"""
scripts/train_realdata.py
==========================
Real data par MLBrain train karo aur backtest chalao.

Prerequisites:
    python scripts/download_data.py   (pehle yeh chalao)

Kya karta hai:
    1. data/historical/ se saare CSV load karta hai
    2. Sweep-focused labels generate karta hai (real market outcomes)
    3. GradientBoosting model train karta hai (walk-forward, no lookahead)
    4. Full v5.1 backtest chalata hai real data par
    5. model/ folder mein trained model save karta hai
    6. backtest/results/ mein full report save karta hai

Usage:
    python scripts/train_realdata.py
    python scripts/train_realdata.py --months 3    # last 3 months only
    python scripts/train_realdata.py --no-backtest # sirf training
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings('ignore')

# Path setup
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("RealTrainer")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = ROOT_DIR / "data" / "historical"
MODEL_DIR   = ROOT_DIR / "model"
RESULTS_DIR = ROOT_DIR / "backtest" / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FEE_RATE       = 0.0002
SLIPPAGE       = 0.0008
TOTAL_COST     = (FEE_RATE + SLIPPAGE) * 2
TP_ATR_MULT    = 1.5
SL_ATR_MULT    = 1.0
MAX_HOLD_BARS  = 24      # 2 hours max
SWEEP_WICK_MIN = 0.45
MIN_CVD_ABS    = 0.08
MIN_OFI_ABS    = 0.06
MIN_VOL_RATIO  = 1.3

FEATURE_COLS = [
    'cvd', 'ofi', 'cvd_lag1', 'cvd_lag2', 'cvd_ma3',
    'ofi_lag1', 'ofi_ma3',
    'vol_ratio', 'vol_ratio_lag',
    'adx', 'adx_lag1', 'is_trending',
    'ret_1', 'ret_3', 'ret_6', 'ret_12',
    'atr_pct', 'bb_width', 'rsi', 'rsi_lag1',
    'body_ratio', 'wick_ratio_lo', 'wick_ratio_hi', 'forced_mult',
    'sweep_depth_bull', 'sweep_depth_bear',
    'is_london', 'is_ny', 'is_active', 'sin_hour', 'cos_hour',
    'spread_pct', 'has_sweep',
]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _add_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add any features missing from CSV (if download_data.py was older version)."""
    df = df.copy()

    # Ensure vol_ratio_lag exists
    if 'vol_ratio_lag' not in df.columns and 'vol_ratio' in df.columns:
        df['vol_ratio_lag'] = df['vol_ratio'].shift(1)

    if 'rsi_lag1' not in df.columns and 'rsi' in df.columns:
        df['rsi_lag1'] = df['rsi'].shift(1)

    if 'adx_lag1' not in df.columns and 'adx' in df.columns:
        df['adx_lag1'] = df['adx'].shift(1)

    if 'has_sweep' not in df.columns:
        df['has_sweep'] = df.get('is_sweep', pd.Series(False, index=df.index)).astype(int)

    # Ensure all feature cols exist (fill missing with 0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# LABEL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_labels(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Label sweep/absorption candles based on REAL forward price action.

    Signal criteria (all must fire):
      - Wick ratio >= 0.45 (sweep structure)
      - CVD confirms direction
      - Volume spike >= 1.3x
      - OFI confirms (optional boost but not mandatory)

    Label = 1: TP hit before SL AND net return > total cost
    Label = 0: SL hit, timeout, or net return <= cost
    """
    df = df.copy()
    labels  = np.full(len(df), np.nan)
    sides   = np.full(len(df), '', dtype=object)

    highs   = df['high'].values
    lows    = df['low'].values
    closes  = df['close'].values
    atr_v   = df['atr'].values if 'atr' in df.columns else np.full(len(df), 0.001)
    wl_v    = df['wick_ratio_lo'].values if 'wick_ratio_lo' in df.columns else np.zeros(len(df))
    wh_v    = df['wick_ratio_hi'].values if 'wick_ratio_hi' in df.columns else np.zeros(len(df))
    cvd_v   = df['cvd'].values if 'cvd' in df.columns else np.zeros(len(df))
    ofi_v   = df['ofi'].values if 'ofi' in df.columns else np.zeros(len(df))
    vol_v   = df['vol_ratio'].values if 'vol_ratio' in df.columns else np.ones(len(df))
    is_sw   = df['is_sweep'].values if 'is_sweep' in df.columns else np.zeros(len(df), bool)
    sw_dir  = df['sweep_dir'].values if 'sweep_dir' in df.columns else np.full(len(df), '')

    tp_count = sl_count = to_count = 0

    for t in range(20, len(df) - MAX_HOLD_BARS - 1):
        atr = atr_v[t]
        if atr <= 0 or np.isnan(atr):
            continue

        # Determine signal side
        side = None

        bull_struct = wl_v[t] >= SWEEP_WICK_MIN or (is_sw[t] and sw_dir[t] == 'BULL')
        bear_struct = wh_v[t] >= SWEEP_WICK_MIN or (is_sw[t] and sw_dir[t] == 'BEAR')

        if bull_struct and cvd_v[t] > MIN_CVD_ABS and vol_v[t] >= MIN_VOL_RATIO:
            side = 'BUY'
        elif bear_struct and cvd_v[t] < -MIN_CVD_ABS and vol_v[t] >= MIN_VOL_RATIO:
            side = 'SELL'

        if side is None:
            continue

        entry = closes[t]
        if side == 'BUY':
            tp = entry + atr * TP_ATR_MULT
            sl = entry - atr * SL_ATR_MULT
        else:
            tp = entry - atr * TP_ATR_MULT
            sl = entry + atr * SL_ATR_MULT

        # Forward simulation on REAL price data
        won = False
        for f in range(1, MAX_HOLD_BARS + 1):
            ft = t + f
            if ft >= len(df):
                break
            if side == 'BUY':
                if highs[ft] >= tp:  won = True;  tp_count += 1; break
                if lows[ft]  <= sl:  won = False; sl_count += 1; break
            else:
                if lows[ft]  <= tp:  won = True;  tp_count += 1; break
                if highs[ft] >= sl:  won = False; sl_count += 1; break
        else:
            to_count += 1

        if won:
            net = atr * TP_ATR_MULT / entry - TOTAL_COST
            labels[t] = 1 if net > 0 else 0
        else:
            labels[t] = 0

        sides[t] = side

    df['label']       = labels
    df['signal_side'] = sides

    labeled = (~np.isnan(labels)).sum()
    pos     = (labels == 1).sum()
    logger.info(
        f"  [{symbol}] labeled={labeled} | "
        f"WR={pos/labeled*100:.1f}% | "
        f"TP={tp_count} SL={sl_count} TO={to_count}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data(months: int = 6) -> dict:
    """Load all CSVs from data/historical/ directory."""
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error(
            f"Manifest not found at {manifest_path}\n"
            f"Run 'python scripts/download_data.py' first!"
        )
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    dfs = {}
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=months * 30)

    for sym, path in manifest['symbols'].items():
        if not os.path.exists(path):
            logger.warning(f"  File not found: {path} — skipping {sym}")
            continue

        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Filter to requested months
        df = df[df['timestamp'] >= cutoff].copy()
        df.reset_index(drop=True, inplace=True)

        if len(df) < 1000:
            logger.warning(f"  {sym}: only {len(df)} rows after filter — skipping")
            continue

        df = _add_missing_features(df)
        dfs[sym] = df
        logger.info(f"  {sym}: {len(df):,} rows loaded")

    if not dfs:
        logger.error("No data loaded. Run download_data.py first.")
        sys.exit(1)

    return dfs


def build_train_dataset(dfs: dict):
    all_X, all_y = [], []

    for sym, df in dfs.items():
        df_labeled = generate_labels(df, sym)
        valid = df_labeled.dropna(subset=['label'] + FEATURE_COLS)
        valid = valid[valid['label'].notna() & (valid['signal_side'] != '')]

        X = valid[FEATURE_COLS].astype(float)
        y = valid['label'].astype(int)
        all_X.append(X)
        all_y.append(y)

    return pd.concat(all_X, ignore_index=True), pd.concat(all_y, ignore_index=True)


def train(dfs: dict) -> dict:
    logger.info("\n=== MLBrain Training on Real Data ===")
    X, y = build_train_dataset(dfs)

    logger.info(f"\nDataset: {len(X):,} signal samples | overall WR={y.mean()*100:.1f}%")

    # Walk-forward split — 70% train, 30% val (no shuffle, preserves time order)
    split   = int(len(X) * 0.70)
    X_tr, X_val = X.iloc[:split],  X.iloc[split:]
    y_tr, y_val = y.iloc[:split],  y.iloc[split:]

    logger.info(f"Train: {len(X_tr):,} | Val: {len(X_val):,}")
    logger.info(f"Train WR: {y_tr.mean()*100:.1f}% | Val WR: {y_val.mean()*100:.1f}%")

    scaler   = StandardScaler()
    Xtr_sc   = scaler.fit_transform(X_tr)
    Xvl_sc   = scaler.transform(X_val)

    sw = compute_sample_weight('balanced', y_tr)

    model = GradientBoostingClassifier(
        n_estimators        = 500,
        max_depth           = 3,
        learning_rate       = 0.03,
        subsample           = 0.75,
        min_samples_leaf    = 25,
        max_features        = 0.75,
        random_state        = 42,
        validation_fraction = 0.1,
        n_iter_no_change    = 40,
        tol                 = 1e-4,
    )

    logger.info("Training GradientBoostingClassifier...")
    t0 = time.time()
    model.fit(Xtr_sc, y_tr, sample_weight=sw)
    logger.info(f"  Done in {time.time()-t0:.1f}s | trees used: {model.n_estimators_}")

    y_prob = model.predict_proba(Xvl_sc)[:, 1]
    auc    = roc_auc_score(y_val, y_prob)

    # Find optimal threshold (precision >= 0.60, recall >= 0.30)
    prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_val, y_prob)
    best_t, best_p, best_r = 0.5, 0.0, 0.0
    for p, r, t in zip(prec_curve, rec_curve, thresh_curve):
        if p >= 0.60 and r >= 0.30 and p > best_p:
            best_p, best_r, best_t = p, r, t

    if best_p == 0.0:
        # Fallback: just use 0.55
        best_t = 0.55
        y_pred_best = (y_prob >= best_t).astype(int)
        best_p = ((y_pred_best == 1) & (y_val == 1)).sum() / (y_pred_best.sum() + 1e-9)
        best_r = ((y_pred_best == 1) & (y_val == 1)).sum() / (y_val.sum() + 1e-9)

    logger.info(f"\n--- Validation ---")
    logger.info(f"  AUC-ROC     : {auc:.4f}")
    logger.info(f"  Best thresh : {best_t:.3f}")
    logger.info(f"  Precision   : {best_p*100:.1f}%")
    logger.info(f"  Recall      : {best_r*100:.1f}%")
    logger.info(f"  AvgPrecision: {average_precision_score(y_val, y_prob):.4f}")
    print(classification_report(y_val, (y_prob >= best_t).astype(int)))

    # Feature importance
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    logger.info("Top 15 features:")
    for name, imp in importances[:15]:
        bar = "█" * int(imp * 400)
        logger.info(f"  {name:<24} {imp:.4f}  {bar}")

    return {
        'model':         model,
        'scaler':        scaler,
        'feature_cols':  FEATURE_COLS,
        'threshold':     float(best_t),
        'auc':           float(auc),
        'precision':     float(best_p),
        'recall':        float(best_r),
        'train_samples': len(X_tr),
        'val_samples':   len(X_val),
        'version':       'v5.1-real',
        'trained_at':    datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST (v5.1 full logic — same as backtest_engine_v2 but self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(dfs: dict, model_payload: dict, initial_balance: float = 100.0) -> dict:
    """
    Run full v5.1 backtest on real data using trained model.
    """
    from backtest.backtest_v51 import BacktestV51
    engine = BacktestV51(
        initial_balance = initial_balance,
        model_payload   = model_payload,
    )
    return engine.run(dfs)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--months',      type=int,  default=6,     help='Months of data to use (default 6)')
    parser.add_argument('--balance',     type=float,default=100.0, help='Starting backtest balance (default 100)')
    parser.add_argument('--no-backtest', action='store_true',       help='Skip backtest, train only')
    parser.add_argument('--model-only',  action='store_true',       help='Load existing model, skip training')
    args = parser.parse_args()

    print("=" * 65)
    print("  MLBrain Real Data Trainer + Backtest  —  v5.1")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"\n[1] Loading real data (last {args.months} months)...")
    dfs = load_all_data(months=args.months)
    logger.info(f"  Loaded {len(dfs)} symbols")

    # ── Train / load model ────────────────────────────────────────────────────
    model_path = MODEL_DIR / "mlbrain_v51_real.pkl"

    if args.model_only and model_path.exists():
        logger.info("\n[2] Loading existing model...")
        model_payload = joblib.load(model_path)
        logger.info(f"  Model loaded | AUC={model_payload.get('auc','?'):.4f}")
    else:
        logger.info("\n[2] Training MLBrain on real data...")
        model_payload = train(dfs)

        # Save model
        joblib.dump(model_payload, model_path)
        logger.info(f"\n  Model saved → {model_path}")

        # Also copy to engine/ for the live bot to use
        live_model_path = ROOT_DIR / "engine" / "mlbrain_model.pkl"
        joblib.dump(model_payload, live_model_path)
        logger.info(f"  Live model  → {live_model_path}")

    # ── Backtest ──────────────────────────────────────────────────────────────
    if not args.no_backtest:
        logger.info("\n[3] Running v5.1 backtest on real data...")

        # Import here to avoid circular import issues
        try:
            from backtest.backtest_v51 import BacktestV51
            engine  = BacktestV51(
                initial_balance = args.balance,
                model_payload   = model_payload,
            )
            results = engine.run(dfs)
            _print_results(results, model_payload)

            # Save results
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            results_path = RESULTS_DIR / f"backtest_{ts}.json"
            save_data = {k: v for k, v in results['summary'].items()}
            save_data['tier_breakdown']  = results['tier_breakdown']
            save_data['setup_breakdown'] = results['setup_breakdown']
            save_data['model_auc']       = model_payload.get('auc', 0)
            with open(results_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            logger.info(f"\n  Results saved → {results_path}")

            tlog = results.get('trade_log')
            if tlog is not None:
                tlog_path = RESULTS_DIR / f"trades_{ts}.csv"
                tlog.to_csv(tlog_path, index=False)
                logger.info(f"  Trade log   → {tlog_path}")

        except ImportError as e:
            logger.error(f"  Backtest engine not found: {e}")
            logger.error("  Run the bot first to ensure backtest/backtest_v51.py exists")

    print("\n  ✅ Done!\n")


def _print_results(results: dict, ml: dict):
    s  = results['summary']
    tb = results['tier_breakdown']
    sb = results['setup_breakdown']
    sym = results.get('symbol_breakdown', {})

    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║      REAL DATA BACKTEST RESULTS  —  v5.1            ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Period         : {s['total_days']:.0f} days (REAL Binance data)     ║")
    print(f"║  Initial Balance: ${s['initial_balance']:.2f}                           ║")
    print(f"║  Final Balance  : ${s['final_balance']:.4f}                        ║")
    print(f"║  Total Return   : {s['total_return_pct']:+.2f}%                           ║")
    print(f"║  Max Drawdown   : {s['max_drawdown_pct']:.2f}%                          ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Total Trades   : {s['total_trades']}                                   ║")
    print(f"║  Trades/Day     : {s['trades_per_day']:.3f}  (target: 0.8–1.2)        ║")
    print(f"║  Win Rate       : {s['win_rate']*100:.1f}%  (target: ≥55%)            ║")
    print(f"║  Avg RR         : {s['avg_rr']:.2f}x  (min: 1.5)                 ║")
    print(f"║  Profit Factor  : {s['profit_factor']:.3f}                             ║")
    print(f"║  Expectancy/tr  : ${s['expectancy_usd']:.4f}                        ║")
    print(f"║  Wins/Losses/TO : {s['wins']}/{s['losses']}/{s['timeouts']}                              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  TIER BREAKDOWN                                      ║")
    print(f"║  PRIMARY   : {tb['primary']['trades']:3d} trades | WR={tb['primary']['win_rate']*100:.1f}% | PnL=${tb['primary']['pnl']:+.3f} ║")
    print(f"║  SECONDARY : {tb['secondary']['trades']:3d} trades | WR={tb['secondary']['win_rate']*100:.1f}% | PnL=${tb['secondary']['pnl']:+.3f} ║")
    print(f"║  RE-ENTRY  : {tb['reentry']['trades']:3d} trades | WR={tb['reentry']['win_rate']*100:.1f}% | PnL=${tb['reentry']['pnl']:+.3f} ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  SETUP BREAKDOWN                                     ║")
    print(f"║  TrapReversal : {sb['trap_reversal']['trades']:3d} trades | WR={sb['trap_reversal']['win_rate']*100:.1f}%               ║")
    print(f"║  Continuation : {sb['continuation']['trades']:3d} trades | WR={sb['continuation']['win_rate']*100:.1f}%               ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  PER-SYMBOL                                          ║")
    for sym_name, sd in sym.items():
        short = sym_name.split('/')[0]
        print(f"║  {short:<6}: {sd['trades']:3d} trades | WR={sd['win_rate']*100:.0f}% | PnL=${sd['pnl']:+.3f}       ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  MLBRAIN (REAL DATA)                                 ║")
    print(f"║  AUC-ROC   : {ml.get('auc',0):.4f}                             ║")
    print(f"║  Precision : {ml.get('precision',0)*100:.1f}% (at threshold {ml.get('threshold',0):.2f})      ║")
    print(f"║  Recall    : {ml.get('recall',0)*100:.1f}%                           ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  VALIDATION CHECKS                                   ║")
    checks = [
        ("Trades/day 0.8-1.5",   0.8 <= s['trades_per_day'] <= 1.5),
        ("Win rate >= 55%",       s['win_rate'] >= 0.55),
        ("RR >= 1.5",             s['avg_rr'] >= 1.50),
        ("Expectancy > 0",        s['expectancy_usd'] > 0),
        ("Profit factor > 1.0",   s['profit_factor'] > 1.0),
        ("Max DD < 20%",          s['max_drawdown_pct'] > -20.0),
        ("MLBrain AUC > 0.62",   ml.get('auc', 0) > 0.62),
    ]
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"║  {status} {name:<40}  ║")
    print("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
