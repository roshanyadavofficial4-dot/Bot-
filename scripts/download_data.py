"""
scripts/download_data.py  — v5.1 yfinance Edition
===================================================
Replit pe chalao — koi API key nahi chahiye, Binance block ka issue nahi.

Kya karta hai:
  1. yfinance se 5-minute OHLCV data download karta hai (last 58 days max)
  2. Volume-independent CVD approximate karta hai (price action based)
  3. OFI, ADX, ATR, RSI, BB-width engineer karta hai
  4. Price-action based sweep detection (volume pe depend nahi)
  5. data/historical/ mein CSV save karta hai

Usage (Replit terminal mein):
    pip install yfinance pandas numpy scikit-learn joblib
    python scripts/download_data.py

Symbols:
  DOGE-USD, XRP-USD, ADA-USD, SOL-USD, BNB-USD (yfinance format)
  
Note:
  yfinance se crypto ka 5m data sirf last 60 days ka milta hai — yeh limit hai.
  Volume spot exchange ka hota hai (futures jaisa nahi) — isliye hum
  volume-independent CVD use karte hain jo OHLC structure se calculate hoti hai.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("DataDownloader")

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# yfinance symbol map → internal ccxt-style symbol
SYMBOLS = {
    "DOGE-USD":  "DOGE/USDT:USDT",
    "XRP-USD":   "XRP/USDT:USDT",
    "ADA-USD":   "ADA/USDT:USDT",
    "SOL-USD":   "SOL/USDT:USDT",
    "BNB-USD":   "BNB/USDT:USDT",
}

TIMEFRAME = "5m"


# ── Feature Engineering ────────────────────────────────────────────────────────

def _calc_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h, l, c, o, v = df['high'], df['low'], df['close'], df['open'], df['volume']

    # ── ATR ───────────────────────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr']     = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / (c + 1e-9)

    # ── ADX ───────────────────────────────────────────────────────────────────
    up   = h.diff()
    down = -l.diff()
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    mdm  = np.where((down > up) & (down > 0), down, 0.0)
    atr14= tr.rolling(14).mean()
    pdi  = 100 * pd.Series(pdm, index=df.index).rolling(14).mean() / (atr14 + 1e-9)
    mdi  = 100 * pd.Series(mdm, index=df.index).rolling(14).mean() / (atr14 + 1e-9)
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    df['adx'] = dx.rolling(14).mean()

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # ── Bollinger Band Width ──────────────────────────────────────────────────
    sma20          = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    df['bb_width'] = std20 / (sma20 + 1e-9)

    # ── Volume features ───────────────────────────────────────────────────────
    df['vol_ma20']  = v.rolling(20).mean()
    df['vol_ratio'] = v / (df['vol_ma20'] + 1e-9)

    # ── Candle structure ──────────────────────────────────────────────────────
    body              = (c - o).abs()
    candle_range      = h - l
    df['body_ratio']    = body / (candle_range + 1e-9)
    df['lower_wick']    = pd.concat([o, c], axis=1).min(axis=1) - l
    df['upper_wick']    = h - pd.concat([o, c], axis=1).max(axis=1)
    df['wick_ratio_lo'] = df['lower_wick'] / (candle_range + 1e-9)
    df['wick_ratio_hi'] = df['upper_wick'] / (candle_range + 1e-9)
    df['forced_mult']   = candle_range / (df['atr'] + 1e-9)

    # ── CVD — Price Action Based (volume-independent) ─────────────────────────
    # Yeh approach yfinance ke unreliable volume se independent hai.
    # Logic: agar candle bullish hai (close > open) toh buying pressure tha.
    # Wick ratio se intensity measure karte hain.
    #
    # Bull candle: close > open
    #   cvd = +wick_ratio_lo * body_ratio (lower wick = stop hunt + absorption)
    # Bear candle: close < open
    #   cvd = -wick_ratio_hi * body_ratio
    #
    # Yeh synthetic CVD real tick-based CVD se weaker hai lekin directionally
    # correct hai — ML ke liye useful signal deta hai.

    bull_candle = (c > o).astype(float)
    bear_candle = (c < o).astype(float)

    raw_cvd = (
        bull_candle * (df['wick_ratio_lo'] * 0.6 + df['body_ratio'] * 0.4) -
        bear_candle * (df['wick_ratio_hi'] * 0.6 + df['body_ratio'] * 0.4)
    )
    # Normalise -1 to +1
    cvd_std     = raw_cvd.rolling(50).std()
    df['cvd']   = (raw_cvd / (cvd_std + 1e-9)).clip(-1, 1).fillna(0.0)

    # Cumulative CVD (daily reset)
    df['date']            = pd.to_datetime(df['timestamp']).dt.date
    df['cvd_cumulative']  = df.groupby('date')['cvd'].cumsum()
    df.drop(columns=['date'], inplace=True)

    # ── OFI — Price momentum based ───────────────────────────────────────────
    # Order Flow Imbalance: price change × relative volume
    # Normalised to -1 to +1
    price_chg   = c.diff()
    vol_rel     = v / (v.rolling(20).mean() + 1e-9)
    raw_ofi     = price_chg * vol_rel
    ofi_std     = raw_ofi.rolling(100).std()
    df['ofi']   = (raw_ofi / (ofi_std + 1e-9)).clip(-1, 1).fillna(0.0)

    # ── Momentum ──────────────────────────────────────────────────────────────
    df['ret_1']  = c.pct_change(1)
    df['ret_3']  = c.pct_change(3)
    df['ret_6']  = c.pct_change(6)
    df['ret_12'] = c.pct_change(12)

    # ── CVD / OFI lags ────────────────────────────────────────────────────────
    df['cvd_lag1']      = df['cvd'].shift(1)
    df['cvd_lag2']      = df['cvd'].shift(2)
    df['cvd_ma3']       = df['cvd'].rolling(3).mean()
    df['ofi_lag1']      = df['ofi'].shift(1)
    df['ofi_ma3']       = df['ofi'].rolling(3).mean()
    df['vol_ratio_lag'] = df['vol_ratio'].shift(1)
    df['rsi_lag1']      = df['rsi'].shift(1)
    df['adx_lag1']      = df['adx'].shift(1)

    # ── Session encoding (UTC) ────────────────────────────────────────────────
    ts             = pd.to_datetime(df['timestamp'], utc=True)
    hour           = ts.dt.hour
    df['hour_utc']   = hour.values
    df['is_london']  = ((hour >= 8)  & (hour < 12)).astype(int).values
    df['is_ny']      = ((hour >= 12) & (hour < 17)).astype(int).values
    df['is_active']  = ((hour >= 8)  & (hour < 20)).astype(int).values
    df['sin_hour']   = np.sin(2 * np.pi * hour / 24).values
    df['cos_hour']   = np.cos(2 * np.pi * hour / 24).values

    # ── Trend ─────────────────────────────────────────────────────────────────
    df['sma50']       = c.rolling(50).mean()
    df['sma200']      = c.rolling(200).mean()   # higher timeframe trend
    df['is_trending'] = (df['adx'] >= 25).astype(int)

    # ── Sweep depth proxies ───────────────────────────────────────────────────
    swing_lo            = l.rolling(12).min().shift(3)
    swing_hi            = h.rolling(12).max().shift(3)
    df['sweep_depth_bull'] = (swing_lo - l)  / (df['atr'] + 1e-9)
    df['sweep_depth_bear'] = (h - swing_hi)  / (df['atr'] + 1e-9)

    # ── Spread placeholder ────────────────────────────────────────────────────
    df['spread_pct'] = 0.0001

    return df


def _detect_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price-action based sweep detection — volume independent.
    Detects wick-based stop hunts using only OHLC structure.
    """
    df       = df.copy()
    n        = len(df)
    is_sweep = np.zeros(n, dtype=bool)
    sw_dir   = np.full(n, '', dtype=object)

    hi  = df['high'].values
    lo  = df['low'].values
    op  = df['open'].values
    cl  = df['close'].values

    lookback = 13

    for t in range(lookback + 3, n):
        swing_lo = np.min(lo[t - lookback - 3 : t - 3])
        swing_hi = np.max(hi[t - lookback - 3 : t - 3])

        for i in range(max(t - 3, 0), t):
            cr = hi[i] - lo[i]
            if cr < 1e-10:
                continue

            # Bull sweep: wick below swing low + close reclaims
            if lo[i] < swing_lo:
                lw = min(op[i], cl[i]) - lo[i]
                wr = lw / cr
                if cl[i] > swing_lo and wr >= 0.45:
                    is_sweep[t]  = True
                    sw_dir[t]    = 'BULL'
                    break

            # Bear sweep: wick above swing high + close reclaims
            if hi[i] > swing_hi:
                uw = hi[i] - max(op[i], cl[i])
                wr = uw / cr
                if cl[i] < swing_hi and wr >= 0.45:
                    is_sweep[t]  = True
                    sw_dir[t]    = 'BEAR'
                    break

    df['is_sweep']  = is_sweep
    df['sweep_dir'] = sw_dir
    return df


# ── Download ───────────────────────────────────────────────────────────────────

def download_symbol(yf_symbol: str, internal_symbol: str) -> pd.DataFrame:
    """Download 5m data using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    logger.info(f"Downloading {yf_symbol} ({internal_symbol}) ...")

    # yfinance 5m data: max 60 days
    ticker = yf.Ticker(yf_symbol)
    df_raw = ticker.history(period="58d", interval="5m", auto_adjust=True)

    if df_raw.empty:
        logger.error(f"  No data returned for {yf_symbol}")
        return pd.DataFrame()

    # Standardise columns
    df_raw.index.name = 'datetime'
    df_raw = df_raw.reset_index()
    df_raw.columns = [c.lower() for c in df_raw.columns]

    df = pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df_raw['datetime'], utc=True)
    df['open']      = df_raw['open'].astype(float)
    df['high']      = df_raw['high'].astype(float)
    df['low']       = df_raw['low'].astype(float)
    df['close']     = df_raw['close'].astype(float)
    df['volume']    = df_raw['volume'].astype(float)
    df['symbol']    = internal_symbol

    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        f"  Raw candles: {len(df):,} | "
        f"{df['timestamp'].iloc[0].strftime('%Y-%m-%d')} → "
        f"{df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}"
    )
    return df


def process_and_save(yf_symbol: str, internal_symbol: str) -> str:
    """Download → features → sweeps → save."""
    safe_name  = yf_symbol.replace('-', '_')
    save_path  = DATA_DIR / f"{safe_name}_{TIMEFRAME}.csv"

    df = download_symbol(yf_symbol, internal_symbol)
    if df.empty:
        return None

    logger.info(f"  Engineering features for {yf_symbol} ...")
    df = _calc_features(df)

    logger.info(f"  Detecting sweeps for {yf_symbol} ...")
    df = _detect_sweeps(df)

    # Drop NaN rows in key columns
    key_cols = ['atr', 'adx', 'rsi', 'cvd', 'ofi', 'vol_ratio']
    before   = len(df)
    df.dropna(subset=key_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(save_path, index=False)

    sweep_count = df['is_sweep'].sum()
    bull_sw     = (df['sweep_dir'] == 'BULL').sum()
    bear_sw     = (df['sweep_dir'] == 'BEAR').sum()

    logger.info(
        f"  Saved {len(df):,} rows (dropped {before-len(df)} NaN rows) → {save_path}"
    )
    logger.info(
        f"  Sweeps: {sweep_count} total  "
        f"(BULL={bull_sw}, BEAR={bear_sw}, "
        f"{sweep_count/len(df)*100:.2f}% of candles)"
    )
    return str(save_path)


def main():
    print("=" * 65)
    print("  Data Downloader v5.1  —  yfinance Edition (Replit-safe)")
    print("=" * 65)
    print(f"  Symbols   : {', '.join(SYMBOLS.keys())}")
    print(f"  Timeframe : {TIMEFRAME}  (last ~58 days — yfinance limit)")
    print(f"  Output    : {DATA_DIR}")
    print(f"  CVD type  : Price-action based (volume-independent)")
    print("=" * 65)
    print()

    saved = {}
    t_total = time.time()

    for yf_sym, int_sym in SYMBOLS.items():
        t0   = time.time()
        path = process_and_save(yf_sym, int_sym)
        if path:
            saved[int_sym] = path
            logger.info(f"  ✅ {yf_sym} done in {time.time()-t0:.0f}s\n")
        else:
            logger.error(f"  ❌ {yf_sym} FAILED\n")
        time.sleep(1.0)   # polite delay between symbols

    print("=" * 65)
    print(f"  Complete in {time.time()-t_total:.0f}s")
    print(f"  Files saved: {len(saved)}/{len(SYMBOLS)}")
    for sym, path in saved.items():
        sz = os.path.getsize(path) / 1024
        print(f"    {sym:<22}  {path}  ({sz:.0f} KB)")
    print()
    print("  Next step:")
    print("    python scripts/train_realdata.py")
    print("=" * 65)

    # Save manifest
    manifest = DATA_DIR / "manifest.json"
    with open(manifest, 'w') as f:
        json.dump({
            'symbols':       saved,
            'timeframe':     TIMEFRAME,
            'source':        'yfinance',
            'cvd_type':      'price_action_based',
            'downloaded_at': datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    logger.info(f"  Manifest → {manifest}")


if __name__ == "__main__":
    main()
