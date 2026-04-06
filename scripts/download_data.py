"""
scripts/download_data.py  — v5.2 Binance Edition (GitHub Actions safe)
========================================================================
GitHub Actions pe Binance REST API kaam karta hai (Azure IPs block nahi hote).
Isliye ab yfinance ki zaroorat nahi — seedha Binance Futures se real data milta hai.

Fayde:
  - Real futures OHLCV data (spot nahi)
  - Real volume — CVD accurate hogi
  - 90 days ka data (yfinance sirf 58 days deta tha)
  - Koi API key nahi chahiye (public endpoint)

Usage:
    python scripts/download_data.py
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
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

# Binance Futures symbols (ccxt format)  →  CSV file stem
SYMBOLS = {
    "DOGE/USDT:USDT": "DOGE_USDT_USDT",
    "XRP/USDT:USDT":  "XRP_USDT_USDT",
    "ADA/USDT:USDT":  "ADA_USDT_USDT",
    "SOL/USDT:USDT":  "SOL_USDT_USDT",
    "BNB/USDT:USDT":  "BNB_USDT_USDT",
}

TIMEFRAME    = "5m"
DAYS_OF_DATA = 90       # 3 mahine ka data
BATCH_LIMIT  = 1000     # Binance max candles per request


# ── Binance data fetch ─────────────────────────────────────────────────────────

def download_symbol(symbol: str, days: int = DAYS_OF_DATA) -> pd.DataFrame:
    """Binance Futures se ccxt ke zariye historical OHLCV fetch karta hai."""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',   # USDT-M Futures
        },
    })

    since_ms   = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_ms     = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_ohlcv  = []
    fetch_from = since_ms

    logger.info(f"Downloading {symbol} ({days} days) from Binance Futures...")

    while fetch_from < now_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe = TIMEFRAME,
                since     = fetch_from,
                limit     = BATCH_LIMIT,
            )
            if not batch:
                break

            all_ohlcv.extend(batch)
            fetch_from = batch[-1][0] + 1   # next ms after last candle
            logger.info(f"  Fetched {len(all_ohlcv):,} candles so far...")
            time.sleep(0.3)

        except ccxt.NetworkError as e:
            logger.warning(f"Network error — retrying in 5s: {e}")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error for {symbol}: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break

    if not all_ohlcv:
        logger.error(f"No data fetched for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['symbol'] = symbol

    logger.info(
        f"  Total: {len(df):,} candles | "
        f"{df['timestamp'].iloc[0].strftime('%Y-%m-%d')} → "
        f"{df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}"
    )
    return df


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
    up    = h.diff()
    down  = -l.diff()
    pdm   = np.where((up > down) & (up > 0), up, 0.0)
    mdm   = np.where((down > up) & (down > 0), down, 0.0)
    atr14 = tr.rolling(14).mean()
    pdi   = 100 * pd.Series(pdm, index=df.index).rolling(14).mean() / (atr14 + 1e-9)
    mdi   = 100 * pd.Series(mdm, index=df.index).rolling(14).mean() / (atr14 + 1e-9)
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
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
    body                = (c - o).abs()
    candle_range        = h - l
    df['body_ratio']    = body / (candle_range + 1e-9)
    df['lower_wick']    = pd.concat([o, c], axis=1).min(axis=1) - l
    df['upper_wick']    = h - pd.concat([o, c], axis=1).max(axis=1)
    df['wick_ratio_lo'] = df['lower_wick'] / (candle_range + 1e-9)
    df['wick_ratio_hi'] = df['upper_wick'] / (candle_range + 1e-9)
    df['forced_mult']   = candle_range / (df['atr'] + 1e-9)

    # ── CVD — Real Volume Based (Binance Futures) ─────────────────────────────
    bull_candle = (c > o).astype(float)
    bear_candle = (c < o).astype(float)
    raw_cvd     = v * (bull_candle - bear_candle)
    cvd_std     = raw_cvd.rolling(50).std()
    df['cvd']   = (raw_cvd / (cvd_std + 1e-9)).clip(-1, 1).fillna(0.0)

    # Cumulative CVD (daily reset)
    df['date']           = pd.to_datetime(df['timestamp']).dt.date
    df['cvd_cumulative'] = df.groupby('date')['cvd'].cumsum()
    df.drop(columns=['date'], inplace=True)

    # ── OFI — Order Flow Imbalance ────────────────────────────────────────────
    price_chg = c.diff()
    vol_rel   = v / (v.rolling(20).mean() + 1e-9)
    raw_ofi   = price_chg * vol_rel
    ofi_std   = raw_ofi.rolling(100).std()
    df['ofi'] = (raw_ofi / (ofi_std + 1e-9)).clip(-1, 1).fillna(0.0)

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
    df['hour_utc']  = hour.values
    df['is_london'] = ((hour >= 8)  & (hour < 12)).astype(int).values
    df['is_ny']     = ((hour >= 12) & (hour < 17)).astype(int).values
    df['is_active'] = ((hour >= 8)  & (hour < 20)).astype(int).values
    df['sin_hour']  = np.sin(2 * np.pi * hour / 24).values
    df['cos_hour']  = np.cos(2 * np.pi * hour / 24).values

    # ── Trend ─────────────────────────────────────────────────────────────────
    df['sma50']       = c.rolling(50).mean()
    df['sma200']      = c.rolling(200).mean()
    df['is_trending'] = (df['adx'] >= 25).astype(int)

    # ── Sweep depth proxies ───────────────────────────────────────────────────
    swing_lo               = l.rolling(12).min().shift(3)
    swing_hi               = h.rolling(12).max().shift(3)
    df['sweep_depth_bull'] = (swing_lo - l) / (df['atr'] + 1e-9)
    df['sweep_depth_bear'] = (h - swing_hi) / (df['atr'] + 1e-9)

    # ── Spread placeholder ────────────────────────────────────────────────────
    df['spread_pct'] = 0.0001

    return df


def _detect_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """Price-action based sweep detection."""
    df       = df.copy()
    n        = len(df)
    is_sweep = np.zeros(n, dtype=bool)
    sw_dir   = np.full(n, '', dtype=object)

    hi = df['high'].values
    lo = df['low'].values
    op = df['open'].values
    cl = df['close'].values

    lookback = 13

    for t in range(lookback + 3, n):
        swing_lo = np.min(lo[t - lookback - 3: t - 3])
        swing_hi = np.max(hi[t - lookback - 3: t - 3])

        for i in range(max(t - 3, 0), t):
            cr = hi[i] - lo[i]
            if cr < 1e-10:
                continue
            if lo[i] < swing_lo:
                lw = min(op[i], cl[i]) - lo[i]
                wr = lw / cr
                if cl[i] > swing_lo and wr >= 0.45:
                    is_sweep[t] = True
                    sw_dir[t]   = 'BULL'
                    break
            if hi[i] > swing_hi:
                uw = hi[i] - max(op[i], cl[i])
                wr = uw / cr
                if cl[i] < swing_hi and wr >= 0.45:
                    is_sweep[t] = True
                    sw_dir[t]   = 'BEAR'
                    break

    df['is_sweep']  = is_sweep
    df['has_sweep'] = is_sweep.astype(int)
    df['sweep_dir'] = sw_dir
    return df


# ── Save ───────────────────────────────────────────────────────────────────────

def process_and_save(symbol: str, file_stem: str) -> str:
    """Download → features → sweeps → save."""
    save_path = DATA_DIR / f"{file_stem}_{TIMEFRAME}.csv"

    df = download_symbol(symbol)
    if df.empty:
        return None

    logger.info(f"  Engineering features for {symbol} ...")
    df = _calc_features(df)

    logger.info(f"  Detecting sweeps for {symbol} ...")
    df = _detect_sweeps(df)

    key_cols = ['atr', 'adx', 'rsi', 'cvd', 'ofi', 'vol_ratio']
    before   = len(df)
    df.dropna(subset=key_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(save_path, index=False)

    sweep_count = int(df['is_sweep'].sum())
    bull_sw     = int((df['sweep_dir'] == 'BULL').sum())
    bear_sw     = int((df['sweep_dir'] == 'BEAR').sum())

    logger.info(f"  Saved {len(df):,} rows (dropped {before - len(df)} NaN rows) → {save_path}")
    logger.info(f"  Sweeps: {sweep_count} total (BULL={bull_sw}, BEAR={bear_sw}, {sweep_count/len(df)*100:.2f}%)")
    return str(save_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Data Downloader v5.2  —  Binance Futures Edition")
    print("=" * 65)
    print(f"  Symbols   : {', '.join(SYMBOLS.keys())}")
    print(f"  Timeframe : {TIMEFRAME}  ({DAYS_OF_DATA} days)")
    print(f"  Source    : Binance USDT-M Futures (no API key needed)")
    print(f"  Output    : {DATA_DIR}")
    print("=" * 65)
    print()

    saved   = {}
    t_total = time.time()

    for symbol, file_stem in SYMBOLS.items():
        t0   = time.time()
        path = process_and_save(symbol, file_stem)
        if path:
            saved[symbol] = path
            logger.info(f"  ✅ {symbol} done in {time.time() - t0:.0f}s\n")
        else:
            logger.error(f"  ❌ {symbol} FAILED\n")
        time.sleep(1.0)

    print("=" * 65)
    print(f"  Complete in {time.time() - t_total:.0f}s")
    print(f"  Files saved: {len(saved)}/{len(SYMBOLS)}")
    for sym, path in saved.items():
        sz = os.path.getsize(path) / 1024
        print(f"    {sym:<25}  {path}  ({sz:.0f} KB)")
    print()
    print("  Next step:")
    print("    python scripts/train_realdata.py")
    print("=" * 65)

    manifest = DATA_DIR / "manifest.json"
    with open(manifest, 'w') as f:
        json.dump({
            'symbols':       saved,
            'timeframe':     TIMEFRAME,
            'source':        'binance_futures_ccxt',
            'cvd_type':      'real_volume_based',
            'days':          DAYS_OF_DATA,
            'downloaded_at': datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    logger.info(f"  Manifest → {manifest}")


if __name__ == "__main__":
    main()
