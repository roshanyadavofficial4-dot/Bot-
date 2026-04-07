import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("DataDownloader")

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "historical"
RAW_DIR  = ROOT_DIR / "data" / "historical" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = {
    "DOGE/USDT:USDT": "DOGE_USDT_USDT",
    "XRP/USDT:USDT":  "XRP_USDT_USDT",
    "ADA/USDT:USDT":  "ADA_USDT_USDT",
    "SOL/USDT:USDT":  "SOL_USDT_USDT",
    "BNB/USDT:USDT":  "BNB_USDT_USDT",
}

TIMEFRAME    = "5m"
DAYS_OF_DATA = 90
BATCH_LIMIT  = 1000

BINANCE_RAW_COLS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_vol', 'trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
]


def load_from_raw_csv(file_stem: str, symbol: str) -> pd.DataFrame:
    raw_path = RAW_DIR / f"{file_stem}.csv"
    if not raw_path.exists():
        return pd.DataFrame()

    logger.info(f"  Local raw CSV found: {raw_path}")
    try:
        # ── Detect header row ────────────────────────────────────────────────
        # Newer Binance CSVs (2024+) include a header: open_time,open,high,...
        # Older ones are headerless. Detect by checking the first cell.
        peek       = pd.read_csv(raw_path, header=None, nrows=1)
        first_cell = str(peek.iloc[0, 0]).strip().lower()
        has_header = first_cell in (
            'open_time', 'timestamp', 'time', 'date', 'open time', 'opentime'
        )

        if has_header:
            df_raw = pd.read_csv(raw_path)
            df_raw.columns = [c.lower().strip().replace(' ', '_') for c in df_raw.columns]
            # Find the open_time column (not close_time)
            ts_col = next(
                (c for c in df_raw.columns if 'time' in c and 'close' not in c),
                df_raw.columns[0]
            )
        else:
            df_raw = pd.read_csv(raw_path, header=None)
            df_raw.columns = BINANCE_RAW_COLS[:df_raw.shape[1]]
            ts_col = 'open_time'

        # ── Parse timestamp safely ───────────────────────────────────────────
        # Use pd.to_numeric first to avoid string / overflow issues.
        # Auto-detect unit: 13-digit = ms, 10-digit = seconds.
        ts_numeric = pd.to_numeric(df_raw[ts_col], errors='coerce')
        sample     = ts_numeric.dropna().iloc[0] if not ts_numeric.dropna().empty else 0
        unit       = 'ms' if sample > 1e12 else 's'

        df = pd.DataFrame()
        df['timestamp'] = pd.to_datetime(ts_numeric, unit=unit, utc=True, errors='coerce')
        df['open']      = pd.to_numeric(df_raw['open'],   errors='coerce')
        df['high']      = pd.to_numeric(df_raw['high'],   errors='coerce')
        df['low']       = pd.to_numeric(df_raw['low'],    errors='coerce')
        df['close']     = pd.to_numeric(df_raw['close'],  errors='coerce')
        df['volume']    = pd.to_numeric(df_raw['volume'], errors='coerce')

        # Drop rows that failed to parse
        df.dropna(subset=['timestamp', 'open', 'close'], inplace=True)
        df['symbol'] = symbol
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        if df.empty:
            logger.error(f"  {raw_path.name}: no valid rows after parsing")
            return pd.DataFrame()

        logger.info(
            f"  Loaded {len(df):,} rows | "
            f"{df['timestamp'].iloc[0].strftime('%Y-%m-%d')} → "
            f"{df['timestamp'].iloc[-1].strftime('%Y-%m-%d')} "
            f"[unit={unit}, header={has_header}]"
        )
        return df

    except Exception as e:
        logger.error(f"  Failed to read local CSV {raw_path}: {e}")
        return pd.DataFrame()


def download_from_binance(symbol: str) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed")
        return pd.DataFrame()

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })

    since_ms   = int((datetime.now(timezone.utc) - timedelta(days=DAYS_OF_DATA)).timestamp() * 1000)
    now_ms     = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_ohlcv  = []
    fetch_from = since_ms

    logger.info(f"Trying Binance Futures download for {symbol}...")

    while fetch_from < now_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=fetch_from, limit=BATCH_LIMIT)
            if not batch:
                break
            all_ohlcv.extend(batch)
            fetch_from = batch[-1][0] + 1
            logger.info(f"  Fetched {len(all_ohlcv):,} candles so far...")
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"  Binance download failed: {e}")
            return pd.DataFrame()

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['symbol'] = symbol
    logger.info(f"  Downloaded {len(df):,} candles from Binance")
    return df


def _calc_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h, l, c, o, v = df['high'], df['low'], df['close'], df['open'], df['volume']

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr']     = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / (c + 1e-9)

    up, down = h.diff(), -l.diff()
    pdm   = np.where((up > down) & (up > 0), up, 0.0)
    mdm   = np.where((down > up) & (down > 0), down, 0.0)
    atr14 = tr.rolling(14).mean()
    pdi   = 100 * pd.Series(pdm, index=df.index).rolling(14).mean() / (atr14 + 1e-9)
    mdi   = 100 * pd.Series(mdm, index=df.index).rolling(14).mean() / (atr14 + 1e-9)
    df['adx'] = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)).rolling(14).mean()

    delta = c.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    df['bb_width'] = c.rolling(20).std() / (c.rolling(20).mean() + 1e-9)

    df['vol_ma20']  = v.rolling(20).mean()
    df['vol_ratio'] = v / (df['vol_ma20'] + 1e-9)

    body             = (c - o).abs()
    candle_range     = h - l
    df['body_ratio']    = body / (candle_range + 1e-9)
    df['lower_wick']    = pd.concat([o, c], axis=1).min(axis=1) - l
    df['upper_wick']    = h - pd.concat([o, c], axis=1).max(axis=1)
    df['wick_ratio_lo'] = df['lower_wick'] / (candle_range + 1e-9)
    df['wick_ratio_hi'] = df['upper_wick'] / (candle_range + 1e-9)
    df['forced_mult']   = candle_range / (df['atr'] + 1e-9)

    bull_candle = (c > o).astype(float)
    bear_candle = (c < o).astype(float)
    raw_cvd     = v * (bull_candle - bear_candle)
    df['cvd']   = (raw_cvd / (raw_cvd.rolling(50).std() + 1e-9)).clip(-1, 1).fillna(0.0)

    df['date']           = pd.to_datetime(df['timestamp']).dt.date
    df['cvd_cumulative'] = df.groupby('date')['cvd'].cumsum()
    df.drop(columns=['date'], inplace=True)

    raw_ofi   = c.diff() * (v / (v.rolling(20).mean() + 1e-9))
    df['ofi'] = (raw_ofi / (raw_ofi.rolling(100).std() + 1e-9)).clip(-1, 1).fillna(0.0)

    df['ret_1']  = c.pct_change(1)
    df['ret_3']  = c.pct_change(3)
    df['ret_6']  = c.pct_change(6)
    df['ret_12'] = c.pct_change(12)

    df['cvd_lag1']      = df['cvd'].shift(1)
    df['cvd_lag2']      = df['cvd'].shift(2)
    df['cvd_ma3']       = df['cvd'].rolling(3).mean()
    df['ofi_lag1']      = df['ofi'].shift(1)
    df['ofi_ma3']       = df['ofi'].rolling(3).mean()
    df['vol_ratio_lag'] = df['vol_ratio'].shift(1)
    df['rsi_lag1']      = df['rsi'].shift(1)
    df['adx_lag1']      = df['adx'].shift(1)

    ts   = pd.to_datetime(df['timestamp'], utc=True)
    hour = ts.dt.hour
    df['hour_utc']  = hour.values
    df['is_london'] = ((hour >= 8)  & (hour < 12)).astype(int).values
    df['is_ny']     = ((hour >= 12) & (hour < 17)).astype(int).values
    df['is_active'] = ((hour >= 8)  & (hour < 20)).astype(int).values
    df['sin_hour']  = np.sin(2 * np.pi * hour / 24).values
    df['cos_hour']  = np.cos(2 * np.pi * hour / 24).values

    df['sma50']       = c.rolling(50).mean()
    df['sma200']      = c.rolling(200).mean()
    df['is_trending'] = (df['adx'] >= 25).astype(int)

    swing_lo               = l.rolling(12).min().shift(3)
    swing_hi               = h.rolling(12).max().shift(3)
    df['sweep_depth_bull'] = (swing_lo - l) / (df['atr'] + 1e-9)
    df['sweep_depth_bear'] = (h - swing_hi) / (df['atr'] + 1e-9)
    df['spread_pct']       = 0.0001

    return df


def _detect_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    df       = df.copy()
    n        = len(df)
    is_sweep = np.zeros(n, dtype=bool)
    sw_dir   = np.full(n, '', dtype=object)
    hi, lo, op, cl = df['high'].values, df['low'].values, df['open'].values, df['close'].values
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
                if cl[i] > swing_lo and lw / cr >= 0.45:
                    is_sweep[t], sw_dir[t] = True, 'BULL'
                    break
            if hi[i] > swing_hi:
                uw = hi[i] - max(op[i], cl[i])
                if cl[i] < swing_hi and uw / cr >= 0.45:
                    is_sweep[t], sw_dir[t] = True, 'BEAR'
                    break

    df['is_sweep']  = is_sweep
    df['has_sweep'] = is_sweep.astype(int)
    df['sweep_dir'] = sw_dir
    return df


def process_and_save(symbol: str, file_stem: str) -> str:
    save_path = DATA_DIR / f"{file_stem}_{TIMEFRAME}.csv"

    df = load_from_raw_csv(file_stem, symbol)

    if df.empty:
        logger.info(f"  No local CSV for {symbol} — trying Binance...")
        df = download_from_binance(symbol)

    if df.empty:
        logger.error(f"  No data available for {symbol} — skipping.")
        return None

    df = _calc_features(df)
    df = _detect_sweeps(df)

    before = len(df)
    df.dropna(subset=['atr', 'adx', 'rsi', 'cvd', 'ofi', 'vol_ratio'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(save_path, index=False)

    sweep_count = int(df['is_sweep'].sum())
    logger.info(f"  Saved {len(df):,} rows (dropped {before - len(df)}) → {save_path}")
    logger.info(f"  Sweeps: {sweep_count} (BULL={int((df['sweep_dir']=='BULL').sum())}, BEAR={int((df['sweep_dir']=='BEAR').sum())})")
    return str(save_path)


def main():
    raw_files = list(RAW_DIR.glob("*.csv"))
    if raw_files:
        logger.info(f"Local raw CSVs found in {RAW_DIR}: {[f.name for f in raw_files]}")
        logger.info("Using offline mode — skipping Binance download.")
    else:
        logger.info(f"No local CSVs in {RAW_DIR} — will try Binance download.")

    saved   = {}
    t_total = time.time()

    for symbol, file_stem in SYMBOLS.items():
        t0   = time.time()
        path = process_and_save(symbol, file_stem)
        if path:
            saved[symbol] = path
            logger.info(f"  OK {symbol} in {time.time()-t0:.0f}s\n")
        else:
            logger.error(f"  FAILED {symbol}\n")
        time.sleep(0.5)

    print(f"\nFiles saved: {len(saved)}/{len(SYMBOLS)}")
    for sym, path in saved.items():
        print(f"  {sym:<25}  ({os.path.getsize(path)//1024} KB)")

    with open(DATA_DIR / "manifest.json", 'w') as f:
        json.dump({
            'symbols':       saved,
            'timeframe':     TIMEFRAME,
            'source':        'local_csv' if raw_files else 'binance_futures_ccxt',
            'downloaded_at': datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    if not saved:
        logger.error(
            "0 files saved. Check that raw CSVs exist in data/historical/raw/ "
            "with correct names (e.g. DOGE_USDT_USDT.csv)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
