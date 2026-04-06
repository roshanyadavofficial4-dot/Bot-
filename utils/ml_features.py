import pandas as pd
import numpy as np

def apply_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies technical indicators for Machine Learning feature engineering.
    This function is shared between the Training Engine and the Live Inference Engine
    to guarantee 100% consistency in feature calculation.
    """
    df = df.copy()
    
    # 1. ATR (Average True Range - 14 period)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # Wilder's Smoothing (alpha = 1/period) for consistency across the project
    df['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # 2. RSI (Relative Strength Index - 14 period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence - 12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 4. Bollinger Band Width (20 period, 2 StdDev)
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    upper_band = sma_20 + (std_20 * 2)
    lower_band = sma_20 - (std_20 * 2)
    # Width normalized by the SMA to make it comparable across different price levels
    df['bb_width'] = (upper_band - lower_band) / sma_20
    
    # 5. Volume ROC (Rate of Change over 5 periods)
    # Replaces infinite values with NaN to be dropped later
    df['vol_roc_5'] = df['volume'].pct_change(periods=5).replace([np.inf, -np.inf], np.nan)
    
    return df
