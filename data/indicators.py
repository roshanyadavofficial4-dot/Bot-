import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger("Indicators")

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Calculates the Supertrend indicator optimized for speed using numpy arrays.
    Returns the Supertrend line and the trend direction (1 for up, -1 for down).
    """
    hl2 = (df['high'] + df['low']) / 2
    atr = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=period
    ).average_true_range()
    
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)
    
    close = df['close'].values
    ub = basic_ub.values
    lb = basic_lb.values
    
    supertrend = np.zeros(len(df))
    direction = np.ones(len(df))
    
    # Initialize first values
    supertrend[0] = 0.0
    direction[0] = 1
    
    for i in range(1, len(df)):
        if close[i] > ub[i-1]:
            direction[i] = 1
        elif close[i] < lb[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
            # Adjust bands based on direction
            if direction[i] == 1 and lb[i] < lb[i-1]:
                lb[i] = lb[i-1]
            if direction[i] == -1 and ub[i] > ub[i-1]:
                ub[i] = ub[i-1]
                
        supertrend[i] = lb[i] if direction[i] == 1 else ub[i]
        
    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)

def add_mtf_trend_confirmation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples 15m OHLCV data to 1h to provide higher timeframe trend confirmation.
    Merges the 1h EMA trend back into the 15m dataframe using forward fill.
    """
    if 'timestamp' not in df.columns:
        logger.warning("Timestamp column missing. Cannot resample to 1h.")
        return df
        
    # Ensure timestamp is datetime and set as index for resampling
    df_temp = df.copy()
    df_temp.set_index('timestamp', inplace=True)
    
    # Resample to 1h
    resample_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_1h = df_temp.resample('1h').agg(resample_dict).dropna()
    
    # Calculate 1h Trend (EMA 9 vs EMA 21)
    ema_9_1h = ta.trend.EMAIndicator(close=df_1h['close'], window=9).ema_indicator()
    ema_21_1h = ta.trend.EMAIndicator(close=df_1h['close'], window=21).ema_indicator()
    
    # 1 if 1h trend is UP, -1 if DOWN
    df_1h['htf_trend_1h'] = np.where(ema_9_1h > ema_21_1h, 1, -1)
    
    # Merge back to 15m dataframe
    # We use merge_asof or reindex with ffill to map the 1h state to the 15m candles
    df_1h_trend = df_1h[['htf_trend_1h']]
    
    # Join back to the original dataframe
    df = df.merge(df_1h_trend, how='left', left_on='timestamp', right_index=True)
    df['htf_trend_1h'] = df['htf_trend_1h'].ffill().fillna(0) # Forward fill the 1h state
    
    return df

def apply_institutional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to apply all institutional-grade technical indicators and 
    custom ML features to the OHLCV DataFrame.
    """
    df = df.copy()
    
    # ==========================================
    # 1. TREND INDICATORS
    # ==========================================
    # Supertrend (10, 3)
    df['supertrend'], df['supertrend_dir'] = calculate_supertrend(df, period=10, multiplier=3.0)
    
    # EMAs
    df['ema_9'] = ta.trend.EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(close=df['close'], window=21).ema_indicator()
    
    # MACD Histogram
    macd = ta.trend.MACD(close=df['close'])
    df['macd_hist'] = macd.macd_diff()
    
    # ==========================================
    # 2. MOMENTUM INDICATORS
    # ==========================================
    # RSI
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # Stochastic RSI
    stoch_rsi = ta.momentum.StochRSIIndicator(close=df['close'], window=14, smooth1=3, smooth2=3)
    df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
    df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
    
    # ==========================================
    # 3. VOLATILITY INDICATORS
    # ==========================================
    # ATR
    df['atr_14'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14
    ).average_true_range()
    
    # Bollinger Bands %B
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_pband'] = bb.bollinger_pband()
    
    # ==========================================
    # 4. CUSTOM ML FEATURES
    # ==========================================
    # Volume Spikes (Current Vol / 20-candle Average Vol)
    avg_vol_20 = df['volume'].rolling(window=20).mean()
    df['volume_spike_ratio'] = np.where(avg_vol_20 > 0, df['volume'] / avg_vol_20, 1.0)
    
    # Candle Pattern Analysis
    df['is_green'] = (df['close'] > df['open']).astype(int)
    
    # Body Size % relative to open price
    df['body_size_percent'] = (abs(df['close'] - df['open']) / df['open']) * 100
    
    # Upper Shadow % relative to open price
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    df['upper_shadow_percent'] = (upper_shadow / df['open']) * 100
    
    # Distance from Support / Resistance (20-period rolling min/max)
    rolling_min_20 = df['low'].rolling(window=20).min()
    rolling_max_20 = df['high'].rolling(window=20).max()
    
    # Distance in percentage
    df['dist_from_support_pct'] = ((df['close'] - rolling_min_20) / rolling_min_20) * 100
    df['dist_from_resistance_pct'] = ((rolling_max_20 - df['close']) / rolling_max_20) * 100
    
    # ==========================================
    # 5. MULTI-TIMEFRAME INTEGRATION (MTF)
    # ==========================================
    df = add_mtf_trend_confirmation(df)
    
    # Drop rows with NaN values resulting from rolling windows
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df
