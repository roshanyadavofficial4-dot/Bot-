import pandas as pd
import numpy as np

def calculate_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates the Exponential Moving Average (EMA) and appends it to the DataFrame.
    """
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

def calculate_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI) using Wilder's Smoothing 
    and appends it to the DataFrame.
    """
    delta = df['close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Wilder's Smoothing (alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculates the Average True Range (ATR) using Wilder's Smoothing
    and appends it to the DataFrame.
    """
    # Calculate the three components of True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    # True Range is the maximum of the three
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Wilder's Smoothing for ATR
    df[f'atr_{period}'] = true_range.ewm(alpha=1/period, adjust=False).mean()
    
    return df
