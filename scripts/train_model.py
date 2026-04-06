import os
import sys
import time
import logging
import ccxt
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Ensure parent directory is in path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ml_features import apply_ml_features

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ML_Trainer")

def fetch_historical_data(symbol: str, timeframe: str, years: int) -> pd.DataFrame:
    """
    Robustly fetches historical OHLCV data using ccxt pagination.
    Respects exchange rate limits using time.sleep().
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Calculate timestamp for 'years' ago
    current_year = pd.Timestamp.utcnow().year
    start_year = current_year - years
    since = exchange.parse8601(f"{start_year}-01-01T00:00:00Z")
    now = exchange.milliseconds()
    
    all_ohlcv = []
    logger.info(f"Fetching {years} years of {timeframe} data for {symbol}...")
    
    while since < now:
        try:
            # Fetch in batches of 1000 (Binance limit)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
                
            # Update 'since' to the last fetched timestamp + 1ms to avoid duplicates
            since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            
            # Anti-ban rate limiting
            time.sleep(0.1)
            
        except ccxt.NetworkError as e:
            logger.warning(f"Network error: {e}. Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Successfully fetched {len(df)} candles.")
    return df

def train_evolution_model():
    """
    The core Machine Learning pipeline: Data Fetching -> Feature Engineering -> 
    Target Creation -> Training -> Persistence.
    """
    symbol = 'BTC/USDT'
    timeframe = '15m'
    look_forward_periods = 48  # Look ahead 12 hours (48 * 15m)
    pump_threshold = 0.03      # 3% pump target
    
    # 1. Fetch Data
    df = fetch_historical_data(symbol, timeframe, years=3)
    if df.empty:
        logger.error("No data fetched. Aborting training.")
        return
        
    # 2. Feature Engineering
    logger.info("Applying feature engineering (ATR, RSI, MACD, BB, Vol ROC)...")
    df = apply_ml_features(df)
    
    # 3. Target Variable Creation (Error 21 Fix - Avoiding Look-ahead Bias & NaN propagation)
    # Future_Pump = 1 if the maximum 'high' in the next N periods is >= 3% above current 'close'
    logger.info(f"Generating binary target (Future_Pump >= {pump_threshold*100}% in next {look_forward_periods} periods)...")
    
    # Calculate max high in the NEXT N periods (excluding current candle)
    # Using reverse rolling max with min_periods=1 to capture the most recent regime data
    df['future_max_high'] = df['high'].iloc[::-1].rolling(window=look_forward_periods, min_periods=1).max().iloc[::-1].shift(-1)
    
    # Binary Target: 1 if future max high >= 3% pump from current close
    df['future_pump'] = (df['future_max_high'] >= df['close'] * (1 + pump_threshold)).astype(int)
    
    # 4. Data Cleaning
    # Drop NaNs created by rolling windows (beginning of df) and the single NaN at the end from shift(-1)
    initial_len = len(df)
    # Drop only rows where features or target are NaN
    features = ['atr_14', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_width', 'vol_roc_5']
    df.dropna(subset=features + ['future_pump'], inplace=True)
    logger.info(f"Dropped {initial_len - len(df)} rows containing NaNs. Remaining rows: {len(df)}")
    
    X = df[features]
    y = df['future_pump']
    
    logger.info(f"Class distribution:\n{y.value_counts()}")
    
    # Split chronologically (no shuffling) to prevent data leakage in time-series
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # 5. Model Training
    logger.info("Training RandomForestClassifier...")
    # Using class_weight='balanced' because pump events (1) are usually rare compared to (0)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy on Test Set: {acc * 100:.2f}%")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # 7. Persistence
    os.makedirs("ml_models", exist_ok=True)
    model_path = "ml_models/evolution_v1.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_evolution_model()
