import os
import sys
import logging
import joblib
import pandas as pd

# Ensure parent directory is in path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ml_features import apply_ml_features

logger = logging.getLogger("AIPredictor")

class AIPredictor:
    """
    Live Inference Engine for Project Digital Evolution.
    Loads the pre-trained ML model and provides real-time win probabilities 
    based on live OHLCV data.
    """
    
    def __init__(self, model_path: str = "ml_models/evolution_v1.pkl"):
        self.model_path = model_path
        self.model = None
        # Must perfectly match the features used in train_model.py
        self.features = ['atr_14', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_width', 'vol_roc_5']
        self._load_model()

    def _load_model(self):
        """Loads the serialized scikit-learn model from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"ML Model not found at {self.model_path}. Running in MOCK mode (0.5-0.8 range) for development.")
            return
            
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Successfully loaded ML model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")

    def get_win_probability(self, df: pd.DataFrame) -> float:
        """
        Applies exact feature engineering to the live dataframe and returns 
        the probability of a pump (Class 1).
        """
        if self.model is None:
            # Mock fallback for development/testing if model isn't trained yet
            # In production, this would return 0.0 or raise an error
            import random
            return round(random.uniform(0.5, 0.8), 4)
            
        if len(df) < 30:
            logger.warning("Insufficient data for ML feature engineering.")
            return 0.0

        try:
            # 1. Apply exact same feature engineering used in training
            df_features = apply_ml_features(df)
            
            # 2. Extract the latest row (current market state)
            latest_row = df_features.iloc[-1:]
            
            # 3. Handle NaNs
            if latest_row[self.features].isnull().values.any():
                return 0.0
                
            X_live = latest_row[self.features]
            
            # 4. Predict Probability
            probabilities = self.model.predict_proba(X_live)
            pump_probability = probabilities[0][1]
            
            return round(pump_probability, 4)
            
        except Exception as e:
            logger.error(f"Error during ML inference: {e}")
            return 0.0
