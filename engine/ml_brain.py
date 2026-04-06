import sqlite3
import json
import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from datetime import datetime, timezone

logger = logging.getLogger("MLBrain")

class MLBrain:
    """
    Phase 106-111: Neural Overlord Upgrade.
    Dynamic Feature Selection & Nightly Hyperparameter Optimization.
    """
    def __init__(self, db_path="trades.db", model_path="model_cache_v15.joblib"):
        self.db_path = db_path
        self.model_path = model_path
        self.model = None
        self.feature_means = None
        self.feature_stds = None
        self.neural_efficiency = 1.0 # Phase 111: Neural Efficiency Score
        self.feature_names = [
            'spread_pct', 'atr', 'imbalance', 'micro_price', 'cvd', 
            'liquidity_depth', 'relative_strength', 'adx', 'ofi', 'btc_dom', 
            'vol_ratio', 'arb_delta', 'wall_dist', 'velocity', 'funding_rate', 
            'h_volatility', 'trend_15m', 'vol_ema_ratio', 'price_range_pos', 
            'sentiment_proxy', 'funding_rate_velocity', 'liquidations_proxy', 
            'social_volume_spike', 'news_sentiment_score', 'btc_whale_tx_count'
        ]
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                if isinstance(data, dict) and 'model' in data:
                    self.model = data['model']
                    self.feature_means = data.get('means')
                    self.feature_stds = data.get('stds')
                    self.active_feature_indices = data.get('active_indices', list(range(len(self.feature_names))))
                    self.neural_efficiency = data.get('neural_efficiency', 1.0)
                else:
                    self.model = data
                    self.active_feature_indices = list(range(len(self.feature_names)))
                logger.info(f"ML Model loaded. Efficiency: {self.neural_efficiency:.2f}x")
            except Exception as e:
                logger.error(f"Error loading ML model: {e}")
                self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            logger.info("ML Brain: Initialized with default parameters.")

    def _z_score_scaling(self, X, is_training=True):
        if is_training:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
        if self.feature_means is None: return X
        return (X - self.feature_means) / (self.feature_stds + 1e-8)

    def save_model(self):
        try:
            joblib.dump({
                'model': self.model,
                'means': self.feature_means,
                'stds': self.feature_stds,
                'active_indices': getattr(self, 'active_feature_indices', list(range(len(self.feature_names)))),
                'neural_efficiency': self.neural_efficiency,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, self.model_path)
            logger.info(f"ML Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving ML model: {e}")

    def optimize_hyperparameters(self):
        """
        Phase 111: Walk-Forward Optimization.
        Fetch last 200 real + shadow trades from DB and optimize.
        """
        try:
            if not os.path.exists(self.db_path): return
            
            conn = sqlite3.connect(self.db_path)
            # Fetch last 200 real trades (Excluding shadow trades as per Phase 133 to prevent drift)
            query_real = "SELECT pnl, market_features FROM trades WHERE exit_price > 0 AND is_shadow = 0 ORDER BY id DESC LIMIT 200"
            
            df = pd.read_sql_query(query_real, conn)
            conn.close()
            
            if len(df) < 50:
                logger.warning(f"Optimization skipped: Insufficient data ({len(df)} trades).")
                return

            features_list, labels = [], []
            for _, row in df.iterrows():
                try:
                    features = json.loads(row['market_features'])
                    f_vector = [features.get(name, 0.0) for name in self.feature_names]
                    features_list.append(f_vector)
                    labels.append(1 if row['pnl'] > 0 else 0)
                except Exception as e:
                    logger.warning(f"Skipping corrupt training row: {e}")
                    continue
            
            if len(features_list) < 50:
                logger.warning(f"Optimization skipped: Insufficient valid features ({len(features_list)} rows).")
                return

            X, y = np.array(features_list), np.array(labels)
            X_scaled = self._z_score_scaling(X, is_training=True)

            # Baseline Performance
            base_scores = cross_val_score(self.model, X_scaled, y, cv=3)
            base_mean_score = np.mean(base_scores)
            
            # Grid Search Optimization
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
            grid_search.fit(X_scaled, y)
            
            # Update Model & Neural Efficiency
            self.model = grid_search.best_estimator_
            self.neural_efficiency = (grid_search.best_score_ / base_mean_score) if base_mean_score > 0 else 1.0
            
            self.save_model()
            logger.info(f"Nightly Optimization Complete: {grid_search.best_params_} | Efficiency: {self.neural_efficiency:.2f}x")
            
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {e}")

    def predict_win_probability(self, current_features: dict) -> tuple:
        """
        Phase 106: Dynamic Feature Selection.
        Returns (win_prob, top_5_drivers).
        """
        if self.model is None: return 0.55, ["Model Not Ready"]
        
        try:
            f_vector = [current_features.get(name, 0.0) for name in self.feature_names]
            
            # Phase 106: Calculate Feature Drivers (Gini Importance)
            # Guard: model must be fitted before accessing feature_importances_
            top_drivers = ["Pending Training"]
            try:
                importances = self.model.feature_importances_
                top_drivers = [self.feature_names[i] for i in np.argsort(importances)[::-1][:5]]
            except Exception:
                pass
            
            X = np.array([f_vector])
            X_scaled = self._z_score_scaling(X, is_training=False)
            
            # Guard: model must be fitted to call predict_proba
            try:
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(self.model)
                probs = self.model.predict_proba(X_scaled)[0]
                win_prob = probs[1] if len(probs) > 1 else 0.5
            except Exception:
                win_prob = 0.55  # Default prior when model not yet trained
            
            return float(win_prob), top_drivers
        except Exception as e:
            logger.error(f"Error during ML prediction: {e}")
            return 0.55, ["Error"]

    def train_model(self):
        """Manual retraining trigger (Phase 49)."""
        try:
            if not os.path.exists(self.db_path): return
            conn = sqlite3.connect(self.db_path)
            # Phase 133: Ignore shadow trades in manual training
            query = "SELECT pnl, market_features FROM trades WHERE exit_price > 0 AND is_shadow = 0 ORDER BY id DESC LIMIT 500"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) < 20: return
            
            features_list, labels = [], []
            for _, row in df.iterrows():
                try:
                    features = json.loads(row['market_features'])
                    f_vector = [features.get(name, 0.0) for name in self.feature_names]
                    features_list.append(f_vector)
                    labels.append(1 if row['pnl'] > 0 else 0)
                except Exception as e:
                    logger.warning(f"Skipping corrupt training row: {e}")
                    continue
                
            X, y = np.array(features_list), np.array(labels)
            X_scaled = self._z_score_scaling(X, is_training=True)
            
            self.model.fit(X_scaled, y)
            self.save_model()
            logger.info("ML Brain: Model retrained successfully.")
        except Exception as e:
            logger.error(f"Manual training failed: {e}")
