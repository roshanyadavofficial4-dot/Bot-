import os
import logging
import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
from datetime import datetime

logger = logging.getLogger("MistakeLogger")

class MistakeLogger:
    """
    Historical Trade Memory & Mistake Logger using ChromaDB.
    Stores market conditions as vector embeddings to find similar past setups
    and prevent the bot from repeating historical mistakes.
    """
    
    def __init__(self, db_path: str = "./data/chroma_db"):
        # Ensure the data directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
        
        # Get or create the collection for trade memory
        # We use cosine similarity to find similar market conditions (feature vectors)
        self.collection = self.client.get_or_create_collection(
            name="trade_memory",
            metadata={"hnsw:space": "cosine"}
        )
        self.expected_dim = 7 # Standard feature count for Project Digital Evolution
        logger.info(f"MistakeLogger initialized. Current memory size: {self.collection.count()} trades.")

    def log_trade(self, trade_id: str, entry_time: str, market_features: list, 
                  reason: str, sl: float, tp: float, pnl: float):
        """
        Logs a completed trade into the vector database.
        """
        if len(market_features) != self.expected_dim:
            logger.error(f"Dimension Mismatch: Expected {self.expected_dim}, got {len(market_features)}. Skipping log.")
            return

        try:
            # We use the numerical market features directly as the vector embedding
            # This allows us to mathematically search for similar past market conditions
            self.collection.add(
                ids=[str(trade_id)],
                embeddings=[market_features],
                metadatas=[{
                    "entry_time": entry_time,
                    "reason": reason,
                    "sl": sl,
                    "tp": tp,
                    "pnl": pnl,
                    "is_win": bool(pnl > 0)
                }]
            )
            logger.info(f"Trade {trade_id} logged to memory. PnL: {pnl:.2f}")
        except Exception as e:
            logger.error(f"Failed to log trade {trade_id} to memory: {e}")

    def search_mistakes(self, current_features: list, k: int = 5) -> dict:
        """
        Queries the database using current market conditions to find similar past trades.
        """
        if len(current_features) != self.expected_dim:
            logger.error(f"Dimension Mismatch: Expected {self.expected_dim}, got {len(current_features)}.")
            return {"memory_score": 0.0, "signal": "CLEAR", "message": "Dimension mismatch."}

        if self.collection.count() < k:
            # Not enough history to make a judgment
            return {
                "memory_score": 0.0,
                "signal": "CLEAR",
                "message": "Insufficient trade history for memory matching."
            }
            
        try:
            results = self.collection.query(
                query_embeddings=[current_features],
                n_results=k
            )
            
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]  # Cosine distances
            
            # Calculate Memory Score (Weighted average of past PnL based on similarity)
            # Closer distance = higher weight
            total_weight = 0.0
            weighted_pnl = 0.0
            win_count = 0
            
            for meta, dist in zip(metadatas, distances):
                # Avoid division by zero if distance is exactly 0
                weight = 1.0 / (dist + 1e-5) 
                pnl = meta['pnl']
                
                weighted_pnl += pnl * weight
                total_weight += weight
                
                if pnl > 0:
                    win_count += 1
                    
            memory_score = weighted_pnl / total_weight if total_weight > 0 else 0.0
            win_rate = win_count / k
            
            # Generate Signal for Risk Manager
            signal = "CLEAR"
            if memory_score < -0.02 or win_rate <= 0.2:
                # If similar past setups resulted in significant losses or terrible win rate
                signal = "WARNING"
                
            logger.info(f"Memory Search | Score: {memory_score:.4f} | Win Rate: {win_rate*100:.1f}% | Signal: {signal}")
            
            return {
                "memory_score": memory_score,
                "win_rate": win_rate,
                "signal": signal,
                "similar_trades_analyzed": k
            }
            
        except Exception as e:
            logger.error(f"Error searching mistakes: {e}")
            return {"memory_score": 0.0, "signal": "CLEAR", "error": str(e)}

    def refactor_memory(self):
        """
        Fetches all historical trades from the vector database and triggers
        a retraining cycle for the ML Price Predictor model.
        """
        logger.info("Initiating Memory Refactoring (Model Retraining)...")
        
        try:
            # Fetch all records (Note: For massive databases, pagination is required)
            all_data = self.collection.get(include=['embeddings', 'metadatas'])
            
            if not all_data['ids']:
                logger.warning("No data available for retraining.")
                return False
                
            # Construct a DataFrame for the ML model
            features = np.array(all_data['embeddings'])
            pnls = [meta['pnl'] for meta in all_data['metadatas']]
            
            # Target variable: 1 if trade was profitable, 0 if loss
            targets = np.array([1 if pnl > 0 else 0 for pnl in pnls])
            
            df = pd.DataFrame(features)
            df['target'] = targets
            df['pnl'] = pnls
            
            logger.info(f"Extracted {len(df)} historical trades for retraining.")
            
            # ---------------------------------------------------------
            # Trigger ML Model Retraining
            # ---------------------------------------------------------
            try:
                # Dynamically import to avoid circular dependencies if price_predictor imports this
                from models.price_predictor import retrain_model
                
                # Pass the historical feature matrix and targets to the model
                success = retrain_model(df)
                if success:
                    logger.info("Successfully refactored memory and retrained ML model.")
                    return True
                else:
                    logger.error("ML model retraining failed.")
                    return False
                    
            except ImportError:
                logger.warning("models.price_predictor module not found or lacks retrain_model(). Skipping actual retraining.")
                return False
                
        except Exception as e:
            logger.error(f"Error during memory refactoring: {e}")
            return False

# Example Usage:
# if __name__ == "__main__":
#     logger = MistakeLogger()
#     # Mock features: [RSI, EMA_dist, Volatility, ...]
#     logger.log_trade("TRD_001", "2026-03-26T10:00:00Z", [45.2, -0.5, 1.2], "RSI Oversold", 50000, 52000, -1.5)
#     logger.log_trade("TRD_002", "2026-03-26T11:00:00Z", [44.8, -0.6, 1.3], "RSI Oversold", 49000, 51000, -2.0)
#     
#     # Search current conditions
#     result = logger.search_mistakes([45.0, -0.55, 1.25])
#     print(result)
