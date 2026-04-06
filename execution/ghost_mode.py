import logging
from collections import deque

logger = logging.getLogger("GhostEngine")

class GhostTradingEngine:
    """
    Ghost Trading Engine.
    Simulates trades and tracks the PnL of the last N virtual trades.
    Acts as a circuit breaker: if recent virtual performance is negative, real trading is disabled.
    """
    def __init__(self, track_limit: int = 3):
        self.track_limit = track_limit
        # deque automatically removes the oldest item when maxlen is reached
        self.virtual_trades = deque(maxlen=track_limit)

    def log_virtual_trade(self, trade_id: str, pnl: float):
        """
        Logs a virtual trade's outcome.
        
        Args:
            trade_id (str): Unique identifier for the virtual trade.
            pnl (float): Profit/Loss percentage of the virtual trade.
        """
        try:
            self.virtual_trades.append({"trade_id": trade_id, "pnl": pnl})
            logger.info(f"Virtual Trade {trade_id} logged with PnL: {pnl:.2f}%")
        except Exception as e:
            logger.error(f"Error logging virtual trade: {e}")

    def is_real_trading_enabled(self) -> bool:
        """
        Evaluates the recent virtual trades to determine if real trading should proceed.
        
        Returns:
            bool: True if cumulative PnL is positive/neutral, False if negative.
        """
        try:
            if len(self.virtual_trades) < self.track_limit:
                # Not enough data to make a statistical decision, default to enabled
                logger.info(f"Ghost Engine: Warming up ({len(self.virtual_trades)}/{self.track_limit} trades). Real trading ENABLED.")
                return True
            
            total_pnl = sum(trade['pnl'] for trade in self.virtual_trades)
            
            if total_pnl < 0:
                logger.warning(f"Ghost Engine: Recent virtual PnL is negative ({total_pnl:.2f}%). Real trading DISABLED.")
                return False
            
            logger.info(f"Ghost Engine: Recent virtual PnL is positive ({total_pnl:.2f}%). Real trading ENABLED.")
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating ghost trades: {e}")
            # Fail-safe: disable trading if the engine crashes
            return False
