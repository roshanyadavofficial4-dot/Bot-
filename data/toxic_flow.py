import logging

logger = logging.getLogger("ToxicFlow")

class OrderFlowToxicityChecker:
    """
    Order Flow Toxicity Checker.
    Monitors short-term volume to detect whale dumps or extreme sell pressure.
    """
    def __init__(self, threshold_multiplier: float = 3.0):
        # 3.0 means 300% of average volume
        self.threshold_multiplier = threshold_multiplier

    def check_toxicity(self, current_sell_volume: float, average_volume: float) -> str:
        """
        Analyzes the latest 1-minute candle volume.
        
        Args:
            current_sell_volume (float): The sell volume in the current/last 1m candle.
            average_volume (float): The rolling average volume (e.g., over the last 20 mins).
            
        Returns:
            str: 'EMERGENCY_EXIT' if toxic, 'SAFE' otherwise.
        """
        try:
            if average_volume <= 0:
                # Avoid division by zero if there's no historical volume
                return "SAFE"
                
            spike_ratio = current_sell_volume / average_volume
            
            if spike_ratio >= self.threshold_multiplier:
                logger.critical(
                    f"TOXIC FLOW DETECTED! Sell volume spike: {spike_ratio*100:.1f}% of average. "
                    f"(Current: {current_sell_volume:.2f}, Avg: {average_volume:.2f})"
                )
                return "EMERGENCY_EXIT"
                
            logger.debug(f"Order flow normal. Spike ratio: {spike_ratio*100:.1f}%")
            return "SAFE"
            
        except Exception as e:
            logger.error(f"Error checking order flow toxicity: {e}")
            # Default to SAFE if calculation fails to avoid false positives
            return "SAFE"
