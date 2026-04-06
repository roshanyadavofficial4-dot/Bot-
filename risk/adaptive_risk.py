import logging

logger = logging.getLogger("AdaptiveRisk")

class ImmuneSystemRiskManager:
    """
    Immune System Risk Manager.
    Adapts position sizing and stop-loss distances based on real-time market volatility.
    """
    def __init__(self, base_sl_pct: float = 0.015, fever_multiplier: float = 2.0):
        self.base_sl_pct = base_sl_pct        # Default Stop Loss distance (e.g., 1.5%)
        self.fever_multiplier = fever_multiplier # Multiplier for SL distance during high volatility

    def calculate_position(self, account_balance: float, risk_per_trade_pct: float, 
                           current_price: float, current_volatility: float, volatility_threshold: float,
                           is_long: bool = True) -> dict:
        """
        Calculates the adaptive position size and stop loss.
        
        Args:
            account_balance (float): Total account capital.
            risk_per_trade_pct (float): Percentage of account to risk (e.g., 0.01 for 1%).
            current_price (float): Current asset price.
            current_volatility (float): Current ATR or StdDev.
            volatility_threshold (float): Threshold above which 'fever mode' activates.
            is_long (bool): True for Long positions, False for Short.
            
        Returns:
            dict: Contains position_size, sl_price, and mode ('NORMAL' or 'FEVER').
        """
        try:
            # Constant Dollar Risk (e.g., $1000 * 0.01 = $10 risk)
            dollar_risk = account_balance * risk_per_trade_pct
            
            # Determine if market is in 'Fever Mode' (High Volatility)
            is_fever_mode = current_volatility > volatility_threshold
            
            if is_fever_mode:
                # Double the SL distance to avoid noise stop-outs
                sl_distance_pct = self.base_sl_pct * self.fever_multiplier
                mode = "FEVER"
                logger.warning(f"FEVER MODE ACTIVATED: Volatility ({current_volatility:.4f}) > Threshold ({volatility_threshold:.4f})")
            else:
                sl_distance_pct = self.base_sl_pct
                mode = "NORMAL"
                
            # Calculate Stop Loss Price
            if is_long:
                sl_price = current_price * (1 - sl_distance_pct)
                price_risk_per_unit = current_price - sl_price
            else:
                sl_price = current_price * (1 + sl_distance_pct)
                price_risk_per_unit = sl_price - current_price
            
            # Calculate Position Size (Units of asset)
            # Position Size = Dollar Risk / Risk Per Unit
            # If Fever Mode is active, price_risk_per_unit is doubled, naturally halving the position size!
            position_size = dollar_risk / (price_risk_per_unit + 1e-9) if price_risk_per_unit > 0 else 0.0
            
            logger.info(
                f"Risk Calc | Mode: {mode} | SL Dist: {sl_distance_pct*100:.1f}% | "
                f"Pos Size: {position_size:.4f} | Risk: ${dollar_risk:.2f}"
            )
            
            return {
                "mode": mode,
                "position_size": position_size,
                "sl_price": sl_price,
                "sl_distance_pct": sl_distance_pct,
                "dollar_risk": dollar_risk
            }
            
        except Exception as e:
            logger.error(f"Error calculating adaptive risk: {e}")
            return {}
