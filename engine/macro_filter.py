import logging
import pandas as pd
from typing import Literal

logger = logging.getLogger("MacroFilter")

class MacroFilter:
    """
    Phase 7: The God's Eye Macro Protocol.
    Monitors Bitcoin (the market leader) to detect flash crashes or heavy bearish momentum.
    Halts altcoin trading during 'DANGER' regimes to prevent catching falling knives.
    """
    
    async def check_macro_regime(self, exchange) -> Literal['SAFE', 'DANGER']:
        """
        Analyzes BTC/USDT:USDT 15m candles to determine the overarching market health.
        """
        symbol = 'BTC/USDT:USDT'
        timeframe = '15m'
        limit = 20
        
        try:
            # Fetch OHLCV data
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < limit:
                logger.warning(f"Insufficient data for {symbol} macro check.")
                return 'SAFE'
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 1. Calculate SMA-20
            sma_20 = df['close'].mean()
            current_price = df['close'].iloc[-1]
            
            # 2. Calculate Price Drop from Highest High in the window
            highest_high = df['high'].max()
            drop_pct = ((highest_high - current_price) / highest_high) * 100
            
            # Logic: Below SMA-20 AND Drop > 2.5% = DANGER
            is_below_sma = current_price < sma_20
            is_heavy_dump = drop_pct > 2.5
            
            if is_below_sma and is_heavy_dump:
                logger.warning(f"GOD'S EYE: BTC DANGER! Price: {current_price} | SMA20: {sma_20:.2f} | Drop: {drop_pct:.2f}%")
                return 'DANGER'
            
            return 'SAFE'
            
        except Exception as e:
            logger.error(f"Error in God's Eye Macro Protocol: {e}")
            return 'SAFE' # Default to SAFE on error to avoid halting incorrectly
