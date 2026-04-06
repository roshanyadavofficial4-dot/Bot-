import logging
import ccxt
import pandas as pd
import sys
import os

# Ensure parent directory is in path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.strategy import Strategy
from engine.risk_manager import RiskManager
from config import INITIAL_CAPITAL

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Backtester")

def run_backtest(symbol: str = 'BTC/USDT', days: int = 30):
    """
    Standalone Backtesting Engine (The Time Machine).
    Simulates the Strategy and Risk Manager over historical OHLCV data 
    to calculate simulated PnL, Win Rate, and Max Drawdown.
    """
    logger.info(f"Starting {days}-day backtest for {symbol}...")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    strategy = Strategy()
    risk_manager = RiskManager()
    
    # Fetch historical data (15m timeframe, 30 days ~ 2880 candles)
    limit = days * 24 * 4
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=limit)
    except Exception as e:
        logger.error(f"Failed to fetch historical data: {e}")
        return
        
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Simulation State
    balance = INITIAL_CAPITAL
    peak_balance = balance
    max_drawdown = 0.0
    
    wins = 0
    losses = 0
    
    in_position = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size = 0.0
    
    logger.info("Simulating historical price action...")
    
    # Iterate through historical data simulating real-time candle closes
    # Start from index 50 to ensure enough data for EMA/RSI/ATR calculation
    for i in range(50, len(df)):
        window_df = df.iloc[:i+1].copy()
        current_candle = window_df.iloc[-1]
        current_price = current_candle['close']
        
        if in_position:
            # 1. Check Exits (Stop-Loss or Take-Profit)
            # EVALUATE SL AGAINST CANDLE LOW (Error 14 Fix)
            if current_candle['low'] <= stop_loss:
                # Stop Loss Hit with 0.1% slippage penalty (Error 13 Fix)
                exit_price = stop_loss * (1 - 0.001)
                loss_amount = (entry_price - exit_price) * position_size
                
                # Apply 0.1% exit fee (Error 12 Fix)
                exit_fee = exit_price * position_size * 0.001
                balance -= (loss_amount + exit_fee)
                
                losses += 1
                in_position = False
                logger.info(f"STOP-LOSS HIT: Exit at {exit_price:.4f} (Slippage applied). Balance: ${balance:.2f}")
                
            # EVALUATE TP AGAINST CANDLE HIGH (Error 14 Fix)
            elif current_candle['high'] >= take_profit:
                # Take Profit Hit
                exit_price = take_profit
                profit_amount = (exit_price - entry_price) * position_size
                
                # Apply 0.1% exit fee (Error 12 Fix)
                exit_fee = exit_price * position_size * 0.001
                balance += (profit_amount - exit_fee)
                
                wins += 1
                in_position = False
                logger.info(f"TAKE-PROFIT HIT: Exit at {exit_price:.4f}. Balance: ${balance:.2f}")
            else:
                # 2. Update Trailing SL
                current_atr = strategy._calculate_atr(window_df, strategy.atr_period).iloc[-1]
                stop_loss = risk_manager.update_trailing_sl(current_price, entry_price, stop_loss, current_atr)
                
            # 3. Update Drawdown Metrics
            if balance > peak_balance:
                peak_balance = balance
            drawdown = (peak_balance - balance) / peak_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        else:
            # 4. Check Entries
            signal_data = strategy.analyze_pair(symbol, window_df)
            if signal_data['signal'] == 'BUY':
                entry_price = signal_data['entry_price']
                stop_loss = signal_data['stop_loss']
                take_profit = signal_data['take_profit']
                
                position_size = risk_manager.calculate_position_size(balance, entry_price, stop_loss)
                if position_size > 0:
                    # Apply 0.1% entry fee (Error 12 Fix)
                    entry_fee = entry_price * position_size * 0.001
                    balance -= entry_fee
                    in_position = True
                    logger.info(f"ENTRY: {symbol} at {entry_price:.4f}. Size: {position_size:.4f}. Balance: ${balance:.2f}")
    
    # Calculate Final Metrics
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_return_pct = ((balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    logger.info("\n========================================")
    logger.info("           BACKTEST RESULTS             ")
    logger.info("========================================")
    logger.info(f"Symbol:           {symbol}")
    logger.info(f"Period:           {days} Days")
    logger.info(f"Starting Capital: ${INITIAL_CAPITAL:.2f}")
    logger.info(f"Ending Capital:   ${balance:.2f}")
    logger.info(f"Total Return:     {total_return_pct:.2f}%")
    logger.info(f"Max Drawdown:     {max_drawdown*100:.2f}%")
    logger.info(f"Total Trades:     {total_trades}")
    logger.info(f"Win Rate:         {win_rate:.2f}% ({wins}W / {losses}L)")
    logger.info("========================================")

if __name__ == "__main__":
    run_backtest()
