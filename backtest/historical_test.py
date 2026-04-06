import logging
import ccxt
import pandas as pd
import numpy as np
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


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Standalone ATR calculation compatible with v5.1 Strategy API.
    Strategy.calculate_atr() returns a float (last value), not a Series.
    This helper returns a full Series for trailing SL updates.
    """
    high  = df['high']
    low   = df['low']
    close = df['close']
    tr1   = high - low
    tr2   = (high - close.shift(1)).abs()
    tr3   = (low  - close.shift(1)).abs()
    tr    = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _simple_trailing_sl(current_price: float, entry_price: float,
                         stop_loss: float, current_atr: float) -> float:
    """
    Simple trailing stop-loss update (mirrors RiskManager logic).
    RiskManager.update_trailing_sl() does not exist in v5.1 — that method
    was removed in the v4 refactor.  This standalone helper replicates
    the ATR-based trailing logic that the orchestrator uses internally.

    For a long (buy) trade:
        new_sl = max(old_sl, current_price - 1.5 * ATR)
    Only moves the SL upward (never down).
    """
    if current_atr <= 0:
        return stop_loss
    new_sl = current_price - 1.5 * current_atr
    return max(stop_loss, new_sl)


def run_backtest(symbol: str = 'BTC/USDT', days: int = 30):
    """
    Standalone Backtesting Engine (The Time Machine).
    Simulates the Strategy and Risk Manager over historical OHLCV data
    to calculate simulated PnL, Win Rate, and Max Drawdown.

    FIX: Updated to use v5.1 Strategy API:
      - Strategy.calculate_atr(df) returns a float — use _calculate_atr() helper instead
      - Strategy.analyze_pair() was removed in v5 — use generate_signal() / generate_signal_full()
      - RiskManager.update_trailing_sl() was removed — use _simple_trailing_sl() helper
      - RiskManager.calculate_position_size() is now async in v5 — use sync fallback here
    """
    logger.info(f"Starting {days}-day backtest for {symbol}...")

    exchange     = ccxt.binance({'enableRateLimit': True})
    strategy     = Strategy()
    risk_manager = RiskManager()

    # Fetch historical data (15m timeframe, 30 days ≈ 2880 candles)
    limit = days * 24 * 4
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=limit)
    except Exception as e:
        logger.error(f"Failed to fetch historical data: {e}")
        return

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Simulation State
    balance      = INITIAL_CAPITAL
    peak_balance = balance
    max_drawdown = 0.0
    wins         = 0
    losses       = 0
    in_position  = False
    entry_price  = 0.0
    stop_loss    = 0.0
    take_profit  = 0.0
    position_size = 0.0

    logger.info("Simulating historical price action...")

    # Start from index 50 to ensure enough data for indicators
    for i in range(50, len(df)):
        window_df      = df.iloc[:i + 1].copy()
        current_candle = window_df.iloc[-1]
        current_price  = current_candle['close']

        if in_position:
            # 1. Check Exits — evaluate SL against candle low, TP against candle high
            if current_candle['low'] <= stop_loss:
                exit_price  = stop_loss * (1 - 0.001)          # 0.1% slippage
                loss_amount = (entry_price - exit_price) * position_size
                exit_fee    = exit_price * position_size * 0.001
                balance    -= (loss_amount + exit_fee)
                losses     += 1
                in_position = False
                logger.info(f"STOP-LOSS HIT: Exit at {exit_price:.4f} (Slippage applied). Balance: ${balance:.2f}")

            elif current_candle['high'] >= take_profit:
                exit_price    = take_profit
                profit_amount = (exit_price - entry_price) * position_size
                exit_fee      = exit_price * position_size * 0.001
                balance      += (profit_amount - exit_fee)
                wins         += 1
                in_position   = False
                logger.info(f"TAKE-PROFIT HIT: Exit at {exit_price:.4f}. Balance: ${balance:.2f}")

            else:
                # 2. Update Trailing SL using standalone helper (v5.1 fix)
                atr_series = _calculate_atr(window_df, period=14)
                current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
                stop_loss = _simple_trailing_sl(current_price, entry_price, stop_loss, current_atr)

            # 3. Update Drawdown Metrics
            if balance > peak_balance:
                peak_balance = balance
            drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        else:
            # 4. Check Entries using v5.1 generate_signal_full() API
            # Build a minimal candidate dict from the OHLCV window
            atr_val  = strategy.calculate_atr(window_df)
            candidate = {
                'symbol':          symbol,
                'last_price':      current_price,
                'micro_price':     current_price,
                'imbalance':       0.0,
                'cvd':             0.0,
                'liquidity_depth': 10000.0,
                'spread_pct':      0.0001,
                'atr':             atr_val,
                'relative_strength': 0.0,
                'adx':             20.0,
                'ofi':             0.08,
                'wall_dist':       0.0,
                'btc_dom':         0.0,
                'btc_dom_roc':     0.0,
                'vol_ratio':       1.3,
                'arb_delta':       0.0,
                'velocity':        0.0,
                'funding_rate':    0.0,
                'h_volatility':    0.0,
                'trend_15m':       0.0,
                'vol_ema_ratio':   1.0,
                'price_range_pos': 0.5,
                'sentiment_proxy': 0.0,
                'funding_rate_velocity': 0.0,
                'liquidations_proxy': 0.0,
                'social_volume_spike': 0.0,
                'news_sentiment_score': 0.0,
                'btc_whale_tx_count':   0.0,
                'df':     window_df,
                'df_15m': window_df,
                'df_1h':  window_df,
                'df_4h':  window_df,
                'trend':  'UP',
            }

            try:
                result = strategy.generate_signal_full(candidate, total_trades=0,
                                                        account_balance=balance)
            except Exception as e:
                logger.debug(f"Signal generation skipped at bar {i}: {e}")
                result = None

            if result and result.signal == 'BUY':
                entry_price   = result.entry_price
                stop_loss     = result.stop_loss
                take_profit   = result.take_profit

                # Sync position sizing fallback (v5.1 calculate_position_size is async)
                risk_usd  = balance * risk_manager.current_risk
                risk_per_unit = abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else entry_price * 0.01
                pos_size  = risk_usd / risk_per_unit if risk_per_unit > 0 else 0.0

                if pos_size > 0:
                    entry_fee    = entry_price * pos_size * 0.001
                    balance     -= entry_fee
                    position_size = pos_size
                    in_position  = True
                    logger.info(f"ENTRY: {symbol} at {entry_price:.4f}. Size: {position_size:.4f}. Balance: ${balance:.2f}")

    # Calculate Final Metrics
    total_trades    = wins + losses
    win_rate        = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_return_pct = ((balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    logger.info("\n========================================")
    logger.info("           BACKTEST RESULTS             ")
    logger.info("========================================")
    logger.info(f"Symbol:           {symbol}")
    logger.info(f"Period:           {days} Days")
    logger.info(f"Starting Capital: ${INITIAL_CAPITAL:.2f}")
    logger.info(f"Ending Capital:   ${balance:.2f}")
    logger.info(f"Total Return:     {total_return_pct:.2f}%")
    logger.info(f"Max Drawdown:     {max_drawdown * 100:.2f}%")
    logger.info(f"Total Trades:     {total_trades}")
    logger.info(f"Win Rate:         {win_rate:.2f}% ({wins}W / {losses}L)")
    logger.info("========================================")


if __name__ == "__main__":
    run_backtest()
