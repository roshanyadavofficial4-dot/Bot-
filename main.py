import asyncio
import logging
import math
from logging.handlers import RotatingFileHandler
import json
import aiosqlite
import ccxt.pro as ccxt_pro
import pandas as pd
from datetime import datetime, timezone
import time
import os
import ccxt
import signal
import sys

# Internal Modules
from config import (
    BINANCE_API_KEY, BINANCE_SECRET, USE_SANDBOX,
    INITIAL_CAPITAL, BASE_ASSET, SCAN_INTERVAL,
    TRADE_START_HOUR_UTC, TRADE_END_HOUR_UTC,
    MAX_ATR_VOLATILITY,
    ALLOWED_SYMBOLS,
    DEFAULT_LEVERAGE, MODE,
    MAX_LEVERAGE, MIN_NOTIONAL, MIN_NOTIONAL_BUFFER,
    STOP_LOSS_PCT, MIN_VOLUME_SPIKE,
    MAX_TRADES_PER_DAY,
)
from engine.scanner import Scanner
from engine.strategy import Strategy
from engine.risk_manager import RiskManager
from engine.macro_filter import MacroFilter
from engine.executor import OrderExecutor
from engine.watchdog import SystemWatchdog
from engine.global_state import BotState
from engine.regime_detector import RegimeDetector
from engine.capital_adaptive import get_adaptive_params   # v4.1: dynamic capital scaling
from engine.execution_model import estimate_round_trip_cost, adjust_price_for_slippage
# Phase 3+4: Intelligent Orchestrator (drop-in replacement for Phase 2 Orchestrator)
# Adds 17-gate pipeline with self-learning, anomaly detection, capital growth management
from engine.phase3_4_orchestrator import IntelligentOrchestrator
from core.notifier import TelegramNotifier
from core.trade_logger import TradeLogger
from core.dashboard_api import update_bot_state
import uvicorn

# Configure logging
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_file = "digital_evolution_v3.log"
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Global state
global_state = BotState()
live_prices = {} # {symbol: (price, timestamp)}
price_lock = asyncio.Lock()
trade_logger = TradeLogger()
daily_state = {"date": "", "balance": 0, "trade_count": 0}
strategy = Strategy()
regime_detector = RegimeDetector()   # v4: global regime detector
orchestrator    = IntelligentOrchestrator(balance=INITIAL_CAPITAL)  # Phase 3+4: intelligent decision authority

async def ws_watchdog(exchange, exchange_id, notifier):
    """
    Phase 4: Logical Deadlock Fix
    Monitors WebSocket heartbeats and forces re-initialization if data is stale.
    """
    logger.info(f"WS Watchdog for {exchange_id} started.")
    while True:
        try:
            await asyncio.sleep(10)
            last_heartbeat = global_state.last_ws_heartbeat.get(exchange_id, 0)
            if time.time() - last_heartbeat > 8: # 8 seconds staleness threshold
                logger.critical(f"WS STALENESS DETECTED for {exchange_id}! Forcing re-initialization...")
                await notifier.send_message(f"⚠️ *WS STALENESS DETECTED*: {exchange_id} data is stale (>8s). Reconnecting...")
                try:
                    await exchange.close()
                except Exception as e:
                    logger.warning(f"WS close error (non-critical): {e}")
        except Exception as e:
            logger.error(f"WS Watchdog Error ({exchange_id}): {e}")

async def db_cleanup_task():
    """Phase 4: Memory Leak Prevention - Periodic DB cleanup."""
    logger.info("DB Cleanup Task started.")
    while True:
        try:
            await asyncio.sleep(300) # Every 5 minutes
            await trade_logger.cleanup_old_trades(days=7)
        except Exception as e:
            logger.error(f"DB Cleanup Task Error: {e}")

async def safe_fetch(func, *args, **kwargs):
    attempts = 0
    while attempts < 5:
        try:
            return await func(*args, **kwargs)
        except ccxt.RateLimitExceeded:
            attempts += 1
            await asyncio.sleep(2 ** attempts)
        except Exception as e:
            raise e
    raise ccxt.RateLimitExceeded("Max retries exceeded")

async def order_watcher(exchange: ccxt_pro.binance, trade_logger, notifier, executor, scanner):
    """
    Phase 4: Math & Predictive Skew Fix
    Listens to Binance User Data Stream to log trades ONLY when filled.
    Prevents Look-Ahead Bias in ML Brain.
    """
    logger.info("Order Watcher (Binance) started.")
    consecutive_failures = 0
    while True:
        try:
            orders = await asyncio.wait_for(exchange.watch_orders(), timeout=5.0)
            global_state.last_ws_heartbeat['binance_user'] = time.time()
            consecutive_failures = 0 # Reset on success
            for order in orders:
                order_id = order['id']
                client_id = order.get('clientOrderId')
                status = order['status']
                
                # Check if this is a pending entry
                pending_data = await global_state.pop_pending_entry(client_id) or await global_state.pop_pending_entry(order_id)
                
                if pending_data:
                    if status == 'closed':
                        symbol = order['symbol']
                        side = order['side']
                        filled_qty = order['filled']
                        avg_price = order['average'] or order['price']
                        
                        # True Latency Calculation
                        signal_ts = pending_data.get('signal_ts')
                        if signal_ts:
                            true_latency_ms = (time.time() - signal_ts) * 1000
                            logger.info(f"TRUE EXECUTION LATENCY (Signal to Fill): {true_latency_ms:.2f}ms")
                        
                        # Now log to ML Brain as active trade
                        trade_info = pending_data['trade_info']
                        trade_info['entry_price'] = avg_price
                        trade_info['qty'] = filled_qty
                        trade_info['timestamp'] = datetime.now(timezone.utc).isoformat()
                        
                        logged_trade = await trade_logger.log_trade(trade_info)
                        trade_id = logged_trade.get('id', 0)
                        
                        # Increment daily trade count
                        await global_state.increment_trade_count()
                        
                        # v4: cooldown is recorded by run_bot after order placement
                        
                        # Only directional trades use SL/TS and Monitor
                        if trade_info.get('type') == 'directional':
                            # Place SL and TS now that we are filled
                            atr = pending_data['atr']
                            stop_loss = avg_price - (1.5 * atr) if side == 'buy' else avg_price + (1.5 * atr)
                            sl_side = 'sell' if side == 'buy' else 'buy'
                            
                            await executor.execute_stop_loss(symbol, sl_side, filled_qty, stop_loss)
                            
                            activation_price = avg_price + (1.5 * atr) if side == 'buy' else avg_price - (1.5 * atr)
                            try:
                                await executor.execute_trailing_stop(symbol, sl_side, filled_qty, activation_price, 1.5)
                            except Exception as e:
                                logger.error(f"Trailing Stop Failed in Watcher: {e}")
                            
                            # Start Position Monitor
                            task = asyncio.create_task(monitor_position(
                                symbol, avg_price, filled_qty, atr, exchange,
                                executor, trade_logger, notifier, trade_id, scanner,
                                risk_mgr=risk_manager
                            ))
                            await global_state.add_active_monitor(symbol, task)
                        
                        await notifier.send_message(f"✅ *ORDER FILLED & LOGGED*\nSymbol: {symbol}\nPrice: {avg_price}\nType: {trade_info.get('type', 'unknown')}")
                    
                    elif status in ['canceled', 'rejected', 'expired']:
                        logger.warning(f"❌ PENDING ENTRY FAILED: {order['symbol']} status {status}")
                        await notifier.send_message(f"⚠️ *ORDER {status.upper()}*\nSymbol: {order['symbol']}\nTrade entry failed.")
                    
                    else:
                        # Order still open, put it back
                        await global_state.add_pending_entry(client_id or order_id, pending_data)
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Order Watcher Error ({consecutive_failures}/3): {e}")
            if consecutive_failures >= 3:
                logger.critical("Circuit breaker triggered for Order Watcher. Restarting task...")
                raise e # Let watchdog restart
            await asyncio.sleep(5 * consecutive_failures) # Exponential backoff

async def price_tracker(exchange: ccxt_pro.binance):
    from config import ALLOWED_SYMBOLS
    # Phase 133: Dynamic WebSocket Sync - Extracting base symbol like 'DOGE/USDT' from 'DOGE/USDT:USDT'
    symbols = [s.split(':')[0] for s in ALLOWED_SYMBOLS]
    consecutive_failures = 0
    while True:
        try:
            tickers = await asyncio.wait_for(exchange.watch_tickers(symbols), timeout=5.0)
            global_state.last_ws_heartbeat['binance_futures'] = time.time()
            consecutive_failures = 0 # Reset on success
            async with price_lock:
                for symbol, ticker in tickers.items():
                    if ticker and 'last' in ticker:
                        live_prices[symbol] = (ticker['last'], datetime.now(timezone.utc))
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"WebSocket Error (Price Tracker) ({consecutive_failures}/3): {e}")
            if consecutive_failures >= 3:
                logger.critical("Circuit breaker triggered for Price Tracker. Restarting task...")
                raise e # Let watchdog restart
            await asyncio.sleep(5 * consecutive_failures) # Exponential backoff

async def get_cached_balance(exchange):
    now = time.time()
    cached = await global_state.get_cache('balance')
    if cached and cached['data'] and now - cached['timestamp'] < 8:
        return cached['data']
    
    data = await safe_fetch(exchange.fetch_balance)
    await global_state.set_cache('balance', data, now)
    return data

async def get_cached_positions(exchange):
    now = time.time()
    cached = await global_state.get_cache('positions')
    if cached and cached['data'] and now - cached['timestamp'] < 8:
        return cached['data']
    
    data = await safe_fetch(exchange.fetch_positions)
    await global_state.set_cache('positions', data, now)
    return data

async def get_fresh_price(exchange, symbol: str) -> float:
    async with price_lock:
        data = live_prices.get(symbol)
    if data:
        price, ts = data
        if (datetime.now(timezone.utc) - ts).total_seconds() <= 2.0: return price
    ticker = await safe_fetch(exchange.fetch_ticker, symbol)
    return ticker['last']

async def ninja_sweeper_protocol(exchange, current_balance, notifier):
    """
    Phase 3: The Ninja Sweeper Protocol.
    Automatically sweeps excess capital to Spot wallet when target is hit.
    """
    if current_balance >= 1200.0:
        try:
            sweep_amount = current_balance - INITIAL_CAPITAL  # FIX: use INITIAL_CAPITAL (~$2.4) not hardcoded $12
            # CCXT internal transfer: asset, amount, from, to
            await safe_fetch(exchange.transfer, BASE_ASSET, sweep_amount, 'future', 'spot')
            
            msg = (
                "🚨 *NINJA SWEEP EXECUTED* 🚨\n"
                "Target of $1,200 HIT!\n"
                f"Transferred ${sweep_amount:,.2f} to Spot Wallet.\n"
                "Futures balance reset to $12. Loop restarting."
            )
            await notifier.send_message(msg)
            logger.info(f"NINJA SWEEP: {sweep_amount} transferred to spot.")
            return True
        except Exception as e:
            logger.error(f"NINJA SWEEP FAILED: {e}")
            await notifier.send_message(f"⚠️ *NINJA SWEEP FAILED*\nError: {e}")
    return False

async def shutdown(sig, signal_loop, tasks_dict, exchange):
    logger.info("Initiating graceful shutdown...")
    try:
        await exchange.close()
    except Exception as e:
        logger.error(f"Error closing exchange: {e}")
        
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
        
    await asyncio.gather(*tasks, return_exceptions=True)
    signal_loop.stop()

async def monitor_position(symbol: str, entry_price: float, pos_size: float, atr: float, exchange, executor, trade_logger, notifier, trade_id: int, scanner: Scanner, risk_mgr=None):
    """
    Phase 20: The Regime Adaptive Matrix
    Active Position Monitor for Stage 1 Scale-out.
    """
    # Phase 46: Sniper Protocol - Strictly 1.5% move for PTP
    from engine.risk_manager import RiskManager
    rm = RiskManager()
    
    # Calculate PTP Target based on side
    if pos_size > 0:
        ptp_target = entry_price * 1.015 
    else:
        ptp_target = entry_price * 0.985
        
    half_qty = abs(pos_size) * 0.5
    open_time = time.time()
    
    # Phase 131: Track current size and PTP status
    current_size = abs(pos_size)
    ptp_hit = False
    
    try:
        while True:
            # Phase 131: Global Time Kill-Switch (VERY TOP)
            # Close after ~4 hours if no progress
            if time.time() - open_time > 14400:
                logger.warning(f"⏰ 4-Hour Time Kill Switch Activated - Closing Position for {symbol}")
                exit_side = 'sell' if pos_size > 0 else 'buy'
                await executor.execute_trade(symbol, exit_side, current_size)
                current_price = await get_fresh_price(exchange, symbol)
                pnl = (current_price - entry_price) * current_size if pos_size > 0 else (entry_price - current_price) * current_size
                pnl_pct = (current_price - entry_price) / entry_price if pos_size > 0 else (entry_price - current_price) / entry_price
                await trade_logger.close_trade(trade_id, current_price, pnl, pnl_pct)
                if risk_mgr:
                    risk_mgr.update_risk_profile(pnl > 0)
                    risk_mgr.record_pnl(pnl, pnl_pct)   # v4
                break

            try:
                current_price = await get_fresh_price(exchange, symbol)
                
                # Phase 131: If PTP hit, monitor for kill switch or exchange-side exit
                if ptp_hit:
                    positions = await get_cached_positions(exchange)
                    active_symbols = {p['symbol'] for p in positions if float(p.get('contracts', 0)) > 0}
                    if symbol not in active_symbols:
                        logger.info(f"Position for {symbol} closed on exchange (SL/TS). Stopping monitor.")
                        break
                
                # BUY Position Exit Logic
                elif pos_size > 0:
                    if current_price >= ptp_target:
                        logger.info(f"STAGE 1 COMPLETE: PTP Target hit for {symbol} at {current_price}. Scaling out 50%.")
                        
                        # 1. Close 50%
                        await executor.close_partial_position(symbol, 'sell', half_qty)
                        
                        # Calculate PnL for logging
                        pnl = (current_price - entry_price) * half_qty
                        pnl_pct = (current_price - entry_price) / entry_price
                        
                        # Log partial exit
                        await trade_logger.close_trade(trade_id, current_price, pnl, pnl_pct, partial_qty=half_qty)
                        
                        # 2. Cancel old SL and Trailing Stop
                        await executor.cancel_all_orders(symbol)
                        
                        # 3. Move SL to Break-Even for remaining 50%
                        await executor.execute_stop_loss(symbol, 'sell', half_qty, entry_price)
                        
                        # 4. Re-issue Trailing Stop for the remaining 50%
                        # BOOST MODE: Widen trailing stop to 2.0% to capture 4-6% moves
                        try:
                            await executor.execute_trailing_stop(symbol, 'sell', half_qty, ptp_target, 2.0)
                        except Exception as e:
                            logger.error(f"Trailing Stop Re-issue Failed: {e}")
                        
                        # 5. Mark PTP as hit in DB
                        await trade_logger.mark_ptp_hit(trade_id)
                        await notifier.send_message(f"💸 *STAGE 1 COMPLETE (50%)*\nSymbol: {symbol}\nPrice: {current_price}\nSL moved to Break-Even.")
                        
                        current_size = half_qty
                        ptp_hit = True
                        continue 

                # SELL Position Exit Logic
                elif pos_size < 0:
                    if current_price <= ptp_target:
                        logger.info(f"STAGE 1 COMPLETE: PTP Target hit for {symbol} at {current_price}. Scaling out 50%.")
                        
                        # 1. Close 50%
                        await executor.close_partial_position(symbol, 'buy', half_qty)
                        
                        # Calculate PnL for logging
                        pnl = (entry_price - current_price) * half_qty
                        pnl_pct = (entry_price - current_price) / entry_price
                        
                        # Log partial exit
                        await trade_logger.close_trade(trade_id, current_price, pnl, pnl_pct, partial_qty=half_qty)
                        
                        # 2. Cancel old SL and Trailing Stop
                        await executor.cancel_all_orders(symbol)
                        
                        # 3. Move SL to Break-Even for remaining 50%
                        await executor.execute_stop_loss(symbol, 'buy', half_qty, entry_price)
                        
                        # 4. Re-issue Trailing Stop for the remaining 50%
                        # BOOST MODE: Widen trailing stop to 2.0% to capture 4-6% moves
                        try:
                            await executor.execute_trailing_stop(symbol, 'buy', half_qty, ptp_target, 2.0)
                        except Exception as e:
                            logger.error(f"Trailing Stop Re-issue Failed: {e}")
                        
                        # 5. Mark PTP as hit in DB
                        await trade_logger.mark_ptp_hit(trade_id)
                        await notifier.send_message(f"💸 *STAGE 1 COMPLETE (50%)*\nSymbol: {symbol}\nPrice: {current_price}\nSL moved to Break-Even.")
                        
                        current_size = half_qty
                        ptp_hit = True
                        continue 
                    
                # Phase 128: Adaptive Sleep (Check faster if in profit)
                in_profit = (current_price > entry_price) if pos_size > 0 else (current_price < entry_price)
                await asyncio.sleep(0.5 if in_profit else 1.5)
            except ccxt.NetworkError:
                logger.warning(f"Network error in monitor_position for {symbol}, retrying...")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                logger.error(f"Error in monitor_position for {symbol}: {e}")
                break
    finally:
        active_monitors = await global_state.get_active_monitors()
        if symbol in active_monitors:
            await global_state.remove_active_monitor(symbol)

async def run_bot():
    global daily_state
    logger.info("--- Digital Evolution v3.4 Institutional High-Conviction ---")
    
    # Phase 81: God-Eye Cluster Initialization
    # In a real scenario, these would be loaded from a secure config/env
    exchanges = []
    try:
        # Phase 130: Binance-only initialization
        binance_config = {'apiKey': BINANCE_API_KEY, 'secret': BINANCE_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'future'}}
        binance = ccxt_pro.binance(binance_config)
        if USE_SANDBOX: binance.set_sandbox_mode(True)
        exchanges.append(binance)
    except Exception as e:
        logger.error(f"Failed to initialize multi-exchange cluster: {e}")
        if not exchanges: return

    exchange = binance # Primary for data fetching
    
    # Phase 135: Market Loading & Safety Setup
    try:
        await exchange.load_markets()
        logger.info("Markets loaded successfully.")
        
        # Set Isolated Margin and Leverage for all allowed symbols at startup
        for symbol in ALLOWED_SYMBOLS:
            try:
                # Use the base symbol for margin/leverage if needed, but CCXT usually handles it
                await exchange.set_margin_mode('ISOLATED', symbol)
                await exchange.set_leverage(DEFAULT_LEVERAGE, symbol)
                logger.info(f"Safety Setup: {symbol} set to ISOLATED {DEFAULT_LEVERAGE}x")
            except Exception as e:
                logger.warning(f"Safety Setup failed for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Failed to load markets or setup safety: {e}")
        return

    # Initialize DB
    await trade_logger.init_db()
    await global_state.init_db()
    
    # Initialize Scanner & Strategy
    scanner = Scanner(exchange, trade_logger)
    # strategy = Strategy() # Now global
    risk_manager = RiskManager()
    macro_filter = MacroFilter()
    executor = OrderExecutor(exchanges)
    notifier = TelegramNotifier(executor, trade_logger, global_state, strategy)
    notifier.start_polling()
    watchdog = SystemWatchdog()
    
    # Immortality Protocol (Phase 8): Ghost Reconciliation
    await watchdog.reconcile_ghost_positions(exchange, trade_logger, notifier)
    
    tracker_task = asyncio.create_task(price_tracker(exchange))
    watcher_task = asyncio.create_task(order_watcher(exchange, trade_logger, notifier, executor, scanner))
    
    # Phase 4: Background Tasks for Stability
    asyncio.create_task(ws_watchdog(exchange, 'binance_futures', notifier))
    asyncio.create_task(ws_watchdog(exchange, 'binance_user', notifier))
    asyncio.create_task(db_cleanup_task())
    
    tasks_dict = {
        'price_tracker': {'task': tracker_task, 'func': price_tracker, 'args': (exchange,)},
        'order_watcher': {'task': watcher_task, 'func': order_watcher, 'args': (exchange, trade_logger, notifier, executor, scanner)}
    }
    asyncio.create_task(watchdog.monitor_tasks(tasks_dict, exchange))
    
    main_loop = asyncio.get_running_loop()
    if os.name != 'nt':
        for sig in (signal.SIGINT, signal.SIGTERM):
            main_loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s, main_loop, tasks_dict, exchange)))
            
    last_report_date = datetime.now(timezone.utc).date()
    
    # Phase 130: Native Async Dashboard Boot
    config = uvicorn.Config("core.dashboard_api:app", host="0.0.0.0", port=3000, log_level="error")
    server = uvicorn.Server(config)
    
    # Phase 133/134: Uvicorn Crash Handling with Retry Limit
    async def start_dashboard():
        retries = 0
        while retries < 3:
            try:
                await server.serve()
                break # If it exits normally
            except Exception as e:
                retries += 1
                logger.error(f"DASHBOARD CRASH (Attempt {retries}/3): {e}")
                if retries >= 3:
                    logger.critical("Dashboard failed permanently.")
                    break
                await asyncio.sleep(5)
            
    asyncio.create_task(start_dashboard())
    logger.info("Dashboard API started on port 3000.")

    while True:
        try:
            # Phase 101: Bi-Directional Control
            if global_state.EMERGENCY_EXIT_TRIGGERED:
                logger.warning("EMERGENCY EXIT TRIGGERED via Telegram.")
                positions = await safe_fetch(exchange.fetch_positions)
                active_fut_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]
                for pos in active_fut_positions:
                    symbol = pos['symbol']
                    qty = float(pos['contracts'])
                    side = 'sell' if pos['side'] == 'long' else 'buy'
                    await executor.execute_emergency_exit(symbol, side, qty)
                await notifier.send_message("✅ *EMERGENCY EXIT COMPLETE*: All positions closed.")
                global_state.EMERGENCY_EXIT_TRIGGERED = False
                global_state.TRADING_ENABLED = False

            if not global_state.TRADING_ENABLED:
                await asyncio.sleep(60)
                continue

            balance_data = await get_cached_balance(exchange)
            current_balance = float(balance_data.get(BASE_ASSET, {}).get('free', 0.0))
            
            # Update Dashboard State
            positions_data = await get_cached_positions(exchange)
            open_trades = [p for p in positions_data if float(p.get('contracts', 0)) > 0]
            update_bot_state(
                open_trades=open_trades,
                balance=current_balance,
                last_decisions=strategy.last_decisions,
                scan_results=scanner.last_scan_results
            )
            
            # Ninja Sweeper Protocol (Phase 3)
            if await ninja_sweeper_protocol(exchange, current_balance, notifier):
                # Re-fetch balance immediately after sweep (Force refresh)
                await global_state.set_cache('balance', None, 0)
                balance_data = await get_cached_balance(exchange)
                current_balance = float(balance_data.get(BASE_ASSET, {}).get('free', 0.0))
            
            # Phase 74: Global Panic Mode
            if scanner.global_panic:
                logger.critical("!!! EMERGENCY: VOLATILITY SHOCK DETECTED !!!")
                # 1. Cancel ALL open orders
                try:
                    await exchange.cancel_all_orders()
                except Exception as e:
                    logger.warning(f"cancel_all_orders failed (non-critical): {e}")
                positions = await get_cached_positions(exchange)
                active_fut_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]
                
                for pos in active_fut_positions:
                    symbol = pos['symbol']
                    qty = float(pos['contracts'])
                    side = 'sell' if pos['side'] == 'long' else 'buy'
                    await executor.execute_emergency_exit(symbol, side, qty)
                    await notifier.send_message(f"🚨 *GLOBAL PANIC EXIT*\nSymbol: {symbol}\nReason: Volatility Shock")
                
                # 3. Stop trading for 15 minutes
                await notifier.send_message("🛑 *BOT HALTED*: Global Panic Mode active. Cooling down for 15 minutes.")
                scanner.reset_panic() 
                await asyncio.sleep(900) # 15 minutes
                continue

            # Global Circuit Breaker (Phase 25)
            if risk_manager.global_circuit_breaker(current_balance, daily_state['balance']):
                logger.error("Trading halted for 1 hour due to Global Circuit Breaker.")
                await asyncio.sleep(3600)
                continue

            # Daily Telegram Report (Phase 9)
            current_date = datetime.now(timezone.utc).date()
            if current_date > last_report_date:
                if global_state.daily_state.get('trade_count', 0) > 0:
                    yesterday_str = last_report_date.strftime("%Y-%m-%d")
                    all_trades = await trade_logger.get_trades_by_date(yesterday_str)
                    daily_trades = [t for t in all_trades if t.get('timestamp', '').startswith(yesterday_str)]
                    daily_pnl = sum(t.get('pnl', 0) for t in daily_trades)
                    
                    # Phase 26: Regime Efficiency
                    trending_pnl = sum(t.get('pnl', 0) for t in daily_trades if t.get('market_features', {}).get('adx', 0) > 25)
                    ranging_pnl = sum(t.get('pnl', 0) for t in daily_trades if t.get('market_features', {}).get('adx', 0) <= 25)
                    
                    # Phase 50: Edge Drift
                    directional_trades = [t for t in daily_trades if t.get('type') == 'directional']
                    if directional_trades:
                        wins = sum(1 for t in directional_trades if t.get('pnl', 0) > 0)
                        actual_win_rate = wins / (len(directional_trades) + 1e-9)
                        avg_win_prob_predicted = sum(t.get('win_prob', 0.5) for t in directional_trades) / (len(directional_trades) + 1e-9)
                        edge_drift = avg_win_prob_predicted - actual_win_rate
                    else:
                        actual_win_rate = 0.0
                        avg_win_prob_predicted = 0.0
                        edge_drift = 0.0
                    
                    # Phase 90: Shadow Performance Reporting
                    shadow_pnl = await trade_logger.get_shadow_performance(yesterday_str)
                    
                    # Phase 111: Neural Efficiency Score
                    neural_score = strategy.ml_brain.neural_efficiency
                    
                    # v4: include live performance metrics
                    perf = risk_manager.get_performance_summary()
                    msg = (
                        "📊 *DAILY COMMAND CENTER REPORT* 📊\n"
                        f"Date: {yesterday_str}\n"
                        f"Trades Taken: {global_state.daily_state['trade_count']}\n"
                        f"Daily PnL: ${daily_pnl:.2f}\n"
                        f"Shadow PnL: ${shadow_pnl:.2f} (Potential) 👻\n"
                        f"Neural Efficiency: {neural_score:.2f}x 🧠\n"
                        f"Current Balance: ${current_balance:.2f}\n"
                        f"Regime Efficiency:\n"
                        f"- Trending PnL: ${trending_pnl:.2f}\n"
                        f"- Ranging PnL: ${ranging_pnl:.2f}\n"
                        f"Edge Drift: {edge_drift*100:.2f}% (Pred: {avg_win_prob_predicted*100:.1f}%, Act: {actual_win_rate*100:.1f}%)\n"
                        f"📈 Performance (last {perf['total_trades']} trades):\n"
                        f"- Win Rate: {perf['win_rate']*100:.1f}%\n"
                        f"- Expectancy: ${perf['expectancy']:.4f}/trade\n"
                        f"- Sharpe: {perf['sharpe']:.2f}\n"
                        f"- Max DD: ${perf['max_dd']:.4f}\n"
                        "ML Brain Status: Active & Learning 🧠"
                    )
                    await notifier.send_message(msg)
                last_report_date = current_date
            
            daily_state = await global_state.update_daily_state(current_balance)

            # v5-CA: anchor session start balance for 8% equity drawdown brake.
            # Called every loop — update_daily_state returns the day-start balance,
            # so this only changes value when a new trading day begins.
            if daily_state.get('balance', 0) > 0:
                risk_manager.set_session_start_balance(daily_state['balance'])

            daily_pnl_pct = (current_balance - daily_state['balance']) / (daily_state['balance'] + 1e-9) if daily_state['balance'] > 0 else 0.0
            risk_check = risk_manager.check_daily_limits(daily_pnl_pct, current_balance)
            if not risk_check['trade_allowed']:
                await asyncio.sleep(60); continue

            # God's Eye Macro Protocol (Phase 7)
            regime = await macro_filter.check_macro_regime(exchange)
            if regime == 'DANGER':
                logger.warning("MACRO REGIME: DANGER (BTC dumping). Halting scans for 5 minutes.")
                await asyncio.sleep(300)
                continue

            # Phase 111: Nightly Evolution
            now_utc = datetime.now(timezone.utc)
            if now_utc.hour == 0 and now_utc.minute == 0 and now_utc.second < 30:
                logger.info("NIGHTLY EVOLUTION: Optimizing hyperparameters and retraining...")
                evolution_loop = asyncio.get_running_loop()
                await evolution_loop.run_in_executor(global_state.executor, strategy.ml_brain.optimize_hyperparameters)
                await notifier.send_message("🧠 *NIGHTLY EVOLUTION*: ML Brain optimized and retrained.")
                await asyncio.sleep(60) # Prevent multiple triggers
                continue

            if not (TRADE_START_HOUR_UTC <= now_utc.hour < TRADE_END_HOUR_UTC):
                await asyncio.sleep(SCAN_INTERVAL); continue

            positions = await get_cached_positions(exchange)
            active_fut_symbols = {p['symbol'] for p in positions if float(p.get('contracts', 0)) > 0}
            
            # Phase 4: Event-Loop Optimization - Fetch only open trades from last 6 hours
            all_trades = await trade_logger.get_open_trades_only(since_hours=6)
            for trade in list(all_trades):
                if trade.get('exit_price', 0) == 0 and trade['type'] == 'directional':
                    if trade['symbol'] not in active_fut_symbols:
                        try:
                            # Phase 4: Math & Predictive Skew Fix (Ghost Reconciliation)
                            # Fetch trades since entry to calculate actual filled and partial quantities
                            entry_ts = trade['timestamp']
                            if isinstance(entry_ts, str):
                                entry_ts_ms = int(datetime.fromisoformat(entry_ts.replace('Z', '+00:00')).timestamp() * 1000)
                            else:
                                entry_ts_ms = int(entry_ts * 1000)

                            my_trades = await safe_fetch(exchange.fetch_my_trades, trade['symbol'], since=entry_ts_ms)
                            if my_trades:
                                # Filter for trades related to this position
                                # For a LONG, entry is BUY, exit is SELL
                                # For a SHORT, entry is SELL, exit is BUY
                                side = trade['side']
                                exit_side = 'sell' if side == 'buy' else 'buy'
                                
                                exit_trades = [t for t in my_trades if t['side'] == exit_side]
                                if exit_trades:
                                    # Calculate weighted average exit price and total quantity closed
                                    total_exit_qty = sum(t['amount'] for t in exit_trades)
                                    total_exit_cost = sum(t['amount'] * t['price'] for t in exit_trades)
                                    exit_price = total_exit_cost / total_exit_qty if total_exit_qty > 0 else my_trades[-1]['price']
                                    
                                    # The actual quantity to log is what was closed on exchange
                                    # (filled_qty - partial_qty) as per user's fix
                                    actual_qty = total_exit_qty
                                    
                                    fee = sum(t.get('fee', {}).get('cost', 0) for t in my_trades)
                                    pnl = (exit_price - trade['entry_price']) * actual_qty if side == 'buy' else (trade['entry_price'] - exit_price) * actual_qty
                                    pnl_pct = (exit_price - trade['entry_price']) / trade['entry_price'] if side == 'buy' else (trade['entry_price'] - exit_price) / trade['entry_price']
                                    
                                    # Phase 131: Cancel monitor if active
                                    active_monitors = await global_state.get_active_monitors()
                                    if trade['symbol'] in active_monitors:
                                        active_monitors[trade['symbol']].cancel()
                                        await global_state.remove_active_monitor(trade['symbol'])

                                    await trade_logger.close_trade(trade['id'], exit_price, pnl=pnl, pnl_pct=pnl_pct, fee=fee)
                                    risk_manager.update_risk_profile(pnl > 0)
                                    risk_manager.record_pnl(pnl, pnl_pct)   # v4: performance tracking
                                    # Phase 3+4: feed result into all intelligence engines via on_trade_closed
                                    current_bal = await get_cached_balance(exchange)
                                    current_balance_usd = float(current_bal.get('USDT', {}).get('free', INITIAL_CAPITAL))
                                    orchestrator.on_trade_closed(
                                        trade_result={
                                            'symbol':       trade['symbol'],
                                            'pnl_usd':      pnl,
                                            'pnl_pct':      pnl_pct,
                                            'won':          pnl > 0,
                                            'side':         side,
                                            'strategy':     trade.get('strategy', 'directional'),
                                            'edge_score':   trade.get('edge_score', 0),
                                            'win_prob_used': trade.get('win_prob', 0.5),
                                            'rr_ratio':     trade.get('rr_ratio', 1.5),
                                            'hold_seconds': (time.time() - (trade.get('timestamp_epoch', time.time()))),
                                        },
                                        market_conditions={
                                            'regime':   trade.get('regime', 'unknown'),
                                            'adx':      trade.get('adx', 0),
                                            'ofi':      trade.get('ofi', 0),
                                            'session':  trade.get('session', 'unknown'),
                                            'hour_utc': datetime.now(timezone.utc).hour,
                                            'atr_pct':  trade.get('atr_pct', 0),
                                            'signal_direction': side,
                                            'efficiency_score': trade.get('efficiency_score', 0.5),
                                        },
                                        current_balance=current_balance_usd,
                                    )
                                    await notifier.send_message(f"✅ *GHOST RECONCILIATION COMPLETE*\nSymbol: {trade['symbol']}\nExit: {exit_price}\nPnL: ${pnl:.2f}")
                                    
                                    # Phase 49/134: Self-Correction Loop (Simplified)
                                    if pnl < 0:
                                        logger.warning(f"Trade {trade['id']} closed at loss. Monitoring for drift.")
                                        # Retrain if needed, but removed risk multipliers as per Phase 134
                                        # loop = asyncio.get_running_loop()
                                        # await loop.run_in_executor(global_state.executor, strategy.ml_brain.train_model)
                        except Exception as e:
                            logger.error(f"Ghost reconciliation failed for {trade['symbol']}: {e}")

            candidates = await scanner.scan_markets()
            if not candidates: await asyncio.sleep(SCAN_INTERVAL); continue
                
            best = candidates[0]; symbol = best['symbol']
            # Phase 129: Critical Bug Fix - Calculate ATR before use
            atr = strategy.calculate_atr(best['df'])
            
            entry_price = await get_fresh_price(exchange, symbol)

            # HAAE: Hybrid Adaptive Acceleration Engine
            # 1. Calculate Recent Momentum
            trades = await trade_logger.get_trades()
            closed_trades = [t for t in trades if t.get('exit_price', 0) > 0]
            last_two_results = []
            if len(closed_trades) >= 2:
                for t in closed_trades[-2:]:
                    last_two_results.append("WIN" if t.get('pnl', 0) > 0 else "LOSS")

            # 2. Define High Edge Condition
            adx = best.get('adx', 0.0)
            ofi_strength = abs(best.get('ofi', 0.0))
            threshold = 0.2 # Strength threshold
            volume_spike = best.get('vol_ratio', 1.0) > MIN_VOLUME_SPIKE
            HIGH_EDGE = (adx >= 22 and ofi_strength > threshold and volume_spike == True)

            # 3. Dynamic Trade Frequency Control
            # v4.1: trade limit derived from capital_adaptive (smooth sigmoid)
            _ap = get_adaptive_params(current_balance)
            adaptive_trade_limit = math.floor(_ap.trade_limit)
            dynamic_max_trades = min(adaptive_trade_limit, MAX_TRADES_PER_DAY)
            logger.debug(
                f"TradeLimit | tier={_ap.capital_tier} "
                f"adaptive={adaptive_trade_limit} config_cap={MAX_TRADES_PER_DAY} "
                f"effective={dynamic_max_trades}"
            )

            if len(active_fut_symbols) >= 1 or daily_state['trade_count'] >= dynamic_max_trades:
                if daily_state['trade_count'] >= dynamic_max_trades:
                    logger.info(f"Dynamic MAX_TRADES ({dynamic_max_trades}) reached. Sniper mode idling...")
                await asyncio.sleep(SCAN_INTERVAL); continue

            # v4: Log performance summary periodically
            perf = risk_manager.get_performance_summary()
            if perf['total_trades'] > 0 and perf['total_trades'] % 5 == 0:
                logger.info(
                    f"PERFORMANCE: trades={perf['total_trades']} "
                    f"win_rate={perf['win_rate']*100:.1f}% "
                    f"expectancy=${perf['expectancy']:.4f} "
                    f"sharpe={perf['sharpe']:.2f} "
                    f"max_dd=${perf['max_dd']:.4f}"
                )

            # 4. HAAE Risk Calculation
            risk = risk_manager.get_base_risk(current_balance)
            # The streak logic is now handled in RiskManager
            
            # 5. Dynamic Target System (RR) — v4.1: balance-aware RR floor
            rr_ratio = risk_manager.get_dynamic_rr_tp(balance=current_balance)

            best['last_price'] = entry_price
            if 'metrics' not in best:
                best['metrics'] = {'imbalance': best.get('imbalance', 0.0), 'micro_price': best.get('micro_price', entry_price)}
            total_trades = len(await trade_logger.get_trades())
            # v5.0: use generate_signal_full() to get the full SignalResult
            # (includes entry, SL, TP, setup_type, absorption, sweep data)
            signal_result = strategy.generate_signal_full(
                best, total_trades=total_trades, account_balance=current_balance
            )
            signal   = signal_result.signal   if signal_result is not None else 'HOLD'
            win_prob = signal_result.win_prob  if signal_result is not None else 0.0

            if signal == 'BUY':
                logger.info(
                    f"[{MODE} MODE] Sniper BUY signal for {symbol} | "
                    f"setup={signal_result.setup_type} WinProb:{win_prob:.2f} "
                    f"RR:{signal_result.rr_ratio:.2f} conf:{signal_result.confidence:.2f}"
                )

                # Phase 2/3/4 Orchestrator -- all gates in sequence.
                # v5.0: pass signal_result so orchestrator uses pre-calculated
                # entry/SL/TP and enriches candidate for EdgeScorer v5.0
                h_volatility = best.get('h_volatility', 0.0)
                orch_decision = orchestrator.evaluate(
                    candidate        = best,
                    signal           = signal,
                    win_prob         = win_prob,
                    current_balance  = current_balance,
                    trades_today     = daily_state['trade_count'],
                    open_trades      = len(active_fut_symbols),
                    daily_pnl_pct    = daily_pnl_pct,
                    rr_ratio         = signal_result.rr_ratio,
                    risk_manager     = risk_manager,
                    signal_result    = signal_result,
                )
                if not orch_decision.execute:
                    logger.info(f"Orchestrator SKIP BUY [{symbol}]: {orch_decision.skip_reason}")
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue
                # Carry orchestrator outputs forward for execution
                regime            = orch_decision.regime
                regime_confidence = orch_decision.regime_confidence
                regime_desc       = f"{regime} ({regime_confidence:.0%}) via orchestrator"
                regime_mults      = orchestrator.regime_engine.get_multipliers(regime)

                # 1. Auto Leverage Calculation
                leverage = risk_manager.calculate_auto_leverage(current_balance, MIN_NOTIONAL, MAX_LEVERAGE)
                if leverage is None:
                    logger.warning(f"Trade skipped: required leverage too high for capital {current_balance}")
                    await asyncio.sleep(SCAN_INTERVAL); continue
                
                # 2. Set Leverage Dynamically
                try:
                    await exchange.set_leverage(int(leverage), symbol)
                    logger.info(f"Auto Leverage set to {int(leverage)}x for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to set leverage for {symbol}: {e}")
                    await asyncio.sleep(SCAN_INTERVAL); continue

                # v5.0: Use strategy's pre-calculated SL/TP (structure-based, not % floor)
                # Falls back to percentage-based if orchestrator did not receive signal_result
                if orch_decision.stop_loss > 0 and orch_decision.take_profit > 0:
                    stop_loss   = orch_decision.stop_loss
                    take_profit = orch_decision.take_profit
                    entry_price = orch_decision.entry_price if orch_decision.entry_price > 0 else entry_price
                else:
                    # Legacy fallback -- % based
                    stop_loss   = entry_price * (1 - STOP_LOSS_PCT)
                    sl_dist     = entry_price * STOP_LOSS_PCT
                    tp_dist     = sl_dist * rr_ratio
                    take_profit = entry_price + tp_dist
                
                # Position Sizing with regime awareness
                pos_size = await risk_manager.calculate_position_size(
                    exchange, current_balance, symbol, entry_price, atr,
                    risk=risk_manager.current_risk, regime=regime
                )
                
                # Hippocampus ML Feature Extraction
                market_features = {
                    'spread_pct': best.get('spread_pct', 0.0),
                    'atr': atr,
                    'imbalance': best.get('imbalance', 0.0),
                    'micro_price': best.get('micro_price', entry_price),
                    'cvd': best.get('cvd', 0.0),
                    'liquidity_depth': best.get('liquidity_depth', 0.0),
                    'relative_strength': best.get('relative_strength', 0.0),
                    'adx': best.get('adx', 0.0),
                    'ofi': best.get('ofi', 0.0),
                    'btc_dom': best.get('btc_dom', 0.0),
                    'vol_ratio': best.get('vol_ratio', 1.0),
                    'arb_delta': best.get('arb_delta', 0.0),
                    'wall_dist': best.get('wall_dist', 100.0),
                    'velocity': best.get('velocity', 0.0),
                    'funding_rate': best.get('funding_rate', 0.0),
                    'h_volatility': best.get('h_volatility', 0.0),
                    'trend_15m': best.get('trend_15m', 0.0),
                    'vol_ema_ratio': best.get('vol_ema_ratio', 1.0),
                    'price_range_pos': best.get('price_range_pos', 0.5),
                    'sentiment_proxy': best.get('sentiment_proxy', 0.0),
                    'funding_rate_velocity': best.get('funding_rate_velocity', 0.0),
                    'liquidations_proxy': best.get('liquidations_proxy', 0.0),
                    'social_volume_spike': best.get('social_volume_spike', 0.0),
                    'news_sentiment_score': best.get('news_sentiment_score', 0.0),
                    'btc_whale_tx_count': best.get('btc_whale_tx_count', 0.0)
                }
                
                if pos_size > 0:
                    # Phase 73: Execution Latency Logger (Pass signal_ts)
                    signal_ts = best.get('signal_timestamp', time.time())
                    order = await executor.execute_trade(symbol, 'buy', pos_size, signal_ts=signal_ts, stop_loss=stop_loss, take_profit=take_profit)
                    if order:
                        # Phase 4: Fix Maker Order Illusion
                        # Store as pending; log only when filled via order_watcher
                        client_id = order.get('clientOrderId') or order.get('id')
                        await global_state.add_pending_entry(client_id, {
                            'trade_info': {
                                'symbol': symbol, 'side': 'buy', 'entry_price': entry_price, 'qty': pos_size, 
                                'exit_price': 0, 'pnl': 0, 'pnl_pct': 0, 'fee': 0, 'slippage': 0, 
                                'timestamp': datetime.now(timezone.utc).isoformat(), 'type': 'directional',
                                'market_features': market_features,
                                'win_prob': win_prob
                            },
                            'atr': atr,
                            'signal_ts': signal_ts
                        })
                        logger.info(f"LIMIT Order placed: {client_id}. Waiting for fill confirmation...")
                        await notifier.send_message(f"⏳ *LIMIT ORDER PLACED*\nSymbol: {symbol}\nPrice: {entry_price}\nWaiting for fill...")
                        risk_manager.record_trade_time()   # v4: cooldown
                else:
                    # Phase 85: Shadow Engine (Log unexecuted signals)
                    await trade_logger.log_shadow_trade({
                        'symbol': symbol, 'side': 'buy', 'entry_price': entry_price, 'qty': 1.0, # Virtual qty
                        'exit_price': 0, 'pnl': 0, 'pnl_pct': 0,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'market_features': market_features,
                        'win_prob': win_prob,
                        'reason': "Insufficient Capital or Notional Guard",
                        'is_shadow': True # Phase 4: Predictive Skew Fix (Tag for ML exclusion)
                    })
                    logger.info(f"SHADOW TRADE: {symbol} logged. Reason: Insufficient Capital or Notional Guard")

            elif signal == 'SELL':
                logger.info(
                    f"[{MODE} MODE] Sniper SELL signal for {symbol} | "
                    f"setup={signal_result.setup_type} WinProb:{win_prob:.2f} "
                    f"RR:{signal_result.rr_ratio:.2f} conf:{signal_result.confidence:.2f}"
                )

                # Phase 2/3/4 Orchestrator -- v5.0: pass signal_result
                h_volatility = best.get('h_volatility', 0.0)
                orch_decision = orchestrator.evaluate(
                    candidate        = best,
                    signal           = signal,
                    win_prob         = win_prob,
                    current_balance  = current_balance,
                    trades_today     = daily_state['trade_count'],
                    open_trades      = len(active_fut_symbols),
                    daily_pnl_pct    = daily_pnl_pct,
                    rr_ratio         = signal_result.rr_ratio,
                    risk_manager     = risk_manager,
                    signal_result    = signal_result,
                )
                if not orch_decision.execute:
                    logger.info(f"Orchestrator SKIP SELL [{symbol}]: {orch_decision.skip_reason}")
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue
                regime            = orch_decision.regime
                regime_confidence = orch_decision.regime_confidence
                regime_desc       = f"{regime} ({regime_confidence:.0%}) via orchestrator"
                regime_mults      = orchestrator.regime_engine.get_multipliers(regime)

                # 1. Auto Leverage Calculation
                leverage = risk_manager.calculate_auto_leverage(current_balance, MIN_NOTIONAL, MAX_LEVERAGE, MIN_NOTIONAL_BUFFER)
                if leverage is None:
                    logger.warning(f"Trade skipped: required leverage too high for capital {current_balance}")
                    await asyncio.sleep(SCAN_INTERVAL); continue
                
                # 2. Set Leverage Dynamically
                try:
                    await exchange.set_leverage(math.ceil(leverage), symbol)
                    logger.info(f"Auto Leverage set to {math.ceil(leverage)}x for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to set leverage for {symbol}: {e}")
                    await asyncio.sleep(SCAN_INTERVAL); continue

                # v5.0: Use strategy's pre-calculated SL/TP (structure-based)
                if orch_decision.stop_loss > 0 and orch_decision.take_profit > 0:
                    stop_loss   = orch_decision.stop_loss
                    take_profit = orch_decision.take_profit
                    entry_price = orch_decision.entry_price if orch_decision.entry_price > 0 else entry_price
                else:
                    # Legacy fallback -- percentage-based
                    stop_loss   = entry_price * (1 + STOP_LOSS_PCT)
                    sl_dist     = entry_price * STOP_LOSS_PCT
                    tp_dist     = sl_dist * rr_ratio
                    take_profit = entry_price - tp_dist
                
                # Position Sizing with regime awareness
                pos_size = await risk_manager.calculate_position_size(
                    exchange, current_balance, symbol, entry_price, atr,
                    risk=risk_manager.current_risk, regime=regime
                )
                
                # Hippocampus ML Feature Extraction
                market_features = {
                    'spread_pct': best.get('spread_pct', 0.0),
                    'atr': atr,
                    'imbalance': best.get('imbalance', 0.0),
                    'micro_price': best.get('micro_price', entry_price),
                    'cvd': best.get('cvd', 0.0),
                    'liquidity_depth': best.get('liquidity_depth', 0.0),
                    'relative_strength': best.get('relative_strength', 0.0),
                    'adx': best.get('adx', 0.0),
                    'ofi': best.get('ofi', 0.0),
                    'btc_dom': best.get('btc_dom', 0.0),
                    'vol_ratio': best.get('vol_ratio', 1.0),
                    'arb_delta': best.get('arb_delta', 0.0),
                    'wall_dist': best.get('wall_dist', 100.0),
                    'velocity': best.get('velocity', 0.0),
                    'funding_rate': best.get('funding_rate', 0.0),
                    'h_volatility': best.get('h_volatility', 0.0),
                    'trend_15m': best.get('trend_15m', 0.0),
                    'vol_ema_ratio': best.get('vol_ema_ratio', 1.0),
                    'price_range_pos': best.get('price_range_pos', 0.5),
                    'sentiment_proxy': best.get('sentiment_proxy', 0.0),
                    'funding_rate_velocity': best.get('funding_rate_velocity', 0.0),
                    'liquidations_proxy': best.get('liquidations_proxy', 0.0),
                    'social_volume_spike': best.get('social_volume_spike', 0.0),
                    'news_sentiment_score': best.get('news_sentiment_score', 0.0),
                    'btc_whale_tx_count': best.get('btc_whale_tx_count', 0.0)
                }
                
                if pos_size > 0:
                    # Phase 73: Execution Latency Logger (Pass signal_ts)
                    signal_ts = best.get('signal_timestamp', time.time())
                    order = await executor.execute_trade(symbol, 'sell', pos_size, signal_ts=signal_ts, stop_loss=stop_loss, take_profit=take_profit)
                    if order:
                        # Phase 4: Fix Maker Order Illusion
                        # Store as pending; log only when filled via order_watcher
                        client_id = order.get('clientOrderId') or order.get('id')
                        await global_state.add_pending_entry(client_id, {
                            'trade_info': {
                                'symbol': symbol, 'side': 'sell', 'entry_price': entry_price, 'qty': pos_size, 
                                'exit_price': 0, 'pnl': 0, 'pnl_pct': 0, 'fee': 0, 'slippage': 0, 
                                'timestamp': datetime.now(timezone.utc).isoformat(), 'type': 'directional',
                                'market_features': market_features,
                                'win_prob': win_prob
                            },
                            'atr': atr,
                            'signal_ts': signal_ts
                        })
                        logger.info(f"LIMIT Order placed: {client_id}. Waiting for fill confirmation...")
                        await notifier.send_message(f"⏳ *LIMIT ORDER PLACED*\nSymbol: {symbol}\nPrice: {entry_price}\nWaiting for fill...")
                        risk_manager.record_trade_time()   # v4: cooldown
                else:
                    # Phase 85: Shadow Engine (Log unexecuted signals)
                    await trade_logger.log_shadow_trade({
                        'symbol': symbol, 'side': 'sell', 'entry_price': entry_price, 'qty': 1.0, # Virtual qty
                        'exit_price': 0, 'pnl': 0, 'pnl_pct': 0,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'market_features': market_features,
                        'win_prob': win_prob,
                        'reason': "Insufficient Capital or Notional Guard",
                        'is_shadow': True # Phase 4: Predictive Skew Fix (Tag for ML exclusion)
                    })
                    logger.info(f"SHADOW TRADE: {symbol} logged. Reason: Insufficient Capital or Notional Guard")
            
            await asyncio.sleep(SCAN_INTERVAL)
        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt).")
        sys.exit(0)
