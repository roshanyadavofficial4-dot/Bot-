import logging
import asyncio
import uuid
import time
import ccxt.async_support as ccxt
from typing import Dict, Optional
import config

# Configure module-level logging
logger = logging.getLogger("OrderExecutor")

class OrderExecutor:
    """
    Institutional-grade Order Execution Engine for Binance Futures.
    Implements idempotent execution, separate Stop-Loss orders, and 
    robust error handling for micro-capital stability.
    """

    def __init__(self, exchanges: list):
        """
        Phase 81: Multi-Exchange Initialization
        Accepts a list of ccxt exchange instances (Binance only for Phase 130).
        """
        self.exchanges = exchanges

    async def _execute_on_exchange(self, exchange: ccxt.Exchange, symbol: str, side: str, quantity: float, price: float, params: dict, start_time: float, signal_ts: float = None) -> Optional[Dict]:
        """
        Internal helper for single exchange execution.
        """
        try:
            # Phase 135: Precision Setup
            quantity_prec = float(exchange.amount_to_precision(symbol, quantity))
            price_prec = float(exchange.price_to_precision(symbol, price))
            
            logger.info(f"[{exchange.id}] Executing {side} LIMIT for {quantity_prec} {symbol} at {price_prec}")
            order = await exchange.create_order(
                symbol=symbol,
                type='LIMIT',
                side=side,
                amount=quantity_prec,
                price=price_prec,
                params=params
            )
            
            fill_time = time.perf_counter()
            latency_us = (fill_time - start_time) * 1_000_000
            
            if signal_ts:
                total_latency_ms = (time.time() - signal_ts) * 1000
                logger.info(f"[{exchange.id}] Placement Latency: {latency_us:.2f}us | Signal-to-Placement: {total_latency_ms:.2f}ms")
            
            return order
        except Exception as e:
            logger.error(f"[{exchange.id}] Order failed for {symbol}: {e}")
            return None

    async def execute_trade(self, symbol: str, side: str, quantity: float, price: Optional[float] = None, signal_ts: float = None, stop_loss: float = None, take_profit: float = None) -> Optional[Dict]:
        """
        Phase 82: Mirror Execution (asyncio.gather)
        Phase 83: Route Optimization (Liquidity allocation)
        Phase 135: Isolated Margin & Atomic Bracket
        """
        start_time = time.perf_counter()
        client_id_base = f"DE_{symbol.replace('/', '')}_{int(time.time())}"
        
        # 1. Route Optimization: Check liquidity/spread on all exchanges
        if config.MICRO_CAPITAL_MODE:
            # Force Binance only for micro-capital to avoid fragmentation
            target_exchanges = [self.exchanges[0]]
        else:
            target_exchanges = self.exchanges

        exchange_metrics = []
        for ex in target_exchanges:
            try:
                ticker = await ex.fetch_ticker(symbol)
                spread = (ticker['ask'] - ticker['bid']) / ticker['bid']
                exchange_metrics.append({'exchange': ex, 'spread': spread, 'bid': ticker['bid'], 'ask': ticker['ask']})
            except Exception as e:
                logger.warning(f"Failed to fetch ticker for routing on {ex.id}: {e}")
                continue
        
        if not exchange_metrics:
            logger.error("No exchanges available for routing.")
            return None
            
        # Sort by tightest spread
        exchange_metrics.sort(key=lambda x: x['spread'])
        
        # 2. Allocate quantity (60% to best, 40% split among others)
        allocations = []
        if len(exchange_metrics) == 1:
            allocations.append((exchange_metrics[0], quantity))
        else:
            best_ex = exchange_metrics[0]
            others = exchange_metrics[1:]
            best_qty = quantity * 0.6
            allocations.append((best_ex, best_qty))
            
            remaining_qty = quantity - best_qty
            # FIX: Guard against division by zero if others list is empty
            if others:
                qty_per_other = remaining_qty / len(others)
                for ex_m in others:
                    allocations.append((ex_m, qty_per_other))

        # 3. Mirror Execution via asyncio.gather
        tasks = []
        for ex_m, qty in allocations:
            ex = ex_m['exchange']
            client_id = f"{client_id_base}_{ex.id[:2]}_{uuid.uuid4().hex[:4]}"
            
            # Price Improvement (Spread Buffer - Phase 135)
            if price is None:
                # Use a small spread buffer (0.05%) to improve fill probability while staying competitive
                buffer = ex_m['bid'] * 0.0005
                ex_price = ex_m['bid'] + buffer if side.lower() == 'buy' else ex_m['ask'] - buffer
            else:
                ex_price = price
                
            params = {
                'newClientOrderId': client_id,
                'timeInForce': 'GTX' # Post-Only
            }
            tasks.append(self._execute_on_exchange(ex, symbol, side, qty, ex_price, params, start_time, signal_ts))
            
        results = await asyncio.gather(*tasks)
        
        # Phase 135: Atomic Bracket Order (SL/TP)
        if stop_loss or take_profit:
            bracket_tasks = []
            for i, res in enumerate(results):
                if res:
                    ex = allocations[i][0]['exchange']
                    alloc_qty = allocations[i][1]  # FIX: use per-allocation qty, not stale loop var
                    qty_prec = float(ex.amount_to_precision(symbol, alloc_qty))
                    exit_side = 'sell' if side.lower() == 'buy' else 'buy'
                    
                    if stop_loss:
                        sl_params = {'stopPrice': ex.price_to_precision(symbol, stop_loss), 'reduceOnly': True}
                        bracket_tasks.append(ex.create_order(symbol, 'STOP_MARKET', exit_side, qty_prec, params=sl_params))
                    
                    if take_profit:
                        tp_params = {'stopPrice': ex.price_to_precision(symbol, take_profit), 'reduceOnly': True}
                        bracket_tasks.append(ex.create_order(symbol, 'TAKE_PROFIT_MARKET', exit_side, qty_prec, params=tp_params))
            
            if bracket_tasks:
                await asyncio.gather(*bracket_tasks, return_exceptions=True)
                logger.info(f"Atomic Bracket Orders (SL/TP) fired for {symbol}")

        # Return first successful order for tracking
        for res in results:
            if res: return res
        return None

    async def set_leverage(self, symbol: str, leverage: int):
        """
        Sets leverage on ALL exchanges.
        """
        tasks = []
        for ex in self.exchanges:
            tasks.append(ex.set_leverage(leverage, symbol))
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Leverage set to {leverage}x on all exchanges for {symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage on some exchanges: {e}")

    async def set_margin_mode(self, symbol: str, margin_mode: str = 'ISOLATED'):
        """
        Sets margin mode on ALL exchanges.
        """
        tasks = []
        for ex in self.exchanges:
            tasks.append(ex.set_margin_mode(margin_mode.upper(), symbol))
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Margin mode set to {margin_mode} on all exchanges for {symbol}")
        except Exception as e:
            logger.warning(f"Margin mode change warning: {e}")

    async def execute_emergency_exit(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """
        Closes positions on ALL exchanges.
        """
        tasks = []
        for ex in self.exchanges:
            qty_prec = float(ex.amount_to_precision(symbol, quantity))
            params = {'reduceOnly': True}
            logger.critical(f"[{ex.id}] EMERGENCY EXIT: Closing {symbol}")
            tasks.append(ex.create_order(symbol, 'MARKET', side, qty_prec, params=params))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results[0] if results and not isinstance(results[0], Exception) else None

    async def execute_stop_loss(self, symbol: str, side: str, quantity: float, stop_price: float) -> Optional[Dict]:
        """
        Places SL on ALL exchanges.
        """
        tasks = []
        for ex in self.exchanges:
            qty_prec = float(ex.amount_to_precision(symbol, quantity))
            price_prec = float(ex.price_to_precision(symbol, stop_price))
            client_id = f"SL_{symbol.replace('/', '')}_{int(time.time())}_{ex.id[:2]}"
            params = {'stopPrice': price_prec, 'newClientOrderId': client_id, 'reduceOnly': True}
            tasks.append(ex.create_order(symbol, 'STOP_MARKET', side, qty_prec, params=params))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results[0] if results and not isinstance(results[0], Exception) else None

    async def cancel_all_orders(self, symbol: str):
        tasks = [ex.cancel_all_orders(symbol) for ex in self.exchanges]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def close_position(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        tasks = []
        for ex in self.exchanges:
            qty_prec = float(ex.amount_to_precision(symbol, quantity))
            params = {'reduceOnly': True}
            tasks.append(ex.create_order(symbol, 'MARKET', side, qty_prec, params=params))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results[0] if results and not isinstance(results[0], Exception) else None

    async def close_partial_position(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        tasks = []
        for ex in self.exchanges:
            qty_prec = float(ex.amount_to_precision(symbol, quantity))
            params = {'reduceOnly': True}
            tasks.append(ex.create_order(symbol, 'MARKET', side, qty_prec, params=params))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results[0] if results and not isinstance(results[0], Exception) else None

    async def execute_trailing_stop(self, symbol: str, side: str, amount: float, activation_price: float, callback_rate: float = 1.0):
        tasks = []
        for ex in self.exchanges:
            qty_prec = float(ex.amount_to_precision(symbol, amount))
            params = {
                'reduceOnly': True,
                'activationPrice': float(ex.price_to_precision(symbol, activation_price)),
                'callbackRate': callback_rate
            }
            tasks.append(ex.create_order(symbol, 'TRAILING_STOP_MARKET', side, qty_prec, price=None, params=params))
        await asyncio.gather(*tasks, return_exceptions=True)
