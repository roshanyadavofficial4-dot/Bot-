import logging
import asyncio

logger = logging.getLogger("SystemWatchdog")

class SystemWatchdog:
    """
    Phase 8: The Immortality Protocol.
    Ensures the bot can run unattended by reconciling ghost positions
    and monitoring critical background tasks.
    """
    
    async def reconcile_ghost_positions(self, exchange, trade_logger, notifier):
        """
        Finds and closes positions that are open on the exchange but missing in the local DB.
        """
        try:
            logger.info("Running Ghost Position Reconciliation...")
            positions = await exchange.fetch_positions()
            active_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]
            
            # Phase 8: The Immortality Protocol.
            # Optimized to fetch only open trades to prevent event-loop blocking.
            all_trades = await trade_logger.get_open_trades_only(since_hours=6)
            local_open_symbols = {
                t['symbol'] for t in all_trades 
                if t.get('exit_price', 0) == 0
            }

            for pos in active_positions:
                symbol = pos['symbol']
                if symbol not in local_open_symbols:
                    logger.warning(f"GHOST POSITION DETECTED: {symbol}. Executing emergency close.")
                    
                    side = 'sell'
                    if pos.get('side') == 'short':
                        side = 'buy'
                    elif 'info' in pos and 'positionAmt' in pos['info']:
                        if float(pos['info']['positionAmt']) < 0:
                            side = 'buy'
                            
                    amount = float(pos.get('contracts', 0))
                    
                    try:
                        await exchange.create_order(
                            symbol=symbol,
                            type='MARKET',
                            side=side,
                            amount=amount,
                            params={'reduceOnly': True}
                        )
                        msg = f"🚨 GHOST POSITION KILLED: {symbol}"
                        await notifier.send_message(msg)
                        logger.info(msg)
                    except Exception as e:
                        logger.error(f"Failed to kill ghost position {symbol}: {e}")
                        
        except Exception as e:
            logger.error(f"Error during ghost position reconciliation: {e}")

    async def monitor_tasks(self, tasks_dict: dict, exchange):
        """
        Monitors background tasks and restarts them if they crash.
        tasks_dict format: {'task_name': {'task': asyncio.Task, 'func': coroutine_function, 'args': tuple}}
        """
        while True:
            try:
                for name, info in tasks_dict.items():
                    task = info['task']
                    if task.done():
                        logger.critical(f"CRITICAL: Task '{name}' has crashed or stopped! Restarting...")
                        try:
                            exc = task.exception()
                            if exc:
                                logger.error(f"Task '{name}' exception: {exc}")
                        except Exception:
                            pass
                            
                        func = info['func']
                        args = info['args']
                        new_task = asyncio.create_task(func(*args))
                        tasks_dict[name]['task'] = new_task
                        
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in task monitor: {e}")
                await asyncio.sleep(60)
