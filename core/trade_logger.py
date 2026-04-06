import aiosqlite
import json
import os
import logging
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger("TradeLogger")

class TradeLogger:
    """
    ML-Ready Trade Logger using aiosqlite.
    Stores full trade history and market features for Reinforcement Learning.
    """
    def __init__(self, db_path="trades.db"):
        self.db_path = db_path

    async def init_db(self):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        side TEXT,
                        entry_price REAL,
                        qty REAL,
                        exit_price REAL,
                        pnl REAL,
                        pnl_pct REAL,
                        fee REAL,
                        slippage REAL,
                        timestamp TEXT,
                        exit_timestamp TEXT,
                        type TEXT,
                        market_features TEXT,
                        exchange_id TEXT,
                        wallet_id TEXT,
                        is_shadow INTEGER DEFAULT 0
                    )
                ''')
                # Phase 85: Shadow Logging
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS shadow_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        side TEXT,
                        entry_price REAL,
                        qty REAL,
                        exit_price REAL,
                        pnl REAL,
                        pnl_pct REAL,
                        timestamp TEXT,
                        exit_timestamp TEXT,
                        market_features TEXT,
                        reason TEXT
                    )
                ''')
                # Phase 27: Partial Exit Accounting
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS partial_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id INTEGER,
                        timestamp TEXT,
                        exit_price REAL,
                        partial_qty REAL,
                        realized_pnl REAL
                    )
                ''')
                await db.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    async def get_open_trades_only(self, since_hours=6):
        """
        Surgical Fix 1: Optimized query to load only open trades from the last X hours.
        Prevents event-loop blocking by avoiding full table loads.
        """
        try:
            since_ts = datetime.now(timezone.utc).timestamp() - (since_hours * 3600)
            since_iso = datetime.fromtimestamp(since_ts, timezone.utc).isoformat()
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute('''
                    SELECT * FROM trades 
                    WHERE exit_price = 0 
                    AND timestamp > ?
                ''', (since_iso,)) as cursor:
                    rows = await cursor.fetchall()
            
            trades_list = []
            for row in rows:
                trade_dict = dict(row)
                try:
                    trade_dict['market_features'] = json.loads(trade_dict['market_features'])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"market_features parse failed for a trade record: {e}")
                    trade_dict['market_features'] = {}
                trades_list.append(trade_dict)
            return trades_list
        except Exception as e:
            logger.error(f"Error fetching open trades: {e}")
            return []

    async def cleanup_old_trades(self, days=7):
        """
        Surgical Fix 2: Memory/Resource Leak Prevention.
        Deletes old trades and vacuums the database.
        """
        try:
            cutoff_ts = datetime.now(timezone.utc).timestamp() - (days * 86400)
            cutoff_iso = datetime.fromtimestamp(cutoff_ts, timezone.utc).isoformat()
            
            async with aiosqlite.connect(self.db_path) as db:
                # Delete old closed trades
                await db.execute('DELETE FROM trades WHERE exit_price > 0 AND exit_timestamp < ?', (cutoff_iso,))
                await db.execute('DELETE FROM shadow_trades WHERE timestamp < ?', (cutoff_iso,))
                await db.execute('DELETE FROM partial_logs WHERE timestamp < ?', (cutoff_iso,))
                await db.execute('VACUUM')
                await db.commit()
                logger.info(f"Database cleanup complete. Trades older than {days} days removed.")
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")

    async def log_trade(self, trade_data):
        if trade_data.get('is_shadow', False):
            # Log it to file for your testing, but DO NOT send to ML Brain
            logger.info("Shadow Trade logged. Skipping ML Training update.")
            return trade_data

        try:
            async with aiosqlite.connect(self.db_path) as db:
                market_features = json.dumps(trade_data.get('market_features', {}))
                
                cursor = await db.execute('''
                    INSERT INTO trades (
                        symbol, side, entry_price, qty, exit_price, pnl, pnl_pct, fee, slippage, timestamp, type, market_features, exchange_id, wallet_id, is_shadow
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data['symbol'],
                    trade_data['side'],
                    trade_data['entry_price'],
                    trade_data['qty'],
                    trade_data.get('exit_price', 0),
                    trade_data.get('pnl', 0),
                    trade_data.get('pnl_pct', 0),
                    trade_data.get('fee', 0),
                    trade_data.get('slippage', 0),
                    trade_data['timestamp'],
                    trade_data['type'],
                    market_features,
                    trade_data.get('exchange_id', 'binance'),
                    trade_data.get('wallet_id', 'main'),
                    trade_data.get('is_shadow', 0)
                ))
                trade_id = cursor.lastrowid
                await db.commit()
                trade_data['id'] = trade_id
                return trade_data
        except Exception as e:
            logger.error(f"Error logging trade to SQLite: {e}")
            return trade_data

    async def log_shadow_trade(self, trade_data):
        """
        Phase 85: Shadow Logging
        Logs trades that were NOT executed due to risk/capital limits.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                features = trade_data.get('market_features', {})
                features['is_shadow'] = trade_data.get('is_shadow', True)
                market_features = json.dumps(features)
                await db.execute('''
                    INSERT INTO shadow_trades (
                        symbol, side, entry_price, qty, exit_price, pnl, pnl_pct, timestamp, market_features, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data['symbol'],
                    trade_data['side'],
                    trade_data['entry_price'],
                    trade_data['qty'],
                    trade_data.get('exit_price', 0),
                    trade_data.get('pnl', 0),
                    trade_data.get('pnl_pct', 0),
                    trade_data['timestamp'],
                    market_features,
                    trade_data.get('reason', 'Unknown')
                ))
                await db.commit()
                logger.info(f"Shadow Trade Logged: {trade_data['symbol']} | Reason: {trade_data['reason']}")
        except Exception as e:
            logger.error(f"Error logging shadow trade: {e}")

    async def mark_ptp_hit(self, trade_id: int):
        """
        Phase 19: Marks a trade as having hit its Partial Take-Profit (PTP) target.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute('SELECT market_features FROM trades WHERE id = ?', (trade_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        features = json.loads(row[0]) if row[0] else {}
                        features['ptp_hit'] = True
                        await db.execute('UPDATE trades SET market_features = ? WHERE id = ?', (json.dumps(features), trade_id))
                        await db.commit()
        except Exception as e:
            logger.error(f"Error marking PTP hit in SQLite: {e}")

    async def close_trade(self, trade_id_or_symbol, exit_price, pnl=0, pnl_pct=0, partial_qty=None, fee=0, slippage=0):
        """
        Phase 27: Partial Exit Accounting
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if isinstance(trade_id_or_symbol, int):
                    cursor = await db.execute('''
                        SELECT id, side, entry_price, qty, fee, slippage FROM trades 
                        WHERE id = ? AND exit_price = 0 
                    ''', (trade_id_or_symbol,))
                else:
                    cursor = await db.execute('''
                        SELECT id, side, entry_price, qty, fee, slippage FROM trades 
                        WHERE symbol = ? AND exit_price = 0 
                        ORDER BY id DESC LIMIT 1
                    ''', (trade_id_or_symbol,))
                row = await cursor.fetchone()
                
                if row:
                    trade_id, side, entry_price, qty, current_fee, current_slippage = row
                    exit_timestamp = datetime.now(timezone.utc).isoformat()
                    
                    if partial_qty is not None and partial_qty < qty:
                        # Partial Exit
                        if side == 'buy':
                            realized_pnl = (exit_price - entry_price) * partial_qty - fee
                        else:
                            realized_pnl = (entry_price - exit_price) * partial_qty - fee
                        
                        new_qty = qty - partial_qty
                        
                        await db.execute('''
                            UPDATE trades SET 
                                qty = ?,
                                fee = ?,
                                slippage = ?
                            WHERE id = ?
                        ''', (new_qty, current_fee + fee, current_slippage + slippage, trade_id))
                        
                        await db.execute('''
                            INSERT INTO partial_logs (trade_id, timestamp, exit_price, partial_qty, realized_pnl)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (trade_id, exit_timestamp, exit_price, partial_qty, realized_pnl))
                        
                        await db.commit()
                        logger.info(f"Partial Trade Closed: {trade_id_or_symbol} | Qty: {partial_qty} | PnL: {realized_pnl:.4f}")
                        return await self.calculate_metrics()
                    else:
                        # Full Exit
                        if side == 'buy':
                            calc_pnl = (exit_price - entry_price) * qty - fee
                        else: # sell or funding_open (short futures)
                            calc_pnl = (entry_price - exit_price) * qty - fee
                        
                        # Use provided pnl if non-zero, else calculated
                        final_pnl = pnl if pnl != 0 else calc_pnl
                        
                        entry_notional = entry_price * qty
                        final_pnl_pct = pnl_pct if pnl_pct != 0 else (final_pnl / entry_notional if entry_notional > 0 else 0)
                        
                        await db.execute('''
                            UPDATE trades SET 
                                exit_price = ?, 
                                exit_timestamp = ?, 
                                pnl = ?, 
                                pnl_pct = ?, 
                                fee = ?, 
                                slippage = ?
                            WHERE id = ?
                        ''', (exit_price, exit_timestamp, final_pnl, final_pnl_pct, current_fee + fee, current_slippage + slippage, trade_id))
                        await db.commit()
                        
                        metrics = await self.calculate_metrics()
                        logger.info(f"Trade Closed: {trade_id_or_symbol} | PnL: {final_pnl:.4f} | Metrics: {metrics}")
                        return metrics
        except Exception as e:
            logger.error(f"Error closing trade in SQLite: {e}")
        return {}

    async def get_blacklisted_symbols(self) -> list:
        """
        Phase 11: The Omni-Shield (Anti-Revenge Blacklist)
        Returns a list of symbols that have had 2 consecutive losses in the last 24 hours.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute('''
                    SELECT symbol, pnl FROM trades 
                    WHERE exit_price > 0 
                    AND type = 'directional' 
                    AND exit_timestamp >= datetime('now', '-1 day', 'utc')
                    ORDER BY exit_timestamp ASC
                ''') as cursor:
                    rows = await cursor.fetchall()
            
            symbol_history = {}
            for symbol, pnl in rows:
                if symbol not in symbol_history:
                    symbol_history[symbol] = []
                symbol_history[symbol].append(pnl)
                
            blacklisted = []
            for symbol, pnls in symbol_history.items():
                if len(pnls) >= 2:
                    # Check if the last two trades were losses
                    if pnls[-1] < 0 and pnls[-2] < 0:
                        blacklisted.append(symbol)
                        
            return blacklisted
        except Exception as e:
            logger.error(f"Error fetching blacklisted symbols: {e}")
            return []

    async def calculate_metrics(self):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute('SELECT pnl, pnl_pct FROM trades WHERE exit_price > 0') as cursor:
                    rows = await cursor.fetchall()
            
            if not rows: return {}
            
            pnls = [r[0] for r in rows]
            pnl_pcts = [r[1] for r in rows]
            
            wins = [p for p in pnls if p > 0]
            win_rate = len(wins) / len(pnls) if pnls else 0
            
            cumulative_pnl = np.cumsum(pnls)
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            last_50_pcts = pnl_pcts[-50:]
            if len(last_50_pcts) > 1:
                avg_return = np.mean(last_50_pcts)
                std_return = np.std(last_50_pcts)
                sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            else: sharpe_ratio = 0
            
            return {
                'total_trades': len(rows),
                'win_rate': round(win_rate, 4),
                'max_drawdown': round(float(max_drawdown), 4),
                'sharpe_ratio': round(float(sharpe_ratio), 4)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics from SQLite: {e}")
            return {}

    async def get_trades(self):
        """
        Returns all trades as a list of dictionaries.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute('SELECT * FROM trades') as cursor:
                    rows = await cursor.fetchall()
            
            trades_list = []
            for row in rows:
                trade_dict = dict(row)
                try:
                    trade_dict['market_features'] = json.loads(trade_dict['market_features'])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"market_features parse failed for a trade record: {e}")
                    trade_dict['market_features'] = {}
                trades_list.append(trade_dict)
            return trades_list
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []

    async def get_shadow_performance(self, date_str: str) -> float:
        """
        Phase 85: Returns cumulative PnL of shadow trades for a specific date.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute('''
                    SELECT SUM(pnl) FROM shadow_trades 
                    WHERE timestamp LIKE ?
                ''', (f"{date_str}%",)) as cursor:
                    row = await cursor.fetchone()
                    return float(row[0]) if row and row[0] else 0.0
        except Exception as e:
            logger.error(f"Error fetching shadow performance: {e}")
            return 0.0

    async def get_trades_by_date(self, date_str):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('''
                SELECT * FROM trades 
                WHERE timestamp LIKE ?
            ''', (f"{date_str}%",)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
