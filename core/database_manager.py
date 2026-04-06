import sqlite3
import logging
import time
from datetime import datetime

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TradeTracker")

def db_retry(func):
    """
    Decorator to retry SQLite operations if the database is locked.
    (Killer #3: Concurrency Hardening)
    """
    def wrapper(*args, **kwargs):
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    attempts += 1
                    wait_time = 0.5 * attempts
                    logger.warning(f"DB LOCKED: Retrying {func.__name__} in {wait_time}s (Attempt {attempts}/{max_attempts})...")
                    time.sleep(wait_time)
                else:
                    raise e
        return func(*args, **kwargs) # Final attempt
    return wrapper

class TradeTracker:
    """
    Lightweight Local Database Manager (Memory Bank).
    Tracks active trades, logs closed trades, and calculates exact Daily PnL 
    including exchange fees. Uses SQLite for robust, concurrent-safe storage.
    Optimized with an in-memory cache to eliminate high-frequency I/O latency.
    """

    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        # In-memory cache for ultra-fast access during high-frequency loops (Error 27 Fix)
        self._active_trades_cache = []
        self._init_db()
        self._refresh_cache()

    def _init_db(self):
        """Initializes the SQLite database and creates tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Table for currently open positions
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS active_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        sl REAL NOT NULL,
                        tp REAL NOT NULL,
                        oco_order_id TEXT,
                        market_features TEXT,
                        open_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Table for historical closed trades
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS closed_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        net_pnl REAL NOT NULL,
                        fee REAL NOT NULL,
                        reason TEXT NOT NULL,
                        open_time TIMESTAMP,
                        close_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            logger.critical(f"Database initialization failed: {e}")

    def _refresh_cache(self):
        """Syncs the in-memory cache with the current state of the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, symbol, entry_price, quantity, sl, tp, oco_order_id, market_features, open_time FROM active_trades')
                columns = [col[0] for col in cursor.description]
                self._active_trades_cache = [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Failed to refresh active trades cache: {e}")

    @db_retry
    def open_trade(self, symbol: str, entry_price: float, quantity: float, sl: float, tp: float, oco_order_id: str = None, market_features: str = None) -> int:
        """
        Logs a newly opened trade into the active_trades table and updates cache.
        Returns the unique trade_id.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO active_trades (symbol, entry_price, quantity, sl, tp, oco_order_id, market_features)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, entry_price, quantity, sl, tp, oco_order_id, market_features))
                trade_id = cursor.lastrowid
                conn.commit()
                
                # Update cache immediately (Error 25 Fix)
                self._refresh_cache()
                
                logger.info(f"DB LOG: Opened {symbol} (ID: {trade_id}) | Entry: ${entry_price:.4f} | Qty: {quantity:.6f}")
                return trade_id
        except sqlite3.Error as e:
            logger.error(f"Failed to log open trade for {symbol}: {e}")
            return -1

    @db_retry
    def close_trade(self, trade_id: int, exit_price: float, reason: str):
        """
        Closes an active trade by its unique trade_id (Error 24 Fix).
        Calculates exact Net PnL, moves to closed_trades, and removes from active_trades.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Fetch the specific trade by ID
                cursor.execute('''
                    SELECT symbol, entry_price, quantity, open_time, market_features 
                    FROM active_trades 
                    WHERE id = ?
                ''', (trade_id,))
                trade = cursor.fetchone()
                
                if not trade:
                    logger.warning(f"DB LOG: No active trade found with ID {trade_id} to close.")
                    return None
                
                symbol, entry_price, quantity, open_time, market_features = trade
                
                # --- Exact Fee & PnL Calculation ---
                entry_value = entry_price * quantity
                exit_value = exit_price * quantity
                total_fee = (entry_value * 0.001) + (exit_value * 0.001)
                gross_pnl = exit_value - entry_value
                net_pnl = gross_pnl - total_fee
                
                # Insert into closed_trades
                cursor.execute('''
                    INSERT INTO closed_trades 
                    (symbol, entry_price, exit_price, quantity, net_pnl, fee, reason, open_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, entry_price, exit_price, quantity, net_pnl, total_fee, reason, open_time))
                
                # Delete from active_trades
                cursor.execute('DELETE FROM active_trades WHERE id = ?', (trade_id,))
                conn.commit()
                
                # Update cache immediately (Error 25 Fix)
                self._refresh_cache()
                
                logger.info(
                    f"DB LOG: Closed {symbol} (ID: {trade_id}, Reason: {reason}) | Exit: ${exit_price:.4f} | "
                    f"Net PnL: ${net_pnl:.4f}"
                )
                return {
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'net_pnl': net_pnl,
                    'open_time': open_time,
                    'market_features': market_features
                }
        except sqlite3.Error as e:
            logger.error(f"Failed to log close trade for ID {trade_id}: {e}")
            return None

    def get_daily_pnl_amount(self) -> float:
        """
        Calculates the exact Daily PnL absolute amount based on today's closed trades.
        (Fix 4: Explicit amount return)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT SUM(net_pnl) 
                    FROM closed_trades 
                    WHERE date(close_time) = date('now')
                ''')
                result = cursor.fetchone()[0]
                return result if result is not None else 0.0
        except sqlite3.Error as e:
            logger.error(f"Failed to calculate daily PnL amount: {e}")
            return 0.0

    def purge_old_trades(self, days: int = 7):
        """
        Prunes historical closed trades older than N days to prevent database bloating.
        (Longevity Fix)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM closed_trades WHERE close_time < datetime('now', ?)", (f'-{days} days',))
                deleted_count = cursor.rowcount
                conn.commit()
                if deleted_count > 0:
                    logger.info(f"DB LOG: Purged {deleted_count} old trades from history.")
        except sqlite3.Error as e:
            logger.error(f"Failed to purge old trades: {e}")

    async def reconcile_trades(self, exchange):
        """
        Orphan/Stale Trade Reconciliation (Killer #2: Atomic State Sync).
        Syncs Binance open orders and trades with local DB.
        """
        logger.info("Starting Atomic Trade Reconciliation...")
        try:
            # 1. Fetch all open orders from Binance
            open_orders = await exchange.fetch_open_orders()
            active_trades = self.get_active_trades()
            active_symbols = {t['symbol'] for t in active_trades}
            
            # 2. Identify 'Ghost' orders (orders on exchange but not in DB)
            # We 'Adopt' them if they are market/limit entries, or cancel them if they are unknown
            for order in open_orders:
                symbol = order['symbol']
                if symbol not in active_symbols:
                    # If it's an OCO part or a stray limit, we might want to adopt or cancel
                    # For this bot, we cancel stray orders to maintain a clean state
                    logger.warning(f"RECONCILE: Found ghost order {order['id']} for {symbol}. Cancelling for safety.")
                    await exchange.cancel_order(order['id'], symbol)
            
            # 3. Identify 'Orphan' trades (trades in DB but no orders on exchange)
            # This usually means the OCO was filled while the bot was offline.
            for trade in active_trades:
                symbol = trade['symbol']
                oco_id = trade.get('oco_order_id')
                
                # Check if any part of the OCO is still open
                oco_still_open = any(
                    str(o.get('info', {}).get('orderListId')) == str(oco_id) 
                    for o in open_orders if o.get('info', {}).get('orderListId')
                )
                
                if not oco_still_open and oco_id:
                    logger.warning(f"RECONCILE: Found orphan trade in DB for {symbol} (OCO {oco_id} filled/missing). Closing as RECONCILE_STALE.")
                    # Fetch last price to close accurately
                    ticker = await exchange.fetch_ticker(symbol)
                    self.close_trade(trade['id'], ticker['last'], reason="RECONCILE_STALE")
                    
            logger.info("Atomic Reconciliation complete.")
        except Exception as e:
            logger.error(f"Atomic Reconciliation failed: {e}")

    def get_active_trades(self) -> list:
        """
        Fetches all currently open trades from the in-memory cache (Error 27 Fix).
        No disk I/O required for high-frequency reads.
        """
        return self._active_trades_cache

    @db_retry
    def update_sl(self, trade_id: int, new_sl: float):
        """Updates the Stop-Loss for an active trade and updates cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE active_trades SET sl = ? WHERE id = ?', (new_sl, trade_id))
                conn.commit()
                
                # Update cache immediately (Error 25 Fix)
                self._refresh_cache()
        except sqlite3.Error as e:
            logger.error(f"Failed to update SL for trade {trade_id}: {e}")
