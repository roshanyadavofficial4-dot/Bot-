import asyncio
import json
import logging
import aiosqlite
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger("GlobalState")

class BotState:
    def __init__(self):
        self.TRADING_ENABLED = True
        self.EMERGENCY_EXIT_TRIGGERED = False
        self._pending_entries = {} # {client_id: {trade_info, atr}}
        self.active_monitors = {} # {symbol: task}
        self.daily_state = {"date": "", "balance": 0, "trade_count": 0}
        self.db_path = "bot_state.db"
        self.last_ws_heartbeat = {} # {exchange_id: timestamp}
        self.executor = ProcessPoolExecutor(max_workers=2)
        
        # Phase 129: API Caching
        self.cache = {
            'balance': {'data': None, 'timestamp': 0},
            'positions': {'data': None, 'timestamp': 0}
        }
        
        # Phase 130/134: Thread-Safety Locks
        self.state_lock = asyncio.Lock()
        self.pending_lock = asyncio.Lock()
        self.cache_lock = asyncio.Lock()
        self.monitor_lock = asyncio.Lock()

    @property
    def pending_entries(self):
        return self._pending_entries

    async def get_pending_entries(self):
        async with self.pending_lock:
            return dict(self._pending_entries)

    async def add_pending_entry(self, client_id, data):
        """Phase 4: Memory Leak Prevention - Cap pending entries at 100 and remove stale entries."""
        async with self.pending_lock:
            # Add timestamp
            data['timestamp'] = datetime.now(timezone.utc).timestamp()
            
            # Cleanup stale entries (older than 1 hour)
            now = datetime.now(timezone.utc).timestamp()
            stale_keys = [k for k, v in self._pending_entries.items() if now - v.get('timestamp', 0) > 3600]
            for k in stale_keys:
                del self._pending_entries[k]
            
            if len(self._pending_entries) >= 100:
                # Remove oldest entry
                oldest_key = next(iter(self._pending_entries))
                del self._pending_entries[oldest_key]
                logger.warning(f"PENDING ENTRIES CAP REACHED: Removed oldest entry {oldest_key}")
            self._pending_entries[client_id] = data

    async def pop_pending_entry(self, key):
        async with self.pending_lock:
            return self._pending_entries.pop(key, None)

    async def get_cache(self, key):
        async with self.cache_lock:
            return self.cache.get(key)

    async def set_cache(self, key, data, timestamp):
        async with self.cache_lock:
            self.cache[key] = {'data': data, 'timestamp': timestamp}

    async def get_active_monitors(self):
        async with self.monitor_lock:
            return dict(self.active_monitors)

    async def add_active_monitor(self, symbol, task):
        async with self.monitor_lock:
            self.active_monitors[symbol] = task

    async def remove_active_monitor(self, symbol):
        async with self.monitor_lock:
            return self.active_monitors.pop(symbol, None)

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            await db.commit()
            
            # Load initial state
            async with db.execute('SELECT key, value FROM state') as cursor:
                async for row in cursor:
                    if row[0] == 'daily_state':
                        async with self.state_lock:
                            self.daily_state = json.loads(row[1])

    async def save_state(self, key, value):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)', (key, json.dumps(value)))
            await db.commit()

    async def update_daily_state(self, current_balance):
        async with self.state_lock:
            now = datetime.now(timezone.utc)
            today_str = now.strftime("%Y-%m-%d")
            if self.daily_state.get("date") != today_str:
                self.daily_state = {"date": today_str, "balance": current_balance, "trade_count": 0}
                await self.save_state('daily_state', self.daily_state)
            return self.daily_state

    async def increment_trade_count(self):
        async with self.state_lock:
            self.daily_state['trade_count'] += 1
            await self.save_state('daily_state', self.daily_state)
