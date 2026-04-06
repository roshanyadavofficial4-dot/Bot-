import logging
import asyncio
import aiohttp
import time
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ADMIN_ID

# Configure module-level logging
logger = logging.getLogger("TelegramNotifier")

class TelegramNotifier:
    """
    Asynchronous Telegram Push Notification Engine.
    Phase 101-105: Bi-Directional Control.
    """

    def __init__(self, executor=None, trade_logger=None, global_state=None, strategy=None):
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.admin_id = TELEGRAM_ADMIN_ID
        self.executor = executor
        self.trade_logger = trade_logger
        self.global_state = global_state
        self.strategy = strategy  # FIX: Circular import fix — strategy injected via constructor
        self.last_update_id = 0
        self.session = None
        
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials missing in .env. Notifications are disabled.")
        else:
            self.base_url = f"https://api.telegram.org/bot{self.token}"

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def send_message(self, message_text: str):
        """Sends a Telegram message asynchronously."""
        if not self.token or not self.chat_id:
            return
            
        try:
            await self._ensure_session()
            payload = {
                "chat_id": self.chat_id,
                "text": message_text,
                "parse_mode": "Markdown"
            }
            async with self.session.post(f"{self.base_url}/sendMessage", json=payload, timeout=5.0) as response:
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    async def _handle_commands(self, text: str, user_id: str):
        """Phase 102: Command Handler"""
        # Phase 101: Admin restriction
        if self.admin_id and str(user_id) != str(self.admin_id):
            logger.warning(f"Unauthorized command attempt from user {user_id}")
            return

        text = text.lower().strip()
        
        if text == "/status":
            pnl = 0.0
            if self.trade_logger:
                today = time.strftime("%Y-%m-%d")
                all_trades = await self.trade_logger.get_trades()
                pnl = sum(t.get('pnl', 0) for t in all_trades if t.get('timestamp', '').startswith(today))
            
            status = "🟢 ENABLED" if (self.global_state and self.global_state.TRADING_ENABLED) else "🔴 PAUSED"
            
            # Accessing neural efficiency from strategy if possible
            # FIX: Removed circular 'from main import strategy' — use injected self.strategy
            neural_score = 1.0
            if self.strategy and hasattr(self.strategy, 'ml_brain'):
                neural_score = self.strategy.ml_brain.neural_efficiency
            
            msg = (
                "📊 *CLUSTER STATUS*\n"
                f"Mode: {status}\n"
                f"Daily PnL: ${pnl:.2f}\n"
                f"Neural Efficiency: {neural_score:.2f}x 🧠\n"
                "Active Drivers: ML Brain v3.5"
            )
            await self.send_message(msg)
            
        elif text == "/pause":
            if self.global_state:
                self.global_state.TRADING_ENABLED = False
            await self.send_message("🛑 *TRADING PAUSED*: Global halt active.")
            
        elif text == "/resume":
            if self.global_state:
                self.global_state.TRADING_ENABLED = True
            await self.send_message("🚀 *TRADING RESUMED*: Scanning for signals.")
            
        elif text == "/close_all":
            await self.send_message("⚠️ *EMERGENCY EXIT*: Closing all positions across the cluster...")
            if self.global_state:
                self.global_state.EMERGENCY_EXIT_TRIGGERED = True

    def start_polling(self):
        """Phase 105: Background Listener (Async Wrapper)"""
        if not self.token: return
        asyncio.create_task(self.poll())

    async def poll(self):
        """Phase 105: Asynchronous Background Listener"""
        logger.info("Telegram Bi-Directional Control: Active & Polling.")
        while True:
            try:
                await self._ensure_session()
                url = f"{self.base_url}/getUpdates?offset={self.last_update_id + 1}&timeout=30"
                async with self.session.get(url, timeout=35) as response:
                    if response.status == 200:
                        data = await response.json()
                        updates = data.get("result", [])
                        for update in updates:
                            self.last_update_id = update["update_id"]
                            message = update.get("message", {})
                            chat_id = message.get("chat", {}).get("id")
                            user_id = message.get("from", {}).get("id")
                            text = message.get("text")
                            
                            if str(chat_id) == str(self.chat_id) and text:
                                await self._handle_commands(text, user_id)
            except Exception as e:
                logger.error(f"Telegram Polling Error: {e}")
            await asyncio.sleep(1)
