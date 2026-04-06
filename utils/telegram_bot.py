import os
import logging
import requests

logger = logging.getLogger("SurgeonsPager")

class SurgeonsPager:
    """
    The 'Surgeon's Pager' - Telegram Alert Module.
    Sends real-time execution and emergency alerts to a Telegram chat.
    """
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def format_message(self, alert_type: str, **kwargs) -> str:
        """
        Formats the alert message with appropriate emojis and structure.
        """
        if alert_type == "EXECUTION":
            side = kwargs.get('side', 'BUY').upper()
            emoji = "🟢" if side == "BUY" else "🔴"
            return (
                f"{emoji} *{side} EXECUTION*\n"
                f"Symbol: {kwargs.get('symbol', 'N/A')}\n"
                f"Price: ${kwargs.get('price', 0.0):.4f}\n"
                f"Confidence: {kwargs.get('confidence', 0.0)}%"
            )
        elif alert_type == "SL_TP":
            event = kwargs.get('event', 'SL').upper()
            emoji = "🛑" if event == "SL" else "🎯"
            return (
                f"{emoji} *{event} HIT*\n"
                f"Symbol: {kwargs.get('symbol', 'N/A')}\n"
                f"Price: ${kwargs.get('price', 0.0):.4f}\n"
                f"PnL: {kwargs.get('pnl', 0.0):.2f}%"
            )
        elif alert_type == "EMERGENCY":
            return (
                f"🚨 *EMERGENCY CRASH WARNING* 🚨\n"
                f"Reason: {kwargs.get('message', 'Unknown anomaly')}\n"
                f"Action Taken: {kwargs.get('action', 'Bot Paused')}"
            )
        
        return f"ℹ️ *INFO*\n{kwargs.get('message', '')}"

    def send_alert(self, alert_type: str, **kwargs) -> bool:
        """
        Sends the formatted message to the configured Telegram chat.
        """
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials missing. Alert not sent.")
            return False
        
        message = self.format_message(alert_type, **kwargs)
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Telegram alert sent: {alert_type}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
