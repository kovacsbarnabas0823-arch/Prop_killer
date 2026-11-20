"""
Asynchronous Operational Alert Dispatcher Module
Author: Senior Quantitative Systems Engineer

Dispatches real-time, non-blocking operational alerts to:
  - Telegram Bot (via TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
  - Discord Channel (via DISCORD_WEBHOOK_URL)
  - Local Structured Logging

Supported Critical Event Types:
  - ORDER_FILLED
  - ORDER_REJECTED
  - CONNECTION_LOST
  - HARD_FLATTEN_EOD
  - DAILY_STOP_HIT
  - CONSISTENCY_CAP_HIT
  - STATE_MISMATCH
  - EOD_RECONCILIATION_SAVED
"""

from __future__ import annotations
import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("AlertDispatcher")

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

EVENT_EMOJIS = {
    "ORDER_FILLED": "🎯",
    "ORDER_REJECTED": "⚠️",
    "CONNECTION_LOST": "🚨",
    "HARD_FLATTEN_EOD": "🏁",
    "DAILY_STOP_HIT": "🛑",
    "CONSISTENCY_CAP_HIT": "🛡️",
    "STATE_MISMATCH": "💥",
    "EOD_RECONCILIATION_SAVED": "📊"
}


class AlertDispatcher:
    def __init__(self):
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

        self.has_telegram = bool(self.telegram_token and self.telegram_chat_id)
        self.has_discord = bool(self.discord_webhook)

        if not (self.has_telegram or self.has_discord):
            logger.info("AlertDispatcher: External alert endpoints unconfigured. Operating in local-log mode.")
        else:
            logger.info(f"AlertDispatcher initialized (Telegram: {self.has_telegram}, Discord: {self.has_discord})")

    def _send_telegram_sync(self, text: str) -> None:
        if not self.has_telegram:
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                pass
        except Exception as e:
            logger.warning(f"Telegram alert delivery failed: {e}")

    def _send_discord_sync(self, title: str, text: str, color: int = 0x3498DB) -> None:
        if not self.has_discord:
            return
        payload = {
            "embeds": [{
                "title": title,
                "description": text,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        try:
            req = urllib.request.Request(
                self.discord_webhook,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (PropKiller/9.0)"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                pass
        except Exception as e:
            logger.warning(f"Discord alert delivery failed: {e}")

    def dispatch(
        self,
        event_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Dispatches alert synchronously in non-blocking thread or executor.
        """
        emoji = EVENT_EMOJIS.get(event_type, "📢")
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S EST")
        header = f"{emoji} **[{event_type}]** - {ts_str}"
        body = f"{header}\n{message}"

        if details:
            body += "\n```json\n" + json.dumps(details, indent=2) + "\n```"

        logger.info(f"ALERT [{event_type}]: {message}")

        # Non-blocking delivery via background thread
        def _bg_send():
            if self.has_telegram:
                self._send_telegram_sync(body)
            if self.has_discord:
                color = 0xE74C3C if "STOP" in event_type or "MISMATCH" in event_type or "REJECTED" in event_type else 0x2ECC71
                self._send_discord_sync(f"{emoji} {event_type}", message, color=color)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.run_in_executor(None, _bg_send)
            else:
                _bg_send()
        except RuntimeError:
            import threading
            threading.Thread(target=_bg_send, daemon=True).start()


# Global Singleton Dispatcher
alert_dispatcher = AlertDispatcher()
