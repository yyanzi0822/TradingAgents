"""Text messages to your phone through a Telegram bot.

Set ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` in ``.env`` and the engine
messages you when it sends an order, when a stop is hit, when something goes
wrong, and with a daily summary. Without them it stays silent.

A failed message never stops trading: it is logged and the cycle carries on.

One-time setup:
    1. In Telegram, talk to @BotFather, send /newbot, and copy the token it gives you
       into .env as TELEGRAM_BOT_TOKEN.
    2. Open your new bot and send it any message.
    3. python -m trading_engine.notify chat-id     # prints your chat id -> TELEGRAM_CHAT_ID
    4. python -m trading_engine.notify test        # you should get a message
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Protocol

logger = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/{method}"
MAX_LEN = 4000  # Telegram's limit is 4096 characters


class Notifier(Protocol):
    def send(self, text: str) -> bool: ...


class NullNotifier:
    """Used when Telegram is not configured, and for dry runs."""

    def send(self, text: str) -> bool:
        return False


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, *, timeout: float = 10.0, post=None):
        self._token = token
        self.chat_id = str(chat_id)
        self.timeout = timeout
        self._post = post or _requests_post

    def send(self, text: str) -> bool:
        if len(text) > MAX_LEN:
            text = text[: MAX_LEN - 20] + "\n… (cut short)"
        try:
            body = self._post(API.format(token=self._token, method="sendMessage"),
                              {"chat_id": self.chat_id, "text": text,
                               "disable_web_page_preview": True}, self.timeout)
            if not body.get("ok"):
                logger.warning("Telegram refused the message: %s", body.get("description", "unknown error"))
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - a notification must never break a trading cycle
            # The request URL contains the bot token, so log only the error type.
            logger.warning("Telegram message not sent (%s)", type(exc).__name__)
            return False


def _requests_post(url: str, payload: dict, timeout: float) -> dict:
    import requests

    return requests.post(url, json=payload, timeout=timeout).json()


def send_blocks(notifier: Notifier, blocks: list[str], *, limit: int = MAX_LEN) -> bool:
    """Send text blocks packed into as few messages as fit Telegram's limit.

    A block is never split across two messages unless it is longer than one
    message by itself. Returns True only if every message was sent.
    """
    messages: list[str] = []
    for block in blocks:
        if messages and len(messages[-1]) + 2 + len(block) <= limit:
            messages[-1] += "\n\n" + block
        else:
            messages.append(block)
    ok = True
    for text in messages:
        ok = notifier.send(text) and ok
    return ok


def from_env() -> Notifier:
    token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return TelegramNotifier(token, chat_id)
    return NullNotifier()


def find_chat_id(token: str, timeout: float = 10.0) -> list[tuple[str, str]]:
    """(chat id, name) for everyone who has messaged the bot recently."""
    import requests

    body = requests.get(API.format(token=token, method="getUpdates"), timeout=timeout).json()
    chats: dict[str, str] = {}
    for update in body.get("result", []):
        chat = (update.get("message") or {}).get("chat") or {}
        if "id" in chat:
            chats[str(chat["id"])] = chat.get("username") or chat.get("first_name") or ""
    return list(chats.items())


def main(argv: list[str]) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    cmd = argv[0] if argv else ""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if cmd not in ("test", "chat-id"):
        print(__doc__)
        return 2
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather and put its token in .env.")
        return 1
    if cmd == "chat-id":
        chats = find_chat_id(token)
        if not chats:
            print("No messages found. Open your bot in Telegram, send it any message, then run this again.")
            return 1
        for chat_id, name in chats:
            print(f"TELEGRAM_CHAT_ID={chat_id}    # {name}")
        return 0
    notifier = from_env()
    if isinstance(notifier, NullNotifier):
        print("TELEGRAM_CHAT_ID is not set. Run: python -m trading_engine.notify chat-id")
        return 1
    ok = notifier.send("Trading engine: test message. Notifications are working.")
    print("sent" if ok else "not sent; check the token and chat id")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
