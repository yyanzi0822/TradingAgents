"""Remote control from Telegram: stop, resume and check the engine from your phone.

Send these to your bot (only TELEGRAM_CHAT_ID is obeyed; anyone else is ignored):

    /stop [reason]   or  停止   turn the kill switch on: every new order is rejected
    /resume          or  恢复   turn it off again
    /status          or  状态   account, positions, stops and the kill switch

/stop does not cancel orders already filled at the broker, and it leaves the
resting stop orders in place: they protect the positions you hold.

Run it as a long-lived process (the setup keeps it running with launchd):

    python -m trading_engine.commands
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from trading_engine.killswitch import KillSwitch
from trading_engine.notify import API, MAX_LEN

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("TRADING_ENGINE_DIR", Path.home() / ".trading_engine"))

ALIASES = {
    "stop": "stop", "停止": "stop", "停": "stop",
    "resume": "resume", "恢复": "resume", "start": "help",
    "status": "status", "状态": "status",
    "help": "help", "帮助": "help",
}

HELP = ("可用命令：\n"
        "/stop [原因] 或 停止 — 打开紧急开关，拒绝所有新订单\n"
        "/resume 或 恢复 — 关闭紧急开关，恢复交易\n"
        "/status 或 状态 — 账户、持仓、止损和开关状态")


def parse_command(text: str) -> tuple[str | None, str]:
    """('stop', 'reason') from '/stop reason', '/stop@my_bot reason' or '停止 reason'."""
    parts = (text or "").strip().split(maxsplit=1)
    if not parts:
        return None, ""
    word = parts[0].lstrip("/").split("@", 1)[0].lower()
    return ALIASES.get(word), (parts[1].strip() if len(parts) > 1 else "")


class CommandBot:
    def __init__(self, token: str, chat_id: str, kill_switch: KillSwitch,
                 status: Callable[[], str], *, offset_path: Path | None = None,
                 get: Callable[[str, dict, float], dict] | None = None,
                 send: Callable[[str], bool] | None = None):
        self._token = token
        self.chat_id = str(chat_id)
        self.kill_switch = kill_switch
        self.status = status
        self.offset_path = offset_path
        self._get = get or _requests_get
        self._send = send or self._telegram_send

    # --- commands ---------------------------------------------------------

    def handle(self, text: str) -> str:
        cmd, arg = parse_command(text)
        ks = self.kill_switch
        if cmd == "stop":
            already = ks.active()
            ks.engage(f"Telegram /stop: {arg}" if arg else "Telegram /stop")
            return ("🛑 紧急开关已经是打开的（已更新原因）。" if already else "🛑 紧急开关已打开。") + \
                   "\n所有新订单都会被拒绝，已有持仓的止损单保持不变。\n发送 /resume 恢复交易。"
        if cmd == "resume":
            if not ks.active():
                return "✅ 紧急开关本来就是关闭的，交易正常。"
            ks.release()
            return "✅ 紧急开关已关闭，下一次运行起恢复交易。"
        if cmd == "status":
            try:
                return self.status()
            except Exception as exc:  # noqa: BLE001 - report, never crash the listener
                logger.warning("status failed: %s", type(exc).__name__)
                state = f"打开 — {ks.reason()}" if ks.active() else "关闭"
                return f"⚠️ 读取账户失败（{type(exc).__name__}）。\n紧急开关：{state}"
        return HELP

    # --- Telegram -----------------------------------------------------------

    def poll_once(self, timeout: float = 25) -> int:
        """Fetch new messages, act on the owner's, and return how many were handled."""
        offset = self._load_offset()
        params = {"timeout": int(timeout), "allowed_updates": '["message"]'}
        if offset is not None:
            params["offset"] = offset
        body = self._get(API.format(token=self._token, method="getUpdates"), params, timeout + 10)
        if not body.get("ok"):
            raise RuntimeError(f"Telegram refused getUpdates: {body.get('description', 'unknown error')}")
        handled = 0
        for update in body.get("result", []):
            self._save_offset(int(update["update_id"]) + 1)
            message = update.get("message") or {}
            chat = str((message.get("chat") or {}).get("id", ""))
            if chat != self.chat_id:
                logger.warning("ignored a message from chat %s (not TELEGRAM_CHAT_ID)", chat or "?")
                continue
            reply = self.handle(message.get("text", ""))
            self._send(reply[:MAX_LEN])
            handled += 1
        return handled

    def run_forever(self) -> None:
        delay = 1.0
        while True:
            try:
                self.poll_once()
                delay = 1.0
            except Exception as exc:  # noqa: BLE001 - network blips must not end the listener
                # The request URL contains the bot token, so log only the error type.
                logger.warning("poll failed (%s); retrying in %.0fs", type(exc).__name__, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)

    def _telegram_send(self, text: str) -> bool:
        from trading_engine.notify import TelegramNotifier
        return TelegramNotifier(self._token, self.chat_id).send(text)

    def _load_offset(self) -> int | None:
        if self.offset_path and self.offset_path.exists():
            try:
                return int(self.offset_path.read_text().strip())
            except ValueError:
                return None
        return None

    def _save_offset(self, offset: int) -> None:
        if self.offset_path:
            self.offset_path.parent.mkdir(parents=True, exist_ok=True)
            self.offset_path.write_text(str(offset))


def _requests_get(url: str, params: dict, timeout: float) -> dict:
    import requests

    return requests.get(url, params=params, timeout=timeout).json()


def _live_status(kill_switch: KillSwitch) -> Callable[[], str]:
    def status() -> str:
        from trading_engine.broker import AlpacaPaperBroker
        from trading_engine.ledger import Ledger
        from trading_engine.summary import build_summary

        return build_summary(AlpacaPaperBroker(), Ledger(DATA_DIR / "ledger.jsonl"), kill_switch)

    return status


def main(argv: list[str]) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # requests' own logger would print the URL, which contains the bot token.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env (see trading_engine/notify.py).")
        return 1
    ks = KillSwitch(DATA_DIR / "KILL_SWITCH")
    bot = CommandBot(token, chat_id, ks, _live_status(ks), offset_path=DATA_DIR / "telegram_offset")
    logger.info("listening for /stop, /resume and /status from chat %s", chat_id)
    bot.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
