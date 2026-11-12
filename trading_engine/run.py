"""Command-line entry point for one trading cycle.

Examples:
    # No broker, no LLM spend on orders: analyze and show what would trade
    python -m trading_engine.run --tickers AAPL,MSFT --dry-run

    # Alpaca paper account (needs ALPACA_API_KEY / ALPACA_SECRET_KEY)
    python -m trading_engine.run --tickers AAPL,MSFT,JPM

    # Whole starter universe, cheap model settings
    python -m trading_engine.run --universe --analysts market,news,fundamentals

A real run sends every ticker's rating, action and reason to Telegram;
add --notify to a dry run to send it too.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from trading_engine import notify
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.runner import Runner, tradingagents_analyzer

UNIVERSE_FILE = Path(__file__).with_name("universe.json")
DATA_DIR = Path(os.environ.get("TRADING_ENGINE_DIR", Path.home() / ".trading_engine"))


def default_ledger_path(dry_run: bool, data_dir: Path = DATA_DIR) -> Path:
    """Dry runs keep their own ledger.

    A dry run can use a made-up account (FakeBroker, $100k). Its equity records
    must not feed the high-water mark of the real paper account, or the
    drawdown breaker measures from a number that account never had.
    """
    return data_dir / ("ledger-dryrun.jsonl" if dry_run else "ledger.jsonl")


def ask_yes_no(question: str) -> bool:
    """Console prompt for --confirm. Anything but y/yes, or no terminal, means no."""
    try:
        return input(f"{question}  Send it? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def load_sectors(path: Path = UNIVERSE_FILE) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))["tickers"]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Run one TradingAgents + risk-engine cycle")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tickers", help="comma-separated, e.g. AAPL,MSFT")
    g.add_argument("--universe", action="store_true", help="every ticker in universe.json")
    p.add_argument("--date", help="analysis date YYYY-MM-DD (default: today)")
    p.add_argument("--analysts", help="subset of analysts, e.g. market,news,fundamentals")
    p.add_argument("--dry-run", action="store_true", help="analyze and size, but never submit")
    p.add_argument("--allow-closed", action="store_true", help="submit even when the market is closed")
    p.add_argument("--confirm", action="store_true",
                   help="show each buy or sell and wait for y before sending it")
    p.add_argument("--settle-seconds", type=float, default=20.0,
                   help="how long to wait for fills before placing stop orders (default 20)")
    p.add_argument("--ledger", help="ledger file (default: ledger.jsonl, or ledger-dryrun.jsonl with --dry-run)")
    p.add_argument("--notify", action="store_true",
                   help="send the per-ticker report to Telegram in a dry run too (a real run always sends it)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sectors = load_sectors()
    tickers = list(sectors) if args.universe else [t for t in args.tickers.split(",") if t.strip()]

    if args.dry_run and not os.environ.get("ALPACA_API_KEY"):
        from trading_engine.broker import FakeBroker
        from trading_engine.dryrun_prices import latest_closes

        broker = FakeBroker(prices=latest_closes(tickers))
    else:
        from trading_engine.broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker()

    runner = Runner(
        broker,
        tradingagents_analyzer(analysts=args.analysts.split(",") if args.analysts else None),
        Ledger(args.ledger or default_ledger_path(args.dry_run)),
        sectors=sectors,
        kill_switch=KillSwitch(DATA_DIR / "KILL_SWITCH"),
        dry_run=args.dry_run,
        require_market_open=not args.allow_closed,
        settle_seconds=args.settle_seconds,
        confirm=ask_yes_no if args.confirm else None,
        notifier=notify.from_env(),
    )
    report = runner.run_cycle(tickers, args.date)
    print(report.render())
    if not args.dry_run or args.notify:
        notifier = notify.from_env()
        if isinstance(notifier, notify.NullNotifier):
            print("(report not sent: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")
        elif not notify.send_blocks(notifier, report.telegram_blocks(dry_run=args.dry_run)):
            print("(report not fully sent to Telegram; see the log)")
    if not report.halted:
        count, last = runner.ledger.head()
        print(f"ledger head: {count} records, {last}")
    return 1 if report.halted else 0


if __name__ == "__main__":
    raise SystemExit(main())
