"""Command-line entry point for one trading cycle.

Examples:
    # No broker, no LLM spend on orders: analyze and show what would trade
    python -m trading_engine.run --tickers AAPL,MSFT --dry-run

    # Alpaca paper account (needs ALPACA_API_KEY / ALPACA_SECRET_KEY)
    python -m trading_engine.run --tickers AAPL,MSFT,JPM

    # Whole starter universe, cheap model settings
    python -m trading_engine.run --universe --analysts market,news,fundamentals
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.runner import Runner, tradingagents_analyzer

UNIVERSE_FILE = Path(__file__).with_name("universe.json")
DATA_DIR = Path(os.environ.get("TRADING_ENGINE_DIR", Path.home() / ".trading_engine"))


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
    p.add_argument("--ledger", default=str(DATA_DIR / "ledger.jsonl"))
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
        Ledger(args.ledger),
        sectors=sectors,
        kill_switch=KillSwitch(DATA_DIR / "KILL_SWITCH"),
        dry_run=args.dry_run,
        require_market_open=not args.allow_closed,
    )
    report = runner.run_cycle(tickers, args.date)
    print(report.render())
    return 1 if report.halted else 0


if __name__ == "__main__":
    raise SystemExit(main())
