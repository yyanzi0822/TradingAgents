"""End-of-day summary: account, positions with their stops, and what happened today.

    python -m trading_engine.summary           # print it
    python -m trading_engine.summary --send    # also send it to Telegram
"""

from __future__ import annotations

from datetime import datetime

from trading_engine.broker import Broker
from trading_engine.config import RiskLimits
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.reconcile import Reconciler, _market_date
from trading_engine.stops import StopBook


def build_summary(broker: Broker, ledger: Ledger, kill_switch: KillSwitch,
                  limits: RiskLimits | None = None, day: str | None = None) -> str:
    limits = limits or RiskLimits()
    day = day or _market_date(None)
    account = broker.get_account()
    stops = StopBook(ledger).current()

    lines = [f"Daily summary {day}"]

    # Change since the last equity reading from an earlier day.
    earlier = [r["data"]["equity"] for r in ledger.records("equity")
               if _market_date(datetime.fromisoformat(r["at"])) < day]
    change = ""
    if earlier and earlier[-1] > 0:
        change = f" ({(account.equity / earlier[-1] - 1):+.2%} since last session)"
    lines.append(f"Equity ${account.equity:,.2f}{change} · cash ${account.cash:,.2f}")

    peak = max([account.equity, ledger.high_water_mark() or 0.0])
    drawdown = 0.0 if peak <= 0 else (peak - account.equity) / peak
    halted = " — BUYS HALTED" if drawdown >= limits.max_drawdown else ""
    lines.append(f"Drawdown from peak {drawdown:.1%} (buys halt at {limits.max_drawdown:.0%}){halted}")

    held = sorted((h for h in account.holdings.values() if h.quantity), key=lambda h: -h.market_value)
    lines.append(f"Positions ({len(held)}):" if held else "Positions: none")
    for h in held:
        stop = f"stop {stops[h.ticker]:,.2f}" if h.ticker in stops else "NO STOP"
        lines.append(f"  {h.ticker} {h.quantity:g} sh ${h.market_value:,.0f} "
                     f"{account.weight(h.ticker):.1%} · {stop}")

    tracked = Reconciler(broker, ledger, limits, kill_switch)._tracked()
    orders = [r["data"] for r in ledger.records("order") if r["data"].get("trade_date") == day]
    stop_outs = [r["data"] for r in ledger.records("stop_filled") if r["data"].get("trade_date") == day]
    errors = [r["data"] for r in ledger.records("error") if r["data"].get("trade_date") == day]
    lines.append(f"Today: {len(orders)} order(s), {len(stop_outs)} stop-out(s), {len(errors)} error(s)")
    for o in orders:
        status = tracked.get(o["client_order_id"], {}).get("status", o.get("status"))
        lines.append(f"  {o['side']} {o['quantity']:g} {o['ticker']} — {status}")
    for s in stop_outs:
        price = f" @ {s['price']:,.2f}" if s.get("price") else ""
        lines.append(f"  stopped out {s['ticker']} {s['quantity']:g}{price}")
    for e in errors:
        lines.append(f"  error {e.get('ticker') or ''}: {str(e.get('error'))[:120]}")

    lines.append(f"Kill switch: ON — {kill_switch.reason()}" if kill_switch.active() else "Kill switch: off")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    from trading_engine import notify
    from trading_engine.broker import AlpacaPaperBroker

    load_dotenv()
    p = argparse.ArgumentParser(description="Print (and optionally send) the daily summary")
    p.add_argument("--send", action="store_true", help="also send it to Telegram")
    args = p.parse_args(argv)

    data_dir = Path(os.environ.get("TRADING_ENGINE_DIR", Path.home() / ".trading_engine"))
    text = build_summary(AlpacaPaperBroker(), Ledger(data_dir / "ledger.jsonl"),
                         KillSwitch(data_dir / "KILL_SWITCH"))
    print(text)
    if args.send and not notify.from_env().send(text):
        print("\n(not sent: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, see trading_engine/notify.py)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
