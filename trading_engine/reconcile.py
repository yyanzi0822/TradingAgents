"""Bring the ledger and the protective stops in line with the broker.

Two jobs, both safe to run at any time and as often as you like:

``refresh_orders``
    The ledger records an order when it is submitted ("accepted"). This asks
    the broker what happened next and records fills, partial fills, expiries
    and cancels. A filled stop order is recorded as a stop-out.

``sync_stops``
    Every position should have exactly one resting stop order at the broker,
    for the whole position, at the price in the stop book. This places missing
    stops, replaces ones whose quantity or price no longer matches, cancels
    stops for positions that are gone, and sells at market if the price is
    already through the stop.

The stop orders rest at the broker (good-till-cancelled), so they trigger even
when this program is not running.

    python -m trading_engine.reconcile            # run both now
    python -m trading_engine.reconcile --dry-run  # show what would change
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from trading_engine.broker import Broker, OrderStatus
from trading_engine.config import RiskLimits
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.models import TERMINAL_STATUSES, OrderRequest, Side
from trading_engine.risk import client_order_id
from trading_engine.stops import StopBook

MARKET_TZ = ZoneInfo("America/New_York")
_TRACKED_KINDS = ("order", "stop_order", "order_update")


class Reconciler:
    def __init__(
        self,
        broker: Broker,
        ledger: Ledger,
        limits: RiskLimits | None = None,
        kill_switch: KillSwitch | None = None,
        *,
        stops: StopBook | None = None,
        dry_run: bool = False,
        require_market_open: bool = True,
        confirm: Callable[[str], bool] | None = None,
    ):
        self.broker = broker
        self.ledger = ledger
        self.limits = limits or RiskLimits()
        self.kill_switch = kill_switch or KillSwitch()
        self.stops = stops or StopBook(ledger)
        self.dry_run = dry_run
        self.require_market_open = require_market_open
        self.confirm = confirm

    def run(self, trade_date: str | None = None) -> list[str]:
        return self.refresh_orders() + self.sync_stops(trade_date)

    # -- order status ---------------------------------------------------------

    def _tracked(self) -> dict[str, dict]:
        """Latest known state of every order this engine has submitted."""
        latest: dict[str, dict] = {}
        for rec in self.ledger.records():
            data = rec["data"]
            oid = data.get("client_order_id")
            if rec["kind"] not in _TRACKED_KINDS or not oid:
                continue
            entry = latest.setdefault(oid, {"role": "stop" if rec["kind"] == "stop_order" else "order",
                                            "filled_quantity": 0.0})
            for key in ("ticker", "side", "quantity", "status", "filled_quantity"):
                if data.get(key) is not None:
                    entry[key] = data[key]
        return latest

    def refresh_orders(self) -> list[str]:
        notes: list[str] = []
        for oid, known in self._tracked().items():
            if known.get("status") in TERMINAL_STATUSES:
                continue
            status = self.broker.get_order(oid)
            if status is None:
                self.ledger.append("order_update", {"client_order_id": oid, "ticker": known.get("ticker"),
                                                    "status": "missing"})
                notes.append(f"{known.get('ticker')}: order {oid[:14]} is not at the broker")
                continue
            if status.status == known.get("status") and status.filled_quantity == known.get("filled_quantity"):
                continue
            self.ledger.append("order_update", {
                "client_order_id": oid, "ticker": status.ticker, "role": known["role"],
                "side": status.side, "quantity": status.quantity, "status": status.status,
                "filled_quantity": status.filled_quantity, "filled_avg_price": status.filled_avg_price,
            })
            notes.append(self._describe(status, known["role"]))
            if known["role"] == "stop" and status.status == "filled":
                self.ledger.append("stop_filled", {
                    "ticker": status.ticker, "trade_date": _market_date(status.filled_at),
                    "quantity": status.filled_quantity, "price": status.filled_avg_price,
                    "client_order_id": oid,
                })
        return notes

    @staticmethod
    def _describe(status: OrderStatus, role: str) -> str:
        what = "stop" if role == "stop" else status.side
        price = f" @ {status.filled_avg_price:,.2f}" if status.filled_avg_price else ""
        if status.status == "filled":
            return f"{status.ticker}: {what} filled {status.filled_quantity:g}{price}"
        if status.filled_quantity and status.status in TERMINAL_STATUSES:
            return (f"{status.ticker}: {what} {status.status} after partial fill "
                    f"{status.filled_quantity:g} of {status.quantity:g}{price}")
        if status.filled_quantity:
            return f"{status.ticker}: {what} partially filled {status.filled_quantity:g} of {status.quantity:g}"
        return f"{status.ticker}: {what} {status.status}"

    # -- protective stops -----------------------------------------------------

    def cancel_stops(self, ticker: str, open_orders: list[OrderStatus] | None = None) -> int:
        """Cancel resting stops for a ticker (they reserve the shares a sell needs)."""
        ticker = ticker.upper()
        orders = self.broker.open_orders() if open_orders is None else open_orders
        cancelled = 0
        for o in orders:
            if o.ticker == ticker and o.is_stop and o.side == Side.SELL.value:
                self._cancel(o, "position is being sold")
                cancelled += 1
        return cancelled

    def _cancel(self, order: OrderStatus, reason: str) -> None:
        result = self.broker.cancel(order.client_order_id)
        self.ledger.append("order_update", {
            "client_order_id": order.client_order_id, "ticker": order.ticker, "role": "stop",
            "status": result.status if result else "missing", "reason": reason,
        })

    def sync_stops(self, trade_date: str | None = None) -> list[str]:
        if not self.limits.enforce_stops:
            return []
        if self.kill_switch.active():
            # No new orders of any kind. Stops already resting at the broker stay there.
            return ["kill switch active: stop orders left as they are"]

        trade_date = trade_date or date.today().isoformat()
        notes: list[str] = []
        account = self.broker.get_account()
        open_orders = self.broker.open_orders()
        resting: dict[str, list[OrderStatus]] = {}
        selling: set[str] = set()
        buying: set[str] = set()
        for o in open_orders:
            if o.side == Side.SELL.value and o.is_stop:
                resting.setdefault(o.ticker, []).append(o)
            elif o.side == Side.SELL.value:
                selling.add(o.ticker)
            else:
                buying.add(o.ticker)
        held = {t: h for t, h in account.holdings.items() if h.quantity > 0}
        book = self.stops.current()

        # Positions that are gone: drop their stops and their stop price.
        for ticker in sorted(set(resting) | set(book)):
            if ticker in held or ticker in buying:
                continue
            for o in resting.pop(ticker, []):
                notes.append(f"{ticker}: cancel stop {o.quantity:g} @ {o.stop_price} (no position)")
                if not self.dry_run:
                    self._cancel(o, "no position")
            if ticker in book and not self.dry_run:
                self.stops.clear(ticker, "position closed")

        market_open: bool | None = None
        for ticker, holding in sorted(held.items()):
            stop = book.get(ticker)
            if stop is None:
                if not holding.avg_entry_price:
                    notes.append(f"{ticker}: no stop on record and no entry price; not protected")
                    continue
                stop = round(holding.avg_entry_price * (1 - self.limits.default_stop_pct), 2)
                if not self.dry_run:
                    stop = self.stops.set(ticker, stop, source="default from entry price")
            if ticker in selling:
                notes.append(f"{ticker}: sell order still open; stop will be set once it settles")
                continue

            want_qty = float(math.floor(holding.quantity))
            existing = resting.get(ticker, [])
            if (len(existing) == 1 and existing[0].remaining == want_qty
                    and existing[0].stop_price is not None and abs(existing[0].stop_price - stop) < 0.005):
                continue

            quote = self.broker.get_quote(ticker)
            if quote.price <= stop:
                if market_open is None:
                    market_open = self.broker.is_market_open()
                notes.append(self._exit_breached(ticker, holding.quantity, stop, quote.price,
                                                 existing, trade_date, market_open))
                continue

            for o in existing:
                notes.append(f"{ticker}: replace stop {o.remaining:g} @ {o.stop_price}")
                if not self.dry_run:
                    self._cancel(o, "quantity or price changed")
            if want_qty <= 0:
                notes.append(f"{ticker}: under one share; stop {stop:,.2f} is checked each run, not at the broker")
                continue
            notes.append(f"{ticker}: {'would place' if self.dry_run else 'placed'} stop {want_qty:g} @ {stop:,.2f}")
            if not self.dry_run:
                self._place_stop(ticker, want_qty, stop)
        return notes

    def _place_stop(self, ticker: str, quantity: float, stop: float) -> None:
        # Alpaca never accepts a client_order_id twice, even for a cancelled
        # order, so the id carries a counter taken from the ledger.
        nonce = len(self.ledger.records("stop_order"))
        raw = f"{ticker}|{quantity:g}|{stop:.2f}|{nonce}"
        order = OrderRequest(
            client_order_id="ta-stop-" + hashlib.sha256(raw.encode()).hexdigest()[:32],
            ticker=ticker, side=Side.SELL, quantity=quantity, stop_price=stop, time_in_force="gtc",
        )
        status = self.broker.submit(order)
        self.ledger.append("stop_order", {
            "client_order_id": order.client_order_id, "broker_order_id": status.broker_order_id,
            "ticker": ticker, "side": Side.SELL.value, "quantity": quantity, "stop_price": stop,
            "status": status.status,
        })

    def _exit_breached(self, ticker: str, quantity: float, stop: float, price: float,
                       existing: list[OrderStatus], trade_date: str, market_open: bool) -> str:
        """The price is already at or below the stop: sell the position at market."""
        summary = f"{ticker}: price {price:,.2f} is through stop {stop:,.2f}"
        if self.dry_run:
            return f"{summary}; would sell {quantity:g} at market"
        if self.require_market_open and not market_open:
            return f"{summary}; market closed, will sell on the next run while it is open"
        if self.confirm and not self.confirm(f"{summary}. Sell {quantity:g} at market?"):
            self.ledger.append("declined", {"ticker": ticker, "trade_date": trade_date, "what": "stop exit"})
            return f"{summary}; exit declined by you"
        for o in existing:
            self._cancel(o, "stop breached; exiting at market")
        order = OrderRequest(
            client_order_id=client_order_id(f"{trade_date}|{ticker}|stop-exit", ticker, Side.SELL),
            ticker=ticker, side=Side.SELL, quantity=quantity,
        )
        status = self.broker.submit(order)
        self.ledger.append("order", {
            "decision_id": f"{trade_date}|{ticker}|stop-exit", "trade_date": trade_date, "ticker": ticker,
            "client_order_id": order.client_order_id, "broker_order_id": status.broker_order_id,
            "side": Side.SELL.value, "quantity": quantity, "status": status.status,
            "duplicate": status.duplicate, "reason": f"price {price} through stop {stop}",
        })
        self.ledger.append("stop_filled", {"ticker": ticker, "trade_date": trade_date, "quantity": quantity,
                                           "price": price, "client_order_id": order.client_order_id})
        return f"{summary}; sold {quantity:g} at market -> {status.status}"


def _market_date(when: datetime | None) -> str:
    when = when or datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(MARKET_TZ).date().isoformat()


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    from trading_engine.broker import AlpacaPaperBroker

    load_dotenv()
    p = argparse.ArgumentParser(description="Update order statuses and protective stops from the broker")
    p.add_argument("--dry-run", action="store_true", help="show what would change; send nothing")
    p.add_argument("--confirm", action="store_true", help="ask before a market sell on a breached stop")
    args = p.parse_args(argv)

    data_dir = Path(os.environ.get("TRADING_ENGINE_DIR", Path.home() / ".trading_engine"))
    from trading_engine.run import ask_yes_no

    reconciler = Reconciler(
        AlpacaPaperBroker(), Ledger(data_dir / "ledger.jsonl"),
        kill_switch=KillSwitch(data_dir / "KILL_SWITCH"),
        dry_run=args.dry_run, confirm=ask_yes_no if args.confirm else None,
    )
    notes = reconciler.refresh_orders() if not args.dry_run else []
    notes += reconciler.sync_stops()
    print("\n".join(notes) if notes else "nothing to do: orders and stops match the broker")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
