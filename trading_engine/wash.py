"""Wash-sale detection from the engine's own ledger.

A wash sale, in short: you sell a stock at a loss and buy the same stock
within 30 days before or after that sale. The loss is then not deductible in
the usual way. This module flags the two ways the engine can cause one:

- **Buying back** a ticker that was sold at a loss in the last 30 days
  (the common case here: stopped out, then the agents rate it Buy again).
- **Selling part** of a position at a loss when shares of it were bought in
  the last 30 days and some are kept.

What it cannot see, so treat every result as "possible", not as a tax answer:

- trades in any other account (another broker, an IRA, a spouse's account),
  which count under the rule;
- "substantially identical" securities other than the same ticker (options,
  another share class, a near-identical fund);
- tax lots. Losses are judged against the broker's *average* entry price,
  while your broker reports per lot, so a sale can be a loss on one lot and a
  gain on average.

Paper trading has no tax effect. This is bookkeeping help, not tax advice.

    python -m trading_engine.wash      # tickers currently inside a window
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from trading_engine.ledger import Ledger
from trading_engine.models import Side

WINDOW_DAYS = 30
_DEAD = frozenset({"canceled", "expired", "rejected", "missing"})


@dataclass(frozen=True)
class Trade:
    ticker: str
    side: str
    day: date
    quantity: float
    price: float | None           # fill price, or the quote when the order was sent
    cost_basis: float | None      # average entry price when sold (sells only)
    client_order_id: str | None = None

    @property
    def is_loss(self) -> bool | None:
        """True / False, or None when price or cost basis is not on record."""
        if self.side != Side.SELL.value or self.price is None or self.cost_basis is None:
            return None
        return self.price < self.cost_basis


def trades(ledger: Ledger) -> list[Trade]:
    """Every buy and sell the engine made, with what is known about its fill."""
    fills: dict[str, dict] = {}
    basis_at_stop: dict[str, float | None] = {}
    for rec in ledger.records():
        data = rec["data"]
        oid = data.get("client_order_id")
        if rec["kind"] == "order_update" and oid:
            fills.setdefault(oid, {}).update({k: v for k, v in data.items() if v is not None})
        elif rec["kind"] == "stop_order" and oid:
            basis_at_stop[oid] = data.get("avg_entry_price")

    out: list[Trade] = []
    seen: set[str] = set()
    for rec in ledger.records("order"):
        data = rec["data"]
        oid = data.get("client_order_id")
        if not data.get("trade_date") or oid in seen:
            continue
        fill = fills.get(oid, {})
        filled = float(fill.get("filled_quantity") or data.get("filled_quantity") or 0.0)
        if fill.get("status", data.get("status")) in _DEAD and filled <= 0:
            continue                                  # never traded
        seen.add(oid)
        out.append(Trade(
            ticker=data["ticker"], side=data["side"], day=date.fromisoformat(data["trade_date"]),
            quantity=filled or float(data.get("quantity") or 0.0),
            price=fill.get("filled_avg_price") or data.get("filled_avg_price") or data.get("est_price"),
            cost_basis=data.get("avg_entry_price"), client_order_id=oid,
        ))
    for rec in ledger.records("stop_filled"):
        data = rec["data"]
        oid = data.get("client_order_id")
        if oid in seen or not data.get("trade_date"):
            continue                                  # a market stop-exit is already an order above
        seen.add(oid)
        out.append(Trade(
            ticker=data["ticker"], side=Side.SELL.value, day=date.fromisoformat(data["trade_date"]),
            quantity=float(data.get("quantity") or 0.0), price=data.get("price"),
            cost_basis=data.get("avg_entry_price", basis_at_stop.get(oid)), client_order_id=oid,
        ))
    return sorted(out, key=lambda t: t.day)


class WashSaleChecker:
    def __init__(self, ledger: Ledger, window_days: int = WINDOW_DAYS):
        self.ledger = ledger
        self.window = timedelta(days=window_days)

    def _recent(self, ticker: str, day: date, side: Side) -> list[Trade]:
        ticker = ticker.upper()
        return [t for t in trades(self.ledger)
                if t.ticker == ticker and t.side == side.value and timedelta(0) <= day - t.day <= self.window]

    def loss_sales_before(self, ticker: str, day: date) -> list[Trade]:
        """Sales in the last 30 days that were losses, or whose result is unknown."""
        return [t for t in self._recent(ticker, day, Side.SELL) if t.is_loss is not False]

    def buys_before(self, ticker: str, day: date) -> list[Trade]:
        return self._recent(ticker, day, Side.BUY)

    def check_buy(self, ticker: str, day: date) -> str | None:
        sales = self.loss_sales_before(ticker, day)
        if not sales:
            return None
        last = sales[-1]
        how = "at a loss" if last.is_loss else "(gain or loss not on record)"
        return (f"possible wash sale: {ticker.upper()} was sold {how} on {last.day}; "
                f"a buy before {self.clear_date(last)} falls inside the 30-day window")

    def check_sell(self, ticker: str, day: date, *, price: float, cost_basis: float | None,
                   quantity: float, held: float) -> str | None:
        """A partial sale at a loss while recently bought shares are kept."""
        if cost_basis is None or price >= cost_basis or quantity >= held:
            return None
        buys = self.buys_before(ticker, day)
        if not buys:
            return None
        return (f"possible wash sale: selling part of {ticker.upper()} at a loss "
                f"within 30 days of buying it on {buys[-1].day}")

    def clear_date(self, sale: Trade) -> date:
        """First day a buy is outside the window of this sale."""
        return sale.day + self.window + timedelta(days=1)

    def windows(self, day: date) -> dict[str, tuple[date, date]]:
        """Tickers inside a window today: ticker -> (loss sale day, first clear day)."""
        open_windows: dict[str, tuple[date, date]] = {}
        for t in trades(self.ledger):
            if t.side == Side.SELL.value and t.is_loss is not False and timedelta(0) <= day - t.day <= self.window:
                open_windows[t.ticker] = (t.day, self.clear_date(t))
        return open_windows


def main(argv: list[str] | None = None) -> int:
    import os
    from pathlib import Path

    from trading_engine.reconcile import _market_date

    data_dir = Path(os.environ.get("TRADING_ENGINE_DIR", Path.home() / ".trading_engine"))
    today = date.fromisoformat(_market_date(None))
    open_windows = WashSaleChecker(Ledger(data_dir / "ledger.jsonl")).windows(today)
    if not open_windows:
        print("No ticker is inside a wash-sale window (from this engine's ledger only).")
        return 0
    print("Buying these again before the date shown is a possible wash sale:")
    for ticker, (sold, clear) in sorted(open_windows.items()):
        print(f"  {ticker}: loss sale {sold}, clear from {clear}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
