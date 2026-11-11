"""Stop prices per position, kept in the ledger.

The stop a position was sized with is recorded when its buy is submitted.
While the position is open the stop can be raised but never lowered: a later
decision from the agents cannot loosen it. It is cleared once the position is
gone, so the next entry starts fresh.
"""

from __future__ import annotations

from trading_engine.ledger import Ledger


class StopBook:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def current(self) -> dict[str, float]:
        """Ticker -> active stop price, replayed from the ledger."""
        stops: dict[str, float] = {}
        for rec in self.ledger.records("stop"):
            ticker, price = rec["data"]["ticker"], rec["data"]["stop_price"]
            if price is None:
                stops.pop(ticker, None)
            else:
                stops[ticker] = float(price)
        return stops

    def set(self, ticker: str, stop_price: float, *, source: str, decision_id: str | None = None) -> float:
        """Record a stop. Returns the stop now in force (the higher of old and new)."""
        ticker = ticker.upper()
        stop_price = round(float(stop_price), 2)
        existing = self.current().get(ticker)
        if existing is not None and stop_price <= existing:
            return existing
        self.ledger.append("stop", {"ticker": ticker, "stop_price": stop_price, "source": source,
                                    "decision_id": decision_id, "previous": existing})
        return stop_price

    def clear(self, ticker: str, reason: str) -> None:
        ticker = ticker.upper()
        if ticker in self.current():
            self.ledger.append("stop", {"ticker": ticker, "stop_price": None, "source": reason})
