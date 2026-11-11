"""Which take-profit levels have been taken on each open position.

Kept in the ledger, like stops. A level is taken once per position: when the
position is closed the record is reset, so the next entry starts fresh.
"""

from __future__ import annotations

from trading_engine.ledger import Ledger


class TakeProfitBook:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def state(self) -> dict[str, dict]:
        """Ticker -> {"levels": set of level indexes taken, "base": shares held at the first one}."""
        state: dict[str, dict] = {}
        for rec in self.ledger.records("take_profit"):
            data = rec["data"]
            if data.get("reset"):
                state.pop(data["ticker"], None)
                continue
            entry = state.setdefault(data["ticker"], {"levels": set(), "base": data["base_quantity"]})
            entry["levels"].add(data["level"])
        return state

    def taken(self, ticker: str) -> set[int]:
        return self.state().get(ticker.upper(), {}).get("levels", set())

    def record(self, ticker: str, level: int, **details) -> None:
        self.ledger.append("take_profit", {"ticker": ticker.upper(), "level": level, **details})

    def reset(self, ticker: str) -> None:
        if ticker.upper() in self.state():
            self.ledger.append("take_profit", {"ticker": ticker.upper(), "reset": True})
