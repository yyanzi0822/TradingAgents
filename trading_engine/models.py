"""Plain data types shared by the risk engine, broker and runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Holding:
    ticker: str
    quantity: float
    market_value: float
    avg_entry_price: float | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    """The broker's view of the account at one moment."""

    equity: float
    cash: float
    holdings: dict[str, Holding] = field(default_factory=dict)
    # Highest equity seen so far; the drawdown breaker measures from here.
    high_water_mark: float | None = None
    as_of: datetime | None = None

    @property
    def gross_exposure(self) -> float:
        return sum(abs(h.market_value) for h in self.holdings.values())

    def weight(self, ticker: str) -> float:
        h = self.holdings.get(ticker.upper())
        return (h.market_value / self.equity) if h and self.equity > 0 else 0.0

    def drawdown(self) -> float:
        hwm = max(self.high_water_mark or self.equity, self.equity)
        return 0.0 if hwm <= 0 else (hwm - self.equity) / hwm


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float
    timestamp: datetime


@dataclass(frozen=True)
class TradeIntent:
    """What the agents want, expressed as a target weight for one ticker."""

    decision_id: str
    ticker: str
    rating: str
    target_weight: float | None           # None = leave the position alone
    stop_loss: float | None = None
    thesis: str = ""
    time_horizon: str | None = None


@dataclass(frozen=True)
class OrderRequest:
    client_order_id: str
    ticker: str
    side: Side
    quantity: float
    limit_price: float | None = None      # None = market order


@dataclass
class RiskResult:
    approved: bool
    order: OrderRequest | None
    reasons: list[str] = field(default_factory=list)
    # Checks that shrank the order rather than rejecting it.
    adjustments: list[str] = field(default_factory=list)
