"""Broker adapters.

``AlpacaPaperBroker`` talks to Alpaca's *paper* endpoint only. It refuses to
start against a live account, so a misconfigured key cannot move real money.

``FakeBroker`` is an in-memory broker for tests and dry runs.

Both are idempotent: an order whose ``client_order_id`` already exists is
returned as-is instead of being submitted again.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Protocol

from trading_engine.models import AccountSnapshot, Holding, OrderRequest, Quote, Side


class LiveTradingRefused(RuntimeError):
    pass


@dataclass
class OrderStatus:
    client_order_id: str
    broker_order_id: str | None
    ticker: str
    side: str
    quantity: float
    status: str                 # new / accepted / filled / partially_filled / rejected / ...
    filled_quantity: float = 0.0
    filled_avg_price: float | None = None
    duplicate: bool = False     # True when we found it already submitted


class Broker(Protocol):
    def get_account(self) -> AccountSnapshot: ...
    def get_quote(self, ticker: str) -> Quote: ...
    def is_market_open(self) -> bool: ...
    def get_order(self, client_order_id: str) -> OrderStatus | None: ...
    def submit(self, order: OrderRequest) -> OrderStatus: ...


# ---------------------------------------------------------------------------
# Alpaca (paper only)
# ---------------------------------------------------------------------------


class AlpacaPaperBroker:
    """Alpaca paper trading via ``alpaca-py`` (``pip install alpaca-py``).

    Reads ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` from the environment.
    """

    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
        base = os.environ.get("ALPACA_BASE_URL", "")
        if base and "paper" not in base:
            raise LiveTradingRefused(f"ALPACA_BASE_URL={base!r} is not a paper endpoint; live trading is disabled in v1")
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError("Alpaca support needs `pip install alpaca-py` (or `pip install .[alpaca]`)") from exc

        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (paper account keys) in .env")

        # paper=True is hard-coded: there is no argument that turns it off.
        self._trading = TradingClient(key, secret, paper=True)
        self._data = StockHistoricalDataClient(key, secret)

        account = self._trading.get_account()
        number = str(getattr(account, "account_number", ""))
        if number and not number.upper().startswith("P"):
            # Alpaca paper account numbers start with "P".
            raise LiveTradingRefused(f"account {number} does not look like a paper account")

    def get_account(self) -> AccountSnapshot:
        acct = self._trading.get_account()
        holdings = {}
        for p in self._trading.get_all_positions():
            holdings[p.symbol.upper()] = Holding(
                ticker=p.symbol.upper(),
                quantity=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price) if p.avg_entry_price else None,
            )
        return AccountSnapshot(
            equity=float(acct.equity),
            cash=float(acct.cash),
            holdings=holdings,
            as_of=datetime.now(UTC),
        )

    def get_quote(self, ticker: str) -> Quote:
        from alpaca.data.requests import StockLatestTradeRequest

        trade = self._data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=ticker.upper()))
        t = trade[ticker.upper()]
        ts = t.timestamp if t.timestamp.tzinfo else t.timestamp.replace(tzinfo=UTC)
        return Quote(ticker.upper(), float(t.price), ts)

    def is_market_open(self) -> bool:
        return bool(self._trading.get_clock().is_open)

    def get_order(self, client_order_id: str) -> OrderStatus | None:
        from alpaca.common.exceptions import APIError

        try:
            o = self._trading.get_order_by_client_id(client_order_id)
        except APIError as exc:
            if getattr(exc, "status_code", None) == 404 or "not found" in str(exc).lower():
                return None
            raise
        return self._status(o)

    def submit(self, order: OrderRequest) -> OrderStatus:
        existing = self.get_order(order.client_order_id)
        if existing is not None:
            return replace(existing, duplicate=True)

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        common = {
            "symbol": order.ticker,
            "qty": order.quantity,
            "side": OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL,
            "time_in_force": TimeInForce.DAY,
            "client_order_id": order.client_order_id,
        }
        req = (LimitOrderRequest(limit_price=order.limit_price, **common)
               if order.limit_price is not None else MarketOrderRequest(**common))
        return self._status(self._trading.submit_order(order_data=req))

    @staticmethod
    def _status(o) -> OrderStatus:
        return OrderStatus(
            client_order_id=o.client_order_id,
            broker_order_id=str(o.id),
            ticker=o.symbol,
            side=str(getattr(o.side, "value", o.side)),
            quantity=float(o.qty or 0),
            status=str(getattr(o.status, "value", o.status)),
            filled_quantity=float(o.filled_qty or 0),
            filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
        )


# ---------------------------------------------------------------------------
# In-memory broker for tests and dry runs
# ---------------------------------------------------------------------------


@dataclass
class FakeBroker:
    cash: float = 100_000.0
    prices: dict[str, float] = field(default_factory=dict)
    positions: dict[str, float] = field(default_factory=dict)
    market_open: bool = True
    orders: dict[str, OrderStatus] = field(default_factory=dict)
    fill_immediately: bool = True

    def get_account(self) -> AccountSnapshot:
        holdings = {
            t: Holding(t, q, q * self.prices[t]) for t, q in self.positions.items() if q
        }
        equity = self.cash + sum(h.market_value for h in holdings.values())
        return AccountSnapshot(equity=equity, cash=self.cash, holdings=holdings,
                               as_of=datetime.now(UTC))

    def get_quote(self, ticker: str) -> Quote:
        return Quote(ticker.upper(), self.prices[ticker.upper()], datetime.now(UTC))

    def is_market_open(self) -> bool:
        return self.market_open

    def get_order(self, client_order_id: str) -> OrderStatus | None:
        return self.orders.get(client_order_id)

    def submit(self, order: OrderRequest) -> OrderStatus:
        if order.client_order_id in self.orders:
            return replace(self.orders[order.client_order_id], duplicate=True)
        price = self.prices[order.ticker]
        status = OrderStatus(order.client_order_id, f"fake-{len(self.orders) + 1}", order.ticker,
                             order.side.value, order.quantity, "accepted")
        if self.fill_immediately:
            sign = 1 if order.side is Side.BUY else -1
            self.positions[order.ticker] = self.positions.get(order.ticker, 0.0) + sign * order.quantity
            self.cash -= sign * order.quantity * price
            status.status, status.filled_quantity, status.filled_avg_price = "filled", order.quantity, price
        self.orders[order.client_order_id] = status
        return status
