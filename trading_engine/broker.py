"""Broker adapters.

``AlpacaPaperBroker`` talks to Alpaca's *paper* endpoint only. It refuses to
start against a live account, so a misconfigured key cannot move real money.

``FakeBroker`` is an in-memory broker for tests and dry runs.

Both are idempotent: an order whose ``client_order_id`` already exists is
returned as-is instead of being submitted again.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Protocol

from trading_engine.models import (
    STOP_ORDER_TYPES,
    TERMINAL_STATUSES,
    AccountSnapshot,
    Holding,
    OrderRequest,
    Quote,
    Side,
)


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
    order_type: str = "market"  # market / limit / stop / stop_limit / ...
    stop_price: float | None = None
    limit_price: float | None = None
    filled_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.status not in TERMINAL_STATUSES

    @property
    def is_stop(self) -> bool:
        return self.order_type in STOP_ORDER_TYPES

    @property
    def remaining(self) -> float:
        return max(self.quantity - self.filled_quantity, 0.0)


class Broker(Protocol):
    def get_account(self) -> AccountSnapshot: ...
    def get_quote(self, ticker: str) -> Quote: ...
    def is_market_open(self) -> bool: ...
    def get_order(self, client_order_id: str) -> OrderStatus | None: ...
    def submit(self, order: OrderRequest) -> OrderStatus: ...
    def open_orders(self) -> list[OrderStatus]: ...
    def cancel(self, client_order_id: str) -> OrderStatus | None: ...


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
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopOrderRequest

        common = {
            "symbol": order.ticker,
            "qty": order.quantity,
            "side": OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL,
            "time_in_force": TimeInForce.GTC if order.time_in_force == "gtc" else TimeInForce.DAY,
            "client_order_id": order.client_order_id,
        }
        if order.stop_price is not None:
            req = StopOrderRequest(stop_price=order.stop_price, **common)
        elif order.limit_price is not None:
            req = LimitOrderRequest(limit_price=order.limit_price, **common)
        else:
            req = MarketOrderRequest(**common)
        return self._status(self._trading.submit_order(order_data=req))

    def open_orders(self) -> list[OrderStatus]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self._trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        return [self._status(o) for o in orders]

    def cancel(self, client_order_id: str, wait_seconds: float = 5.0) -> OrderStatus | None:
        """Cancel and wait until the broker confirms.

        Alpaca cancels asynchronously, and shares stay reserved by the order
        until it does. Returning early would make the next sell of the same
        shares fail, so this polls until the order is closed (or gives up and
        returns the last state seen).
        """
        current = self.get_order(client_order_id)
        if current is None or not current.is_open:
            return current
        self._trading.cancel_order_by_id(current.broker_order_id)
        deadline = time.monotonic() + wait_seconds
        while True:
            current = self.get_order(client_order_id)
            if current is None or not current.is_open or time.monotonic() >= deadline:
                return current
            time.sleep(0.25)

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
            order_type=AlpacaPaperBroker._order_type(o),
            stop_price=AlpacaPaperBroker._price(o, "stop_price"),
            limit_price=AlpacaPaperBroker._price(o, "limit_price"),
            filled_at=getattr(o, "filled_at", None),
        )

    @staticmethod
    def _order_type(o) -> str:
        kind = getattr(o, "order_type", None) or getattr(o, "type", None)
        return str(getattr(kind, "value", kind) or "market")

    @staticmethod
    def _price(o, name: str) -> float | None:
        value = getattr(o, name, None)
        return float(value) if value else None


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
    entry_prices: dict[str, float] = field(default_factory=dict)

    def get_account(self) -> AccountSnapshot:
        holdings = {
            t: Holding(t, q, q * self.prices[t], self.entry_prices.get(t))
            for t, q in self.positions.items() if q
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
        if order.side is Side.SELL:
            # Like a real broker: shares reserved by an open sell cannot be sold again.
            reserved = sum(o.remaining for o in self.open_orders()
                           if o.ticker == order.ticker and o.side == Side.SELL.value)
            available = self.positions.get(order.ticker, 0.0) - reserved
            if order.quantity > available + 1e-9:
                raise RuntimeError(f"insufficient qty available for {order.ticker}: "
                                   f"requested {order.quantity:g}, available {available:g}")
        is_stop = order.stop_price is not None
        status = OrderStatus(order.client_order_id, f"fake-{len(self.orders) + 1}", order.ticker,
                             order.side.value, order.quantity, "accepted",
                             order_type="stop" if is_stop else ("limit" if order.limit_price else "market"),
                             stop_price=order.stop_price, limit_price=order.limit_price)
        self.orders[order.client_order_id] = status
        if self.fill_immediately and not is_stop:
            self.fill(order.client_order_id)
        return status

    def open_orders(self) -> list[OrderStatus]:
        return [o for o in self.orders.values() if o.is_open]

    def cancel(self, client_order_id: str) -> OrderStatus | None:
        status = self.orders.get(client_order_id)
        if status is not None and status.is_open:
            status.status = "canceled"
        return status

    # -- test helpers: make the fake market move ------------------------------

    def fill(self, client_order_id: str, quantity: float | None = None, price: float | None = None) -> OrderStatus:
        """Fill an open order, fully or (with ``quantity``) partially."""
        status = self.orders[client_order_id]
        qty = min(quantity if quantity is not None else status.remaining, status.remaining)
        price = price if price is not None else self.prices[status.ticker]
        sign = 1 if status.side == Side.BUY.value else -1
        held = self.positions.get(status.ticker, 0.0)
        if sign > 0:
            cost = self.entry_prices.get(status.ticker, price) * held + price * qty
            self.entry_prices[status.ticker] = cost / (held + qty)
        self.positions[status.ticker] = held + sign * qty
        self.cash -= sign * qty * price
        status.filled_quantity += qty
        status.filled_avg_price = price
        status.status = "filled" if status.remaining <= 1e-9 else "partially_filled"
        if status.status == "filled":
            status.filled_at = datetime.now(UTC)
        return status

    def set_price(self, ticker: str, price: float) -> None:
        """Move the price; resting stop sells at or above it trigger."""
        self.prices[ticker] = price
        for o in self.open_orders():
            if o.ticker == ticker and o.is_stop and o.stop_price is not None and price <= o.stop_price:
                self.fill(o.client_order_id, price=price)
