"""AlpacaPaperBroker against the real alpaca-py request/enum types, with the
network clients stubbed out (no keys or connectivity needed)."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("alpaca")

import alpaca.data.historical as data_mod  # noqa: E402
import alpaca.trading.client as trading_mod  # noqa: E402
from alpaca.common.exceptions import APIError  # noqa: E402
from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: E402
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest  # noqa: E402

from trading_engine.broker import AlpacaPaperBroker, LiveTradingRefused  # noqa: E402
from trading_engine.models import OrderRequest, Side  # noqa: E402


def not_found():
    return APIError('{"code": 40410000, "message": "order not found"}',
                    SimpleNamespace(response=SimpleNamespace(status_code=404), request=None))


class StubTrading:
    instances = []

    def __init__(self, key, secret, paper=False, **_):
        self.paper = paper
        self.account_number = "PA3XYZ"
        self.orders = {}
        self.submitted = []
        StubTrading.instances.append(self)

    def get_account(self):
        return SimpleNamespace(account_number=self.account_number, equity="100000", cash="60000")

    def get_all_positions(self):
        return [SimpleNamespace(symbol="AAPL", qty="100", market_value="20000", avg_entry_price="180.5"),
                SimpleNamespace(symbol="JPM", qty="80", market_value="20000", avg_entry_price=None)]

    def get_clock(self):
        return SimpleNamespace(is_open=True)

    def get_order_by_client_id(self, client_id):
        if client_id not in self.orders:
            raise not_found()
        return self.orders[client_id]

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        o = SimpleNamespace(id=f"alp-{len(self.submitted)}", client_order_id=order_data.client_order_id,
                            symbol=order_data.symbol, side=order_data.side, qty=order_data.qty,
                            status=SimpleNamespace(value="accepted"), filled_qty="0", filled_avg_price=None)
        self.orders[order_data.client_order_id] = o
        return o


class StubData:
    def __init__(self, key, secret, **_):
        pass

    def get_stock_latest_trade(self, req):
        sym = req.symbol_or_symbols
        return {sym: SimpleNamespace(price=201.25, timestamp=datetime(2026, 10, 27, 15, 0))}


@pytest.fixture
def broker(monkeypatch):
    StubTrading.instances.clear()
    monkeypatch.setattr(trading_mod, "TradingClient", StubTrading)
    monkeypatch.setattr(data_mod, "StockHistoricalDataClient", StubData)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    return AlpacaPaperBroker()


def test_always_constructs_paper_client(broker):
    assert StubTrading.instances[0].paper is True


def test_refuses_live_base_url(monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(LiveTradingRefused):
        AlpacaPaperBroker("k", "s")


def test_refuses_non_paper_account(monkeypatch):
    class LiveAccount(StubTrading):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.account_number = "3XYZ123"

    monkeypatch.setattr(trading_mod, "TradingClient", LiveAccount)
    monkeypatch.setattr(data_mod, "StockHistoricalDataClient", StubData)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    with pytest.raises(LiveTradingRefused):
        AlpacaPaperBroker("k", "s")


def test_missing_keys(monkeypatch):
    monkeypatch.setattr(trading_mod, "TradingClient", StubTrading)
    monkeypatch.setattr(data_mod, "StockHistoricalDataClient", StubData)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        AlpacaPaperBroker()


def test_account_mapping(broker):
    acct = broker.get_account()
    assert acct.equity == 100_000 and acct.cash == 60_000
    assert acct.holdings["AAPL"].quantity == 100 and acct.holdings["AAPL"].avg_entry_price == 180.5
    assert acct.holdings["JPM"].avg_entry_price is None
    assert acct.gross_exposure == 40_000


def test_quote_is_timezone_aware(broker):
    q = broker.get_quote("aapl")
    assert q.ticker == "AAPL" and q.price == 201.25 and q.timestamp.tzinfo is UTC


def test_market_order_built_with_real_alpaca_types(broker):
    status = broker.submit(OrderRequest("ta-abc", "AAPL", Side.BUY, 10))
    req = StubTrading.instances[0].submitted[0]
    assert isinstance(req, MarketOrderRequest)
    assert req.side == OrderSide.BUY and req.time_in_force == TimeInForce.DAY
    assert req.client_order_id == "ta-abc" and req.qty == 10
    assert status.status == "accepted" and not status.duplicate


def test_limit_order(broker):
    broker.submit(OrderRequest("ta-lim", "AAPL", Side.SELL, 5, limit_price=205.0))
    req = StubTrading.instances[0].submitted[0]
    assert isinstance(req, LimitOrderRequest) and req.limit_price == 205.0 and req.side == OrderSide.SELL


def test_resubmit_is_idempotent(broker):
    o = OrderRequest("ta-dup", "AAPL", Side.BUY, 10)
    first = broker.submit(o)
    second = broker.submit(o)
    assert len(StubTrading.instances[0].submitted) == 1
    assert second.duplicate and not first.duplicate
    assert second.broker_order_id == first.broker_order_id


def test_unknown_order_returns_none(broker):
    assert broker.get_order("ta-missing") is None


def test_other_api_errors_propagate(broker):
    def boom(client_id):
        raise APIError('{"code": 50010000, "message": "internal"}',
                       SimpleNamespace(response=SimpleNamespace(status_code=500), request=None))

    StubTrading.instances[0].get_order_by_client_id = boom
    with pytest.raises(APIError):
        broker.submit(OrderRequest("ta-x", "AAPL", Side.BUY, 1))
