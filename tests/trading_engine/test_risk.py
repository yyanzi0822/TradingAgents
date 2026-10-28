from datetime import UTC, datetime, timedelta

import pytest

from trading_engine.config import RiskLimits
from trading_engine.models import AccountSnapshot, Holding, Quote, Side, TradeIntent
from trading_engine.risk import RiskEngine, client_order_id

NOW = datetime(2026, 10, 27, 15, 0, tzinfo=UTC)
SECTORS = {"AAPL": "Information Technology", "MSFT": "Information Technology",
           "NVDA": "Information Technology", "JPM": "Financials", "XOM": "Energy"}


def engine(**overrides):
    return RiskEngine(RiskLimits(**overrides), SECTORS)


def account(equity=100_000.0, cash=100_000.0, holdings=None, hwm=None):
    return AccountSnapshot(equity=equity, cash=cash, holdings=holdings or {}, high_water_mark=hwm)


def quote(ticker="AAPL", price=100.0, age_s=0):
    return Quote(ticker, price, NOW - timedelta(seconds=age_s))


def intent(ticker="AAPL", weight=0.05, stop=None, rating="Buy"):
    return TradeIntent("d1", ticker, rating, weight, stop_loss=stop)


def test_default_stop_8pct_leaves_single_stock_cap_binding():
    # $500 risk / $8 per share = 62 shares ($6,250) > 5% cap ($5,000) -> 50.
    r = engine().evaluate(intent(), account(), quote(), now=NOW)
    assert r.approved and r.order.side is Side.BUY
    assert r.order.quantity == 50


def test_buy_capped_by_risk_per_trade_with_wide_stop():
    # 0.5% of 100k = $500 risk; $25 stop distance -> 20 shares ($2,000 < 5% cap).
    r = engine().evaluate(intent(stop=75.0), account(), quote(), now=NOW)
    assert r.order.quantity == 20
    assert any("risk per trade" in a for a in r.adjustments)


def test_tight_stop_allows_full_single_stock_cap():
    # $1 stop distance -> 500 shares allowed by risk; 5% cap = $5,000 = 50 shares.
    r = engine().evaluate(intent(stop=99.0), account(), quote(), now=NOW)
    assert r.order.quantity == 50


def test_single_stock_cap_counts_existing_position():
    held = {"AAPL": Holding("AAPL", 40, 4_000.0)}
    r = engine().evaluate(intent(weight=0.10, stop=99.0), account(cash=96_000, holdings=held), quote(), now=NOW)
    assert r.order.quantity == 10  # only $1,000 of room left under 5%


def test_sector_cap():
    held = {"MSFT": Holding("MSFT", 100, 10_000.0), "NVDA": Holding("NVDA", 95, 9_500.0)}
    r = engine().evaluate(intent(stop=99.0), account(cash=80_500, holdings=held), quote(), now=NOW)
    # 20% sector = $20,000; $19,500 used -> $500 room -> 5 shares.
    assert r.order.quantity == 5
    assert any("sector cap" in a for a in r.adjustments)


def test_gross_exposure_cap():
    held = {"JPM": Holding("JPM", 1, 79_900.0)}
    r = engine().evaluate(intent(stop=99.0), account(cash=20_100, holdings=held), quote(), now=NOW)
    assert r.order.quantity == 1  # $100 of room under 80%


def test_cash_cap():
    r = engine().evaluate(intent(stop=99.0), account(cash=2_000), quote(), now=NOW)
    assert r.order.quantity == 20


def test_drawdown_breaker_blocks_buys_but_not_sells():
    acct = account(equity=91_000, cash=86_000, hwm=100_000,
                   holdings={"AAPL": Holding("AAPL", 50, 5_000.0)})
    new_buy = engine().evaluate(intent("JPM", 0.05, stop=99.0), acct, quote("JPM"), now=NOW)
    assert not new_buy.approved and "drawdown" in new_buy.reasons[0]
    sell = engine().evaluate(intent("AAPL", 0.0, rating="Sell"), acct, quote(), now=NOW)
    assert sell.approved and sell.order.side is Side.SELL and sell.order.quantity == 50


def test_kill_switch_rejects_everything():
    r = engine().evaluate(intent(), account(), quote(), kill_switch_active=True, now=NOW)
    assert not r.approved and r.order is None


def test_stale_quote_rejected():
    r = engine().evaluate(intent(), account(), quote(age_s=3600), now=NOW)
    assert not r.approved and "stale" in r.reasons[0]


@pytest.mark.parametrize("price", [0.0, -5.0, float("nan")])
def test_bad_price_rejected(price):
    r = engine().evaluate(intent(), account(), quote(price=price), now=NOW)
    assert not r.approved


def test_long_only_rejects_negative_target():
    r = engine().evaluate(intent(weight=-0.05), account(), quote(), now=NOW)
    assert not r.approved


def test_never_sells_more_than_held():
    held = {"AAPL": Holding("AAPL", 30, 3_000.0)}
    r = engine().evaluate(intent(weight=0.0, rating="Sell"), account(holdings=held, cash=97_000), quote(), now=NOW)
    assert r.order.quantity == 30


def test_sell_with_nothing_held_is_noop():
    r = engine().evaluate(intent(weight=0.0, rating="Sell"), account(), quote(), now=NOW)
    assert r.approved and r.order is None


def test_hold_is_noop():
    r = engine().evaluate(intent(weight=None, rating="Hold"), account(), quote(), now=NOW)
    assert r.approved and r.order is None


def test_unknown_sector_rejected():
    r = engine().evaluate(intent("ZZZZ", stop=99.0), account(), quote("ZZZZ"), now=NOW)
    assert not r.approved and "sector" in r.reasons[0]


def test_stop_above_price_ignored():
    r = engine(default_stop_pct=0.25).evaluate(intent(stop=150.0), account(), quote(), now=NOW)
    assert r.order.quantity == 20  # sized off the 25% default stop, not the bogus one
    assert any("ignored stop" in a for a in r.adjustments)


def test_client_order_id_is_deterministic():
    a = client_order_id("d1", "aapl", Side.BUY, 10)
    assert a == client_order_id("d1", "AAPL", Side.BUY, 10.0)
    assert a != client_order_id("d2", "AAPL", Side.BUY, 10)
    assert len(a) <= 48  # Alpaca's client_order_id limit
