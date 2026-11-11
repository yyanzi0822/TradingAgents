"""Wash-sale detection: buying back after a loss sale, and partial loss sales."""

from datetime import date, timedelta

import pytest

from trading_engine.broker import FakeBroker
from trading_engine.config import RiskLimits
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.reconcile import Reconciler, _market_date
from trading_engine.runner import Runner
from trading_engine.summary import build_summary
from trading_engine.wash import WashSaleChecker, trades

SECTORS = {"AAPL": "Information Technology", "JPM": "Financials"}
D0 = date(2026, 9, 1)


def day(n):
    return (D0 + timedelta(days=n)).isoformat()


class Inbox:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)
        return True


def stub(signals):
    def analyze(ticker, trade_date, portfolio):
        return {"trader_investment_plan": "**Stop Loss**: 95.00"}, signals[ticker]
    return analyze


@pytest.fixture
def env(tmp_path):
    return (FakeBroker(prices={"AAPL": 100.0, "JPM": 200.0}), Ledger(tmp_path / "ledger.jsonl"),
            KillSwitch(tmp_path / "KILL"), Inbox())


def cycle(env, signal, n, policy="warn", **kw):
    broker, ledger, ks, inbox = env
    runner = Runner(broker, stub({"AAPL": signal}), ledger, sectors=SECTORS, kill_switch=ks, notifier=inbox,
                    limits=RiskLimits(wash_sale_policy=policy, take_profit_levels=()),
                    portfolio_builder=lambda a: None, **kw)
    return runner.run_cycle(["AAPL"], day(n)).outcomes[0]


def sell_at(env, price, n, signal="Sell"):
    env[0].prices["AAPL"] = price
    return cycle(env, signal, n)


# --- buying back after a loss ----------------------------------------------------


def test_rebuy_within_30_days_of_a_loss_sale_is_flagged_but_still_sent(env):
    broker, ledger, _, inbox = env
    cycle(env, "Buy", 0)
    sell_at(env, 97.0, 3)                                   # bought at 100, sold at 97
    out = cycle(env, "Buy", 10)

    assert out.action == "submitted" and broker.positions["AAPL"] > 0
    assert f"possible wash sale: AAPL was sold at a loss on {day(3)}" in out.detail
    assert f"a buy before {day(34)}" in out.detail            # 30 days after the sale, plus one
    flagged = ledger.records("wash_sale")
    assert len(flagged) == 1 and flagged[0]["data"]["blocked"] is False
    assert "⚠️ possible wash sale" in inbox.messages[-1]


def test_block_policy_refuses_the_buy(env):
    broker, ledger, _, _ = env
    cycle(env, "Buy", 0)
    sell_at(env, 97.0, 3)
    out = cycle(env, "Buy", 10, policy="block")

    assert out.action == "rejected" and out.detail.startswith("buy blocked: possible wash sale")
    assert broker.positions["AAPL"] == 0
    assert ledger.records("wash_sale")[0]["data"]["blocked"] is True


def test_window_is_30_days_inclusive_and_clear_on_day_31(env):
    cycle(env, "Buy", 0)
    sell_at(env, 97.0, 3)
    checker = WashSaleChecker(env[1])
    assert checker.check_buy("AAPL", date.fromisoformat(day(3))) is not None    # same day
    assert checker.check_buy("AAPL", date.fromisoformat(day(33))) is not None   # day 30
    assert checker.check_buy("AAPL", date.fromisoformat(day(34))) is None       # day 31
    assert "possible wash sale" not in cycle(env, "Buy", 34).detail


def test_rebuy_after_a_sale_at_a_gain_is_not_flagged(env):
    cycle(env, "Buy", 0)
    sell_at(env, 120.0, 3)
    out = cycle(env, "Buy", 10)
    assert out.action == "submitted" and "wash" not in out.detail
    assert not env[1].records("wash_sale")


def test_stop_out_opens_a_window_and_says_so(env):
    broker, ledger, ks, inbox = env
    cycle(env, "Buy", 0)
    broker.set_price("AAPL", 94.0)                           # resting stop fills at the broker
    notes = Reconciler(broker, ledger, kill_switch=ks, notifier=inbox).run()

    today = date.fromisoformat(_market_date(None))
    clear = today + timedelta(days=31)
    assert f"AAPL: sold at a loss; a rebuy before {clear} is a possible wash sale" in notes
    assert f"a rebuy before {clear}" in inbox.messages[-1]
    assert WashSaleChecker(ledger).windows(today) == {"AAPL": (today, clear)}
    assert "Wash-sale windows" in build_summary(broker, ledger, ks)
    assert f"AAPL: sold at a loss {today}, clear from {clear}" in build_summary(broker, ledger, ks)


def test_market_exit_on_a_breached_stop_is_counted_once(env):
    broker, ledger, ks, _ = env
    cycle(env, "Buy", 0)
    for o in list(broker.open_orders()):
        broker.cancel(o.client_order_id)                     # stop order lost
    broker.prices["AAPL"] = 90.0
    Reconciler(broker, ledger, kill_switch=ks).run(day(2))   # sells at market
    sells = [t for t in trades(ledger) if t.side == "sell"]
    assert len(sells) == 1 and sells[0].is_loss is True and sells[0].cost_basis == 100.0


# --- selling part of a position at a loss -----------------------------------------


def test_partial_sale_at_a_loss_soon_after_buying_is_flagged_and_still_sent(env):
    broker, ledger, _, _ = env
    cycle(env, "Buy", 0)
    out = sell_at(env, 97.0, 5, signal="Underweight")
    assert out.action == "submitted" and broker.positions["AAPL"] == 25
    assert f"selling part of AAPL at a loss within 30 days of buying it on {day(0)}" in out.detail
    assert ledger.records("wash_sale")[0]["data"]["side"] == "sell"


def test_sells_are_never_blocked(env):
    broker, _, _, _ = env
    cycle(env, "Buy", 0)
    env[0].prices["AAPL"] = 97.0
    out = cycle(env, "Underweight", 5, policy="block")
    assert out.action == "submitted" and broker.positions["AAPL"] == 25


def test_full_exit_at_a_loss_is_not_itself_flagged(env):
    cycle(env, "Buy", 0)
    out = sell_at(env, 97.0, 5)
    assert out.action == "submitted" and "wash" not in out.detail


def test_partial_sale_at_a_gain_or_long_after_buying_is_not_flagged(env):
    cycle(env, "Buy", 0)
    assert "wash" not in sell_at(env, 110.0, 5, signal="Underweight").detail
    checker = WashSaleChecker(env[1])
    assert checker.check_sell("AAPL", date.fromisoformat(day(40)), price=97.0, cost_basis=100.0,
                              quantity=10, held=25) is None


# --- edges --------------------------------------------------------------------------


def test_policy_off_checks_nothing(env):
    cycle(env, "Buy", 0)
    sell_at(env, 97.0, 3)
    out = cycle(env, "Buy", 10, policy="off")
    assert "wash" not in out.detail and not env[1].records("wash_sale")
    assert "Wash-sale windows" not in build_summary(env[0], env[1], env[2], RiskLimits(wash_sale_policy="off"),
                                                    day=day(10))


def test_a_sell_that_never_filled_opens_no_window(env):
    broker, ledger, ks, _ = env
    cycle(env, "Buy", 0)
    broker.fill_immediately = False
    sell_at(env, 97.0, 3)                                   # accepted, not filled
    oid = ledger.records("order")[-1]["data"]["client_order_id"]
    broker.orders[oid].status = "expired"
    Reconciler(broker, ledger, kill_switch=ks).run(day(3))
    assert WashSaleChecker(ledger).check_buy("AAPL", date.fromisoformat(day(10))) is None


def test_sale_with_no_cost_basis_on_record_is_treated_as_possible(env):
    broker, ledger, _, _ = env
    broker.positions["AAPL"] = 10                            # held before the engine, no entry price
    sell_at(env, 97.0, 3)
    note = WashSaleChecker(ledger).check_buy("AAPL", date.fromisoformat(day(10)))
    assert note and "gain or loss not on record" in note


def test_other_tickers_are_unaffected(env):
    cycle(env, "Buy", 0)
    sell_at(env, 97.0, 3)
    assert WashSaleChecker(env[1]).check_buy("JPM", date.fromisoformat(day(10))) is None
