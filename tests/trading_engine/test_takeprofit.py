"""Take profit in steps: a third at +1.5%, a third at +2.5%, the rest runs."""

import pytest

from trading_engine.broker import FakeBroker
from trading_engine.config import RiskLimits
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.reconcile import Reconciler
from trading_engine.runner import Runner
from trading_engine.takeprofit import TakeProfitBook

SECTORS = {"AAPL": "Information Technology", "JPM": "Financials"}


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


def cycle(env, signal, day, **kw):
    broker, ledger, ks, inbox = env
    return Runner(broker, stub({"AAPL": signal}), ledger, sectors=SECTORS, kill_switch=ks, notifier=inbox,
                  portfolio_builder=lambda a: None, **kw).run_cycle(["AAPL"], day).outcomes[0]


def reconcile(env, day="2026-10-28", **kw):
    broker, ledger, ks, inbox = env
    return Reconciler(broker, ledger, kill_switch=ks, notifier=inbox, **kw).run(day)


def stop_orders(broker):
    return [(o.quantity, o.stop_price) for o in broker.open_orders() if o.is_stop]


@pytest.fixture
def held(env):
    cycle(env, "Buy", "2026-10-26")                           # 50 shares at 100, stop 95
    env[3].messages.clear()
    return env


def test_first_level_sells_a_third_and_resizes_the_stop(held):
    broker, ledger, _, inbox = held
    broker.prices["AAPL"] = 101.5
    notes = reconcile(held)

    assert broker.positions["AAPL"] == 34                     # 16 of 50 sold
    assert "AAPL: up +1.5% (take-profit at +1.5%); sold 16 of 50 at market -> filled" in notes
    assert stop_orders(broker) == [(34.0, 95.0)]
    assert inbox.messages == ["💰 TAKE PROFIT AAPL: up +1.5% from 100.00, sold 16 of 50 @ ~101.50 (filled)"]
    assert ledger.records("order")[-1]["data"]["reason"] == "take profit at +1.5%"


def test_nothing_below_the_first_level(held):
    broker = held[0]
    broker.prices["AAPL"] = 101.4
    assert reconcile(held) == [] and broker.positions["AAPL"] == 50


def test_each_level_is_taken_once(held):
    broker = held[0]
    broker.prices["AAPL"] = 101.6
    reconcile(held)
    assert reconcile(held, "2026-10-29") == []                # still +1.6%: nothing more
    assert broker.positions["AAPL"] == 34


def test_second_level_sells_another_third_of_the_original(held):
    broker = held[0]
    broker.prices["AAPL"] = 101.5
    reconcile(held)
    broker.prices["AAPL"] = 102.5
    reconcile(held, "2026-10-29")
    assert broker.positions["AAPL"] == 18                     # 16 + 16 sold, 18 runs
    assert stop_orders(broker) == [(18.0, 95.0)]
    broker.prices["AAPL"] = 110.0
    assert reconcile(held, "2026-10-30") == []                # no third level


def test_jump_past_both_levels_sells_two_thirds_at_once(held):
    broker, ledger, _, _ = held
    broker.prices["AAPL"] = 103.0
    notes = reconcile(held)
    assert broker.positions["AAPL"] == 18
    assert "take-profit at +1.5% and +2.5%" in notes[0]
    assert TakeProfitBook(ledger).taken("AAPL") == {0, 1}


def test_buy_rating_does_not_buy_back_what_was_sold(held):
    broker = held[0]
    broker.prices["AAPL"] = 101.5
    reconcile(held, "2026-10-27")
    out = cycle(held, "Buy", "2026-10-28")
    assert out.detail == "profit already taken on this position; not adding to it"
    assert broker.positions["AAPL"] == 34


def test_levels_start_again_after_the_position_is_closed(held):
    broker, ledger, _, _ = held
    broker.prices["AAPL"] = 101.5
    reconcile(held, "2026-10-27")
    cycle(held, "Sell", "2026-10-28")
    assert broker.positions["AAPL"] == 0
    reconcile(held, "2026-10-29")
    assert TakeProfitBook(ledger).taken("AAPL") == set()

    out = cycle(held, "Buy", "2026-10-30")                    # new position at 101.5
    assert out.action == "submitted" and broker.positions["AAPL"] > 0
    shares = broker.positions["AAPL"]
    broker.prices["AAPL"] = 103.1                             # +1.6% on the new entry
    reconcile(held, "2026-11-02")
    assert broker.positions["AAPL"] == shares - shares // 3


def test_dry_run_market_closed_kill_switch_and_decline_send_nothing(held):
    broker, ledger, ks, _ = held
    broker.prices["AAPL"] = 101.5
    assert reconcile(held, dry_run=True) == ["AAPL: up +1.5% (take-profit at +1.5%); would sell 16 of 50"]

    broker.market_open = False
    assert "market closed" in reconcile(held)[0]
    broker.market_open = True

    asked = []
    notes = reconcile(held, confirm=lambda q: asked.append(q) or False)
    assert "Sell 16 of 50 at market?" in asked[0] and "declined by you" in notes[0]

    ks.engage("test")
    assert reconcile(held) == ["kill switch active: stop orders left as they are"]
    assert broker.positions["AAPL"] == 50 and not ledger.records("take_profit")


def test_no_levels_means_no_take_profit(held):
    broker, ledger, ks, _ = held
    broker.prices["AAPL"] = 110.0
    Reconciler(broker, ledger, RiskLimits(take_profit_levels=()), ks).run("2026-10-28")
    assert broker.positions["AAPL"] == 50


def test_position_too_small_to_split_is_reported(env):
    broker = env[0]
    broker.positions["JPM"], broker.entry_prices["JPM"] = 2, 200.0
    broker.prices["JPM"] = 204.0
    notes = reconcile(env)
    assert "JPM: up +2.0% (take-profit at +1.5%); position too small to sell a part of" in notes
    assert broker.positions["JPM"] == 2


def test_waits_while_another_order_for_the_ticker_is_open(held):
    broker = held[0]
    broker.fill_immediately = False
    broker.prices["AAPL"] = 101.5
    reconcile(held)                                           # take-profit sell accepted, not filled
    assert broker.positions["AAPL"] == 50
    broker.prices["AAPL"] = 102.6
    notes = reconcile(held, "2026-10-29")
    assert any("take-profit waits for the open order" in n for n in notes)
    assert len([o for o in broker.orders.values() if not o.is_stop and o.side == "sell"]) == 1
