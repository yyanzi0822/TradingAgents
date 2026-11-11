"""Protective stops and broker reconciliation, end to end on the fake broker."""

import pytest

from trading_engine.broker import FakeBroker
from trading_engine.config import RiskLimits
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.models import AccountSnapshot, Quote, TradeIntent
from trading_engine.reconcile import Reconciler, _market_date
from trading_engine.risk import RiskEngine
from trading_engine.runner import Runner
from trading_engine.stops import StopBook

SECTORS = {"AAPL": "Information Technology", "MSFT": "Information Technology", "JPM": "Financials"}
TODAY = _market_date(None)


def plan(stop):
    return f"**Action**: Buy\n\n**Stop Loss**: {stop:.2f}\n\nFINAL TRANSACTION PROPOSAL: **BUY**"


def stub(signals, stop=95.0):
    calls = []

    def analyze(ticker, trade_date, portfolio):
        calls.append(ticker)
        return {"trader_investment_plan": plan(stop), "final_trade_decision": "thesis"}, signals[ticker]

    analyze.calls = calls
    return analyze


@pytest.fixture
def env(tmp_path):
    broker = FakeBroker(prices={"AAPL": 100.0, "MSFT": 100.0, "JPM": 200.0})
    return broker, Ledger(tmp_path / "ledger.jsonl"), KillSwitch(tmp_path / "KILL")


def runner(env, analyze, **kw):
    broker, ledger, ks = env
    return Runner(broker, analyze, ledger, sectors=SECTORS, kill_switch=ks,
                  portfolio_builder=lambda a: None, **kw)


def reconciler(env, **kw):
    broker, ledger, ks = env
    return Reconciler(broker, ledger, kill_switch=ks, **kw)


def stops_at(broker, ticker):
    return [(o.quantity, o.stop_price) for o in broker.open_orders() if o.ticker == ticker and o.is_stop]


# --- stops are placed and kept in step with the position ----------------------


def test_filled_buy_gets_a_resting_stop_at_the_price_it_was_sized_with(env):
    broker, ledger, _ = env
    # 0.5% of 100k = $500 risk; $5 stop distance -> 100 shares; the 5% Buy target makes it 50.
    report = runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-27")

    assert broker.positions["AAPL"] == 50
    assert stops_at(broker, "AAPL") == [(50.0, 95.0)]
    assert ledger.records("stop_order")[0]["data"]["stop_price"] == 95.0
    assert any("placed stop 50 @ 95.00" in n for n in report.notes)


def test_running_again_does_not_stack_a_second_stop(env):
    broker, ledger, _ = env
    runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-27")
    assert reconciler(env).run() == []
    assert reconciler(env).run() == []
    assert len(stops_at(broker, "AAPL")) == 1 and len(ledger.records("stop_order")) == 1


def test_stop_out_is_recorded_and_the_ticker_is_not_rebought_that_day(env):
    broker, ledger, _ = env
    runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-26")
    broker.set_price("AAPL", 94.0)                      # the resting stop triggers at the broker
    assert broker.positions["AAPL"] == 0

    analyze = stub({"AAPL": "Buy"})
    report = runner(env, analyze).run_cycle(["AAPL"], TODAY)

    assert report.outcomes[0].detail == "stopped out today" and not analyze.calls
    assert ledger.records("stop_filled")[0]["data"]["quantity"] == 50
    assert any("stop filled 50 @ 94.00" in n for n in report.notes)
    assert StopBook(ledger).current() == {} and not broker.open_orders()


def test_sell_cancels_the_stop_first_and_leaves_none_behind(env):
    broker, ledger, _ = env
    runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-26")
    report = runner(env, stub({"AAPL": "Sell"})).run_cycle(["AAPL"], "2026-10-27")

    assert report.outcomes[0].action == "submitted"      # the fake broker rejects selling reserved shares
    assert broker.positions["AAPL"] == 0 and not broker.open_orders()
    assert StopBook(ledger).current() == {}


def test_partial_sell_resizes_the_stop_to_what_is_left(env):
    broker, _, _ = env
    runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-26")
    runner(env, stub({"AAPL": "Underweight"})).run_cycle(["AAPL"], "2026-10-27")
    assert broker.positions["AAPL"] == 25
    assert stops_at(broker, "AAPL") == [(25.0, 95.0)]


def test_stop_can_be_raised_but_never_lowered(tmp_path):
    book = StopBook(Ledger(tmp_path / "ledger.jsonl"))
    assert book.set("AAPL", 95.0, source="t") == 95.0
    assert book.set("AAPL", 90.0, source="t") == 95.0     # the agents cannot loosen it
    assert book.set("AAPL", 97.0, source="t") == 97.0
    book.clear("AAPL", "closed")
    assert book.current() == {}
    assert book.set("AAPL", 90.0, source="t") == 90.0     # a new position starts fresh


def test_existing_position_without_a_recorded_stop_gets_the_default(env):
    broker, ledger, _ = env
    broker.positions["JPM"], broker.entry_prices["JPM"] = 10, 200.0
    reconciler(env).run()
    assert stops_at(broker, "JPM") == [(10.0, 184.0)]      # 8% below entry
    assert StopBook(ledger).current() == {"JPM": 184.0}


def test_position_with_no_entry_price_is_reported_not_guessed(env):
    broker, _, _ = env
    broker.positions["JPM"] = 10
    notes = reconciler(env).run()
    assert notes == ["JPM: no stop on record and no entry price; not protected"]
    assert not broker.open_orders()


# --- price already through the stop -------------------------------------------


def breached(env):
    broker, _, _ = env
    broker.positions["JPM"], broker.entry_prices["JPM"] = 10, 200.0
    broker.prices["JPM"] = 180.0                           # default stop is 184


def test_price_through_the_stop_sells_at_market(env):
    broker, ledger, _ = env
    breached(env)
    notes = reconciler(env).run(TODAY)
    assert broker.positions["JPM"] == 0
    assert "through stop 184.00; sold 10 at market" in notes[0]
    assert ledger.records("order")[0]["data"]["decision_id"].endswith("|stop-exit")
    assert reconciler(env).run(TODAY) == [] or broker.positions["JPM"] == 0


def test_breached_stop_waits_when_the_market_is_closed(env):
    broker, _, _ = env
    breached(env)
    broker.market_open = False
    notes = reconciler(env).run()
    assert "market closed" in notes[0] and broker.positions["JPM"] == 10 and not broker.orders


def test_breached_stop_exit_can_be_declined_with_confirm(env):
    broker, ledger, _ = env
    breached(env)
    asked = []
    notes = reconciler(env, confirm=lambda q: asked.append(q) or False).run()
    assert asked and "declined" in notes[0]
    assert broker.positions["JPM"] == 10 and ledger.records("declined")


# --- safety rails --------------------------------------------------------------


def test_kill_switch_stops_new_stop_orders_but_leaves_resting_ones(env):
    broker, _, ks = env
    runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-27")
    broker.positions["JPM"], broker.entry_prices["JPM"] = 10, 200.0
    ks.engage("test")
    assert reconciler(env).sync_stops() == ["kill switch active: stop orders left as they are"]
    assert stops_at(broker, "AAPL") == [(50.0, 95.0)] and not stops_at(broker, "JPM")


def test_dry_run_reports_and_sends_nothing(env):
    broker, ledger, _ = env
    broker.positions["JPM"], broker.entry_prices["JPM"] = 10, 200.0
    notes = reconciler(env, dry_run=True).run()
    assert notes == ["JPM: would place stop 10 @ 184.00"]
    assert not broker.orders and not ledger.records("stop") and not ledger.records("stop_order")


def test_enforce_stops_off_places_nothing(env):
    broker, ledger, ks = env
    Runner(broker, stub({"AAPL": "Buy"}), ledger, sectors=SECTORS, kill_switch=ks,
           limits=RiskLimits(enforce_stops=False), portfolio_builder=lambda a: None).run_cycle(["AAPL"], "2026-10-27")
    assert broker.positions["AAPL"] == 50 and not broker.open_orders()


def test_cycle_does_not_trade_if_it_cannot_reconcile(env):
    broker, _, ks = env

    def down():
        raise ConnectionError("broker unreachable")

    broker.open_orders = down
    broker.positions["JPM"], broker.entry_prices["JPM"] = 10, 200.0
    analyze = stub({"AAPL": "Buy"})
    report = runner(env, analyze).run_cycle(["AAPL"], "2026-10-27")
    assert report.halted and not analyze.calls and not broker.orders
    assert not ks.active()                                  # a network blip is retried next run


# --- unfilled and partly filled orders -----------------------------------------


def test_unfilled_buy_blocks_a_second_order_and_gets_its_stop_once_filled(env):
    broker, ledger, _ = env
    broker.fill_immediately = False
    runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-26")
    assert not stops_at(broker, "AAPL")                     # nothing to protect yet
    assert StopBook(ledger).current() == {"AAPL": 95.0}     # but the stop price is kept

    analyze = stub({"AAPL": "Buy"})
    report = runner(env, analyze).run_cycle(["AAPL"], "2026-10-27")
    assert "order still open at broker" in report.outcomes[0].detail and not analyze.calls

    buy_id = ledger.records("order")[0]["data"]["client_order_id"]
    broker.fill(buy_id)
    notes = reconciler(env).run()
    assert "AAPL: buy filled 50 @ 100.00" in notes
    assert stops_at(broker, "AAPL") == [(50.0, 95.0)]


def test_partial_fill_then_expiry_is_recorded_and_the_filled_part_protected(env):
    broker, ledger, _ = env
    broker.fill_immediately = False
    runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-26")
    buy_id = ledger.records("order")[0]["data"]["client_order_id"]
    broker.fill(buy_id, quantity=20)
    broker.orders[buy_id].status = "expired"                # day order ran out at the close

    notes = reconciler(env).run()
    assert "AAPL: buy expired after partial fill 20 of 50 @ 100.00" in notes
    assert stops_at(broker, "AAPL") == [(20.0, 95.0)]
    assert reconciler(env).run() == []                      # terminal: not reported twice


def test_unfilled_buys_count_against_sector_gross_and_cash_caps(env):
    broker, _, _ = env
    broker.fill_immediately = False
    # Four tech names would each get 5%; the sector cap is 20% and MSFT is the
    # only other tech ticker here, so use the risk engine directly for the cap.
    report = runner(env, stub({"AAPL": "Buy", "MSFT": "Buy"})).run_cycle(["AAPL", "MSFT"], "2026-10-27")
    assert [o.action for o in report.outcomes] == ["submitted", "submitted"]

    engine = RiskEngine(RiskLimits(max_sector_weight=0.20), SECTORS)
    acct = AccountSnapshot(equity=100_000, cash=100_000, pending_buys={"AAPL": 5_000, "NVDA": 14_500})
    engine.sectors["NVDA"] = "Information Technology"
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    result = engine.evaluate(TradeIntent("d", "MSFT", "Buy", 0.05), acct, Quote("MSFT", 100.0, now), now=now)
    assert result.order.quantity == 5                       # $500 of sector room left, not $5,000
    assert any("sector cap" in a for a in result.adjustments)

    acct = AccountSnapshot(equity=100_000, cash=3_000, pending_buys={"JPM": 2_000})
    result = engine.evaluate(TradeIntent("d", "MSFT", "Buy", 0.05), acct, Quote("MSFT", 100.0, now), now=now)
    assert result.order.quantity == 10                      # cash already promised to the open order

    acct = AccountSnapshot(equity=100_000, cash=100_000, pending_buys={"MSFT": 4_000})
    result = engine.evaluate(TradeIntent("d", "MSFT", "Buy", 0.05), acct, Quote("MSFT", 100.0, now), now=now)
    assert result.order.quantity == 10                      # same ticker: only the remaining 1%


# --- confirm --------------------------------------------------------------------


def test_confirm_shows_the_order_and_a_no_sends_nothing(env):
    broker, ledger, _ = env
    asked = []
    report = runner(env, stub({"AAPL": "Buy"}), confirm=lambda q: asked.append(q) or False).run_cycle(
        ["AAPL"], "2026-10-27")
    assert asked == ["AAPL (Buy): buy 50 @ ~100.00"]
    assert report.outcomes[0].action == "declined" and not broker.orders
    assert ledger.records("declined") and ledger.records("decision")


def test_confirm_yes_sends_the_order(env):
    broker, _, _ = env
    runner(env, stub({"AAPL": "Buy"}), confirm=lambda q: True).run_cycle(["AAPL"], "2026-10-27")
    assert broker.positions["AAPL"] == 50 and stops_at(broker, "AAPL") == [(50.0, 95.0)]
