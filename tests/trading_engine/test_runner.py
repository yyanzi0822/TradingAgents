import json

import pytest

from trading_engine.broker import FakeBroker
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.models import AccountSnapshot, OrderRequest, Side
from trading_engine.runner import Runner
from trading_engine.sizing import intent_from_decision, target_weight_for

SECTORS = {"AAPL": "Information Technology", "JPM": "Financials", "XOM": "Energy"}
TRADER_PLAN = "**Action**: Buy\n\n**Entry Price**: 100\n\n**Stop Loss**: 99.00\n\nFINAL TRANSACTION PROPOSAL: **BUY**"


def stub(signals: dict[str, str], fail: set[str] = frozenset()):
    calls = []

    def analyze(ticker, trade_date, portfolio):
        calls.append(ticker)
        if ticker in fail:
            raise RuntimeError("LLM timeout")
        return {"trader_investment_plan": TRADER_PLAN, "final_trade_decision": f"{signals[ticker]} thesis"}, signals[ticker]

    analyze.calls = calls
    return analyze


@pytest.fixture
def env(tmp_path):
    broker = FakeBroker(prices={"AAPL": 100.0, "JPM": 200.0, "XOM": 50.0})
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ks = KillSwitch(tmp_path / "KILL")
    return broker, ledger, ks


def make(env, analyze, **kw):
    broker, ledger, ks = env
    return Runner(broker, analyze, ledger, sectors=SECTORS, kill_switch=ks,
                  portfolio_builder=lambda a: None, **kw)


def test_buy_flows_through_to_a_fill_and_is_logged_first(env):
    broker, ledger, _ = env
    report = make(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-27")
    assert report.outcomes[0].action == "submitted"
    assert broker.positions["AAPL"] == 50  # 5% of $100k at $100
    kinds = [r["kind"] for r in ledger.records()]
    assert kinds.index("decision") < kinds.index("order")
    assert ledger.verify() == (True, "ok")


def test_rerun_same_day_does_not_double_trade(env):
    broker, _, _ = env
    analyze = stub({"AAPL": "Buy"})
    make(env, analyze).run_cycle(["AAPL"], "2026-10-27")
    second = make(env, analyze).run_cycle(["AAPL"], "2026-10-27")
    assert second.outcomes[0].action == "skipped"
    assert broker.positions["AAPL"] == 50
    assert analyze.calls == ["AAPL"]  # no second LLM run either


def test_broker_level_idempotency():
    b = FakeBroker(prices={"AAPL": 100.0})
    o = OrderRequest("ta-x", "AAPL", Side.BUY, 10)
    first, again = b.submit(o), b.submit(o)
    assert not first.duplicate and again.duplicate
    assert b.positions["AAPL"] == 10


def test_review_signal_never_trades(env):
    broker, _, _ = env
    report = make(env, stub({"AAPL": "REVIEW"})).run_cycle(["AAPL"], "2026-10-27")
    assert report.outcomes[0].action == "none"
    assert broker.orders == {}


def test_dry_run_submits_nothing(env):
    broker, _, _ = env
    report = make(env, stub({"AAPL": "Buy"}), dry_run=True).run_cycle(["AAPL"], "2026-10-27")
    assert report.outcomes[0].action == "dry-run"
    assert broker.orders == {}


def test_market_closed_holds_orders(env):
    broker, _, _ = env
    broker.market_open = False
    report = make(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-27")
    assert report.outcomes[0].action == "skipped" and "market closed" in report.outcomes[0].detail
    assert broker.orders == {}


def test_kill_switch_halts_before_any_analysis(env):
    _, _, ks = env
    ks.engage("test")
    analyze = stub({"AAPL": "Buy"})
    report = make(env, analyze).run_cycle(["AAPL"], "2026-10-27")
    assert report.halted and analyze.calls == []


def test_repeated_errors_auto_suspend(env):
    _, _, ks = env
    analyze = stub({"AAPL": "Buy", "JPM": "Buy", "XOM": "Buy"}, fail={"AAPL", "JPM"})
    report = make(env, analyze, max_errors=2).run_cycle(["AAPL", "JPM", "XOM"], "2026-10-27")
    assert ks.active() and report.halted
    assert report.outcomes[2].action == "skipped"


def test_tampered_ledger_engages_kill_switch(env):
    broker, ledger, ks = env
    make(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-27")
    lines = ledger.path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["data"]["signal"] = "Sell"
    lines[1] = json.dumps(rec, sort_keys=True)
    ledger.path.write_text("\n".join(lines) + "\n")
    report = make(env, stub({"JPM": "Buy"})).run_cycle(["JPM"], "2026-10-28")
    assert ks.active() and report.halted


def test_sell_exits_position(env):
    broker, _, _ = env
    broker.positions["XOM"] = 40
    broker.cash -= 40 * 50
    report = make(env, stub({"XOM": "Sell"})).run_cycle(["XOM"], "2026-10-27")
    assert report.outcomes[0].action == "submitted"
    assert broker.positions["XOM"] == 0


def test_sizing_maps_ratings():
    acct = AccountSnapshot(equity=100_000, cash=90_000, holdings={})
    assert target_weight_for("Buy", "AAPL", acct) == 0.05
    assert target_weight_for("Overweight", "AAPL", acct) == 0.03
    assert target_weight_for("Hold", "AAPL", acct) is None
    assert target_weight_for("Sell", "AAPL", acct) == 0.0
    assert target_weight_for("REVIEW", "AAPL", acct) is None
    assert target_weight_for("Underweight", "AAPL", acct) is None  # nothing held


def test_stop_loss_parsed_from_trader_plan():
    acct = AccountSnapshot(equity=100_000, cash=100_000)
    i = intent_from_decision(decision_id="d", ticker="aapl", rating="Buy", account=acct,
                             final_state={"trader_investment_plan": TRADER_PLAN})
    assert i.stop_loss == 99.0 and i.ticker == "AAPL"
    none = intent_from_decision(decision_id="d", ticker="AAPL", rating="Buy", account=acct,
                                final_state={"trader_investment_plan": "**Stop Loss**: not provided"})
    assert none.stop_loss is None
