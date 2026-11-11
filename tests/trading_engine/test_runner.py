import json

import pytest

from trading_engine.broker import FakeBroker
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.models import AccountSnapshot, Holding, OrderRequest, Side
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
    assert target_weight_for("Overweight", "AAPL", acct) is None
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


# --- regressions -------------------------------------------------------------


class LostReplyBroker(FakeBroker):
    """Accepts the first order, then raises as if the reply never arrived."""

    def submit(self, order):
        status = FakeBroker.submit(self, order)
        if len(self.orders) == 1 and not getattr(self, "_failed", False):
            self._failed = True
            raise TimeoutError("reply lost")
        return status


def test_retry_after_lost_reply_does_not_buy_twice(tmp_path):
    broker = LostReplyBroker(prices={"AAPL": 200.0}, fill_immediately=False)
    env = (broker, Ledger(tmp_path / "ledger.jsonl"), KillSwitch(tmp_path / "KILL"))
    analyze = stub({"AAPL": "Buy"})

    first = make(env, analyze).run_cycle(["AAPL"], "2026-10-27")
    assert first.outcomes[0].action == "error"

    broker.prices["AAPL"] = 203.0          # a different quantity would be sized now
    second = make(env, analyze).run_cycle(["AAPL"], "2026-10-27")

    assert len(broker.orders) == 1
    assert second.outcomes[0].action == "duplicate"
    assert analyze.calls == ["AAPL"]       # the retry did not pay for a second analysis
    recovered = env[1].records("order")
    assert len(recovered) == 1 and recovered[0]["data"]["recovered"] is True

    third = make(env, analyze).run_cycle(["AAPL"], "2026-10-27")
    assert third.outcomes[0].detail == "already traded today"


def test_retry_does_not_trade_when_the_agents_flip_sides(tmp_path):
    broker = LostReplyBroker(prices={"AAPL": 200.0}, positions={"AAPL": 10}, fill_immediately=False)
    env = (broker, Ledger(tmp_path / "ledger.jsonl"), KillSwitch(tmp_path / "KILL"))
    make(env, stub({"AAPL": "Sell"})).run_cycle(["AAPL"], "2026-10-27")
    make(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-27")
    assert [o.side for o in broker.orders.values()] == ["sell"]


def test_dry_run_ledger_is_separate_from_the_real_one(tmp_path):
    from trading_engine.run import default_ledger_path

    dry, real = default_ledger_path(True, tmp_path), default_ledger_path(False, tmp_path)
    assert dry != real

    # A dry run on a made-up $100k book, then a real $50k account.
    Runner(FakeBroker(prices={"AAPL": 100.0}), stub({"AAPL": "Buy"}), Ledger(dry), sectors=SECTORS,
           kill_switch=KillSwitch(tmp_path / "KILL"), dry_run=True,
           portfolio_builder=lambda a: None).run_cycle(["AAPL"], "2026-10-26")
    report = Runner(FakeBroker(cash=50_000, prices={"AAPL": 100.0}), stub({"AAPL": "Buy"}), Ledger(real),
                    sectors=SECTORS, kill_switch=KillSwitch(tmp_path / "KILL"),
                    portfolio_builder=lambda a: None).run_cycle(["AAPL"], "2026-10-27")
    assert report.outcomes[0].action == "submitted"


def test_half_written_ledger_line_engages_kill_switch_instead_of_crashing(env):
    broker, ledger, ks = env
    ledger.append("equity", {"equity": 100_000.0, "cash": 100_000.0})
    with ledger.path.open("a", encoding="utf-8") as f:
        f.write('{"kind": "eq')
    analyze = stub({"AAPL": "Buy"})

    report = make(env, analyze).run_cycle(["AAPL"], "2026-10-27")

    assert ks.active() and "unreadable record at line 2" in ks.reason()
    assert report.halted and not analyze.calls and not broker.orders


def test_saved_head_detects_records_removed_from_the_end(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for i in range(3):
        ledger.append("equity", {"equity": float(i), "cash": 0.0})
    head = ledger.head()
    assert ledger.verify(expected_head=head) == (True, "ok")

    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    ledger.path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    assert ledger.verify() == (True, "ok")                # the chain alone cannot see it
    ok, msg = ledger.verify(expected_head=head)
    assert not ok and "expects 3" in msg

    ledger.append("equity", {"equity": 9.0, "cash": 0.0})  # same length, different history
    ok, msg = ledger.verify(expected_head=head)
    assert not ok and "does not match" in msg


def test_downgrade_from_buy_to_overweight_keeps_the_position(env):
    broker, _, _ = env
    make(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-26")
    assert broker.positions["AAPL"] == 50                       # 5%

    report = make(env, stub({"AAPL": "Overweight"})).run_cycle(["AAPL"], "2026-10-27")
    assert broker.positions["AAPL"] == 50                       # not sold down to 3%
    assert report.outcomes[0].action == "none" and "hold" in report.outcomes[0].detail


def test_overweight_does_not_trade(env):
    broker, ledger, _ = env
    report = make(env, stub({"AAPL": "Overweight"})).run_cycle(["AAPL"], "2026-10-26")
    assert not broker.orders and report.outcomes[0].action == "none"
    assert ledger.records("decision")[0]["data"]["signal"] == "Overweight"   # still logged


def test_add_only_rating_tops_up_but_never_trims():
    from trading_engine.config import SizingPolicy

    grown = AccountSnapshot(100_000, 93_000, {"AAPL": Holding("AAPL", 70, 7_000.0)})
    small = AccountSnapshot(100_000, 99_000, {"AAPL": Holding("AAPL", 10, 1_000.0)})
    assert target_weight_for("Buy", "AAPL", grown) is None                # 7% held: keep
    assert target_weight_for("Buy", "AAPL", small) == 0.05               # 1% held: top up
    assert target_weight_for("Buy", "AAPL", grown, SizingPolicy(add_only_ratings=())) == 0.05


def test_buy_on_a_position_that_has_grown_past_5pct_sells_nothing(env):
    broker, _, _ = env
    from trading_engine.config import RiskLimits

    no_tp = RiskLimits(take_profit_levels=())                    # this test is about sizing only
    make(env, stub({"AAPL": "Buy"}), limits=no_tp).run_cycle(["AAPL"], "2026-10-26")
    broker.prices["AAPL"] = 140.0                                # 50 shares are now ~6.9%
    report = make(env, stub({"AAPL": "Buy"}), limits=no_tp).run_cycle(["AAPL"], "2026-10-27")
    assert broker.positions["AAPL"] == 50 and report.outcomes[0].action == "none"


def test_underweight_and_sell_still_reduce(env):
    broker, _, _ = env
    make(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-26")
    make(env, stub({"AAPL": "Underweight"})).run_cycle(["AAPL"], "2026-10-27")
    assert broker.positions["AAPL"] == 25
    make(env, stub({"AAPL": "Sell"})).run_cycle(["AAPL"], "2026-10-28")
    assert broker.positions["AAPL"] == 0
