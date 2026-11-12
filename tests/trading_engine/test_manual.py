import pytest

from trading_engine.broker import FakeBroker
from trading_engine.commands import CommandBot
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.manual import ManualDesk, rating_override
from trading_engine.models import Side
from trading_engine.reconcile import _market_date
from trading_engine.runner import Runner
from trading_engine.stops import StopBook

TODAY = _market_date(None)


@pytest.fixture
def desk(tmp_path):
    broker = FakeBroker(prices={"CLS": 100.0, "COHR": 50.0}, positions={"COHR": 40})
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ks = KillSwitch(tmp_path / "KILL")
    return ManualDesk(broker, ledger, ks, settle_seconds=0), broker, ledger, ks


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_bot(tmp_path, desk_obj, clock=None):
    return CommandBot("TOKEN", "1", KillSwitch(tmp_path / "KILL"), lambda: "status",
                      send=lambda t: True, get=lambda *a: {"ok": True, "result": []},
                      desk=lambda: desk_obj, clock=clock or Clock())


def test_small_buy_has_no_warnings_and_needs_confirm(tmp_path, desk):
    d, broker, ledger, _ = desk
    bot = make_bot(tmp_path, d)
    preview = bot.handle("/buy CLS 20")
    assert "手动买入 20 股 CLS" in preview and "没有违反风控规则" in preview
    assert broker.orders == {}                       # nothing before /confirm
    done = bot.handle("确认")
    assert done.startswith("✅ 已提交")
    assert broker.positions["CLS"] == 20
    order = ledger.records("order")[-1]["data"]
    assert order["source"] == "manual" and order["decision_id"].startswith("manual|")
    assert StopBook(ledger).current()["CLS"] == 95.0  # default 5% stop
    # The stop order is resting at the broker after the sync.
    assert any(o.is_stop and o.ticker == "CLS" for o in broker.open_orders())


def test_big_buy_warns_but_still_runs_after_confirm(tmp_path, desk):
    d, broker, _, ks = desk
    ks.engage("testing")
    broker.market_open = False
    bot = make_bot(tmp_path, d)
    preview = bot.handle("/buy CLS 500")             # $50k of a ~$102k account
    for words in ("风控提醒", "紧急开关", "不是开盘时间", "超过单只上限", "每笔上限"):
        assert words in preview, words
    assert bot.handle("/confirm").startswith("✅ 已提交")
    assert broker.positions["CLS"] == 500


def test_sell_defaults_to_whole_position_and_never_goes_short(tmp_path, desk):
    d, broker, _, _ = desk
    bot = make_bot(tmp_path, d)
    assert "卖出 40 股 COHR" in bot.handle("/sell COHR")
    assert "不能卖 50 股" in bot.handle("/sell COHR 50")   # replaces nothing: refused
    assert "没有 CLS 持仓" in bot.handle("卖 CLS 1")
    assert bot.handle("/confirm").startswith("✅")         # the earlier 40-share sell
    assert broker.positions.get("COHR", 0) == 0


def test_no_drops_and_confirm_expires(tmp_path, desk):
    d, broker, _, _ = desk
    clock = Clock()
    bot = make_bot(tmp_path, d, clock)
    bot.handle("/buy CLS 1")
    assert bot.handle("/no") == "已放弃。"
    assert bot.handle("/confirm") == "没有待确认的操作。"
    bot.handle("/buy CLS 1")
    clock.t = 301
    assert "作废" in bot.handle("/confirm")
    assert broker.orders == {}


def test_bad_input_is_explained(tmp_path, desk):
    d, _, _, _ = desk
    bot = make_bot(tmp_path, d)
    assert "格式" in bot.handle("/buy CLS")
    assert "正整数" in bot.handle("/buy CLS 1.5")
    assert "评级要写" in bot.handle("/rate CLS maybe")


def test_rate_override_is_used_by_the_next_cycle(tmp_path, desk):
    d, broker, ledger, ks = desk
    bot = make_bot(tmp_path, d)
    assert "用你的评级 Buy" in bot.handle("评级 CLS 买入")
    assert rating_override(ledger, "CLS", TODAY) is None   # not before /confirm
    bot.handle("/confirm")
    assert rating_override(ledger, "CLS", TODAY) == "Buy"

    def analyze(ticker, trade_date, portfolio):
        return {"trader_investment_plan": "**Stop Loss**: 95.00", "final_trade_decision": "x"}, "Sell"

    runner = Runner(broker, analyze, ledger, sectors={"CLS": "Information Technology"}, kill_switch=ks,
                    portfolio_builder=lambda a: None)
    report = runner.run_cycle(["CLS"], TODAY)
    out = report.outcomes[0]
    assert out.signal == "Buy" and out.ai_signal == "Sell" and out.action == "submitted"
    assert "AI 是 Sell" in report.telegram_blocks()[1]
    decision = ledger.records("decision")[-1]["data"]
    assert decision["overridden"] and decision["ai_signal"] == "Sell"

    bot.handle("/rate CLS AI")
    bot.handle("/confirm")
    assert rating_override(ledger, "CLS", TODAY) is None


def test_cancel_leaves_the_stop_order(tmp_path, desk):
    d, broker, _, _ = desk
    broker.fill_immediately = False
    bot = make_bot(tmp_path, d)
    bot.handle("/buy CLS 5")
    bot.handle("/confirm")
    assert "1 张挂单" in bot.handle("/cancel CLS")
    assert bot.handle("/confirm").startswith("✅ 已撤销")
    assert not [o for o in broker.open_orders() if o.ticker == "CLS" and not o.is_stop]


def test_manual_desk_refuses_unknown_side_quantity():
    with pytest.raises(ValueError):
        ManualDesk(FakeBroker(prices={"X": 1.0}), None, None).preview_order(Side.BUY, "X", None)
