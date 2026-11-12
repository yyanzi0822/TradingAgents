"""Telegram messages: what is sent, and that a failure never breaks a cycle."""

import logging

import pytest

from trading_engine.broker import FakeBroker
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.notify import NullNotifier, TelegramNotifier, from_env
from trading_engine.reconcile import Reconciler, _market_date
from trading_engine.runner import Runner
from trading_engine.summary import build_summary

SECTORS = {"AAPL": "Information Technology", "JPM": "Financials"}
PLAN = "**Stop Loss**: 95.00"
TODAY = _market_date(None)


class Inbox:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)
        return True


def stub(signals, fail=()):
    def analyze(ticker, trade_date, portfolio):
        if ticker in fail:
            raise RuntimeError("LLM timeout")
        return {"trader_investment_plan": PLAN}, signals[ticker]
    return analyze


@pytest.fixture
def env(tmp_path):
    return (FakeBroker(prices={"AAPL": 100.0, "JPM": 200.0}), Ledger(tmp_path / "ledger.jsonl"),
            KillSwitch(tmp_path / "KILL"), Inbox())


def runner(env, analyze, **kw):
    broker, ledger, ks, inbox = env
    return Runner(broker, analyze, ledger, sectors=SECTORS, kill_switch=ks, notifier=inbox,
                  portfolio_builder=lambda a: None, **kw)


def test_order_message_has_side_size_price_and_stop(env):
    runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-27")
    assert env[3].messages == ["BUY 50 AAPL @ ~100.00, stop 95.00\nRating: Buy · filled"]


def test_hold_sends_nothing(env):
    runner(env, stub({"AAPL": "Hold"})).run_cycle(["AAPL"], "2026-10-27")
    assert env[3].messages == []


def test_stop_hit_is_messaged_when_reconciliation_sees_it(env):
    broker, ledger, ks, inbox = env
    runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], "2026-10-26")
    inbox.messages.clear()
    broker.set_price("AAPL", 94.0)
    Reconciler(broker, ledger, kill_switch=ks, notifier=inbox).run()
    assert len(inbox.messages) == 1
    assert inbox.messages[0].startswith("🛑 STOP HIT AAPL: sold 50 @ 94.00\nsold at a loss; a rebuy before ")


def test_errors_and_kill_switch_are_messaged(env):
    r = runner(env, stub({}, fail={"AAPL", "JPM"}), max_errors=2)
    r.run_cycle(["AAPL", "JPM"], "2026-10-27")
    msgs = env[3].messages
    assert msgs[0].startswith("⚠️ ERROR AAPL:") and msgs[1].startswith("⚠️ ERROR JPM:")
    assert msgs[2].startswith("🚨 KILL SWITCH ON: 2 errors in cycle")


def test_dry_run_sends_no_messages(env):
    runner(env, stub({"AAPL": "Buy"}), dry_run=True).run_cycle(["AAPL"], "2026-10-27")
    assert env[3].messages == []


def test_telegram_payload_and_long_messages_are_cut(monkeypatch):
    calls = []

    def post(url, payload, timeout):
        calls.append((url, payload))
        return {"ok": True}

    assert TelegramNotifier("123:ABC", 42, post=post).send("x" * 5000)
    url, payload = calls[0]
    assert url == "https://api.telegram.org/bot123:ABC/sendMessage"
    assert payload["chat_id"] == "42" and len(payload["text"]) <= 4000


def test_a_failed_message_never_raises_and_never_logs_the_token(caplog):
    def post(url, payload, timeout):
        raise ConnectionError(f"could not reach {url}")

    with caplog.at_level(logging.WARNING):
        assert TelegramNotifier("123:SECRET", 42, post=post).send("hi") is False
        assert TelegramNotifier("123:SECRET", 42, post=lambda *a: {"ok": False, "description": "chat not found"}
                                ).send("hi") is False
    assert "SECRET" not in caplog.text and "chat not found" in caplog.text


def test_trading_carries_on_when_telegram_is_down(env):
    broker, ledger, ks, _ = env

    def post(url, payload, timeout):
        raise TimeoutError

    Runner(broker, stub({"AAPL": "Buy"}), ledger, sectors=SECTORS, kill_switch=ks,
           notifier=TelegramNotifier("t", 1, post=post),
           portfolio_builder=lambda a: None).run_cycle(["AAPL"], "2026-10-27")
    assert broker.positions["AAPL"] == 50 and not ks.active()


def test_from_env_is_silent_without_both_settings(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert isinstance(from_env(), NullNotifier)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    assert isinstance(from_env(), NullNotifier)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    assert isinstance(from_env(), TelegramNotifier)


def test_daily_summary_lists_positions_stops_and_todays_activity(env):
    broker, ledger, ks, _ = env
    broker.positions["JPM"], broker.entry_prices["JPM"], broker.cash = 10, 200.0, 98_000.0
    runner(env, stub({"AAPL": "Buy"})).run_cycle(["AAPL"], TODAY)
    broker.set_price("JPM", 180.0)                    # JPM's default stop (190) triggers
    Reconciler(broker, ledger, kill_switch=ks).run()

    text = build_summary(broker, ledger, ks, day=TODAY)
    assert f"Daily summary {TODAY}" in text
    assert "AAPL 50 sh $5,000 5.0% · stop 95.00" in text
    assert "Today: 1 order(s), 1 stop-out(s), 0 error(s)" in text
    assert "buy 50 AAPL — filled" in text and "stopped out JPM 10 @ 180.00" in text
    assert text.endswith("Kill switch: off")


# --- per-ticker cycle report -------------------------------------------------

THESIS = ("**Rating**: Hold\n\n**Executive Summary**: Keep the position; the **trend** holds.\n\n"
          "**Investment Thesis**: long text that is not sent")


def test_report_lists_every_ticker_with_rating_action_and_reason(env):
    def analyze(ticker, trade_date, portfolio):
        signal = {"AAPL": "Buy", "JPM": "Hold"}[ticker]
        return {"trader_investment_plan": PLAN, "final_trade_decision": THESIS}, signal

    report = runner(env, analyze, dry_run=True).run_cycle(["AAPL", "JPM"], TODAY)
    blocks = report.telegram_blocks(dry_run=True)
    assert "dry run" in blocks[0]
    aapl, jpm = blocks[1], blocks[2]
    assert aapl.startswith("AAPL · Buy（买入）") and "模拟（未下单）" in aapl
    assert jpm.startswith("JPM · Hold（持有）") and "不操作" in jpm
    # The executive summary only, without markdown bold.
    assert "理由: Keep the position; the trend holds." in jpm
    assert "Investment Thesis" not in jpm


def test_executive_summary_falls_back_and_is_shortened():
    from trading_engine.runner import executive_summary
    assert executive_summary("") == ""
    assert executive_summary("plain text") == "plain text"
    assert executive_summary("x" * 1000, limit=50).endswith("…")
    assert len(executive_summary("x" * 1000, limit=50)) == 50


def test_send_blocks_packs_messages_without_splitting_a_block():
    from trading_engine.notify import send_blocks
    inbox = Inbox()
    assert send_blocks(inbox, ["a" * 30, "b" * 30, "c" * 30], limit=70)
    assert inbox.messages == ["a" * 30 + "\n\n" + "b" * 30, "c" * 30]
