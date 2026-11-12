from types import SimpleNamespace

from langchain_core.messages import AIMessage

from trading_engine.broker import FakeBroker
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.runner import Runner
from trading_engine.usage import Usage, UsageCounter, fmt_tokens


def fake_call(counter, tokens_in, tokens_out):
    counter.on_chat_model_start({}, [])
    msg = AIMessage(content="x", usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out,
                                                 "total_tokens": tokens_in + tokens_out})
    counter.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=msg)]]))


def analyzer(cost, fail=()):
    counter = UsageCounter()

    def analyze(ticker, trade_date, portfolio):
        for _ in range(cost[ticker]):
            fake_call(counter, 5_000, 500)
        if ticker in fail:
            raise RuntimeError("rate limited")
        return {"trader_investment_plan": "**Stop Loss**: 95.00", "final_trade_decision": "x"}, "Hold"

    analyze.usage = counter
    analyze.models = {"provider": "chatgpt", "deep": "d", "quick": "q", "analysts": "all"}
    return analyze


def run(tmp_path, cost, fail=()):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    runner = Runner(FakeBroker(prices={"AAA": 10.0, "BBB": 20.0}), analyzer(cost, fail), ledger,
                    sectors={"AAA": "X", "BBB": "Y"}, kill_switch=KillSwitch(tmp_path / "KILL"),
                    portfolio_builder=lambda a: None)
    return runner.run_cycle(list(cost), "2026-11-12"), ledger


def test_each_ticker_usage_is_logged_and_totalled(tmp_path):
    report, ledger = run(tmp_path, {"AAA": 3, "BBB": 2})
    usage = {r["data"]["ticker"]: r["data"] for r in ledger.records("usage")}
    assert usage["AAA"]["llm_calls"] == 3 and usage["AAA"]["tokens_in"] == 15_000
    assert usage["BBB"]["tokens"] == 2 * 5_500
    assert usage["AAA"]["provider"] == "chatgpt" and usage["AAA"]["trade_date"] == "2026-11-12"
    assert report.usage.llm_calls == 5 and report.usage.tokens == 5 * 5_500
    text = report.render()
    assert "usage:" in text and "TOTAL" in text
    assert "🔢 用量：5 次调用，2.8万 token" in report.telegram_blocks()[-1]


def test_usage_is_logged_even_when_the_analysis_fails(tmp_path):
    report, ledger = run(tmp_path, {"AAA": 4}, fail={"AAA"})
    assert report.outcomes[0].action == "error"
    assert ledger.records("usage")[0]["data"]["llm_calls"] == 4


def test_counter_resets_per_ticker_and_formats():
    c = UsageCounter()
    fake_call(c, 100, 10)
    c.start()
    assert c.snapshot().llm_calls == 0
    assert Usage(1, 2, 3) + Usage(1, 2, 3) == Usage(2, 4, 6)
    assert fmt_tokens(176_400) == "17.6万" and fmt_tokens(900) == "900"


def test_analyzer_without_a_counter_still_runs(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    runner = Runner(FakeBroker(prices={"AAA": 10.0}),
                    lambda t, d, p: ({"final_trade_decision": "x"}, "Hold"), ledger,
                    sectors={"AAA": "X"}, kill_switch=KillSwitch(tmp_path / "K"), portfolio_builder=lambda a: None)
    report = runner.run_cycle(["AAA"], "2026-11-12")
    assert report.usage is None and not ledger.records("usage")
