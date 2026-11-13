import json

from trading_engine import publish
from trading_engine.publish import (
    build,
    forward_returns,
    key_points,
    ledger_ratings,
    scorecard,
    scrub,
)


def write_ledger(path, decisions):
    with path.open("w") as f:
        for d in decisions:
            f.write(json.dumps({"kind": "decision", "data": d}) + "\n")


def write_state(reports, ticker, day, rating="Sell"):
    p = reports / ticker / "TradingAgentsStrategy_logs" / f"full_states_log_{day}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "final_trade_decision": f"**Rating**: {rating}\n\n**Executive Summary**: Keep the $100,000 portfolio in cash. "
                                "Margins are shrinking.\n\n**Investment Thesis**: x",
        "market_report": "m", "news_report": "n", "fundamentals_report": "f"}))


def test_scrub_drops_account_sentences():
    text = ("Revenue grew 16%. Keep the $100,000 in cash for now. With no shares held, wait. "
            "Size the position against a loss budget. The stock trades above its 200-day average.")
    assert scrub(text) == "Revenue grew 16%. The stock trades above its 200-day average."


def test_public_rating_is_the_agents_not_a_manual_override(tmp_path):
    led = tmp_path / "ledger.jsonl"
    write_ledger(led, [
        {"trade_date": "2026-11-12", "ticker": "AAA", "signal": "Buy", "ai_signal": "Sell", "overridden": True},
        {"trade_date": "2026-11-12", "ticker": "BBB", "signal": "Hold"},
        {"trade_date": "2026-11-12", "ticker": "CCC", "signal": "REVIEW"},
    ])
    assert ledger_ratings(led) == [{"day": "2026-11-12", "ticker": "AAA", "rating": "Sell"},
                                   {"day": "2026-11-12", "ticker": "BBB", "rating": "Hold"}]


def test_key_points_from_model_are_scrubbed_and_capped():
    def llm(prompt):
        assert "never mention any portfolio" in prompt.lower()
        return 'Sure: {"headline": "AAA is rated Sell. Hold the $100,000 portfolio in cash.", ' \
               '"technical": ["a", "b", "c", "d"], "bull": ["Keep cash in the account.", "growth"]}'
    p = key_points({"final_trade_decision": "x"}, "AAA", "Sell", llm)
    assert p["headline"] == "AAA is rated Sell."
    assert p["technical"] == ["a", "b", "c"] and p["bull"] == ["growth"] and p["news"] == []


def test_key_points_fall_back_without_a_model():
    p = key_points({"final_trade_decision": "**Executive Summary**: Keep the $100,000 portfolio in cash. "
                                            "Margins are shrinking."}, "AAA", "Sell", None)
    assert p["headline"] == "Margins are shrinking."


def test_forward_returns_and_scorecard():
    days = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"]
    closes = {"AAA": dict(zip(days, [100, 101, 102, 103, 104, 110, 111], strict=True)),
              "BBB": dict(zip(days, [50, 50, 50, 50, 50, 45, 45], strict=True)),
              "SPY": dict(zip(days, [400, 400, 400, 400, 400, 404, 404], strict=True))}
    ratings = [{"day": "2026-11-02", "ticker": "AAA", "rating": "Buy"},
               {"day": "2026-11-02", "ticker": "BBB", "rating": "Sell"},
               {"day": "2026-11-06", "ticker": "AAA", "rating": "Buy"}]   # not 5 sessions old yet
    res = forward_returns(ratings, closes)
    assert len(res) == 2
    aaa = next(r for r in res if r["ticker"] == "AAA")
    assert aaa["exit_day"] == "2026-11-09" and round(aaa["ret"], 3) == 0.10 and round(aaa["excess"], 3) == 0.09
    card = {c["group"].split()[0]: c for c in scorecard(res)}
    assert card["Bullish"]["hit"] == 1.0 and card["Bearish"]["hit"] == 1.0


def test_build_writes_pages_without_account_data(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    write_state(reports, "AAA", "2026-11-11", "Buy")
    write_state(reports, "AAA", "2026-11-12", "Sell")
    write_state(reports, "BBB", "2026-11-12", "Hold")
    ratings = [{"day": "2026-11-11", "ticker": "AAA", "rating": "Buy"},
               {"day": "2026-11-12", "ticker": "AAA", "rating": "Sell"},
               {"day": "2026-11-12", "ticker": "BBB", "rating": "Hold"},
               {"day": "2026-11-12", "ticker": "ZZZ", "rating": "Buy"}]     # no saved report: skipped
    site = tmp_path / "site"
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return '{"headline": "Rated on <b>margins</b>.", "technical": ["t"]}'

    written = build(site, ratings, llm=llm, reports_dir=reports, closes_fn=lambda *a: {}, today="2026-11-12")
    assert {"index.html", "archive.html", "performance.html", "about.html", "stocks/AAA.html",
            "days/2026-11-11.html", "days/2026-11-12.html"} <= set(written)
    assert "stocks/ZZZ.html" not in written
    index = (site / "index.html").read_text()
    assert "▼ from Buy" in index                                # rating change is shown
    assert "&lt;b&gt;margins&lt;/b&gt;" in index                # model text is escaped
    assert "100,000" not in "".join(p.read_text() for p in site.rglob("*.html"))
    assert len(calls) == 6                                      # key points + scores per stock
    build(site, ratings, llm=llm, reports_dir=reports, closes_fn=lambda *a: {}, today="2026-11-12")
    assert len(calls) == 6                                      # both are cached
    assert json.loads((site / "data" / "ratings.json").read_text())[-1]["ticker"] == "BBB"


def test_build_survives_a_price_feed_failure(tmp_path):
    reports = tmp_path / "r"
    write_state(reports, "AAA", "2026-11-12")

    def broken(*a):
        raise ConnectionError("down")

    written = build(tmp_path / "s", [{"day": "2026-11-12", "ticker": "AAA", "rating": "Sell"}],
                    llm=None, reports_dir=reports, closes_fn=broken, today="2026-11-12")
    assert "performance.html" in written


def test_disclaimer_is_on_every_page(tmp_path):
    reports = tmp_path / "r"
    write_state(reports, "AAA", "2026-11-12")
    site = tmp_path / "s"
    build(site, [{"day": "2026-11-12", "ticker": "AAA", "rating": "Sell"}], llm=None, reports_dir=reports,
          closes_fn=lambda *a: {}, today="2026-11-12")
    for page in site.rglob("*.html"):
        assert publish.DISCLAIMER[:40] in page.read_text(), page


def test_day_list_runs_from_buy_to_sell_with_company_profiles(tmp_path):
    led, reports, site = tmp_path / "ledger.jsonl", tmp_path / "reports", tmp_path / "site"
    day = "2026-11-12"
    picks = {"AAA": "Sell", "BBB": "Buy", "CCC": "Hold", "DDD": "Overweight"}
    write_ledger(led, [{"trade_date": day, "ticker": t, "signal": r} for t, r in picks.items()])
    for t, r in picks.items():
        write_state(reports, t, day, r)

    def profile(ticker, state):
        return {"name": f"{ticker} Corp", "sector": "Industrials", "industry": "Widgets",
                "about": f"{ticker} Corp makes widgets."}

    build(site, ledger_ratings(led), llm=None, reports_dir=reports, closes_fn=lambda *a: {},
          today=day, profile_fn=profile)
    index = (site / "index.html").read_text()
    order = [index.index(f"stocks/{t}.html") for t in ("BBB", "DDD", "CCC", "AAA")]
    assert order == sorted(order)
    assert "BBB Corp" in index and "Industrials" in index
    stock = (site / "stocks" / "BBB.html").read_text()
    assert "About the company" in stock and "BBB Corp makes widgets." in stock and "Widgets" in stock
    assert json.loads((site / "data" / "companies.json").read_text())["AAA"]["name"] == "AAA Corp"


def test_company_intro_keeps_a_name_with_holding_in_it(monkeypatch):
    import sys
    import types
    fake = types.SimpleNamespace(Ticker=lambda t: types.SimpleNamespace(
        info={"longName": "Credo Technology Group Holding Ltd", "industry": "Semiconductors"}))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    prof = publish.make_profile_fn(lambda prompt: "Credo Technology Group Holding Ltd makes chips. "
                                                  "It serves data centers.")("CRDO", None)
    assert prof["about"].startswith("Credo Technology Group Holding Ltd makes chips.")
    assert prof["industry"] == "Semiconductors"


def test_trend_arrow_against_the_previous_day():
    assert publish.trend("Hold", None) == ""                       # no earlier rating: nothing
    assert "▲ from Sell" in publish.trend("Hold", "Sell")
    assert "▼ from Buy" in publish.trend("Overweight", "Buy")
    assert "▲ from Moderate Sell" in publish.trend("Hold", "Underweight")
    assert "unchanged" in publish.trend("Sell", "Sell")


def test_dimension_scores_are_validated_and_shown(tmp_path):
    assert publish.dimension_scores({}, "X", None) == {}
    got = publish.dimension_scores({}, "X", lambda p: '{"technical": "weak", "fundamentals": "Great", "news": "Good"}')
    assert got == {"technical": "Weak", "news": "Good"}                 # unknown levels dropped
    reports, site = tmp_path / "r", tmp_path / "s"
    write_state(reports, "AAA", "2026-11-12", "Hold")

    def llm(prompt):
        if "Grade AAA" in prompt:
            return '{"technical": "Poor", "fundamentals": "Strong", "news": "Mixed"}'
        return '{"headline": "Rated Hold.", "technical": ["t"]}'

    build(site, [{"day": "2026-11-12", "ticker": "AAA", "rating": "Hold"}], llm=llm, reports_dir=reports,
          closes_fn=lambda *a: {}, today="2026-11-12")
    index = (site / "index.html").read_text()
    assert 't-sell">Poor' in index and 't-buy">Strong' in index and 't-hold">Mixed' in index
    assert 't-sell">Poor' in (site / "stocks" / "AAA.html").read_text()


def test_every_column_can_sort(tmp_path):
    reports, site = tmp_path / "r", tmp_path / "s"
    write_state(reports, "AAA", "2026-11-12", "Hold")
    build(site, [{"day": "2026-11-12", "ticker": "AAA", "rating": "Hold"}], llm=None, reports_dir=reports,
          closes_fn=lambda *a: {}, today="2026-11-12")
    index = (site / "index.html").read_text()
    for key in ("stock", "technical", "fundamentals", "news", "rating"):
        assert f'data-key="{key}"' in index, key
    assert 'data-stock="AAA" data-rating="2" data-technical="9"' in index   # no grade sorts last
    assert "<select>" in index and "rows.sort" in index


def test_public_labels_replace_broker_jargon():
    assert publish.label("Overweight") == "Moderate Buy" and publish.label("Underweight") == "Moderate Sell"
    assert publish.label("Buy") == "Buy"
    assert publish.plain("CRDO is rated Overweight as growth holds.") == "CRDO is rated Moderate Buy as growth holds."
    assert "Moderate Sell" in publish.badge("Underweight")
