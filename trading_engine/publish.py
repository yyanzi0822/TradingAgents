"""AI Stock Daily: a public static site with each day's ratings and key points.

    python -m trading_engine.publish --site ~/dev/ai-stock-daily            # build
    python -m trading_engine.publish --site ~/dev/ai-stock-daily --push     # build, commit, push

Reads the agents' decisions from the ledger and their saved reports, and writes
plain HTML (no build step) for GitHub Pages or any static host.

Nothing about the account is published: no positions, cash, orders or stops,
and the rating shown is always the agents' own, never a manual /rate override.
Key points are rewritten by one quick-model call per ticker with instructions to
leave the portfolio out, and a filter drops any sentence that still mentions it.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("TRADING_ENGINE_DIR", Path.home() / ".trading_engine"))
REPORTS_DIR = Path.home() / ".tradingagents" / "logs"
SITE_NAME = "AI Stock Daily"
HORIZON_DAYS = 5
BULLISH, BEARISH = {"Buy", "Overweight"}, {"Underweight", "Sell"}
RATING_CLASS = {"Buy": "buy", "Overweight": "ow", "Hold": "hold", "Underweight": "uw", "Sell": "sell"}
RATING_ORDER = {r: i for i, r in enumerate(RATING_CLASS)}         # Buy first, Sell last
UNIVERSE_FILE = Path(__file__).with_name("universe.json")

DISCLAIMER = ("AI-generated research for education and information only. Not investment advice, not a "
              "recommendation to buy or sell any security, and not tailored to anyone's situation. The "
              "analysis can be wrong, incomplete or out of date. Do your own research; past results do not "
              "predict future returns.")

# Sentences that talk about an account rather than the stock.
# Company language such as "operating cash flow" or "competitive position" is left alone.
PRIVATE = re.compile("|".join([
    r"\$\s?\d[\d,]*(?:\.\d+)?\s*k?\b[^.]*\b(?:portfolio|cash|account)\b",       # "$100,000 in cash"
    r"\b(?:portfolio|holdings?|allocation)\b",
    r"\b(?:in|our|the|your|(?:keep|hold|retain|preserv|maintain|stay)\w*(?:\s+\w+){0,2})\s+cash\b"
    r"(?!\s+(?:flow|generation|conversion|burn|position|balance sheet|and|from))",
    r"\b(?:the|our|your|this|in the|an?)\s+account\b",
    r"\b(?:no|existing|our|your) shares\b|\bshares (?:held|owned)\b",
    r"\bposition\s+siz\w*|\b(?:size|sizing|initiat\w*|open\w*|add\w* to|trim\w*|exit\w*|enter\w*|"
    r"establish\w*|build\w*|stag\w*|start\w*|current|existing|no|new|small|full)\s+(?:a |the |any |that )?"
    r"(?:\w+ )?position\b",
]), re.I)


# --- collect ---------------------------------------------------------------------

@dataclass
class Call:
    day: str
    ticker: str
    rating: str
    points: dict


def ledger_ratings(ledger_path: Path) -> list[dict]:
    """The agents' rating per (day, ticker): the last decision, never a manual override."""
    seen: dict[tuple[str, str], dict] = {}
    if not ledger_path.exists():
        return []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") != "decision":
            continue
        d = rec["data"]
        rating = d.get("ai_signal") or d.get("signal")
        if rating not in RATING_CLASS:
            continue
        seen[(d["trade_date"], d["ticker"])] = {"day": d["trade_date"], "ticker": d["ticker"], "rating": rating}
    return sorted(seen.values(), key=lambda r: (r["day"], r["ticker"]))


def load_state(ticker: str, day: str, reports_dir: Path = REPORTS_DIR) -> dict | None:
    p = reports_dir / ticker / "TradingAgentsStrategy_logs" / f"full_states_log_{day}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def scrub(text: str) -> str:
    """Drop sentences that mention an account, portfolio, cash or holdings."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(str(text or "").split()))
    return " ".join(s for s in sentences if s and not PRIVATE.search(s)).strip()


KEY_POINTS_PROMPT = """You turn an equity research team's internal notes on {ticker} into a short public brief.

Rules:
- Write about the stock and the company only. Never mention any portfolio, account, cash balance,
  holdings, position size, shares owned, allocation or what "we" should do with money.
- Plain, neutral English. Concrete numbers from the notes are welcome; do not invent any.
- Describe the view; never give an instruction. Write "{ticker} is rated Sell because ...",
  never "Sell {ticker}" or "investors should ...".
- The final rating is {rating}; explain it, do not change it.

Return only JSON with these keys:
{{"headline": "one sentence, at most 25 words",
  "technical": ["up to 3 short bullets"],
  "fundamentals": ["up to 3 short bullets"],
  "news": ["up to 3 short bullets"],
  "bull": ["up to 2 short bullets"],
  "bear": ["up to 2 short bullets"],
  "watch": ["up to 2 things that would change the view"]}}

NOTES
=== Final decision ===
{final}
=== Technical ===
{market}
=== News and macro ===
{news}
=== Fundamentals ===
{fundamentals}
"""


def key_points(state: dict, ticker: str, rating: str, llm: Callable[[str], str] | None) -> dict:
    """Public key points for one ticker; falls back to a scrubbed summary without a model."""
    points: dict = {}
    if llm is not None:
        prompt = KEY_POINTS_PROMPT.format(
            ticker=ticker, rating=rating,
            final=str(state.get("final_trade_decision", ""))[:4000],
            market=str(state.get("market_report", ""))[:6000],
            news=str(state.get("news_report", ""))[:6000],
            fundamentals=str(state.get("fundamentals_report", ""))[:8000])
        try:
            raw = llm(prompt)
            m = re.search(r"\{.*\}", raw, re.S)
            points = json.loads(m.group(0)) if m else {}
        except Exception as exc:  # noqa: BLE001 - fall back below
            logger.warning("key points for %s failed: %s", ticker, type(exc).__name__)
    if not points.get("headline"):
        from trading_engine.runner import executive_summary
        points["headline"] = executive_summary(str(state.get("final_trade_decision", "")), 400)
    clean: dict = {"headline": scrub(points.get("headline", ""))}
    for key in ("technical", "fundamentals", "news", "bull", "bear", "watch"):
        items = points.get(key) or []
        clean[key] = [s for s in (scrub(i) for i in items if isinstance(i, str)) if s][:3]
    return clean


def make_llm() -> Callable[[str], str]:
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.llm_clients.factory import create_tier_client

    model = create_tier_client(DEFAULT_CONFIG.copy(), "quick").get_llm()

    def ask(prompt: str) -> str:
        out = model.invoke(prompt)
        content = out.content
        if isinstance(content, list):
            content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
        return str(content)

    return ask


# --- company profiles ------------------------------------------------------------------

def universe_sector(ticker: str) -> str:
    try:
        return json.loads(UNIVERSE_FILE.read_text())["tickers"].get(ticker, "")
    except (OSError, ValueError, KeyError):
        return ""


PROFILE_PROMPT = """Write a neutral two-sentence introduction to {name} ({ticker}) for a stock research site:
what the company does, its main products or customers, and where it fits in its industry ({industry}).
No opinions, no ratings, no stock-price talk, no investment language. Do not copy the description;
write it in your own words, and never say information is missing. Return only the two sentences.

COMPANY DESCRIPTION
{summary}

RESEARCH NOTES
{notes}
"""


def make_profile_fn(llm: Callable[[str], str] | None) -> Callable[[str, dict | None], dict]:
    """Company name, sector and industry from Yahoo Finance, and a short intro written by the model."""
    def profile(ticker: str, state: dict | None) -> dict:
        info: dict = {}
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
        except Exception as exc:  # noqa: BLE001 - fall back to the universe sector
            logger.warning("profile for %s unavailable: %s", ticker, type(exc).__name__)
        name = info.get("longName") or info.get("shortName") or ticker
        out = {"name": name, "sector": universe_sector(ticker) or info.get("sector", ""),
               "industry": info.get("industry", ""), "about": ""}
        if llm is not None:
            try:
                notes = str((state or {}).get("fundamentals_report", ""))[:4000]
                text = llm(PROFILE_PROMPT.format(name=name, ticker=ticker,
                                                 summary=str(info.get("longBusinessSummary", ""))[:3000],
                                                 industry=out["industry"] or out["sector"], notes=notes))
                # Mask the name so "Holding Ltd." is not taken for holdings or a sentence end.
                out["about"] = scrub(text.strip().replace(name, "THECOMPANY")).replace("THECOMPANY", name)[:700]
            except Exception as exc:  # noqa: BLE001
                logger.warning("intro for %s failed: %s", ticker, type(exc).__name__)
        return out
    return profile


def default_profile(ticker: str, state: dict | None) -> dict:
    return {"name": ticker, "sector": universe_sector(ticker), "industry": "", "about": ""}


# --- performance -------------------------------------------------------------------

def yf_closes(tickers: list[str], start: str, end: str) -> dict[str, dict[str, float]]:
    import yfinance as yf

    out: dict[str, dict[str, float]] = {}
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, group_by="ticker")
    for t in tickers:
        try:
            s = (data[t] if len(tickers) > 1 else data)["Close"].dropna()
        except KeyError:
            continue
        out[t] = {d.strftime("%Y-%m-%d"): float(v) for d, v in s.items()}
    return out


def forward_returns(ratings: list[dict], closes: dict[str, dict[str, float]],
                    benchmark: str = "SPY", horizon: int = HORIZON_DAYS) -> list[dict]:
    """Return from the close on the rating day to the close `horizon` sessions later, and vs the benchmark."""
    bench = closes.get(benchmark, {})
    out = []
    for r in ratings:
        series = closes.get(r["ticker"], {})
        days = sorted(series)
        if r["day"] not in series or r["day"] not in bench:
            continue
        i = days.index(r["day"])
        if i + horizon >= len(days):
            continue
        later = days[i + horizon]
        if later not in bench:
            continue
        ret = series[later] / series[r["day"]] - 1
        bret = bench[later] / bench[r["day"]] - 1
        out.append({**r, "exit_day": later, "ret": ret, "excess": ret - bret})
    return out


def scorecard(results: list[dict]) -> list[dict]:
    rows = []
    for name, group in (("Bullish (Buy, Overweight)", BULLISH), ("Hold", {"Hold"}),
                        ("Bearish (Underweight, Sell)", BEARISH)):
        xs = [r for r in results if r["rating"] in group]
        if not xs:
            continue
        right = [r for r in xs if (r["excess"] > 0 if group is BULLISH else r["excess"] < 0)]
        rows.append({"group": name, "n": len(xs),
                     "avg_ret": sum(r["ret"] for r in xs) / len(xs),
                     "avg_excess": sum(r["excess"] for r in xs) / len(xs),
                     "hit": None if group == {"Hold"} else len(right) / len(xs)})
    return rows


# --- render ---------------------------------------------------------------------------

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#e7e4dd;--accent:#2f5d50;
--buy:#1f7a4d;--ow:#4c9a6a;--hold:#8a8a8f;--uw:#c27c2c;--sell:#b5402f}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ececec;--muted:#9a9aa0;--card:#1d1d1f;
--line:#2c2c2e;--accent:#8cc5b0}}
a{color:var(--accent)}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:24px 16px 64px}
header{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:baseline;border-bottom:1px solid var(--line);
padding-bottom:14px;margin-bottom:22px}header .brand{font-weight:700;font-size:20px;color:var(--fg);
text-decoration:none}nav a{color:var(--muted);text-decoration:none;margin-right:16px}nav a:hover{color:var(--fg)}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:32px 0 12px}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h3{margin:0 0 6px;font-size:18px}.card a{color:inherit;text-decoration:none}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;color:#fff;font-size:13px;font-weight:600}
.buy{background:var(--buy)}.ow{background:var(--ow)}.hold{background:var(--hold)}.uw{background:var(--uw)}
.sell{background:var(--sell)}.chg{font-size:13px;color:var(--muted);margin-left:6px}
table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:8px 6px;
border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
ul{margin:6px 0 0;padding-left:20px}li{margin:4px 0}
.list{list-style:none;padding:0;margin:0}.list li{margin:0;border-bottom:1px solid var(--line)}
.list li a{display:grid;grid-template-columns:110px 1fr 16px;gap:4px 14px;padding:14px 6px;border-radius:8px}
.list li a:hover{background:var(--card)}.arrow{color:var(--muted);align-self:center}.list .tk{font-weight:700;font-size:17px}
.list a{color:inherit;text-decoration:none}.list a:hover .tk{color:var(--accent)}.co{color:var(--muted);font-size:14px}
.tag{display:inline-block;font-size:12px;padding:1px 8px;border:1px solid var(--line);border-radius:999px;
color:var(--muted);margin-left:6px}.grp{margin:28px 0 4px;font-size:15px;color:var(--muted)}
@media (max-width:560px){.list li a{grid-template-columns:1fr 16px}.list li a>div{grid-column:1}}
.disclaimer{margin-top:40px;padding:14px;border:1px solid var(--line);border-radius:10px;font-size:13px;
color:var(--muted)}.scroll{overflow-x:auto}
"""


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def badge(rating: str) -> str:
    return f'<span class="badge {RATING_CLASS.get(rating, "hold")}">{esc(rating)}</span>'


def page(title: str, body: str, depth: int = 0) -> str:
    up = "../" * depth
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · {SITE_NAME}</title><meta name="description" content="Daily AI equity research briefs.">
<style>{CSS}</style></head><body><div class="wrap">
<header><a class="brand" href="{up}index.html">{SITE_NAME}</a>
<nav><a href="{up}index.html">Today</a><a href="{up}archive.html">Archive</a>
<a href="{up}performance.html">Track record</a><a href="{up}about.html">About</a></nav></header>
{body}
<div class="disclaimer"><strong>Disclaimer.</strong> {esc(DISCLAIMER)}</div>
</div></body></html>"""


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>" if items else '<p class="muted">—</p>'


def by_rating(calls: list[Call]) -> list[Call]:
    return sorted(calls, key=lambda c: (RATING_ORDER.get(c.rating, 9), c.ticker))


def day_body(day: str, calls: list[Call], previous: dict[str, str], depth: int,
             profiles: dict[str, dict] | None = None) -> str:
    """The day's ratings as one list, grouped from Buy at the top to Sell at the bottom."""
    up, profiles = "../" * depth, profiles or {}
    groups = []
    for rating in RATING_CLASS:
        mine = [c for c in by_rating(calls) if c.rating == rating]
        if not mine:
            continue
        rows = []
        for c in mine:
            prof = profiles.get(c.ticker, {})
            prev = previous.get(c.ticker)
            chg = f'<span class="chg">was {esc(prev)}</span>' if prev and prev != c.rating else ""
            name = prof.get("name") if prof.get("name") not in (None, "", c.ticker) else ""
            sector = f'<span class="tag">{esc(prof["sector"])}</span>' if prof.get("sector") else ""
            rows.append(f'<li><a href="{up}stocks/{esc(c.ticker)}.html"><span><span class="tk">{esc(c.ticker)}'
                        f'</span><br>{badge(c.rating)}</span><div><div class="co">{esc(name)}{sector}{chg}</div>'
                        f'{esc(c.points.get("headline"))}</div><span class="arrow">›</span></a></li>')
        groups.append(f'<h2 class="grp">{badge(rating)} {len(mine)}</h2><ul class="list">{"".join(rows)}</ul>')
    when = datetime.strptime(day, "%Y-%m-%d").strftime("%A, %B %-d, %Y")
    return (f"<h1>{esc(when)}</h1><p class='muted'>{len(calls)} stocks, from Buy to Sell · ratings from a "
            f"multi-agent AI research pipeline · <a href='{up}about.html'>how it works</a></p>"
            f"{''.join(groups)}")


def stock_body(ticker: str, latest: Call, history: list[dict], results: dict,
               profile: dict | None = None) -> str:
    p, prof = latest.points, profile or {}
    sections = [("Technical", "technical"), ("Fundamentals", "fundamentals"), ("News & macro", "news"),
                ("Bull case", "bull"), ("Bear case", "bear"), ("What would change the view", "watch")]
    cols = "".join(f'<div class="card"><h3>{t}</h3>{bullets(p.get(k, []))}</div>' for t, k in sections)
    rows = []
    for h in sorted(history, key=lambda r: r["day"], reverse=True):
        r = results.get((h["day"], ticker))
        perf = (f"{r['ret']:+.1%}", f"{r['excess']:+.1%}") if r else ("pending", "")
        rows.append(f"<tr><td><a href='../days/{h['day']}.html'>{h['day']}</a></td><td>{badge(h['rating'])}</td>"
                    f"<td>{perf[0]}</td><td>{perf[1]}</td></tr>")
    where = " · ".join(esc(x) for x in (prof.get("sector"), prof.get("industry")) if x)
    name = prof.get("name") if prof.get("name") not in (None, "", ticker) else ""
    intro = (f"<div class='card'><h3>About the company</h3><p class='co' style='margin:0 0 6px'>{where}</p>"
             f"<p style='margin:0'>{esc(prof['about'])}</p></div>") if prof.get("about") else (
             f"<p class='co'>{where}</p>" if where else "")
    return (f"<h1>{esc(ticker)} {badge(latest.rating)}</h1><p class='muted'>{esc(name)}"
            f"{' · ' if name else ''}Latest brief: {latest.day}</p>{intro}"
            f"<h2>Today's view</h2><p>{esc(p.get('headline'))}</p><div class='cols'>{cols}</div>"
            f"<h2>Rating history</h2><div class='scroll'><table><tr><th>Date</th><th>Rating</th>"
            f"<th>{HORIZON_DAYS}-day return</th><th>vs S&amp;P 500</th></tr>{''.join(rows)}</table></div>")


def performance_body(card: list[dict], results: list[dict]) -> str:
    if not card:
        return (f"<h1>Track record</h1><p>Each rating is scored by the stock's return over the next "
                f"{HORIZON_DAYS} trading days, against the S&amp;P 500 (SPY). The first scores appear "
                f"{HORIZON_DAYS} trading days after the first published ratings.</p>")
    def hit(c: dict) -> str:
        return "—" if c["hit"] is None else f"{c['hit']:.0%}"

    rows = "".join(f"<tr><td>{esc(c['group'])}</td><td>{c['n']}</td><td>{c['avg_ret']:+.2%}</td>"
                   f"<td>{c['avg_excess']:+.2%}</td><td>{hit(c)}</td></tr>"
                   for c in card)
    recent = "".join(f"<tr><td>{r['day']}</td><td><a href='stocks/{esc(r['ticker'])}.html'>{esc(r['ticker'])}</a>"
                     f"</td><td>{badge(r['rating'])}</td><td>{r['ret']:+.1%}</td><td>{r['excess']:+.1%}</td></tr>"
                     for r in sorted(results, key=lambda r: r["day"], reverse=True)[:60])
    return (f"<h1>Track record</h1><p class='muted'>Return from the close on the rating day to the close "
            f"{HORIZON_DAYS} trading days later; “vs S&amp;P 500” subtracts SPY over the same days. "
            f"“Right” means a bullish call beat SPY or a bearish call lagged it. Small samples are noisy.</p>"
            f"<div class='scroll'><table><tr><th>Ratings</th><th>Calls</th><th>Avg return</th>"
            f"<th>Avg vs S&amp;P 500</th><th>Right</th></tr>{rows}</table></div>"
            f"<h2>Recent scored calls</h2><div class='scroll'><table><tr><th>Date</th><th>Stock</th><th>Rating</th>"
            f"<th>Return</th><th>vs S&amp;P 500</th></tr>{recent}</table></div>")


ABOUT = f"""<h1>About</h1>
<p>{SITE_NAME} publishes one brief per stock each US trading day. A team of AI agents reads price data and
technical indicators, company filings and financial statements, and recent news and macro data. Bull and bear
researchers argue the case, a trader drafts a plan, risk reviewers challenge it, and a portfolio-manager agent
gives the final five-level rating: Buy, Overweight, Hold, Underweight or Sell.</p>
<p>The research uses the open-source TradingAgents framework. Briefs are generated automatically and are not
reviewed by a person before publishing. Every rating is scored on the Track record page so you can judge
how useful it has been.</p>"""


# --- build -------------------------------------------------------------------------------

def build(site: Path, ratings: list[dict], *, llm: Callable[[str], str] | None,
          reports_dir: Path = REPORTS_DIR, closes_fn=yf_closes, today: str | None = None,
          profile_fn: Callable[[str, dict | None], dict] = default_profile) -> list[str]:
    """Write the site; returns the pages written. Key points are cached under data/."""
    data_dir = site / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    history_path = data_dir / "ratings.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    known = {(r["day"], r["ticker"]) for r in history}
    for r in ratings:
        if (r["day"], r["ticker"]) not in known:
            history.append(r)
    history.sort(key=lambda r: (r["day"], r["ticker"]))

    calls: list[Call] = []
    for r in history:
        cache = data_dir / r["day"] / f"{r['ticker']}.json"
        if cache.exists():
            points = json.loads(cache.read_text())
        else:
            state = load_state(r["ticker"], r["day"], reports_dir)
            if state is None:
                continue
            points = key_points(state, r["ticker"], r["rating"], llm)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(points, indent=1))
        calls.append(Call(r["day"], r["ticker"], r["rating"], points))
    history = [r for r in history if any(c.day == r["day"] and c.ticker == r["ticker"] for c in calls)]
    history_path.write_text(json.dumps(history, indent=1))
    if not calls:
        return []

    # Company profiles, made once per ticker and cached.
    profiles_path = data_dir / "companies.json"
    profiles: dict[str, dict] = json.loads(profiles_path.read_text()) if profiles_path.exists() else {}
    for t in sorted({c.ticker for c in calls}):
        if not profiles.get(t, {}).get("about") or profile_fn is default_profile:
            latest = max(c.day for c in calls if c.ticker == t)
            fresh = profile_fn(t, load_state(t, latest, reports_dir))
            if fresh.get("about") or t not in profiles:
                profiles[t] = fresh
    profiles_path.write_text(json.dumps(profiles, indent=1))

    # Track record.
    results: list[dict] = []
    first = min(c.day for c in calls)
    try:
        end = (datetime.strptime(today or date.today().isoformat(), "%Y-%m-%d") + timedelta(days=1)).date()
        closes = closes_fn(sorted({c.ticker for c in calls} | {"SPY"}),
                           first, end.isoformat())
        results = forward_returns(history, closes)
    except Exception as exc:  # noqa: BLE001 - the site still builds without scores
        logger.warning("performance unavailable: %s", type(exc).__name__)
    by_key = {(r["day"], r["ticker"]): r for r in results}

    written = []

    def write(rel: str, text: str) -> None:
        p = site / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        written.append(rel)

    days = sorted({c.day for c in calls})
    previous: dict[str, str] = {}
    for d in days:
        todays = [c for c in calls if c.day == d]
        write(f"days/{d}.html", page(d, day_body(d, todays, previous, 1, profiles), 1))
        prev_snapshot = dict(previous)
        previous.update({c.ticker: c.rating for c in todays})
        if d == days[-1]:
            write("index.html", page("Today", day_body(d, todays, prev_snapshot, 0, profiles)))
    archive = "".join(f"<tr><td><a href='days/{d}.html'>{d}</a></td><td>"
                      f"{' '.join(f'{esc(c.ticker)} {badge(c.rating)}' for c in by_rating([c for c in calls if c.day == d]))}</td></tr>"
                      for d in reversed(days))
    write("archive.html", page("Archive", f"<h1>Archive</h1><div class='scroll'><table>{archive}</table></div>"))
    for t in sorted({c.ticker for c in calls}):
        mine = [c for c in calls if c.ticker == t]
        write(f"stocks/{t}.html", page(t, stock_body(t, mine[-1], [r for r in history if r["ticker"] == t],
                                                     by_key, profiles.get(t)), 1))
    write("performance.html", page("Track record", performance_body(scorecard(results), results)))
    write("about.html", page("About", ABOUT))
    write(".nojekyll", "")
    return written


def push(site: Path, message: str) -> str:
    git = os.environ.get("GIT", "git")
    cred = ["-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]
    run = lambda *a: subprocess.run([git, "-C", str(site), *a], capture_output=True, text=True)  # noqa: E731
    run("add", "-A")
    if run("diff", "--cached", "--quiet").returncode == 0:
        return "nothing to publish"
    run("-c", "user.name=AI Stock Daily", "-c", "user.email=noreply@users.noreply.github.com",
        "commit", "-q", "-m", message)
    out = subprocess.run([git, *cred, "-C", str(site), "push", "-q"], capture_output=True, text=True)
    return "pushed" if out.returncode == 0 else f"push failed: {out.stderr.strip()[:200]}"


def main(argv: list[str]) -> int:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=f"Build (and optionally publish) {SITE_NAME}")
    p.add_argument("--site", required=True, help="folder of the site (a git clone to --push)")
    p.add_argument("--ledger", default=str(DATA_DIR / "ledger.jsonl"))
    p.add_argument("--no-llm", action="store_true", help="skip the key-points call (summary only)")
    p.add_argument("--push", action="store_true", help="commit and push the site")
    args = p.parse_args(argv)

    site = Path(args.site).expanduser()
    llm = None if args.no_llm else make_llm()
    written = build(site, ledger_ratings(Path(args.ledger)), llm=llm, profile_fn=make_profile_fn(llm))
    print(f"wrote {len(written)} files to {site}")
    if args.push and written:
        print(push(site, f"Briefs for {date.today().isoformat()}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
