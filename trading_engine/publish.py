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
# Public wording: the engine's Overweight/Underweight are broker jargon, so readers see these instead.
LABELS = {"Overweight": "Moderate Buy", "Underweight": "Moderate Sell"}
_JARGON = re.compile(r"\b(Overweight|Underweight)\b")


def label(rating: str) -> str:
    return LABELS.get(rating, rating)


def plain(text: str) -> str:
    """Swap the jargon in model-written text ("rated Overweight") for the public labels."""
    return _JARGON.sub(lambda m: LABELS[m.group(1)], str(text or ""))

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


LEVELS = ("Strong", "Good", "Mixed", "Weak", "Poor")
LEVEL_CLASS = dict(zip(LEVELS, ("buy", "ow", "hold", "uw", "sell"), strict=True))
DIMENSIONS = (("technical", "Technical"), ("fundamentals", "Fundamentals"), ("news", "News"))

SCORES_PROMPT = """Grade {ticker} on three separate dimensions from the research notes below.
Use exactly one of: Strong, Good, Mixed, Weak, Poor.
- technical: price trend, momentum and moving averages (Strong = clear uptrend, Poor = clear downtrend)
- fundamentals: growth, profitability, cash flow, balance sheet and valuation together
- news: whether recent company news and macro backdrop are favorable for the stock
Grade each dimension on its own evidence, not on the final rating.
Return only JSON: {{"technical": "...", "fundamentals": "...", "news": "..."}}

=== Technical ===
{market}
=== Fundamentals ===
{fundamentals}
=== News and macro ===
{news}
"""


def dimension_scores(state: dict, ticker: str, llm: Callable[[str], str] | None) -> dict:
    """Strong/Good/Mixed/Weak/Poor for technical, fundamentals and news; empty without a model."""
    if llm is None:
        return {}
    try:
        raw = llm(SCORES_PROMPT.format(ticker=ticker, market=str(state.get("market_report", ""))[:6000],
                                       fundamentals=str(state.get("fundamentals_report", ""))[:8000],
                                       news=str(state.get("news_report", ""))[:6000]))
        m = re.search(r"\{.*\}", raw, re.S)
        got = json.loads(m.group(0)) if m else {}
    except Exception as exc:  # noqa: BLE001 - the site shows a dash instead
        logger.warning("scores for %s failed: %s", ticker, type(exc).__name__)
        return {}
    out = {}
    for key, _ in DIMENSIONS:
        level = str(got.get(key, "")).strip().capitalize()
        if level in LEVELS:
            out[key] = level
    return out


def level_tag(level: str | None) -> str:
    if not level:
        return '<span class="lv muted">—</span>'
    return f'<span class="lv t-{LEVEL_CLASS[level]}">{esc(level)}</span>'


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
    for name, group in (("Bullish (Buy, Moderate Buy)", BULLISH), ("Hold", {"Hold"}),
                        ("Bearish (Moderate Sell, Sell)", BEARISH)):
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
:root{--bg:#ffffff;--surface:#ffffff;--secondary:#f5f5f5;--fg:#0a0a0a;--muted:#737373;--faint:#a3a3a3;
--line:#e5e5e5;--input:#e5e5e5;--ring:#a3a3a3;--accent:#0a0a0a;--radius:10px;
--buy:#15803d;--buy-bar:#16a34a;--ow:#4d7c0f;--ow-bar:#65a30d;--hold:#525252;--hold-bar:#a3a3a3;
--uw:#b45309;--uw-bar:#f59e0b;--sell:#dc2626;--sell-bar:#dc2626;
--shadow:0 1px 3px 0 rgb(0 0 0/.06),0 1px 2px -1px rgb(0 0 0/.06);--shadow-xs:0 1px 2px 0 rgb(0 0 0/.05)}
@media (prefers-color-scheme:dark){:root{--bg:#0a0a0a;--surface:#171717;--secondary:#262626;--fg:#fafafa;
--muted:#a3a3a3;--faint:#737373;--line:rgb(255 255 255/.1);--input:rgb(255 255 255/.15);--ring:#737373;--accent:#fafafa;
--buy:#4ade80;--ow:#8fbf7a;--hold:#d4d4d4;--hold-bar:#737373;--uw:#fbbf24;--sell:#f87171;--shadow:none;--shadow-xs:none}}
*{box-sizing:border-box;border-color:var(--line)}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 Geist,ui-sans-serif,system-ui,-apple-system,
"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased;font-feature-settings:"rlig" 1,"calt" 1}
::selection{background:var(--fg);color:var(--bg)}
a{color:var(--fg);text-decoration:underline;text-underline-offset:4px;text-decoration-color:var(--line)}
a:hover{text-decoration-color:var(--fg)}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px}
.top{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 92%,transparent);
backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.bar{display:flex;align-items:center;justify-content:space-between;gap:16px;height:56px}
.brand{display:flex;align-items:center;gap:8px;font-weight:600;font-size:15px;color:var(--fg);text-decoration:none}
.logo{width:24px;height:24px;border-radius:6px;background:var(--fg);color:var(--bg);display:grid;place-items:center;
font-size:10.5px;font-weight:700;letter-spacing:.02em}
nav{display:flex;gap:2px;overflow-x:auto}nav a{color:var(--muted);font-weight:500;font-size:14px;padding:6px 12px;
border-radius:8px;white-space:nowrap;text-decoration:none;transition:color .15s,background .15s}
nav a:hover{color:var(--fg);background:var(--secondary)}nav a.on{color:var(--fg);background:var(--secondary)}
main{padding:40px 0 24px}
h1{font-size:30px;line-height:1.2;letter-spacing:-.025em;margin:0;font-weight:600}
h2{font-size:18px;letter-spacing:-.015em;font-weight:600;margin:40px 0 4px}
h3{font-size:14px;margin:0 0 10px;font-weight:600;line-height:1}
.eyebrow{font-size:13px;font-weight:500;color:var(--muted);margin:0 0 6px}
.lede{color:var(--muted);font-size:15px;margin:8px 0 0}.muted{color:var(--muted)}
.pill{display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:500;
white-space:nowrap;border:1px solid var(--line);color:var(--fg);background:var(--surface)}
.pill::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--hold-bar)}
.p-buy::before{background:var(--buy-bar)}.p-ow::before{background:var(--ow-bar)}.p-uw::before{background:var(--uw-bar)}
.p-sell::before{background:var(--sell-bar)}
.chip{display:inline-block;font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:6px;padding:1px 7px;
margin-right:6px;white-space:nowrap}
.chg{font-size:12px;color:var(--muted);white-space:nowrap}
.dist{margin-top:28px;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px 24px;
box-shadow:var(--shadow)}.dbar{display:flex;height:8px;border-radius:999px;overflow:hidden;gap:2px;background:var(--secondary)}
.dbar span{display:block}.b-buy{background:var(--buy-bar)}.b-ow{background:var(--ow-bar)}.b-hold{background:var(--hold-bar)}
.b-uw{background:var(--uw-bar)}.b-sell{background:var(--sell-bar)}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:14px;font-size:13px;color:var(--muted)}
.legend b{color:var(--fg);font-weight:500;margin-left:6px;font-variant-numeric:tabular-nums}
.legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;vertical-align:0}
.row.head{padding:0 16px;height:40px;background:var(--surface);font-size:14px;font-weight:500;color:var(--muted);
border-bottom:1px solid var(--line)}.row.head:hover{background:var(--surface)}
.sort{all:unset;cursor:pointer;white-space:nowrap;display:inline-flex;align-items:center;gap:6px;font:inherit;color:inherit;
border-radius:6px;padding:4px 8px;margin-left:-8px;transition:background .15s,color .15s}
.sort:hover{color:var(--fg);background:var(--secondary)}.sort:focus-visible{outline:none;box-shadow:0 0 0 3px
color-mix(in srgb,var(--ring) 50%,transparent)}.sort[data-dir]{color:var(--fg)}
.ar{display:inline-flex;flex-direction:column;justify-content:center;gap:2px}
.ar::before,.ar::after{content:"";border-left:3.5px solid transparent;border-right:3.5px solid transparent;opacity:.4}
.ar::before{border-bottom:4.5px solid currentColor}.ar::after{border-top:4.5px solid currentColor}
.sort[data-dir=asc] .ar::before,.sort[data-dir=desc] .ar::after{opacity:1}
.row.head .rt.sort{display:flex;flex-direction:row;justify-content:flex-start;margin:0;padding:4px 8px 4px 16px;
background:none;border:0;font-weight:500}
.row.head .rt.sort:hover{background:var(--secondary)}
.msort{display:none;font-size:14px;color:var(--muted);margin-bottom:12px}
.msort select{font:inherit;color:var(--fg);background:var(--surface);border:1px solid var(--input);border-radius:8px;
height:36px;padding:0 10px;margin-left:8px;box-shadow:var(--shadow-xs)}
.sec{font-size:12px;color:var(--muted);margin-top:2px}
.scs{display:grid;grid-template-columns:100px 130px 80px;gap:0 6px;align-items:center}.lbl{display:none}
.lv{font-weight:400;font-size:14px}.note{font-size:13px;color:var(--muted);margin:14px 2px 0}
.ch{display:flex;justify-content:space-between;align-items:center;gap:10px}.why{font-size:14px;line-height:1.55}
.rt{font-weight:600;font-size:14px;align-self:stretch;display:flex;flex-direction:column;justify-content:center;
border-left:1px solid var(--line);border-right:1px solid var(--line);padding-left:16px;margin:-14px 0;
background:color-mix(in srgb,var(--secondary) 55%,transparent)}
.rt .chg{font-weight:500;font-size:12px;margin-top:2px}.chg.up{color:var(--buy)}.chg.down{color:var(--sell)}
.t-buy{color:var(--buy)}.t-ow{color:var(--ow)}.t-hold{color:var(--hold)}.t-uw{color:var(--uw)}.t-sell{color:var(--sell)}
.rows{background:var(--surface);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow)}
.row{display:grid;grid-template-columns:170px 328px 150px 1fr 100px;gap:6px 18px;align-items:center;padding:14px 16px;
color:var(--fg);border-top:1px solid var(--line);text-decoration:none;transition:background .15s}
.row:nth-child(2){border-top:0}.row:hover{background:color-mix(in srgb,var(--secondary) 50%,transparent)}
.go{font-size:13px;font-weight:500;color:var(--muted);white-space:nowrap;text-align:right;transition:color .15s}
.go i{font-style:normal;display:inline-block;transition:transform .15s}
.row:hover .go{color:var(--fg)}.row:hover .go i{transform:translateX(3px)}
.hint{font-size:14px;color:var(--muted);margin:0 0 16px}
.tk{font-weight:600;font-size:15px}.nm{font-size:13px;color:var(--muted);line-height:1.4;margin-top:2px}
.back{display:inline-block;font-size:14px;color:var(--muted);margin-bottom:20px;text-decoration:none}
.back:hover{color:var(--fg)}
.shead{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:12px 24px}
.shead h1{display:flex;align-items:center;gap:12px;font-size:36px}.shead h1 .pill{font-size:13px;padding:3px 10px}
.shead .nm{font-size:15px;margin-top:6px}.asof{font-size:13px;color:var(--muted)}
.view{margin-top:28px;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px 24px;
box-shadow:var(--shadow);position:relative;overflow:hidden}
.view::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--hold-bar)}
.view p{margin:0;font-size:17px;line-height:1.55;letter-spacing:-.01em}.view .eyebrow{margin-bottom:6px;font-size:13px}
.v-buy::before{background:var(--buy-bar)}.v-ow::before{background:var(--ow-bar)}.v-uw::before{background:var(--uw-bar)}
.v-sell::before{background:var(--sell-bar)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px 24px;box-shadow:var(--shadow)}
h2+.card,h2+.grid3,h2+.tbl,h2+.rows{margin-top:16px}
.about p{margin:0;line-height:1.65}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-top:16px}
.card ul{margin:0;padding-left:18px}.card li{margin:8px 0;font-size:14px;line-height:1.55}
.card li::marker{color:var(--faint)}.bull h3 span{color:var(--buy)}.bear h3 span{color:var(--sell)}.watch{margin-top:16px}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:10px 16px;border-top:1px solid var(--line)}
th{color:var(--muted);font-weight:500;height:40px;border-top:0}
tr:hover td{background:color-mix(in srgb,var(--secondary) 50%,transparent)}
.tbl{background:var(--surface);border:1px solid var(--line);border-radius:14px;overflow:auto;box-shadow:var(--shadow)}
.pos{color:var(--buy)}.neg{color:var(--sell)}
.prose{max-width:680px}.prose p{font-size:15px;line-height:1.75;color:color-mix(in srgb,var(--fg) 85%,var(--muted))}
footer{border-top:1px solid var(--line);margin-top:64px;padding:24px 0 48px;font-size:13px;color:var(--muted);line-height:1.6}
footer .wrap>div{max-width:760px}
@media (max-width:820px){.grid3{grid-template-columns:1fr}.grid2{grid-template-columns:1fr}}
@media (max-width:640px){.wrap{padding:0 16px}h1{font-size:26px}.shead h1{font-size:30px}main{padding-top:28px}
.row{grid-template-columns:1fr auto;padding:14px 16px;gap:8px 12px}.row:first-of-type{border-top:0}
.row .rt{grid-column:2;grid-row:1;text-align:right;border-left:0;border-right:0;padding-left:0;margin:0;background:none}
.row .rt .lbl{display:block;font-weight:400;margin:0}.row .scs{grid-column:1/3;display:flex;flex-wrap:wrap;gap:6px 16px}
.row .lbl{display:inline;color:var(--muted);font-size:12px;margin-right:5px}
.row .why{grid-column:1/3}.row.head{display:none}.msort{display:block}
.row .go{grid-column:1/3;text-align:left}.hint{display:none}
.bar{height:auto;padding-top:12px;padding-bottom:12px;flex-direction:column;align-items:flex-start;gap:8px}nav{max-width:100%;margin-left:-12px}}
"""

NAV = [("Today", "index.html"), ("Archive", "archive.html"), ("Track record", "performance.html"),
       ("About", "about.html")]


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def badge(rating: str) -> str:
    return f'<span class="pill p-{RATING_CLASS.get(rating, "hold")}">{esc(label(rating))}</span>'


def page(title: str, body: str, depth: int = 0) -> str:
    up = "../" * depth
    nav = "".join(f'<a href="{up}{href}"{" class=on" if name == title else ""}>{name}</a>' for name, href in NAV)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · {SITE_NAME}</title><meta name="description" content="Daily AI equity research briefs.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<header class="top"><div class="wrap bar"><a class="brand" href="{up}index.html"><span class="logo">AI</span>{SITE_NAME}</a>
<nav>{nav}</nav></div></header>
<main class="wrap">{body}</main>
<footer><div class="wrap"><div><strong>Disclaimer.</strong> {esc(DISCLAIMER)}</div></div></footer>
</body></html>"""


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(plain(i))}</li>" for i in items) + "</ul>" if items else '<p class="muted">—</p>'


def by_rating(calls: list[Call]) -> list[Call]:
    return sorted(calls, key=lambda c: (RATING_ORDER.get(c.rating, 9), c.ticker))


def display_name(prof: dict, ticker: str) -> str:
    name = prof.get("name") or ""
    return "" if name == ticker else name


# Click a column header (or pick from the phone menu) to sort; click again to reverse.
# Lower numbers are better (Buy, Strong); stocks without a grade always go last.
SORT_JS = """<script>
document.querySelectorAll('.rows').forEach(function(t){
  var hs=t.querySelectorAll('.sort'), menu=t.previousElementSibling.querySelector('select');
  function sort(k,asc){
    hs.forEach(function(x){x.removeAttribute('data-dir');x.removeAttribute('aria-sort');
      if(x.dataset.key===k){x.dataset.dir=asc?'asc':'desc';x.setAttribute('aria-sort',asc?'ascending':'descending');}});
    var rows=Array.prototype.slice.call(t.querySelectorAll('a.row'));
    rows.sort(function(a,b){
      var x=a.dataset[k], y=b.dataset[k], c;
      if(k==='stock'){c=x.localeCompare(y);}else{
        if(x==='9'||y==='9'){return (x==='9')-(y==='9')||a.dataset.stock.localeCompare(b.dataset.stock);}
        c=x-y;}
      return (asc?c:-c)||a.dataset.rating-b.dataset.rating||a.dataset.stock.localeCompare(b.dataset.stock);});
    rows.forEach(function(r){t.appendChild(r);});
    if(menu){menu.value=k;}
  }
  hs.forEach(function(h){h.addEventListener('click',function(){sort(h.dataset.key,h.dataset.dir!=='asc');});});
  if(menu){menu.addEventListener('change',function(){sort(menu.value,true);});}
});
</script>"""


def trend(rating: str, prev: str | None) -> str:
    """Arrow against the previous day's rating: up if it improved toward Buy, down if it fell; nothing without one."""
    if not prev or prev not in RATING_ORDER:
        return ""
    move = RATING_ORDER[prev] - RATING_ORDER.get(rating, RATING_ORDER[prev])
    if move > 0:
        return f'<div class="chg up" title="Up from {esc(label(prev))}">▲ from {esc(label(prev))}</div>'
    if move < 0:
        return f'<div class="chg down" title="Down from {esc(label(prev))}">▼ from {esc(label(prev))}</div>'
    return '<div class="chg" title="Same as the previous day">– unchanged</div>'


def day_body(day: str, calls: list[Call], previous: dict[str, str], depth: int,
             profiles: dict[str, dict] | None = None) -> str:
    """The day's ratings as one table, sorted from Buy at the top to Sell at the bottom."""
    up, profiles = "../" * depth, profiles or {}
    counts = {r: sum(c.rating == r for c in calls) for r in RATING_CLASS}
    segs = "".join(f'<span class="b-{RATING_CLASS[r]}" style="flex:{n}"></span>' for r, n in counts.items() if n)
    legend = "".join(f'<span><i class="b-{RATING_CLASS[r]}"></i>{label(r)}<b>{n}</b></span>' for r, n in counts.items())
    rows = []
    for c in by_rating(calls):
        prof = profiles.get(c.ticker, {})
        prev = previous.get(c.ticker)
        chg = trend(c.rating, prev)
        scores = c.points.get("scores") or {}
        cells = "".join(f'<div class="sc"><span class="lbl">{label}</span>{level_tag(scores.get(key))}</div>'
                        for key, label in DIMENSIONS)
        rank = {key: LEVELS.index(scores[key]) if scores.get(key) in LEVELS else 9 for key, _ in DIMENSIONS}
        data = (f'data-stock="{esc(c.ticker)}" data-rating="{RATING_ORDER.get(c.rating, 9)}" '
                + " ".join(f'data-{key}="{v}"' for key, v in rank.items()))
        rows.append(f'<a class="row" {data} href="{up}stocks/{esc(c.ticker)}.html">'
                    f'<div><div class="tk">{esc(c.ticker)}</div><div class="nm">{esc(display_name(prof, c.ticker))}'
                    f'</div><div class="sec">{esc(prof.get("sector", ""))}</div></div>'
                    f'<div class="scs">{cells}</div>'
                    f'<div class="rt t-{RATING_CLASS[c.rating]}"><span class="lbl">Final rating</span>'
                    f'{esc(label(c.rating))}{chg}</div>'
                    f'<div class="why">{esc(plain(c.points.get("headline")))}</div><span class="go">Full report <i>→</i></span></a>')
    def sorter(key: str, label: str, cls: str = "", on: bool = False) -> str:
        state = ' data-dir="asc" aria-sort="ascending"' if on else ""
        return f'<button type="button" class="sort {cls}" data-key="{key}"{state}>{label}<span class="ar"></span></button>'

    options = "".join(f'<option value="{k}">{lbl}</option>' for k, lbl in
                      (("rating", "Final rating"), ("technical", "Technical"), ("fundamentals", "Fundamentals"),
                       ("news", "News"), ("stock", "Stock A–Z")))
    table = (f'<label class="msort">Sort by <select>{options}</select></label>'
             f'<div class="rows"><div class="row head">{sorter("stock", "Stock")}<div class="scs">'
             f'{sorter("technical", "Technical")}{sorter("fundamentals", "Fundamentals")}{sorter("news", "News")}'
             f'</div>{sorter("rating", "Final rating", "rt", on=True)}'
             f'<div>Summary</div><span></span></div>{"".join(rows)}</div>{SORT_JS}'
             f'<p class="note">Technical, Fundamentals and News are graded separately by the AI on a five-step '
             f'scale: {" · ".join(level_tag(lv) for lv in LEVELS)}.</p>')
    d = datetime.strptime(day, "%Y-%m-%d")
    return (f'<p class="eyebrow">Daily brief · {d.strftime("%A")}</p><h1>{d.strftime("%B %-d, %Y")}</h1>'
            f'<p class="lede">AI ratings for {len(calls)} US stocks, ranked from Buy to Sell. '
            f'<a href="{up}about.html">How it works</a></p>'
            f'<div class="dist"><div class="dbar">{segs}</div><div class="legend">{legend}</div></div>'
            f'<h2>All ratings</h2><p class="hint">Click a row for the full report · '
            f'click a column header to sort</p>{table}')


def stock_body(ticker: str, latest: Call, history: list[dict], results: dict,
               profile: dict | None = None) -> str:
    p, prof = latest.points, profile or {}
    cls = RATING_CLASS.get(latest.rating, "hold")
    where = " · ".join(esc(x) for x in (prof.get("sector"), prof.get("industry")) if x)
    about = (f'<h2>About the company</h2><div class="card about"><p>{esc(prof["about"])}</p></div>'
             if prof.get("about") else "")
    scores = p.get("scores") or {}

    def card(title: str, key: str, extra: str = "") -> str:
        grade = level_tag(scores[key]) if key in scores else ""
        return (f'<div class="card {extra}"><h3 class="ch"><span>{title}</span>{grade}</h3>'
                f'{bullets(p.get(key, []))}</div>')
    rows = []
    for h in sorted(history, key=lambda r: r["day"], reverse=True):
        r = results.get((h["day"], ticker))
        if r:
            perf = "".join(f'<td class="{"pos" if v >= 0 else "neg"}">{v:+.1%}</td>' for v in (r["ret"], r["excess"]))
        else:
            perf = '<td class="muted">pending</td><td></td>'
        rows.append(f"<tr><td><a href='../days/{h['day']}.html'>{h['day']}</a></td><td>{badge(h['rating'])}</td>"
                    f"{perf}</tr>")
    return (f'<a class="back" href="../index.html">← Today’s ratings</a>'
            f'<div class="shead"><div>{f"<p class=eyebrow>{where}</p>" if where else ""}'
            f'<h1>{esc(ticker)} {badge(latest.rating)}</h1>'
            f'<div class="nm">{esc(display_name(prof, ticker))}</div></div>'
            f'<div class="asof">Brief for {latest.day}</div></div>'
            f'<div class="view v-{cls}"><p class="eyebrow">Today’s view</p><p>{esc(plain(p.get("headline")))}</p></div>'
            f'{about}<h2>The analysis</h2><div class="grid3">{card("Technical", "technical")}'
            f'{card("Fundamentals", "fundamentals")}{card("News &amp; macro", "news")}</div>'
            f'<div class="grid2">{card("Bull case", "bull", "bull")}{card("Bear case", "bear", "bear")}</div>'
            f'{card("What would change the view", "watch", "watch")}'
            f'<h2>Rating history</h2><div class="tbl"><table><tr><th>Date</th><th>Rating</th>'
            f'<th>{HORIZON_DAYS}-day return</th><th>vs S&amp;P 500</th></tr>{"".join(rows)}</table></div>')


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
            f"<div class='tbl'><table><tr><th>Ratings</th><th>Calls</th><th>Avg return</th>"
            f"<th>Avg vs S&amp;P 500</th><th>Right</th></tr>{rows}</table></div>"
            f"<h2>Recent scored calls</h2><div class='tbl'><table><tr><th>Date</th><th>Stock</th><th>Rating</th>"
            f"<th>Return</th><th>vs S&amp;P 500</th></tr>{recent}</table></div>")


ABOUT = f"""<div class="prose"><p class="eyebrow">About</p><h1>How {SITE_NAME} works</h1>
<p>{SITE_NAME} publishes one brief per stock each US trading day. A team of AI agents reads price data and
technical indicators, company filings and financial statements, and recent news and macro data. Bull and bear
researchers argue the case, a trader drafts a plan, risk reviewers challenge it, and a portfolio-manager agent
gives the final five-level rating: Buy, Moderate Buy, Hold, Moderate Sell or Sell.</p>
<p>The research uses the open-source TradingAgents framework. Briefs are generated automatically and are not
reviewed by a person before publishing. Every rating is scored on the Track record page so you can judge
how useful it has been.</p></div>"""


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
        points = json.loads(cache.read_text()) if cache.exists() else None
        state = None
        if points is None or ("scores" not in points and llm is not None):
            state = load_state(r["ticker"], r["day"], reports_dir)
            if state is None and points is None:
                continue
        if points is None:
            points = key_points(state, r["ticker"], r["rating"], llm)
        if "scores" not in points and state is not None and llm is not None:
            points["scores"] = dimension_scores(state, r["ticker"], llm)
        if state is not None:
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
    write("archive.html", page("Archive", f"<p class='eyebrow'>Archive</p><h1>Every daily brief</h1><h2>By date</h2><div class='tbl'><table>{archive}</table></div>"))
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
