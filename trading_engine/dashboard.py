"""Local monitoring dashboard (read-only): account, ratings, analyst reports, orders and run logs.

    streamlit run trading_engine/dashboard.py        # then open http://localhost:8501

It only reads: the ledger, the saved analyst reports, the per-run logs in
~/.trading_engine/runs/ and the Alpaca paper account. Trading stays in Telegram,
where every manual action is previewed and confirmed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATA_DIR = Path(os.environ.get("TRADING_ENGINE_DIR", Path.home() / ".trading_engine"))
RUNS_DIR = DATA_DIR / "runs"
REPORTS_DIR = Path.home() / ".tradingagents" / "logs"
LEDGERS = {"正式账本（模拟盘）": DATA_DIR / "ledger.jsonl", "Dry run 账本": DATA_DIR / "ledger-dryrun.jsonl"}
RATING_ZH = {"Buy": "买入", "Overweight": "增持", "Hold": "持有", "Underweight": "减持",
             "Sell": "卖出", "REVIEW": "无法解读"}
NOISE = re.compile(r"HTTP Request:|httpx|urllib3")

st.set_page_config(page_title="交易引擎监控", page_icon="📈", layout="wide")


# --- data -----------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_ledger(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@st.cache_data(ttl=60)
def load_account() -> dict:
    from trading_engine.broker import AlpacaPaperBroker

    b = AlpacaPaperBroker()
    a = b.get_account()
    return {"equity": a.equity, "cash": a.cash,
            "holdings": [{"ticker": h.ticker, "qty": h.quantity, "value": h.market_value,
                          "entry": h.avg_entry_price} for h in a.holdings.values() if h.quantity],
            "market_open": b.is_market_open()}


def records(ledger: list[dict], kind: str) -> pd.DataFrame:
    rows = [{"at": r["at"], **r["data"]} for r in ledger if r.get("kind") == kind]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["at"] = pd.to_datetime(df["at"], utc=True).dt.tz_convert("America/Los_Angeles")
    return df


def kill_switch() -> str | None:
    p = DATA_DIR / "KILL_SWITCH"
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def scheduled_jobs() -> pd.DataFrame:
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    names = {"com.tradingengine.daily": "早上 7:00 交易", "com.tradingengine.summary": "下午 1:15 总结",
             "com.tradingengine.commands": "Telegram 监听"}
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2] in names:
            pid, code, label = parts
            running = pid.strip() != "-"
            rows.append({"任务": names[label], "状态": "🟢 运行中" if running else "⏰ 等待下次运行",
                         "上次退出码": code, "launchd 名称": label})
    return pd.DataFrame(rows)


def stop_book(ledger: list[dict]) -> dict[str, float]:
    stops: dict[str, float] = {}
    for r in ledger:
        if r.get("kind") == "stop":
            t, p = r["data"]["ticker"], r["data"]["stop_price"]
            if p is None:
                stops.pop(t, None)
            else:
                stops[t] = float(p)
    return stops


def summary_of(thesis: str) -> str:
    from trading_engine.runner import executive_summary
    return executive_summary(thesis or "", 600)


# --- page --------------------------------------------------------------------------

st.title("📈 交易引擎监控")
top = st.columns([3, 1])
which = top[0].radio("账本", list(LEDGERS), horizontal=True, label_visibility="collapsed")
if top[1].button("🔄 刷新"):
    st.cache_data.clear()
ledger = load_ledger(str(LEDGERS[which]))
dry = which.startswith("Dry")

tabs = st.tabs(["概览", "评级", "分析报告", "订单与止损", "运行日志", "任务状态"])

# 概览 -------------------------------------------------------------------------------
with tabs[0]:
    ks = kill_switch()
    if ks:
        st.error(f"🛑 紧急开关已打开：{ks}　（Telegram 发 /resume 恢复）")
    try:
        acct = load_account()
    except Exception as exc:  # noqa: BLE001
        acct = None
        st.warning(f"读取 Alpaca 账户失败：{type(exc).__name__}")
    eq = records(ledger, "equity")
    if acct:
        from trading_engine.config import RiskLimits
        lim = RiskLimits()
        peak = max([acct["equity"], *(eq["equity"].tolist() if not eq.empty and not dry else [])])
        dd = 0.0 if peak <= 0 else (peak - acct["equity"]) / peak
        invested = sum(h["value"] for h in acct["holdings"])
        c = st.columns(5)
        c[0].metric("账户净值", f"${acct['equity']:,.0f}")
        c[1].metric("现金", f"${acct['cash']:,.0f}")
        c[2].metric("总仓位", f"{invested / acct['equity']:.1%}" if acct["equity"] else "-",
                    help=f"上限 {lim.max_gross_exposure:.0%}")
        c[3].metric("从高点回撤", f"{dd:.1%}", help=f"到 {lim.max_drawdown:.0%} 停止买入")
        c[4].metric("美股", "开盘中" if acct["market_open"] else "已收盘")

        st.subheader("持仓")
        stops = stop_book(load_ledger(str(LEDGERS["正式账本（模拟盘）"])))
        if acct["holdings"]:
            rows = []
            for h in sorted(acct["holdings"], key=lambda h: -h["value"]):
                cost = (h["entry"] or 0) * h["qty"]
                rows.append({"股票": h["ticker"], "股数": h["qty"], "市值": h["value"],
                             "占比": h["value"] / acct["equity"], "成本价": h["entry"],
                             "浮动盈亏": h["value"] - cost if h["entry"] else None,
                             "止损价": stops.get(h["ticker"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={"市值": st.column_config.NumberColumn(format="$%.0f"),
                                        "占比": st.column_config.NumberColumn(format="percent"),
                                        "成本价": st.column_config.NumberColumn(format="$%.2f"),
                                        "浮动盈亏": st.column_config.NumberColumn(format="$%.0f"),
                                        "止损价": st.column_config.NumberColumn(format="$%.2f")})
        else:
            st.caption("目前没有持仓。")

    st.subheader("净值曲线")
    if eq.empty:
        st.caption("还没有净值记录：每次运行都会记一笔。")
    else:
        st.line_chart(eq.set_index("at")["equity"], height=260)

# 评级 -------------------------------------------------------------------------------
with tabs[1]:
    dec = records(ledger, "decision")
    if dec.empty:
        st.caption("还没有评级记录。")
    else:
        risk = records(ledger, "risk")
        orders = records(ledger, "order")
        latest_day = dec["trade_date"].max()
        days = sorted(dec["trade_date"].unique(), reverse=True)
        day = st.selectbox("日期", days, index=days.index(latest_day))
        today = dec[dec["trade_date"] == day].drop_duplicates("ticker", keep="last")
        rows = []
        for _, d in today.iterrows():
            r = risk[risk["decision_id"] == d["decision_id"]].tail(1) if not risk.empty else pd.DataFrame()
            o = orders[orders["decision_id"] == d["decision_id"]].tail(1) if not orders.empty else pd.DataFrame()
            if not o.empty:
                action = f"已下单 {o.iloc[0]['side']} {o.iloc[0]['quantity']:g}"
            elif not r.empty and not r.iloc[0]["approved"]:
                action = "被风控拦下：" + "; ".join(r.iloc[0]["reasons"])
            elif not r.empty and r.iloc[0].get("order") is None:
                action = "不操作"
            else:
                action = "模拟（未下单）" if dry else "-"
            ai = d.get("ai_signal") if d.get("overridden") is True else None
            rows.append({"股票": d["ticker"], "评级": f"{d['signal']}（{RATING_ZH.get(d['signal'], '')}）",
                         "AI 原评级": ai or "", "目标仓位": d.get("target_weight"),
                         "止损": d.get("stop_loss"), "操作": action, "理由": summary_of(d.get("thesis", ""))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"目标仓位": st.column_config.NumberColumn(format="percent"),
                                    "理由": st.column_config.TextColumn(width="large")})

        st.subheader("评级历史")
        st.caption("用来观察 AI 是否经常改变主意。")
        hist = dec.drop_duplicates(["trade_date", "ticker"], keep="last")
        pivot = hist.pivot(index="trade_date", columns="ticker", values="signal").sort_index(ascending=False)
        st.dataframe(pivot.map(lambda s: RATING_ZH.get(s, s) if isinstance(s, str) else ""),
                     use_container_width=True)

# 分析报告 ------------------------------------------------------------------------
with tabs[2]:
    files = sorted(REPORTS_DIR.glob("*/TradingAgentsStrategy_logs/full_states_log_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        st.caption("还没有分析报告。")
    else:
        tickers = sorted({f.parts[-3] for f in files})
        c = st.columns(2)
        t = c[0].selectbox("股票", tickers, index=tickers.index(files[0].parts[-3]))  # newest first
        dates = [f.stem.replace("full_states_log_", "") for f in files if f.parts[-3] == t]
        d = c[1].selectbox("日期", dates)
        state = json.loads((REPORTS_DIR / t / "TradingAgentsStrategy_logs" / f"full_states_log_{d}.json")
                           .read_text(encoding="utf-8"))
        rating = state.get("final_rating") or ""
        st.markdown(f"### {t} · {d} · 最终评级：**{rating}（{RATING_ZH.get(rating, '')}）**")
        sections = [("最终决定", state.get("final_trade_decision")),
                    ("市场/技术分析师", state.get("market_report")),
                    ("新闻分析师", state.get("news_report")),
                    ("基本面分析师", state.get("fundamentals_report")),
                    ("情绪分析师", state.get("sentiment_report")),
                    ("看多研究员", (state.get("investment_debate_state") or {}).get("bull_history")),
                    ("看空研究员", (state.get("investment_debate_state") or {}).get("bear_history")),
                    ("研究经理建议", state.get("investment_plan")),
                    ("交易员方案", state.get("trader_investment_plan")),
                    ("风险讨论", (state.get("risk_debate_state") or {}).get("history"))]
        sections = [(name, text) for name, text in sections if text]
        for tab, (_name, text) in zip(st.tabs([n for n, _ in sections]), sections, strict=True):
            tab.markdown(text)
        with st.expander("运行设置"):
            st.json(state.get("run_settings", {}))

# 订单与止损 ------------------------------------------------------------------------
with tabs[3]:
    orders = records(ledger, "order")
    if orders.empty:
        st.caption("还没有订单。")
    else:
        orders["来源"] = orders.get("source", pd.Series(index=orders.index)).fillna("AI").replace({"manual": "手动"})
        cols = [c for c in ["at", "trade_date", "ticker", "side", "quantity", "status", "est_price",
                            "filled_avg_price", "来源"] if c in orders]
        st.dataframe(orders[cols].sort_values("at", ascending=False), hide_index=True, use_container_width=True)
    st.subheader("止损变动")
    stops = records(ledger, "stop")
    if stops.empty:
        st.caption("还没有止损记录。")
    else:
        st.dataframe(stops[["at", "ticker", "stop_price", "previous", "source"]].sort_values("at", ascending=False),
                     hide_index=True, use_container_width=True)
    events = pd.concat([records(ledger, k).assign(类型=k) for k in
                        ("stop_filled", "take_profit", "manual", "override", "error")
                        if not records(ledger, k).empty], ignore_index=True) if ledger else pd.DataFrame()
    if not events.empty:
        st.subheader("其他事件（止损成交、止盈、手动操作、评级覆盖、错误）")
        st.dataframe(events.sort_values("at", ascending=False), hide_index=True, use_container_width=True)

# 运行日志 --------------------------------------------------------------------------
with tabs[4]:
    logs = sorted(RUNS_DIR.glob("*.log"), reverse=True)
    if not logs:
        st.caption("还没有运行日志。")
    else:
        c = st.columns([3, 1])
        pick = c[0].selectbox("哪一次运行", logs, format_func=lambda p: p.stem.replace("_", " "))
        hide = c[1].toggle("隐藏 HTTP 请求行", value=True)
        text = pick.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        calls = sum(1 for line in lines if "POST https://api.openai.com" in line)
        result = [line for line in lines if re.match(r"^(Cycle |  [A-Z.]+ +\S|ledger head|Daily summary)", line)]
        errors = [line for line in lines if " ERROR " in line or line.startswith("Traceback")]
        m = st.columns(3)
        m[0].metric("ChatGPT 调用次数", calls)
        m[1].metric("错误", len(errors))
        m[2].metric("文件大小", f"{pick.stat().st_size / 1024:.0f} KB")
        if result:
            st.markdown("**结果**")
            st.code("\n".join(result), language=None)
        st.markdown("**完整日志**")
        shown = [line for line in lines if not (hide and NOISE.search(line))]
        st.code("\n".join(shown[-3000:]), language=None)
        st.download_button("下载这份日志", text, file_name=pick.name)

# 任务状态 --------------------------------------------------------------------------
with tabs[5]:
    jobs = scheduled_jobs()
    if jobs.empty:
        st.warning("没有找到定时任务（launchd 里没有 com.tradingengine.*）。")
    else:
        st.dataframe(jobs, hide_index=True, use_container_width=True)
    ks = kill_switch()
    st.markdown(f"**紧急开关：** {'🛑 打开 — ' + ks if ks else '✅ 关闭'}")
    for name in ("com.tradingengine.commands.log", "commands.log"):
        p = DATA_DIR / name
        if p.exists():
            st.markdown(f"**Telegram 监听日志**（{name}，最后 40 行）")
            st.code("\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]), language=None)
            break
    st.caption(f"页面更新时间：{datetime.now():%Y-%m-%d %H:%M:%S}（数据缓存 60 秒，点“刷新”立即更新）")
