"""Your own orders and rating overrides, sent from Telegram.

Every manual action is a two-step: ``preview_*`` describes it and lists what the
risk rules would say, and nothing reaches the broker until you confirm and
``execute`` runs. The rules only warn here; the decision is yours. Two things
are refused outright because the engine cannot hold them: a short position
(selling more than you hold) and a fractional or non-positive quantity.

Manual orders go into the ledger like the engine's own (``order`` records, with
``source: manual``), so fills, stops, the daily summary and the drawdown
breaker all see them. A manual buy gets the default stop below its price.

A rating override (``/rate CLS Buy``) is used by the next cycle on that market
day instead of the agents' rating; that cycle still sizes and checks the order
with the usual rules.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from trading_engine.broker import Broker
from trading_engine.config import RiskLimits
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.models import TERMINAL_STATUSES, OrderRequest, Side
from trading_engine.reconcile import Reconciler, _market_date
from trading_engine.risk import client_order_id
from trading_engine.stops import StopBook
from trading_engine.wash import WashSaleChecker

RATINGS = {"buy": "Buy", "买入": "Buy", "overweight": "Overweight", "增持": "Overweight",
           "hold": "Hold", "持有": "Hold", "underweight": "Underweight", "减持": "Underweight",
           "sell": "Sell", "卖出": "Sell"}
CLEAR_RATING = {"ai", "auto", "clear", "自动", "清除"}


@dataclass
class Pending:
    """A previewed action waiting for your confirmation."""

    kind: str                       # order / cancel / rate
    ticker: str
    text: str                       # what you are asked to confirm
    side: Side | None = None
    quantity: float | None = None
    rating: str | None = None       # None with kind "rate" = clear the override
    order_ids: list[str] = field(default_factory=list)


def rating_override(ledger: Ledger, ticker: str, trade_date: str) -> str | None:
    """Your rating for this ticker on this market day, if you set one."""
    rating = None
    for rec in ledger.records("override"):
        d = rec["data"]
        if d.get("ticker") == ticker.upper() and d.get("trade_date") == trade_date:
            rating = d.get("rating")
    return rating


class ManualDesk:
    def __init__(self, broker: Broker, ledger: Ledger, kill_switch: KillSwitch,
                 limits: RiskLimits | None = None, *, settle_seconds: float = 10.0):
        self.broker = broker
        self.ledger = ledger
        self.kill_switch = kill_switch
        self.limits = limits or RiskLimits()
        self.stops = StopBook(ledger)
        self.settle_seconds = settle_seconds

    # --- previews ------------------------------------------------------------

    def preview_order(self, side: Side, ticker: str, quantity: float | None) -> Pending:
        ticker = ticker.upper()
        account = self.broker.get_account()
        held = account.holdings.get(ticker)
        held_qty = held.quantity if held else 0.0
        if side is Side.SELL:
            if held_qty <= 0:
                raise ValueError(f"你没有 {ticker} 持仓，不能卖出（引擎不做空）。")
            quantity = held_qty if quantity is None else quantity
            if quantity > held_qty:
                raise ValueError(f"你只持有 {held_qty:g} 股 {ticker}，不能卖 {quantity:g} 股（引擎不做空）。")
        if quantity is None:
            raise ValueError("买入要写股数，例如 /buy CLS 20")
        if quantity <= 0 or (not self.limits.allow_fractional and quantity != int(quantity)):
            raise ValueError("股数必须是正整数。")

        quote = self.broker.get_quote(ticker)
        value = quantity * quote.price
        warnings = self._warnings(side, ticker, quantity, quote, account, held)
        verb = "买入" if side is Side.BUY else "卖出"
        lines = [f"📝 手动{verb} {quantity:g} 股 {ticker}，市价约 ${quote.price:,.2f}（约 ${value:,.0f}）"]
        if side is Side.BUY:
            stop = quote.price * (1 - self.limits.default_stop_pct)
            lines.append(f"成交后挂止损 ${stop:,.2f}（-{self.limits.default_stop_pct:.0%}）")
        lines += self._warning_lines(warnings)
        lines.append("回复 /confirm 执行，/no 放弃（5 分钟内有效）")
        return Pending("order", ticker, "\n".join(lines), side=side, quantity=quantity)

    def preview_cancel(self, ticker: str) -> Pending:
        ticker = ticker.upper()
        orders = [o for o in self.broker.open_orders() if o.ticker.upper() == ticker and not o.is_stop]
        if not orders:
            raise ValueError(f"{ticker} 没有未成交的挂单（止损单不算，/cancel 不会撤止损单）。")
        listing = "\n".join(f"• {o.side} {o.remaining:g} 股（{o.status}）" for o in orders)
        text = (f"📝 撤销 {ticker} 的 {len(orders)} 张挂单：\n{listing}\n止损单保持不变。\n"
                "回复 /confirm 执行，/no 放弃（5 分钟内有效）")
        return Pending("cancel", ticker, text, order_ids=[o.client_order_id for o in orders])

    def preview_rate(self, ticker: str, word: str) -> Pending:
        ticker = ticker.upper()
        key = word.strip().lower()
        day = _market_date(None)
        if key in CLEAR_RATING:
            if rating_override(self.ledger, ticker, day) is None:
                raise ValueError(f"{ticker} 今天没有你设置的评级。")
            text = f"📝 取消你对 {ticker} 的评级，{day} 的下一次运行恢复使用 AI 的评级。"
            return Pending("rate", ticker, text + "\n回复 /confirm 执行，/no 放弃", rating=None)
        rating = RATINGS.get(key)
        if rating is None:
            raise ValueError("评级要写 Buy / Overweight / Hold / Underweight / Sell"
                             "（或 买入 / 增持 / 持有 / 减持 / 卖出），清除用 /rate CLS AI")
        text = (f"📝 {ticker} 在 {day} 的下一次运行用你的评级 {rating}，不用 AI 的。\n"
                "仓位仍由引擎按规则计算，引擎下的单照常过风控；想买确定的股数请用 /buy。\n"
                "回复 /confirm 执行，/no 放弃（5 分钟内有效）")
        return Pending("rate", ticker, text, rating=rating)

    # --- execution -------------------------------------------------------------

    def execute(self, p: Pending) -> str:
        if p.kind == "order":
            return self._execute_order(p)
        if p.kind == "cancel":
            for oid in p.order_ids:
                self.broker.cancel(oid)
                self.ledger.append("manual", {"action": "cancel", "ticker": p.ticker, "client_order_id": oid})
            return f"✅ 已撤销 {p.ticker} 的 {len(p.order_ids)} 张挂单。"
        if p.kind == "rate":
            day = _market_date(None)
            self.ledger.append("override", {"ticker": p.ticker, "trade_date": day, "rating": p.rating})
            if p.rating is None:
                return f"✅ 已取消 {p.ticker} 的手动评级，恢复使用 AI 评级。"
            return f"✅ {p.ticker} 今天的下一次运行会用你的评级 {p.rating}。"
        raise ValueError(f"unknown action {p.kind}")

    def _execute_order(self, p: Pending) -> str:
        assert p.side is not None and p.quantity is not None
        day = _market_date(None)
        decision_id = f"manual|{day}|{p.ticker}|{uuid.uuid4().hex[:8]}"
        account = self.broker.get_account()
        quote = self.broker.get_quote(p.ticker)
        open_orders = self.broker.open_orders()
        reconciler = Reconciler(self.broker, self.ledger, self.limits, self.kill_switch, stops=self.stops)
        self.ledger.append("manual", {"action": "order", "decision_id": decision_id, "trade_date": day,
                                      "ticker": p.ticker, "side": p.side.value, "quantity": p.quantity,
                                      "price": quote.price})
        # A sell needs the shares the resting stop reserves; the sync below puts it back.
        reconciler.cancel_stops(p.ticker, open_orders)
        order = OrderRequest(client_order_id(decision_id, p.ticker, p.side), p.ticker, p.side, p.quantity)
        status = self.broker.submit(order)
        if p.side is Side.BUY and self.limits.enforce_stops:
            self.stops.set(p.ticker, quote.price * (1 - self.limits.default_stop_pct),
                           source="manual buy, default stop", decision_id=decision_id)
        held = account.holdings.get(p.ticker)
        self.ledger.append("order", {
            "decision_id": decision_id, "trade_date": day, "ticker": p.ticker, "source": "manual",
            "client_order_id": order.client_order_id, "broker_order_id": status.broker_order_id,
            "side": p.side.value, "quantity": p.quantity, "status": status.status,
            "duplicate": status.duplicate, "est_price": quote.price,
            "filled_quantity": status.filled_quantity, "filled_avg_price": status.filled_avg_price,
            "avg_entry_price": getattr(held, "avg_entry_price", None),
        })
        self._settle(order.client_order_id)
        notes = reconciler.refresh_orders() + reconciler.sync_stops(day)
        final = self.broker.get_order(order.client_order_id)
        state = final.status if final else status.status
        verb = "买入" if p.side is Side.BUY else "卖出"
        lines = [f"✅ 已提交：手动{verb} {p.quantity:g} 股 {p.ticker} — {state}"]
        lines += [f"• {n}" for n in notes]
        return "\n".join(lines)

    def _settle(self, oid: str) -> None:
        deadline = time.monotonic() + self.settle_seconds
        while time.monotonic() < deadline:
            s = self.broker.get_order(oid)
            if s is None or s.status in TERMINAL_STATUSES:
                return
            time.sleep(1.0)

    # --- rule checks (warnings only) ------------------------------------------------

    def _warnings(self, side, ticker, quantity, quote, account, held) -> list[str]:
        lim, out = self.limits, []
        if self.kill_switch.active():
            out.append(f"紧急开关是打开的（{self.kill_switch.reason()}）")
        try:
            if not self.broker.is_market_open():
                out.append("现在不是开盘时间：订单会排队到下一次开盘，成交价可能和现在差很多")
        except Exception:  # noqa: BLE001 - a clock failure is only a missing warning
            pass
        age = (datetime.now(UTC) - quote.timestamp).total_seconds()
        if age > lim.max_quote_age_seconds:
            out.append(f"报价是 {age / 3600:.1f} 小时前的")
        if side is Side.SELL:
            return out
        equity = account.equity or 0.0
        cost = quantity * quote.price
        held_value = held.market_value if held else 0.0
        if cost > account.cash:
            out.append(f"现金不足：需要 ${cost:,.0f}，现金 ${account.cash:,.0f}（会用到保证金）")
        if equity > 0:
            weight = (held_value + cost) / equity
            if lim.max_position_weight is not None and weight > lim.max_position_weight:
                out.append(f"买后 {ticker} 占账户 {weight:.0%}，超过单只上限 {lim.max_position_weight:.0%}")
            invested = sum(h.market_value for h in account.holdings.values()) + cost
            if invested / equity > lim.max_gross_exposure:
                out.append(f"买后总仓位 {invested / equity:.0%}，超过上限 {lim.max_gross_exposure:.0%}")
            loss = cost * lim.default_stop_pct
            if loss > equity * lim.risk_per_trade:
                out.append(f"碰到止损约亏 ${loss:,.0f}（{loss / equity:.1%}），超过每笔上限 {lim.risk_per_trade:.1%}")
            peak = max(equity, self.ledger.high_water_mark() or 0.0)
            if peak > 0 and (peak - equity) / peak >= lim.max_drawdown:
                out.append(f"账户已从高点回撤 {(peak - equity) / peak:.1%}，按规则应停止买入")
        wash = WashSaleChecker(self.ledger).check_buy(ticker, date.fromisoformat(_market_date(None)))
        if wash:
            out.append(f"可能构成洗售：{wash}")
        return out

    @staticmethod
    def _warning_lines(warnings: list[str]) -> list[str]:
        if not warnings:
            return ["✅ 没有违反风控规则"]
        return ["⚠️ 风控提醒（只提醒，不拦截）："] + [f"• {w}" for w in warnings]
