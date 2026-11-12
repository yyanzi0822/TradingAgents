"""One end-to-end trading cycle.

Before any ticker: reconcile with the broker (``reconcile``) so order
statuses and protective stops are current.

For each ticker:
    1. read the account and open orders from the broker (record equity for
       the drawdown breaker; unfilled buys count against the caps)
    2. run the TradingAgents graph with the real book as portfolio context
    3. turn its rating into a target weight (``sizing``)
    4. write the decision to the ledger  <- before anything is sent
    5. run the deterministic risk checks (``risk``)
    6. submit the approved order, idempotently, and log the broker's answer
       (the ticker's resting stop is cancelled first; a buy records its stop price)

After the last ticker: wait briefly for fills, then reconcile again so every
new position has its stop order resting at the broker.

Safety rails:
    - the kill switch stops the cycle before any analysis or order
    - one order per ticker per trading day (a re-run cannot double-trade)
    - after ``max_errors`` failures in one cycle the kill switch is engaged
    - orders are only sent while the market is open, unless told otherwise
    - a ticker with an order still open at the broker, or stopped out today,
      is skipped
    - ``confirm`` (optional) asks before each buy or sell is sent
    - ``dry_run`` does everything except submit
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from trading_engine.broker import Broker
from trading_engine.config import RiskLimits, SizingPolicy
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.models import TERMINAL_STATUSES, AccountSnapshot, Side
from trading_engine.notify import Notifier, NullNotifier
from trading_engine.reconcile import Reconciler, _market_date
from trading_engine.risk import RiskEngine, client_order_id
from trading_engine.sizing import intent_from_decision
from trading_engine.stops import StopBook
from trading_engine.takeprofit import TakeProfitBook
from trading_engine.wash import WashSaleChecker

logger = logging.getLogger(__name__)

# (ticker, trade_date, portfolio_context) -> (final_state, signal)
AnalyzeFn = Callable[[str, str, object], tuple[dict, str]]


@dataclass
class TickerOutcome:
    ticker: str
    signal: str | None = None
    action: str = "none"          # none / submitted / duplicate / rejected / skipped / declined / error / dry-run
    detail: str = ""
    thesis: str = ""              # the agents' final decision text, for the Telegram report


@dataclass
class CycleReport:
    trade_date: str
    outcomes: list[TickerOutcome] = field(default_factory=list)
    halted: str | None = None
    # What reconciliation changed: fills, stops placed or moved, stop-outs.
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Cycle {self.trade_date}" + (f" — HALTED: {self.halted}" if self.halted else "")]
        for o in self.outcomes:
            lines.append(f"  {o.ticker:<6} {str(o.signal or '-'):<11} {o.action:<10} {o.detail}")
        if self.notes:
            lines.append("  orders and stops:")
            lines.extend(f"    {n}" for n in self.notes)
        return "\n".join(lines)

    def telegram_blocks(self, *, dry_run: bool = False) -> list[str]:
        """The report as Telegram text: a header, then one block per ticker.

        Every ticker gets its rating, what the engine did and why, and the
        agents' executive summary, so a Hold or a rejected order is visible too.
        """
        head = f"📊 交易周期 {self.trade_date}" + ("（dry run，不下单）" if dry_run else "")
        if self.halted:
            head += f"\n🚨 已停止: {self.halted}"
        blocks = [head]
        for o in self.outcomes:
            label = ACTION_LABELS.get(o.action, o.action)
            rating = f"{o.signal}（{RATING_LABELS.get(o.signal, o.signal)}）" if o.signal else "-"
            block = f"{o.ticker} · {rating}\n操作: {label}" + (f" — {o.detail}" if o.detail else "")
            summary = executive_summary(o.thesis)
            if summary:
                block += f"\n理由: {summary}"
            blocks.append(block)
        if self.notes:
            blocks.append("订单和止损:\n" + "\n".join(f"• {n}" for n in self.notes))
        return blocks


RATING_LABELS = {"Buy": "买入", "Overweight": "增持", "Hold": "持有",
                 "Underweight": "减持", "Sell": "卖出", "REVIEW": "无法解读"}
ACTION_LABELS = {"none": "不操作", "submitted": "已下单", "duplicate": "订单已存在",
                 "rejected": "被风控拦下", "skipped": "跳过", "declined": "你已拒绝",
                 "error": "出错", "dry-run": "模拟（未下单）"}


def executive_summary(thesis: str, limit: int = 400) -> str:
    """The "Executive Summary" paragraph of a decision, shortened for a phone."""
    if not thesis:
        return ""
    m = re.search(r"\*\*Executive Summary\*\*:?\s*(.+?)(?:\n\s*\n|\n\*\*|$)", thesis, re.S)
    text = (m.group(1) if m else thesis).replace("**", "").strip()
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def tradingagents_analyzer(config: dict | None = None, analysts: list[str] | None = None) -> AnalyzeFn:
    """The default analyzer: one TradingAgentsGraph run per ticker."""
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(config or {})
    kwargs = {"selected_analysts": analysts} if analysts else {}
    graph = TradingAgentsGraph(config=cfg, **kwargs)

    def analyze(ticker: str, trade_date: str, portfolio) -> tuple[dict, str]:
        return graph.propagate(ticker, trade_date, portfolio=portfolio)

    return analyze


def portfolio_context(account: AccountSnapshot):
    """Convert the broker's account into TradingAgents' PortfolioContext."""
    from tradingagents.portfolio import PortfolioContext

    return PortfolioContext.model_validate({
        "cash": account.cash,
        "currency": "USD",
        "positions": [
            {"ticker": h.ticker, "quantity": h.quantity, "average_price": h.avg_entry_price}
            for h in account.holdings.values()
        ],
    })


class Runner:
    def __init__(
        self,
        broker: Broker,
        analyze: AnalyzeFn,
        ledger: Ledger,
        *,
        sectors: dict[str, str],
        limits: RiskLimits | None = None,
        sizing: SizingPolicy | None = None,
        kill_switch: KillSwitch | None = None,
        dry_run: bool = False,
        require_market_open: bool = True,
        max_errors: int = 3,
        portfolio_builder: Callable[[AccountSnapshot], object] = portfolio_context,
        settle_seconds: float = 0.0,
        confirm: Callable[[str], bool] | None = None,
        notifier: Notifier | None = None,
    ):
        self.broker = broker
        self.analyze = analyze
        self.ledger = ledger
        self.limits = limits or RiskLimits()
        self.sizing = sizing or SizingPolicy()
        self.risk = RiskEngine(self.limits, sectors)
        self.kill_switch = kill_switch or KillSwitch()
        self.dry_run = dry_run
        self.require_market_open = require_market_open
        self.max_errors = max_errors
        self.portfolio_builder = portfolio_builder
        self.settle_seconds = settle_seconds
        self.confirm = confirm
        self.stops = StopBook(ledger)
        self.wash = WashSaleChecker(ledger)
        # Dry runs send nothing, messages included.
        self.notifier = NullNotifier() if dry_run or notifier is None else notifier
        self.reconciler = Reconciler(broker, ledger, self.limits, self.kill_switch, stops=self.stops,
                                     dry_run=dry_run, require_market_open=require_market_open,
                                     confirm=confirm, notifier=self.notifier)

    def _engage_kill_switch(self, reason: str) -> None:
        self.kill_switch.engage(reason)
        self.notifier.send(f"🚨 KILL SWITCH ON: {reason}\nNo new orders until you turn it off "
                           f"(python -m trading_engine.killswitch off).")

    def _account(self, open_orders=None) -> AccountSnapshot:
        acct = self.broker.get_account()
        self.ledger.append("equity", {"equity": acct.equity, "cash": acct.cash})
        # Buys the broker has accepted but not filled are not positions yet.
        pending: dict[str, float] = {}
        for o in (self.broker.open_orders() if open_orders is None else open_orders):
            if o.side == Side.BUY.value and o.remaining > 0:
                price = o.limit_price or self.broker.get_quote(o.ticker).price
                pending[o.ticker] = pending.get(o.ticker, 0.0) + o.remaining * price
        return AccountSnapshot(
            equity=acct.equity, cash=acct.cash, holdings=acct.holdings,
            high_water_mark=self.ledger.high_water_mark(), as_of=acct.as_of,
            pending_buys=pending,
        )

    def _already_traded(self, trade_date: str, ticker: str) -> bool:
        return any(r["data"].get("trade_date") == trade_date and r["data"].get("ticker") == ticker
                   for r in self.ledger.records("order"))

    def _wash_check(self, order, ticker: str, trade_date: str, price: float,
                    account: AccountSnapshot) -> str | None:
        if self.limits.wash_sale_policy == "off":
            return None
        day = date.fromisoformat(trade_date)
        if order.side is Side.BUY:
            return self.wash.check_buy(ticker, day)
        held = account.holdings.get(ticker)
        if held is None:
            return None
        return self.wash.check_sell(ticker, day, price=price, cost_basis=held.avg_entry_price,
                                    quantity=order.quantity, held=held.quantity)

    def _stopped_out(self, trade_date: str, ticker: str) -> bool:
        return any(r["data"].get("trade_date") == trade_date and r["data"].get("ticker") == ticker
                   for r in self.ledger.records("stop_filled"))

    def _settle(self, order_ids: list[str]) -> None:
        """Give this cycle's orders a moment to fill before stops are placed."""
        deadline = time.monotonic() + self.settle_seconds
        while order_ids and time.monotonic() < deadline:
            states = [self.broker.get_order(oid) for oid in order_ids]
            if all(s is None or s.status in TERMINAL_STATUSES for s in states):
                return
            time.sleep(min(2.0, self.settle_seconds))

    def _reconcile(self, report: CycleReport, trade_date: str) -> bool:
        try:
            report.notes.extend(self.reconciler.run(trade_date))
            return True
        except Exception as exc:  # noqa: BLE001 - reported, and decides whether the cycle goes on
            logger.exception("reconciliation failed")
            self.ledger.append("error", {"trade_date": trade_date, "ticker": None,
                                         "error": f"reconcile: {exc!r}"})
            report.notes.append(f"reconciliation failed: {exc!r}"[:200])
            return False

    def _order_at_broker(self, decision_id: str, ticker: str):
        """An order for this decision that the broker has but the ledger does not.

        Happens when a submit reached the broker and the reply was lost. Either
        side counts: a retry must not trade again even if the agents changed
        their mind on the second run.
        """
        for side in Side:
            status = self.broker.get_order(client_order_id(decision_id, ticker, side))
            if status is not None:
                return status
        return None

    def run_cycle(self, tickers: list[str], trade_date: str | None = None) -> CycleReport:
        # The market's date, not this machine's: late evening in California is
        # already the next day in New York.
        trade_date = trade_date or _market_date(None)
        report = CycleReport(trade_date)
        version = self.limits.version_tag()

        ok, msg = self.ledger.verify()
        if not ok:
            self._engage_kill_switch(f"ledger integrity check failed: {msg}")
        if self.kill_switch.active():
            report.halted = f"kill switch: {self.kill_switch.reason()}"
            return report

        # Trading on a stale view of orders and stops is worse than not trading.
        if not self._reconcile(report, trade_date):
            report.halted = "could not reconcile with the broker; nothing traded"
            self.notifier.send(f"⚠️ Cycle {trade_date} did not run: could not reconcile with the broker.")
            return report

        market_open = self.broker.is_market_open()
        errors = 0
        submitted: list[str] = []

        for raw in tickers:
            ticker = raw.strip().upper()
            outcome = TickerOutcome(ticker)
            report.outcomes.append(outcome)

            if self._already_traded(trade_date, ticker):
                outcome.action, outcome.detail = "skipped", "already traded today"
                continue
            if self._stopped_out(trade_date, ticker):
                outcome.action, outcome.detail = "skipped", "stopped out today"
                continue
            if self.kill_switch.active():
                outcome.action, outcome.detail = "skipped", "kill switch engaged mid-cycle"
                continue

            try:
                decision_id = f"{trade_date}|{ticker}|{version}"
                if not self.dry_run:
                    found = self._order_at_broker(decision_id, ticker)
                    if found is not None:
                        self.ledger.append("order", {
                            "decision_id": decision_id, "trade_date": trade_date, "ticker": ticker,
                            "client_order_id": found.client_order_id,
                            "broker_order_id": found.broker_order_id, "side": found.side,
                            "quantity": found.quantity, "status": found.status,
                            "duplicate": True, "recovered": True,
                        })
                        outcome.action = "duplicate"
                        outcome.detail = (f"order already at broker ({found.side} {found.quantity:g}, "
                                          f"{found.status}); recorded, not resubmitted")
                        continue

                open_orders = self.broker.open_orders()
                in_flight = [o for o in open_orders if o.ticker == ticker and not o.is_stop]
                if in_flight:
                    o = in_flight[0]
                    outcome.action = "skipped"
                    outcome.detail = f"order still open at broker ({o.side} {o.remaining:g}, {o.status})"
                    continue

                account = self._account(open_orders)
                final_state, signal = self.analyze(ticker, trade_date, self.portfolio_builder(account))
                outcome.signal = signal
                intent = intent_from_decision(decision_id=decision_id, ticker=ticker, rating=signal,
                                              account=account, final_state=final_state, policy=self.sizing)
                outcome.thesis = intent.thesis
                # Profit was taken on this position: a Buy must not buy the sold part back.
                profit_taken = (signal in self.sizing.add_only_ratings and intent.target_weight is not None
                                and bool(TakeProfitBook(self.ledger).taken(ticker)))
                if profit_taken:
                    intent = replace(intent, target_weight=None)

                # Log the decision before the market can answer.
                self.ledger.append("decision", {
                    "decision_id": decision_id, "trade_date": trade_date, "ticker": ticker,
                    "signal": signal, "target_weight": intent.target_weight,
                    "stop_loss": intent.stop_loss, "risk_version": version,
                    "thesis": intent.thesis,
                })

                quote = self.broker.get_quote(ticker)
                result = self.risk.evaluate(intent, account, quote,
                                            kill_switch_active=self.kill_switch.active())
                self.ledger.append("risk", {
                    "decision_id": decision_id, "ticker": ticker, "approved": result.approved,
                    "reasons": result.reasons, "adjustments": result.adjustments,
                    "price": quote.price, "quote_time": quote.timestamp.isoformat(),
                    "stop_price": result.stop_price,
                    "order": None if result.order is None else {
                        "side": result.order.side.value, "quantity": result.order.quantity,
                        "client_order_id": result.order.client_order_id},
                })

                if not result.approved:
                    outcome.action, outcome.detail = "rejected", "; ".join(result.reasons)
                    continue
                if result.order is None:
                    outcome.detail = ("profit already taken on this position; not adding to it" if profit_taken
                                      else "; ".join(result.reasons) or "no change")
                    continue

                o = result.order
                summary = f"{o.side.value} {o.quantity:g} @ ~{quote.price:,.2f}"
                if result.adjustments:
                    summary += f" ({'; '.join(result.adjustments)})"

                wash = self._wash_check(o, ticker, trade_date, quote.price, account)
                if wash:
                    blocked = o.side is Side.BUY and self.limits.wash_sale_policy == "block"
                    self.ledger.append("wash_sale", {
                        "decision_id": decision_id, "trade_date": trade_date, "ticker": ticker,
                        "side": o.side.value, "quantity": o.quantity, "detail": wash, "blocked": blocked,
                    })
                    if blocked:
                        outcome.action, outcome.detail = "rejected", f"buy blocked: {wash}"
                        continue
                    summary += f" [{wash}]"
                if self.dry_run:
                    outcome.action, outcome.detail = "dry-run", summary
                    continue
                if self.require_market_open and not market_open:
                    outcome.action, outcome.detail = "skipped", f"market closed; would {summary}"
                    continue

                if self.confirm and not self.confirm(f"{ticker} ({signal}): {summary}"):
                    self.ledger.append("declined", {"decision_id": decision_id, "trade_date": trade_date,
                                                    "ticker": ticker, "what": summary})
                    outcome.action, outcome.detail = "declined", f"you declined: {summary}"
                    continue

                # Take the resting stop down before trading the ticker. A sell
                # needs the shares it reserves, and Alpaca can refuse a buy
                # while an opposite-side order is open (wash-trade check).
                # The end-of-cycle sync puts a stop back for whatever is held,
                # including when this order fails.
                self.reconciler.cancel_stops(ticker, open_orders)
                status = self.broker.submit(o)
                submitted.append(o.client_order_id)
                if o.side is Side.BUY and self.limits.enforce_stops and result.stop_price:
                    self.stops.set(ticker, result.stop_price, source="sized with this stop",
                                   decision_id=decision_id)
                self.ledger.append("order", {
                    "decision_id": decision_id, "trade_date": trade_date, "ticker": ticker,
                    "client_order_id": o.client_order_id, "broker_order_id": status.broker_order_id,
                    "side": o.side.value, "quantity": o.quantity, "status": status.status,
                    "duplicate": status.duplicate,
                    # For wash-sale detection: what it traded at, and what it cost.
                    "est_price": quote.price, "filled_quantity": status.filled_quantity,
                    "filled_avg_price": status.filled_avg_price,
                    "avg_entry_price": getattr(account.holdings.get(ticker), "avg_entry_price", None),
                })
                outcome.action = "duplicate" if status.duplicate else "submitted"
                outcome.detail = f"{summary} -> {status.status}"
                if not status.duplicate:
                    stop_note = f", stop {result.stop_price:,.2f}" if o.side is Side.BUY and result.stop_price else ""
                    self.notifier.send(f"{o.side.value.upper()} {o.quantity:g} {ticker} @ ~{quote.price:,.2f}"
                                       f"{stop_note}\nRating: {signal} · {status.status}"
                                       + (f"\n⚠️ {wash}" if wash else ""))

            except Exception as exc:  # noqa: BLE001 - one bad ticker must not kill the cycle
                errors += 1
                logger.exception("cycle error for %s", ticker)
                self.ledger.append("error", {"trade_date": trade_date, "ticker": ticker, "error": repr(exc)})
                outcome.action, outcome.detail = "error", repr(exc)[:200]
                self.notifier.send(f"⚠️ ERROR {ticker}: {repr(exc)[:300]}")
                if errors >= self.max_errors:
                    self._engage_kill_switch(f"{errors} errors in cycle {trade_date}")
                    report.halted = f"auto-suspended after {errors} errors"

        if not self.dry_run:
            self._settle(submitted)
        self._reconcile(report, trade_date)
        return report
