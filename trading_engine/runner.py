"""One end-to-end trading cycle.

For each ticker:
    1. read the account from the broker (record equity for the drawdown breaker)
    2. run the TradingAgents graph with the real book as portfolio context
    3. turn its rating into a target weight (``sizing``)
    4. write the decision to the ledger  <- before anything is sent
    5. run the deterministic risk checks (``risk``)
    6. submit the approved order, idempotently, and log the broker's answer

Safety rails:
    - the kill switch stops the cycle before any analysis or order
    - one order per ticker per trading day (a re-run cannot double-trade)
    - after ``max_errors`` failures in one cycle the kill switch is engaged
    - orders are only sent while the market is open, unless told otherwise
    - ``dry_run`` does everything except submit
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from trading_engine.broker import Broker
from trading_engine.config import RiskLimits, SizingPolicy
from trading_engine.killswitch import KillSwitch
from trading_engine.ledger import Ledger
from trading_engine.models import AccountSnapshot
from trading_engine.risk import RiskEngine
from trading_engine.sizing import intent_from_decision

logger = logging.getLogger(__name__)

# (ticker, trade_date, portfolio_context) -> (final_state, signal)
AnalyzeFn = Callable[[str, str, object], tuple[dict, str]]


@dataclass
class TickerOutcome:
    ticker: str
    signal: str | None = None
    action: str = "none"          # none / submitted / duplicate / rejected / skipped / error / dry-run
    detail: str = ""


@dataclass
class CycleReport:
    trade_date: str
    outcomes: list[TickerOutcome] = field(default_factory=list)
    halted: str | None = None

    def render(self) -> str:
        lines = [f"Cycle {self.trade_date}" + (f" — HALTED: {self.halted}" if self.halted else "")]
        for o in self.outcomes:
            lines.append(f"  {o.ticker:<6} {str(o.signal or '-'):<11} {o.action:<10} {o.detail}")
        return "\n".join(lines)


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

    def _account(self) -> AccountSnapshot:
        acct = self.broker.get_account()
        self.ledger.append("equity", {"equity": acct.equity, "cash": acct.cash})
        return AccountSnapshot(
            equity=acct.equity, cash=acct.cash, holdings=acct.holdings,
            high_water_mark=self.ledger.high_water_mark(), as_of=acct.as_of,
        )

    def _already_traded(self, trade_date: str, ticker: str) -> bool:
        return any(r["data"].get("trade_date") == trade_date and r["data"].get("ticker") == ticker
                   for r in self.ledger.records("order"))

    def run_cycle(self, tickers: list[str], trade_date: str | None = None) -> CycleReport:
        trade_date = trade_date or date.today().isoformat()
        report = CycleReport(trade_date)
        version = self.limits.version_tag()

        ok, msg = self.ledger.verify()
        if not ok:
            self.kill_switch.engage(f"ledger integrity check failed: {msg}")
        if self.kill_switch.active():
            report.halted = f"kill switch: {self.kill_switch.reason()}"
            return report

        market_open = self.broker.is_market_open()
        errors = 0

        for raw in tickers:
            ticker = raw.strip().upper()
            outcome = TickerOutcome(ticker)
            report.outcomes.append(outcome)

            if self._already_traded(trade_date, ticker):
                outcome.action, outcome.detail = "skipped", "already traded today"
                continue
            if self.kill_switch.active():
                outcome.action, outcome.detail = "skipped", "kill switch engaged mid-cycle"
                continue

            try:
                account = self._account()
                final_state, signal = self.analyze(ticker, trade_date, self.portfolio_builder(account))
                outcome.signal = signal
                decision_id = f"{trade_date}|{ticker}|{version}"
                intent = intent_from_decision(decision_id=decision_id, ticker=ticker, rating=signal,
                                              account=account, final_state=final_state, policy=self.sizing)

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
                    "order": None if result.order is None else {
                        "side": result.order.side.value, "quantity": result.order.quantity,
                        "client_order_id": result.order.client_order_id},
                })

                if not result.approved:
                    outcome.action, outcome.detail = "rejected", "; ".join(result.reasons)
                    continue
                if result.order is None:
                    outcome.detail = "; ".join(result.reasons) or "no change"
                    continue

                o = result.order
                summary = f"{o.side.value} {o.quantity:g} @ ~{quote.price:,.2f}"
                if result.adjustments:
                    summary += f" ({'; '.join(result.adjustments)})"
                if self.dry_run:
                    outcome.action, outcome.detail = "dry-run", summary
                    continue
                if self.require_market_open and not market_open:
                    outcome.action, outcome.detail = "skipped", f"market closed; would {summary}"
                    continue

                status = self.broker.submit(o)
                self.ledger.append("order", {
                    "decision_id": decision_id, "trade_date": trade_date, "ticker": ticker,
                    "client_order_id": o.client_order_id, "broker_order_id": status.broker_order_id,
                    "side": o.side.value, "quantity": o.quantity, "status": status.status,
                    "duplicate": status.duplicate,
                })
                outcome.action = "duplicate" if status.duplicate else "submitted"
                outcome.detail = f"{summary} -> {status.status}"

            except Exception as exc:  # noqa: BLE001 - one bad ticker must not kill the cycle
                errors += 1
                logger.exception("cycle error for %s", ticker)
                self.ledger.append("error", {"trade_date": trade_date, "ticker": ticker, "error": repr(exc)})
                outcome.action, outcome.detail = "error", repr(exc)[:200]
                if errors >= self.max_errors:
                    self.kill_switch.engage(f"{errors} errors in cycle {trade_date}")
                    report.halted = f"auto-suspended after {errors} errors"

        return report
