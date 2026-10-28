"""Deterministic risk engine.

Every order passes through ``RiskEngine.evaluate`` before it can reach the
broker. Nothing here calls an LLM, and nothing the LLM writes can loosen a
limit: the intent only proposes a target weight, and this module decides the
quantity (if any) that is allowed.

Order of checks:
1. kill switch            -> reject everything
2. stale or bad quote     -> reject
3. no target              -> no order (Hold / unreadable decision)
4. sells                  -> always allowed (they reduce risk); never short
5. drawdown breaker       -> reject new buys
6. buys are capped by the smallest of: single-stock cap, sector headroom,
   gross-exposure headroom, available cash, and risk-per-trade sizing
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime

from trading_engine.config import RiskLimits
from trading_engine.models import (
    AccountSnapshot,
    OrderRequest,
    Quote,
    RiskResult,
    Side,
    TradeIntent,
)

UNKNOWN_SECTOR = "Unknown"


def client_order_id(decision_id: str, ticker: str, side: Side, quantity: float) -> str:
    """Deterministic order id: re-submitting the same decision cannot double-trade.

    Alpaca rejects a second order with an id it has already seen, and the
    broker wrapper checks for it first as well.
    """
    raw = f"{decision_id}|{ticker.upper()}|{side.value}|{quantity:.6f}"
    return "ta-" + hashlib.sha256(raw.encode()).hexdigest()[:40]


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None, sectors: dict[str, str] | None = None):
        self.limits = limits or RiskLimits()
        self.sectors = {k.upper(): v for k, v in (sectors or {}).items()}

    def sector_of(self, ticker: str) -> str:
        return self.sectors.get(ticker.upper(), UNKNOWN_SECTOR)

    def sector_weight(self, account: AccountSnapshot, sector: str) -> float:
        if account.equity <= 0:
            return 0.0
        total = sum(h.market_value for t, h in account.holdings.items() if self.sector_of(t) == sector)
        return total / account.equity

    def _round_qty(self, qty: float) -> float:
        if self.limits.allow_fractional:
            return math.floor(qty * 1e4) / 1e4
        return float(math.floor(qty))

    def evaluate(
        self,
        intent: TradeIntent,
        account: AccountSnapshot,
        quote: Quote,
        *,
        kill_switch_active: bool = False,
        now: datetime | None = None,
    ) -> RiskResult:
        lim = self.limits
        ticker = intent.ticker.upper()
        reasons: list[str] = []
        adjustments: list[str] = []

        # 1. Kill switch.
        if kill_switch_active:
            return RiskResult(False, None, ["kill switch is active"])

        # 2. Quote sanity and staleness.
        now = now or datetime.now(UTC)
        if quote.price is None or not math.isfinite(quote.price) or quote.price <= 0:
            return RiskResult(False, None, [f"invalid price for {ticker}: {quote.price}"])
        age = (now - quote.timestamp).total_seconds()
        if age > lim.max_quote_age_seconds:
            return RiskResult(False, None, [f"stale quote for {ticker}: {age:.0f}s old"])
        if account.equity <= 0:
            return RiskResult(False, None, ["account equity is not positive"])

        # 3. Nothing to do.
        if intent.target_weight is None:
            return RiskResult(True, None, ["no target weight (hold)"])

        target_w = intent.target_weight
        if target_w < 0 and lim.long_only:
            return RiskResult(False, None, ["negative target weight but strategy is long-only"])

        current = account.holdings.get(ticker)
        current_qty = current.quantity if current else 0.0
        current_value = current_qty * quote.price
        target_value = max(target_w, 0.0) * account.equity
        delta_value = target_value - current_value

        if abs(delta_value) < lim.min_order_notional:
            return RiskResult(True, None, [f"change of ${delta_value:,.0f} is below minimum order size"])

        # 4. Sells reduce risk: allowed, but never more than held.
        if delta_value < 0:
            qty = self._round_qty(min(-delta_value / quote.price, current_qty))
            if target_w == 0.0:
                qty = current_qty  # full exit sells everything, including fractions
            if qty <= 0:
                return RiskResult(True, None, ["nothing to sell"])
            return self._approve(intent, ticker, Side.SELL, qty, reasons, adjustments)

        # 5. Drawdown circuit breaker blocks new risk.
        dd = account.drawdown()
        if dd >= lim.max_drawdown:
            return RiskResult(False, None, [f"drawdown {dd:.1%} >= breaker {lim.max_drawdown:.0%}; buys halted"])

        # 6. Buys: take the tightest of every cap.
        caps: dict[str, float] = {}

        caps["single-stock cap"] = lim.max_position_weight * account.equity - current_value

        sector = self.sector_of(ticker)
        if sector == UNKNOWN_SECTOR:
            reasons.append(f"{ticker} has no sector mapping; add it to the universe file")
            return RiskResult(False, None, reasons)
        caps["sector cap"] = (lim.max_sector_weight - self.sector_weight(account, sector)) * account.equity

        caps["gross exposure cap"] = lim.max_gross_exposure * account.equity - account.gross_exposure
        caps["available cash"] = account.cash

        stop = intent.stop_loss
        if stop is None or not (0 < stop < quote.price):
            if stop is not None:
                adjustments.append(f"ignored stop {stop} (not below price); used default stop")
            stop = quote.price * (1 - lim.default_stop_pct)
        risk_per_share = quote.price - stop
        risk_budget = lim.risk_per_trade * account.equity
        caps["risk per trade"] = (risk_budget / risk_per_share) * quote.price

        allowed_value = min(delta_value, *caps.values())
        binding = min(caps, key=caps.get)
        if allowed_value < delta_value:
            adjustments.append(f"buy reduced by {binding}: ${delta_value:,.0f} -> ${max(allowed_value, 0):,.0f}")

        if allowed_value < lim.min_order_notional:
            return RiskResult(False, None, [f"no room to buy {ticker}: limited by {binding}"], adjustments)

        qty = self._round_qty(allowed_value / quote.price)
        if qty <= 0:
            return RiskResult(False, None, [f"{ticker} price too high for allowed size"], adjustments)
        return self._approve(intent, ticker, Side.BUY, qty, reasons, adjustments)

    @staticmethod
    def _approve(intent, ticker, side, qty, reasons, adjustments) -> RiskResult:
        order = OrderRequest(
            client_order_id=client_order_id(intent.decision_id, ticker, side, qty),
            ticker=ticker,
            side=side,
            quantity=qty,
        )
        return RiskResult(True, order, reasons, adjustments)
