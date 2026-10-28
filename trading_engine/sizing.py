"""Turn a TradingAgents decision into a TradeIntent.

The agents return a 5-tier rating (Buy / Overweight / Hold / Underweight /
Sell) or ``REVIEW`` when their output could not be read. Only the rating is
used to size; free-text fields (thesis, horizon) are carried along for the log.
A ``REVIEW`` signal never trades.
"""

from __future__ import annotations

import re

from trading_engine.config import SizingPolicy
from trading_engine.models import AccountSnapshot, TradeIntent

REVIEW = "REVIEW"


def target_weight_for(rating: str, ticker: str, account: AccountSnapshot,
                      policy: SizingPolicy | None = None) -> float | None:
    policy = policy or SizingPolicy()
    if rating == REVIEW or rating not in policy.targets:
        return None
    if rating == "Underweight":
        current = account.weight(ticker)
        return current * policy.underweight_keep_fraction if current > 0 else None
    return policy.targets[rating]


def intent_from_decision(
    *,
    decision_id: str,
    ticker: str,
    rating: str,
    account: AccountSnapshot,
    final_state: dict | None = None,
    policy: SizingPolicy | None = None,
) -> TradeIntent:
    state = final_state or {}
    return TradeIntent(
        decision_id=decision_id,
        ticker=ticker.upper(),
        rating=rating,
        target_weight=target_weight_for(rating, ticker, account, policy),
        stop_loss=_stop_from_state(state),
        thesis=str(state.get("final_trade_decision", ""))[:4000],
        time_horizon=None,
    )


_STOP_RE = re.compile(r"\*\*Stop Loss\*\*:\s*\$?([0-9][0-9,]*\.?[0-9]*)")


def _stop_from_state(state: dict) -> float | None:
    """Read the trader's stop loss from its rendered proposal.

    ``render_trader_proposal`` writes ``**Stop Loss**: <value>`` (or
    ``not provided``) into ``trader_investment_plan``. Anything unparseable
    returns None and the risk engine falls back to its default stop.
    """
    match = _STOP_RE.search(str(state.get("trader_investment_plan", "")))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None
