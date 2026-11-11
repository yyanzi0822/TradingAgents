"""Experiment configuration: risk limits and sizing.

These are the *initial experimental limits* from the development plan, not
validated parameters. Changing any of them creates a new experimental version
(see ``RiskLimits.version_tag``), so results from different settings are never
mixed in one track record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RiskLimits:
    # Position and exposure caps, as fractions of account equity.
    # Single-stock cap. None = no cap. This is a ceiling, not a target: how
    # much is actually bought comes from SizingPolicy below, and the sector
    # cap and risk per trade can bind first.
    max_position_weight: float | None = 0.30
    max_sector_weight: float | None = None  # one sector; None = no sector cap
    max_gross_exposure: float = 0.80       # total invested

    # Planned loss if the stop is hit, as a fraction of equity.
    risk_per_trade: float = 0.005
    # Stop distance assumed when the agents give no stop loss.
    default_stop_pct: float = 0.08

    # Halt new buys once equity is this far below its high-water mark.
    max_drawdown: float = 0.08

    # Keep a resting stop order at the broker for every position, at the stop
    # the position was sized with. Off = stops only size the trade.
    enforce_stops: bool = True

    # Take profit in steps: (gain over the average entry price, fraction of
    # the position to sell). Each level is taken once per position; the
    # fraction is of the shares held when the first level is reached, so two
    # levels of 1/3 leave 1/3 running. () = no take-profit. Checked whenever
    # the engine reconciles, not continuously at the broker.
    take_profit_levels: tuple[tuple[float, float], ...] = ((0.015, 1 / 3), (0.025, 1 / 3))

    # Wash-sale detection (see wash.py): "warn" flags a possible wash sale in
    # the output, the ledger and the message; "block" also refuses the buy;
    # "off" does nothing. Sells are never blocked.
    wash_sale_policy: str = "warn"

    # Reject decisions made on prices older than this.
    max_quote_age_seconds: int = 15 * 60

    # Ignore orders smaller than this (avoids churn from tiny rebalances).
    min_order_notional: float = 100.0

    # v1 scope.
    long_only: bool = True
    allow_fractional: bool = False

    def version_tag(self) -> str:
        """Stable short hash of the settings, recorded with every decision."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class SizingPolicy:
    """Target weight per rating, before risk caps are applied.

    ``None`` means "keep the current position unchanged".
    A weight of 0.0 means "exit the position".
    """

    targets: dict[str, float | None] = field(default_factory=lambda: {
        "Buy": 0.05,
        "Overweight": None,    # not traded: treated like Hold
        "Hold": None,
        "Underweight": None,   # handled as "halve the position" in sizing.py
        "Sell": 0.0,
    })
    underweight_keep_fraction: float = 0.5
    # Ratings that may add to a position but never sell it down. A Buy on a
    # position that has grown past its target leaves it alone; only
    # Underweight, Sell and a stop reduce a position.
    add_only_ratings: tuple[str, ...] = ("Buy",)
