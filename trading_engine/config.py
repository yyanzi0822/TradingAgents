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
    max_position_weight: float = 0.05      # single stock
    max_sector_weight: float = 0.20        # one sector
    max_gross_exposure: float = 0.80       # total invested

    # Planned loss if the stop is hit, as a fraction of equity.
    risk_per_trade: float = 0.005
    # Stop distance assumed when the agents give no stop loss.
    default_stop_pct: float = 0.08

    # Halt new buys once equity is this far below its high-water mark.
    max_drawdown: float = 0.08

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
        "Overweight": 0.03,
        "Hold": None,
        "Underweight": None,   # handled as "halve the position" in sizing.py
        "Sell": 0.0,
    })
    underweight_keep_fraction: float = 0.5
