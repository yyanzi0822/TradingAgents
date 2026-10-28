"""Deterministic execution layer around TradingAgents.

The LLM agents (``tradingagents``) decide *what* they would like to do. This
package decides *whether it is allowed* and *how it is executed*:

- ``risk``      hard limits the LLM cannot override
- ``sizing``    maps a 5-tier rating to a target portfolio weight
- ``broker``    Alpaca paper trading (live trading refused) and a fake broker
- ``ledger``    append-only, hash-chained log of every decision and order
- ``killswitch`` file-based emergency stop
- ``runner``    one end-to-end cycle: analyze -> size -> risk check -> log -> order

Design principle (from the project plan): AI makes investment decisions;
deterministic software enforces risk limits, order execution rules and
performance measurement.
"""

__all__ = ["RiskLimits", "RiskEngine"]

from trading_engine.config import RiskLimits
from trading_engine.risk import RiskEngine
