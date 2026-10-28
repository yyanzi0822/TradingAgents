# Trading Engine (paper trading layer)

`trading_engine/` wraps TradingAgents with the deterministic parts of the
project plan: hard risk limits, Alpaca **paper** execution, an audit ledger
and a kill switch. The upstream `tradingagents/` package is untouched, so
upstream updates can still be merged.

> AI makes investment decisions. Deterministic software enforces risk limits,
> order execution rules, and performance measurement.

## Flow of one cycle

```
broker account ──► TradingAgents (analysts → debate → trader → PM) ──► rating
                                                                         │
                         sizing: Buy 5% · Overweight 3% · Hold keep ·    ▼
                         Underweight halve · Sell exit · REVIEW never trades
                                                                         │
             ledger: decision written BEFORE any order  ◄────────────────┤
                                                                         ▼
          risk engine (no LLM): kill switch → stale quote → long-only →
          drawdown breaker → min(5% stock, 20% sector, 80% gross, cash,
          0.5% risk-per-trade from the trader's stop or an 8% default)
                                                                         │
                                                                         ▼
                 Alpaca paper order, idempotent client_order_id → ledger
```

## Setup

```bash
pip install -e ".[alpaca]"
cp .env.example .env     # add one LLM key + ALPACA_API_KEY / ALPACA_SECRET_KEY (paper keys)
```

## Run

```bash
# Analyze and size only; uses Yahoo prices if no Alpaca key is set
python -m trading_engine.run --tickers AAPL,MSFT --dry-run

# Paper trade (orders only go out while the market is open)
python -m trading_engine.run --tickers AAPL,MSFT,JPM

# Whole starter universe (trading_engine/universe.json), fewer analysts to save cost
python -m trading_engine.run --universe --analysts market,news,fundamentals

# Emergency stop / resume
python -m trading_engine.killswitch on "something looks wrong"
python -m trading_engine.killswitch off
```

The ledger and kill-switch file live in `~/.trading_engine/` (override with
`TRADING_ENGINE_DIR`).

## Safety rails

| Rail | Where |
|---|---|
| Live trading refused (non-paper URL or account number) | `broker.AlpacaPaperBroker` |
| 5% stock / 20% sector / 80% gross / 0.5% risk per trade | `risk.RiskEngine` |
| 8% drawdown breaker halts buys (sells still allowed) | `risk.RiskEngine` + ledger high-water mark |
| Stale (>15 min) or invalid quotes rejected | `risk.RiskEngine` |
| Unknown-sector tickers rejected | `risk.RiskEngine` + `universe.json` |
| One order per ticker per day; deterministic `client_order_id` | `runner.Runner`, `risk.client_order_id` |
| Decision logged before order; hash-chained ledger | `ledger.Ledger` |
| Tampered ledger or 3 errors in a cycle → kill switch | `runner.Runner` |

All limits are in `trading_engine/config.py`. Each decision records
`risk_version` (a hash of the limits), so changing a limit starts a new
experimental version instead of silently mixing results.

## Tests

```bash
pytest tests/trading_engine
```

## Not built yet

- Opportunity screener, Market Regime agent, Strategy agent (holding periods per strategy)
- Exits driven by stop loss / time horizon between analysis runs
- Broker reconciliation job for partial fills and orders accepted but not filled
- Portfolios B–D (SPY, momentum baseline, AI fixed 10-day horizon) and the evaluation layer
- Scheduling (market-hours cron) and monitoring
