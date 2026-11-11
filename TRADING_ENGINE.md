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
                                                                         │
                                                                         ▼
        reconcile: record fills → one resting GTC stop order per position
```

Reconciliation also runs at the start of every cycle, before any analysis.

## It trades without asking

A cycle analyzes, sizes and sends orders on its own, and the stop orders rest
at Alpaca, so they trigger even when this program is not running. Nothing asks
you first unless you turn it on:

- `--confirm` shows each buy or sell (and a market exit on a breached stop) and
  waits for `y`. Placing or moving a protective stop is never asked.
- `--dry-run` sends nothing.
- The kill switch refuses every new order. Stops already resting at the broker
  stay there and can still trigger.

## Messages on your phone (Telegram)

One-time setup:

1. In Telegram, talk to `@BotFather`, send `/newbot`, and put the token it gives
   you in `.env` as `TELEGRAM_BOT_TOKEN`.
2. Open your new bot and send it any message.
3. `python -m trading_engine.notify chat-id` prints your `TELEGRAM_CHAT_ID`; add it to `.env`.
4. `python -m trading_engine.notify test` should reach your phone.

You then get a message for every order sent, every stop that is hit, an order
that expires or is rejected, any error, and the kill switch turning on.
`python -m trading_engine.summary --send` sends the end-of-day summary. Dry
runs send nothing. A message that fails to send is logged and never stops a
cycle. Treat the bot token like a password: anyone holding it can post as your bot.

## Stops and reconciliation

- **Where the stop comes from.** The price a buy was sized with: the trader's
  stop loss, or 8% below the price when it gives none. A position with no
  recorded stop gets 8% below its average entry price.
- **Only up.** While a position is open its stop can be raised, never lowered,
  so a later decision from the agents cannot loosen it.
- **Kept in step.** Each run checks that every position has exactly one stop
  order for its whole quantity, and replaces it after a partial sell or an
  added buy. The stop is taken down just before the engine trades that ticker
  and put back at the end of the cycle.
- **Already through the stop.** If the price is at or below the stop and no
  stop order caught it, the position is sold at market (only while the market
  is open, unless `--allow-closed`).
- **Stopped out.** A filled stop is recorded, and that ticker is not bought
  again the same day.
- **Unfilled orders.** A ticker with an order still open is skipped, and
  unfilled buys count against the stock, sector, gross-exposure and cash caps.
  Fills, partial fills and expiries are written to the ledger as they are seen.

Run it on its own at any time (for example a few minutes after a cycle, or
before the open):

```bash
python -m trading_engine.reconcile            # update orders and stops now
python -m trading_engine.reconcile --dry-run  # show what would change
```

Limits worth knowing: a stop order is a market order once triggered, so a gap
down fills below the stop price; positions under one share cannot hold a stop
order and are only checked when the engine runs; nothing exits a position on
its time horizon yet.

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

# Ask before every buy or sell
python -m trading_engine.run --tickers AAPL,MSFT --confirm

# Emergency stop / resume
python -m trading_engine.killswitch on "something looks wrong"
python -m trading_engine.killswitch off
```

The ledger and kill-switch file live in `~/.trading_engine/` (override with
`TRADING_ENGINE_DIR`). Dry runs write to `ledger-dryrun.jsonl`, not `ledger.jsonl`,
so a made-up dry-run account never feeds the real account's drawdown breaker.

## Safety rails

| Rail | Where |
|---|---|
| Live trading refused (non-paper URL or account number) | `broker.AlpacaPaperBroker` |
| 5% stock / 20% sector / 80% gross / 0.5% risk per trade | `risk.RiskEngine` |
| 8% drawdown breaker halts buys (sells still allowed) | `risk.RiskEngine` + ledger high-water mark |
| Stale (>15 min) or invalid quotes rejected | `risk.RiskEngine` |
| Resting GTC stop order per position; stop only moves up | `reconcile.Reconciler`, `stops.StopBook` |
| Unfilled buys count against every cap; ticker with an open order skipped | `risk.RiskEngine`, `runner.Runner` |
| No trading if the broker cannot be reconciled | `runner.Runner` |
| Optional confirmation before each order (`--confirm`) | `runner.Runner` |
| Unknown-sector tickers rejected | `risk.RiskEngine` + `universe.json` |
| One order per ticker per day; `client_order_id` from date + ticker + side, and the broker is checked for an earlier attempt before analysis | `runner.Runner`, `risk.client_order_id` |
| Decision logged before order; hash-chained ledger (catches damage and mid-file edits; not tamper-proof unless you save `ledger head` elsewhere) | `ledger.Ledger` |
| Damaged ledger or 3 errors in a cycle → kill switch | `runner.Runner` |

All limits are in `trading_engine/config.py`. Each decision records
`risk_version` (a hash of the limits), so changing a limit starts a new
experimental version instead of silently mixing results.

## Tests

```bash
pytest tests/trading_engine
```

## Not built yet

- Opportunity screener, Market Regime agent, Strategy agent (holding periods per strategy)
- Exits driven by time horizon (stop-loss exits are built: see above)
- Portfolios B–D (SPY, momentum baseline, AI fixed 10-day horizon) and the evaluation layer
- Scheduling (market-hours cron) and monitoring
