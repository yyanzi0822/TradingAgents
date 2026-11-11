"""Prices for dry runs without an Alpaca key, from Yahoo (already a TradingAgents dependency)."""

from __future__ import annotations


def latest_closes(tickers: list[str]) -> dict[str, float]:
    import yfinance as yf

    prices: dict[str, float] = {}
    for t in tickers:
        hist = yf.Ticker(t).history(period="5d")
        if hist.empty:
            raise RuntimeError(f"no recent price for {t}")
        prices[t.upper()] = float(hist["Close"].iloc[-1])
    return prices
