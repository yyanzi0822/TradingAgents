"""'Today' is the exchange's day, so a west-coast evening run is not rejected.

At 22:56 in Los Angeles on 2026-11-11 it is already 2026-11-12 in New York. The
trading engine dates its cycle on the New York clock; the analysis graph must
accept that date rather than call it "in the future".
"""

from datetime import UTC, datetime

import pytest

from trading_engine.reconcile import _market_date
from tradingagents.dataflows import date_window
from tradingagents.graph import trading_graph

# 2026-11-11 22:56 Pacific == 2026-11-12 01:56 Eastern == 06:56 UTC
WEST_COAST_EVENING = datetime(2026, 11, 12, 6, 56, tzinfo=UTC)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return WEST_COAST_EVENING.astimezone(tz) if tz else WEST_COAST_EVENING.replace(tzinfo=None)


@pytest.fixture
def west_coast_evening(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    monkeypatch.delenv("TRADINGAGENTS_MARKET_TZ", raising=False)
    monkeypatch.setattr(date_window, "datetime", _FrozenDatetime)


def test_today_is_the_new_york_date(west_coast_evening):
    assert date_window.get_current_date() == "2026-11-12"


def test_engine_trade_date_is_accepted_by_the_graph(west_coast_evening):
    engine_date = _market_date(WEST_COAST_EVENING)
    assert engine_date == "2026-11-12"
    assert trading_graph._validate_trade_date(engine_date) == engine_date


def test_the_next_market_day_is_still_the_future(west_coast_evening):
    with pytest.raises(ValueError, match="future"):
        trading_graph._validate_trade_date("2026-11-13")


def test_market_time_zone_can_be_changed(west_coast_evening, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_MARKET_TZ", "America/Los_Angeles")
    assert date_window.get_current_date() == "2026-11-11"
