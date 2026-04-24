"""Tests for the /api/rainfall endpoint and its pure-function helpers."""

import pandas as pd
import pytest


class TestPickTier:
    """Verify tier selection at every threshold boundary."""

    def test_window_over_180_days_is_daily(self):
        from app.queries import pick_tier
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=365)
        assert pick_tier(start, end) == "daily"

    def test_window_exactly_181_days_is_daily(self):
        from app.queries import pick_tier
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=181)
        assert pick_tier(start, end) == "daily"

    def test_window_exactly_180_days_is_hourly(self):
        from app.queries import pick_tier
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=180)
        assert pick_tier(start, end) == "hourly"

    def test_window_exactly_7_days_is_hourly(self):
        from app.queries import pick_tier
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=7)
        assert pick_tier(start, end) == "hourly"

    def test_window_exactly_6_days_is_raw(self):
        from app.queries import pick_tier
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=6)
        assert pick_tier(start, end) == "raw"

    def test_single_day_is_raw(self):
        from app.queries import pick_tier
        ts = pd.Timestamp("2020-01-01")
        assert pick_tier(ts, ts) == "raw"
