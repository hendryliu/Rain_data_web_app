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


# These must match the constants in app/queries.py. Kept locally in the test
# file so Step 1 can run (and fail) before the implementation lands.
VALID_MIN = pd.Timestamp("2016-01-01")
VALID_MAX = pd.Timestamp("2024-12-31 23:59:59")


class TestResolveWindow:
    """Verify parameter resolution for the rainfall endpoint."""

    @pytest.fixture
    def sample_df(self):
        """DataFrame with timestamps from 2020-01-01 to 2020-07-18 (200 days)."""
        ts = pd.date_range("2020-01-01", periods=200 * 24 * 12, freq="5min")
        return pd.DataFrame({"timestamp": ts, "reading_value": 1.0})

    def test_no_params_returns_data_range(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(sample_df, None, None, None)
        assert start == pd.Timestamp("2020-01-01 00:00")
        expected_end = pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=5) * (200 * 288 - 1)
        assert end == expected_end

    def test_year_shortcut(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(sample_df, None, None, 2023)
        assert start == pd.Timestamp("2023-01-01 00:00")
        assert end == pd.Timestamp("2023-12-31 23:59:59")

    def test_year_overrides_start_end(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df,
            pd.Timestamp("2020-05-01"),
            pd.Timestamp("2020-06-01"),
            2023,
        )
        assert start == pd.Timestamp("2023-01-01 00:00")
        assert end == pd.Timestamp("2023-12-31 23:59:59")

    def test_partial_start_fills_end_from_data(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df, pd.Timestamp("2020-02-01"), None, None
        )
        assert start == pd.Timestamp("2020-02-01 00:00")
        assert end == sample_df["timestamp"].max()

    def test_partial_end_fills_start_from_data(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df, None, pd.Timestamp("2020-06-01"), None
        )
        assert start == sample_df["timestamp"].min()
        assert end == pd.Timestamp("2020-06-01 00:00")

    def test_window_entirely_before_valid_range_clamps_to_min(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df,
            pd.Timestamp("2013-01-01"),
            pd.Timestamp("2014-01-01"),
            None,
        )
        assert start == VALID_MIN
        assert end == VALID_MIN

    def test_window_entirely_after_valid_range_clamps_to_max(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df,
            pd.Timestamp("2030-01-01"),
            pd.Timestamp("2031-01-01"),
            None,
        )
        assert start == VALID_MAX
        assert end == VALID_MAX

    def test_window_straddling_valid_range_is_clamped(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df,
            pd.Timestamp("2013-01-01"),
            pd.Timestamp("2030-01-01"),
            None,
        )
        assert start == VALID_MIN
        assert end == VALID_MAX
