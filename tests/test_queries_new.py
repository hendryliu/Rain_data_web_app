"""Tests for queries added in the registry expansion (2026-05-07)."""

import pytest

from app import queries


def test_yearly_trend_single_year_no_slope(fixture_processed_dir):
    out = queries.yearly_trend("S99")
    assert out["chart_type"] == "line"
    # Only one (partial) year — trend cannot be computed.
    assert "Need" in out["text"]
    series_names = [s["name"] for s in out["data"]["series"]]
    assert series_names == ["Actual"]


def test_yearly_trend_two_years_includes_trend_series(fixture_processed_dir_two_years):
    out = queries.yearly_trend("S99")
    series_names = [s["name"] for s in out["data"]["series"]]
    # Both years are partial in this fixture (200 days < 300) so still 'Need ≥ 2 full years'.
    # Slope is only fit on full years.
    assert series_names == ["Actual"]


def test_yearly_trend_returns_chart_with_actual_values(fixture_processed_dir_two_years):
    out = queries.yearly_trend("S99")
    actual = next(s for s in out["data"]["series"] if s["name"] == "Actual")
    assert len(actual["values"]) == 2
    assert all(v > 0 for v in actual["values"])
