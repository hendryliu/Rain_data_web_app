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


def test_year_comparison_two_years_returns_grouped_bar(fixture_processed_dir_two_years):
    out = queries.year_comparison("S99", year_a=2020, year_b=2021)
    assert out["chart_type"] == "grouped_bar"
    assert out["data"]["labels"] == queries.MONTH_NAMES
    series = out["data"]["series"]
    assert {s["name"] for s in series} == {"2020", "2021"}
    assert all(len(s["values"]) == 12 for s in series)


def test_year_comparison_missing_year_returns_zeros_for_that_series(fixture_processed_dir):
    # 2019 has no parquet; series should still be 12 entries, all zero.
    out = queries.year_comparison("S99", year_a=2020, year_b=2019)
    s_2019 = next(s for s in out["data"]["series"] if s["name"] == "2019")
    assert s_2019["values"] == [0.0] * 12


def test_station_ranking_orders_by_total_desc(fixture_processed_dir_multi_station):
    queries._cross_station_yearly_totals.cache_clear()
    out = queries.station_ranking(year=2020, n=10)
    assert out["type"] == "table"
    assert out["columns"] == ["Rank", "Station", "Total (mm)"]
    rows = out["rows"]
    # S88 (2.0 mm/reading) ranks above S99 (1.0 mm/reading).
    assert rows[0][1] == "Wettest Test Station"
    assert rows[1][1] == "Synthetic Test Station"


def test_station_ranking_skips_stations_missing_year(fixture_processed_dir_multi_station):
    queries._cross_station_yearly_totals.cache_clear()
    out = queries.station_ranking(year=2019)  # no station has 2019
    assert out["rows"] == []


def test_station_ranking_respects_n(fixture_processed_dir_multi_station):
    queries._cross_station_yearly_totals.cache_clear()
    out = queries.station_ranking(year=2020, n=1)
    assert len(out["rows"]) == 1
    assert out["rows"][0][1] == "Wettest Test Station"
