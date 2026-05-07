"""Tests for shared helpers in app/queries.py."""

import pytest

from app import queries


@pytest.mark.parametrize("month, label", [
    (1, "NE"),  (2, "NE"),  (3, "NE"),  (12, "NE"),
    (4, "Pre-SW"), (5, "Pre-SW"),
    (6, "SW"),  (7, "SW"),  (8, "SW"),  (9, "SW"),
    (10, "Pre-NE"), (11, "Pre-NE"),
])
def test_season_label_buckets(month, label):
    assert queries._season_label(month) == label


@pytest.mark.parametrize("bad_month", [0, 13, -1, 100])
def test_season_label_rejects_out_of_range(bad_month):
    with pytest.raises(ValueError):
        queries._season_label(bad_month)


def test_all_station_ids_returns_sorted_list(fixture_processed_dir_multi_station):
    queries._load_stations_index.cache_clear()
    ids = queries._all_station_ids()
    assert ids == ["S88", "S99"]


def test_all_station_ids_single_station(fixture_processed_dir):
    queries._load_stations_index.cache_clear()
    ids = queries._all_station_ids()
    assert ids == ["S99"]


def test_cross_station_yearly_totals_includes_both(fixture_processed_dir_multi_station):
    queries._cross_station_yearly_totals.cache_clear()
    totals = queries._cross_station_yearly_totals(2020)
    assert set(totals.keys()) == {"S88", "S99"}
    # S88 readings are 2x S99's → total is also 2x.
    assert abs(totals["S88"] - 2 * totals["S99"]) < 1e-3


def test_cross_station_yearly_totals_skips_missing_year(fixture_processed_dir_multi_station):
    queries._cross_station_yearly_totals.cache_clear()
    # Neither station has 2019 — result should be empty (not raise).
    totals = queries._cross_station_yearly_totals(2019)
    assert totals == {}


def test_regional_series_monthly_year_scoped(fixture_processed_dir_multi_station):
    queries._regional_series.cache_clear()
    s = queries._regional_series(2020, "monthly")
    # FIXTURE_DAYS=200 starting Jan 1 → ~7 months. Mean across S99 (1.0/reading)
    # and S88 (2.0/reading) per day = 1.5 mm/reading × 288 readings/day → 432 mm/day
    # → monthly sum is per-month-day-count × 432, then averaged across the 2 stations.
    assert len(s) >= 6
    assert (s > 0).all()


def test_regional_series_rejects_invalid_mode(fixture_processed_dir_multi_station):
    queries._regional_series.cache_clear()
    with pytest.raises(ValueError):
        queries._regional_series(2020, "weekly")


def test_regional_series_uses_mean_not_sum(fixture_processed_dir_multi_station):
    """Mean of (1.0/reading) and (2.0/reading) sources should be 1.5/reading."""
    queries._regional_series.cache_clear()
    s_daily = queries._regional_series(2020, "daily")
    # 288 readings/day × 1.5 mean = 432 mm/day expected for any covered day.
    assert abs(s_daily.iloc[0] - 432.0) < 1.0
