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
