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
