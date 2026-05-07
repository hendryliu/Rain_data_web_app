"""Sanity tests for the multi-station fixture."""

import json

from app import queries


def test_multi_station_fixture_has_two_stations(fixture_processed_dir_multi_station):
    stations_path = fixture_processed_dir_multi_station / "stations.json"
    stations = json.loads(stations_path.read_text())
    ids = sorted(s["id"] for s in stations)
    assert ids == ["S88", "S99"]


def test_multi_station_fixture_s88_total_is_2x_s99(fixture_processed_dir_multi_station):
    df_s99 = queries._load_station("S99", year=2020)
    df_s88 = queries._load_station("S88", year=2020)
    total_s99 = float(df_s99["reading_value"].sum())
    total_s88 = float(df_s88["reading_value"].sum())
    assert abs(total_s88 - 2 * total_s99) < 1e-3
