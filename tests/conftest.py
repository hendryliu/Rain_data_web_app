"""Test fixtures for the rainfall API.

Builds a synthetic processed/ directory under a tmp dir with one station ('S99')
in the partitioned layout. The default fixture has a single year (2020) with
200 days at 5-minute intervals; an extension fixture adds 2021 for multi-year
tests. Each reading is exactly 1.0 mm so daily sums are 288.0 and hourly sums
are 12.0.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient


FIXTURE_DAYS = 200  # > 180 → default window picks the daily tier
FIXTURE_STATION_ID = "S99"
FIXTURE_STATION_NAME = "Synthetic Test Station"
DEFAULT_YEAR = 2020
EXTRA_YEAR = 2021


def _write_year(year_path, year, days):
    """Write one year's parquet with `days` of 5-minute readings at 1.0 mm each."""
    periods = days * 24 * 12
    start = pd.Timestamp(f"{year}-01-01 00:00:00")
    timestamps = pd.date_range(start, periods=periods, freq="5min", tz="Asia/Singapore")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "reading_value": pd.Series([1.0] * periods, dtype="float32"),
    })
    df.to_parquet(year_path, index=False)


@pytest.fixture
def fixture_processed_dir(tmp_path, monkeypatch):
    """processed/ with one station ('S99') containing only year 2020."""
    processed = tmp_path / "processed"
    rainfall = processed / "rainfall"
    station_dir = rainfall / FIXTURE_STATION_ID
    station_dir.mkdir(parents=True)

    _write_year(station_dir / f"{DEFAULT_YEAR}.parquet", DEFAULT_YEAR, FIXTURE_DAYS)

    stations = [{
        "id": FIXTURE_STATION_ID,
        "name": FIXTURE_STATION_NAME,
        "lng": 103.8,
        "lat": 1.35,
    }]
    (processed / "stations.json").write_text(json.dumps(stations))

    from app import queries
    monkeypatch.setattr(queries, "PROCESSED_DIR", str(processed))
    queries._load_station.cache_clear()
    queries._load_stations_index.cache_clear()

    yield processed

    queries._load_station.cache_clear()
    queries._load_stations_index.cache_clear()


@pytest.fixture
def fixture_processed_dir_two_years(fixture_processed_dir):
    """Same fixture, plus a 2021.parquet file for multi-year tests.

    Note: clears _load_station's LRU after writing the new file, otherwise the
    cached single-year frame from any prior call would mask the new year.
    """
    station_dir = fixture_processed_dir / "rainfall" / FIXTURE_STATION_ID
    _write_year(station_dir / f"{EXTRA_YEAR}.parquet", EXTRA_YEAR, FIXTURE_DAYS)

    from app import queries
    queries._load_station.cache_clear()

    return fixture_processed_dir


@pytest.fixture
def client(fixture_processed_dir, monkeypatch):
    """FastAPI TestClient rooted at the synthetic processed dir."""
    from app.main import app
    from app import main
    monkeypatch.setattr(main, "PROCESSED_DIR", str(fixture_processed_dir))
    return TestClient(app)
