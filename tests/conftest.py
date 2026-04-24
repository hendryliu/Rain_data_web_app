"""Test fixtures for the rainfall API.

Builds a synthetic per-station parquet under a tmp directory and monkeypatches
app.queries.PROCESSED_DIR so the production loader reads from it. One station
('S99') with 200 days of data at 5-minute intervals; each reading is exactly
1.0 mm, so daily sums are 288.0 and hourly sums are 12.0.
"""

import json
import os
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient


FIXTURE_START = pd.Timestamp("2020-01-01 00:00:00")
FIXTURE_DAYS = 200  # Spans > 180 so default window hits the daily tier.
FIXTURE_STATION_ID = "S99"
FIXTURE_STATION_NAME = "Synthetic Test Station"


@pytest.fixture
def fixture_processed_dir(tmp_path, monkeypatch):
    """Build a processed/ directory with one synthetic station."""
    processed = tmp_path / "processed"
    rainfall = processed / "rainfall"
    rainfall.mkdir(parents=True)

    # 200 days * 24 hours * 12 readings/hour = 57,600 rows
    periods = FIXTURE_DAYS * 24 * 12
    timestamps = pd.date_range(FIXTURE_START, periods=periods, freq="5min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "reading_value": [1.0] * periods,
    })
    df.to_parquet(rainfall / f"{FIXTURE_STATION_ID}.parquet", index=False)

    stations = [{
        "id": FIXTURE_STATION_ID,
        "name": FIXTURE_STATION_NAME,
        "lng": 103.8,
        "lat": 1.35,
    }]
    (processed / "stations.json").write_text(json.dumps(stations))

    from app import queries
    monkeypatch.setattr(queries, "PROCESSED_DIR", str(processed))
    # Clear caches that might hold references to the real processed dir.
    queries._load_station.cache_clear()
    queries._load_stations_index.cache_clear()

    yield processed

    # Post-test cleanup: clear caches again so the next test starts fresh.
    queries._load_station.cache_clear()
    queries._load_stations_index.cache_clear()


@pytest.fixture
def client(fixture_processed_dir):
    """FastAPI TestClient rooted at the synthetic processed dir."""
    from app.main import app
    # Also patch main's PROCESSED_DIR in case the endpoint uses it directly.
    from app import main
    main.PROCESSED_DIR = str(fixture_processed_dir)
    return TestClient(app)
