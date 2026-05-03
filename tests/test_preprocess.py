"""Integration test for preprocess against synthetic CSVs."""

import os

import pandas as pd
import pytest


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def preprocess_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "processed"
    data_dir.mkdir()
    out_dir.mkdir()

    from scripts import preprocess
    monkeypatch.setattr(preprocess, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(preprocess, "RAINFALL_DIR", str(out_dir / "rainfall"))
    return data_dir, out_dir


def test_preprocess_writes_partitioned_layout(preprocess_env):
    data_dir, out_dir = preprocess_env
    rows = []
    for hour in range(0, 48):
        # Two days spanning 2020-12-31 and 2021-01-01
        ts = pd.Timestamp("2020-12-31 00:00:00") + pd.Timedelta(hours=hour)
        rows.append({
            "timestamp": ts.isoformat(),
            "station_id": "S00",
            "station_name": "Test",
            "location_longitude": 103.8,
            "location_latitude": 1.35,
            "reading_value": 0.1,
        })
    _write_csv(data_dir / "data_2020.csv", rows)

    from scripts.preprocess import main
    main()

    # Both years are produced even though the CSV is named "2020".
    assert (out_dir / "rainfall" / "S00" / "2020.parquet").exists()
    assert (out_dir / "rainfall" / "S00" / "2021.parquet").exists()
    # No flat per-station file is produced.
    assert not (out_dir / "rainfall" / "S00.parquet").exists()
    # No _tmp directory left behind.
    assert not (out_dir / "_tmp").exists()
    # stations.json written.
    assert (out_dir / "stations.json").exists()


def test_preprocess_year_comes_from_timestamps(preprocess_env):
    data_dir, out_dir = preprocess_env
    # Filename says 2018, but every timestamp is in 2019.
    rows = [{
        "timestamp": pd.Timestamp(f"2019-06-{day:02d} 12:00:00").isoformat(),
        "station_id": "S00",
        "station_name": "Test",
        "location_longitude": 103.8,
        "location_latitude": 1.35,
        "reading_value": 1.0,
    } for day in range(1, 6)]
    _write_csv(data_dir / "data_2018.csv", rows)

    from scripts.preprocess import main
    main()

    assert (out_dir / "rainfall" / "S00" / "2019.parquet").exists()
    assert not (out_dir / "rainfall" / "S00" / "2018.parquet").exists()
