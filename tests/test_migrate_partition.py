"""Tests for the one-shot parquet partitioning migrator."""

import pandas as pd
import pytest

from scripts.migrate_partition import migrate


def _write_flat_station(rainfall_dir, station_id, years):
    """Write a synthetic flat per-station parquet covering the given years."""
    frames = []
    for year in years:
        ts = pd.date_range(
            f"{year}-01-01", f"{year}-12-31 23:00:00",
            freq="1h", tz="Asia/Singapore",
        )
        frames.append(pd.DataFrame({
            "timestamp": ts,
            "reading_value": pd.Series([0.5] * len(ts), dtype="float32"),
        }))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(rainfall_dir / f"{station_id}.parquet", index=False)
    return df


def test_migrator_partitions_two_years(tmp_path):
    rainfall = tmp_path / "rainfall"
    rainfall.mkdir()
    df = _write_flat_station(rainfall, "S00", [2017, 2018])

    migrate(str(rainfall))

    # Original flat file deleted.
    assert not (rainfall / "S00.parquet").exists()
    # Per-year files created.
    assert (rainfall / "S00" / "2017.parquet").exists()
    assert (rainfall / "S00" / "2018.parquet").exists()

    df_2017 = pd.read_parquet(rainfall / "S00" / "2017.parquet")
    df_2018 = pd.read_parquet(rainfall / "S00" / "2018.parquet")
    # All rows accounted for and split by actual timestamp year.
    assert len(df_2017) + len(df_2018) == len(df)
    assert (df_2017["timestamp"].dt.year == 2017).all()
    assert (df_2018["timestamp"].dt.year == 2018).all()


def test_migrator_idempotent_on_already_migrated_station(tmp_path):
    rainfall = tmp_path / "rainfall"
    s00 = rainfall / "S00"
    s00.mkdir(parents=True)
    # Pre-existing partitioned layout, no flat file.
    pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=3, freq="h", tz="Asia/Singapore"),
        "reading_value": pd.Series([1.0, 1.0, 1.0], dtype="float32"),
    }).to_parquet(s00 / "2020.parquet", index=False)

    # Should not raise and should not modify the existing file.
    before = (s00 / "2020.parquet").stat().st_mtime_ns
    migrate(str(rainfall))
    after = (s00 / "2020.parquet").stat().st_mtime_ns
    assert before == after


def test_migrator_leaves_flat_in_place_on_count_mismatch(tmp_path, monkeypatch):
    rainfall = tmp_path / "rainfall"
    rainfall.mkdir()
    _write_flat_station(rainfall, "S00", [2017])

    # Force a verification failure by stubbing the row-count check.
    from scripts import migrate_partition
    monkeypatch.setattr(
        migrate_partition, "_total_row_count",
        lambda paths: -1,
    )

    with pytest.raises(RuntimeError, match="row count"):
        migrate(str(rainfall))

    # Flat file is still there because verification failed.
    assert (rainfall / "S00.parquet").exists()


def test_migrator_handles_empty_directory(tmp_path):
    rainfall = tmp_path / "rainfall"
    rainfall.mkdir()
    # No stations at all — should not raise.
    migrate(str(rainfall))
