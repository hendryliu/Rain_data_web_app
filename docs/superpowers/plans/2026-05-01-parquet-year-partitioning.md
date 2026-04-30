# Parquet Year-Partitioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Partition each station's parquet into one file per year so analytical queries read only the years they need.

**Architecture:** Storage layout flips from `processed/rainfall/{id}.parquet` to `processed/rainfall/{id}/{year}.parquet`. `_load_station` accepts an optional `year` arg. Analytical queries pass `year` through; tier helpers (chart endpoint) keep loading all years and are untouched. A one-shot migrator converts pre-existing flat parquets, and `preprocess.py` is updated so fresh runs produce the new layout directly.

**Tech Stack:** Python 3.x, pandas, pyarrow, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-05-01-parquet-year-partitioning-design.md`

---

## File map

| File | Status | Purpose |
|---|---|---|
| `scripts/migrate_partition.py` | new | One-shot: convert each `{id}.parquet` → `{id}/{year}.parquet` |
| `tests/test_migrate_partition.py` | new | Migrator unit tests |
| `app/queries.py` | modify | `_load_station` accepts `year=None`; new `_empty_df`, `_available_years`; analytical queries pass `year` through; `_filter_year` removed |
| `tests/conftest.py` | modify | Fixture switches to partitioned layout; adds 2nd year |
| `tests/test_loader.py` | new | `_load_station` year-scoped behavior |
| `tests/test_queries_year_scoped.py` | new | Functional regression tests for analytical queries on partitioned data |
| `scripts/preprocess.py` | modify | Pass 2 deleted; Pass 1 writes directly to `{id}/{year}.parquet`; year from timestamps |
| `tests/test_preprocess.py` | new | Preprocess integration test against tiny synthetic CSVs |

Tier helpers (`daily_series`, `hourly_series`, `raw_series`) and the `/api/rainfall` endpoint are NOT modified. The 23 existing tests in `test_rainfall_api.py` should pass unchanged after Task 2.

---

## Task 1: Migrator script + tests

**Files:**
- Create: `scripts/migrate_partition.py`
- Create: `tests/test_migrate_partition.py`

This task is fully self-contained — it doesn't import from `app/` and doesn't change any existing file. Running it before Task 2 has no effect on the running app.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_migrate_partition.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_migrate_partition.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.migrate_partition'` (or similar import error).

- [ ] **Step 3: Implement the migrator**

Create `scripts/migrate_partition.py`:

```python
"""One-shot migrator: split flat per-station parquets into year partitions.

For each `processed/rainfall/{id}.parquet`:
  1. Read into memory.
  2. Group by timestamp.dt.year.
  3. Write each year to `processed/rainfall/{id}/{year}.parquet`.
  4. Verify row counts match.
  5. Delete the original flat file.

If `processed/rainfall/{id}/` already exists as a directory, skip the station
(assume already migrated). Failures leave the flat file in place — rerun safely.
"""

import os
import sys

import pandas as pd


def _total_row_count(paths: list[str]) -> int:
    return sum(len(pd.read_parquet(p)) for p in paths)


def migrate(rainfall_dir: str) -> None:
    if not os.path.isdir(rainfall_dir):
        raise FileNotFoundError(f"Not a directory: {rainfall_dir}")

    migrated, skipped, failed = 0, 0, 0

    for entry in sorted(os.listdir(rainfall_dir)):
        full = os.path.join(rainfall_dir, entry)

        # Already a partitioned directory → skip.
        if os.path.isdir(full):
            skipped += 1
            continue

        # Only flat parquets are candidates.
        if not entry.endswith(".parquet"):
            continue

        station_id = entry[: -len(".parquet")]
        station_dir = os.path.join(rainfall_dir, station_id)

        # If both flat file and station dir exist, skip — manual cleanup needed.
        if os.path.isdir(station_dir):
            print(f"  {station_id}: skipped (subdir already exists)")
            skipped += 1
            continue

        try:
            df = pd.read_parquet(full)
            original_count = len(df)

            os.makedirs(station_dir, exist_ok=True)
            written = []
            for year, group in df.groupby(df["timestamp"].dt.year):
                out = os.path.join(station_dir, f"{int(year)}.parquet")
                group.reset_index(drop=True).to_parquet(out, index=False)
                written.append(out)

            new_count = _total_row_count(written)
            if new_count != original_count:
                # Roll back: remove the partial subdir, leave flat file intact.
                for p in written:
                    os.remove(p)
                os.rmdir(station_dir)
                raise RuntimeError(
                    f"{station_id}: row count mismatch "
                    f"(flat={original_count}, partitioned={new_count})"
                )

            os.remove(full)
            migrated += 1
            print(f"  {station_id}: migrated ({len(written)} years, {original_count} rows)")
        except Exception as e:
            failed += 1
            print(f"  {station_id}: FAILED — {e}")
            raise

    print(f"\nDone. migrated={migrated} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "processed", "rainfall"
    )
    migrate(target)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_migrate_partition.py -v`
Expected: PASS — all four tests green.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_partition.py tests/test_migrate_partition.py
git commit -m "Add one-shot migrator for partitioning per-station parquets"
```

---

## Task 2: Loader refactor + fixture migration

**Files:**
- Modify: `tests/conftest.py` (fixture switches to partitioned layout, adds 2nd year)
- Modify: `app/queries.py:12-26` (`_load_station` accepts `year`; cache `maxsize` 32→128)
- Modify: `app/queries.py` (add `_empty_df`, `_available_years`)
- Create: `tests/test_loader.py` (year-scoped behavior tests)

This task changes the loader and the fixture together — they have to flip atomically. Existing 23 tests in `test_rainfall_api.py` should pass after this task because they don't introspect the on-disk shape.

- [ ] **Step 1: Update conftest.py to partitioned layout**

Two fixtures: the default `fixture_processed_dir` keeps the original single-year (2020) shape so existing tests in `test_rainfall_api.py` (which assert exact point counts of 200) continue to pass; `fixture_processed_dir_two_years` extends it with a 2021 file for tests that need multi-year data.

Replace `tests/conftest.py` contents with:

```python
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
```

- [ ] **Step 2: Run existing tests to confirm they break**

Run: `pytest tests/test_rainfall_api.py -v`
Expected: most tests FAIL with `ValueError: No data for station S99` or similar — the loader still expects `S99.parquet` but the fixture now writes `S99/2020.parquet` + `S99/2021.parquet`.

(This is expected — we'll fix it in Step 3.)

- [ ] **Step 3: Update `_load_station` in app/queries.py**

In `app/queries.py`, replace the current `_load_station` (lines 12–26) and add the helpers. Keep the existing module-level `import` block; add `glob` to the imports if not already there.

Imports at top of file:

```python
import glob
import json
import os
from functools import lru_cache

import pandas as pd
```

Replace `_load_station` (and add `_empty_df`, `_available_years` directly below it) with:

```python
def _empty_df() -> pd.DataFrame:
    """Schema-shaped empty frame returned when a year file is absent."""
    return pd.DataFrame({
        "timestamp": pd.Series([], dtype="datetime64[ns]"),
        "reading_value": pd.Series([], dtype="float64"),
    })


@lru_cache(maxsize=128)
def _load_station(station_id: str, year: int | None = None) -> pd.DataFrame:
    """Load a station's readings.

    If `year` is given, returns only that year's data (empty DataFrame if the
    year file is absent for an existing station). If `year` is None, returns
    all years concatenated. Raises ValueError if the station has no data
    directory at all.
    """
    station_dir = os.path.join(PROCESSED_DIR, "rainfall", station_id)
    if not os.path.isdir(station_dir):
        raise ValueError(f"No data for station {station_id}")

    if year is not None:
        path = os.path.join(station_dir, f"{int(year)}.parquet")
        if not os.path.exists(path):
            return _empty_df()
        df = pd.read_parquet(path)
    else:
        files = sorted(glob.glob(os.path.join(station_dir, "*.parquet")))
        if not files:
            return _empty_df()
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)

    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["reading_value"] = df["reading_value"].astype("float64")
    return df


def _available_years(station_id: str) -> list[int]:
    """Return sorted list of years for which this station has a parquet file."""
    station_dir = os.path.join(PROCESSED_DIR, "rainfall", station_id)
    if not os.path.isdir(station_dir):
        raise ValueError(f"No data for station {station_id}")
    years = []
    for fname in os.listdir(station_dir):
        if not fname.endswith(".parquet"):
            continue
        stem = fname[: -len(".parquet")]
        try:
            years.append(int(stem))
        except ValueError:
            continue
    return sorted(years)
```

(Leave `_load_stations_index`, `_filter_year`, and everything below unchanged in this task — `_filter_year` will be removed in Task 3.)

- [ ] **Step 4: Run existing tests to confirm they pass**

Run: `pytest tests/test_rainfall_api.py -v`
Expected: PASS — all 23 tests green. The tier helpers (`daily_series` etc.) call `_load_station(station_id)` with no year, hitting the all-years concat path; with the default single-year fixture they get the 200-day 2020 frame, identical in shape to what the pre-partition fixture produced, so existing assertions (including the two `len(points) == 200` checks) still hold.

- [ ] **Step 5: Write loader tests**

Create `tests/test_loader.py`:

```python
"""Tests for _load_station's partitioned-layout behavior."""

import pandas as pd
import pytest

from app import queries


class TestLoadStation:
    def test_year_scoped_returns_only_that_year(self, fixture_processed_dir):
        df = queries._load_station("S99", year=2020)
        assert len(df) > 0
        assert (df["timestamp"].dt.year == 2020).all()

    def test_other_year_scoped_returns_only_that_year(self, fixture_processed_dir_two_years):
        df = queries._load_station("S99", year=2021)
        assert len(df) > 0
        assert (df["timestamp"].dt.year == 2021).all()

    def test_missing_year_returns_empty_not_error(self, fixture_processed_dir):
        df = queries._load_station("S99", year=9999)
        assert df.empty
        assert list(df.columns) == ["timestamp", "reading_value"]

    def test_unknown_station_raises(self, fixture_processed_dir):
        with pytest.raises(ValueError, match="No data for station UNKNOWN"):
            queries._load_station("UNKNOWN")

    def test_no_year_returns_all_years_concatenated(self, fixture_processed_dir_two_years):
        all_df = queries._load_station("S99")
        df_2020 = queries._load_station("S99", year=2020)
        df_2021 = queries._load_station("S99", year=2021)
        assert len(all_df) == len(df_2020) + len(df_2021)
        years = set(all_df["timestamp"].dt.year.unique())
        assert years == {2020, 2021}

    def test_loader_strips_tz(self, fixture_processed_dir):
        df = queries._load_station("S99", year=2020)
        assert df["timestamp"].dt.tz is None

    def test_loader_casts_to_float64(self, fixture_processed_dir):
        df = queries._load_station("S99", year=2020)
        assert df["reading_value"].dtype == "float64"


class TestAvailableYears:
    def test_single_year_fixture(self, fixture_processed_dir):
        assert queries._available_years("S99") == [2020]

    def test_two_year_fixture_returns_sorted(self, fixture_processed_dir_two_years):
        assert queries._available_years("S99") == [2020, 2021]

    def test_unknown_station_raises(self, fixture_processed_dir):
        with pytest.raises(ValueError, match="No data for station UNKNOWN"):
            queries._available_years("UNKNOWN")
```

- [ ] **Step 6: Run loader tests to verify they pass**

Run: `pytest tests/test_loader.py -v`
Expected: PASS — 10 tests green (7 in TestLoadStation, 3 in TestAvailableYears).

- [ ] **Step 7: Commit**

```bash
git add app/queries.py tests/conftest.py tests/test_loader.py
git commit -m "Switch _load_station to partitioned layout with optional year arg"
```

---

## Task 3: Wire `year` through analytical query call-sites

**Files:**
- Modify: `app/queries.py` (7 analytical functions; `yearly_totals`; remove `_filter_year`)
- Create: `tests/test_queries_year_scoped.py` (functional regression tests)

After this task, analytical queries that take a `year` argument actually load only that year's parquet. Tier helpers (chart endpoint) are still untouched.

- [ ] **Step 1: Write functional tests for analytical queries**

Create `tests/test_queries_year_scoped.py`:

```python
"""Functional tests verifying analytical queries work on the partitioned layout.

The default fixture has S99 with 200 days of data in 2020, every reading 1.0
mm. Total = 200 * 288 = 57,600 mm. Multi-year tests use the two-year fixture.
"""

from app import queries


def test_monthly_totals_returns_only_requested_year(fixture_processed_dir):
    out = queries.monthly_totals("S99", 2020)
    assert "2020" in out["title"]
    # 200 days starting 2020-01-01 → roughly 6.5 months of data; expect 7 month buckets.
    assert 6 <= len(out["data"]["labels"]) <= 7
    # Each reading is 1.0 mm and there are 288 readings/day → values are large positive.
    assert all(v > 0 for v in out["data"]["values"])


def test_monthly_totals_for_year_with_no_file_returns_empty(fixture_processed_dir):
    out = queries.monthly_totals("S99", 2019)
    assert out["data"]["labels"] == []
    assert out["data"]["values"] == []


def test_yearly_totals_lists_all_available_years(fixture_processed_dir_two_years):
    out = queries.yearly_totals("S99")
    # Both fixture years should appear (one or both may be marked partial via "*").
    labels_clean = [lab.rstrip("*") for lab in out["data"]["labels"]]
    assert set(labels_clean) == {"2020", "2021"}


def test_yearly_totals_single_year(fixture_processed_dir):
    out = queries.yearly_totals("S99")
    labels_clean = [lab.rstrip("*") for lab in out["data"]["labels"]]
    assert labels_clean == ["2020"]


def test_hourly_pattern_year_scoped(fixture_processed_dir):
    out = queries.hourly_pattern("S99", year=2020)
    assert "2020" in out["title"]
    assert len(out["data"]["labels"]) == 24  # one bar per hour of day


def test_hourly_pattern_no_year_uses_all_years(fixture_processed_dir):
    out = queries.hourly_pattern("S99")
    # No year in title; still 24 hour buckets.
    assert len(out["data"]["labels"]) == 24


def test_compare_stations_year_scoped(fixture_processed_dir):
    # Comparing the station against itself for one year — both series should match.
    out = queries.compare_stations("S99", "S99", year=2020)
    assert "2020" in out["title"]
    s = out["data"]["series"]
    assert s[0]["values"] == s[1]["values"]


def test_longest_dry_spell_no_dry_days_in_fixture(fixture_processed_dir):
    # Fixture has 1.0 mm every reading → 0 dry days.
    out = queries.longest_dry_spell("S99", year=2020)
    assert "No dry days" in out["text"]


def test_rainiest_week_finds_a_week(fixture_processed_dir):
    out = queries.rainiest_week("S99", year=2020)
    assert "Rainiest 7-day period" in out["text"]


def test_top_rainy_days_returns_n_rows(fixture_processed_dir):
    out = queries.top_rainy_days("S99", year=2020, n=5)
    assert len(out["rows"]) == 5


def test_station_summary_year_scoped(fixture_processed_dir):
    out = queries.station_summary("S99", year=2020)
    assert "2020" in out["title"]
    rows = dict(out["rows"])
    # 200 days × 288 readings/day × 1.0 mm = 57,600 mm total.
    assert "57600" in rows["Total Rainfall"] or "57,600" in rows["Total Rainfall"]
```

- [ ] **Step 2: Run tests to verify they fail (or partially pass on accidentally-correct paths)**

Run: `pytest tests/test_queries_year_scoped.py -v`
Expected: most tests pass already because `_filter_year` still works against the all-years concatenation. The point of this task isn't that these tests start failing — it's that we change the queries to load less data while keeping the same behavior. Note any tests that fail; they reveal a real semantic change.

- [ ] **Step 3: Update analytical queries to pass `year` to `_load_station`**

In `app/queries.py`, replace each `_filter_year(_load_station(station_id), year)` call with `_load_station(station_id, year=year)`. Specifically:

```python
# monthly_totals (line ~125)
def monthly_totals(station_id: str, year: int) -> dict:
    df = _load_station(station_id, year=year)
    ...  # rest unchanged

# top_rainy_days (line ~184)
def top_rainy_days(station_id: str, year: int | None = None, n: int = 10) -> dict:
    n = max(1, min(int(n), 100))
    df = _load_station(station_id, year=year)
    ...  # rest unchanged

# compare_stations (line ~200)
def compare_stations(station_id_1: str, station_id_2: str, year: int | None = None) -> dict:
    name1 = _station_name(station_id_1)
    name2 = _station_name(station_id_2)
    df1 = _load_station(station_id_1, year=year)
    df2 = _load_station(station_id_2, year=year)
    ...  # rest unchanged

# longest_dry_spell (line ~232)
def longest_dry_spell(station_id: str, year: int | None = None) -> dict:
    df = _load_station(station_id, year=year)
    ...  # rest unchanged

# station_summary (line ~269)
def station_summary(station_id: str, year: int | None = None) -> dict:
    df = _load_station(station_id, year=year)
    ...  # rest unchanged

# rainiest_week (line ~295)
def rainiest_week(station_id: str, year: int | None = None) -> dict:
    df = _load_station(station_id, year=year)
    ...  # rest unchanged

# hourly_pattern (line ~337)
def hourly_pattern(station_id: str, year: int | None = None) -> dict:
    df = _load_station(station_id, year=year)
    ...  # rest unchanged
```

- [ ] **Step 4: Update `yearly_totals` to enumerate available years**

In `app/queries.py`, replace `yearly_totals` with the per-year version:

```python
def yearly_totals(station_id: str) -> dict:
    years = _available_years(station_id)

    yearly: dict[int, float] = {}
    coverage: dict[int, int] = {}
    for y in years:
        df_y = _load_station(station_id, year=y)
        if len(df_y) == 0:
            continue
        ts = df_y["timestamp"]
        yearly[y] = float(df_y["reading_value"].sum())
        coverage[y] = int(ts.dt.normalize().nunique())

    labels: list[str] = []
    values: list[float] = []
    full_year_totals: list[float] = []
    partial_years: list[int] = []
    for y in sorted(yearly.keys()):
        total = round(yearly[y], 1)
        is_partial = coverage[y] < PARTIAL_YEAR_DAYS
        labels.append(f"{y}*" if is_partial else str(y))
        values.append(total)
        if is_partial:
            partial_years.append(y)
        else:
            full_year_totals.append(total)

    if full_year_totals:
        avg = sum(full_year_totals) / len(full_year_totals)
        text = f"Average (full years only): {avg:.1f} mm/year"
    elif values:
        text = "Not enough complete years to compute an average."
    else:
        text = "No data"

    if partial_years:
        text += (
            f" Partial years marked with * (excluded from average): "
            f"{', '.join(str(y) for y in partial_years)}."
        )

    return {
        "type": "chart",
        "chart_type": "bar",
        "title": f"Yearly Rainfall — {_station_name(station_id)}",
        "data": {"labels": labels, "values": values},
        "text": text,
    }
```

- [ ] **Step 5: Remove the now-unused `_filter_year` helper**

In `app/queries.py`, delete:

```python
def _filter_year(df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    if year is not None:
        return df[df["timestamp"].dt.year == year]
    return df
```

(Lines ~39–42 of the original file. Confirm with `grep -n "_filter_year" app/queries.py` — should return zero hits after removal.)

- [ ] **Step 6: Run all tests to confirm semantics unchanged**

Run: `pytest tests/ -v`
Expected: all tests pass — `test_rainfall_api.py` (23), `test_loader.py` (10), `test_queries_year_scoped.py` (11), `test_migrate_partition.py` (4). Total ~48.

- [ ] **Step 7: Commit**

```bash
git add app/queries.py tests/test_queries_year_scoped.py
git commit -m "Thread year through analytical queries; remove _filter_year"
```

---

## Task 4: Update `preprocess.py` to write partitioned layout directly

**Files:**
- Modify: `scripts/preprocess.py` (delete Pass 2, write directly to `{station}/{year}.parquet`, year from timestamps)
- Create: `tests/test_preprocess.py` (integration test on a tiny synthetic CSV)

After this task, fresh runs of the preprocess pipeline produce the same layout the migrator produces.

- [ ] **Step 1: Write integration test**

Create `tests/test_preprocess.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preprocess.py -v`
Expected: FAIL — the current preprocess uses Pass 2 to merge into `S00.parquet` and uses filename year, so neither test will pass.

- [ ] **Step 3: Rewrite `scripts/preprocess.py`**

Replace the entire file with:

```python
"""Preprocess rainfall CSVs into partitioned per-station Parquet files.

For each yearly CSV: read once, group by (station_id, timestamp_year), and
write directly to processed/rainfall/{station_id}/{year}.parquet, appending
across CSV files when the same (station, year) bucket appears more than once
(e.g., a CSV's rows spilling into the previous or next calendar year).
"""

import glob
import json
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
RAINFALL_DIR = os.path.join(OUTPUT_DIR, "rainfall")


def main():
    os.makedirs(RAINFALL_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        print("No CSV files found in data/")
        return

    print(f"Found {len(csv_files)} CSV files")

    stations: dict[str, dict] = {}
    # (station_id, year) -> list of dataframes, flushed and concatenated at end
    buckets: dict[tuple[str, int], list[pd.DataFrame]] = {}

    for csv_path in csv_files:
        print(f"Reading {os.path.basename(csv_path)}...")
        df = pd.read_csv(
            csv_path,
            usecols=[
                "timestamp",
                "station_id",
                "station_name",
                "location_longitude",
                "location_latitude",
                "reading_value",
            ],
            dtype={
                "station_id": "string",
                "station_name": "string",
                "location_longitude": "float64",
                "location_latitude": "float64",
                "reading_value": "float32",
            },
        )

        for _, row in (
            df[["station_id", "station_name", "location_longitude", "location_latitude"]]
            .drop_duplicates(subset="station_id", keep="last")
            .iterrows()
        ):
            stations[row["station_id"]] = {
                "id": row["station_id"],
                "name": row["station_name"],
                "lng": row["location_longitude"],
                "lat": row["location_latitude"],
            }

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[["station_id", "timestamp", "reading_value"]]
        df["__year"] = df["timestamp"].dt.year

        for (station_id, year), group in df.groupby(["station_id", "__year"], sort=False):
            buckets.setdefault((station_id, int(year)), []).append(
                group[["timestamp", "reading_value"]].reset_index(drop=True)
            )

        del df

    # Flush each bucket once.
    print(f"Writing {len(buckets)} (station, year) parquet files...")
    written = 0
    for (station_id, year), frames in buckets.items():
        station_dir = os.path.join(RAINFALL_DIR, station_id)
        os.makedirs(station_dir, exist_ok=True)
        combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        out = os.path.join(station_dir, f"{year}.parquet")
        combined.to_parquet(out, index=False)
        written += 1
        if written % 50 == 0:
            print(f"  wrote {written}/{len(buckets)}")

    stations_list = sorted(stations.values(), key=lambda s: s["name"])
    stations_path = os.path.join(OUTPUT_DIR, "stations.json")
    with open(stations_path, "w") as f:
        json.dump(stations_list, f)
    print(f"Wrote {len(stations_list)} stations to {stations_path}")
    print(f"Done. Wrote {written} parquet files.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preprocess.py -v`
Expected: PASS — both integration tests green.

- [ ] **Step 5: Run the whole test suite**

Run: `pytest tests/ -v`
Expected: PASS — all tests green (~50 total: 23 + 10 + 11 + 4 + 2).

- [ ] **Step 6: Commit**

```bash
git add scripts/preprocess.py tests/test_preprocess.py
git commit -m "Preprocess writes partitioned layout directly; year from timestamps"
```

---

## Manual smoke (post-merge, against real data)

After merging the branch:

1. Run the migrator against the real `processed/` directory:
   ```bash
   python -m scripts.migrate_partition processed/rainfall
   ```
   Expect log output `migrated=N skipped=0 failed=0` for whatever station count is present.
2. Start the app: `python -m app.main` (or the project's standard run command).
3. Click a station on the map → bar chart loads (daily resolution by default).
4. Pick a year from the dropdown → chart switches to hourly, station detail visible.
5. In the chat sidebar, run a few pre-built queries: `monthly_totals` for 2020, `hourly_pattern`, `longest_dry_spell`. Each should return plausible numbers.
6. Confirm an unknown station ID still 404s (try `curl http://localhost:8000/api/rainfall/UNKNOWN`).

If anything looks wrong, check that the migrator actually deleted the old flat parquets — running with old + new files coexisting is the most likely smoke-test failure mode.

---

## Out of scope / explicitly NOT changed

- `daily_series`, `hourly_series`, `raw_series` and the `/api/rainfall` chart endpoint.
- Pre-computed aggregation tiers at preprocess time.
- Pyarrow predicate pushdown / partition discovery (could replace explicit `glob` + `concat` later).
- Multi-user deployment caching concerns.
