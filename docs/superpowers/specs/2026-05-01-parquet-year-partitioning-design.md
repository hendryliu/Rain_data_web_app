# Parquet Year-Partitioning — Design

**Date:** 2026-05-01
**Status:** Approved
**Scope:** `processed/rainfall/*.parquet` storage layout, `app/queries.py` loader, analytical query path, preprocess pipeline + one-shot migrator

## Problem

Every analytical query in `app/queries.py` (monthly_totals, yearly_totals, hourly_pattern, longest_dry_spell, rainiest_week, compare_stations, ...) starts by calling `_load_station(station_id)`, which reads the entire per-station parquet (~9 years, ~950k rows, ~25 MB on disk). Most of those queries then immediately filter to a single year. The wasted I/O is the dominant cost on a cold cache, and it scales linearly with how many years of data exist — so adding more years makes existing queries slower.

The rainfall-downsampling work fixed the chart endpoint, but it deliberately left the analytical path alone. This change addresses that path.

## Goals

1. Year-scoped analytical queries read only the year(s) they need.
2. Existing analytical query semantics unchanged — no behavior changes visible to the LLM-driven chat or to the manual query callers.
3. One migration story for users with already-processed data; one preprocessing story going forward; both produce identical output.
4. Tier helpers (`daily_series`, `hourly_series`, `raw_series`) are intentionally out of scope — they read all years anyway, so they don't benefit, and changing them risks regressions in the rainfall endpoint we just stabilized.

## Non-goals

- Changing the rainfall chart endpoint or its tier helpers.
- Pre-computing aggregations (already tracked separately).
- Switching to a different storage engine (DuckDB / SQLite considered, rejected — parquet directory layout is enough for this workload and avoids adding a dependency).
- Multi-station files or year-keyed multi-station bundles.

## Design

### Section 1 — Storage layout

Flat directory per station, one parquet per year of data:

```
processed/
  rainfall/
    S06/
      2017.parquet
      2018.parquet
      ...
      2024.parquet
    S07/
      ...
    S99/
      2020.parquet      (test fixture)
  stations.json         (unchanged)
```

- One subdirectory per station; the subdirectory's existence is the "station has data" signal.
- One `{year}.parquet` per year in which that station has at least one reading. Years with zero readings are absent files (not zero-row files).
- Schema unchanged: `timestamp` datetime64 (tz preserved as written by pyarrow), `reading_value` float32, plus identifying columns. The loader is still responsible for stripping tz and casting `reading_value` to float64 at read time.
- `stations.json` unchanged.

### Section 2 — Preprocess + migrator

**`scripts/preprocess.py` changes:**

- Delete Pass 2 entirely. Pass 1 already produces per-(station, year) parquets — wire its output path directly to `processed/rainfall/{station_id}/{year}.parquet` and stop using `_tmp/`.
- Switch year extraction from `year_from_filename(path)` to year-from-actual-timestamp. This handles edge rows where a 2018 CSV contains a few late-2017 readings, and matches migrator behavior so both paths produce identical output.
- Schema unchanged.

**`scripts/migrate_partition.py` (new, one-shot):**

For each `processed/rainfall/{id}.parquet`:

1. Read into DataFrame.
2. Group by `timestamp.dt.year`.
3. Write each group to `processed/rainfall/{id}/{year}.parquet`.
4. Verify summed row counts across new files equals the original row count.
5. Only after verification: delete the original `{id}.parquet`.

Idempotency: if `processed/rainfall/{id}/` already exists as a directory, skip the station (assume already migrated). Log `migrated/skipped/failed` per station.

Safety: write-then-verify-then-delete order means an interrupted run leaves the old file in place; rerunning resumes cleanly.

### Section 3 — Loader API + call-site updates

**`app/queries.py` — `_load_station` signature:**

```python
@lru_cache(maxsize=128)
def _load_station(station_id: str, year: int | None = None) -> pd.DataFrame:
    station_dir = os.path.join(PROCESSED_DIR, "rainfall", station_id)
    if not os.path.isdir(station_dir):
        raise ValueError(f"No data for station {station_id}")

    if year is not None:
        path = os.path.join(station_dir, f"{year}.parquet")
        if not os.path.exists(path):
            return _empty_df()             # missing year → empty, not error
        df = pd.read_parquet(path)
    else:
        files = sorted(glob(os.path.join(station_dir, "*.parquet")))
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)

    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["reading_value"] = df["reading_value"].astype("float64")
    return df
```

`maxsize` bumped from 32 → 128 to accommodate per-(station, year) keys (~91 stations × ~9 years + per-station "all years" entries).

**Call-site updates — analytical query path only:**

- `monthly_totals(station_id, year)` → `_load_station(station_id, year=year)` instead of filtering after a full load.
- `yearly_totals(station_id)` → call `_load_station(station_id, year=y)` per year and aggregate.
- `hourly_pattern(station_id, year=None)` → `_load_station(station_id, year=year)`.
- `longest_dry_spell(station_id, year=None)` and `rainiest_week(station_id, year=None)` → same.
- `compare_stations(...)` and other multi-station helpers → pass `year` through.

**Tier helpers untouched.** `daily_series`, `hourly_series`, `raw_series` continue calling `_load_station(station_id)` (loads all years). The rainfall chart endpoint behavior is unchanged.

### Section 4 — Testing

**Unit tests (extend `tests/test_rainfall_api.py` or add a new file):**

- `_load_station(id, year=2020)` returns only that year's rows.
- `_load_station(id, year=9999)` returns empty DataFrame (missing-year is non-error).
- `_load_station("UNKNOWN")` still raises `ValueError`.
- `_load_station(id)` (no year) returns concatenation of all available years; row-count equals sum of per-year files.
- Loader strips tz and casts to float64 (regression on the smoke-test bugs).

**Fixture update (`tests/conftest.py`):**

- Switch S99 fixture from `processed/rainfall/S99.parquet` to `processed/rainfall/S99/2020.parquet`.
- Add `processed/rainfall/S99/2021.parquet` so the multi-year concat path is actually exercised.
- Existing 23 tests should pass unchanged — they don't introspect on-disk shape.

**Migrator test (`tests/test_migrate_partition.py`, new):**

- Build a temp dir with one synthetic `S00.parquet` spanning two years.
- Run the migrator.
- Assert: `S00/2017.parquet` and `S00/2018.parquet` exist with correct rows; original `S00.parquet` deleted.
- Run again → idempotency check, no error, no changes.

**Manual smoke (post-merge, against real data):**

1. Run migrator against `processed/`.
2. `python -m app.main` → click a station → daily chart loads.
3. Pick a single year via dropdown → chart switches.
4. Pre-built chat queries: monthly totals 2020, hourly pattern, dry spell. Verify each returns plausible numbers.
5. Unknown station ID still 404s.

## Error handling

| Condition | Behavior |
|---|---|
| Station directory missing | `ValueError` from `_load_station` (existing 404 path in endpoint preserved) |
| Year file missing for an existing station | Empty DataFrame returned; queries see "no data this year" rather than an error |
| Migrator: row-count mismatch after writing partitions | Abort that station, leave original file in place, log failure |
| Migrator: station subdir already exists | Skip station, log "already migrated" |
| Preprocess: row's timestamp year disagrees with filename year | Honor the timestamp; row goes into its actual year's file |

## Scope summary

| File | Change |
|---|---|
| `scripts/preprocess.py` | Delete Pass 2; Pass 1 writes directly to `{station}/{year}.parquet`; year extracted from timestamps |
| `scripts/migrate_partition.py` | New, one-shot migrator |
| `app/queries.py` | `_load_station` accepts `year`; analytical query call-sites updated; cache `maxsize` 32 → 128 |
| `tests/conftest.py` | Fixture switches to subdir layout; adds a second year |
| `tests/test_rainfall_api.py` | Add loader tests; existing tests unchanged |
| `tests/test_migrate_partition.py` | New |

No changes to the rainfall chart endpoint, the LLM integration, or the frontend.

## Out of scope / follow-ups

- Pre-computed aggregation tiers at preprocess time (separate issue).
- Pyarrow predicate pushdown / partition discovery API (could replace the explicit `glob` + `concat` once the directory layout is stable).
- Multi-user deployment concerns (cache memory, stale-cache invalidation when files change underfoot).
