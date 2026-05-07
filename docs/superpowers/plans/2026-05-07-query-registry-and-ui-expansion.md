# Query Registry Expansion + UI Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six new analytical queries (yearly_trend, year_comparison, station_ranking, regional_total, monsoon_breakdown, season_comparison) and update the frontend with tabbed quick-queries, a generic schema-driven parameter modal, and station search above the map.

**Architecture:** Backend queries are added to `app/queries.py` alongside four shared helpers. Each query returns the existing `dict` result shape (chart/table/text) — a new `chart_type: "line"` is introduced for time-series outputs. The frontend `index.html` is extended with: tab strip + tab-aware chip rendering, a single param-modal helper that reads the registry's `params` schema, a station search input bound to the existing `allStations` array, and a small Plotly branch for the line type. A new `GET /api/years` endpoint replaces the hardcoded year `<option>` list.

**Tech Stack:** Python 3.11, FastAPI, pandas, pyarrow (Parquet), pytest. Frontend: vanilla JS + Leaflet + Plotly (CDN). uv for env management.

**Spec:** `docs/superpowers/specs/2026-05-07-query-registry-and-ui-expansion-design.md`

---

## Conventions

- All `pytest` commands are run via `uv run pytest`.
- Frontend tasks have no automated tests — each ends with explicit manual browser verification steps.
- Commit messages use imperative mood, no `feat:`/`fix:` prefixes (matching repo style). Each commit is a working unit on its own.
- Working directory is the repo root: `C:\Users\Jian\github\Rain_data_web_app`.
- All work is on the `develop` branch.

---

## Task 1: Extend test fixture with a multi-station fixture

Add a second station `S88` (with two years) plus a parameterized `_write_year` so we can give it a different reading value for ranking tests.

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/test_fixture_multi_station.py` (sanity check)

- [ ] **Step 1: Update `_write_year` to accept a `value` parameter (default 1.0)**

In `tests/conftest.py`, replace the `_write_year` function:

```python
def _write_year(year_path, year, days, value: float = 1.0):
    """Write one year's parquet with `days` of 5-minute readings at `value` mm each."""
    periods = days * 24 * 12
    start = pd.Timestamp(f"{year}-01-01 00:00:00")
    timestamps = pd.date_range(start, periods=periods, freq="5min", tz="Asia/Singapore")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "reading_value": pd.Series([value] * periods, dtype="float32"),
    })
    df.to_parquet(year_path, index=False)
```

- [ ] **Step 2: Add `EXTRA_STATION_ID`/`EXTRA_STATION_NAME` constants and the new fixture**

After the existing constants block in `tests/conftest.py`:

```python
EXTRA_STATION_ID = "S88"
EXTRA_STATION_NAME = "Wettest Test Station"
EXTRA_STATION_VALUE = 2.0  # readings here are 2.0 mm so totals exceed S99
```

After the `fixture_processed_dir_two_years` fixture, add:

```python
@pytest.fixture
def fixture_processed_dir_multi_station(fixture_processed_dir_two_years):
    """fixture_processed_dir_two_years plus station S88 with 2020 and 2021 at 2.0 mm.

    S88 totals are 2x S99's, which lets us test rankings deterministically.
    """
    rainfall = fixture_processed_dir_two_years / "rainfall"
    extra_dir = rainfall / EXTRA_STATION_ID
    extra_dir.mkdir(parents=True)
    _write_year(extra_dir / f"{DEFAULT_YEAR}.parquet", DEFAULT_YEAR, FIXTURE_DAYS, value=EXTRA_STATION_VALUE)
    _write_year(extra_dir / f"{EXTRA_YEAR}.parquet", EXTRA_YEAR, FIXTURE_DAYS, value=EXTRA_STATION_VALUE)

    stations_path = fixture_processed_dir_two_years / "stations.json"
    stations = json.loads(stations_path.read_text())
    stations.append({
        "id": EXTRA_STATION_ID,
        "name": EXTRA_STATION_NAME,
        "lng": 103.85,
        "lat": 1.30,
    })
    stations_path.write_text(json.dumps(stations))

    from app import queries
    queries._load_station.cache_clear()
    queries._load_stations_index.cache_clear()

    return fixture_processed_dir_two_years
```

- [ ] **Step 3: Write a sanity test for the new fixture**

Create `tests/test_fixture_multi_station.py`:

```python
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
```

- [ ] **Step 4: Run the sanity test**

Run: `uv run pytest tests/test_fixture_multi_station.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `uv run pytest -v`
Expected: all green; existing tests still pass with the parameterized `_write_year`.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_fixture_multi_station.py
git commit -m "Add multi-station test fixture with parameterized reading value"
```

---

## Task 2: `_season_label` helper

Maps a calendar month to one of `"NE" / "Pre-SW" / "SW" / "Pre-NE"`.

**Files:**
- Modify: `app/queries.py` (add helper)
- Test: `tests/test_query_helpers.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_query_helpers.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_query_helpers.py -v`
Expected: FAIL with `AttributeError: module 'app.queries' has no attribute '_season_label'`.

- [ ] **Step 3: Implement `_season_label` in `app/queries.py`**

After the `MONTH_NAMES` constant (around line 81), add:

```python
SEASON_BUCKETS = {
    "NE": (1, 2, 3, 12),
    "Pre-SW": (4, 5),
    "SW": (6, 7, 8, 9),
    "Pre-NE": (10, 11),
}


def _season_label(month: int) -> str:
    """Map a calendar month (1-12) to a Singapore monsoon bucket.

    Convention: NE includes Dec of the same year (no cross-year span).
    """
    for label, months in SEASON_BUCKETS.items():
        if month in months:
            return label
    raise ValueError(f"month must be 1-12, got {month}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_query_helpers.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_query_helpers.py
git commit -m "Add _season_label helper for monsoon bucket classification"
```

---

## Task 3: `_all_station_ids` helper

Returns the list of station IDs from `stations.json`.

**Files:**
- Modify: `app/queries.py`
- Test: `tests/test_query_helpers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_query_helpers.py`:

```python
def test_all_station_ids_returns_sorted_list(fixture_processed_dir_multi_station):
    queries._load_stations_index.cache_clear()
    ids = queries._all_station_ids()
    assert ids == ["S88", "S99"]


def test_all_station_ids_single_station(fixture_processed_dir):
    queries._load_stations_index.cache_clear()
    ids = queries._all_station_ids()
    assert ids == ["S99"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_query_helpers.py::test_all_station_ids_returns_sorted_list -v`
Expected: FAIL with `AttributeError: module 'app.queries' has no attribute '_all_station_ids'`.

- [ ] **Step 3: Implement `_all_station_ids`**

In `app/queries.py`, after `_load_stations_index`:

```python
def _all_station_ids() -> list[str]:
    """Sorted list of station IDs from stations.json."""
    return sorted(_load_stations_index().keys())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_query_helpers.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_query_helpers.py
git commit -m "Add _all_station_ids helper"
```

---

## Task 4: `_cross_station_yearly_totals` helper

Returns `{station_id: total_mm}` for one year, skipping stations missing that year.

**Files:**
- Modify: `app/queries.py`
- Test: `tests/test_query_helpers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_query_helpers.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_query_helpers.py::test_cross_station_yearly_totals_includes_both -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `_cross_station_yearly_totals`**

In `app/queries.py`, after `_all_station_ids`:

```python
@lru_cache(maxsize=32)
def _cross_station_yearly_totals(year: int) -> dict[str, float]:
    """Total rainfall (mm) per station for the given year.

    Stations whose year-file is absent or empty are skipped (not zero-ranked).
    """
    out: dict[str, float] = {}
    for sid in _all_station_ids():
        try:
            df = _load_station(sid, year=year)
        except ValueError:
            continue
        if len(df) == 0:
            continue
        out[sid] = float(df["reading_value"].sum())
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_query_helpers.py -v`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_query_helpers.py
git commit -m "Add _cross_station_yearly_totals helper"
```

---

## Task 5: `_regional_series` helper

Cross-station mean rainfall as a `pd.Series`, monthly or daily.

**Files:**
- Modify: `app/queries.py`
- Test: `tests/test_query_helpers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_query_helpers.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_query_helpers.py::test_regional_series_monthly_year_scoped -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `_regional_series`**

In `app/queries.py`, after `_cross_station_yearly_totals`:

```python
@lru_cache(maxsize=32)
def _regional_series(year: int | None, mode: str) -> pd.Series:
    """Cross-station mean rainfall for the given year (or all years).

    `mode` is 'monthly' (resample daily totals to month-start sums) or 'daily'
    (raw daily totals). Returns a Series indexed by timestamp; mean is taken
    across stations for each timestamp.
    """
    if mode not in ("monthly", "daily"):
        raise ValueError(f"mode must be 'monthly' or 'daily', got {mode!r}")

    per_station: list[pd.Series] = []
    for sid in _all_station_ids():
        try:
            df = _load_station(sid, year=year)
        except ValueError:
            continue
        if len(df) == 0:
            continue
        s = df.groupby(df["timestamp"].dt.normalize())["reading_value"].sum()
        if mode == "monthly":
            s = s.resample("MS").sum()
        per_station.append(s)

    if not per_station:
        return pd.Series([], dtype="float64")

    aligned = pd.concat(per_station, axis=1)
    return aligned.mean(axis=1)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_query_helpers.py -v`
Expected: 23 passed.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_query_helpers.py
git commit -m "Add _regional_series helper for cross-station means"
```

---

## Task 6: `yearly_trend` query

Returns yearly totals plus a fitted linear trend over full years only.

**Files:**
- Modify: `app/queries.py` (function + registry)
- Test: `tests/test_queries_new.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_queries_new.py`:

```python
"""Tests for queries added in the registry expansion (2026-05-07)."""

import pytest

from app import queries


def test_yearly_trend_single_year_no_slope(fixture_processed_dir):
    out = queries.yearly_trend("S99")
    assert out["chart_type"] == "line"
    # Only one (partial) year — trend cannot be computed.
    assert "Need" in out["text"]
    series_names = [s["name"] for s in out["data"]["series"]]
    assert series_names == ["Actual"]


def test_yearly_trend_two_years_includes_trend_series(fixture_processed_dir_two_years):
    out = queries.yearly_trend("S99")
    series_names = [s["name"] for s in out["data"]["series"]]
    # Both years are partial in this fixture (200 days < 300) so still 'Need ≥ 2 full years'.
    # Slope is only fit on full years.
    assert series_names == ["Actual"]


def test_yearly_trend_returns_chart_with_actual_values(fixture_processed_dir_two_years):
    out = queries.yearly_trend("S99")
    actual = next(s for s in out["data"]["series"] if s["name"] == "Actual")
    assert len(actual["values"]) == 2
    assert all(v > 0 for v in actual["values"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_queries_new.py::test_yearly_trend_single_year_no_slope -v`
Expected: FAIL with `AttributeError: module 'app.queries' has no attribute 'yearly_trend'`.

- [ ] **Step 3: Implement `yearly_trend` and `_linear_fit` helper**

In `app/queries.py`, before `monthly_totals`, add the helper:

```python
def _linear_fit(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and intercept for ys = slope*xs + intercept."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den if den else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept
```

After the existing `yearly_totals` function, add:

```python
def yearly_trend(station_id: str) -> dict:
    """Yearly totals + linear trend (fit only on full years)."""
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

    sorted_years = sorted(yearly.keys())
    if not sorted_years:
        return {"type": "text", "title": "Yearly Trend", "text": "No data."}

    labels = [str(y) for y in sorted_years]
    actuals = [round(yearly[y], 1) for y in sorted_years]

    full_years = [y for y in sorted_years if coverage[y] >= PARTIAL_YEAR_DAYS]
    series = [{"name": "Actual", "values": actuals}]

    if len(full_years) >= 2:
        slope, intercept = _linear_fit(full_years, [yearly[y] for y in full_years])
        fitted = [round(slope * y + intercept, 1) for y in sorted_years]
        series.append({"name": "Trend", "values": fitted})
        text = f"Trend: {slope:+.1f} mm/year (linear fit over {len(full_years)} full years)"
    else:
        text = f"Need ≥ 2 full years to compute a trend (have {len(full_years)})."

    return {
        "type": "chart",
        "chart_type": "line",
        "title": f"Yearly Trend — {_station_name(station_id)}",
        "data": {"labels": labels, "series": series},
        "text": text,
    }
```

Then add the registry entry inside `QUERY_REGISTRY` (alphabetical order is fine; the placement only affects LLM prompt order which is not sensitive):

```python
    "yearly_trend": {
        "function": yearly_trend,
        "description": "Yearly totals plus a linear trend line for one station across all years",
        "params": {
            "station_id": {"type": "str", "required": True},
        },
    },
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_queries_new.py::test_yearly_trend_single_year_no_slope tests/test_queries_new.py::test_yearly_trend_two_years_includes_trend_series tests/test_queries_new.py::test_yearly_trend_returns_chart_with_actual_values -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_queries_new.py
git commit -m "Add yearly_trend query with linear trend fit"
```

---

## Task 7: `year_comparison` query

Side-by-side monthly bars for the same station in two different years.

**Files:**
- Modify: `app/queries.py`
- Test: `tests/test_queries_new.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_queries_new.py`:

```python
def test_year_comparison_two_years_returns_grouped_bar(fixture_processed_dir_two_years):
    out = queries.year_comparison("S99", year_a=2020, year_b=2021)
    assert out["chart_type"] == "grouped_bar"
    assert out["data"]["labels"] == queries.MONTH_NAMES
    series = out["data"]["series"]
    assert {s["name"] for s in series} == {"2020", "2021"}
    assert all(len(s["values"]) == 12 for s in series)


def test_year_comparison_missing_year_returns_zeros_for_that_series(fixture_processed_dir):
    # 2019 has no parquet; series should still be 12 entries, all zero.
    out = queries.year_comparison("S99", year_a=2020, year_b=2019)
    s_2019 = next(s for s in out["data"]["series"] if s["name"] == "2019")
    assert s_2019["values"] == [0.0] * 12
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_queries_new.py::test_year_comparison_two_years_returns_grouped_bar -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `year_comparison` and register**

In `app/queries.py`, after `compare_stations`:

```python
def year_comparison(station_id: str, year_a: int, year_b: int) -> dict:
    """Compare one station's monthly rainfall across two years."""
    name = _station_name(station_id)

    def _monthly(year: int) -> list[float]:
        df = _load_station(station_id, year=year)
        if len(df) == 0:
            return [0.0] * 12
        m = df.groupby(df["timestamp"].dt.month)["reading_value"].sum()
        return [round(float(m.get(i + 1, 0.0)), 1) for i in range(12)]

    values_a = _monthly(year_a)
    values_b = _monthly(year_b)

    return {
        "type": "chart",
        "chart_type": "grouped_bar",
        "title": f"{name}: {year_a} vs {year_b}",
        "data": {
            "labels": MONTH_NAMES,
            "series": [
                {"name": str(year_a), "values": values_a},
                {"name": str(year_b), "values": values_b},
            ],
        },
        "text": f"Total — {year_a}: {sum(values_a):.1f} mm, {year_b}: {sum(values_b):.1f} mm",
    }
```

Add to `QUERY_REGISTRY`:

```python
    "year_comparison": {
        "function": year_comparison,
        "description": "Compare monthly rainfall for one station across two years",
        "params": {
            "station_id": {"type": "str", "required": True},
            "year_a": {"type": "int", "required": True},
            "year_b": {"type": "int", "required": True},
        },
    },
```

Note: `year_a`/`year_b` need year-range validation. The existing `execute_query` validator only checks the literal key `"year"`. Update the validator:

In `execute_query`, replace the year-range block:

```python
    # Validate year range — applies to any param whose name starts with 'year'.
    for name, val in coerced.items():
        if name.startswith("year") and val is not None:
            if not (2016 <= int(val) <= 2024):
                raise ValueError(f"{name} must be between 2016 and 2024")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_queries_new.py -v`
Expected: all yearly_trend + year_comparison tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_queries_new.py
git commit -m "Add year_comparison query and broaden year-range validation"
```

---

## Task 8: `station_ranking` query

Ranks all stations by total rainfall for a given year.

**Files:**
- Modify: `app/queries.py`
- Test: `tests/test_queries_new.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_queries_new.py`:

```python
def test_station_ranking_orders_by_total_desc(fixture_processed_dir_multi_station):
    queries._cross_station_yearly_totals.cache_clear()
    out = queries.station_ranking(year=2020, n=10)
    assert out["type"] == "table"
    assert out["columns"] == ["Rank", "Station", "Total (mm)"]
    rows = out["rows"]
    # S88 (2.0 mm/reading) ranks above S99 (1.0 mm/reading).
    assert rows[0][1] == "Wettest Test Station"
    assert rows[1][1] == "Synthetic Test Station"


def test_station_ranking_skips_stations_missing_year(fixture_processed_dir_multi_station):
    queries._cross_station_yearly_totals.cache_clear()
    out = queries.station_ranking(year=2019)  # no station has 2019
    assert out["rows"] == []


def test_station_ranking_respects_n(fixture_processed_dir_multi_station):
    queries._cross_station_yearly_totals.cache_clear()
    out = queries.station_ranking(year=2020, n=1)
    assert len(out["rows"]) == 1
    assert out["rows"][0][1] == "Wettest Test Station"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_queries_new.py::test_station_ranking_orders_by_total_desc -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `station_ranking` and register**

In `app/queries.py`, after `year_comparison`:

```python
def station_ranking(year: int, n: int = 20) -> dict:
    """Rank all stations by total rainfall for a given year."""
    n = max(1, min(int(n), 100))
    totals = _cross_station_yearly_totals(year)
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:n]

    rows = [
        [rank, _station_name(sid), round(total, 1)]
        for rank, (sid, total) in enumerate(ranked, start=1)
    ]

    if rows:
        text = (
            f"Rainiest in {year}: {rows[0][1]} ({rows[0][2]} mm) "
            f"across {len(totals)} stations with data."
        )
    else:
        text = f"No stations have data for {year}."

    return {
        "type": "table",
        "title": f"Station Ranking — {year}",
        "columns": ["Rank", "Station", "Total (mm)"],
        "rows": rows,
        "text": text,
    }
```

Add to `QUERY_REGISTRY`:

```python
    "station_ranking": {
        "function": station_ranking,
        "description": "Rank all stations by total rainfall for a given year",
        "params": {
            "year": {"type": "int", "required": True},
            "n": {"type": "int", "required": False, "default": 20},
        },
    },
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_queries_new.py -v`
Expected: all tests so far pass.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_queries_new.py
git commit -m "Add station_ranking query"
```

---

## Task 9: `regional_total` query

Cross-station mean rainfall as a time series (monthly or daily), one or all years.

**Files:**
- Modify: `app/queries.py`
- Test: `tests/test_queries_new.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_queries_new.py`:

```python
def test_regional_total_year_scoped_monthly(fixture_processed_dir_multi_station):
    queries._regional_series.cache_clear()
    out = queries.regional_total(year=2020, mode="monthly")
    assert out["chart_type"] == "line"
    assert "2020" in out["title"]
    series = out["data"]["series"]
    assert len(series) == 1
    assert series[0]["name"].lower().startswith("all")
    # 200 days starting Jan → roughly 7 month buckets.
    assert 6 <= len(out["data"]["labels"]) <= 7
    assert all(v > 0 for v in series[0]["values"])


def test_regional_total_invalid_mode_raises(fixture_processed_dir_multi_station):
    queries._regional_series.cache_clear()
    with pytest.raises(ValueError):
        queries.regional_total(year=2020, mode="weekly")


def test_regional_total_no_data_returns_text(fixture_processed_dir_multi_station):
    queries._regional_series.cache_clear()
    out = queries.regional_total(year=2019, mode="monthly")
    assert out["type"] == "text"
    assert "No data" in out["text"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_queries_new.py::test_regional_total_year_scoped_monthly -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `regional_total` and register**

In `app/queries.py`, after `station_ranking`:

```python
def regional_total(year: int | None = None, mode: str = "monthly") -> dict:
    """All-Singapore mean rainfall over time, monthly or daily."""
    series = _regional_series(year, mode)
    if len(series) == 0:
        return {"type": "text", "title": "Regional Total", "text": "No data."}

    if mode == "monthly":
        labels = [ts.strftime("%Y-%m") for ts in series.index]
    else:
        labels = [str(ts.date()) for ts in series.index]
    values = [round(float(v), 2) for v in series.values]

    year_label = f" ({year})" if year else " (all years)"
    n_stations = len(_all_station_ids())
    return {
        "type": "chart",
        "chart_type": "line",
        "title": f"Regional Mean Rainfall{year_label}",
        "data": {
            "labels": labels,
            "series": [{"name": "All-SG mean", "values": values}],
        },
        "text": f"Mean across {n_stations} stations.",
    }
```

Add to `QUERY_REGISTRY`:

```python
    "regional_total": {
        "function": regional_total,
        "description": "All-Singapore mean rainfall over time across stations; mode is 'monthly' or 'daily'",
        "params": {
            "year": {"type": "int", "required": False},
            "mode": {"type": "str", "required": False, "default": "monthly"},
        },
    },
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_queries_new.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_queries_new.py
git commit -m "Add regional_total query for all-SG cross-station mean"
```

---

## Task 10: `monsoon_breakdown` query

For one station and one year, total rainfall in each of NE / Pre-SW / SW / Pre-NE.

**Files:**
- Modify: `app/queries.py`
- Test: `tests/test_queries_new.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_queries_new.py`:

```python
def test_monsoon_breakdown_returns_four_buckets(fixture_processed_dir):
    out = queries.monsoon_breakdown("S99", year=2020)
    assert out["chart_type"] == "bar"
    assert out["data"]["labels"] == ["NE", "Pre-SW", "SW", "Pre-NE"]
    assert len(out["data"]["values"]) == 4
    # Fixture has 200 days starting Jan 1 → covers NE (Jan-Mar), Pre-SW (Apr-May),
    # most of SW (Jun-Jul, partial Aug). Pre-NE (Oct-Nov) and Dec NE should be 0.
    by_label = dict(zip(out["data"]["labels"], out["data"]["values"]))
    assert by_label["Pre-NE"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_queries_new.py::test_monsoon_breakdown_returns_four_buckets -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `monsoon_breakdown` and register**

In `app/queries.py`, after `regional_total`:

```python
def monsoon_breakdown(station_id: str, year: int) -> dict:
    """Total rainfall in each monsoon bucket for one station-year."""
    df = _load_station(station_id, year=year)

    bucket_order = ["NE", "Pre-SW", "SW", "Pre-NE"]
    if len(df) == 0:
        return {
            "type": "chart",
            "chart_type": "bar",
            "title": f"Monsoon Breakdown — {_station_name(station_id)} ({year})",
            "data": {"labels": bucket_order, "values": [0.0, 0.0, 0.0, 0.0]},
            "text": "No data.",
        }

    months = df["timestamp"].dt.month
    labels = months.map(_season_label)
    by_bucket = df.groupby(labels)["reading_value"].sum()
    values = [round(float(by_bucket.get(b, 0.0)), 1) for b in bucket_order]

    total = sum(values)
    if total > 0:
        pcts = [f"{b} {v / total * 100:.0f}%" for b, v in zip(bucket_order, values)]
        text = "Share of yearly total — " + ", ".join(pcts)
    else:
        text = "No rainfall recorded."

    return {
        "type": "chart",
        "chart_type": "bar",
        "title": f"Monsoon Breakdown — {_station_name(station_id)} ({year})",
        "data": {"labels": bucket_order, "values": values},
        "text": text,
    }
```

Add to `QUERY_REGISTRY`:

```python
    "monsoon_breakdown": {
        "function": monsoon_breakdown,
        "description": "Rainfall split into NE (Jan-Mar+Dec), Pre-SW (Apr-May), SW (Jun-Sep), Pre-NE (Oct-Nov) for one station-year",
        "params": {
            "station_id": {"type": "str", "required": True},
            "year": {"type": "int", "required": True},
        },
    },
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_queries_new.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_queries_new.py
git commit -m "Add monsoon_breakdown query (NE/Pre-SW/SW/Pre-NE buckets)"
```

---

## Task 11: `season_comparison` query

Per-year totals per monsoon bucket (with Pre-SW + Pre-NE collapsed to "Inter") for one station, all years.

**Files:**
- Modify: `app/queries.py`
- Test: `tests/test_queries_new.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_queries_new.py`:

```python
def test_season_comparison_three_series_per_year(fixture_processed_dir_two_years):
    out = queries.season_comparison("S99")
    assert out["chart_type"] == "grouped_bar"
    assert out["data"]["labels"] == ["2020", "2021"]
    series_names = sorted(s["name"] for s in out["data"]["series"])
    assert series_names == ["Inter", "NE", "SW"]
    # Each series has one value per year.
    for s in out["data"]["series"]:
        assert len(s["values"]) == 2


def test_season_comparison_no_data_returns_text(fixture_processed_dir):
    # Make _available_years return nothing by querying a fictional station id
    # not registered in stations.json. station validation belongs to execute_query;
    # at the function level, we expect ValueError if the station dir is missing.
    with pytest.raises(ValueError):
        queries.season_comparison("S404")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_queries_new.py::test_season_comparison_three_series_per_year -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `season_comparison` and register**

In `app/queries.py`, after `monsoon_breakdown`:

```python
def season_comparison(station_id: str) -> dict:
    """Per-year totals per monsoon season (NE / SW / Inter) for one station, all years.

    Inter combines Pre-SW + Pre-NE for chart readability (3 series, not 4).
    """
    years = _available_years(station_id)
    if not years:
        return {"type": "text", "title": "Season Comparison", "text": "No data."}

    # Gather per (year, season) totals.
    agg: dict[int, dict[str, float]] = {y: {"NE": 0.0, "SW": 0.0, "Inter": 0.0} for y in years}
    for y in years:
        df = _load_station(station_id, year=y)
        if len(df) == 0:
            continue
        months = df["timestamp"].dt.month
        labels = months.map(lambda m: _season_label(m))
        # Collapse Pre-SW + Pre-NE into "Inter" for this view.
        labels = labels.replace({"Pre-SW": "Inter", "Pre-NE": "Inter"})
        sums = df.groupby(labels)["reading_value"].sum()
        for season in ("NE", "SW", "Inter"):
            agg[y][season] = round(float(sums.get(season, 0.0)), 1)

    sorted_years = sorted(years)
    labels_out = [str(y) for y in sorted_years]
    series = [
        {"name": "NE",    "values": [agg[y]["NE"] for y in sorted_years]},
        {"name": "SW",    "values": [agg[y]["SW"] for y in sorted_years]},
        {"name": "Inter", "values": [agg[y]["Inter"] for y in sorted_years]},
    ]

    return {
        "type": "chart",
        "chart_type": "grouped_bar",
        "title": f"Season Comparison — {_station_name(station_id)}",
        "data": {"labels": labels_out, "series": series},
        "text": f"Years: {', '.join(labels_out)}.",
    }
```

Add to `QUERY_REGISTRY`:

```python
    "season_comparison": {
        "function": season_comparison,
        "description": "Per-year monsoon season totals (NE, SW, Inter-monsoon) for one station across all years",
        "params": {
            "station_id": {"type": "str", "required": True},
        },
    },
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_queries_new.py -v`
Expected: all tests pass.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass — existing + new.

- [ ] **Step 6: Commit**

```bash
git add app/queries.py tests/test_queries_new.py
git commit -m "Add season_comparison query (NE/SW/Inter across years)"
```

---

## Task 12: `GET /api/years` endpoint

Returns the sorted union of years across all stations.

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_api_years.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_years.py`:

```python
"""Tests for the /api/years endpoint."""


def test_api_years_returns_sorted_list(client):
    r = client.get("/api/years")
    assert r.status_code == 200
    years = r.json()
    assert years == [2020]


def test_api_years_with_two_years(fixture_processed_dir_two_years, monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    from app.main import app
    monkeypatch.setattr(main, "PROCESSED_DIR", str(fixture_processed_dir_two_years))
    c = TestClient(app)
    r = c.get("/api/years")
    assert r.status_code == 200
    assert r.json() == [2020, 2021]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_api_years.py -v`
Expected: FAIL with 404 (endpoint not registered).

- [ ] **Step 3: Implement the endpoint**

In `app/main.py`, after the `get_stations` endpoint (around line 42), add:

```python
@app.get("/api/years")
def get_years():
    """Return the sorted union of years across all stations."""
    rainfall_dir = os.path.join(PROCESSED_DIR, "rainfall")
    if not os.path.isdir(rainfall_dir):
        return []
    years: set[int] = set()
    for sid in os.listdir(rainfall_dir):
        station_dir = os.path.join(rainfall_dir, sid)
        if not os.path.isdir(station_dir):
            continue
        for fname in os.listdir(station_dir):
            if not fname.endswith(".parquet"):
                continue
            stem = fname[: -len(".parquet")]
            try:
                years.add(int(stem))
            except ValueError:
                continue
    return sorted(years)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_api_years.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api_years.py
git commit -m "Add GET /api/years endpoint"
```

---

## Task 13: Expose `default` values in `/api/chat/queries`

The frontend param modal needs registry defaults to pre-fill inputs.

**Files:**
- Modify: `app/main.py:159-172`
- Test: `tests/test_api_queries_schema.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_queries_schema.py`:

```python
"""Tests for /api/chat/queries response shape."""


def test_chat_queries_includes_default_for_top_rainy_days(client):
    r = client.get("/api/chat/queries")
    assert r.status_code == 200
    queries = {q["id"]: q for q in r.json()}
    assert "top_rainy_days" in queries
    n_param = queries["top_rainy_days"]["params"]["n"]
    assert n_param["default"] == 10


def test_chat_queries_omits_default_when_param_has_none(client):
    r = client.get("/api/chat/queries")
    queries = {q["id"]: q for q in r.json()}
    # station_id has no default; field should be absent.
    assert "default" not in queries["station_summary"]["params"]["station_id"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_api_queries_schema.py -v`
Expected: FAIL with `KeyError: 'default'` or assertion failure.

- [ ] **Step 3: Update `get_queries` to include `default` when present**

In `app/main.py`, replace the `get_queries` function:

```python
@app.get("/api/chat/queries")
def get_queries():
    """Return available queries for frontend buttons."""
    out = []
    for qid, entry in QUERY_REGISTRY.items():
        params = {}
        for k, v in entry["params"].items():
            param_info = {"type": v["type"], "required": v["required"]}
            if "default" in v:
                param_info["default"] = v["default"]
            params[k] = param_info
        out.append({
            "id": qid,
            "description": entry["description"],
            "params": params,
        })
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_api_queries_schema.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api_queries_schema.py
git commit -m "Expose param defaults in /api/chat/queries"
```

---

## Task 14: Add `chart_type: "line"` renderer to the frontend

The smallest frontend change. Verify with `regional_total` since it already works backend-side.

**Files:**
- Modify: `app/static/index.html` (the chart-rendering block, around lines 476-495)

- [ ] **Step 1: Add the line branch to the renderer**

In `app/static/index.html`, find the chart-rendering block in `renderResult`:

```js
  if (result.type === 'chart') {
    const chartEl = div.querySelector('.inline-chart');
    if (result.chart_type === 'grouped_bar') {
      const traces = result.data.series.map(s => ({
        x: result.data.labels, y: s.values,
        type: 'bar', name: s.name
      }));
      Plotly.newPlot(chartEl, traces, {
        barmode: 'group', margin: { t: 10, b: 30, l: 40, r: 10 },
        legend: { orientation: 'h', y: -0.2 }
      }, { responsive: true, displayModeBar: false });
    } else {
      Plotly.newPlot(chartEl, [{
        x: result.data.labels, y: result.data.values,
        type: 'bar', marker: { color: '#42a5f5' }
      }], {
        margin: { t: 10, b: 30, l: 40, r: 10 }
      }, { responsive: true, displayModeBar: false });
    }
  }
```

Replace with:

```js
  if (result.type === 'chart') {
    const chartEl = div.querySelector('.inline-chart');
    if (result.chart_type === 'grouped_bar') {
      const traces = result.data.series.map(s => ({
        x: result.data.labels, y: s.values,
        type: 'bar', name: s.name
      }));
      Plotly.newPlot(chartEl, traces, {
        barmode: 'group', margin: { t: 10, b: 30, l: 40, r: 10 },
        legend: { orientation: 'h', y: -0.2 }
      }, { responsive: true, displayModeBar: false });
    } else if (result.chart_type === 'line') {
      const traces = result.data.series.map(s => ({
        x: result.data.labels, y: s.values,
        type: 'scatter', mode: 'lines+markers', name: s.name
      }));
      Plotly.newPlot(chartEl, traces, {
        margin: { t: 10, b: 30, l: 40, r: 10 },
        legend: { orientation: 'h', y: -0.2 }
      }, { responsive: true, displayModeBar: false });
    } else {
      Plotly.newPlot(chartEl, [{
        x: result.data.labels, y: result.data.values,
        type: 'bar', marker: { color: '#42a5f5' }
      }], {
        margin: { t: 10, b: 30, l: 40, r: 10 }
      }, { responsive: true, displayModeBar: false });
    }
  }
```

- [ ] **Step 2: Manual browser verification**

Start the dev server: `uv run uvicorn app.main:app --port 8000`. Open `http://localhost:8000`. In the browser console, run:

```js
fetch('/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({query_id: 'regional_total', params: {year: 2023, mode: 'monthly'}})
}).then(r => r.json()).then(d => console.log(d.result));
```

You should see a result with `chart_type: "line"` and a non-empty series. Then trigger this from the chat panel by clicking a query whose result you can inspect (any chart query) and confirm no console errors. The line renderer can be visually confirmed once `regional_total` is wired into a chip in Task 16.

Stop the dev server (Ctrl+C).

- [ ] **Step 3: Commit**

```bash
git add app/static/index.html
git commit -m "Add 'line' chart_type renderer to frontend"
```

---

## Task 15: Year dropdown sourced from `/api/years`

Replace the hardcoded `<option>` elements with options fetched at page load.

**Files:**
- Modify: `app/static/index.html` (lines 142-153 and the page-init script block)

- [ ] **Step 1: Strip the hardcoded options**

In `app/static/index.html`, replace the year-select HTML (lines 142-153):

```html
<select id="year-select" style="display:none">
  <option value="">All years</option>
</select>
```

- [ ] **Step 2: Populate options on page load**

Find the page-init script section after `// --- Load stations ---` (around line 200). After the existing fetch for `/api/stations`, add a new fetch:

```js
// --- Load available years ---
fetch('/api/years')
  .then(r => r.json())
  .then(years => {
    const sel = document.getElementById('year-select');
    years.forEach(y => {
      const opt = document.createElement('option');
      opt.value = String(y);
      opt.textContent = String(y);
      sel.appendChild(opt);
    });
  });
```

- [ ] **Step 3: Manual browser verification**

Start the dev server: `uv run uvicorn app.main:app --port 8000`. Open `http://localhost:8000`. Click any station marker. Verify the year dropdown shows "All years" plus 2016–2024 (or whatever years are present in `processed/`). Selecting any year should still load the chart correctly.

Stop the dev server (Ctrl+C).

- [ ] **Step 4: Commit**

```bash
git add app/static/index.html
git commit -m "Source year dropdown from /api/years instead of hardcoded options"
```

---

## Task 16: Tabbed quick-query panel

Replace the flat chip strip with category tabs.

**Files:**
- Modify: `app/static/index.html` (CSS, HTML for `#quick-queries`, and the `QUICK_QUERIES`/`buildQQButtons`/`updateQQButtons` JS)

- [ ] **Step 1: Add tab CSS**

In the `<style>` block, after the `.qq-btn` rules (around line 68), add:

```css
  /* Tabs */
  #qq-tabs {
    display: flex; padding: 8px 10px 0; gap: 4px;
    background: white; border-bottom: 1px solid #f0f0f0;
  }
  .qq-tab {
    padding: 6px 12px; font-size: 0.8em;
    border: none; background: transparent; cursor: pointer;
    color: #666; border-bottom: 2px solid transparent;
    border-radius: 4px 4px 0 0;
  }
  .qq-tab:hover { color: #1976d2; }
  .qq-tab.active { color: #1976d2; border-bottom-color: #1976d2; font-weight: 600; }
  #quick-queries { padding: 8px 10px 10px; }
  .qq-tab-panel { display: none; flex-wrap: wrap; gap: 6px; }
  .qq-tab-panel.active { display: flex; }
```

- [ ] **Step 2: Update the chat-panel HTML**

In `index.html`, find:

```html
<div id="chat-panel">
  <div id="chat-header">Rainfall Chat</div>
  <div id="quick-queries"></div>
```

Replace with:

```html
<div id="chat-panel">
  <div id="chat-header">Rainfall Chat</div>
  <div id="qq-tabs"></div>
  <div id="quick-queries"></div>
```

- [ ] **Step 3: Update the JS to render tabs and per-tab panels**

Replace the `QUICK_QUERIES` array (currently around line 343) with:

```js
const QQ_TABS = [
  { id: 'station', label: 'Station' },
  { id: 'trends',  label: 'Trends' },
  { id: 'multi',   label: 'Multi-station' },
  { id: 'season',  label: 'Seasonal' },
];

const QUICK_QUERIES = [
  // Station tab
  { id: 'station_summary', label: 'Station Stats',  tab: 'station', needsStation: true },
  { id: 'monthly_totals',  label: 'Monthly Totals', tab: 'station', needsStation: true, needsYear: true },
  { id: 'yearly_totals',   label: 'Yearly Totals',  tab: 'station', needsStation: true },
  { id: 'top_rainy_days',  label: 'Top Rainy Days', tab: 'station', needsStation: true },
  { id: 'longest_dry_spell', label: 'Dry Spell',    tab: 'station', needsStation: true },
  { id: 'hourly_pattern',  label: 'Hourly Pattern', tab: 'station', needsStation: true },
  { id: 'rainiest_week',   label: 'Rainiest Week',  tab: 'station', needsStation: true },

  // Trends tab
  { id: 'compare_stations', label: 'Compare Stations', tab: 'trends', needsStation: true, special: true },
  { id: 'yearly_trend',     label: 'Yearly Trend',     tab: 'trends', needsStation: true },
  { id: 'year_comparison',  label: 'Year Comparison',  tab: 'trends', needsStation: true, special: true },

  // Multi-station tab
  { id: 'station_ranking', label: 'Station Ranking', tab: 'multi', special: true },
  { id: 'regional_total',  label: 'Regional Total',  tab: 'multi' },

  // Seasonal tab
  { id: 'monsoon_breakdown', label: 'Monsoon Breakdown', tab: 'season', needsStation: true, needsYear: true },
  { id: 'season_comparison', label: 'Season Comparison', tab: 'season', needsStation: true },
];
```

Replace `buildQQButtons` and `updateQQButtons` with:

```js
function buildQQButtons() {
  // Build tab strip.
  const tabsContainer = document.getElementById('qq-tabs');
  QQ_TABS.forEach(t => {
    const btn = document.createElement('button');
    btn.className = 'qq-tab';
    btn.dataset.tabId = t.id;
    btn.textContent = t.label;
    btn.onclick = () => activateTab(t.id);
    tabsContainer.appendChild(btn);
  });

  // Build per-tab chip panels.
  const container = document.getElementById('quick-queries');
  QQ_TABS.forEach(t => {
    const panel = document.createElement('div');
    panel.className = 'qq-tab-panel';
    panel.dataset.tabId = t.id;
    QUICK_QUERIES.filter(q => q.tab === t.id).forEach(q => {
      const btn = document.createElement('button');
      btn.className = 'qq-btn' + (q.special ? ' special' : '');
      btn.textContent = q.label;
      btn.disabled = true;
      btn.dataset.queryId = q.id;
      btn.onclick = () => handleQQ(q);
      panel.appendChild(btn);
    });
    container.appendChild(panel);
  });

  const initial = sessionStorage.getItem('qq.activeTab') || 'station';
  activateTab(initial);
}

function activateTab(tabId) {
  document.querySelectorAll('.qq-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tabId === tabId);
  });
  document.querySelectorAll('.qq-tab-panel').forEach(p => {
    p.classList.toggle('active', p.dataset.tabId === tabId);
  });
  sessionStorage.setItem('qq.activeTab', tabId);
}

function updateQQButtons() {
  document.querySelectorAll('.qq-btn').forEach(btn => {
    const q = QUICK_QUERIES.find(qq => qq.id === btn.dataset.queryId);
    btn.disabled = (q.needsStation && !selectedStation);
  });
}
```

The existing `handleQQ` continues to handle clicks. New "special" queries (`year_comparison`, `station_ranking`) will be wired through the param modal in Task 17.

- [ ] **Step 4: Manual browser verification**

Start the dev server: `uv run uvicorn app.main:app --port 8000`. Open `http://localhost:8000`. Verify:

1. Four tabs appear: Station / Trends / Multi-station / Seasonal.
2. The default tab is "Station" with 7 chips visible.
3. Clicking a tab swaps the visible chips.
4. The selection persists after a page reload.
5. Clicking `Station Stats` (after picking a station) still works as before.

Stop the dev server.

- [ ] **Step 5: Commit**

```bash
git add app/static/index.html
git commit -m "Group quick-queries into category tabs"
```

---

## Task 17: Generic schema-driven parameter modal

One modal serves every query that needs extra inputs. Replaces the bespoke `compare-stations` modal.

**Files:**
- Modify: `app/static/index.html` (delete `#compare-overlay` and its JS, add `#param-modal` and `openParamPicker`)

- [ ] **Step 1: Replace the compare overlay with the param modal HTML**

Find the existing `<!-- Compare station picker -->` block (around line 174) and replace it with:

```html
<!-- Generic parameter picker -->
<div id="param-overlay">
  <div id="param-dialog">
    <h3 id="param-title">Configure</h3>
    <div id="param-fields"></div>
    <div id="param-actions">
      <button id="param-cancel" type="button">Cancel</button>
      <button id="param-submit" type="button">Run</button>
    </div>
  </div>
</div>
```

- [ ] **Step 2: Replace the compare-overlay CSS with param-overlay CSS**

In the `<style>` block, replace the `/* Compare picker */` rules with:

```css
  /* Generic parameter picker modal */
  #param-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.3); z-index: 2000;
    align-items: center; justify-content: center;
  }
  #param-overlay.active { display: flex; }
  #param-dialog {
    background: white; border-radius: 10px; padding: 20px; width: 340px;
    max-height: 70vh; display: flex; flex-direction: column; gap: 10px;
  }
  #param-dialog h3 { margin-bottom: 4px; }
  .param-row { display: flex; flex-direction: column; gap: 4px; }
  .param-row label { font-size: 0.85em; color: #555; font-weight: 600; }
  .param-row input, .param-row select {
    padding: 8px; border: 1px solid #ccc; border-radius: 6px; font-size: 0.95em;
  }
  .param-row .station-results {
    list-style: none; max-height: 140px; overflow-y: auto;
    border: 1px solid #eee; border-radius: 6px; margin-top: 4px;
    display: none;
  }
  .param-row .station-results.active { display: block; }
  .param-row .station-results li { padding: 6px 10px; cursor: pointer; }
  .param-row .station-results li:hover, .param-row .station-results li.selected { background: #e3f2fd; }
  #param-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 6px; }
  #param-actions button {
    padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 0.9em; border: 1px solid #ccc;
    background: white;
  }
  #param-submit { background: #1976d2; border-color: #1976d2; color: white; }
  #param-submit:disabled { opacity: 0.5; cursor: default; }
```

- [ ] **Step 3: Add `openParamPicker` and remove the old compare functions**

Delete the existing `function openCompare()`, `closeCompare()`, `filterCompareList()` block (around lines 393-424).

In their place, add:

```js
// --- Schema-driven parameter modal ---
let paramSchemaCache = null;

async function loadParamSchema() {
  if (paramSchemaCache) return paramSchemaCache;
  const r = await fetch('/api/chat/queries');
  const arr = await r.json();
  paramSchemaCache = Object.fromEntries(arr.map(q => [q.id, q]));
  return paramSchemaCache;
}

async function openParamPicker(queryId, prefilled = {}) {
  const schema = (await loadParamSchema())[queryId];
  if (!schema) return null;

  const overlay = document.getElementById('param-overlay');
  const fieldsEl = document.getElementById('param-fields');
  const titleEl = document.getElementById('param-title');
  const submitBtn = document.getElementById('param-submit');
  const cancelBtn = document.getElementById('param-cancel');

  titleEl.textContent = schema.description || 'Configure parameters';
  fieldsEl.innerHTML = '';

  const values = { ...prefilled };

  // Render one row per missing required param (and optional params with no prefilled value).
  Object.entries(schema.params).forEach(([name, spec]) => {
    if (name in values && values[name] != null && values[name] !== '') return;

    const row = document.createElement('div');
    row.className = 'param-row';
    const lbl = document.createElement('label');
    lbl.textContent = name + (spec.required ? '' : ' (optional)');
    row.appendChild(lbl);

    if (name.startsWith('station_id')) {
      // Station search input + filtered list.
      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'Search station…';
      const list = document.createElement('ul');
      list.className = 'station-results';
      const refresh = () => {
        const q = input.value.toLowerCase();
        list.innerHTML = '';
        const exclude = new Set(
          Object.entries(values)
            .filter(([k, v]) => k.startsWith('station_id') && v && k !== name)
            .map(([, v]) => v)
        );
        allStations
          .filter(s => !exclude.has(s.id))
          .filter(s => s.name.toLowerCase().includes(q) || s.id.toLowerCase().includes(q))
          .slice(0, 8)
          .forEach(s => {
            const li = document.createElement('li');
            li.textContent = `${s.name} (${s.id})`;
            li.onclick = () => {
              values[name] = s.id;
              input.value = s.name;
              list.classList.remove('active');
              updateSubmit();
            };
            list.appendChild(li);
          });
        list.classList.toggle('active', list.children.length > 0 && document.activeElement === input);
      };
      input.addEventListener('input', refresh);
      input.addEventListener('focus', refresh);
      input.addEventListener('blur', () => setTimeout(() => list.classList.remove('active'), 150));
      row.appendChild(input);
      row.appendChild(list);
    } else if (name.startsWith('year')) {
      const sel = document.createElement('select');
      const blank = document.createElement('option');
      blank.value = ''; blank.textContent = spec.required ? 'Pick a year…' : '(any)';
      sel.appendChild(blank);
      // Populate from the same list as the main year select.
      const mainSel = document.getElementById('year-select');
      Array.from(mainSel.options).forEach(o => {
        if (!o.value) return;
        const opt = document.createElement('option');
        opt.value = o.value; opt.textContent = o.textContent;
        sel.appendChild(opt);
      });
      sel.addEventListener('change', () => {
        values[name] = sel.value ? parseInt(sel.value, 10) : null;
        updateSubmit();
      });
      row.appendChild(sel);
    } else if (spec.type === 'int') {
      const input = document.createElement('input');
      input.type = 'number'; input.min = '1'; input.max = '100';
      if (spec.default !== undefined) {
        input.value = String(spec.default);
        values[name] = parseInt(spec.default, 10);
      }
      input.addEventListener('input', () => {
        values[name] = input.value ? parseInt(input.value, 10) : null;
        updateSubmit();
      });
      row.appendChild(input);
    } else {
      // Generic string input (e.g. 'mode' for regional_total).
      const input = document.createElement('input');
      input.type = 'text';
      if (spec.default !== undefined) {
        input.value = String(spec.default);
        values[name] = String(spec.default);
      }
      input.addEventListener('input', () => {
        values[name] = input.value || null;
        updateSubmit();
      });
      row.appendChild(input);
    }

    fieldsEl.appendChild(row);
  });

  function updateSubmit() {
    const ok = Object.entries(schema.params).every(([n, s]) => {
      if (!s.required) return true;
      return values[n] != null && values[n] !== '';
    });
    submitBtn.disabled = !ok;
  }
  updateSubmit();

  overlay.classList.add('active');

  return new Promise(resolve => {
    function done(result) {
      overlay.classList.remove('active');
      submitBtn.onclick = null;
      cancelBtn.onclick = null;
      resolve(result);
    }
    submitBtn.onclick = () => done(values);
    cancelBtn.onclick = () => done(null);
  });
}
```

- [ ] **Step 4: Rewrite `handleQQ` to use the param picker uniformly**

Replace the existing `handleQQ` function with:

```js
async function handleQQ(q) {
  const schema = (await loadParamSchema())[q.id];
  if (!schema) {
    addMessage('assistant', `Unknown query: ${q.id}`);
    return;
  }

  // Seed prefilled values from current selection state.
  const prefilled = {};
  if (selectedStation) {
    if ('station_id' in schema.params) prefilled.station_id = selectedStation.id;
    // For two-station queries (compare_stations), seed the *first* slot.
    if ('station_id_1' in schema.params) prefilled.station_id_1 = selectedStation.id;
  }
  const yearVal = document.getElementById('year-select').value;
  if (yearVal && 'year' in schema.params) {
    prefilled.year = parseInt(yearVal, 10);
  }

  // Determine which required params remain unmet.
  const missingRequired = Object.entries(schema.params).some(([name, spec]) => {
    if (!spec.required) return false;
    return !(name in prefilled);
  });

  let params;
  if (missingRequired || q.special) {
    params = await openParamPicker(q.id, prefilled);
    if (params == null) return;
  } else {
    params = prefilled;
  }

  const summary = Object.entries(params).map(([k, v]) => `${k}=${v}`).join(', ');
  addMessage('user', `${q.label}${summary ? ` (${summary})` : ''}`);
  sendQuery(q.id, params);
}
```

- [ ] **Step 5: Manual browser verification**

Start the dev server: `uv run uvicorn app.main:app --port 8000`. Verify:

1. Click `Compare Stations` (Trends tab) → modal opens with a station-picker row for `station_id_2`. Search works; selecting a station enables the Run button. Run produces the comparison chart.
2. Click `Year Comparison` (Trends tab, after picking a station) → modal opens with two year dropdowns. Both required to enable Run.
3. Click `Station Ranking` (Multi-station tab) → modal opens with a year dropdown and an `n` number field defaulting to `20`.
4. Click `Station Stats` (Station tab) with a station + year already selected → no modal, runs immediately.

Stop the dev server.

- [ ] **Step 6: Commit**

```bash
git add app/static/index.html
git commit -m "Replace compare modal with schema-driven param picker"
```

---

## Task 18: Station search above the map

Full-width search input above the Leaflet map. Selecting a result selects + flies to the station.

**Files:**
- Modify: `app/static/index.html` (HTML for `#left-col`, CSS, and JS init)

- [ ] **Step 1: Add HTML for the search bar**

In `index.html`, change the `#left-col` opening:

```html
<div id="left-col">
  <div id="map"></div>
```

To:

```html
<div id="left-col">
  <div id="station-search">
    <input id="station-search-input" type="text" placeholder="Search station…" autocomplete="off">
    <ul id="station-search-results"></ul>
  </div>
  <div id="map"></div>
```

- [ ] **Step 2: Add CSS for the search bar**

In the `<style>` block, after the `#left-col` rules (around line 16), add:

```css
  #station-search {
    position: relative;
    padding: 6px 10px; background: white;
    border-bottom: 1px solid #e0e0e0;
  }
  #station-search-input {
    width: 100%; padding: 6px 10px; border: 1px solid #ccc;
    border-radius: 6px; font-size: 0.9em;
  }
  #station-search-input:focus { outline: none; border-color: #1976d2; }
  #station-search-results {
    position: absolute; left: 10px; right: 10px; top: 100%;
    background: white; border: 1px solid #ddd; border-top: none;
    list-style: none; max-height: 240px; overflow-y: auto;
    z-index: 1500; display: none;
  }
  #station-search-results.active { display: block; }
  #station-search-results li {
    padding: 6px 10px; cursor: pointer;
  }
  #station-search-results li:hover, #station-search-results li.kbd-active { background: #e3f2fd; }
```

- [ ] **Step 3: Wire up the search behavior**

After the `// --- Load stations ---` block in the script (where `allStations` gets populated), add:

```js
// --- Station search ---
(function setupStationSearch() {
  const input = document.getElementById('station-search-input');
  const list = document.getElementById('station-search-results');
  let kbdIdx = -1;

  function render(matches) {
    list.innerHTML = '';
    matches.forEach((s, i) => {
      const li = document.createElement('li');
      li.textContent = `${s.name} (${s.id})`;
      if (i === kbdIdx) li.classList.add('kbd-active');
      li.onmousedown = (e) => {  // mousedown so it fires before blur clears the list
        e.preventDefault();
        pick(s);
      };
      list.appendChild(li);
    });
    list.classList.toggle('active', matches.length > 0);
  }

  function currentMatches() {
    const q = input.value.toLowerCase().trim();
    if (!q) return [];
    return allStations
      .filter(s => s.name.toLowerCase().includes(q) || s.id.toLowerCase().includes(q))
      .slice(0, 8);
  }

  function pick(s) {
    selectStation(s);
    map.setView([s.lat, s.lng], 15);
    input.value = '';
    list.classList.remove('active');
    kbdIdx = -1;
  }

  input.addEventListener('input', () => {
    kbdIdx = -1;
    render(currentMatches());
  });
  input.addEventListener('focus', () => render(currentMatches()));
  input.addEventListener('blur', () => setTimeout(() => list.classList.remove('active'), 150));
  input.addEventListener('keydown', (e) => {
    const matches = currentMatches();
    if (!matches.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      kbdIdx = (kbdIdx + 1) % matches.length;
      render(matches);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      kbdIdx = (kbdIdx - 1 + matches.length) % matches.length;
      render(matches);
    } else if (e.key === 'Enter' && kbdIdx >= 0) {
      e.preventDefault();
      pick(matches[kbdIdx]);
    } else if (e.key === 'Escape') {
      list.classList.remove('active');
    }
  });
})();
```

- [ ] **Step 4: Manual browser verification**

Start the dev server: `uv run uvicorn app.main:app --port 8000`. Verify:

1. The search input appears above the map, full width.
2. Typing "cle" shows matching station entries (e.g. Clementi).
3. ↓/↑ navigates the dropdown; Enter selects.
4. Clicking a result centers the map on that station, opens the popup, and loads its rainfall chart.
5. Esc closes the dropdown.

Stop the dev server.

- [ ] **Step 5: Commit**

```bash
git add app/static/index.html
git commit -m "Add station search above the map"
```

---

## Task 19: Update workflow.md

Document the four QQ tabs and the station search briefly.

**Files:**
- Modify: `workflow.md` (the "Chat features (quick queries)" and "Use the app" sections)

- [ ] **Step 1: Update the "Use the app" section**

In `workflow.md`, find the bulleted list under `### 4. Use the app`:

```
- The map shows all rainfall stations across Singapore
- Click a station marker to see its name and load its rainfall chart
- Use the year dropdown to filter by year (2016–2024) or view all years
- Use the chat sidebar on the right for data analysis (see below)
```

Replace with:

```
- The map shows all rainfall stations across Singapore
- Click a marker, or use the search bar above the map to find a station by name or ID
- Use the year dropdown to filter by year (years available in the dataset are loaded from the API)
- Use the chat sidebar on the right for data analysis (see below)
```

- [ ] **Step 2: Update the "Chat features (quick queries)" section**

Replace the body of `### 5. Chat features (quick queries)` with:

```markdown
The chat sidebar provides pre-built query buttons grouped into four tabs:

- **Station** — single-station summaries: Stats, Monthly Totals, Yearly Totals, Top Rainy Days, Dry Spell, Hourly Pattern, Rainiest Week.
- **Trends** — Compare Stations, Yearly Trend (with linear fit), Year Comparison.
- **Multi-station** — Station Ranking (rank all stations by year), Regional Total (cross-station mean over time).
- **Seasonal** — Monsoon Breakdown (NE / Pre-SW / SW / Pre-NE for one year), Season Comparison (NE/SW/Inter across years).

Most chips run immediately using the station/year selected in the map and dropdown. Queries that need extra inputs (e.g. a second station or a second year) open a parameter picker.
```

- [ ] **Step 3: Commit**

```bash
git add workflow.md
git commit -m "Document QQ tabs and station search in workflow.md"
```

---

## Final verification

After Task 19, run the full test suite once more and check the dev server end-to-end.

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Manual end-to-end check**

Start the dev server: `uv run uvicorn app.main:app --port 8000`. Verify the full happy paths:

1. Page loads — four tabs visible, search bar above the map, year dropdown populated.
2. Search "Clementi" → select → marker opens, chart renders.
3. Click `Yearly Trend` (Trends tab) → line chart with a single "Actual" series (no trend line because only partial-year data is in the demo dataset; verify the text reads accordingly).
4. Click `Station Ranking` → modal opens, pick year 2023 → table sorted by total mm desc.
5. Click `Regional Total` (Multi-station, no required params after defaults) → line chart of all-SG mean for "all years".
6. Click `Monsoon Breakdown` → modal opens (needs year), pick year → 4-bar chart.
7. Click `Season Comparison` → grouped bar chart with NE/SW/Inter across years.
8. Click `Compare Stations` → modal opens with station picker, run produces grouped-bar.
9. Click `Year Comparison` → modal opens with two year dropdowns, run produces grouped-bar.
10. Resize the browser to mobile width → tabs and chips reflow without breaking.

If anything misbehaves, file a follow-up task in the next plan rather than stuffing fixes into existing commits.

Stop the dev server.

- [ ] **Step 3: Push to origin**

The user has SSH key passphrase prompts that block automation, so push manually:

```bash
git push origin develop
```
