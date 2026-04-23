# Rainfall Endpoint Downsampling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the all-or-nothing `/api/rainfall` endpoint with a tiered downsampling API (daily / hourly / raw) that picks resolution from the requested time window, and wire the frontend chart to re-fetch on zoom.

**Architecture:** Server owns the tier policy. Client sends a time window (`start`/`end`/`year`); backend picks one of three tiers (daily sum, hourly sum, raw 5-minute), computes on the fly with `lru_cache`'d per-station aggregates, and returns `{resolution, points: [...]}`. Frontend subscribes to Plotly's `plotly_relayout` event, debounces 300 ms, and re-fetches with the new window.

**Tech Stack:** FastAPI + Pydantic (existing), pandas (existing), pytest + FastAPI `TestClient` (new), Plotly.js (existing, already on CDN).

**Spec:** `docs/superpowers/specs/2026-04-23-rainfall-downsampling-design.md`

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add `pytest` |
| `tests/__init__.py` | Create | Package marker (empty) |
| `tests/conftest.py` | Create | Synthetic-station parquet fixture + TestClient |
| `tests/test_rainfall_api.py` | Create | 3 test classes, ~12 cases |
| `app/queries.py` | Modify | Add `pick_tier`, `_resolve_window`, `daily_series`, `hourly_series`, `raw_series` |
| `app/main.py` | Modify | Rewrite `/api/rainfall` endpoint |
| `app/static/index.html` | Modify | New `loadRainfall(opts)`, new `handleZoom`, updated year-select listener |

---

## Pre-flight: environment setup

The venv in this repo has historically been incomplete (no `pip`). The user runs commands like `! <cmd>` to pass them through interactively when needed.

Before starting Task 1, confirm pytest is available:

```bash
python -m pytest --version
```

If it errors, install it with whichever Python the repo uses, e.g.:

```bash
python -m pip install pytest
# or, if using the repo venv:
.venv/Scripts/python.exe -m pip install pytest
```

---

## Task 1: Set up test infrastructure

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Background:** The project has no tests today. This task establishes the minimum infrastructure: a dev dependency, a package dir, and a conftest that builds a synthetic per-station parquet in a tmp directory and points `app.queries.PROCESSED_DIR` at it. All later tests reuse this fixture.

- [ ] **Step 1: Add pytest to requirements.txt**

Append one line to `requirements.txt`:

```
fastapi
uvicorn[standard]
pandas
pyarrow
httpx
pytest
```

- [ ] **Step 2: Create tests package**

Create `tests/__init__.py` as an empty file:

```python
```

- [ ] **Step 3: Create tests/conftest.py with fixtures**

Create `tests/conftest.py` with the full content below:

```python
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
```

- [ ] **Step 4: Verify pytest discovers the empty test dir**

Run:
```bash
python -m pytest tests/ -v
```

Expected: `no tests ran in 0.XXs` (zero failures, zero passes).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/__init__.py tests/conftest.py
git commit -m "Add pytest infrastructure: conftest with synthetic station fixture

Introduces pytest as a dev dependency and tests/conftest.py which builds
a tmp processed/ directory with one synthetic station (S99, 200 days of
1mm/5-min readings) and points the loader at it via monkeypatch."
```

---

## Task 2: Implement `pick_tier`

**Files:**
- Modify: `app/queries.py` (add function)
- Create: `tests/test_rainfall_api.py` (first class)

**Background:** `pick_tier` is the pure policy function. Given `(start, end)`, it returns one of `"daily"`, `"hourly"`, `"raw"`. Boundaries:
- `days > 180` → `"daily"`
- `days in [7, 180]` → `"hourly"`
- `days < 7` → `"raw"`

- [ ] **Step 1: Write the failing tests for `pick_tier`**

Create `tests/test_rainfall_api.py` with:

```python
"""Tests for the /api/rainfall endpoint and its pure-function helpers."""

import pandas as pd
import pytest


class TestPickTier:
    """Verify tier selection at every threshold boundary."""

    def test_window_over_180_days_is_daily(self):
        from app.queries import pick_tier
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=365)
        assert pick_tier(start, end) == "daily"

    def test_window_exactly_181_days_is_daily(self):
        from app.queries import pick_tier
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=181)
        assert pick_tier(start, end) == "daily"

    def test_window_exactly_180_days_is_hourly(self):
        from app.queries import pick_tier
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=180)
        assert pick_tier(start, end) == "hourly"

    def test_window_exactly_7_days_is_hourly(self):
        from app.queries import pick_tier
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=7)
        assert pick_tier(start, end) == "hourly"

    def test_window_exactly_6_days_is_raw(self):
        from app.queries import pick_tier
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=6)
        assert pick_tier(start, end) == "raw"

    def test_single_day_is_raw(self):
        from app.queries import pick_tier
        ts = pd.Timestamp("2020-01-01")
        assert pick_tier(ts, ts) == "raw"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python -m pytest tests/test_rainfall_api.py::TestPickTier -v
```

Expected: all 6 tests ERROR with `ImportError: cannot import name 'pick_tier' from 'app.queries'`.

- [ ] **Step 3: Implement `pick_tier` in `app/queries.py`**

Add after the existing `MONTH_NAMES` block (around line 39), before the existing query functions:

```python
# --- /api/rainfall tier policy ---

def pick_tier(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Select the aggregation tier for a time window.

    >180 days → daily; 7-180 days → hourly; <7 days → raw 5-min.
    """
    days = (end - start).days
    if days > 180:
        return "daily"
    if days >= 7:
        return "hourly"
    return "raw"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_rainfall_api.py::TestPickTier -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_rainfall_api.py
git commit -m "Add pick_tier policy function + tests

Pure function: maps (start, end) window to one of 'daily', 'hourly',
'raw'. Thresholds at 7 and 180 days."
```

---

## Task 3: Implement `_resolve_window`

**Files:**
- Modify: `app/queries.py` (add function)
- Modify: `tests/test_rainfall_api.py` (add class)

**Background:** `_resolve_window(df, start, end, year)` converts optional API parameters into concrete `(start_ts, end_ts)`. Rules:
- `year` given → returns `(YYYY-01-01 00:00, YYYY-12-31 23:59:59)`, `start`/`end` ignored.
- No params → uses `(df.timestamp.min(), df.timestamp.max())`.
- Partial bounds → fill missing side from df min/max.
- Resulting window is clamped to `[2016-01-01, 2024-12-31 23:59:59]`.
- Does not raise on `start > end` — the endpoint handles that check.

- [ ] **Step 1: Write the failing tests for `_resolve_window`**

Append to `tests/test_rainfall_api.py`:

```python
# These must match the constants in app/queries.py. Kept locally in the test
# file so Step 1 can run (and fail) before the implementation lands.
VALID_MIN = pd.Timestamp("2016-01-01")
VALID_MAX = pd.Timestamp("2024-12-31 23:59:59")


class TestResolveWindow:
    """Verify parameter resolution for the rainfall endpoint."""

    @pytest.fixture
    def sample_df(self):
        """DataFrame with timestamps from 2020-01-01 to 2020-07-18 (200 days)."""
        ts = pd.date_range("2020-01-01", periods=200 * 24 * 12, freq="5min")
        return pd.DataFrame({"timestamp": ts, "reading_value": 1.0})

    def test_no_params_returns_data_range(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(sample_df, None, None, None)
        assert start == pd.Timestamp("2020-01-01 00:00")
        # Last timestamp of 200 days of 5-min data at 288/day:
        # = 2020-01-01 + 200*288 - 1 periods of 5 min
        expected_end = pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=5) * (200 * 288 - 1)
        assert end == expected_end

    def test_year_shortcut(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(sample_df, None, None, 2023)
        assert start == pd.Timestamp("2023-01-01 00:00")
        assert end == pd.Timestamp("2023-12-31 23:59:59")

    def test_year_overrides_start_end(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df,
            pd.Timestamp("2020-05-01"),
            pd.Timestamp("2020-06-01"),
            2023,
        )
        assert start == pd.Timestamp("2023-01-01 00:00")
        assert end == pd.Timestamp("2023-12-31 23:59:59")

    def test_partial_start_fills_end_from_data(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df, pd.Timestamp("2020-02-01"), None, None
        )
        assert start == pd.Timestamp("2020-02-01 00:00")
        assert end == sample_df["timestamp"].max()

    def test_partial_end_fills_start_from_data(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df, None, pd.Timestamp("2020-06-01"), None
        )
        assert start == sample_df["timestamp"].min()
        assert end == pd.Timestamp("2020-06-01 00:00")

    def test_window_entirely_before_valid_range_clamps_to_min(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df,
            pd.Timestamp("2013-01-01"),
            pd.Timestamp("2014-01-01"),
            None,
        )
        assert start == VALID_MIN
        assert end == VALID_MIN

    def test_window_entirely_after_valid_range_clamps_to_max(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df,
            pd.Timestamp("2030-01-01"),
            pd.Timestamp("2031-01-01"),
            None,
        )
        assert start == VALID_MAX
        assert end == VALID_MAX

    def test_window_straddling_valid_range_is_clamped(self, sample_df):
        from app.queries import _resolve_window
        start, end = _resolve_window(
            sample_df,
            pd.Timestamp("2013-01-01"),
            pd.Timestamp("2030-01-01"),
            None,
        )
        assert start == VALID_MIN
        assert end == VALID_MAX
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python -m pytest tests/test_rainfall_api.py::TestResolveWindow -v
```

Expected: all ERROR with `ImportError: cannot import name '_resolve_window'`.

- [ ] **Step 3: Implement `_resolve_window` in `app/queries.py`**

Add just below `pick_tier` (inside the same `--- /api/rainfall tier policy ---` section):

```python
VALID_MIN = pd.Timestamp("2016-01-01")
VALID_MAX = pd.Timestamp("2024-12-31 23:59:59")


def _resolve_window(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    year: int | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert optional API params into a concrete (start, end) window.

    If `year` is given, it wins over start/end. Otherwise partial bounds are
    filled from the DataFrame's timestamp range. The final window is clamped
    to [VALID_MIN, VALID_MAX]; start > end is not treated as an error here —
    the endpoint checks that separately after clamping.
    """
    if year is not None:
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)
    else:
        if start is None:
            start = df["timestamp"].min()
        if end is None:
            end = df["timestamp"].max()

    # Clamp into the valid range.
    start = max(start, VALID_MIN)
    start = min(start, VALID_MAX)
    end = max(end, VALID_MIN)
    end = min(end, VALID_MAX)

    return start, end
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python -m pytest tests/test_rainfall_api.py::TestResolveWindow -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add app/queries.py tests/test_rainfall_api.py
git commit -m "Add _resolve_window helper + tests

Resolves (start, end, year) API parameters into a concrete window.
Year shortcut wins; partial bounds fill from data; result clamped to
[2016-01-01, 2024-12-31]."
```

---

## Task 4: Implement the three series helpers

**Files:**
- Modify: `app/queries.py` (add three functions)
- Modify: `tests/test_rainfall_api.py` (no new tests yet — these are verified via the endpoint tests in Task 5; keeping unit-test scope per spec)

**Background:** The three series helpers produce per-station aggregates. `daily_series` and `hourly_series` are cached with `@lru_cache(maxsize=32)` because they do non-trivial groupby work; `raw_series` is a thin wrapper over the already-cached `_load_station` and doesn't need its own cache.

Series helpers return `pd.Series` indexed by timestamp. The endpoint will slice with `.loc[start:end]`.

- [ ] **Step 1: Add the three helpers to `app/queries.py`**

Add directly below `_resolve_window`:

```python
@lru_cache(maxsize=32)
def daily_series(station_id: str) -> pd.Series:
    """Daily rainfall totals for one station, indexed by calendar date midnight."""
    df = _load_station(station_id)
    return df.groupby(df["timestamp"].dt.normalize())["reading_value"].sum()


@lru_cache(maxsize=32)
def hourly_series(station_id: str) -> pd.Series:
    """Hourly rainfall totals for one station, indexed by hour-floored timestamp."""
    df = _load_station(station_id)
    return df.groupby(df["timestamp"].dt.floor("h"))["reading_value"].sum()


def raw_series(station_id: str) -> pd.Series:
    """Raw 5-minute readings for one station, indexed by timestamp."""
    df = _load_station(station_id).set_index("timestamp")
    return df["reading_value"]
```

- [ ] **Step 2: Verify the full test suite still passes**

Run:
```bash
python -m pytest tests/ -v
```

Expected: 14 passed (6 from Task 2 + 8 from Task 3). No regressions.

- [ ] **Step 3: Spot-check the helpers interactively**

Run a one-off check against the fixture to confirm nothing obvious is wrong:

```bash
python -c "
import sys, tempfile, json
from pathlib import Path
import pandas as pd
tmp = Path(tempfile.mkdtemp())
(tmp / 'rainfall').mkdir()
ts = pd.date_range('2020-01-01', periods=200*288, freq='5min')
pd.DataFrame({'timestamp': ts, 'reading_value': [1.0]*len(ts)}).to_parquet(tmp / 'rainfall' / 'S99.parquet', index=False)
(tmp / 'stations.json').write_text(json.dumps([{'id':'S99','name':'S','lng':0,'lat':0}]))
from app import queries
queries.PROCESSED_DIR = str(tmp)
queries._load_station.cache_clear()
d = queries.daily_series('S99')
h = queries.hourly_series('S99')
r = queries.raw_series('S99')
print(f'daily: {len(d)} entries, first value = {d.iloc[0]}')
print(f'hourly: {len(h)} entries, first value = {h.iloc[0]}')
print(f'raw: {len(r)} entries, first value = {r.iloc[0]}')
"
```

Expected output:
```
daily: 200 entries, first value = 288.0
hourly: 4800 entries, first value = 12.0
raw: 57600 entries, first value = 1.0
```

- [ ] **Step 4: Commit**

```bash
git add app/queries.py
git commit -m "Add daily/hourly/raw series helpers for rainfall endpoint

daily_series and hourly_series are @lru_cache'd per-station; raw_series
is a thin view over the already-cached _load_station."
```

---

## Task 5: Rewrite the `/api/rainfall` endpoint

**Files:**
- Modify: `app/main.py` (replace `get_rainfall` body)
- Modify: `tests/test_rainfall_api.py` (add endpoint test class)

**Background:** The endpoint resolves the window, picks a tier, selects the appropriate cached series, slices by the window, and returns `{resolution, points}`. The old `?year=2023`-returns-raw behavior is gone; `year` is now a shortcut that feeds into the tier-picking logic.

- [ ] **Step 1: Write the failing endpoint tests**

Append to `tests/test_rainfall_api.py`:

```python
class TestRainfallEndpoint:
    """Integration tests against the FastAPI app using TestClient."""

    def test_default_window_returns_daily(self, client):
        # 200-day fixture > 180 day threshold → daily
        r = client.get("/api/rainfall/S99")
        assert r.status_code == 200
        body = r.json()
        assert body["resolution"] == "daily"
        assert len(body["points"]) == 200
        # Spot-check a point
        assert body["points"][0]["value"] == 288.0

    def test_seven_day_window_returns_hourly(self, client):
        r = client.get(
            "/api/rainfall/S99",
            params={"start": "2020-01-01T00:00:00", "end": "2020-01-08T00:00:00"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["resolution"] == "hourly"
        # 7 days * 24 hours + 1 boundary hour = 169
        assert 168 <= len(body["points"]) <= 169

    def test_three_day_window_returns_raw(self, client):
        r = client.get(
            "/api/rainfall/S99",
            params={"start": "2020-01-01T00:00:00", "end": "2020-01-03T00:00:00"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["resolution"] == "raw"
        # 2 full days * 288 + 1 boundary = 577
        assert 576 <= len(body["points"]) <= 577

    def test_unknown_station_returns_404(self, client):
        r = client.get("/api/rainfall/UNKNOWN")
        assert r.status_code == 404

    def test_start_after_end_returns_400(self, client):
        r = client.get(
            "/api/rainfall/S99",
            params={"start": "2020-06-01T00:00:00", "end": "2020-05-01T00:00:00"},
        )
        assert r.status_code == 400

    def test_year_shortcut(self, client):
        # Fixture is all in 2020; ?year=2020 covers the full year (366 days) → daily
        r = client.get("/api/rainfall/S99", params={"year": 2020})
        assert r.status_code == 200
        body = r.json()
        assert body["resolution"] == "daily"
        # Points limited to what's in the fixture (200 days of data within the 366-day window)
        assert len(body["points"]) == 200

    def test_window_outside_data_returns_empty_points(self, client):
        # Window inside valid range but with no fixture data
        r = client.get(
            "/api/rainfall/S99",
            params={"start": "2022-01-01T00:00:00", "end": "2022-02-01T00:00:00"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["points"] == []
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:
```bash
python -m pytest tests/test_rainfall_api.py::TestRainfallEndpoint -v
```

Expected: all 7 fail — the current endpoint returns a bare list, not an object with `resolution` and `points`.

- [ ] **Step 3: Rewrite the endpoint in `app/main.py`**

At the top of `app/main.py`, add the import for `datetime`:

```python
from datetime import datetime
```

Update the imports from `.queries` to include the new names:

```python
from .queries import (
    QUERY_REGISTRY,
    _load_station,
    _resolve_window,
    daily_series,
    execute_query,
    hourly_series,
    pick_tier,
    raw_series,
)
```

Replace the entire `get_rainfall` function (currently at `app/main.py:34-48`) with:

```python
@app.get("/api/rainfall/{station_id}")
def get_rainfall(
    station_id: str,
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    year: int | None = Query(None),
):
    try:
        df = _load_station(station_id)
    except ValueError:
        raise HTTPException(404, f"No data for station {station_id}")

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    window_start, window_end = _resolve_window(df, start_ts, end_ts, year)
    if window_start > window_end:
        raise HTTPException(400, "start must be <= end")

    tier = pick_tier(window_start, window_end)
    series_fn = {"daily": daily_series, "hourly": hourly_series, "raw": raw_series}[tier]
    series = series_fn(station_id)
    clipped = series.loc[window_start:window_end]

    return {
        "resolution": tier,
        "points": [
            {"timestamp": ts.isoformat(), "value": round(float(v), 3)}
            for ts, v in clipped.items()
        ],
    }
```

Add the `import pandas as pd` line near the top if it's not already there (it was removed in an earlier refactor):

```python
import pandas as pd
```

- [ ] **Step 4: Run the endpoint tests to verify they pass**

Run:
```bash
python -m pytest tests/test_rainfall_api.py::TestRainfallEndpoint -v
```

Expected: 7 passed.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run:
```bash
python -m pytest tests/ -v
```

Expected: 21 passed total (6 + 8 + 7).

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_rainfall_api.py
git commit -m "Rewrite /api/rainfall to use tiered downsampling

Endpoint now accepts ?start=&end= (or ?year= as shortcut), resolves the
window, picks daily/hourly/raw tier, and returns {resolution, points}.
Replaces the old 'return every 5-min reading' behavior.

Tests cover default window, 7-day and 3-day windows, unknown station,
start>end, year shortcut, and empty-window responses."
```

---

## Task 6: Update frontend to consume the new API

**Files:**
- Modify: `app/static/index.html`

**Background:** The frontend now sends `?start&end` (or `?year`) instead of just `?year`, reads the `resolution` field from the wrapper response, switches chart type (bar for daily/hourly, scatter for raw) and y-axis label accordingly, and binds a debounced `plotly_relayout` handler for zoom-driven re-fetches.

No automated tests here — per the spec, frontend coverage is manual. That check happens in Task 7.

- [ ] **Step 1: Update the module-scoped state declarations**

Find the existing block in `app/static/index.html` (around line 189):

```javascript
let selectedStation = null;
let allStations = [];
let llmAvailable = false;
let rainfallController = null;
let chartIdCounter = 0;
```

Replace with:

```javascript
let selectedStation = null;
let allStations = [];
let llmAvailable = false;
let rainfallController = null;
let chartIdCounter = 0;
let currentResolution = null;
let zoomDebounceTimer = null;
```

- [ ] **Step 2: Replace the `loadRainfall` function**

Find the entire existing `loadRainfall` function (around lines 222-248). Replace with:

```javascript
function loadRainfall(opts = {}) {
  // opts may contain {start: Date, end: Date, year: number}
  const params = new URLSearchParams();
  if (opts.year) {
    params.set('year', opts.year);
  } else {
    if (opts.start) params.set('start', opts.start.toISOString());
    if (opts.end)   params.set('end', opts.end.toISOString());
  }
  const queryString = params.toString();
  const url = `/api/rainfall/${selectedStation.id}` + (queryString ? `?${queryString}` : '');

  document.getElementById('loading').style.display = '';
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('chart').style.display = '';

  if (rainfallController) rainfallController.abort();
  rainfallController = new AbortController();

  fetch(url, { signal: rainfallController.signal })
    .then(r => r.json())
    .then(data => {
      document.getElementById('loading').style.display = 'none';
      currentResolution = data.resolution;

      const unitLabel =
        data.resolution === 'raw'    ? 'mm / 5-min' :
        data.resolution === 'hourly' ? 'mm / hour'  :
                                       'mm / day';
      const chartType = data.resolution === 'raw' ? 'scatter' : 'bar';

      const trace = {
        x: data.points.map(p => p.timestamp),
        y: data.points.map(p => p.value),
        type: chartType,
        name: 'Rainfall',
      };
      if (chartType === 'scatter') {
        trace.mode = 'lines';
        trace.line = { color: '#1976d2', width: 1 };
      } else {
        trace.marker = { color: '#42a5f5' };
      }

      Plotly.react('chart', [trace], {
        title: `Rainfall — ${selectedStation.name} (${data.resolution})`,
        xaxis: { title: 'Time' },
        yaxis: { title: unitLabel },
        margin: { t: 40, b: 50, l: 60, r: 20 },
      }, { responsive: true });

      // Bind zoom handler once. Plotly.react preserves the node but not
      // necessarily prior listeners bound via .on() — re-binding is safe
      // because we removeAllListeners for this event first.
      const chartEl = document.getElementById('chart');
      if (chartEl.removeAllListeners) chartEl.removeAllListeners('plotly_relayout');
      chartEl.on('plotly_relayout', handleZoom);
    })
    .catch(err => {
      if (err.name === 'AbortError') return;
      document.getElementById('loading').style.display = 'none';
    });
}
```

- [ ] **Step 3: Add the `handleZoom` function**

Insert directly after the `loadRainfall` function:

```javascript
function handleZoom(ev) {
  // Plotly emits this on any layout change. Only react to user-initiated zoom/pan.
  const hasRange = ('xaxis.range[0]' in ev) || ('xaxis.range' in ev);
  if (!hasRange) return;

  clearTimeout(zoomDebounceTimer);
  zoomDebounceTimer = setTimeout(() => {
    const rawStart = ev['xaxis.range[0]'] ?? ev['xaxis.range'][0];
    const rawEnd   = ev['xaxis.range[1]'] ?? ev['xaxis.range'][1];
    loadRainfall({ start: new Date(rawStart), end: new Date(rawEnd) });
  }, 300);
}
```

- [ ] **Step 4: Update the year-select listener**

Find the existing listener (around line 216-218):

```javascript
document.getElementById('year-select').addEventListener('change', () => {
  if (selectedStation) loadRainfall();
});
```

Replace with:

```javascript
document.getElementById('year-select').addEventListener('change', () => {
  if (!selectedStation) return;
  const y = document.getElementById('year-select').value;
  loadRainfall(y ? { year: parseInt(y, 10) } : {});
});
```

- [ ] **Step 5: Verify no Python tests regressed**

The frontend change is isolated; run pytest just to confirm:

```bash
python -m pytest tests/ -v
```

Expected: 21 passed. No changes.

- [ ] **Step 6: Commit**

```bash
git add app/static/index.html
git commit -m "Wire frontend to tiered /api/rainfall

loadRainfall now accepts {start, end, year} opts and builds the query
string accordingly. Reads {resolution, points} from the response;
switches chart type (bar for daily/hourly, scatter for raw) and y-axis
label (mm/day, mm/h, mm/5-min). Adds handleZoom with 300ms debounce
bound to plotly_relayout for drill-down zoom re-fetches. Year-select
now narrows the window instead of filtering raw data."
```

---

## Task 7: Manual smoke test + wrap-up

**Files:**
- None to modify (verification task)

**Background:** The design explicitly calls out that Plotly rendering, zoom events, and debounce behavior aren't covered by automated tests. This task runs the spec's manual smoke test checklist against real data.

- [ ] **Step 1: Start the dev server**

Run (in a dedicated terminal — the server blocks):

```bash
uvicorn app.main:app --reload --port 8000
```

- [ ] **Step 2: Open the app**

Navigate to `http://localhost:8000` in a browser.

- [ ] **Step 3: Run the smoke-test checklist**

Work through each item. Record any failures and stop; don't proceed until all pass.

1. Click a station on the map.
   - **Expected:** Bar chart appears. Title ends with `(daily)`. Y-axis reads `mm / day`. ~3,300 bars.
2. Select a year from the dropdown (e.g., 2023).
   - **Expected:** Chart switches to bars for that year. Title ends with `(daily)` because a full year = 365 days > 180.
3. Use Plotly's box-select zoom to select a ~30 day window inside the chart.
   - **Expected:** After ~300ms, chart re-fetches and shows **hourly** bars. Y-axis reads `mm / hour`.
4. Zoom further into a ~3 day window.
   - **Expected:** After ~300ms, chart re-fetches and shows **raw** scatter line. Y-axis reads `mm / 5-min`.
5. Zoom back out by double-clicking in the chart (Plotly's "reset axes" gesture).
   - **Expected:** Chart returns to daily view.
6. Click a different station on the map.
   - **Expected:** Chart resets; no stale data from the previous station's zoom level.
7. Open DevTools → Network tab. Pan the chart rapidly with the middle-mouse-button drag (or whichever pan gesture Plotly uses).
   - **Expected:** At most 1 fetch fires per ~300ms of no-activity. Rapid continuous pan should *not* fire a flurry of requests. Earlier in-flight requests should show as cancelled (from `AbortController`).

- [ ] **Step 4: Stop the dev server**

In the server terminal, press `Ctrl+C`.

- [ ] **Step 5: Run the full test suite one more time to confirm all green**

```bash
python -m pytest tests/ -v
```

Expected: 21 passed.

- [ ] **Step 6: Final commit if any tiny fixes were needed**

If the smoke test surfaced any small corrections (e.g., a typo in a label, an off-by-one in the URL builder), amend them and commit:

```bash
# only if changes were needed
git add -A
git commit -m "Smoke-test fixes: <brief description>"
```

If everything passed cleanly, nothing to commit — move on.

---

## Verification summary

After Task 7, the branch should have:
- 6 new commits (Tasks 1–6) plus optionally a 7th (Task 7 tweaks)
- 21 passing pytest cases
- No lint/syntax errors (existing code didn't use linting; this plan doesn't add any)
- Green smoke-test checklist

If any of these are red, return to the relevant task and fix before reporting the feature as complete.
