# Rainfall Endpoint Downsampling — Design

**Date:** 2026-04-23
**Status:** Approved
**Scope:** `/api/rainfall/{station_id}` endpoint + frontend chart consumer

## Problem

The current `/api/rainfall/{station_id}` endpoint returns every 5-minute reading
as a JSON array. For a station with 9 years of data that is roughly 950,000
points per request — around 100 MB of JSON, pushing the browser to render a
Plotly scatter it cannot handle smoothly. The default frontend call (no year
filter) triggers exactly this worst case on every station click.

## Goals

1. Keep per-response payload in the 1–5k-point range regardless of time window.
2. Support a drill-down UX: coarse overview by default, finer detail as the
   user zooms in.
3. Frontend is the only consumer and is updated in the same commit, so the
   response-shape change is not user-visible. No external callers to preserve.
4. Keep the backend implementation small and testable without pre-computing
   parquet partitions (that's a separate open item).

## Non-goals

- Pre-aggregation at preprocess time (tracked as a separate issue; this design
  assumes on-demand aggregation with in-process caching).
- Partitioning the parquet files by year (same separate issue).
- Supporting window queries from third-party callers; this is an internal API.

## Design

### Resolution tiers

Three fixed tiers:

| Tier | Aggregation | Typical size |
|---|---|---|
| `daily` | Sum of readings per calendar day | ~3,300 points for 9 years |
| `hourly` | Sum of readings per hour | ~4,320 points for 180 days |
| `raw` | Unaggregated 5-minute readings | ~2,016 points for 7 days |

Rainfall is additive, so the aggregation rule is always `sum`. `mean` would
conflate intensity with duration and is not useful for this domain.

### Tier policy (server-side)

Given a requested window `[start, end]`:

```python
def pick_tier(start, end):
    days = (end - start).days
    if days > 180: return "daily"
    if days >= 7:  return "hourly"
    return "raw"
```

Thresholds:
- Window > 180 days → `daily`
- Window 7–180 days → `hourly`
- Window < 7 days → `raw`

Boundaries are stable under normal zoom gestures and produce predictable point
counts. The client does not choose a tier — the server owns the policy.

### API

```
GET /api/rainfall/{station_id}
    ?start=<ISO timestamp>    optional
    ?end=<ISO timestamp>      optional
    ?year=<YYYY>              optional shortcut
```

Parameter resolution rules:

- No params → window is `(earliest, latest)` of the station's data.
- `year=YYYY` → window is `(YYYY-01-01 00:00, YYYY-12-31 23:59)`. `start`/`end`
  are ignored when `year` is set.
- Partial bounds (`start` only or `end` only) → fill the missing side from the
  station's data range.
- `start > end` → HTTP 400.
- Window extending before 2016-01-01 or after 2024-12-31 → silently clamped to
  the valid range (do not error).

Response shape (wrapper object, breaking change from current bare array):

```json
{
  "resolution": "daily" | "hourly" | "raw",
  "points": [
    {"timestamp": "2023-05-14T00:00:00", "value": 12.4}
  ]
}
```

### Backend implementation

File: `app/queries.py` gains three helpers plus the tier policy.

```python
@lru_cache(maxsize=32)
def daily_series(station_id: str) -> pd.Series:
    df = _load_station(station_id)
    return df.groupby(df["timestamp"].dt.normalize())["reading_value"].sum()

@lru_cache(maxsize=32)
def hourly_series(station_id: str) -> pd.Series:
    df = _load_station(station_id)
    return df.groupby(df["timestamp"].dt.floor("h"))["reading_value"].sum()

def raw_series(station_id: str) -> pd.Series:
    df = _load_station(station_id).set_index("timestamp")
    return df["reading_value"]

def pick_tier(start: pd.Timestamp, end: pd.Timestamp) -> str:
    days = (end - start).days
    if days > 180: return "daily"
    if days >= 7:  return "hourly"
    return "raw"
```

A `_resolve_window(df, start, end, year)` helper encapsulates the parameter
resolution rules listed above and returns `(start_ts, end_ts)`.

Memory footprint with caches warm across all 91 stations:
- Daily: ~5 MB (91 × 3,300 × ~16 bytes)
- Hourly: ~115 MB (91 × 79,000 × ~16 bytes)

`maxsize=32` on both caches caps resident hourly data to ~40 MB. LRU eviction
under pressure is acceptable for a single-user local app. Multi-user deployment
would move to pre-computed parquet partitions (separate issue).

File: `app/main.py` endpoint becomes:

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

    window_start, window_end = _resolve_window(df, start, end, year)
    if window_start > window_end:
        raise HTTPException(400, "start must be <= end")

    tier = pick_tier(window_start, window_end)
    series = {
        "daily": daily_series,
        "hourly": hourly_series,
        "raw": raw_series,
    }[tier](station_id)
    clipped = series.loc[window_start:window_end]

    return {
        "resolution": tier,
        "points": [
            {"timestamp": ts.isoformat(), "value": round(float(v), 3)}
            for ts, v in clipped.items()
        ],
    }
```

### Frontend implementation

File: `app/static/index.html`. All changes localised to `loadRainfall` and the
year-select listener.

New module-scoped state:

```javascript
let currentResolution = null;
let zoomDebounceTimer = null;
```

(`rainfallController` already exists from Tier A.)

Updated `loadRainfall(opts = {})` — accepts `{start?, end?, year?}`:

- Builds query string from `opts`.
- Cancels any in-flight request via `rainfallController`.
- On response, reads `data.resolution` and uses it to:
  - Pick chart type: `bar` for `daily`/`hourly`, `scatter` for `raw`.
  - Pick y-axis label: `mm/day`, `mm/h`, `mm/5-min`.
  - Set chart title to include the resolution.
- Calls `Plotly.react` (not `newPlot`) so the DOM node and event handlers
  survive across updates.
- Binds `plotly_relayout` to `handleZoom` once after the first plot.

Zoom handler:

```javascript
function handleZoom(ev) {
  if (!('xaxis.range[0]' in ev) && !('xaxis.range' in ev)) return;
  clearTimeout(zoomDebounceTimer);
  zoomDebounceTimer = setTimeout(() => {
    const start = new Date(ev['xaxis.range[0]'] ?? ev['xaxis.range'][0]);
    const end   = new Date(ev['xaxis.range[1]'] ?? ev['xaxis.range'][1]);
    loadRainfall({ start, end });
  }, 300);
}
```

Year-select listener:

```javascript
document.getElementById('year-select').addEventListener('change', () => {
  if (!selectedStation) return;
  const y = document.getElementById('year-select').value;
  loadRainfall(y ? { year: parseInt(y) } : {});
});
```

Station-click path (`selectStation`) is unchanged: it calls `loadRainfall()`
with no args, which resolves to the full station range → daily overview.

### Error handling

| Condition | Response |
|---|---|
| Station ID not in parquet dir | HTTP 404 `"No data for station {id}"` |
| Malformed `start`/`end` (not ISO) | HTTP 422 via FastAPI Pydantic parsing |
| `start > end` | HTTP 400 `"start must be <= end"` |
| Both `year` and `start`/`end` given | `year` wins; others ignored (documented) |
| Window extends outside 2016–2024 | Silently clamped to valid range |
| Window entirely outside 2016–2024 | Clamped window is empty; returns HTTP 200 `{resolution, points: []}` |
| Window contains no readings | HTTP 200 with `{resolution, points: []}`; frontend shows empty chart |
| Client aborts fetch | No server action needed |

### Testing

Introduces `pytest` as a dev dependency. Scope is minimal — pure-function
units plus a thin endpoint smoke test.

```
tests/
  conftest.py             # ~20 lines: fixture parquet, TestClient
  test_rainfall_api.py    # ~80 lines, 12 test cases
```

Coverage:

1. **`TestPickTier`** — exhaustive branch coverage for the tier policy,
   including exact-boundary cases at 7 and 180 days.
2. **`TestResolveWindow`** — uses a small in-memory DataFrame to verify
   parameter resolution: no params, `year`, partial bounds, `start > end`,
   out-of-range clamping.
3. **`TestRainfallEndpoint`** — uses FastAPI `TestClient` against a tiny
   fixture parquet (3 stations, 1 week of synthetic data):
   - `GET /api/rainfall/S01` → 200, valid `resolution`, non-empty `points`.
   - `GET /api/rainfall/UNKNOWN` → 404.
   - `GET /api/rainfall/S01?year=2023` → 200 with an appropriate tier.
   - `GET /api/rainfall/S01?start=X&end=Y` with `X > Y` → 400.

Not automatically tested (tracked as manual smoke tests in the implementation
plan):

- Plotly rendering and zoom-event wiring.
- Debounce behavior (that rapid pans don't flood the server).
- Performance on the real 9-year dataset.

Manual smoke test checklist, to be run after implementation:

1. Click a station → bar chart with daily bars appears, resolution label shows
   "daily".
2. Select a year from the dropdown → chart switches to that year, shows hourly
   bars.
3. Zoom into a one-week window with Plotly's selection tool → hourly bars swap
   in (may fall through to raw if the selection is tight).
4. Zoom further into a single day → raw 5-minute scatter swaps in.
5. Zoom back out to multi-year view → daily bars return.
6. Rapid pan with the mouse does not produce a burst of backend requests
   (debounce holds until the user stops).

## Scope summary

| File | Change |
|---|---|
| `app/queries.py` | +25 lines: three series helpers + `pick_tier` + `_resolve_window` |
| `app/main.py` | `/api/rainfall` endpoint rewritten (~15 → ~25 lines) |
| `app/static/index.html` | +30 lines net in `loadRainfall`, new `handleZoom`, updated year-select listener |
| `requirements.txt` | `+pytest` (or split into `requirements-dev.txt`) |
| `tests/conftest.py` | New, ~20 lines |
| `tests/test_rainfall_api.py` | New, ~80 lines / 12 cases |

No other modules touched. No changes to the preprocess pipeline or the LLM
integration.

## Out of scope / follow-ups

These are intentionally deferred:

- **Pre-computed parquet per tier at preprocess time** — removes the runtime
  groupby cost and the in-process hourly-cache memory concern. Worth doing if
  the app is ever deployed multi-user.
- **Parquet year-partitioning + pyarrow filter pushdown** — cuts I/O for
  year-scoped queries. Separate from this design.
- **Dynamic point-count-target tier (Option B from brainstorming)** — if
  fixed-threshold tiering proves too coarse in practice, revisit.
