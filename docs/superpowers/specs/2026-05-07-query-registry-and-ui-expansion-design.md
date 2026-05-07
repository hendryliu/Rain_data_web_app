# Design: Query registry expansion + UI tabs / param picker / station search

**Date:** 2026-05-07
**Branch:** develop

## Goal

Broaden the analytical query registry with six new queries (year-over-year trends, multi-station rankings, monsoon seasonality) and update the frontend so adding queries doesn't degrade the UX. Adds: tabbed quick-query panel, schema-driven parameter modal that replaces the bespoke compare-stations modal, and a station search input above the map.

## Non-goals

- Map ranking heatmap / choropleth-style station coloring (held for follow-up).
- Chat panel resize handle, loading skeletons, empty-state polish.
- Frontend test framework (none today; manual browser verification only).
- Extreme-event / threshold queries (deferred — not in this batch).

## New queries

| ID | Signature | Output | Answers |
|---|---|---|---|
| `yearly_trend` | `(station_id)` | line chart of yearly totals + fitted linear trend; text = slope mm/year | "Is Clementi getting wetter over the years?" |
| `year_comparison` | `(station_id, year_a, year_b)` | grouped_bar of monthly totals, two series | "How did 2023 compare to 2024 at Clementi?" |
| `station_ranking` | `(year, n=20)` | table; rank, name, total mm | "Which stations were rainiest in 2023?" |
| `regional_total` | `(year=None, mode='monthly')` — `mode` ∈ `{"monthly", "daily"}` | line chart, all-Singapore mean across stations over time | "What was the SG-wide rainfall pattern in 2023?" |
| `monsoon_breakdown` | `(station_id, year)` | bar with 4 buckets: NE / Pre-SW / SW / Pre-NE; totals + % | "How does monsoon vs inter-monsoon split at Clementi in 2023?" |
| `season_comparison` | `(station_id)` | grouped_bar; x = year, three series (NE / SW / Inter) | "How does each monsoon season vary year-to-year at Clementi?" |

### Conventions

- **Monsoon buckets are calendar-month within the requested year**: NE = Jan-Mar + Dec of the same year, SW = Jun-Sep, Pre-SW = Apr-May, Pre-NE = Oct-Nov. Documented in each query's description so users (and the LLM) understand it doesn't span across years.
- **`season_comparison`** collapses Pre-SW + Pre-NE into one "Inter" series for chart readability (3 series instead of 4).
- **`yearly_trend`** reuses the existing `PARTIAL_YEAR_DAYS` rule from `yearly_totals`. The slope is fit only on full years; partial years are still plotted but visually marked.
- **`regional_total`** uses the *mean* across stations (not the sum). Mean stays interpretable as "typical mm in SG"; sum would inflate as station coverage changes year to year.
- **`station_ranking`** silently skips stations with no data file for the requested year (rather than ranking them as 0 mm and inflating the bottom of the list).

### Output shape additions

Two existing chart types: `bar`, `grouped_bar`. Adding one new chart_type `"line"`:

```json
{
  "type": "chart",
  "chart_type": "line",
  "title": "...",
  "data": {
    "labels": ["2016", "2017", ...],
    "series": [
      {"name": "Total", "values": [2100, 2350, ...]},
      {"name": "Trend", "values": [2150, 2200, ...]}
    ]
  },
  "text": "..."
}
```

`yearly_trend` returns two series (data + fitted line). `regional_total` returns one series, but uses the same series-list shape for consistency.

## Backend helpers

Added to `app/queries.py`:

- `_all_station_ids() -> list[str]` — list station IDs from `stations.json`.
- `_season_label(month: int) -> str` — returns `"NE" / "Pre-SW" / "SW" / "Pre-NE"`. Lookup table; raises `ValueError` for invalid month.
- `_cross_station_yearly_totals(year: int) -> dict[str, float]` — used by `station_ranking`. Skips stations missing the year file.
- `_regional_series(year: int | None, mode: str) -> pd.Series` — used by `regional_total`. Loads each station, resamples to monthly or daily, takes the cross-station mean. Reindexed to a contiguous date range so cross-station gaps don't bridge ranges (mirroring `rainiest_week` gap-handling).

Both `_cross_station_yearly_totals` and `_regional_series` are `@lru_cache`-decorated so a repeat call is instant.

## Frontend changes

### Tabbed quick-query panel

Replace the flat `#quick-queries` div with a tab strip + per-tab panel.

```
┌─────────────────────────────────────────┐
│ [Station] [Trends] [Multi] [Seasonal]   │
├─────────────────────────────────────────┤
│ ( Station Stats ) ( Monthly Totals )    │
│ ( Yearly Totals ) ( Top Rainy Days  )   │
│ ( Dry Spell     ) ( Hourly Pattern  )   │
│ ( Rainiest Week )                       │
└─────────────────────────────────────────┘
```

Tab assignment:

| Tab | Queries |
|---|---|
| Station | station_summary, monthly_totals, yearly_totals, top_rainy_days, longest_dry_spell, hourly_pattern, rainiest_week |
| Trends | compare_stations, yearly_trend, year_comparison |
| Multi-station | station_ranking, regional_total |
| Seasonal | monsoon_breakdown, season_comparison |

Selected tab persists in `sessionStorage` (key `qq.activeTab`); default `Station`. Tab strip is keyboard-accessible (arrow keys move focus, Enter activates) — basic role=tab/tablist semantics.

### Schema-driven parameter modal

Replace `#compare-overlay` with a generic modal `#param-modal`. Function:

```js
async function openParamPicker(queryId, prefilled = {}) -> Promise<paramDict | null>
```

Reads the query's `params` schema (already exposed at `/api/chat/queries`). For each unmet required param, renders an input:

| Schema name | Input rendered |
|---|---|
| `station_id`, `station_id_2`, `station_id_1` | Searchable station picker (the existing compare-modal list, extracted to a reusable component) |
| `year`, `year_a`, `year_b` | Year dropdown (sourced from `/api/years`) |
| `n` | Number input (range 1–100); pre-fills with the registry's `default` for that param |

Resolves to a complete param dict on submit; resolves to `null` on cancel.

Click flow for chips:

1. Build params from current selectedStation + year-select.
2. Determine which required params from the registry are still missing.
3. If none missing → call `sendQuery(queryId, params)` immediately.
4. Otherwise → `openParamPicker(queryId, prefilled=params)` and run on resolve.

This deletes the `Compare Stations` special-case branching in `handleQQ` — it becomes the same as everything else, just with two missing `station_id_*` params.

### Station search above the map

New row above the map containing a full-width `<input>` with placeholder "Search station…". Behavior:

- Live filter `allStations` by `name.toLowerCase().includes(query)` (also matches station ID).
- Dropdown shows up to 8 matches, keyboard navigable (↑/↓, Enter, Esc).
- Selecting a result: `selectStation(s)`, `map.setView([s.lat, s.lng], 15)`, opens that marker's popup.
- Search input clears after selection.

### `chart_type: "line"` renderer

One additional `else if` branch in `renderResult` covering both single-series and multi-series via the `data.series` shape (same shape as `grouped_bar`). Uses Plotly `type: 'scatter', mode: 'lines+markers'`.

### Year dropdown sourced from API

New endpoint `GET /api/years` returns the sorted union of years across stations (e.g. `[2016, …, 2024]`). Frontend populates the year `<select>` from this on page load. Hardcoded `<option>` tags removed from `index.html`.

## Performance

- `station_ranking(year)` reads 91 station-year files. Existing `_load_station` cache (`lru_cache(maxsize=128)`) absorbs repeat calls. Cold-cache wall time is acceptable for a chat-driven action.
- `regional_total(year=None, mode='monthly')` would otherwise read 91 × 9 = 819 files cold and exceed the 128-entry cache. Mitigation: `_regional_series` is itself `lru_cache`-decorated so subsequent calls are O(1). First call may take a few seconds; acceptable.
- No background prefetch added.

## LLM integration

New queries appear in the system prompt automatically (it's built from `QUERY_REGISTRY`). Description text needs to make these distinctions easy:

- `monthly_totals`: one station, one year, monthly bars.
- `year_comparison`: one station, two years, monthly bars side by side.
- `compare_stations`: two stations, one year, monthly bars side by side.

Each description will explicitly call out the parameter shape ("requires X and Y") to reduce mis-routing.

## Testing

New pytest cases (mirroring existing patterns in `tests/test_queries_year_scoped.py`):

- One happy-path test per new query against the fixture.
- `_season_label` table-driven test for all 12 months + `ValueError` for invalid month.
- `station_ranking` test asserting stations missing the year are excluded (not ranked as 0).
- `regional_total` test verifying mean (not sum) across a 2-station fixture.
- `yearly_trend` test verifying slope sign matches a constructed monotonic fixture.

Existing fixture in `tests/conftest.py` may need a small extension (a third station and a multi-year setup) to support multi-station and trend tests.

Frontend changes (tabs, modal, search, line renderer, /api/years) are verified manually in the browser. No JS test framework is added.

## File-by-file impact

| File | Change |
|---|---|
| `app/queries.py` | +6 query functions, +4 helpers, registry entries |
| `app/main.py` | +1 endpoint `GET /api/years` |
| `app/static/index.html` | Tabbed QQ panel, generic param modal (replaces compare modal), station search, line chart_type renderer, dynamic year options |
| `tests/conftest.py` | Fixture extended for multi-station and multi-year cases |
| `tests/test_queries_*.py` | New test files for new queries and helpers |
| `workflow.md` | Document the four QQ tabs and station search briefly |

## Risks and trade-offs

- **Cross-station scans on cold cache.** Acceptable for chat-style use; not acceptable if a future feature calls them on every keystroke. Documented but no preemptive optimization.
- **Monsoon convention is calendar-month-within-year.** True NE monsoon spans Dec–Mar across two years. The simpler convention is documented; users wanting cross-year analysis can use `year_comparison`.
- **Regional mean vs sum.** Mean is more interpretable but loses absolute volume. We show one and not both to keep the LLM prompt smaller; can add a sum variant later if requested.
- **Tab UI in `sessionStorage`** — survives reloads in same tab, doesn't sync across tabs. Acceptable; this is a single-page app with no auth.
