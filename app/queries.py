"""Pre-built rainfall query functions and registry."""

import glob
import json
import os
from functools import lru_cache

import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")


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


@lru_cache(maxsize=1)
def _load_stations_index() -> dict:
    with open(os.path.join(PROCESSED_DIR, "stations.json")) as f:
        return {s["id"]: s["name"] for s in json.load(f)}


def _all_station_ids() -> list[str]:
    """Sorted list of station IDs from stations.json."""
    return sorted(_load_stations_index().keys())


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


def _station_name(station_id: str) -> str:
    return _load_stations_index().get(station_id, station_id)


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

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


VALID_MIN = pd.Timestamp("2016-01-01")
VALID_MAX = pd.Timestamp("2024-12-31 23:59:59")


def _clamp(ts: pd.Timestamp) -> pd.Timestamp:
    """Clamp a timestamp into the valid data range."""
    return min(max(ts, VALID_MIN), VALID_MAX)


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
    start = _clamp(start)
    end = _clamp(end)

    return start, end


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


def monthly_totals(station_id: str, year: int) -> dict:
    df = _load_station(station_id, year=year)
    monthly = df.groupby(df["timestamp"].dt.month)["reading_value"].sum()
    labels = [MONTH_NAMES[m - 1] for m in monthly.index]
    values = [round(v, 1) for v in monthly.values]
    return {
        "type": "chart",
        "chart_type": "bar",
        "title": f"Monthly Rainfall — {_station_name(station_id)} ({year})",
        "data": {"labels": labels, "values": values},
        "text": f"Total for {year}: {sum(values):.1f} mm",
    }


PARTIAL_YEAR_DAYS = 300  # fewer recorded days than this → treat as partial


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


def top_rainy_days(station_id: str, year: int | None = None, n: int = 10) -> dict:
    n = max(1, min(int(n), 100))
    df = _load_station(station_id, year=year)
    daily = df.groupby(df["timestamp"].dt.date)["reading_value"].sum()
    top = daily.nlargest(n)
    rows = [[str(date), round(val, 1)] for date, val in top.items()]
    year_label = f" ({year})" if year else ""
    return {
        "type": "table",
        "title": f"Top {n} Rainiest Days — {_station_name(station_id)}{year_label}",
        "columns": ["Date", "Rainfall (mm)"],
        "rows": rows,
        "text": f"Rainiest day: {rows[0][0]} with {rows[0][1]} mm" if rows else "No data",
    }


def compare_stations(
    station_id_1: str, station_id_2: str, year: int | None = None
) -> dict:
    name1 = _station_name(station_id_1)
    name2 = _station_name(station_id_2)

    df1 = _load_station(station_id_1, year=year)
    df2 = _load_station(station_id_2, year=year)

    m1 = df1.groupby(df1["timestamp"].dt.month)["reading_value"].sum()
    m2 = df2.groupby(df2["timestamp"].dt.month)["reading_value"].sum()

    labels = MONTH_NAMES
    values1 = [round(m1.get(i + 1, 0), 1) for i in range(12)]
    values2 = [round(m2.get(i + 1, 0), 1) for i in range(12)]

    year_label = f" ({year})" if year else ""
    return {
        "type": "chart",
        "chart_type": "grouped_bar",
        "title": f"Comparison{year_label}",
        "data": {
            "labels": labels,
            "series": [
                {"name": name1, "values": values1},
                {"name": name2, "values": values2},
            ],
        },
        "text": f"Total — {name1}: {sum(values1):.1f} mm, {name2}: {sum(values2):.1f} mm",
    }


def longest_dry_spell(station_id: str, year: int | None = None) -> dict:
    df = _load_station(station_id, year=year)
    daily = df.groupby(df["timestamp"].dt.normalize())["reading_value"].sum()
    daily = daily.sort_index()

    if len(daily) == 0:
        return {"type": "text", "title": "Longest Dry Spell", "text": "No data."}

    # Reindex to a contiguous calendar so sensor outages don't silently bridge
    # two separate dry streaks. Gap days become NaN; NaN == 0 is False, so
    # they correctly break runs.
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    gap_days = int(daily.isna().sum())

    dry = daily == 0
    if not dry.any():
        return {"type": "text", "title": "Longest Dry Spell", "text": "No dry days found."}

    groups = (dry != dry.shift()).cumsum()
    counts = dry[dry].groupby(groups[dry]).count()
    longest = int(counts.max())
    longest_group_id = counts.idxmax()
    group_dates = dry[groups == longest_group_id].index
    start, end = group_dates[0].date(), group_dates[-1].date()

    year_label = f" ({year})" if year else ""
    text = f"{longest} consecutive dry days, from {start} to {end}."
    if gap_days:
        text += f" (Note: {gap_days} day(s) with no recorded readings were excluded.)"
    return {
        "type": "text",
        "title": f"Longest Dry Spell — {_station_name(station_id)}{year_label}",
        "text": text,
    }


def station_summary(station_id: str, year: int | None = None) -> dict:
    df = _load_station(station_id, year=year)
    daily = df.groupby(df["timestamp"].dt.date)["reading_value"].sum()

    total = round(daily.sum(), 1)
    mean_daily = round(daily.mean(), 2)
    max_daily = round(daily.max(), 1)
    max_date = str(daily.idxmax()) if len(daily) > 0 else "N/A"
    rainy_days = int((daily > 0).sum())
    total_days = len(daily)

    year_label = f" ({year})" if year else ""
    return {
        "type": "table",
        "title": f"Station Summary — {_station_name(station_id)}{year_label}",
        "columns": ["Metric", "Value"],
        "rows": [
            ["Total Rainfall", f"{total} mm"],
            ["Daily Average", f"{mean_daily} mm"],
            ["Max Daily Rainfall", f"{max_daily} mm ({max_date})"],
            ["Rainy Days", f"{rainy_days} / {total_days}"],
        ],
        "text": f"Total: {total} mm across {rainy_days} rainy days out of {total_days}.",
    }


def rainiest_week(station_id: str, year: int | None = None) -> dict:
    df = _load_station(station_id, year=year)
    daily = df.groupby(df["timestamp"].dt.normalize())["reading_value"].sum()
    daily = daily.sort_index()

    if len(daily) == 0:
        return {"type": "text", "title": "Rainiest Week", "text": "No data."}

    # Reindex to a contiguous date range so a 7-row rolling window is always
    # 7 calendar days. Windows that touch a gap evaluate to NaN and are
    # skipped by idxmax.
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)

    rolling = daily.rolling(7).sum()
    if rolling.notna().sum() == 0:
        return {
            "type": "text",
            "title": "Rainiest Week",
            "text": "Not enough contiguous data for a 7-day window.",
        }

    peak_end = rolling.idxmax()
    peak_start = peak_end - pd.Timedelta(days=6)
    peak_val = round(rolling.max(), 1)

    week_data = daily.loc[peak_start:peak_end]
    rows = [
        [str(d.date()), 0.0 if pd.isna(v) else round(v, 1)]
        for d, v in week_data.items()
    ]

    year_label = f" ({year})" if year else ""
    return {
        "type": "table",
        "title": f"Rainiest Week — {_station_name(station_id)}{year_label}",
        "columns": ["Date", "Rainfall (mm)"],
        "rows": rows,
        "text": f"Rainiest 7-day period: {peak_start.date()} to {peak_end.date()} with {peak_val} mm total.",
    }


def hourly_pattern(station_id: str, year: int | None = None) -> dict:
    df = _load_station(station_id, year=year)
    ts = df["timestamp"]
    # Sum the 12 five-minute readings inside each (date, hour) so we get an
    # actual hourly total, then average those totals across days.
    hourly_totals = df.groupby([ts.dt.normalize(), ts.dt.hour])["reading_value"].sum()
    hourly = hourly_totals.groupby(level=1).mean()
    labels = [f"{h:02d}:00" for h in hourly.index]
    values = [round(v, 3) for v in hourly.values]

    year_label = f" ({year})" if year else ""
    return {
        "type": "chart",
        "chart_type": "bar",
        "title": f"Avg Hourly Rainfall — {_station_name(station_id)}{year_label}",
        "data": {"labels": labels, "values": values},
        "text": f"Peak hour: {labels[values.index(max(values))]} ({max(values):.3f} mm/h avg)"
        if values
        else "No data",
    }


# --- Registry ---

QUERY_REGISTRY = {
    "monthly_totals": {
        "function": monthly_totals,
        "description": "Monthly rainfall totals for a station in a given year",
        "params": {
            "station_id": {"type": "str", "required": True},
            "year": {"type": "int", "required": True},
        },
    },
    "yearly_totals": {
        "function": yearly_totals,
        "description": "Yearly rainfall totals for a station across all years",
        "params": {
            "station_id": {"type": "str", "required": True},
        },
    },
    "yearly_trend": {
        "function": yearly_trend,
        "description": "Yearly totals plus a linear trend line for one station across all years",
        "params": {
            "station_id": {"type": "str", "required": True},
        },
    },
    "top_rainy_days": {
        "function": top_rainy_days,
        "description": "Top N rainiest days for a station",
        "params": {
            "station_id": {"type": "str", "required": True},
            "year": {"type": "int", "required": False},
            "n": {"type": "int", "required": False, "default": 10},
        },
    },
    "compare_stations": {
        "function": compare_stations,
        "description": "Compare monthly rainfall between two stations",
        "params": {
            "station_id_1": {"type": "str", "required": True},
            "station_id_2": {"type": "str", "required": True},
            "year": {"type": "int", "required": False},
        },
    },
    "longest_dry_spell": {
        "function": longest_dry_spell,
        "description": "Find the longest consecutive period with no rainfall",
        "params": {
            "station_id": {"type": "str", "required": True},
            "year": {"type": "int", "required": False},
        },
    },
    "station_summary": {
        "function": station_summary,
        "description": "Statistics summary: total, mean, max, rainy days count",
        "params": {
            "station_id": {"type": "str", "required": True},
            "year": {"type": "int", "required": False},
        },
    },
    "rainiest_week": {
        "function": rainiest_week,
        "description": "Find the 7-day period with the highest total rainfall",
        "params": {
            "station_id": {"type": "str", "required": True},
            "year": {"type": "int", "required": False},
        },
    },
    "hourly_pattern": {
        "function": hourly_pattern,
        "description": "Average rainfall by hour of day, showing daily patterns",
        "params": {
            "station_id": {"type": "str", "required": True},
            "year": {"type": "int", "required": False},
        },
    },
}


def execute_query(query_id: str, params: dict) -> dict:
    """Validate and execute a registered query."""
    if query_id not in QUERY_REGISTRY:
        raise ValueError(f"Unknown query: {query_id}")

    entry = QUERY_REGISTRY[query_id]
    schema = entry["params"]

    # Validate required params
    for name, spec in schema.items():
        if spec["required"] and name not in params:
            raise ValueError(f"Missing required parameter: {name}")

    # Type coercion
    coerced = {}
    for name, spec in schema.items():
        if name in params and params[name] is not None:
            val = params[name]
            if spec["type"] == "int":
                coerced[name] = int(val)
            else:
                coerced[name] = str(val)
        elif "default" in spec:
            coerced[name] = spec["default"]

    # Validate station IDs exist
    stations = _load_stations_index()
    for name, val in coerced.items():
        if "station_id" in name and val not in stations:
            raise ValueError(f"Unknown station: {val}")

    # Validate year range
    if "year" in coerced and coerced["year"] is not None:
        if not (2016 <= coerced["year"] <= 2024):
            raise ValueError("Year must be between 2016 and 2024")

    return entry["function"](**coerced)
