"""Pre-built rainfall query functions and registry."""

import json
import os
from functools import lru_cache

import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")


@lru_cache(maxsize=32)
def _load_station(station_id: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "rainfall", f"{station_id}.parquet")
    if not os.path.exists(path):
        raise ValueError(f"No data for station {station_id}")
    return pd.read_parquet(path)


def _load_stations_index() -> dict:
    with open(os.path.join(PROCESSED_DIR, "stations.json")) as f:
        return {s["id"]: s["name"] for s in json.load(f)}


def _station_name(station_id: str) -> str:
    return _load_stations_index().get(station_id, station_id)


def _filter_year(df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    if year is not None:
        return df[df["timestamp"].dt.year == year]
    return df


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def monthly_totals(station_id: str, year: int) -> dict:
    df = _filter_year(_load_station(station_id), year)
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


def yearly_totals(station_id: str) -> dict:
    df = _load_station(station_id)
    yearly = df.groupby(df["timestamp"].dt.year)["reading_value"].sum()
    labels = [str(y) for y in yearly.index]
    values = [round(v, 1) for v in yearly.values]
    return {
        "type": "chart",
        "chart_type": "bar",
        "title": f"Yearly Rainfall — {_station_name(station_id)}",
        "data": {"labels": labels, "values": values},
        "text": f"Average: {sum(values) / len(values):.1f} mm/year"
        if values
        else "No data",
    }


def top_rainy_days(station_id: str, year: int | None = None, n: int = 10) -> dict:
    df = _filter_year(_load_station(station_id), year)
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

    df1 = _filter_year(_load_station(station_id_1), year)
    df2 = _filter_year(_load_station(station_id_2), year)

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
    df = _filter_year(_load_station(station_id), year)
    daily = df.groupby(df["timestamp"].dt.date)["reading_value"].sum()
    daily = daily.sort_index()

    dry = daily == 0
    if not dry.any():
        return {"type": "text", "title": "Longest Dry Spell", "text": "No dry days found."}

    # Find longest consecutive dry streak
    groups = (dry != dry.shift()).cumsum()
    dry_groups = dry[dry].groupby(groups[dry])
    longest = dry_groups.count().max()
    longest_group_id = dry_groups.count().idxmax()
    group_dates = dry[groups == longest_group_id].index
    start, end = group_dates[0], group_dates[-1]

    year_label = f" ({year})" if year else ""
    return {
        "type": "text",
        "title": f"Longest Dry Spell — {_station_name(station_id)}{year_label}",
        "text": f"{longest} consecutive dry days, from {start} to {end}.",
    }


def station_summary(station_id: str, year: int | None = None) -> dict:
    df = _filter_year(_load_station(station_id), year)
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
    df = _filter_year(_load_station(station_id), year)
    daily = df.groupby(df["timestamp"].dt.date)["reading_value"].sum()
    daily = daily.sort_index()

    if len(daily) < 7:
        return {"type": "text", "title": "Rainiest Week", "text": "Not enough data."}

    rolling = daily.rolling(7).sum()
    peak_end = rolling.idxmax()
    peak_start = daily.index[daily.index.get_loc(peak_end) - 6]
    peak_val = round(rolling.max(), 1)

    week_data = daily.loc[peak_start:peak_end]
    rows = [[str(d), round(v, 1)] for d, v in week_data.items()]

    year_label = f" ({year})" if year else ""
    return {
        "type": "table",
        "title": f"Rainiest Week — {_station_name(station_id)}{year_label}",
        "columns": ["Date", "Rainfall (mm)"],
        "rows": rows,
        "text": f"Rainiest 7-day period: {peak_start} to {peak_end} with {peak_val} mm total.",
    }


def hourly_pattern(station_id: str, year: int | None = None) -> dict:
    df = _filter_year(_load_station(station_id), year)
    hourly = df.groupby(df["timestamp"].dt.hour)["reading_value"].mean()
    labels = [f"{h:02d}:00" for h in hourly.index]
    values = [round(v, 3) for v in hourly.values]

    year_label = f" ({year})" if year else ""
    return {
        "type": "chart",
        "chart_type": "bar",
        "title": f"Avg Rainfall by Hour — {_station_name(station_id)}{year_label}",
        "data": {"labels": labels, "values": values},
        "text": f"Peak hour: {labels[values.index(max(values))]} ({max(values):.3f} mm avg)"
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
        if name in params:
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
