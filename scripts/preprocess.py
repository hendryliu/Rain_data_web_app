"""Preprocess rainfall CSVs into per-station Parquet files.

Streaming design: each yearly CSV is loaded once, split per station, and
flushed to a temp parquet partition. After all CSVs are processed, each
station's year-parts are concatenated, sorted, and written to its final
parquet. Peak memory is roughly one year of narrow-column data.
"""

import glob
import json
import os
import re
import shutil

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
RAINFALL_DIR = os.path.join(OUTPUT_DIR, "rainfall")
TMP_DIR = os.path.join(OUTPUT_DIR, "_tmp")

YEAR_RE = re.compile(r"(\d{4})")


def year_from_filename(path: str) -> str:
    m = YEAR_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not extract year from {path}")
    return m.group(1)


def main():
    os.makedirs(RAINFALL_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        print("No CSV files found in data/")
        return

    print(f"Found {len(csv_files)} CSV files")

    stations: dict[str, dict] = {}

    # Pass 1: per CSV, split by station and write temp parquet partitions.
    for csv_path in csv_files:
        year = year_from_filename(csv_path)
        print(f"[{year}] Reading {os.path.basename(csv_path)}...")
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

        # Track station metadata (last-seen coords/name wins)
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

        print(f"[{year}] Writing per-station temp parquet for {df['station_id'].nunique()} stations...")
        for station_id, group in df.groupby("station_id", sort=False):
            station_tmp = os.path.join(TMP_DIR, station_id)
            os.makedirs(station_tmp, exist_ok=True)
            out = group[["timestamp", "reading_value"]].reset_index(drop=True)
            out.to_parquet(os.path.join(station_tmp, f"{year}.parquet"), index=False)

        del df

    # Write stations.json
    stations_list = sorted(stations.values(), key=lambda s: s["name"])
    stations_path = os.path.join(OUTPUT_DIR, "stations.json")
    with open(stations_path, "w") as f:
        json.dump(stations_list, f)
    print(f"Wrote {len(stations_list)} stations to {stations_path}")

    # Pass 2: merge each station's yearly parts into final parquet.
    print("Merging per-station parquet files...")
    station_dirs = sorted(os.listdir(TMP_DIR))
    for i, station_id in enumerate(station_dirs, 1):
        station_tmp = os.path.join(TMP_DIR, station_id)
        parts = sorted(glob.glob(os.path.join(station_tmp, "*.parquet")))
        frames = [pd.read_parquet(p) for p in parts]
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        combined.to_parquet(os.path.join(RAINFALL_DIR, f"{station_id}.parquet"), index=False)
        if i % 10 == 0 or i == len(station_dirs):
            print(f"  merged {i}/{len(station_dirs)} stations")

    shutil.rmtree(TMP_DIR)
    print(f"Done. Wrote Parquet files for {len(station_dirs)} stations.")


if __name__ == "__main__":
    main()
