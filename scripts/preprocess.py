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
