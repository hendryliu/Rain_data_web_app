"""Preprocess rainfall CSVs into Parquet files partitioned by station."""

import json
import glob
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

    # Collect all stations and rainfall data
    stations = {}
    all_frames = []

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
                "station_id": str,
                "station_name": str,
                "reading_value": float,
            },
        )

        # Extract unique stations (use last seen coordinates/name per station)
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

        # Keep only columns needed for rainfall parquet
        all_frames.append(df[["station_id", "timestamp", "reading_value"]])

    # Write stations.json
    stations_list = sorted(stations.values(), key=lambda s: s["name"])
    stations_path = os.path.join(OUTPUT_DIR, "stations.json")
    with open(stations_path, "w") as f:
        json.dump(stations_list, f)
    print(f"Wrote {len(stations_list)} stations to {stations_path}")

    # Concatenate all data and write per-station parquet files
    print("Concatenating all data...")
    combined = pd.concat(all_frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp")

    print("Writing per-station Parquet files...")
    for station_id, group in combined.groupby("station_id"):
        out = group[["timestamp", "reading_value"]].reset_index(drop=True)
        parquet_path = os.path.join(RAINFALL_DIR, f"{station_id}.parquet")
        out.to_parquet(parquet_path, index=False)

    print(f"Done. Wrote Parquet files for {combined['station_id'].nunique()} stations.")


if __name__ == "__main__":
    main()
