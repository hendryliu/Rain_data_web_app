"""One-shot migrator: split flat per-station parquets into year partitions.

For each `processed/rainfall/{id}.parquet`:
  1. Read into memory.
  2. Group by timestamp.dt.year.
  3. Write each year to `processed/rainfall/{id}/{year}.parquet`.
  4. Verify row counts match.
  5. Delete the original flat file.

If `processed/rainfall/{id}/` already exists as a directory, skip the station
(assume already migrated). Failures leave the flat file in place — rerun safely.
"""

import os
import sys

import pandas as pd


def _total_row_count(paths: list[str]) -> int:
    return sum(len(pd.read_parquet(p)) for p in paths)


def migrate(rainfall_dir: str) -> None:
    if not os.path.isdir(rainfall_dir):
        raise FileNotFoundError(f"Not a directory: {rainfall_dir}")

    migrated, skipped, failed = 0, 0, 0

    for entry in sorted(os.listdir(rainfall_dir)):
        full = os.path.join(rainfall_dir, entry)

        # Already a partitioned directory → skip.
        if os.path.isdir(full):
            skipped += 1
            continue

        # Only flat parquets are candidates.
        if not entry.endswith(".parquet"):
            continue

        station_id = entry[: -len(".parquet")]
        station_dir = os.path.join(rainfall_dir, station_id)

        # If both flat file and station dir exist, skip — manual cleanup needed.
        if os.path.isdir(station_dir):
            print(f"  {station_id}: skipped (subdir already exists)")
            skipped += 1
            continue

        try:
            df = pd.read_parquet(full)
            original_count = len(df)

            os.makedirs(station_dir, exist_ok=True)
            written = []
            for year, group in df.groupby(df["timestamp"].dt.year):
                out = os.path.join(station_dir, f"{int(year)}.parquet")
                group.reset_index(drop=True).to_parquet(out, index=False)
                written.append(out)

            new_count = _total_row_count(written)
            if new_count != original_count:
                # Roll back: remove the partial subdir, leave flat file intact.
                for p in written:
                    os.remove(p)
                os.rmdir(station_dir)
                raise RuntimeError(
                    f"{station_id}: row count mismatch "
                    f"(flat={original_count}, partitioned={new_count})"
                )

            os.remove(full)
            migrated += 1
            print(f"  {station_id}: migrated ({len(written)} years, {original_count} rows)")
        except Exception as e:
            failed += 1
            print(f"  {station_id}: FAILED — {e}")
            raise

    print(f"\nDone. migrated={migrated} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "processed", "rainfall"
    )
    migrate(target)
