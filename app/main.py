"""FastAPI server for Singapore rainfall data."""

import json
import os

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")

app = FastAPI(title="Singapore Rainfall")


@app.get("/api/stations")
def get_stations():
    path = os.path.join(PROCESSED_DIR, "stations.json")
    if not os.path.exists(path):
        raise HTTPException(500, "stations.json not found — run preprocess.py first")
    with open(path) as f:
        return json.load(f)


@app.get("/api/rainfall/{station_id}")
def get_rainfall(station_id: str, year: int | None = Query(None)):
    path = os.path.join(PROCESSED_DIR, "rainfall", f"{station_id}.parquet")
    if not os.path.exists(path):
        raise HTTPException(404, f"No data for station {station_id}")

    df = pd.read_parquet(path)
    if year is not None:
        df = df[df["timestamp"].dt.year == year]

    records = df.rename(columns={"reading_value": "value"}).to_dict(orient="records")
    # Convert timestamps to ISO strings
    for r in records:
        r["timestamp"] = r["timestamp"].isoformat()
    return records


# Serve frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
