"""FastAPI server for Singapore rainfall data."""

import json
import logging
import os
from datetime import datetime

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .llm import query_llm
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

logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")

app = FastAPI(title="Singapore Rainfall")
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/api/stations")
def get_stations():
    path = os.path.join(PROCESSED_DIR, "stations.json")
    if not os.path.exists(path):
        raise HTTPException(500, "stations.json not found — run preprocess.py first")
    with open(path) as f:
        return json.load(f)


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

    # Normalize to pd.Timestamp so _resolve_window's max/min compare like types.
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    window_start, window_end = _resolve_window(df, start_ts, end_ts, year)
    # _resolve_window clamps but does not reorder; validate after resolution.
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


# --- Chat endpoints ---


class ChatRequest(BaseModel):
    message: str | None = None
    query_id: str | None = None
    params: dict | None = None
    context: dict | None = None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Pre-built query path
    if req.query_id:
        params = req.params or {}
        try:
            result = execute_query(req.query_id, params)
            result["_query_id"] = req.query_id
            result["_params"] = params
            return {
                "role": "assistant",
                "content": result.get("text", ""),
                "result": result,
            }
        except ValueError as e:
            return {"role": "assistant", "content": str(e), "result": None}

    # Free-form message path via LLM
    if req.message:
        try:
            llm_result = await query_llm(req.message, req.context)

            if llm_result["query"] is None:
                return {
                    "role": "assistant",
                    "content": llm_result["explanation"],
                    "result": None,
                }

            result = execute_query(llm_result["query"], llm_result["params"])
            result["_query_id"] = llm_result["query"]
            result["_params"] = llm_result["params"]
            content = llm_result.get("explanation", result.get("text", ""))
            return {"role": "assistant", "content": content, "result": result}
        except ValueError as e:
            # Validation errors (unknown station, bad year, etc.) are safe to surface
            return {"role": "assistant", "content": str(e), "result": None}
        except Exception:
            logger.exception("Unhandled error in free-form chat")
            return {
                "role": "assistant",
                "content": "Sorry, something went wrong processing your request.",
                "result": None,
            }

    raise HTTPException(400, "Provide either query_id or message")


@app.get("/api/chat/health")
async def chat_health():
    """Check if the LLM backend (llama.cpp) is reachable."""
    url = os.environ.get("LLAMA_CPP_URL", "http://localhost:8080") + "/health"
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(url)
        return {"llm_available": r.status_code == 200}
    except Exception:
        return {"llm_available": False}


@app.get("/api/chat/queries")
def get_queries():
    """Return available queries for frontend buttons."""
    return [
        {
            "id": qid,
            "description": entry["description"],
            "params": {
                k: {"type": v["type"], "required": v["required"]}
                for k, v in entry["params"].items()
            },
        }
        for qid, entry in QUERY_REGISTRY.items()
    ]


# Serve frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
