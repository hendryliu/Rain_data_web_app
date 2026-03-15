"""FastAPI server for Singapore rainfall data."""

import json
import os

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .llm import query_llm
from .queries import QUERY_REGISTRY, execute_query

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
            return {
                "role": "assistant",
                "content": result.get("text", ""),
                "result": result,
            }
        except ValueError as e:
            return {"role": "assistant", "content": str(e), "result": None}

    # Free-form message path via LLM
    if req.message:
        llm_result = await query_llm(req.message, req.context)

        if llm_result["query"] is None:
            return {
                "role": "assistant",
                "content": llm_result["explanation"],
                "result": None,
            }

        try:
            result = execute_query(llm_result["query"], llm_result["params"])
            content = llm_result.get("explanation", result.get("text", ""))
            return {"role": "assistant", "content": content, "result": result}
        except ValueError as e:
            return {"role": "assistant", "content": str(e), "result": None}

    raise HTTPException(400, "Provide either query_id or message")


@app.get("/api/chat/health")
def chat_health():
    """Check if the LLM backend (llama.cpp) is reachable."""
    try:
        import httpx

        r = httpx.get(
            os.environ.get("LLAMA_CPP_URL", "http://localhost:8080") + "/health",
            timeout=2,
        )
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
