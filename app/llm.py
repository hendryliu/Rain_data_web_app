"""LLM integration via llama.cpp's OpenAI-compatible API."""

import json
import os
import re

import httpx

from .queries import QUERY_REGISTRY, _load_stations_index

LLAMA_CPP_URL = os.environ.get("LLAMA_CPP_URL", "http://localhost:8080")
LLAMA_CPP_MODEL = os.environ.get("LLAMA_CPP_MODEL", "default")


def _build_station_list() -> str:
    stations = _load_stations_index()
    return "\n".join(f"{sid}={name}" for sid, name in sorted(stations.items()))


def _build_query_catalog() -> str:
    """Render the query list for the system prompt from QUERY_REGISTRY.

    Single source of truth: adding a query to the registry updates the prompt
    automatically, with no risk of the two drifting apart.
    """
    lines = []
    for qid, entry in QUERY_REGISTRY.items():
        required = [n for n, s in entry["params"].items() if s["required"]]
        optional = [f"{n}?" for n, s in entry["params"].items() if not s["required"]]
        sig = ", ".join(required + optional)
        lines.append(f"- {qid}({sig}) — {entry['description']}")
    return "\n".join(lines)


def _build_system_prompt(context: dict | None = None) -> str:
    # Only include full station list if user is asking about a station by name
    # without context. If context has a selected station, we can keep it short.
    if context and context.get("selected_station"):
        station_info = f"Currently selected station: {context['selected_station']} ({_load_stations_index().get(context['selected_station'], '')})"
    else:
        station_info = f"Stations:\n{_build_station_list()}"

    return f"""Rainfall data assistant. Respond ONLY with JSON.

{station_info}

Queries:
{_build_query_catalog()}

JSON format: {{"query":"<id>","params":{{...}},"explanation":"<short>"}}
If unanswerable: {{"query":null,"explanation":"<why>"}}
Use context station if none mentioned. Match station names to IDs.

Examples (assume S06 Paya Lebar is selected):
User: "which month had the most rain in 2023"
→ {{"query":"monthly_totals","params":{{"station_id":"S06","year":2023}},"explanation":"Monthly totals for 2023; the tallest bar is the rainiest month."}}

User: "rainiest days last year"
→ {{"query":"top_rainy_days","params":{{"station_id":"S06","year":2023}},"explanation":"Top 10 rainiest days in 2023."}}

User: "compare with Upper Changi"
→ {{"query":"compare_stations","params":{{"station_id_1":"S06","station_id_2":"S24"}},"explanation":"Monthly comparison against Upper Changi Road North."}}"""


async def query_llm(message: str, context: dict | None = None) -> dict:
    """Send a message to llama.cpp and parse the structured response."""
    system_prompt = _build_system_prompt(context)
    user_content = message

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{LLAMA_CPP_URL}/v1/chat/completions",
                json={
                    "model": LLAMA_CPP_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 512,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
    except Exception as e:
        return {"query": None, "explanation": f"Could not reach LLM server: {e}"}

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return {"query": None, "explanation": "Unexpected response from LLM."}

    return _parse_llm_response(content)


def _parse_llm_response(content: str) -> dict:
    """Extract and validate JSON from LLM response text."""
    # Most well-behaved responses are pure JSON. Only fall back to a {...}
    # extraction if the model wrapped the JSON in prose or markdown.
    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            return {"query": None, "explanation": "Could not parse LLM response."}
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            return {"query": None, "explanation": "Invalid JSON in LLM response."}

    query_id = parsed.get("query")
    explanation = parsed.get("explanation", "")
    params = parsed.get("params", {})

    if query_id is None:
        return {"query": None, "explanation": explanation}

    if query_id not in QUERY_REGISTRY:
        return {"query": None, "explanation": f"Unknown query type: {query_id}"}

    # Validate station IDs
    stations = _load_stations_index()
    for key, val in params.items():
        if "station_id" in key and val not in stations:
            return {"query": None, "explanation": f"Unknown station: {val}"}

    return {"query": query_id, "params": params, "explanation": explanation}
