"""LLM integration via llama.cpp's OpenAI-compatible API."""

import json
import os
import re

import httpx

from .queries import QUERY_REGISTRY, _load_stations_index

LLAMA_CPP_URL = os.environ.get("LLAMA_CPP_URL", "http://localhost:8080")
LLAMA_CPP_MODEL = os.environ.get("LLAMA_CPP_MODEL", "default")


def _build_system_prompt() -> str:
    stations = _load_stations_index()
    station_list = ", ".join(f"{sid} ({name})" for sid, name in sorted(stations.items()))

    query_descriptions = []
    for qid, entry in QUERY_REGISTRY.items():
        params_desc = ", ".join(
            f"{k} ({v['type']}, {'required' if v['required'] else 'optional'})"
            for k, v in entry["params"].items()
        )
        query_descriptions.append(f"- {qid}: {entry['description']}. Parameters: {params_desc}")

    return f"""You are a Singapore rainfall data assistant. Users ask questions about historical rainfall data (2016-2024) from weather stations across Singapore.

Your job is to translate the user's question into a structured query. Respond ONLY with a JSON object — no other text.

Available stations:
{station_list}

Available queries:
{chr(10).join(query_descriptions)}

Response format (JSON only):
{{"query": "<query_id>", "params": {{<parameters>}}, "explanation": "<brief one-line explanation of what you're computing>"}}

If the question cannot be answered with the available queries, respond with:
{{"query": null, "explanation": "<brief explanation of why and what queries are available>"}}

Rules:
- Match station names to their IDs (e.g., "Changi" -> the station ID containing "Changi")
- If the user mentions a station by name, find the closest matching station ID
- If a year is mentioned, include it in params
- For "compare" questions, use compare_stations with two station IDs
- Default n=10 for top_rainy_days unless the user specifies otherwise
- If a station is provided in context but not in the question, use the context station"""


async def query_llm(message: str, context: dict | None = None) -> dict:
    """Send a message to llama.cpp and parse the structured response."""
    system_prompt = _build_system_prompt()

    user_content = message
    if context and context.get("selected_station"):
        user_content += f"\n(Currently selected station: {context['selected_station']})"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
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
                },
            )
            response.raise_for_status()
    except httpx.HTTPError as e:
        return {"query": None, "explanation": f"Could not reach LLM server: {e}"}

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return {"query": None, "explanation": "Unexpected response from LLM."}

    return _parse_llm_response(content)


def _parse_llm_response(content: str) -> dict:
    """Extract and validate JSON from LLM response text."""
    # Try to find JSON in the response (may be wrapped in markdown code blocks)
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
