# Build-up: Interactive Chat Interface for Rainfall Data Analysis

## Overview

Added a chat sidebar to the Singapore rainfall web app that lets users query and analyze rainfall data through pre-built quick queries and (optionally) natural language via a local LLM (llama.cpp).

## What was built

### Backend query engine (`app/queries.py`)

- 8 pre-built analytical queries using pandas against Parquet data:
  - **Monthly totals** — monthly rainfall sums for a station in a given year
  - **Yearly totals** — yearly rainfall sums across all available years
  - **Top N rainiest days** — ranked list of days with highest rainfall
  - **Compare stations** — side-by-side monthly comparison of two stations
  - **Longest dry spell** — longest consecutive period with zero rainfall
  - **Station summary** — statistics: total, daily average, max, rainy day count
  - **Rainiest week** — the 7-day period with highest cumulative rainfall
  - **Hourly pattern** — average rainfall by hour of day
- Query registry with parameter validation and type coercion
- LRU-cached Parquet file loading for performance

### Chat API endpoints (`app/main.py`)

- `POST /api/chat` — dispatches pre-built queries or forwards free-form text to the LLM; returns structured results (charts, tables, text)
- `GET /api/chat/queries` — lists available queries for frontend buttons
- `GET /api/chat/health` — checks if llama.cpp is reachable

### LLM integration (`app/llm.py`)

- Connects to llama.cpp's OpenAI-compatible API (`/v1/chat/completions`)
- System prompt built dynamically from the query registry and station list
- LLM outputs structured JSON (query ID + params), never executable code
- Response parsing with validation against the query registry
- Configurable via `LLAMA_CPP_URL` and `LLAMA_CPP_MODEL` environment variables

### Frontend chat UI (`app/static/index.html`)

- Three-panel layout: map + chart (left 70%) and chat sidebar (right 30%)
- Quick-query buttons that use the currently selected station and year
- Compare Stations picker dialog for selecting a second station
- Inline Plotly charts and HTML tables rendered in chat messages
- Collapsible chat panel with responsive mobile layout (bottom sheet on screens < 768px)
- Free-form text input enabled only when LLM is available

### Infrastructure

- `docker-compose.yml` — optional llama.cpp service via `--profile llm`
- `requirements.txt` — added `httpx` for async HTTP to llama.cpp
- `.gitignore` — added `models/` directory

## How to use

### Quick queries (no LLM needed)

1. Start the server: `uvicorn app.main:app --port 8000`
2. Open `http://localhost:8000`
3. Click a station on the map
4. Use the quick-query buttons in the chat sidebar

### Free-form queries (requires llama.cpp)

1. Download a GGUF model (e.g., Llama 3 8B Instruct Q4_K_M) into `models/model.gguf`
2. Start llama.cpp server on port 8080, or run `docker compose --profile llm up`
3. The chat text input will activate automatically when the LLM is detected

## Architecture decisions

- **Structured JSON dispatch, not code generation** — the LLM picks a query ID + parameters from a fixed registry. No `exec()` or `eval()` is ever used. This is safe by design.
- **Progressive enhancement** — quick-query buttons always work; the LLM is an optional add-on for natural language.
- **No new frameworks** — kept the existing vanilla JS + CDN approach. No build step added.
