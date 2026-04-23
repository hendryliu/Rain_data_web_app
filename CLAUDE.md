# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A FastAPI + vanilla-JS web app for exploring historical rainfall across Singapore. Users click stations on a Leaflet map to view per-station time series, and use a chat sidebar (pre-built queries plus an optional local LLM via llama.cpp) for higher-level analysis.

Layout:
- `app/main.py` — FastAPI routes: `/api/stations`, `/api/rainfall/{id}`, `/api/chat*`, plus static frontend
- `app/queries.py` — analytical query registry (monthly totals, dry spells, station comparisons, etc.) backed by per-station Parquet files
- `app/llm.py` — llama.cpp client; the LLM picks a query ID + params from the registry, never executes code
- `app/static/index.html` — single-file UI (Leaflet map + Plotly charts + chat panel)
- `scripts/preprocess.py` — CSV → Parquet conversion + `stations.json` extraction
- `processed/` — generated artifacts (Parquet per station, stations index)

## Data

CSV files in `data/` contain 5-minute rainfall readings from weather stations across Singapore, spanning 2016–2024 (~7 GB total).

**CSV schema:** `date, timestamp, update_timestamp, station_id, station_name, station_device_id, location_longitude, location_latitude, reading_update_timestamp, reading_value, reading_type, reading_unit`

- Readings are in millimeters (mm), recorded at 5-minute intervals
- Each row corresponds to one station's reading at one timestamp
- Station locations are given as longitude/latitude pairs
- Files are large (700 MB–1.1 GB each for 2017–2024; 2016 is ~900 KB)
