# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A web application project for visualizing historical rainfall data across Singapore. Currently in the data-collection phase — no application code has been written yet.

## Data

CSV files in `data/` contain 5-minute rainfall readings from weather stations across Singapore, spanning 2016–2024 (~7 GB total).

**CSV schema:** `date, timestamp, update_timestamp, station_id, station_name, station_device_id, location_longitude, location_latitude, reading_update_timestamp, reading_value, reading_type, reading_unit`

- Readings are in millimeters (mm), recorded at 5-minute intervals
- Each row corresponds to one station's reading at one timestamp
- Station locations are given as longitude/latitude pairs
- Files are large (700 MB–1.1 GB each for 2017–2024; 2016 is ~900 KB)
