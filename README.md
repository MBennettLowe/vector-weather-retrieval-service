Vector Weather Retrieval Service

Homework submission for DataExpert.io — Weather Intelligence: Unstructured Data → Lakebase Vector Search → REST API (assignment 4938).

This project extends the databricks-lakebase-app-day-2 reference app pattern (structured sync + pgvector-backed retrieval-augmented search) to a new unstructured data source: weather. It harvests free-text weather alerts and forecast narratives from the National Weather Service API, embeds them with sentence-transformers, stores the vectors in Postgres/Lakebase via pgvector, and exposes a semantic search endpoint over the result.

POST /weather/search {"query": "flash flood risk this weekend"}

returns the most semantically relevant weather documents, ranked by cosine similarity.

Architecture
                 ┌────────────────────┐
  NWS API  ───▶  │  weather_client.py │  harvest + normalize
 (api.weather    └─────────┬──────────┘
   .gov)                   │
                            ▼
                 ┌────────────────────┐
                  POST /weather/sync  │  upsert into Lakebase
                 └─────────┬──────────┘
                            ▼
                 ┌────────────────────┐
                  weather_documents   │  raw text + metadata (Postgres)
                 └─────────┬──────────┘
                            ▼
        notebooks/ingest_weather_embeddings.py   chunk + embed
                            │
                            ▼
                 ┌────────────────────┐
                  weather_embeddings  │  vector(384) + pgvector index
                 └─────────┬──────────┘
                            ▼
                 ┌────────────────────┐
                 POST /weather/search │  cosine similarity via `<=>`
                 └────────────────────┘
Project layout
File	Purpose
weather_client.py	NWS API client: geocodes locations, resolves grid points, fetches active alerts + forecasts, normalizes to document records.
lakebase.py	get_connection() context manager (psycopg2 + RealDictCursor) and idempotent DDL for weather_documents / weather_embeddings.
app.py	Flask app exposing POST /weather/sync and POST /weather/search.
notebooks/ingest_weather_embeddings.py	Batch job: chunk unembedded weather_documents rows, embed with sentence-transformers, write to weather_embeddings via psycopg2.extras.execute_values.
tests/	Unit tests for chunking and NWS response normalization (no network/DB required).
README_WEATHER.md	Deliverable write-up: data source rationale, schema decisions, run instructions, limitations.

See README_WEATHER.md for the full write-up required by the assignment's deliverables section.

Quickstart
bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in Lakebase credentials + NWS_USER_AGENT

python lakebase.py                 # 1. run migrations (creates tables + pgvector index)
python app.py                      # 2. start the Flask app

curl -X POST localhost:5000/weather/sync \
  -H "Content-Type: application/json" \
  -d '{"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}'

python notebooks/ingest_weather_embeddings.py   # 3. embed newly synced documents

curl -X POST localhost:5000/weather/search \
  -H "Content-Type: application/json" \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'

Run tests:

bash
pip install pytest
pytest tests/
Status
 Part 1 — Harvest (weather_client.py, POST /weather/sync)
 Part 2 — Vectorize (notebooks/ingest_weather_embeddings.py)
 Part 3 — Retrieve (POST /weather/search)
 Stretch goals (see README_WEATHER.md for notes on what's implemented vs. left as future work)