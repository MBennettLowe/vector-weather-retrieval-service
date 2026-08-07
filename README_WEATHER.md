# Weather Intelligence Pipeline — Design Notes

This document is the deliverable write-up required by assignment 4938: data source rationale, schema decisions, how to run the pipeline end-to-end, and known limitations.

## 1. Data source: National Weather Service API (`api.weather.gov`)

I used the recommended NWS API rather than the alternatives, for three reasons:

1. **No API key.** `api.weather.gov` only requires a descriptive `User-Agent` header (set via the `NWS_USER_AGENT` env var), so the assignment can focus on harvesting/vectorization/retrieval instead of auth/secrets plumbing.
2. **Rich free text for embedding.** Two fields map cleanly onto the project's "unstructured narrative text" requirement: an alert's `description` + `instruction` (e.g. *"A Flash Flood Warning means..."*), and a forecast period's `detailedForecast` (e.g. *"Sunny, with a high near 78. Northwest wind around 6 mph."*). Both are genuinely free-form prose, not templated key/value data, so semantic search over them is meaningful.
3. **Two source types, one schema.** Alerts and forecasts have different cadences and lengths but normalize into the same document shape (`source_type` discriminates them), which let me demonstrate the multi-source filtering pattern (`source_type = 'alert' | 'forecast'`) without needing a second external API.

I did **not** mix in OpenWeatherMap or NOAA CPC discussion text — the assignment says not to mix sources unless going for the multi-source extra credit, and NWS alerts + forecasts already gave two distinct narrative shapes to exercise the chunking logic.

## 2. Schema decisions

### `weather_documents`

Mirrors `ticker_news_documents`:

| Column | Type | Notes |
|---|---|---|
| `id` | `TEXT PRIMARY KEY` | NWS alert `properties.id` (a stable URN) for alerts; `sha256(location + period name + startTime)[:32]` for forecast periods, since NWS doesn't give forecast periods a stable id. |
| `location` | `TEXT` | The city/state or lat,lon string the caller requested. |
| `source_type` | `TEXT CHECK IN ('alert','forecast')` | Discriminator for filtering/retrieval. |
| `headline` | `TEXT` | Alert `event` (e.g. "Flash Flood Warning") or forecast period `name` (e.g. "Tonight"). |
| `narrative_text` | `TEXT` | The text that gets embedded — alert `description` + `instruction` concatenated, or forecast `detailedForecast`. |
| `issued_at` / `effective_at` | `TIMESTAMPTZ` | Alert `sent`/`effective`, or forecast period `startTime` for both (NWS doesn't separate issue vs. effective time for forecast periods). |
| `payload` | `JSONB` | Full raw feature/period JSON, for provenance/debugging. |
| `synced_at` | `TIMESTAMPTZ` | Set at upsert time. |

Upserts key on `id` (`ON CONFLICT (id) DO UPDATE`) so re-running `/weather/sync` refreshes existing rows (e.g. an alert's `instruction` text changing) instead of duplicating them.

### `weather_embeddings`

Mirrors `ticker_news_embeddings`:

| Column | Type | Notes |
|---|---|---|
| `id` | `BIGSERIAL PRIMARY KEY` | |
| `document_id` | `TEXT REFERENCES weather_documents(id) ON DELETE CASCADE` | |
| `chunk_index` | `INTEGER` | 0-based position within the source document. |
| `chunk_text` | `TEXT` | The exact substring that was embedded. |
| `embedding` | `VECTOR(384)` | pgvector column; see model choice below. |
| `model_name` | `TEXT` | Recorded per-row so the table can hold embeddings from more than one model generation if the model is ever upgraded. |
| `created_at` | `TIMESTAMPTZ` | |

`UNIQUE (document_id, chunk_index)` + `ON CONFLICT DO NOTHING` on insert makes the embedding job idempotent — re-running it after a partial failure doesn't create duplicate vectors.

Index: `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`, falling back to `ivfflat` if the Postgres/pgvector build doesn't support HNSW (`lakebase.run_migrations()` tries HNSW first and catches the error).

### Chunking parameters

`CHUNK_SIZE = 800`, `CHUNK_OVERLAP = 100` — kept identical to the existing news pipeline's convention rather than inventing new numbers, since:

- Most NWS forecast `detailedForecast` strings are well under 800 characters and end up as a single chunk.
- Combined alert `description + instruction` text for significant events (e.g. tornado warnings) regularly exceeds 800 characters, so chunking does matter there; a 100-character overlap keeps a sentence that straddles a chunk boundary from losing context in either half.

### Embedding model

`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions — the same model the existing news/ticker pipeline uses. Keeping the model identical (rather than picking something "better" for weather text) means `weather_embeddings` and `ticker_news_embeddings` stay dimensionally compatible and could be queried with the same distance-operator conventions if the two were ever unioned or compared side by side.

## 3. Running the pipeline end-to-end

```bash
pip install -r requirements.txt
cp .env.example .env        # set LAKEBASE_* and NWS_USER_AGENT

python lakebase.py          # create weather_documents / weather_embeddings + pgvector index
python app.py                # starts Flask on :5000, also re-runs migrations on boot

# 1. Harvest + upsert raw documents
curl -X POST localhost:5000/weather/sync \
  -H "Content-Type: application/json" \
  -d '{"locations": ["Chicago, IL", "Austin, TX", "Miami, FL"], "limit": 50}'

# 2. Chunk + embed newly synced documents
python notebooks/ingest_weather_embeddings.py

# 3. Semantic search
curl -X POST localhost:5000/weather/search \
  -H "Content-Type: application/json" \
  -d '{"query": "risk of flooding near rivers", "top_k": 5}'
```

`notebooks/ingest_weather_embeddings.py` only processes documents that don't already have rows in `weather_embeddings` (a `NOT EXISTS` check keyed on `document_id`), so it can be re-run safely after every `/weather/sync` call to pick up new documents incrementally.

## 4. Known limitations / what I'd improve with more time

- **No re-embedding on text change.** If an alert's `narrative_text` is updated by a re-sync (`ON CONFLICT DO UPDATE`), the ingestion script's `NOT EXISTS` check won't re-embed it, since a row already exists in `weather_embeddings` for that `document_id`. A production version would compare a content hash and re-embed on change.
- **No LLM summary endpoint yet.** The stretch goal of a `GET /weather/search` variant that also returns an LLM-generated natural-language summary (basic RAG) is not implemented — the current `/weather/search` returns raw ranked matches only.
- **No scheduled re-sync.** Alerts are time-sensitive; in production this would run on a Databricks Job / cron every 5–15 minutes rather than being triggered manually via `/weather/sync`.
- **Single-source filtering not exposed via the API.** `source_type` is stored and could be used to let `/weather/search` filter to only `alert` or only `forecast` results, but the current endpoint searches across both.
- **HNSW vs. ivfflat benchmark not included.** `lakebase.py` prefers HNSW and falls back to ivfflat, but I didn't benchmark query latency with vs. without the index (one of the optional stretch goals).
- **Geocoding depends on a third-party service.** Locations given as free text (rather than `"lat,lon"`) are resolved via Nominatim/OpenStreetMap, which is not part of the NWS API itself and has its own rate limits — for heavier use this should be cached or replaced with a bundled geocoding dataset.
