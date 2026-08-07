"""
app.py

Flask REST API for the Weather Intelligence pipeline. Adds two endpoints
to the reference app pattern (mirroring /news/sync and its retrieval
counterpart):

    POST /weather/sync    -- harvest + normalize + upsert weather documents
    POST /weather/search  -- semantic search over embedded weather documents

This file is intentionally self-contained (it can run standalone for
local development/grading) but is written so the two route functions can
be copy-pasted directly into an existing `app.py` that already defines
`/news/sync` and `/news/search` in the reference app.
"""

import os

import psycopg2.extras
from flask import Flask, jsonify, request

import lakebase
import weather_client

app = Flask(__name__)

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Loaded lazily on first request so `flask --help` / import-time tooling
# doesn't have to pull down model weights.
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


VALID_SOURCE_TYPES = {"alert", "forecast"}


@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    """Harvest unstructured weather text for the requested locations and
    upsert it into `weather_documents`.

    Body: {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
    """
    body = request.get_json(silent=True) or {}
    locations = body.get("locations")
    limit = body.get("limit", 50)

    if not locations or not isinstance(locations, list):
        return jsonify({"error": "'locations' must be a non-empty list of strings"}), 400

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return jsonify({"error": "'limit' must be an integer"}), 400
    limit = max(1, min(limit, 200))

    documents = weather_client.harvest_locations(locations, limit=limit)

    valid_docs = [d for d in documents if d["source_type"] in VALID_SOURCE_TYPES]
    errors = [d for d in documents if d["source_type"] not in VALID_SOURCE_TYPES]

    synced_count = 0
    if valid_docs:
        with lakebase.get_connection() as conn:
            with conn.cursor() as cur:
                upsert_sql = """
                    INSERT INTO weather_documents
                        (id, location, source_type, headline, narrative_text,
                         issued_at, effective_at, payload, synced_at)
                    VALUES
                        (%(id)s, %(location)s, %(source_type)s, %(headline)s,
                         %(narrative_text)s, %(issued_at)s, %(effective_at)s,
                         %(payload)s, %(synced_at)s)
                    ON CONFLICT (id) DO UPDATE SET
                        headline = EXCLUDED.headline,
                        narrative_text = EXCLUDED.narrative_text,
                        issued_at = EXCLUDED.issued_at,
                        effective_at = EXCLUDED.effective_at,
                        payload = EXCLUDED.payload,
                        synced_at = EXCLUDED.synced_at;
                """
                for doc in valid_docs:
                    row = dict(doc)
                    row["payload"] = psycopg2.extras.Json(row["payload"])
                    cur.execute(upsert_sql, row)
                    synced_count += 1

    return jsonify(
        {
            "locations_requested": locations,
            "documents_synced": synced_count,
            "errors": [{"location": e["location"], "message": e["narrative_text"]} for e in errors],
        }
    )


@app.route("/weather/search", methods=["POST"])
def weather_search():
    """Semantic search over `weather_embeddings` via pgvector cosine distance.

    Body: {"query": "flash flood risk this weekend", "top_k": 5}
    """
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    top_k = body.get("top_k", 5)

    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "'query' is required and must be a non-empty string"}), 400

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5
    top_k = max(1, min(top_k, 20))

    model = get_model()
    query_embedding = model.encode(query.strip()).tolist()

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM weather_embeddings;")
            if cur.fetchone()["count"] == 0:
                return jsonify(
                    {
                        "query": query,
                        "results": [],
                        "message": "No weather documents have been embedded yet. "
                        "Run POST /weather/sync followed by the embedding "
                        "ingestion script first.",
                    }
                )

            cur.execute(
                """
                SELECT d.location, d.headline, d.narrative_text, e.chunk_text,
                       1 - (e.embedding <=> %(embedding)s::vector) AS similarity
                FROM weather_embeddings e
                JOIN weather_documents d ON d.id = e.document_id
                ORDER BY e.embedding <=> %(embedding)s::vector
                LIMIT %(top_k)s;
                """,
                {"embedding": query_embedding, "top_k": top_k},
            )
            rows = cur.fetchall()

    results = [
        {
            "location": row["location"],
            "headline": row["headline"],
            "chunk_text": row["chunk_text"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]

    return jsonify({"query": query, "results": results})


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    lakebase.run_migrations()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
