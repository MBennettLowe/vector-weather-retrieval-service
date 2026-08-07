"""
lakebase.py

Postgres/Lakebase connection helper and DDL for the weather intelligence
pipeline. Mirrors the connection pattern used elsewhere in the reference
app: a `get_connection()` context manager built on psycopg2 with
RealDictCursor, plus idempotent `CREATE TABLE IF NOT EXISTS` migrations
that can be run safely on every startup.

Tables
------
weather_documents
    One row per harvested unstructured weather item (alert or forecast
    period), mirroring the `ticker_news_documents` pattern.

weather_embeddings
    One row per embedded chunk of a `weather_documents.narrative_text`,
    mirroring `ticker_news_embeddings`. Uses pgvector's `vector(384)`
    type to match `sentence-transformers/all-MiniLM-L6-v2`.
"""

import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

DDL_WEATHER_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS weather_documents (
    id              TEXT PRIMARY KEY,
    location        TEXT NOT NULL,
    source_type     TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
    headline        TEXT,
    narrative_text  TEXT NOT NULL,
    issued_at       TIMESTAMPTZ,
    effective_at    TIMESTAMPTZ,
    payload         JSONB,
    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

DDL_WEATHER_DOCUMENTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);
CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);
"""

# NOTE: pgvector must already be enabled on the Lakebase instance
# (CREATE EXTENSION IF NOT EXISTS vector;) -- per the assignment this is
# already the case, so we don't attempt to run that DDL here (it usually
# requires superuser privileges that an app role may not have).
DDL_WEATHER_EMBEDDINGS = f"""
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id              BIGSERIAL PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       VECTOR({EMBEDDING_DIM}) NOT NULL,
    model_name      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);
"""

# HNSW index for fast approximate cosine-similarity search. Falls back to
# ivfflat if the Postgres/pgvector version does not support HNSW.
DDL_WEATHER_EMBEDDINGS_INDEX_HNSW = """
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);
"""

DDL_WEATHER_EMBEDDINGS_INDEX_IVFFLAT = """
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_ivfflat
    ON weather_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


def _connection_kwargs():
    return {
        "host": os.environ.get("LAKEBASE_HOST", "localhost"),
        "port": os.environ.get("LAKEBASE_PORT", "5432"),
        "dbname": os.environ.get("LAKEBASE_DB", "lakebase"),
        "user": os.environ.get("LAKEBASE_USER", "lakebase_user"),
        "password": os.environ.get("LAKEBASE_PASSWORD", ""),
    }


@contextmanager
def get_connection():
    """Yield a psycopg2 connection configured with RealDictCursor.

    Mirrors the existing `lakebase.py` helper used by the news/ticker
    pipeline so the weather pipeline plugs into the same connection
    pooling/config conventions. Commits on clean exit, rolls back on
    exception, and always closes the connection.
    """
    conn = psycopg2.connect(
        cursor_factory=psycopg2.extras.RealDictCursor, **_connection_kwargs()
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_migrations():
    """Create the weather_documents / weather_embeddings tables and
    indexes if they don't already exist. Safe to call on every app
    startup or ingestion run.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_WEATHER_DOCUMENTS)
            cur.execute(DDL_WEATHER_DOCUMENTS_INDEX)
            cur.execute(DDL_WEATHER_EMBEDDINGS)
            try:
                cur.execute(DDL_WEATHER_EMBEDDINGS_INDEX_HNSW)
            except psycopg2.Error:
                conn.rollback()
                cur.execute(DDL_WEATHER_EMBEDDINGS_INDEX_IVFFLAT)


if __name__ == "__main__":
    run_migrations()
    print("weather_documents / weather_embeddings migrations applied.")
