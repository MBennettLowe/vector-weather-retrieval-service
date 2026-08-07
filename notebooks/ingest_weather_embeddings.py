"""
notebooks/ingest_weather_embeddings.py

Plain-Python (psycopg2-based) batch job that embeds `weather_documents`
rows and writes the vectors into `weather_embeddings`. Mirrors
`notebooks/ingest_ticker_news_embeddings.py`, but deliberately avoids
`spark.write.jdbc`, which does not work reliably against Lakebase in
this environment.

Run directly:

    python notebooks/ingest_weather_embeddings.py

Or import `run()` from a notebook cell / Databricks job.
"""

import os
import sys

# Allow running this file directly (`python notebooks/ingest_weather_embeddings.py`)
# by adding the project root to sys.path so `import lakebase` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2.extras  # noqa: E402
from psycopg2.extras import execute_values  # noqa: E402

import lakebase  # noqa: E402

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Most NWS narrative text (alert descriptions/instructions, detailed
# forecasts) is well under a thousand characters, but combined
# alert description + instruction text can run long. We reuse the
# project's existing sliding-window chunking convention so both the
# news and weather pipelines stay consistent and queryable together.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

BATCH_SIZE = 64  # documents embedded per model.encode() call


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split `text` into overlapping chunks of at most `chunk_size` chars.

    Short text (the common case for NWS forecast periods) returns a
    single chunk. Long text (combined alert description + instruction)
    is split with `overlap` characters of context carried into the next
    chunk so semantic meaning isn't cut at an arbitrary boundary.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def fetch_unembedded_documents(conn, limit=500):
    """Return weather_documents rows that have no corresponding rows in
    weather_embeddings yet (or that were re-synced after last embedding
    -- i.e. narrative_text changed -- handled here via a simple
    NOT EXISTS check keyed on document_id).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.narrative_text
            FROM weather_documents d
            WHERE d.narrative_text IS NOT NULL
              AND length(trim(d.narrative_text)) > 0
              AND NOT EXISTS (
                  SELECT 1 FROM weather_embeddings e WHERE e.document_id = d.id
              )
            ORDER BY d.synced_at DESC
            LIMIT %s;
            """,
            (limit,),
        )
        return cur.fetchall()


def write_embeddings(conn, rows, model_name=EMBEDDING_MODEL_NAME):
    """Batch-insert (document_id, chunk_index, chunk_text, embedding,
    model_name) rows into weather_embeddings using execute_values for
    throughput, with an ON CONFLICT no-op so re-runs are idempotent.
    """
    if not rows:
        return 0

    insert_sql = """
        INSERT INTO weather_embeddings
            (document_id, chunk_index, chunk_text, embedding, model_name)
        VALUES %s
        ON CONFLICT (document_id, chunk_index) DO NOTHING;
    """
    template = "(%s, %s, %s, %s::vector, %s)"

    values = [
        (document_id, chunk_index, chunk_text, embedding, model_name)
        for (document_id, chunk_index, chunk_text, embedding) in rows
    ]

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, values, template=template, page_size=BATCH_SIZE)

    return len(values)


def run(batch_limit=500):
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    total_written = 0

    with lakebase.get_connection() as conn:
        documents = fetch_unembedded_documents(conn, limit=batch_limit)
        print(f"Found {len(documents)} unembedded weather_documents rows.")

        # Build a flat (document_id, chunk_index, chunk_text) list across
        # all documents so embedding happens in efficient batches rather
        # than one document at a time.
        flat_chunks = []
        for doc in documents:
            for idx, chunk in enumerate(chunk_text(doc["narrative_text"])):
                flat_chunks.append((doc["id"], idx, chunk))

        print(f"Chunked into {len(flat_chunks)} total chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

        for i in range(0, len(flat_chunks), BATCH_SIZE):
            batch = flat_chunks[i : i + BATCH_SIZE]
            texts = [c[2] for c in batch]
            embeddings = model.encode(texts, show_progress_bar=False).tolist()

            rows = [
                (doc_id, chunk_index, chunk_text_, embedding)
                for (doc_id, chunk_index, chunk_text_), embedding in zip(batch, embeddings)
            ]
            written = write_embeddings(conn, rows)
            total_written += written
            print(f"  wrote {written} embeddings (running total: {total_written})")

    print(f"Done. {total_written} embeddings written across {len(documents)} documents.")
    return total_written


if __name__ == "__main__":
    run()
