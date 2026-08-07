import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notebooks.ingest_weather_embeddings import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text(None) == []


def test_short_text_returns_single_chunk():
    text = "Sunny, with a high near 78. Northwest wind around 6 mph."
    assert chunk_text(text) == [text]


def test_long_text_is_split_with_overlap():
    text = "A" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=100)

    assert len(chunks) > 1
    # every chunk except possibly the last is exactly chunk_size long
    for chunk in chunks[:-1]:
        assert len(chunk) == 800
    # consecutive chunks overlap by `overlap` characters
    assert chunks[0][-CHUNK_OVERLAP:] == chunks[1][:CHUNK_OVERLAP]
    # the full text is covered (chunks reconstruct at least len(text) chars
    # once overlap is accounted for)
    assert chunks[-1][-1] == "A"


def test_chunk_size_default_matches_spec():
    assert CHUNK_SIZE == 800
    assert CHUNK_OVERLAP == 100
