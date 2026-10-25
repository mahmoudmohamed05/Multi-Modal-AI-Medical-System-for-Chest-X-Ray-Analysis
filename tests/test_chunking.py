import pytest
from app.rag.document_loader import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ", 100, 10) == []


def test_chunking_preserves_content_approximately():
    text = "A" * 250
    chunks = chunk_text(text, 100, 20)
    assert len(chunks) >= 3
    assert all(chunks)
    assert max(map(len, chunks)) <= 100


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", 100, 100)
