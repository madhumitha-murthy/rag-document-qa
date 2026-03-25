import pytest
from app.pdf_processor import split_text_into_chunks


def test_split_produces_chunks():
    text = "Hello world. " * 100
    chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1


def test_split_respects_chunk_size():
    text = "word " * 500
    chunk_size = 200
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=0)
    for chunk in chunks:
        assert len(chunk) <= chunk_size + 20  # small tolerance for splitter behaviour


def test_split_empty_text_returns_empty():
    chunks = split_text_into_chunks("", chunk_size=500, chunk_overlap=50)
    assert chunks == []


def test_split_short_text_returns_single_chunk():
    text = "Short text."
    chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text
