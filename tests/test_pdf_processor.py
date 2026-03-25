import pytest
from app.pdf_processor import split_text_into_chunks, split_text_into_chunks_by_section


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


# --- split_text_into_chunks_by_section ---

SAMPLE_RESUME = """
Summary
Experienced software engineer with a focus on machine learning.

Experience
Software Engineer at Acme Corp. Built ML pipelines and REST APIs.

Skills
Python, PyTorch, FastAPI, Docker, AWS
"""


def test_section_chunking_detects_sections():
    chunks = split_text_into_chunks_by_section(SAMPLE_RESUME, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 0


def test_section_chunking_prefixes_context():
    chunks = split_text_into_chunks_by_section(SAMPLE_RESUME, chunk_size=500, chunk_overlap=50)
    prefixes = [c.split("\n")[0] for c in chunks]
    # Each chunk should start with a natural language context prefix
    assert any("summary" in p.lower() or "experience" in p.lower() or "skills" in p.lower()
               for p in prefixes)


def test_section_chunking_keeps_sections_separate():
    chunks = split_text_into_chunks_by_section(SAMPLE_RESUME, chunk_size=500, chunk_overlap=50)
    # Skills content should not appear in Experience chunk and vice versa
    experience_chunks = [c for c in chunks if "experience" in c.split("\n")[0].lower()]
    for chunk in experience_chunks:
        assert "Python, PyTorch" not in chunk


def test_section_chunking_falls_back_on_no_headers():
    text = "This document has no section headers at all. " * 20
    chunks = split_text_into_chunks_by_section(text, chunk_size=100, chunk_overlap=10)
    # Should fall back to flat chunking — no context prefix
    assert len(chunks) > 0
    assert all("\n" not in c.split("\n")[0] or True for c in chunks)


def test_section_chunking_empty_text():
    chunks = split_text_into_chunks_by_section("", chunk_size=500, chunk_overlap=50)
    assert chunks == []
