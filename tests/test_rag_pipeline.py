import pytest
from app.rag_pipeline import build_prompt


def test_build_prompt_contains_question():
    prompt = build_prompt("What is RAG?", ["RAG stands for Retrieval-Augmented Generation."])
    assert "What is RAG?" in prompt


def test_build_prompt_contains_all_chunks():
    chunks = ["Chunk one content.", "Chunk two content."]
    prompt = build_prompt("Any question?", chunks)
    for chunk in chunks:
        assert chunk in prompt


def test_build_prompt_empty_chunks():
    prompt = build_prompt("Any question?", [])
    assert "Any question?" in prompt
    assert "Context:" in prompt
