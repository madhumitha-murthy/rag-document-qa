import numpy as np
import pytest
from app.embeddings import embed_texts, embed_query


def test_embed_texts_shape():
    texts = ["Hello world", "This is a test sentence."]
    embeddings = embed_texts(texts)
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_embed_query_shape():
    query_emb = embed_query("What is the main topic?")
    assert query_emb.shape == (1, 384)
    assert query_emb.dtype == np.float32


def test_embed_texts_returns_different_vectors():
    texts = ["The cat sat on the mat.", "Quantum computing uses qubits."]
    embeddings = embed_texts(texts)
    assert not np.allclose(embeddings[0], embeddings[1])


def test_model_is_cached():
    from app.embeddings import get_model
    m1 = get_model()
    m2 = get_model()
    assert m1 is m2
