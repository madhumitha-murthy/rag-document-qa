"""
Qdrant vector store — drop-in replacement for vector_store.py.

Qdrant is a production-grade vector database (vs FAISS which is a library).

Run Qdrant locally with Docker before using this module:
    docker run -p 6333:6333 qdrant/qdrant

For quick local testing without Docker, swap QDRANT_URL to in-memory mode:
    QdrantClient(":memory:")
"""

import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "rag_documents"
VECTOR_SIZE     = 384          # all-MiniLM-L6-v2 output dimension
QDRANT_URL      = "http://localhost:6333"  # swap to ":memory:" for in-memory mode

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def build_and_save_index(embeddings: np.ndarray, chunks: list[str]) -> None:
    """
    Create (or recreate) a Qdrant collection and upload embeddings + chunks.
    Re-ingesting a PDF drops the existing collection and rebuilds from scratch.
    """
    client = get_client()

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={"text": chunks[i]},
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[qdrant] Indexed {len(chunks)} chunks into '{COLLECTION_NAME}'")


def search(query_embedding: np.ndarray, top_k: int = 3) -> tuple[list[str], float]:
    """
    Search Qdrant collection and return top-k matching chunks and avg cosine score.
    Higher score = more similar (cosine distance, range 0–1).
    """
    client = get_client()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding[0].tolist(),
        limit=top_k,
    )

    chunks    = [r.payload["text"] for r in results]
    avg_score = float(np.mean([r.score for r in results])) if results else 0.0

    return chunks, avg_score
