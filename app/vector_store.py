import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"


def build_and_save_index(embeddings: np.ndarray, chunks: list[str]) -> None:
    """Build a FAISS flat L2 index and save to disk along with chunk texts."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index():
    """Load FAISS index and chunks from disk."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("No FAISS index found. Please upload a PDF first.")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def search(query_embedding: np.ndarray, top_k: int = 3) -> tuple[list[str], float]:
    """Search FAISS index and return top-k matching chunks and avg retrieval score."""
    index, chunks = load_index()
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    avg_distance = float(np.mean(distances[0]))
    return results, avg_distance
