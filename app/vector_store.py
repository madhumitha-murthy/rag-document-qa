import faiss
import numpy as np
import pickle
import os

from app.s3_utils import upload_file_to_s3, download_file_from_s3, s3_key_for

INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"

S3_INDEX_KEY = s3_key_for("index.faiss")
S3_CHUNKS_KEY = s3_key_for("chunks.pkl")


def build_and_save_index(embeddings: np.ndarray, chunks: list[str]) -> None:
    """Build a FAISS flat L2 index, save to disk, and persist to S3."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    # Persist to S3 for durable cross-session storage
    upload_file_to_s3(INDEX_PATH, S3_INDEX_KEY)
    upload_file_to_s3(CHUNKS_PATH, S3_CHUNKS_KEY)


def load_index():
    """Load FAISS index and chunks. Falls back to S3 if not found locally."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        # Try to restore from S3
        index_found = download_file_from_s3(S3_INDEX_KEY, INDEX_PATH)
        chunks_found = download_file_from_s3(S3_CHUNKS_KEY, CHUNKS_PATH)
        if not index_found or not chunks_found:
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
