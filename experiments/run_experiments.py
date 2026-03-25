"""
MLflow Experiment: Compare 3 RAG retrieval configurations.

Configs:
  1. chunk_size=500,  top_k=3  (baseline)
  2. chunk_size=1000, top_k=3  (larger chunks)
  3. chunk_size=500,  top_k=5  (more retrieved chunks)

Usage:
  1. Place a test PDF at faiss_index/test.pdf
  2. Run: python experiments/run_experiments.py
  3. View results: mlflow ui  (opens at http://localhost:5000)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.pdf_processor import extract_text_from_pdf, split_text_into_chunks
from app.embeddings import embed_texts, embed_query
from app.vector_store import build_and_save_index, search
from app.rag_pipeline import build_prompt, get_llm
from app.mlflow_tracker import log_experiment
from langchain_core.messages import HumanMessage, SystemMessage
import time

# ── Configure these ──────────────────────────────────────────────
PDF_PATH = "faiss_index/test.pdf"

# Ground-truth questions with expected keywords for keyword-based recall
TEST_QUESTIONS = [
    {
        "question": "What are Madhumitha's research publications?",
        "keywords": ["IEEE", "npj", "EACL", "published", "workshop"],
    },
    {
        "question": "What is Madhumitha's work experience?",
        "keywords": ["I2R", "NTU", "Zoho", "Research", "Teaching"],
    },
    {
        "question": "What ML frameworks and tools does Madhumitha know?",
        "keywords": ["PyTorch", "TensorFlow", "HuggingFace", "scikit-learn", "FastAPI"],
    },
    {
        "question": "What projects has Madhumitha built?",
        "keywords": ["document", "fine-tuning", "anomaly", "Dravidian", "sentiment"],
    },
    {
        "question": "What are Madhumitha's educational qualifications?",
        "keywords": ["MSc", "NTU", "Nanyang", "Bachelor", "Anna University"],
    },
]
# ─────────────────────────────────────────────────────────────────

CONFIGS = [
    {"name": "config_1_chunk500_top3",  "chunk_size": 500,  "top_k": 3},
    {"name": "config_2_chunk1000_top3", "chunk_size": 1000, "top_k": 3},
    {"name": "config_3_chunk500_top5",  "chunk_size": 500,  "top_k": 5},
]


def compute_keyword_recall(answer: str, keywords: list[str]) -> float:
    """Return fraction of expected keywords found in the answer (case-insensitive)."""
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(found / len(keywords), 4)


def build_index_for_config(pdf_path: str, chunk_size: int) -> None:
    """Extract text, chunk, embed, and save FAISS index. Called once per config."""
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=50)
    embeddings = embed_texts(chunks)
    build_and_save_index(embeddings, chunks)


def run_question(question: str, top_k: int) -> dict:
    """Run retrieval + generation for a single question against the current index."""
    start = time.time()
    query_emb = embed_query(question)
    retrieved, avg_distance = search(query_emb, top_k=top_k)

    prompt = build_prompt(question, retrieved)
    llm = get_llm()
    messages = [
        SystemMessage(content="You are a document Q&A assistant."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    latency = round(time.time() - start, 3)

    return {
        "answer": response.content,
        "latency": latency,
        "avg_distance": round(avg_distance, 4),
    }


def main():
    if not os.path.exists(PDF_PATH):
        print(f"PDF not found at {PDF_PATH}.")
        print("Place a test PDF at that path and re-run.")
        return

    print(f"Running 3 experiments on: {PDF_PATH}")
    print(f"Questions: {len(TEST_QUESTIONS)} per config\n")

    for cfg in CONFIGS:
        print(f"── {cfg['name']} ──")

        # Build index ONCE per config, not once per question
        print(f"  Building index (chunk_size={cfg['chunk_size']})...")
        build_index_for_config(PDF_PATH, cfg["chunk_size"])

        latencies, distances, recalls, answer_lengths = [], [], [], []

        for item in TEST_QUESTIONS:
            result = run_question(item["question"], top_k=cfg["top_k"])
            recall = compute_keyword_recall(result["answer"], item["keywords"])

            latencies.append(result["latency"])
            distances.append(result["avg_distance"])
            recalls.append(recall)
            answer_lengths.append(len(result["answer"]))

            print(f"  Q: {item['question'][:60]}...")
            print(f"     recall={recall:.2f} | latency={result['latency']}s | dist={result['avg_distance']}")

        avg_latency = round(sum(latencies) / len(latencies), 3)
        avg_distance = round(sum(distances) / len(distances), 4)
        avg_recall = round(sum(recalls) / len(recalls), 4)
        avg_answer_length = int(sum(answer_lengths) / len(answer_lengths))

        log_experiment(
            config_name=cfg["name"],
            chunk_size=cfg["chunk_size"],
            top_k=cfg["top_k"],
            latency=avg_latency,
            answer_length=avg_answer_length,
            num_chunks_retrieved=cfg["top_k"],
            retrieval_score=avg_distance,
            recall=avg_recall,
        )

        print(f"\n  AVG → latency={avg_latency}s | keyword_recall={avg_recall} | retrieval_score={avg_distance}\n")

    print("All experiments logged.")
    print("View results: mlflow ui   →  http://localhost:5000")


if __name__ == "__main__":
    main()
