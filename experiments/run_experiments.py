"""
MLflow Experiment: Compare 3 RAG retrieval configurations.

Configs:
  1. chunk_size=500,  top_k=3  (baseline)
  2. chunk_size=1000, top_k=3  (larger chunks)
  3. chunk_size=500,  top_k=5  (more retrieved chunks)

Usage:
  1. Upload a PDF first via /upload endpoint (or run the API and upload manually).
  2. Set TEST_QUESTION below to a question relevant to your PDF.
  3. Run: python experiments/run_experiments.py
  4. View results: mlflow ui  (opens at http://localhost:5000)
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
TEST_QUESTIONS = [
    "What are Madhumitha's research publications?",
    "What is Madhumitha's work experience?",
    "What ML frameworks and tools does Madhumitha know?",
    "What projects has Madhumitha built?",
    "What are Madhumitha's educational qualifications?",
]
# ─────────────────────────────────────────────────────────────────

CONFIGS = [
    {"name": "config_1_chunk500_top3",  "chunk_size": 500,  "top_k": 3},
    {"name": "config_2_chunk1000_top3", "chunk_size": 1000, "top_k": 3},
    {"name": "config_3_chunk500_top5",  "chunk_size": 500,  "top_k": 5},
]


def run_single_config(pdf_path: str, question: str, chunk_size: int, top_k: int) -> dict:
    """Run full RAG pipeline for a given config and return results."""
    # Re-chunk and re-index for this config
    from app.pdf_processor import extract_text_from_pdf, split_text_into_chunks
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=50)
    embeddings = embed_texts(chunks)
    build_and_save_index(embeddings, chunks)

    # Retrieve
    start = time.time()
    query_emb = embed_query(question)
    retrieved, avg_distance = search(query_emb, top_k=top_k)

    # Generate answer
    prompt = build_prompt(question, retrieved)
    llm = get_llm()
    messages = [
        SystemMessage(content="You are a document Q&A assistant."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    latency = round(time.time() - start, 3)

    answer = response.content
    answer_found = 0 if "could not find" in answer.lower() else 1

    return {
        "answer": answer,
        "latency": latency,
        "num_chunks": len(retrieved),
        "retrieval_score": round(avg_distance, 4),
        "answer_found": answer_found,
    }


def main():
    if not os.path.exists(PDF_PATH):
        print(f"PDF not found at {PDF_PATH}.")
        print("Place a test PDF at that path and re-run.")
        return

    print(f"Running 3 experiments on: {PDF_PATH}")
    print(f"Questions: {len(TEST_QUESTIONS)} questions per config\n")

    for cfg in CONFIGS:
        print(f"Running {cfg['name']} ...")

        latencies, retrieval_scores, answer_founds, answer_lengths = [], [], [], []

        for question in TEST_QUESTIONS:
            result = run_single_config(
                PDF_PATH,
                question,
                chunk_size=cfg["chunk_size"],
                top_k=cfg["top_k"],
            )
            latencies.append(result["latency"])
            retrieval_scores.append(result["retrieval_score"])
            answer_founds.append(result["answer_found"])
            answer_lengths.append(len(result["answer"]))
            print(f"  Q: {question[:60]}...")
            print(f"     answer_found={result['answer_found']} | latency={result['latency']}s | retrieval_score={result['retrieval_score']}")

        # Average metrics across all questions
        avg_latency = round(sum(latencies) / len(latencies), 3)
        avg_retrieval_score = round(sum(retrieval_scores) / len(retrieval_scores), 4)
        avg_answer_found = round(sum(answer_founds) / len(answer_founds), 2)
        avg_answer_length = int(sum(answer_lengths) / len(answer_lengths))

        log_experiment(
            config_name=cfg["name"],
            chunk_size=cfg["chunk_size"],
            top_k=cfg["top_k"],
            latency=avg_latency,
            answer="x" * avg_answer_length,
            num_chunks_retrieved=cfg["top_k"],
            retrieval_score=avg_retrieval_score,
            answer_found=avg_answer_found,
        )

        print(f"\n  AVG → latency={avg_latency}s | retrieval_score={avg_retrieval_score} | answer_found={avg_answer_found}\n")

    print("All experiments logged.")
    print("View results: mlflow ui   →  http://localhost:5000")


if __name__ == "__main__":
    main()
