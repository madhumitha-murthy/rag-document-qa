"""
RAGAS evaluation — measures RAG output quality on a set of test questions.

Metrics:
  faithfulness      — does the answer stay within the retrieved context?
  answer_relevancy  — is the answer relevant to the question asked?

Usage:
    1. Start the API and ingest a PDF first (POST /upload)
    2. Set GROQ_API_KEY in your .env file
    3. Run:  python evaluate_rag.py

Results are printed to console and logged to MLflow.
View in MLflow UI:  mlflow ui  →  http://localhost:5000
"""

import os
import mlflow
import numpy as np
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy  # noqa: deprecated warning is fine — collections API requires OpenAI only
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from app.rag_pipeline import query_rag


class GroqSingleN(ChatGroq):
    """Groq wrapper that strips n>1 requests — Groq only supports n=1."""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

load_dotenv()

# ── Test questions — edit to match your ingested PDF ───────────────
TEST_QUESTIONS = [
    "What degree is the candidate currently pursuing and at which university?",
    "What programming languages does the candidate know?",
    "What machine learning projects has the candidate worked on?",
    "Where has the candidate worked or done research?",
    "What tools and frameworks does the candidate have experience with?",
    "What is the candidate's educational background?",
    "What NLP or deep learning experience does the candidate have?",
]

mlflow.set_experiment("ragas-evaluation")


def run_ragas_evaluation() -> dict:
    # Configure RAGAS to use Groq (same LLM as the pipeline)
    groq_llm = LangchainLLMWrapper(GroqSingleN(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
    ))
    hf_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    faithfulness.llm = groq_llm
    answer_relevancy.llm = groq_llm
    answer_relevancy.embeddings = hf_embeddings

    # ── Run RAG pipeline on each test question ─────────────────────
    questions, answers, contexts = [], [], []

    print("Running RAG pipeline on test questions...\n")
    for q in TEST_QUESTIONS:
        result = query_rag(q, top_k=5)
        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["retrieved_chunks"])
        print(f"  Q: {q}")
        print(f"  A: {result['answer'][:120]}...")
        print()

    # ── Build HuggingFace Dataset (required by RAGAS) ──────────────
    dataset = Dataset.from_dict({
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    })

    # ── Run RAGAS ──────────────────────────────────────────────────
    print("Running RAGAS evaluation...")
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    faithfulness_score    = round(float(np.nanmean(scores["faithfulness"])),    3)
    answer_relevancy_score = round(float(np.nanmean(scores["answer_relevancy"])), 3)

    print(f"\nRAGAS Results:")
    print(f"  Faithfulness:     {faithfulness_score}  (1.0 = fully grounded in context, zero hallucination)")
    print(f"  Answer Relevancy: {answer_relevancy_score}  (1.0 = fully relevant to question)")

    # ── Log to MLflow ──────────────────────────────────────────────
    with mlflow.start_run(run_name="ragas-eval"):
        mlflow.log_metric("ragas_faithfulness",     faithfulness_score)
        mlflow.log_metric("ragas_answer_relevancy", answer_relevancy_score)
        mlflow.log_param("num_questions",    len(TEST_QUESTIONS))
        mlflow.log_param("top_k",            5)
        mlflow.log_param("embedding_model",  "all-MiniLM-L6-v2")
        mlflow.log_param("llm",              "llama-3.3-70b-versatile (Groq)")

    print("\n✓ Scores logged to MLflow.")
    print("  Run: mlflow ui  →  open http://localhost:5000")

    return {"faithfulness": faithfulness_score, "answer_relevancy": answer_relevancy_score}


if __name__ == "__main__":
    run_ragas_evaluation()
