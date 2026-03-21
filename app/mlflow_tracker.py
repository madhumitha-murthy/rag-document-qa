import mlflow
import os

EXPERIMENT_NAME = "rag-retrieval-configs"

mlflow.set_tracking_uri(os.path.abspath("mlruns"))


def log_experiment(
    config_name: str,
    chunk_size: int,
    top_k: int,
    latency: float,
    answer: str,
    num_chunks_retrieved: int,
    retrieval_score: float,
    answer_found: int,
) -> None:
    """Log a single RAG config run to MLflow."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=config_name):
        # Params
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("llm", "llama-3.3-70b-versatile")

        # Metrics
        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("answer_length_chars", len(answer))
        mlflow.log_metric("num_chunks_retrieved", num_chunks_retrieved)
        mlflow.log_metric("retrieval_score", retrieval_score)   # lower = more relevant
        mlflow.log_metric("answer_found", answer_found)         # 1 = answered, 0 = not found
