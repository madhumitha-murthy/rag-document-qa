import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np

from app.embeddings import embed_query, embed_texts
from app.vector_store import search

load_dotenv()

# Refusal phrase injected into the prompt — if the LLM echoes this, the
# answer is a clean refusal rather than a hallucination.
_REFUSAL_PHRASE = "I could not find the answer in the document."

# Grounding score thresholds (cosine similarity between answer and best-matching
# retrieved chunk).  Calibrated on the RAG eval set; tune via env vars if needed.
_THRESHOLD_LOW  = float(os.getenv("GROUNDING_THRESHOLD_LOW",  "0.50"))  # >= low risk
_THRESHOLD_MED  = float(os.getenv("GROUNDING_THRESHOLD_MED",  "0.35"))  # >= medium risk
# < _THRESHOLD_MED → high risk


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
    )


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "{_REFUSAL_PHRASE}"

Context:
{context}

Question: {question}
Answer:"""


def grounding_check(answer: str, context_chunks: list[str]) -> tuple[float, str]:
    """
    Measure how well the LLM answer is grounded in the retrieved context.

    Computes cosine similarity between the answer embedding and each retrieved
    chunk embedding, then takes the maximum — i.e., the best-matching chunk.
    A low score means the answer contains content not present in the source
    document, which is the primary hallucination signal in RAG systems.

    Returns
    -------
    grounding_score : float
        Max cosine similarity in [0, 1].  Higher = better grounded.
    hallucination_risk : str
        "refused"  — LLM correctly said it could not find the answer.
        "low"      — grounding_score >= GROUNDING_THRESHOLD_LOW (0.50)
        "medium"   — grounding_score >= GROUNDING_THRESHOLD_MED (0.35)
        "high"     — grounding_score <  GROUNDING_THRESHOLD_MED (0.35)
    """
    # If the LLM refused, that is the correct safe behaviour — not a hallucination.
    if _REFUSAL_PHRASE.lower() in answer.lower():
        return 0.0, "refused"

    if not context_chunks:
        return 0.0, "high"

    # Embed answer (1, 384) and all chunks (N, 384)
    answer_emb = embed_query(answer)          # shape (1, 384)
    chunk_embs = embed_texts(context_chunks)  # shape (N, 384)

    # Cosine similarity: dot product of L2-normalised vectors
    answer_norm = answer_emb / (np.linalg.norm(answer_emb, axis=1, keepdims=True) + 1e-10)
    chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-10)
    similarities = (answer_norm @ chunk_norms.T).flatten()  # shape (N,)

    grounding_score = float(np.max(similarities))

    if grounding_score >= _THRESHOLD_LOW:
        hallucination_risk = "low"
    elif grounding_score >= _THRESHOLD_MED:
        hallucination_risk = "medium"
    else:
        hallucination_risk = "high"

    return round(grounding_score, 4), hallucination_risk


def query_rag(question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question
    2. Retrieve top-k chunks from FAISS
    3. Build prompt and call Groq LLM
    4. Run grounding check on the answer
    Returns answer, retrieved chunks, latency, grounding score, and hallucination risk.
    """
    start = time.time()

    query_embedding = embed_query(question)
    retrieved_chunks, avg_distance = search(query_embedding, top_k=top_k)

    prompt = build_prompt(question, retrieved_chunks)
    llm = get_llm()

    messages = [
        SystemMessage(content="You are a document Q&A assistant."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    answer = response.content

    grounding_score, hallucination_risk = grounding_check(answer, retrieved_chunks)

    latency = round(time.time() - start, 3)

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "latency_seconds": latency,
        "grounding_score": grounding_score,
        "hallucination_risk": hallucination_risk,
    }
