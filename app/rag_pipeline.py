import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.embeddings import embed_query
from app.vector_store import search

load_dotenv()


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
    )


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find the answer in the document."

Context:
{context}

Question: {question}
Answer:"""


def query_rag(question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question
    2. Retrieve top-k chunks from FAISS
    3. Build prompt and call Groq LLM
    Returns answer, retrieved chunks, and latency.
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

    latency = round(time.time() - start, 3)

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "latency_seconds": latency,
    }
