# RAG-Powered Document Q&A API

A production-ready REST API that lets you upload any PDF and ask natural language questions about it — answers are grounded in your document using **Retrieval-Augmented Generation (RAG)**.

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-blue)
![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3.3_70B-F55036)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/Experiment_Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)

---

## What It Does

1. **Upload** a PDF → text is extracted, chunked, and embedded into a FAISS vector store
2. **Ask** a natural language question → relevant chunks are retrieved and passed to Groq's Llama 3.3 70B
3. **Receive** a grounded answer with source chunks and response latency

The included **web UI** (dark theme) lets you do all of this from a browser without touching the API directly.

---

## Architecture

```
PDF Upload
    │
    ▼
pdfplumber (text extraction)
    │
    ▼
RecursiveCharacterTextSplitter  ← chunk_size=500, overlap=50
    │
    ▼
HuggingFace all-MiniLM-L6-v2   ← 384-dim sentence embeddings
    │
    ▼
FAISS IndexFlatL2               ← persisted vector store
    │
    ▼  (at query time: top-k similarity search)
    │
LangChain RAG Pipeline
    │
    ▼
Groq Llama 3.3-70B-versatile   ← temp=0.2 for deterministic answers
    │
    ▼
JSON Response  { answer, retrieved_chunks, latency_seconds }
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| PDF Parsing | pdfplumber |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Store | FAISS (IndexFlatL2) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Orchestration | LangChain, LangChain-Groq |
| Experiment Tracking | MLflow |
| Containerization | Docker + Docker Compose |
| Frontend | Vanilla JS (dark-theme web UI, no framework) |

---

## Quick Start

### Option A — Local

```bash
# 1. Clone and install
git clone https://github.com/madhumitha-murthy/rag-qa-api
cd rag-qa-api
pip install -r requirements.txt

# 2. Add your Groq API key (free at https://console.groq.com)
cp .env.example .env
# Open .env and set GROQ_API_KEY=your_key_here

# 3. Start the server
uvicorn app.main:app --reload
```

- **API + Web UI:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs

### Option B — Docker

```bash
docker-compose up --build
```

- **API + Web UI:** http://localhost:8000
- **MLflow UI:** http://localhost:5000

---

## API Reference

### `POST /upload` — Index a PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

**Response:**
```json
{
  "message": "PDF uploaded and indexed successfully.",
  "filename": "your_document.pdf",
  "num_chunks": 42
}
```

---

### `POST /query` — Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "top_k": 3}'
```

**Response:**
```json
{
  "answer": "The document covers...",
  "retrieved_chunks": ["chunk 1 text...", "chunk 2 text...", "chunk 3 text..."],
  "latency_seconds": 2.14
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Natural language question about the uploaded PDF |
| `top_k` | integer | 3 | Number of document chunks to retrieve (3, 5, or 7) |

---

## Experiment Tracking (MLflow)

Run and compare different retrieval configurations to evaluate answer quality and latency:

```bash
# Place a test PDF at faiss_index/test.pdf, then run:
python experiments/run_experiments.py

# Open results dashboard
mlflow ui   # → http://localhost:5000
```

Tracked metrics per run: `latency_seconds`, `answer_length_chars`, `num_chunks_retrieved`, `retrieval_score` (avg FAISS distance — lower is better), `answer_found` (% of questions answered)

| Config | chunk_size | top_k | Avg Latency | Answer Recall | Notes |
|---|---|---|---|---|---|
| config_1 | 500 | 3 | 0.862s | 60% | Baseline |
| config_2 | 1000 | 3 | 0.554s | **80%** | Best quality |
| config_3 | 500 | 5 | 0.503s | 60% | More chunks |

---

## Project Structure

```
rag-qa-api/
├── app/
│   ├── main.py              # FastAPI app — /upload and /query endpoints
│   ├── pdf_processor.py     # PDF text extraction and recursive chunking
│   ├── embeddings.py        # Sentence Transformer embeddings (cached)
│   ├── vector_store.py      # FAISS index build, persist, and search
│   ├── rag_pipeline.py      # LangChain RAG chain + Groq LLM integration
│   ├── mlflow_tracker.py    # Experiment logging helpers
│   └── static/
│       └── index.html       # Dark-theme web UI (drag & drop, real-time feedback)
├── experiments/
│   └── run_experiments.py   # MLflow experiment runner (3 configs)
├── Dockerfile
├── docker-compose.yml       # API + MLflow UI services
├── requirements.txt
└── .env.example             # Environment variable template
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

All other parameters (chunk size, embedding model, LLM temperature) are set in the source and documented in the code.
