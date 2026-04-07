import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.pdf_processor import extract_text_from_pdf, split_text_into_chunks_by_section
from app.embeddings import embed_texts
from app.vector_store import build_and_save_index
from app.rag_pipeline import query_rag
from app.s3_utils import upload_file_to_s3, s3_key_for

app = FastAPI(
    title="RAG-Powered Document Q&A API",
    description="Upload a PDF and ask questions grounded in its content.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]
    latency_seconds: float
    grounding_score: float
    hallucination_risk: str


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file.
    Extracts text, splits into chunks, embeds them, and saves FAISS index.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        chunks = split_text_into_chunks_by_section(text, chunk_size=500, chunk_overlap=50)
        embeddings = embed_texts(chunks)
        build_and_save_index(embeddings, chunks)

        # Upload original PDF to S3 for durable storage
        upload_file_to_s3(tmp_path, s3_key_for(file.filename))

        return JSONResponse({
            "message": "PDF uploaded and indexed successfully.",
            "filename": file.filename,
            "num_chunks": len(chunks),
        })
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Ask a question about the uploaded PDF.
    Returns a grounded answer using RAG.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = query_rag(request.question, top_k=request.top_k)
        return QueryResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
