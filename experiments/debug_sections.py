"""Debug script to inspect section detection and retrieved chunks."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from app.pdf_processor import extract_text_from_pdf, split_text_into_chunks_by_section, _SECTION_PATTERN
from app.embeddings import embed_texts, embed_query
from app.vector_store import build_and_save_index, search

PDF_PATH = "faiss_index/test.pdf"

# Build index
text = extract_text_from_pdf(PDF_PATH)
chunks = split_text_into_chunks_by_section(text, chunk_size=1000, chunk_overlap=50)
embeddings = embed_texts(chunks)
build_and_save_index(embeddings, chunks)

# Check retrieved chunks for the two failing questions
QUERIES = [
    "What are Madhumitha's research publications?",
    "Where has Madhumitha worked and what were her roles and responsibilities?",
]

for query in QUERIES:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)
    query_emb = embed_query(query)
    retrieved, avg_dist = search(query_emb, top_k=3)
    for i, chunk in enumerate(retrieved):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:300])
