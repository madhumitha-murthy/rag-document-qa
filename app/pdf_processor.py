import re
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Known section headers in a resume/document
SECTION_HEADERS = [
    "Summary", "Education", "Experience", "Projects",
    "Publications", "Skills", "Achievements",
]

# Natural language prefix per section — makes embeddings semantically queryable
SECTION_CONTEXT = {
    "Summary":      "Professional summary and background of Madhumitha:",
    "Education":    "Educational qualifications and degrees of Madhumitha:",
    "Experience":   "Work experience, employment history, and job roles of Madhumitha:",
    "Projects":     "Projects built and developed by Madhumitha:",
    "Publications": "Research publications and papers authored by Madhumitha:",
    "Skills":       "Technical skills, tools, and frameworks known by Madhumitha:",
    "Achievements": "Awards, achievements, and competition results of Madhumitha:",
}

# Regex: match a section header on its own line, allowing leading whitespace
_SECTION_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    return chunks


def split_text_into_chunks_by_section(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[str]:
    """
    Section-aware chunking: detects section headers, splits text within each
    section, and prepends the section name to every chunk.

    This keeps content from different sections (Experience, Education, etc.)
    from being mixed together, improving retrieval precision.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Find all section boundaries
    matches = list(_SECTION_PATTERN.finditer(text))

    # If no headers detected, fall back to flat chunking
    if not matches:
        return split_text_into_chunks(text, chunk_size, chunk_overlap)

    # Build (section_name, section_text) pairs
    sections = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))

    # Chunk within each section with natural language context prefix
    # so the embedding captures both the section type and content
    all_chunks = []
    for section_name, section_text in sections:
        prefix = SECTION_CONTEXT.get(section_name, f"{section_name} of Madhumitha:")
        for chunk in splitter.split_text(section_text):
            all_chunks.append(f"{prefix}\n{chunk}")

    return all_chunks
