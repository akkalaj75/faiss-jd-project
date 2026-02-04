import os
import re
from typing import List, Tuple

MAX_CHARS_PER_FILE = 200_000      # safety: trim very large text
MAX_CHUNKS_PER_FILE = 500         # safety cap to avoid runaway chunking
DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 100
MAX_FILE_SIZE_MB = 5              # skip >5MB files by default


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[str]:
    text = clean_text(text)

    # Safety: trim extremely large text
    if len(text) > MAX_CHARS_PER_FILE:
        text = text[:MAX_CHARS_PER_FILE]

    # Safety: overlap must be < chunk_size
    overlap = min(overlap, chunk_size - 1)

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n and len(chunks) < MAX_CHUNKS_PER_FILE:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])

        next_start = end - overlap
        if next_start <= start:  # prevent infinite loops
            next_start = end
        start = next_start

    return chunks


def load_docs(docs_dir: str = "docs") -> Tuple[List[str], List[dict]]:
    chunks: List[str] = []
    meta: List[dict] = []

    for fname in os.listdir(docs_dir):
        if not fname.lower().endswith(".txt"):
            continue

        path = os.path.join(docs_dir, fname)

        # Skip very large files by size on disk
        try:
            size_bytes = os.path.getsize(path)
            if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
                print(f"Skipping large file (>{MAX_FILE_SIZE_MB}MB): {fname}")
                continue
        except Exception:
            pass

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        doc_chunks = chunk_text(text)

        for i, ch in enumerate(doc_chunks):
            chunks.append(ch)
            meta.append({"file": fname, "chunk_id": i})

    return chunks, meta
