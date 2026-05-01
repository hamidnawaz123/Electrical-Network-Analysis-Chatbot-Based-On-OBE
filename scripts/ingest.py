"""
scripts/ingest.py
-----------------
INGESTION PIPELINE — run this ONCE before starting the app.

What it does (step by step):
  1. Load all PDFs and Word files from  docs/ena_docs/
  2. Extract raw text from each file
  3. Split text into small chunks (500 chars, 50 overlap)
  4. Embed each chunk using a sentence-transformer model
  5. Save chunks + embeddings into ChromaDB (local vector store)

Run:
    python scripts/ingest.py
"""

import os
import sys
import pdfplumber
import chromadb

from docx import Document
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# ── Paths ────────────────────────────────────────────────────────────────────

DOCS_DIR   = r"D:\Genai\docs\ena_docs"
CHROMA_DIR = r"D:\Genai\embeddings\chroma_db"

# ── Config ───────────────────────────────────────────────────────────────────

COLLECTION_NAME = "ena_docs"
CHUNK_SIZE      = 500   # characters per chunk
CHUNK_OVERLAP   = 50    # overlap between consecutive chunks
EMBED_MODEL     = "all-MiniLM-L6-v2"   # lightweight, runs on CPU


# ── Step 1: Load documents ───────────────────────────────────────────────────

def load_pdf(filepath: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def load_docx(filepath: str) -> str:
    """Extract all text from a Word (.docx) file."""
    doc = Document(filepath)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def load_all_documents(docs_dir: str) -> list[dict]:
    """
    Walk the docs folder and load every PDF and Word file.
    Returns a list of dicts: {"filename": ..., "text": ...}
    """
    documents = []

    if not os.path.exists(docs_dir):
        print(f"[ERROR] Docs folder not found: {docs_dir}")
        sys.exit(1)

    files = os.listdir(docs_dir)
    if not files:
        print(f"[ERROR] No files found in {docs_dir}")
        sys.exit(1)

    for filename in files:
        filepath = os.path.join(docs_dir, filename)
        ext = filename.lower().split(".")[-1]

        if ext == "pdf":
            print(f"  [PDF]  Loading: {filename}")
            text = load_pdf(filepath)
            documents.append({"filename": filename, "text": text})

        elif ext == "docx":
            print(f"  [DOCX] Loading: {filename}")
            text = load_docx(filepath)
            documents.append({"filename": filename, "text": text})

        else:
            print(f"  [SKIP] Unsupported format: {filename}")

    return documents


# ── Step 2: Chunk text ───────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks.

    Example (chunk_size=10, overlap=3):
      text = "ABCDEFGHIJKLMNOP"
      chunks = ["ABCDEFGHIJ", "HIJKLMNOP", ...]
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


def build_chunks(documents: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    """
    Convert all documents into chunks.
    Returns:
      - ids       : unique ID for each chunk  e.g. "syllabus_0", "syllabus_1"
      - texts     : the chunk text
      - metadatas : source filename for each chunk
    """
    ids, texts, metadatas = [], [], []

    for doc in documents:
        source = doc["filename"]
        doc_chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        base_name = source.replace(".", "_")

        for i, chunk in enumerate(doc_chunks):
            ids.append(f"{base_name}_{i}")
            texts.append(chunk)
            metadatas.append({"source": source})

        print(f"  Chunked '{source}' → {len(doc_chunks)} chunks")

    return ids, texts, metadatas


# ── Step 3: Embed and store in ChromaDB ─────────────────────────────────────

def build_vector_store(ids, texts, metadatas):
    """
    Embed all chunks and save them to ChromaDB.
    This creates a persistent local vector store.
    """

    # Load embedding model (downloads on first run, ~80MB)
    print(f"\n[3] Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    # Embed all chunks at once (batch for speed)
    print(f"    Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Connect to ChromaDB (persistent = saves to disk)
    print(f"\n[4] Saving to ChromaDB at: {CHROMA_DIR}")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection if it exists (fresh rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("    Deleted old collection (rebuilding fresh).")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # Store in batches of 100 (ChromaDB has a per-call limit)
    BATCH = 100
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids        = ids[i : i + BATCH],
            documents  = texts[i : i + BATCH],
            embeddings = embeddings[i : i + BATCH],
            metadatas  = metadatas[i : i + BATCH],
        )

    print(f"    Stored {len(ids)} chunks in collection '{COLLECTION_NAME}'.")
    return collection


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ENA OBE Assistant — Ingestion Pipeline")
    print("=" * 55)

    # Step 1: Load
    print(f"\n[1] Loading documents from: {DOCS_DIR}")
    documents = load_all_documents(DOCS_DIR)
    print(f"    Loaded {len(documents)} document(s).")

    # Step 2: Chunk
    print(f"\n[2] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    ids, texts, metadatas = build_chunks(documents)
    print(f"    Total chunks: {len(ids)}")

    # Step 3: Embed + Store
    build_vector_store(ids, texts, metadatas)

    print("\n[DONE] Ingestion complete. ChromaDB is ready.")
    print("       Now run:  uvicorn main:app --reload")
    print("=" * 55)


if __name__ == "__main__":
    main()
