"""
src/pipeline.py
---------------
RAG RETRIEVAL PIPELINE

What it does:
  1. Takes a user query (question or topic)
  2. Embeds the query using the same model used in ingest.py
  3. Searches ChromaDB for the top-k most relevant chunks
  4. Returns those chunks as context for Claude

Used by:
  - src/routers/chat.py       (chatbot)
  - src/routers/questions.py  (question generator)
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths & Config ────────────────────────────────────────────────────────────

CHROMA_DIR      = r"D:\Genai\embeddings\chroma_db"
COLLECTION_NAME = "ena_docs"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # must match ingest.py
TOP_K           = 4                     # number of chunks to retrieve

# ── Load model and DB once (at import time) ───────────────────────────────────

print("[pipeline] Loading embedding model...")
_model = SentenceTransformer(EMBED_MODEL)

print("[pipeline] Connecting to ChromaDB...")
_client     = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_collection(COLLECTION_NAME)
print(f"[pipeline] Connected. Collection has {_collection.count()} chunks.")


# ── Main retrieval function ───────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Given a query string, return the top-k most relevant chunks.

    Returns a list of dicts:
      [
        {"text": "...", "source": "syllabus.pdf"},
        {"text": "...", "source": "ENA_Lab_Manual.pdf"},
        ...
      ]
    """

    # Step 1: embed the query
    query_embedding = _model.encode(query).tolist()

    # Step 2: search ChromaDB
    results = _collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
    )

    # Step 3: unpack results into clean list
    chunks = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text":   text,
            "source": metadata.get("source", "unknown"),
        })

    return chunks


def build_context(chunks: list[dict]) -> str:
    """
    Combine retrieved chunks into a single context string for Claude.

    Format:
      [Source: syllabus.pdf]
      <chunk text>

      [Source: ENA_Lab_Manual.pdf]
      <chunk text>
      ...
    """
    parts = []
    for chunk in chunks:
        parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(parts)
