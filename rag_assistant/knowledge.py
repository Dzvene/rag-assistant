"""ChromaDB wrapper — stores and searches knowledge chunks."""

import os
import chromadb
from chromadb.config import Settings
from . import embedder

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db")

COLLECTIONS = {
    "interview": "Interview prep — CV, tech stack, answers",
    "german":    "German language — grammar rules, vocabulary",
}

_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_collection(name: str):
    return _get_client().get_or_create_collection(name)


def add_chunks(collection: str, chunks: list[dict]):
    """Add chunks to a collection.

    Each chunk: {"id": str, "text": str, "meta": dict}
    """
    col = _get_collection(collection)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)
    col.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c.get("meta", {}) for c in chunks],
    )


def search(collection: str, query: str, n: int = 4) -> list[dict]:
    """Return top-n relevant chunks for a query."""
    col = _get_collection(collection)
    if col.count() == 0:
        return []
    q_emb = embedder.embed([query])
    results = col.query(
        query_embeddings=q_emb,
        n_results=min(n, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({"text": doc, "meta": meta, "score": 1 - dist})
    return out


def delete_collection(collection: str):
    _get_client().delete_collection(collection)


def stats() -> dict:
    client = _get_client()
    return {
        name: client.get_or_create_collection(name).count()
        for name in COLLECTIONS
    }
