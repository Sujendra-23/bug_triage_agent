import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

import chromadb

from .embeddings import get_embedding_function
from .hybrid import bm25_rank, rrf_fuse

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "past_incidents"
CRAWLED_COLLECTION_NAME = "crawled_docs"

# vector = embeddings only, bm25 = lexical only, hybrid = both merged with reciprocal-rank fusion
RETRIEVAL_MODES = ("hybrid", "vector", "bm25")


def _get_collection(name: str = COLLECTION_NAME):
    """Return (or create) a ChromaDB collection using the configured embedding function."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def add_incidents(incidents: List[dict]):
    """
    Seed the vector store with past incidents.

    Each incident dict should have:
        id       : str
        text     : str  (the incident description + resolution)
        metadata : dict (optional extra fields)
    """
    collection = _get_collection()
    collection.upsert(
        ids=[i["id"] for i in incidents],
        documents=[i["text"] for i in incidents],
        metadatas=[i.get("metadata") or None for i in incidents],
    )
    print(f"Seeded {len(incidents)} incidents into ChromaDB.")


def vector_rank(collection, query: str, limit: int) -> List[str]:
    """Ids ordered by embedding similarity."""
    count = collection.count()
    if count == 0:
        return []
    response = collection.query(query_texts=[query], n_results=min(limit, count), include=[])
    return response["ids"][0]


def rank_ids(collection, query: str, limit: int, mode: str = "hybrid") -> List[str]:
    """Rank a collection's chunk ids for one query under the given retrieval mode."""
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
    pool = max(limit * 4, 20)  # candidates pulled from each ranker before fusion

    def lexical() -> List[str]:
        everything = collection.get(include=["documents"])
        return bm25_rank(query, everything["ids"], everything["documents"], pool)

    if mode == "vector":
        return vector_rank(collection, query, limit)
    if mode == "bm25":
        return lexical()[:limit]
    return rrf_fuse([vector_rank(collection, query, pool), lexical()])[:limit]


def _format(document: str, metadata: Optional[dict]) -> str:
    """Crawled chunks carry their source URL so the analyzer can cite where an answer came from."""
    if metadata and metadata.get("url"):
        return f"[source: {metadata['url']}]\n{document}"
    return document


def retrieve(
    queries: List[str],
    n_results: int = 3,
    mode: Optional[str] = None,
    collection_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Retrieve the most relevant past incidents and crawled docs for a list of diagnostic questions.

    Returns a flat de-duplicated list of texts.

    mode: hybrid (default), vector or bm25. Env RETRIEVAL_MODE sets the default.
    collection_names: defaults to env RETRIEVAL_COLLECTIONS (past_incidents,crawled_docs).
    Empty collections are skipped.
    """
    mode = mode or os.getenv("RETRIEVAL_MODE", "hybrid")
    if collection_names is None:
        raw = os.getenv("RETRIEVAL_COLLECTIONS", f"{COLLECTION_NAME},{CRAWLED_COLLECTION_NAME}")
        collection_names = [n.strip() for n in raw.split(",") if n.strip()]

    seen: set = set()
    results: List[str] = []
    for name in collection_names:
        collection = _get_collection(name)
        if collection.count() == 0:
            continue
        for query in queries:
            ids = rank_ids(collection, query, n_results, mode)
            if not ids:
                continue
            got = collection.get(ids=ids, include=["documents", "metadatas"])
            by_id: Dict[str, tuple] = {
                i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
            }
            for doc_id in ids:  # keep the ranked order
                key = f"{name}:{doc_id}"
                if key not in seen and doc_id in by_id:
                    seen.add(key)
                    results.append(_format(*by_id[doc_id]))
    return results
