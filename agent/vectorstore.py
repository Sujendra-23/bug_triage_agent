import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "past_incidents"


def _get_collection():
    """Return (or create) the ChromaDB collection using local sentence-transformers embeddings."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
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
        metadatas=[i.get("metadata", {}) for i in incidents],
    )
    print(f"Seeded {len(incidents)} incidents into ChromaDB.")


def retrieve(queries: List[str], n_results: int = 3) -> List[str]:
    """
    Retrieve the most relevant past incidents for a list of diagnostic questions.

    Returns a flat de-duplicated list of incident texts.
    """
    collection = _get_collection()
    seen_ids: set = set()
    results: List[str] = []

    for query in queries:
        response = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "ids"],
        )
        for doc_list, id_list in zip(response["documents"], response["ids"]):
            for doc, doc_id in zip(doc_list, id_list):
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    results.append(doc)

    return results
