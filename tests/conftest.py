"""Run the whole suite offline: hash embeddings (no model download), a temp ChromaDB, no API keys."""

import os
import sys
from pathlib import Path

os.environ["EMBEDDING_BACKEND"] = "hash"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
import pytest

from agent.embeddings import HashEmbeddingFunction


@pytest.fixture()
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return client.get_or_create_collection(
        "crawl-test", embedding_function=HashEmbeddingFunction(), metadata={"hnsw:space": "cosine"}
    )


os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")  # nodes.py builds the client at import time
