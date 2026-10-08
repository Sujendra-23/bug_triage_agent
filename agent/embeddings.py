"""
embeddings.py — Pluggable embedding function for the vector store.

EMBEDDING_BACKEND=sentence-transformers (default)
    Local sentence-transformers model, EMBEDDING_MODEL (default all-MiniLM-L6-v2).
    Set EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2 to embed non-English text directly.

EMBEDDING_BACKEND=hash
    Deterministic character n-gram hashing. No model download and no network, so the test
    suite and CI run offline. It captures spelling overlap, NOT meaning, so treat any
    retrieval numbers produced with it as a harness check, not a quality measurement.
"""

import hashlib
import math
import os
import re
from typing import List

from chromadb import EmbeddingFunction

DEFAULT_MODEL = "all-MiniLM-L6-v2"
HASH_DIM = 512


class HashEmbeddingFunction(EmbeddingFunction):
    """Character 3-5 gram hashing into a fixed-size, L2-normalised vector."""

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        normalized = re.sub(r"\s+", " ", text.lower())
        for n in (3, 4, 5):
            for i in range(len(normalized) - n + 1):
                gram = normalized[i : i + n]
                h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
                vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def name() -> str:
        return "hash-ngram"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbeddingFunction":
        return HashEmbeddingFunction(dim=config.get("dim", HASH_DIM))


def get_embedding_function():
    backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()
    if backend == "hash":
        return HashEmbeddingFunction()
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    )
