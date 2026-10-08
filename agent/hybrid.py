"""
hybrid.py — Hybrid retrieval: BM25 (lexical) + vector (semantic), merged with
reciprocal-rank fusion.

Why both: vector search finds paraphrases ("connections never released" ~ "pool exhausted")
but can miss exact tokens such as error codes, env var names and version strings. BM25 is
the opposite. RRF merges the two rankings using ranks only, so it needs no score calibration.
"""

import re
from typing import Dict, List, Sequence

from rank_bm25 import BM25Okapi

RRF_K = 60  # standard constant from the original RRF paper

_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:[-.][a-z0-9_]+)*")


def tokenize(text: str) -> List[str]:
    """Lowercase tokens; keeps identifiers like PAYMENT_GATEWAY_URL and v2.3.1 whole,
    and also emits their parts so partial matches (payment, gateway) still score."""
    tokens: List[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        parts = re.split(r"[_.\-]", tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    return tokens


def bm25_rank(query: str, ids: Sequence[str], documents: Sequence[str], limit: int) -> List[str]:
    """Return up to `limit` ids ordered by BM25 score. Documents with score 0 are dropped."""
    if not ids:
        return []
    bm25 = BM25Okapi([tokenize(d) or ["_"] for d in documents])
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    return [ids[i] for i in ranked[:limit] if scores[i] > 0]


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> List[str]:
    """Reciprocal-rank fusion: score(d) = sum over rankings of 1 / (k + rank(d)), rank from 1."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # ties broken by first appearance so the result is deterministic
    first_seen = {}
    for ranking in rankings:
        for doc_id in ranking:
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))
