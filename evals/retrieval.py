"""
retrieval.py — Retrieval eval: recall@k for vector-only, BM25-only and hybrid (RRF).

Corpus: the seeded past incidents, split into sections (symptoms, root cause, resolution, ...)
so a question has to land on the right *chunk*, the way a crawled docs page would be.
A question counts as a hit if any of the top-k chunks belongs to its relevant incident.

For non-English questions the planner writes English diagnostic questions, so retrieval is
scored on `retrieval_query` (what the retriever actually sees) and language detection is
checked separately.

Run:  python -m evals.run_retrieval            (uses the configured embedding backend)
      EMBEDDING_BACKEND=hash python -m evals.run_retrieval   (offline stand-in, see agent/embeddings.py)
"""

import json
import re
from pathlib import Path
from typing import Dict, List

from agent.vectorstore import RETRIEVAL_MODES, rank_ids

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
_SECTION_RE = re.compile(r"^(Incident|Symptoms|Root Cause|Resolution|Prevention|Tags):", re.MULTILINE)


def load_questions() -> List[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def incident_chunks(incidents: List[dict]) -> List[dict]:
    """Split each incident into one chunk per labelled section. Chunk id: INC-001#root-cause."""
    chunks = []
    for inc in incidents:
        marks = list(_SECTION_RE.finditer(inc["text"]))
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(inc["text"])
            label = m.group(1).lower().replace(" ", "-")
            chunks.append({"id": f"{inc['id']}#{label}", "text": inc["text"][m.start():end].strip()})
    return chunks


def build_collection(client, name: str, incidents: List[dict], embedding_function):
    collection = client.get_or_create_collection(
        name=name, embedding_function=embedding_function, metadata={"hnsw:space": "cosine"}
    )
    chunks = incident_chunks(incidents)
    collection.upsert(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks])
    return collection


def recall_at_k(collection, questions: List[dict], mode: str, k: int = 5) -> float:
    hits = 0
    for q in questions:
        ids = rank_ids(collection, q.get("retrieval_query", q["query"]), k, mode)
        hits += any(i.split("#")[0] == q["relevant"] for i in ids)
    return hits / len(questions)


def mrr(collection, questions: List[dict], mode: str, k: int = 5) -> float:
    """Mean reciprocal rank of the first relevant chunk within the top k (0 if absent)."""
    total = 0.0
    for q in questions:
        ids = rank_ids(collection, q.get("retrieval_query", q["query"]), k, mode)
        for rank, i in enumerate(ids, start=1):
            if i.split("#")[0] == q["relevant"]:
                total += 1.0 / rank
                break
    return total / len(questions)


def evaluate(collection, questions: List[dict], k: int = 5) -> Dict[str, Dict[str, float]]:
    """recall@k per mode, overall and per question kind."""
    kinds = sorted({q["kind"] for q in questions})
    out: Dict[str, Dict[str, float]] = {}
    for mode in RETRIEVAL_MODES:
        row = {"all": recall_at_k(collection, questions, mode, k)}
        for kind in kinds:
            row[kind] = recall_at_k(collection, [q for q in questions if q["kind"] == kind], mode, k)
        out[mode] = row
    return out


def format_summary(collection, questions: List[dict], k: int = 5) -> str:
    """Recall@1, recall@k and MRR@k per mode, overall. recall@k alone saturates on a small corpus."""
    lines = [f"{'mode':<10}{'recall@1':>10}{'recall@'+str(k):>10}{'MRR@'+str(k):>10}"]
    for mode in RETRIEVAL_MODES:
        lines.append(
            f"{mode:<10}{recall_at_k(collection, questions, mode, 1):>10.2f}"
            f"{recall_at_k(collection, questions, mode, k):>10.2f}{mrr(collection, questions, mode, k):>10.2f}"
        )
    return "\n".join(lines)


def format_table(results: Dict[str, Dict[str, float]], k: int) -> str:
    cols = list(next(iter(results.values())).keys())
    lines = [f"recall@{k}".ljust(10) + "".join(c.rjust(13) for c in cols)]
    for mode, row in results.items():
        lines.append(mode.ljust(10) + "".join(f"{row[c]:.2f}".rjust(13) for c in cols))
    return "\n".join(lines)
