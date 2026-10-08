"""
Retrieval eval as a test. Uses the offline hash embedder (see agent/embeddings.py), which matches
spelling rather than meaning, so these thresholds check the harness and the fusion logic. For
real quality numbers run `python -m evals.run_retrieval` with the sentence-transformers backend.
"""

import pytest

from agent.language import detect_language
from agent.vectorstore import rank_ids
from evals.retrieval import build_collection, evaluate, format_summary, format_table, incident_chunks, load_questions
from seed_vectorstore import PAST_INCIDENTS


@pytest.fixture()
def eval_collection(tmp_path):
    import chromadb
    from agent.embeddings import HashEmbeddingFunction

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return build_collection(client, "eval", PAST_INCIDENTS, HashEmbeddingFunction())


def test_eval_set_shape():
    qs = load_questions()
    assert 20 <= len(qs) <= 30
    assert len({q["id"] for q in qs}) == len(qs)
    assert sum(q["kind"] == "non-english" for q in qs) >= 3
    known = {i["id"] for i in PAST_INCIDENTS}
    assert all(q["relevant"] in known for q in qs)


def test_non_english_questions_are_detected_as_such():
    for q in load_questions():
        if q["kind"] == "non-english":
            assert detect_language(q["query"]) == q["lang"], q["id"]
            assert "retrieval_query" in q


def test_incidents_split_into_sections():
    chunks = incident_chunks(PAST_INCIDENTS)
    assert len(chunks) >= 4 * len(PAST_INCIDENTS)
    assert {"INC-001#root-cause", "INC-008#resolution"} <= {c["id"] for c in chunks}


def test_recall_at_5_vector_vs_bm25_vs_hybrid(eval_collection, capsys):
    results = evaluate(eval_collection, load_questions(), k=5)
    print("\n" + format_summary(eval_collection, load_questions(), 5) + "\n\n" + format_table(results, 5))
    assert results["hybrid"]["all"] >= results["vector"]["all"]
    assert results["hybrid"]["all"] >= 0.8
    # exact-token questions are the case hybrid exists for
    assert results["hybrid"]["exact"] >= results["vector"]["exact"]


def test_hybrid_surfaces_exact_identifier_that_vector_misses(eval_collection):
    ids = rank_ids(eval_collection, "PAYMENT_GATEWAY_URL", 3, "hybrid")
    assert ids[0].startswith("INC-008")


def test_invalid_mode_rejected(eval_collection):
    with pytest.raises(ValueError):
        rank_ids(eval_collection, "x", 3, "semantic")
