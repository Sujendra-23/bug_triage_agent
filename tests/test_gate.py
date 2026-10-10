"""The retrieval gate: comparison logic and exit codes. Runs offline on the hash embedder."""

import json

import pytest

from evals import gate

BASE = {"vector": {"recall@1": 0.7, "recall@k": 0.9, "mrr@k": 0.8}, "bm25": {"recall@1": 0.6, "recall@k": 0.8, "mrr@k": 0.7}}


def test_compare_passes_when_equal_or_better():
    better = {m: {k: v + 0.05 for k, v in s.items()} for m, s in BASE.items()}
    assert gate.compare(BASE, BASE, 0.0) == []
    assert gate.compare(better, BASE, 0.0) == []


def test_compare_flags_each_metric_below_baseline():
    worse = {"vector": {"recall@1": 0.65, "recall@k": 0.9, "mrr@k": 0.8}, "bm25": dict(BASE["bm25"])}
    problems = gate.compare(worse, BASE, 0.01)
    assert len(problems) == 1 and "vector recall@1" in problems[0]


def test_compare_allows_drops_within_tolerance():
    nudged = {m: {k: v - 0.005 for k, v in s.items()} for m, s in BASE.items()}
    assert gate.compare(nudged, BASE, 0.01) == []


def test_compare_flags_a_missing_mode():
    assert any("hybrid" in p for p in gate.compare({}, {"hybrid": {"recall@1": 0.5}}, 0.01))


def test_check_comparable_reports_model_and_question_changes():
    run = {"embedding_model": "m2", "k": 5, "questions": 23}
    recorded = {"embedding_model": "m1", "k": 5, "questions": 22}
    reasons = gate.check_comparable(run, recorded)
    assert len(reasons) == 2 and any("embedding_model" in r for r in reasons)


def test_refuses_the_hash_embedder_without_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    assert gate.main(["--baseline", str(tmp_path / "b.json")]) == 2


def test_write_then_check_round_trip_and_regression_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    path = tmp_path / "baseline.json"
    assert gate.main(["--allow-hash", "--write-baseline", "--baseline", str(path)]) == 0
    assert gate.main(["--allow-hash", "--baseline", str(path)]) == 0

    record = json.loads(path.read_text())
    record["scores"]["hybrid"]["recall@1"] = 1.0
    record["scores"]["hybrid"]["mrr@k"] = 1.0
    record["scores"]["vector"]["recall@1"] = 1.0
    path.write_text(json.dumps(record))
    assert gate.main(["--allow-hash", "--baseline", str(path)]) == 1


def test_missing_or_mismatched_baseline_is_an_error_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    assert gate.main(["--allow-hash", "--baseline", str(tmp_path / "none.json")]) == 2
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"embedding_model": "all-MiniLM-L6-v2", "k": 5, "questions": 22, "scores": {}}))
    assert gate.main(["--allow-hash", "--baseline", str(path)]) == 2
