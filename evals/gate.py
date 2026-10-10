"""
gate.py — Retrieval quality gate. Fails (exit 1) when retrieval gets worse than a stored baseline.

    python -m evals.gate                    compare the current scores with evals/baseline.json
    python -m evals.gate --write-baseline   record the current scores as the new baseline

Scores are recall@1, recall@k and MRR@k for vector, BM25 and hybrid retrieval over the eval set in
evals/questions.json. They only mean something with a real embedding model, so the gate refuses to
run on the offline hash stand-in (EMBEDDING_BACKEND=hash) and refuses to compare against a baseline
recorded with a different model or question set. After an intentional change to retrieval, chunking,
the embedding model or the questions, re-record the baseline and commit it with the change.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

BASELINE_PATH = Path(__file__).parent / "baseline.json"
DEFAULT_TOLERANCE = 0.01  # absolute; one eval question is worth about 0.045 of recall
METRICS = ("recall@1", "recall@k", "mrr@k")

Scores = Dict[str, Dict[str, float]]


def compute_scores(collection, questions: List[dict], k: int) -> Scores:
    """recall@1, recall@k and MRR@k per retrieval mode."""
    from agent.vectorstore import RETRIEVAL_MODES
    from evals.retrieval import mrr, recall_at_k

    return {
        mode: {
            "recall@1": recall_at_k(collection, questions, mode, 1),
            "recall@k": recall_at_k(collection, questions, mode, k),
            "mrr@k": mrr(collection, questions, mode, k),
        }
        for mode in RETRIEVAL_MODES
    }


def compare(scores: Scores, baseline: Scores, tolerance: float) -> List[str]:
    """Return one message per metric that dropped below baseline minus tolerance."""
    problems = []
    for mode, expected in baseline.items():
        for metric in METRICS:
            if metric not in expected:
                continue
            actual = scores.get(mode, {}).get(metric)
            if actual is None:
                problems.append(f"{mode} {metric}: missing from current results")
            elif actual < expected[metric] - tolerance:
                problems.append(f"{mode} {metric}: {actual:.3f} is below baseline {expected[metric]:.3f}")
    return problems


def format_report(scores: Scores, baseline: Scores, k: int, markdown: bool = False) -> str:
    header = ["mode", "metric", "baseline", "current", "delta"]
    rows = []
    for mode, current in scores.items():
        for metric in METRICS:
            base = baseline.get(mode, {}).get(metric)
            label = metric.replace("@k", f"@{k}")
            if base is None:
                rows.append([mode, label, "-", f"{current[metric]:.3f}", "-"])
            else:
                rows.append([mode, label, f"{base:.3f}", f"{current[metric]:.3f}", f"{current[metric] - base:+.3f}"])
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([fmt.format(*header)] + [fmt.format(*r) for r in rows])


def describe_run(k: int, question_count: int) -> dict:
    backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()
    model = "hash-ngram" if backend == "hash" else os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return {"embedding_backend": backend, "embedding_model": model, "k": k, "questions": question_count}


def check_comparable(run: dict, recorded: dict) -> List[str]:
    """Reasons the current run cannot be compared with the recorded baseline."""
    return [
        f"baseline was recorded with {key}={recorded.get(key)!r} but this run has {run[key]!r}"
        for key in ("embedding_model", "k", "questions")
        if recorded.get(key) != run[key]
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--allow-hash", action="store_true", help="run on the offline hash embedder (tests only)")
    args = parser.parse_args(argv)

    import chromadb

    from agent.embeddings import get_embedding_function
    from evals.retrieval import build_collection, load_questions
    from seed_vectorstore import PAST_INCIDENTS

    if os.getenv("EMBEDDING_BACKEND", "").lower() == "hash" and not args.allow_hash:
        print("Refusing to run: EMBEDDING_BACKEND=hash only checks spelling, so its scores are not a quality "
              "measurement. Unset it to use the real embedding model.", file=sys.stderr)
        return 2

    questions = load_questions()
    with tempfile.TemporaryDirectory() as tmp:
        client = chromadb.PersistentClient(path=tmp)
        collection = build_collection(client, "gate_incidents", PAST_INCIDENTS, get_embedding_function())
        scores = compute_scores(collection, questions, args.k)
    run = describe_run(args.k, len(questions))

    if args.write_baseline:
        record = {**run, "scores": {m: {k_: round(v, 4) for k_, v in s.items()} for m, s in scores.items()}}
        args.baseline.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
        return 0

    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}. Record one with: python -m evals.gate --write-baseline", file=sys.stderr)
        return 2
    recorded = json.loads(args.baseline.read_text(encoding="utf-8"))
    mismatches = check_comparable(run, recorded)
    if mismatches:
        print("Cannot compare with the stored baseline:\n  " + "\n  ".join(mismatches)
              + "\nRe-record it with: python -m evals.gate --write-baseline", file=sys.stderr)
        return 2

    baseline = recorded["scores"]
    print(f"{run['embedding_model']}, {run['questions']} questions, k={args.k}, tolerance {args.tolerance}\n")
    print(format_report(scores, baseline, args.k))
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Retrieval eval vs baseline ({run['embedding_model']})\n\n"
                     + format_report(scores, baseline, args.k, markdown=True) + "\n")

    problems = compare(scores, baseline, args.tolerance)
    if problems:
        print("\nRetrieval regressed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print("\nOK: no metric is below its baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
