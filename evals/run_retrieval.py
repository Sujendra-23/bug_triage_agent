"""python -m evals.run_retrieval [--k 5]  — print recall@k for vector, bm25 and hybrid retrieval."""

import argparse
import tempfile

import chromadb
from dotenv import load_dotenv

load_dotenv()

from agent.embeddings import get_embedding_function
from evals.retrieval import build_collection, evaluate, format_summary, format_table, load_questions
from seed_vectorstore import PAST_INCIDENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        client = chromadb.PersistentClient(path=tmp)
        collection = build_collection(client, "eval_incidents", PAST_INCIDENTS, get_embedding_function())
        questions = load_questions()
        print(f"{len(questions)} questions, {collection.count()} chunks")
        print(format_summary(collection, questions, args.k))
        print()
        print(format_table(evaluate(collection, questions, args.k), args.k))


if __name__ == "__main__":
    main()
