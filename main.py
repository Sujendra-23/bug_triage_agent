"""
main.py — CLI entry point for the Bug Triage Agent.

Usage:
    python main.py
    python main.py --report "API timeouts after deploying v2.3.1"
"""

import argparse
from dotenv import load_dotenv

load_dotenv()

from agent import agent
from sample_bug_reports import SAMPLE_1 as SAMPLE_BUG_REPORT


def run(bug_report: str):
    print("=" * 60)
    print("BUG TRIAGE AGENT")
    print("=" * 60)
    print(f"\nBug Report:\n{bug_report.strip()}\n")
    print("=" * 60)

    initial_state = {
        "bug_report": bug_report.strip(),
        "diagnostic_questions": [],
        "retrieved_contexts": [],
        "analysis": "",
        "is_sufficient": False,
        "iterations": 0,
        "report": None,
    }

    final_state = agent.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL INCIDENT REPORT")
    print("=" * 60)
    print(final_state["report"])

    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bug Triage Agent")
    parser.add_argument(
        "--report",
        type=str,
        default=SAMPLE_BUG_REPORT,
        help="Bug report text to analyze",
    )
    args = parser.parse_args()
    run(args.report)
