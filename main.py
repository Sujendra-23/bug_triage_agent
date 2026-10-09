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


def run(bug_report: str, remediation_request: dict | None = None):
    print("=" * 60)
    print("BUG TRIAGE AGENT")
    print("=" * 60)
    print(f"\nBug Report:\n{bug_report.strip()}\n")
    print("=" * 60)

    initial_state = {
        "bug_report": bug_report.strip(),
        "language": "en",
        "diagnostic_questions": [],
        "retrieved_contexts": [],
        "analysis": "",
        "is_sufficient": False,
        "validated": False,
        "iterations": 0,
        "report": None,
        "remediation_request": remediation_request,
        "remediation": None,
    }

    final_state = agent.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL INCIDENT REPORT")
    print("=" * 60)
    print(final_state["report"])

    remediation = final_state.get("remediation")
    if remediation:
        print("\n" + "=" * 60)
        print(f"AUTO-REMEDIATION: {remediation['status']}")
        print("=" * 60)
        print(remediation["reason"])
        if remediation.get("pr_url"):
            print(f"Draft PR: {remediation['pr_url']}")
        if remediation.get("escalation"):
            print("\n" + remediation["escalation"])

    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bug Triage Agent")
    parser.add_argument(
        "--report",
        type=str,
        default=SAMPLE_BUG_REPORT,
        help="Bug report text to analyze",
    )
    fix = parser.add_argument_group(
        "auto-remediation",
        "After a validated report, patch the code in an isolated git worktree, re-run the failing test, "
        "and open a draft PR only if it passes (otherwise escalate with the report and the failed attempts).",
    )
    fix.add_argument("--repo", help="path to the git repository to fix (enables auto-remediation)")
    fix.add_argument("--test-cmd", help='failing test command, e.g. "python -m pytest tests/test_x.py"')
    fix.add_argument("--max-attempts", type=int, help="patch attempts before escalating (default: $REMEDIATION_MAX_ATTEMPTS or 3)")
    fix.add_argument("--base", default="HEAD", help="ref the fix branches from (default HEAD)")
    fix.add_argument("--pr-base", help="branch the PR targets (default: the branch --base is on)")
    fix.add_argument("--setup-cmd", help="command to run once in the worktree first, e.g. 'npm ci'")
    fix.add_argument("--escalation-dir", help="also write escalation reports here as markdown")
    args = parser.parse_args()

    request = None
    if args.repo:
        if not args.test_cmd:
            parser.error("--repo requires --test-cmd")
        request = {
            "repo_path": args.repo, "test_command": args.test_cmd, "base_ref": args.base, "pr_base": args.pr_base,
            "setup_command": args.setup_cmd, "max_attempts": args.max_attempts, "escalation_dir": args.escalation_dir,
        }
    run(args.report, request)
