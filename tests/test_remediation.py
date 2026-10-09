"""Auto-remediation with a stubbed LLM, a throwaway git repo and a stubbed PR opener.

Nothing here talks to a network, pushes, or opens a real pull request.
"""

import json
import subprocess
import sys

import pytest

import agent.nodes as nodes
from agent import remediation as rem
from agent.graph import after_report
from agent.remediation import PullRequest, RemediationConfig, remediate

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
REPORT = "## Root Cause\n`add` subtracts instead of adding."


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture()
def repo(tmp_path):
    path = tmp_path / "target"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "t")
    (path / "calc.py").write_text(BUGGY)
    (path / "test_calc.py").write_text(TEST)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "buggy")
    return path


def cfg_for(repo, **kw):
    return RemediationConfig(repo_path=str(repo), test_command=[sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_calc.py"], **kw)


class ScriptedLLM:
    """Returns the queued replies in order; records every prompt."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)

        class R:
            pass

        r = R()
        r.content = self.replies.pop(0)
        return r


def edit(path="calc.py", search="return a - b", replace="return a + b", why="use addition"):
    return json.dumps({"explanation": why, "edits": [{"path": path, "search": search, "replace": replace}]})


WRONG = edit(search="return a - b", replace="return a * b", why="try multiplication")


class FakeOpener:
    def __init__(self, fail=False):
        self.calls: list[PullRequest] = []
        self.fail = fail

    def __call__(self, pr):
        self.calls.append(pr)
        if self.fail:
            raise RuntimeError("gh: not authenticated")
        return "https://github.com/example/target/pull/7"


def branches(repo):
    return git(repo, "branch", "--list", "triage/*").split()


# ── happy paths ──────────────────────────────────────────────────────────────

def test_passing_patch_opens_exactly_one_draft_pr_and_keeps_the_fix_branch(repo):
    opener, llm = FakeOpener(), ScriptedLLM(edit())
    result = remediate(REPORT, "analysis", llm, cfg_for(repo), opener)

    assert result.status == rem.STATUS_PR_OPENED and result.pr_url.endswith("/pull/7")
    assert len(result.attempts) == 1 and result.attempts[0].passed
    assert len(opener.calls) == 1
    pr = opener.calls[0]
    assert pr.draft is True and pr.base == "main" and pr.branch == result.branch
    assert "add" in pr.body and "Root Cause" in pr.body
    # The fix is committed on the branch, not in the user's checkout.
    assert git(repo, "show", f"{result.branch}:calc.py") == FIXED
    assert (repo / "calc.py").read_text() == BUGGY
    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "worktree", "list").count("\n") == 1  # temp worktree removed


def test_retries_with_the_previous_failure_in_the_prompt_until_it_passes(repo):
    opener, llm = FakeOpener(), ScriptedLLM(WRONG, WRONG, edit())
    result = remediate(REPORT, "analysis", llm, cfg_for(repo), opener)

    assert result.status == rem.STATUS_PR_OPENED
    assert [a.passed for a in result.attempts] == [False, False, True]
    assert result.attempts[0].test_exit_code == 1 and "assert" in result.attempts[0].test_output
    assert "return a * b" in llm.prompts[1]            # attempt 2 saw attempt 1's diff…
    assert "do not repeat" in llm.prompts[1].lower()   # …and was told not to repeat it
    assert len(opener.calls) == 1


# ── escalation ───────────────────────────────────────────────────────────────

def test_never_passing_escalates_after_default_three_attempts_and_opens_no_pr(repo, tmp_path):
    opener, llm = FakeOpener(), ScriptedLLM(WRONG, WRONG, WRONG)
    result = remediate(REPORT, "analysis", llm, cfg_for(repo, escalation_dir=str(tmp_path / "esc")), opener)

    assert result.status == rem.STATUS_ESCALATED and opener.calls == []
    assert len(result.attempts) == 3 and not any(a.passed for a in result.attempts)
    assert "Root Cause" in result.escalation and "Attempt 3" in result.escalation and "return a * b" in result.escalation
    assert result.escalation_path and open(result.escalation_path).read() == result.escalation
    # Cleaned up: no leftover branch, worktree or edits.
    assert branches(repo) == [] and (repo / "calc.py").read_text() == BUGGY
    assert git(repo, "worktree", "list").count("\n") == 1


def test_max_attempts_is_configurable(repo):
    llm = ScriptedLLM(WRONG, WRONG)
    result = remediate(REPORT, "a", llm, cfg_for(repo, max_attempts=2), FakeOpener())
    assert result.status == rem.STATUS_ESCALATED and len(result.attempts) == 2 and llm.replies == []


def test_max_attempts_default_comes_from_the_environment(repo, monkeypatch):
    monkeypatch.setenv("REMEDIATION_MAX_ATTEMPTS", "1")
    assert cfg_for(repo).max_attempts == 1
    monkeypatch.setenv("REMEDIATION_MAX_ATTEMPTS", "nonsense")
    assert cfg_for(repo).max_attempts == 3
    monkeypatch.delenv("REMEDIATION_MAX_ATTEMPTS")
    assert cfg_for(repo).max_attempts == 3


@pytest.mark.parametrize("reply,needle", [
    ("this is not json", "not valid JSON"),
    (json.dumps({"edits": []}), "non-empty"),
    (edit(search="no such text"), "matched 0 times"),
    (edit(path="../outside.py", search="", replace="x"), "escapes the repository"),
    (edit(path=".git/config", search="[core]", replace="[x]"), "refusing to edit"),
])
def test_unusable_patches_are_failed_attempts_not_crashes(repo, reply, needle):
    result = remediate(REPORT, "a", ScriptedLLM(reply), cfg_for(repo, max_attempts=1), FakeOpener())
    assert result.status == rem.STATUS_ESCALATED
    assert needle in result.attempts[0].error and result.attempts[0].test_exit_code is None


def test_patch_may_not_edit_the_failing_test(repo):
    cheat = edit(path="test_calc.py", search="== 5", replace="== -1", why="change the expectation")
    opener = FakeOpener()
    result = remediate(REPORT, "a", ScriptedLLM(cheat), cfg_for(repo, max_attempts=1), opener)
    assert result.status == rem.STATUS_ESCALATED and opener.calls == []
    assert "protected" in result.attempts[0].error
    assert (repo / "test_calc.py").read_text() == TEST


def test_a_test_that_already_passes_is_escalated_without_asking_the_llm(repo):
    (repo / "calc.py").write_text(FIXED)
    git(repo, "commit", "-qam", "already fixed")
    llm, opener = ScriptedLLM(), FakeOpener()
    result = remediate(REPORT, "a", llm, cfg_for(repo), opener)
    assert result.status == rem.STATUS_ESCALATED and "did not fail at baseline" in result.reason
    assert result.attempts == [] and llm.prompts == [] and opener.calls == []


def test_a_failing_test_command_timeout_counts_as_a_failed_attempt(repo):
    slow = RemediationConfig(repo_path=str(repo), test_command=[sys.executable, "-c", "import time; time.sleep(30)"],
                             test_timeout=1, max_attempts=1)
    result = remediate(REPORT, "a", ScriptedLLM(), slow, FakeOpener())
    assert result.status == rem.STATUS_ESCALATED and "timed out" in result.baseline_output


# ── PR handling ──────────────────────────────────────────────────────────────

def test_pr_failure_keeps_the_verified_branch_and_reports_it(repo):
    opener = FakeOpener(fail=True)
    result = remediate(REPORT, "a", ScriptedLLM(edit()), cfg_for(repo), opener)
    assert result.status == rem.STATUS_PR_FAILED and "not authenticated" in result.reason
    assert result.branch in branches(repo) and result.escalation and result.branch in result.escalation


def test_dry_run_verifies_but_never_calls_an_opener(repo):
    result = remediate(REPORT, "a", ScriptedLLM(edit()), cfg_for(repo), None)
    assert result.status == rem.STATUS_VERIFIED and result.pr_url is None
    assert git(repo, "show", f"{result.branch}:calc.py") == FIXED


# ── graph wiring ─────────────────────────────────────────────────────────────

def test_graph_ends_at_the_reporter_unless_remediation_was_requested():
    assert after_report({"remediation_request": None}) == "__end__"
    assert after_report({}) == "__end__"
    assert after_report({"remediation_request": {"repo_path": "x", "test_command": "pytest"}}) == "remediator"


def test_remediator_node_runs_the_flow_when_the_analysis_was_validated(repo, monkeypatch):
    opener = FakeOpener()
    monkeypatch.setattr(nodes, "llm", ScriptedLLM(edit()))
    monkeypatch.setattr(nodes, "pull_request_opener", opener)
    request = {"repo_path": str(repo), "test_command": f"{sys.executable} -m pytest -q -p no:cacheprovider test_calc.py"}
    out = nodes.remediator({"validated": True, "report": REPORT, "analysis": "a", "remediation_request": request})
    assert out["remediation"]["status"] == rem.STATUS_PR_OPENED and len(opener.calls) == 1


def test_remediator_node_refuses_to_patch_an_unvalidated_analysis(repo, monkeypatch):
    opener, llm = FakeOpener(), ScriptedLLM()
    monkeypatch.setattr(nodes, "llm", llm)
    monkeypatch.setattr(nodes, "pull_request_opener", opener)
    request = {"repo_path": str(repo), "test_command": "pytest"}
    out = nodes.remediator({"validated": False, "report": REPORT, "analysis": "a", "remediation_request": request})
    assert out["remediation"]["status"] == rem.STATUS_ESCALATED and "never validated" in out["remediation"]["reason"]
    assert llm.prompts == [] and opener.calls == [] and "Root Cause" in out["remediation"]["escalation"]


def test_validator_marks_forced_and_malformed_passes_as_not_validated(monkeypatch):
    class Reply:
        def __init__(self, text):
            self.text = text

        def invoke(self, _):
            class R:
                content = self.text

            return R()

    base = {"bug_report": "x", "analysis": "a", "iterations": 1}
    monkeypatch.setattr(nodes, "llm", Reply('{"sufficient": true}'))
    assert nodes.validator(base)["validated"] is True
    monkeypatch.setattr(nodes, "llm", Reply("not json"))
    assert nodes.validator(base) == {"is_sufficient": True, "validated": False}
    assert nodes.validator({**base, "iterations": 3}) == {"is_sufficient": True, "validated": False}


def test_full_graph_with_remediation_request_opens_a_pr_through_the_stub(repo, tmp_path, monkeypatch):
    import agent.vectorstore as vs
    from agent.graph import build_graph
    from agent.vectorstore import _get_collection

    monkeypatch.setattr(vs, "CHROMA_PERSIST_DIR", str(tmp_path / "db"))
    _get_collection("past_incidents").upsert(ids=["INC-1"], documents=["Arithmetic helper returned the wrong sign"])

    class Graph(ScriptedLLM):
        def invoke(self, messages):
            prompt = messages[0].content
            if "diagnostic questions" in prompt:
                self.replies = [json.dumps(["wrong sign", "add function", "regression", "test failure"])]
            elif "quality reviewer" in prompt:
                self.replies = [json.dumps({"sufficient": True})]
            elif "You are fixing a bug" in prompt:
                self.replies = [edit()]
            else:
                self.replies = ["analysis or report text"]
            return super().invoke(messages)

    opener = FakeOpener()
    monkeypatch.setattr(nodes, "llm", Graph())
    monkeypatch.setattr(nodes, "pull_request_opener", opener)
    final = build_graph().invoke({
        "bug_report": "add(2, 3) returns -1", "language": "en", "diagnostic_questions": [], "retrieved_contexts": [],
        "analysis": "", "is_sufficient": False, "validated": False, "iterations": 0, "report": None,
        "remediation_request": {"repo_path": str(repo), "test_command": f"{sys.executable} -m pytest -q -p no:cacheprovider test_calc.py"},
        "remediation": None,
    })
    assert final["remediation"]["status"] == rem.STATUS_PR_OPENED and len(opener.calls) == 1


def test_gh_opener_pushes_to_origin_and_creates_a_draft_pr(repo, tmp_path, monkeypatch):
    """The real GhPullRequestOpener against a local bare 'origin' and a fake `gh` that records its arguments."""
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-q", "origin", "main")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh-args.txt"
    gh = bin_dir / "gh"
    gh.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{log}"\necho https://github.com/example/target/pull/9\n')
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")

    result = remediate(REPORT, "a", ScriptedLLM(edit()), cfg_for(repo), rem.GhPullRequestOpener())

    assert result.status == rem.STATUS_PR_OPENED and result.pr_url.endswith("/pull/9")
    assert result.branch in git(origin, "branch", "--list", "triage/*")  # pushed
    args = log.read_text().split("\n")
    assert args[:3] == ["pr", "create", "--draft"] and "--base" in args and "main" in args
