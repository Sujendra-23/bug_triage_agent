"""
remediation.py — turn a validated root-cause report into a verified fix.

Flow (see `remediate`):
  1. Create an isolated git worktree of the target repo on a fresh branch (the user's checkout is never touched).
  2. Run the failing test once to confirm it fails at baseline.
  3. Ask the LLM for a patch (search/replace edits), apply it, re-run the test. On failure, feed the test output
     back and try again, up to `max_attempts` times (REMEDIATION_MAX_ATTEMPTS, default 3).
  4. Only when the test passes: commit and open a *draft* pull request through the injected opener.
  5. Otherwise: escalate with the report and every failed attempt, and discard the branch.

The module has no LangChain/LangGraph imports so it can be tested with a stubbed LLM and a temporary git repo.
The LLM is any object with `.invoke([message])` returning something with `.content`.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

DEFAULT_MAX_ATTEMPTS = 3

# Files a patch may not touch: otherwise "make the failing test pass" could be satisfied by editing the test.
DEFAULT_PROTECTED_GLOBS = (
    "tests/*", "test/*", "*/tests/*", "*/test/*", "__tests__/*", "*/__tests__/*",
    "test_*.py", "*/test_*.py", "*_test.py", "*_test.go", "*.test.*", "*.spec.*", "conftest.py", "*/conftest.py",
    ".github/*",
)

STATUS_PR_OPENED = "pr_opened"
STATUS_PR_FAILED = "fix_verified_pr_failed"
STATUS_VERIFIED = "fix_verified"      # dry run: test passes on a local branch, nothing pushed
STATUS_ESCALATED = "escalated"


def default_max_attempts() -> int:
    try:
        return max(1, int(os.environ.get("REMEDIATION_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)))
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass
class RemediationConfig:
    repo_path: str
    test_command: List[str]                      # argv of the failing test, e.g. ["python", "-m", "pytest", "tests/test_x.py"]
    max_attempts: int = field(default_factory=default_max_attempts)
    base_ref: str = "HEAD"                       # commit/branch the fix branches from
    pr_base: Optional[str] = None                # branch the PR targets; default = branch HEAD was on
    setup_command: Optional[List[str]] = None    # run once in the worktree before the baseline (e.g. npm ci)
    test_timeout: int = 300
    protected_globs: Sequence[str] = DEFAULT_PROTECTED_GLOBS
    escalation_dir: Optional[str] = None         # if set, escalation reports are also written here as markdown

    @classmethod
    def from_dict(cls, d: dict) -> "RemediationConfig":
        d = dict(d)
        for key in ("test_command", "setup_command"):
            if isinstance(d.get(key), str):
                d[key] = shlex.split(d[key])
        d = {k: v for k, v in d.items() if v is not None}
        return cls(**d)


@dataclass
class Attempt:
    number: int
    explanation: str = ""
    edits: List[dict] = field(default_factory=list)
    diff: str = ""
    error: str = ""                  # why the patch could not be produced/applied (no test was run)
    test_exit_code: Optional[int] = None
    test_output: str = ""
    passed: bool = False


@dataclass
class PullRequest:
    repo_path: str
    worktree: str
    branch: str
    base: str
    title: str
    body: str
    draft: bool = True


@dataclass
class RemediationResult:
    status: str
    reason: str
    branch: Optional[str] = None
    pr_url: Optional[str] = None
    baseline_output: str = ""
    attempts: List[Attempt] = field(default_factory=list)
    escalation: Optional[str] = None  # markdown for a human; set whenever status != pr_opened
    escalation_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


PullRequestOpener = Callable[[PullRequest], str]


# ── git / process helpers ────────────────────────────────────────────────────

def _run(argv: Sequence[str], cwd: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _git(cwd: str, *args: str, check: bool = True) -> str:
    p = _run(["git", *args], cwd)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip() or p.stdout.strip()}")
    return p.stdout


def _tail(text: str, limit: int = 6000) -> str:
    return text if len(text) <= limit else "…(truncated)…\n" + text[-limit:]


def _run_test(argv: Sequence[str], cwd: str, timeout: int) -> tuple[int, str]:
    try:
        p = _run(argv, cwd, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"")
        out = out.decode(errors="replace") if isinstance(out, bytes) else out
        return 124, _tail(out) + f"\n[test timed out after {timeout}s]"
    except FileNotFoundError as e:
        return 127, f"[test command not found: {e}]"
    return p.returncode, _tail(p.stdout + p.stderr)


# ── patch proposal / application ─────────────────────────────────────────────

def _is_protected(rel: str, globs: Sequence[str], extra: Sequence[str]) -> bool:
    rel = rel.replace(os.sep, "/")
    return rel in extra or any(fnmatch.fnmatch(rel, g) for g in globs)


def _test_target_files(test_command: Sequence[str], worktree: str) -> List[str]:
    """Paths named on the test command line (e.g. tests/test_x.py::test_a) are off-limits to the patch."""
    found = []
    for arg in test_command:
        candidate = arg.split("::")[0]
        if candidate and not candidate.startswith("-") and (Path(worktree) / candidate).is_file():
            found.append(candidate.replace(os.sep, "/"))
    return found


def parse_edits(raw: str) -> tuple[str, List[dict]]:
    """Parse the LLM reply: {"explanation": "...", "edits": [{"path", "search", "replace"}]}."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"reply was not valid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("edits"), list) or not data["edits"]:
        raise ValueError('reply must be a JSON object with a non-empty "edits" list')
    edits = []
    for i, e in enumerate(data["edits"]):
        if not isinstance(e, dict) or not isinstance(e.get("path"), str) or not isinstance(e.get("replace"), str):
            raise ValueError(f'edit {i} needs string "path" and "replace" (and "search")')
        search = e.get("search", "")
        if not isinstance(search, str):
            raise ValueError(f'edit {i}: "search" must be a string')
        edits.append({"path": e["path"], "search": search, "replace": e["replace"]})
    return str(data.get("explanation", "")).strip(), edits


def apply_edits(worktree: str, edits: List[dict], protected_globs: Sequence[str], protected_files: Sequence[str]) -> None:
    """Validate every edit first, then write them all; raises ValueError (nothing written) on any problem.

    An edit replaces exactly one occurrence of `search` in `path`; an empty `search` creates a new file.
    """
    root = Path(worktree).resolve()
    plan: dict[Path, str] = {}
    for e in edits:
        target = (root / e["path"].replace("\\", "/")).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise ValueError(f"path escapes the repository: {e['path']}")
        rel_norm = target.relative_to(root).as_posix()
        if rel_norm == ".git" or rel_norm.startswith(".git/"):
            raise ValueError(f"refusing to edit {rel_norm}")
        if _is_protected(rel_norm, protected_globs, protected_files):
            raise ValueError(f"{rel_norm} is a protected test/CI file; fix the code under test instead")
        current = plan.get(target)
        if current is None:
            current = target.read_text() if target.is_file() else None
        if e["search"] == "":
            if current is not None:
                raise ValueError(f"{rel_norm} already exists; give a non-empty search to edit it")
            plan[target] = e["replace"]
            continue
        if current is None:
            raise ValueError(f"{rel_norm} does not exist")
        count = current.count(e["search"])
        if count != 1:
            raise ValueError(f"search text must match exactly once in {rel_norm} (matched {count} times)")
        plan[target] = current.replace(e["search"], e["replace"], 1)
    for target, content in plan.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def _files_from_output(output: str, worktree: str, limit: int = 4) -> List[str]:
    """Tracked, non-test source files that appear in a traceback or compiler error."""
    tracked = set(_git(worktree, "ls-files").split("\n"))
    seen: List[str] = []
    for m in re.finditer(r"([\w./\\-]+\.[A-Za-z0-9]+)", output):
        p = m.group(1)
        for cand in (p, os.path.relpath(p, worktree) if os.path.isabs(p) else p):
            cand = cand.replace("\\", "/")
            if cand in tracked and cand not in seen:
                seen.append(cand)
    return seen[:limit]


def _build_prompt(report: str, analysis: str, cfg: RemediationConfig, worktree: str, attempts: List[Attempt],
                  last_output: str, protected_files: Sequence[str]) -> str:
    snippets = []
    budget = 24000
    files = _files_from_output(last_output, worktree)
    for rel in files:
        if _is_protected(rel, cfg.protected_globs, protected_files):
            continue
        text = (Path(worktree) / rel).read_text(errors="replace")[:budget]
        budget -= len(text)
        snippets.append(f"### {rel}\n```\n{text}\n```")
        if budget <= 0:
            break
    tracked = _git(worktree, "ls-files").split("\n")[:200]
    history = ""
    if attempts:
        history = "\n\nPrevious attempts that did NOT make the test pass (do not repeat them):\n" + "\n".join(
            f"- attempt {a.number}: {a.error or 'applied, test still failing'}"
            + (f"\n  diff:\n{_indent(a.diff[:2000])}" if a.diff else "")
            for a in attempts
        )
    return f"""You are fixing a bug in a git repository. A root-cause analysis has already been validated.

Incident report:
{report}

Root-cause analysis:
{analysis}

Failing test command: {shlex.join(cfg.test_command)}
Latest output of that command:
{last_output}
{history}

Tracked files (first 200):
{chr(10).join(tracked)}

Relevant file contents:
{chr(10).join(snippets) or '(none found in the test output; infer from the file list)'}

Write the smallest patch that makes the failing test pass by changing the code under test. You may NOT edit
tests, conftest files or CI config. Respond ONLY with JSON:
{{"explanation": "one or two sentences", "edits": [{{"path": "relative/path", "search": "exact existing text, unique in the file", "replace": "new text"}}]}}
Use an empty "search" only to create a new file. Each "search" must match the file exactly once.
"""


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


# ── pull request ─────────────────────────────────────────────────────────────

class GhPullRequestOpener:
    """Pushes the branch and opens a *draft* PR with the GitHub CLI. Only `remediate` calls it, only after a pass."""

    def __call__(self, pr: PullRequest) -> str:
        _git(pr.repo_path, "push", "-u", "origin", pr.branch)
        argv = ["gh", "pr", "create", "--draft", "--base", pr.base, "--head", pr.branch,
                "--title", pr.title, "--body", pr.body]
        p = _run(argv, pr.repo_path)
        if p.returncode != 0:
            raise RuntimeError(f"gh pr create failed: {p.stderr.strip()}")
        return p.stdout.strip().splitlines()[-1]


def _pr_body(report: str, attempts: List[Attempt], test_command: Sequence[str]) -> str:
    winning = attempts[-1]
    return (
        "Automated fix proposed by the bug triage agent after a validated root-cause analysis. "
        "Opened as a draft: please review before merging.\n\n"
        f"**Verified by:** `{shlex.join(test_command)}` failed before the patch and passes after it "
        f"(attempt {winning.number} of {len(attempts)} used).\n\n"
        f"**What changed:** {winning.explanation or '(no explanation given)'}\n\n"
        f"<details><summary>Incident report</summary>\n\n{report}\n\n</details>\n"
    )


# ── escalation ───────────────────────────────────────────────────────────────

def format_escalation(report: str, reason: str, baseline_output: str, attempts: List[Attempt], cfg: RemediationConfig) -> str:
    lines = [
        "# Auto-remediation escalated to a human", "",
        f"**Reason:** {reason}", "",
        f"**Failing test:** `{shlex.join(cfg.test_command)}`  ",
        f"**Repository:** `{cfg.repo_path}`  ",
        f"**Attempts used:** {len(attempts)} of {cfg.max_attempts}", "",
        "## Incident report", "", report or "(none)", "",
    ]
    if baseline_output:
        lines += ["## Baseline test output", "", "```", baseline_output, "```", ""]
    for a in attempts:
        lines += [f"## Attempt {a.number}", ""]
        if a.explanation:
            lines += [f"Intent: {a.explanation}", ""]
        if a.error:
            lines += [f"Could not apply: {a.error}", ""]
        if a.diff:
            lines += ["```diff", a.diff, "```", ""]
        if a.test_exit_code is not None:
            lines += [f"Test exit code {a.test_exit_code}:", "", "```", a.test_output, "```", ""]
    return "\n".join(lines)


# ── orchestration ────────────────────────────────────────────────────────────

def remediate(report: str, analysis: str, llm, cfg: RemediationConfig, open_pr: Optional[PullRequestOpener]) -> RemediationResult:
    """Try to fix the failing test in an isolated worktree; open a draft PR only if it passes.

    `open_pr=None` verifies the fix and keeps the branch but never pushes (a dry run).
    """
    repo = str(Path(cfg.repo_path).resolve())
    _git(repo, "rev-parse", "--git-dir")  # fail fast if this is not a repository
    pr_base = cfg.pr_base or _git(repo, "rev-parse", "--abbrev-ref", cfg.base_ref).strip()
    base_sha = _git(repo, "rev-parse", cfg.base_ref).strip()
    branch = f"triage/fix-{uuid.uuid4().hex[:8]}"
    work_root = tempfile.mkdtemp(prefix="triage-worktree-")
    worktree = os.path.join(work_root, "wt")
    _git(repo, "worktree", "add", "-q", "-b", branch, worktree, base_sha)

    attempts: List[Attempt] = []
    baseline_output = ""
    keep_branch = False
    try:
        if cfg.setup_command:
            code, out = _run_test(cfg.setup_command, worktree, cfg.test_timeout)
            if code != 0:
                return _escalate(report, f"setup command failed (exit {code})", out, attempts, cfg)

        code, baseline_output = _run_test(cfg.test_command, worktree, cfg.test_timeout)
        if code == 0:
            return _escalate(report, "the test did not fail at baseline (already fixed or flaky), so there is nothing to verify a fix against",
                             baseline_output, attempts, cfg)

        protected_files = _test_target_files(cfg.test_command, worktree)
        last_output = baseline_output
        for n in range(1, cfg.max_attempts + 1):
            attempt = Attempt(number=n)
            attempts.append(attempt)
            _git(worktree, "reset", "--hard", "-q", base_sha)
            _git(worktree, "clean", "-fdxq")

            prompt = _build_prompt(report, analysis, cfg, worktree, attempts[:-1], last_output, protected_files)
            try:
                from langchain_core.messages import HumanMessage  # local import keeps the module LangChain-free for tests
                message = HumanMessage(content=prompt)
            except ImportError:  # pragma: no cover
                message = type("Msg", (), {"content": prompt})()
            try:
                attempt.explanation, attempt.edits = parse_edits(llm.invoke([message]).content)
                apply_edits(worktree, attempt.edits, cfg.protected_globs, protected_files)
            except Exception as e:  # bad JSON, bad path, search not unique, protected file, LLM error …
                attempt.error = str(e)
                continue

            _git(worktree, "add", "-A")
            attempt.diff = _git(worktree, "diff", "--cached", base_sha)
            _git(worktree, "reset", "-q")  # leave changes unstaged for the test run's benefit; diff already captured
            attempt.test_exit_code, attempt.test_output = _run_test(cfg.test_command, worktree, cfg.test_timeout)
            last_output = attempt.test_output
            if attempt.test_exit_code == 0:
                attempt.passed = True
                break

        if not (attempts and attempts[-1].passed):
            return _escalate(report, f"test still failing after {len(attempts)} attempt(s)", baseline_output, attempts, cfg)

        # The test passes: commit on the fix branch.
        _git(worktree, "add", "-A")
        identity = []
        if not _git(worktree, "config", "user.email", check=False).strip():
            identity = ["-c", "user.name=Bug Triage Agent", "-c", "user.email=bug-triage-agent@users.noreply.github.com"]
        title = f"Fix: {attempts[-1].explanation or 'failing test'}"[:100]
        _git(worktree, *identity, "commit", "-q", "-m", title,
             "-m", f"Automated fix verified by: {shlex.join(cfg.test_command)}")
        keep_branch = True

        if open_pr is None:
            return RemediationResult(STATUS_VERIFIED, "fix verified; PR not opened (dry run)", branch=branch,
                                     baseline_output=baseline_output, attempts=attempts)
        pr = PullRequest(repo_path=repo, worktree=worktree, branch=branch, base=pr_base, title=title,
                         body=_pr_body(report, attempts, cfg.test_command), draft=True)
        try:
            url = open_pr(pr)
        except Exception as e:
            result = RemediationResult(STATUS_PR_FAILED, f"fix verified but the pull request could not be opened: {e}",
                                       branch=branch, baseline_output=baseline_output, attempts=attempts)
            result.escalation = format_escalation(report, result.reason + f" The verified fix is on local branch {branch}.",
                                                  baseline_output, attempts, cfg)
            _write_escalation(result, cfg)
            return result
        return RemediationResult(STATUS_PR_OPENED, "test passes; draft PR opened", branch=branch, pr_url=url,
                                 baseline_output=baseline_output, attempts=attempts)
    finally:
        _run(["git", "worktree", "remove", "--force", worktree], repo)
        shutil.rmtree(work_root, ignore_errors=True)
        if not keep_branch:
            _run(["git", "branch", "-D", branch], repo)


def _escalate(report: str, reason: str, baseline_output: str, attempts: List[Attempt], cfg: RemediationConfig) -> RemediationResult:
    result = RemediationResult(STATUS_ESCALATED, reason, baseline_output=baseline_output, attempts=attempts)
    result.escalation = format_escalation(report, reason, baseline_output, attempts, cfg)
    _write_escalation(result, cfg)
    return result


def _write_escalation(result: RemediationResult, cfg: RemediationConfig) -> None:
    if cfg.escalation_dir and result.escalation:
        Path(cfg.escalation_dir).mkdir(parents=True, exist_ok=True)
        path = Path(cfg.escalation_dir) / f"escalation-{uuid.uuid4().hex[:8]}.md"
        path.write_text(result.escalation)
        result.escalation_path = str(path)
