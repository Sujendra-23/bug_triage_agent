"""
nodes.py — Each function here is a LangGraph node.

A node receives the full AgentState and returns a dict of fields to update.
LangGraph merges the returned dict into the state before calling the next node.
"""

import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from .language import detect_language, language_name
from .remediation import (STATUS_ESCALATED, GhPullRequestOpener, RemediationConfig, RemediationResult,
                          format_escalation, remediate)
from .state import AgentState
from .vectorstore import retrieve

# Shared LLM instance — all nodes use the same model
llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2048, temperature=0)


def _language_rule(state: AgentState) -> str:
    """Instruction appended to the analyzer and reporter prompts. Empty for English."""
    code = state.get("language") or "en"
    if code == "en":
        return ""
    return (
        f"\nWrite your entire response in {language_name(code)}. Keep code, log lines, "
        "identifiers and URLs exactly as they are."
    )


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: PLANNER
# Breaks the bug report into focused diagnostic sub-questions that will guide
# the RAG retrieval step.
# ─────────────────────────────────────────────────────────────────────────────

def planner(state: AgentState) -> dict:
    """
    Generate 3-5 targeted diagnostic questions from the bug report.
    These questions drive what we search for in the vector store.
    """
    print("\n[PLANNER] Generating diagnostic questions...")

    language = detect_language(state["bug_report"])
    print(f"[PLANNER] Detected language: {language_name(language)} ({language})")

    prompt = f"""You are a senior software engineer specializing in incident analysis.

Given this bug report, generate exactly 4 focused diagnostic questions that will help 
identify the root cause. Each question should target a specific aspect of the failure.
Write the questions in English even if the bug report is in another language, because
the incident database and documentation are in English.

Bug Report:
{state["bug_report"]}

Respond ONLY with a JSON array of strings. Example:
["question 1", "question 2", "question 3", "question 4"]
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    questions = json.loads(response.content.strip())

    print(f"[PLANNER] Generated {len(questions)} diagnostic questions.")
    for i, q in enumerate(questions, 1):
        print(f"  Q{i}: {q}")

    return {
        "language": language,
        "diagnostic_questions": questions,
        "iterations": 0,
        "retrieved_contexts": [],
        "is_sufficient": False,
        "validated": False,
        "report": None,
        "remediation": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: RETRIEVER
# Queries ChromaDB with the diagnostic questions to fetch relevant past incidents.
# This is the RAG retrieval step.
# ─────────────────────────────────────────────────────────────────────────────

def retriever(state: AgentState) -> dict:
    """
    Use diagnostic questions to retrieve relevant past incidents from ChromaDB.
    On retry loops we broaden the query by also including the current analysis.
    """
    print(f"\n[RETRIEVER] Fetching context (iteration {state['iterations'] + 1})...")

    queries = state["diagnostic_questions"]

    # On retry iterations, add the current analysis as an extra query
    # so we search from a different angle
    if state["iterations"] > 0 and state.get("analysis"):
        queries = queries + [state["analysis"]]

    contexts = retrieve(queries, n_results=3)
    print(f"[RETRIEVER] Retrieved {len(contexts)} relevant past incidents.")

    return {
        "retrieved_contexts": contexts,   # operator.add appends to existing list
        "iterations": state["iterations"] + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: ANALYZER
# Reasons over the retrieved context to form a root-cause hypothesis.
# ─────────────────────────────────────────────────────────────────────────────

def analyzer(state: AgentState) -> dict:
    """
    Synthesize retrieved incidents with the bug report to produce a
    root-cause hypothesis and list of likely causes.
    """
    print("\n[ANALYZER] Performing root-cause analysis...")

    context_block = "\n\n---\n\n".join(state["retrieved_contexts"])

    prompt = f"""You are a principal engineer performing root-cause analysis.

Bug Report:
{state["bug_report"]}

Relevant Past Incidents (retrieved from incident database):
{context_block}

Based on the bug report and the past incidents above, provide:
1. The most likely root cause
2. Supporting evidence from the retrieved incidents
3. Two or three alternative hypotheses if the primary cause is uncertain
4. Immediate mitigation steps

Be specific and technical. Reference the past incidents directly where relevant.
Where a retrieved item starts with "[source: URL]", cite that URL.
{_language_rule(state)}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content.strip()

    print("[ANALYZER] Analysis complete.")
    return {"analysis": analysis}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: VALIDATOR
# Self-checks whether the analysis is confident and complete enough.
# Returns is_sufficient=True to proceed to the reporter, or False to retry.
# ─────────────────────────────────────────────────────────────────────────────

def validator(state: AgentState) -> dict:
    """
    Evaluate the quality of the current analysis.
    If it is vague or missing key information, trigger another retrieval loop.
    Cap at 3 iterations to avoid infinite loops.
    """
    print("\n[VALIDATOR] Checking analysis quality...")

    # Hard stop — never loop more than 3 times
    if state["iterations"] >= 3:
        print("[VALIDATOR] Max iterations reached. Proceeding to report.")
        return {"is_sufficient": True, "validated": False}

    prompt = f"""You are a quality reviewer for incident analyses.

Bug Report:
{state["bug_report"]}

Current Analysis:
{state["analysis"]}

Does this analysis:
1. Identify a specific root cause (not just "could be many things")?
2. Provide concrete evidence or references?
3. Give actionable mitigation steps?

Respond with ONLY a JSON object: {{"sufficient": true}} or {{"sufficient": false, "reason": "what is missing"}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    malformed = False
    try:
        result = json.loads(response.content.strip())
        is_sufficient = result.get("sufficient", False)
        reason = result.get("reason", "")
    except json.JSONDecodeError:
        # If the model doesn't return valid JSON, treat as sufficient to avoid loops
        is_sufficient = True
        reason = ""
        malformed = True

    if is_sufficient:
        print("[VALIDATOR] Analysis is sufficient. Proceeding to report generation.")
    else:
        print(f"[VALIDATOR] Analysis needs improvement: {reason}. Retrying retrieval...")

    return {"is_sufficient": is_sufficient, "validated": is_sufficient and not malformed}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5: REPORTER
# Produces the final structured incident report.
# ─────────────────────────────────────────────────────────────────────────────

def reporter(state: AgentState) -> dict:
    """
    Generate a structured Markdown incident report from the finalized analysis.
    """
    print("\n[REPORTER] Generating final incident report...")

    prompt = f"""You are a technical writer creating a formal incident report.

Bug Report:
{state["bug_report"]}

Root-Cause Analysis:
{state["analysis"]}

Generate a structured incident report in Markdown with these exact sections
(keep the "##" markers, translate the headings and the content):

## Incident Summary
(One paragraph overview)

## Root Cause
(The identified root cause with confidence level: High / Medium / Low)

## Evidence
(Bullet list of supporting evidence)

## Impact
(What is affected and how severely)

## Immediate Mitigation
(Step-by-step actions to stop the bleeding right now)

## Permanent Fix
(What needs to be changed in the codebase/infrastructure long term)

## Prevention
(How to prevent this class of issue in the future — monitoring, testing, process)

## Timeline
(Estimated time to implement each fix)

{_language_rule(state)}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content.strip()

    print("[REPORTER] Report generated.")
    return {"report": report}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6: REMEDIATOR (optional)
# Runs only when the caller supplied a remediation_request. Writes a patch in an isolated
# git worktree, re-runs the failing test, and opens a draft PR only if it passes.
# ─────────────────────────────────────────────────────────────────────────────

# Swapped out in tests; the real opener pushes the branch and runs `gh pr create --draft`.
pull_request_opener = GhPullRequestOpener()


def remediator(state: AgentState) -> dict:
    print("\n[REMEDIATOR] Attempting auto-remediation...")
    cfg = RemediationConfig.from_dict(state["remediation_request"])

    if not state.get("validated"):
        # The iteration cap forced the report through; do not patch code on a root cause nobody accepted.
        reason = "the root-cause analysis was never validated (iteration cap reached), so no patch was attempted"
        result = RemediationResult(STATUS_ESCALATED, reason)
        result.escalation = format_escalation(state.get("report") or "", reason, "", [], cfg)
        print(f"[REMEDIATOR] {reason}")
        return {"remediation": result.to_dict()}

    result = remediate(state.get("report") or "", state.get("analysis") or "", llm, cfg, pull_request_opener)
    print(f"[REMEDIATOR] {result.status}: {result.reason}")
    return {"remediation": result.to_dict()}
