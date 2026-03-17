"""
nodes.py — Each function here is a LangGraph node.

A node receives the full AgentState and returns a dict of fields to update.
LangGraph merges the returned dict into the state before calling the next node.
"""

import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState
from .vectorstore import retrieve

# Shared LLM instance — all nodes use the same model
llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2048, temperature=0)


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

    prompt = f"""You are a senior software engineer specializing in incident analysis.

Given this bug report, generate exactly 4 focused diagnostic questions that will help 
identify the root cause. Each question should target a specific aspect of the failure.

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
        "diagnostic_questions": questions,
        "iterations": 0,
        "retrieved_contexts": [],
        "is_sufficient": False,
        "report": None,
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
"""

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
        return {"is_sufficient": True}

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

    try:
        result = json.loads(response.content.strip())
        is_sufficient = result.get("sufficient", False)
        reason = result.get("reason", "")
    except json.JSONDecodeError:
        # If the model doesn't return valid JSON, treat as sufficient to avoid loops
        is_sufficient = True
        reason = ""

    if is_sufficient:
        print("[VALIDATOR] Analysis is sufficient. Proceeding to report generation.")
    else:
        print(f"[VALIDATOR] Analysis needs improvement: {reason}. Retrying retrieval...")

    return {"is_sufficient": is_sufficient}


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

Generate a structured incident report in Markdown with these exact sections:

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
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content.strip()

    print("[REPORTER] Report generated.")
    return {"report": report}
