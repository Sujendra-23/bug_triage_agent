"""
graph.py — Defines and compiles the LangGraph StateGraph.

This is the heart of the agent. It wires together all the nodes
and defines the conditional routing logic.
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import planner, retriever, analyzer, validator, reporter


def should_retry(state: AgentState) -> str:
    """
    Conditional edge function — called after the validator node.

    Returns:
        "retriever"  — if the analysis needs improvement (retry loop)
        "reporter"   — if the analysis is sufficient (proceed to final report)
    """
    if state["is_sufficient"]:
        return "reporter"
    return "retriever"


def build_graph() -> StateGraph:
    """
    Construct and compile the bug triage agent graph.

    Graph topology:
        planner → retriever → analyzer → validator ──(sufficient)──► reporter → END
                      ▲                       │
                      └──────(insufficient)───┘
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner",   planner)
    graph.add_node("retriever", retriever)
    graph.add_node("analyzer",  analyzer)
    graph.add_node("validator", validator)
    graph.add_node("reporter",  reporter)

    # Entry point
    graph.set_entry_point("planner")

    # Fixed edges
    graph.add_edge("planner",   "retriever")
    graph.add_edge("retriever", "analyzer")
    graph.add_edge("analyzer",  "validator")
    graph.add_edge("reporter",  END)

    # Conditional edge — the retry loop
    graph.add_conditional_edges(
        "validator",
        should_retry,
        {
            "retriever": "retriever",   # loop back for more context
            "reporter":  "reporter",    # proceed to final report
        },
    )

    return graph.compile()


# Singleton — import this directly in main.py and api.py
agent = build_graph()
