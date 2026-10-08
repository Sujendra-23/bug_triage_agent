from typing import TypedDict, Annotated, List, Optional
import operator


class AgentState(TypedDict):
    """
    The shared state passed between every node in the LangGraph.

    Fields are updated immutably — LangGraph merges each node's
    returned dict into this state before calling the next node.
    """

    # Input
    bug_report: str

    # ISO 639-1 code of the bug report's language (set by the planner). Retrieval stays in
    # English; the analysis and final report are written in this language.
    language: str

    # Planner output: sub-questions to guide retrieval
    diagnostic_questions: List[str]

    # Retriever output: past incidents fetched from ChromaDB
    retrieved_contexts: Annotated[List[str], operator.add]

    # Analyzer output: current hypothesis about root cause
    analysis: str

    # Validator output: whether the analysis is good enough
    is_sufficient: bool

    # How many retrieval+analysis loops we have done (circuit breaker)
    iterations: int

    # Final structured report (produced by reporter node)
    report: Optional[str]
