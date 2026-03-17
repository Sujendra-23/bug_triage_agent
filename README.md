# Bug Triage Agent

An intelligent bug triage and root-cause analysis agent built with **LangGraph** and **RAG** (ChromaDB vector store).

Given a bug report, the agent:
1. **Plans** — breaks the bug into diagnostic sub-questions
2. **Retrieves** — searches a vector store of past incidents using RAG
3. **Analyzes** — reasons over retrieved context to form root-cause hypotheses
4. **Validates** — self-checks if the analysis is sufficient, retries retrieval if not
5. **Reports** — generates a structured incident report with RCA and fix recommendations

## Architecture

```
bug_report
    │
    ▼
[planner] ──► [retriever] ──► [analyzer] ──► [validator]
                   ▲                               │
                   │          insufficient         │
                   └───────────────────────────────┘
                                                   │ sufficient
                                                   ▼
                                              [reporter]
                                                   │
                                                   ▼
                                           structured_report
```

## Tech Stack

- **LangGraph** — agent orchestration, stateful graph, conditional routing
- **LangChain + Claude (claude-sonnet-4-6)** — LLM reasoning at each node
- **ChromaDB** — local vector store for RAG (past incidents)
- **FastAPI** — REST API to serve the agent
- **Sentence Transformers** — embeddings for vector search

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your Anthropic API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here

# 3. Seed the vector store with sample past incidents
python seed_vectorstore.py

# 4. Run the agent directly
python main.py

# 5. Or run the FastAPI server
uvicorn api:app --reload
# Then POST to http://localhost:8000/triage
```

## Example

```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"bug_report": "API response time increased by 300% after deploying v2.3.1. Redis connection timeouts appearing in logs."}'
```

## Project Structure

```
bug_triage_agent/
├── agent/
│   ├── state.py          # TypedDict state definition
│   ├── nodes.py          # All LangGraph node functions
│   ├── graph.py          # StateGraph definition and compilation
│   └── vectorstore.py    # ChromaDB RAG retrieval
├── seed_vectorstore.py   # Seeds ChromaDB with sample past incidents
├── main.py               # CLI entry point
├── api.py                # FastAPI server
├── requirements.txt
└── .env.example
```
