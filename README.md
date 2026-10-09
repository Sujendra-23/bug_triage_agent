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
- **rank-bm25** — lexical ranking for hybrid retrieval
- **BeautifulSoup** — HTML cleaning for the crawler

## Grounding in a crawled docs site, hybrid retrieval, and other languages

Besides the seeded past incidents, the agent can retrieve from a public docs site it crawls itself.

**Crawler** (`crawl_site.py`, `agent/crawler.py`): reads a sitemap (or sitemap index), fetches same-host pages
(robots.txt honoured, polite delay), strips navigation and boilerplate, splits the text into overlapping chunks and
embeds them into the `crawled_docs` ChromaDB collection. Re-runs are incremental:

| Situation | What happens |
|---|---|
| Page text unchanged (content hash) | skipped, nothing re-embedded |
| Page text changed | old chunks replaced, new ones embedded |
| Page left the sitemap, or returns 404/410 | its chunks are deleted |
| Page fails with a 5xx or network error | old copy kept, failure reported |
| Sitemap itself cannot be fetched | nothing is deleted |

```bash
python crawl_site.py --sitemap https://docs.example.com/sitemap.xml                  # once
python crawl_site.py --sitemap https://docs.example.com/sitemap.xml --every-hours 24  # nightly loop
# or cron:  0 2 * * *  cd /path/to/bug_triage_agent && .venv/bin/python crawl_site.py --sitemap <url>
```

**Hybrid retrieval** (`agent/hybrid.py`, `agent/vectorstore.py`): each diagnostic question is ranked by embedding
similarity and by BM25, and the two rankings are merged with reciprocal-rank fusion. Vector search finds paraphrases,
BM25 catches exact tokens such as `PAYMENT_GATEWAY_URL` or `DEADLINE_EXCEEDED`. Set `RETRIEVAL_MODE=vector|bm25|hybrid`
(default `hybrid`). Crawled chunks reach the analyzer prefixed with `[source: URL]`, and the analyzer is told to cite it.

**Other languages** (`agent/language.py`): the planner detects the bug report's language, always writes its diagnostic
questions in English (the incidents and docs are English), and the analyzer and reporter write in the report's
language. `POST /triage` returns the detected `language`. To embed non-English text directly, set
`EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2` and re-seed.

**Retrieval eval** (`evals/`): 22 questions (exact-token, paraphrase and 4 non-English) over the seeded incidents split
into 60 section chunks. Non-English questions are scored on the English query the planner would produce.

```bash
python -m evals.run_retrieval                      # real embeddings (sentence-transformers)
EMBEDDING_BACKEND=hash python -m evals.run_retrieval   # offline stand-in
```

`EMBEDDING_BACKEND=hash` matches spelling, not meaning, and exists so the tests and CI run with no model download.
Numbers from it check the harness and the fusion logic; they are not a quality measurement. With it, on this small
corpus, BM25 alone scored best (recall@5 1.00), hybrid 0.91 and vector-only 0.86. Run the command without the
override to measure your real embedding model.

## Auto-remediation (optional)

After a report the validator accepted, the agent can try to fix the bug itself. It only runs when you pass a repo and
a failing test; without them the graph ends at the reporter exactly as before.

```
reporter ─► remediator:  worktree on a new branch ─► run failing test (must fail) ─► LLM proposes edits ─► re-run test
                              ▲                                                            │ still failing, attempts left
                              └────────────────────────────────────────────────────────────┘
            test passes  ─► commit + open a DRAFT pull request
            attempts used up ─► escalate: report + every failed attempt (diff and test output), branch discarded
```

```bash
python main.py --report "add(2, 3) returns -1" \
  --repo ../my-service --test-cmd "python -m pytest tests/test_math.py" --max-attempts 3
```

- **Isolated**: all edits happen in a temporary `git worktree` of the target repo; your checkout is never modified.
- **Bounded**: `--max-attempts` / `REMEDIATION_MAX_ATTEMPTS` (default 3). Each retry sees the previous diffs and the latest
  test output.
- **PR only on green**: a draft PR is opened (branch pushed, `gh pr create --draft`) only after the test exits 0. A
  test that already passes at baseline is escalated instead, since there is nothing to verify a fix against.
- **Cannot cheat**: edits to test files, `conftest.py`, the test file named on the command line, and `.github/` are rejected.
  Edits must match the existing text exactly once; anything else counts as a failed attempt.
- **Needs a validated report**: if the validator only passed because the iteration cap forced it, no patch is attempted.
- **Escalation** (`RemediationResult.escalation`, printed by the CLI, written to `--escalation-dir` if given) contains the
  incident report, the baseline output and each attempt.
- The target repo's dependencies must be available to the test command. A fresh worktree has no `node_modules`/venv, so
  use `--setup-cmd` (for example `npm ci`) when the repo needs one.
- Remediation is deliberately not exposed through `POST /triage`: it runs a caller-supplied command.

Not exercised against real GitHub: the tests use a throwaway repo, a scripted LLM and a stubbed PR opener (plus the real
`GhPullRequestOpener` against a local bare remote with a fake `gh`).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q          # offline: hash embeddings, stubbed LLM, temp ChromaDB, no API key
```

Covers BM25 and RRF, language detection, the crawler (sitemaps, cleaning, chunking, add/skip/update/delete/robots/outage
cases), the retrieval eval, the language handling in the planner, analyzer and reporter, `/triage` end to end, and
auto-remediation (fix on first try, retry then pass, escalation after N attempts, rejected patches, test-file protection,
no PR on failure, PR failure handling, graph wiring).

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
│   ├── vectorstore.py    # ChromaDB RAG retrieval (vector, BM25 or hybrid)
│   ├── hybrid.py         # BM25 + reciprocal-rank fusion
│   ├── crawler.py        # Sitemap crawler with change detection
│   ├── remediation.py    # Worktree patch loop, retries, draft PR / escalation
│   ├── embeddings.py     # sentence-transformers or offline hash embeddings
│   └── language.py       # Language detection
├── seed_vectorstore.py   # Seeds ChromaDB with sample past incidents
├── crawl_site.py         # Crawls a sitemap into the crawled_docs collection
├── evals/                # Retrieval eval: recall@k and MRR for vector, BM25, hybrid
├── tests/                # pytest suite (runs offline)
├── main.py               # CLI entry point
├── api.py                # FastAPI server
├── requirements.txt
└── .env.example
```
