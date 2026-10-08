"""Planner/analyzer/reporter language handling and retrieval wiring, with the LLM stubbed out."""

import json

import pytest

import agent.nodes as nodes
from agent.vectorstore import _get_collection, retrieve


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)

        class R:
            content = self.reply

        return R()


def state(**kw):
    base = {"bug_report": "x", "language": "en", "diagnostic_questions": [], "retrieved_contexts": [],
            "analysis": "a", "is_sufficient": False, "iterations": 0, "report": None}
    base.update(kw)
    return base


def test_planner_detects_language_and_asks_for_english_questions(monkeypatch):
    fake = FakeLLM(json.dumps(["q1", "q2", "q3", "q4"]))
    monkeypatch.setattr(nodes, "llm", fake)
    out = nodes.planner(state(bug_report="La latencia de la API aumentó un 300% después del despliegue de la versión 2.3.1"))
    assert out["language"] == "es"
    assert "in English even if the bug report is in another language" in fake.prompts[0]


def test_english_reports_get_no_language_instruction(monkeypatch):
    fake = FakeLLM("analysis")
    monkeypatch.setattr(nodes, "llm", fake)
    nodes.analyzer(state(language="en"))
    assert "Write your entire response in" not in fake.prompts[0]


@pytest.mark.parametrize("node,fields", [(nodes.analyzer, {}), (nodes.reporter, {})])
def test_non_english_reports_answer_in_the_reports_language(monkeypatch, node, fields):
    fake = FakeLLM("texto")
    monkeypatch.setattr(nodes, "llm", fake)
    node(state(language="de", retrieved_contexts=["ctx"]))
    assert "Write your entire response in German" in fake.prompts[0]


def test_retrieve_merges_incident_and_crawled_collections_with_sources(tmp_path, monkeypatch):
    import agent.vectorstore as vs

    monkeypatch.setattr(vs, "CHROMA_PERSIST_DIR", str(tmp_path / "db"))
    _get_collection("past_incidents").upsert(ids=["INC-1"], documents=["Redis pool exhausted by leaked connections"])
    _get_collection("crawled_docs").upsert(
        ids=["p#0"], documents=["Release Redis connections with a context manager"],
        metadatas=[{"url": "https://docs.example.com/pool"}],
    )
    out = retrieve(["redis connections leak"], n_results=2)
    assert any(t.startswith("[source: https://docs.example.com/pool]") for t in out)
    assert any("leaked connections" in t for t in out)


def test_retrieve_skips_empty_collections(tmp_path, monkeypatch):
    import agent.vectorstore as vs

    monkeypatch.setattr(vs, "CHROMA_PERSIST_DIR", str(tmp_path / "db"))
    assert retrieve(["anything"]) == []


def test_full_graph_and_api_carry_language_end_to_end(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.vectorstore as vs
    import api

    monkeypatch.setattr(vs, "CHROMA_PERSIST_DIR", str(tmp_path / "db"))
    _get_collection("past_incidents").upsert(ids=["INC-1"], documents=["Kafka consumer lag after timeout change"])

    class Scripted(FakeLLM):
        def invoke(self, messages):
            prompt = messages[0].content
            self.prompts.append(prompt)
            if "diagnostic questions" in prompt:
                self.reply = json.dumps(["kafka consumer lag", "timeout change", "throughput drop", "config change"])
            elif "quality reviewer" in prompt:
                self.reply = json.dumps({"sufficient": True})
            else:
                self.reply = "respuesta"
            return super().invoke(messages)

    fake = Scripted("")
    monkeypatch.setattr(nodes, "llm", fake)
    r = TestClient(api.app).post("/triage", json={"bug_report": "Los clientes no reciben los correos de confirmación del pedido desde el cambio de configuración"})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "es" and body["iterations"] == 1
    assert any("Write your entire response in Spanish" in p for p in fake.prompts)
