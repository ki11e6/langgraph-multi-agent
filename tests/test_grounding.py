"""Tests for Phase 1 grounding: structured sources, and the deterministic
`validate` node that refuses ungrounded or fabricated-citation drafts.
"""

from __future__ import annotations

from langgraph.graph import END

from agents import graph as graph_module
from agents import tools as tools_module
from agents.graph import (
    AgentState,
    route_after_validate,
    validate_node,
)


def _sources(n: int) -> list[dict]:
    return [
        {"title": f"T{i}", "url": f"https://ex/{i}", "snippet": "s", "content": "c"}
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# validate_node -- deterministic grounding gate
# ---------------------------------------------------------------------------


def test_validate_refuses_when_no_sources():
    out = validate_node(AgentState(draft="A claim [1].", sources=[]))
    assert out["validation_error"]  # non-empty => refuse


def test_validate_refuses_draft_without_citations():
    draft = "A confident claim with no citation."
    out = validate_node(AgentState(draft=draft, sources=_sources(2)))
    assert out["validation_error"]


def test_validate_refuses_dangling_citation():
    # Draft cites [5] but only 2 sources exist -> fabricated/dangling reference.
    out = validate_node(AgentState(draft="Per the docs [5], do X.", sources=_sources(2)))
    assert out["validation_error"]


def test_validate_passes_grounded_draft():
    out = validate_node(AgentState(draft="First [1]. Second [2].", sources=_sources(2)))
    assert out["validation_error"] == ""


# ---------------------------------------------------------------------------
# routing after validate
# ---------------------------------------------------------------------------


def test_route_after_validate_refuses_to_end():
    assert route_after_validate(AgentState(validation_error="no sources")) == END


def test_route_after_validate_ok_goes_to_reviewer():
    assert route_after_validate(AgentState(validation_error="")) == "reviewer"


# ---------------------------------------------------------------------------
# web_search -- returns structured sources with url + fetched content
# ---------------------------------------------------------------------------


def test_web_search_returns_structured_sources(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "_raw_search",
        lambda q: [{"title": "Doc", "link": "https://ex/doc", "snippet": "a snippet"}],
    )
    monkeypatch.setattr(tools_module, "_fetch_text", lambda url: "full page text")

    results = tools_module.web_search.invoke("query")
    assert isinstance(results, list) and results
    r = results[0]
    assert r["url"] == "https://ex/doc"
    assert r["title"] == "Doc"
    assert r["content"] == "full page text"


def test_web_search_falls_back_to_snippet_when_fetch_fails(monkeypatch):
    monkeypatch.setattr(
        tools_module,
        "_raw_search",
        lambda q: [{"title": "Doc", "link": "https://ex/doc", "snippet": "a snippet"}],
    )
    monkeypatch.setattr(tools_module, "_fetch_text", lambda url: "")  # fetch failed

    results = tools_module.web_search.invoke("query")
    assert results[0]["content"] == "a snippet"


def test_web_search_empty_returns_empty_list(monkeypatch):
    monkeypatch.setattr(tools_module, "_raw_search", lambda q: [])
    assert tools_module.web_search.invoke("query") == []


# ---------------------------------------------------------------------------
# researcher_node -- populates sources in state
# ---------------------------------------------------------------------------


def test_researcher_populates_sources(monkeypatch):
    from langchain_core.messages import HumanMessage

    fake = [{"title": "Doc", "url": "https://ex/doc", "snippet": "s", "content": "grounded text"}]

    class _Tool:
        @staticmethod
        def invoke(_q):
            return fake

    monkeypatch.setattr(graph_module, "web_search", _Tool())
    out = graph_module.researcher_node(AgentState(messages=[HumanMessage(content="q")]))
    assert out["sources"] == fake
    assert "grounded text" in out["research_notes"]
    assert "https://ex/doc" in out["research_notes"]
