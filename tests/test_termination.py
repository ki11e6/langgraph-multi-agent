"""Tests for the writer<->reviewer loop termination logic.

Covers the fix for the non-terminating refinement loop: the reviewer's verdict
must be persisted to state and drive routing deterministically, with
``MAX_REVISIONS`` acting as a hard stop -- no reliance on the LLM supervisor.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from agents import graph as graph_module
from agents.graph import (
    MAX_REVISIONS,
    AgentState,
    _as_text,
    build_graph,
    reviewer_node,
    route_after_review,
)


class _FakeLLM:
    """Minimal stand-in for a chat model: returns a canned reply."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def invoke(self, _messages):
        return AIMessage(content=self._reply)


# ---------------------------------------------------------------------------
# AC2 / AC3 -- route_after_review is deterministic
# ---------------------------------------------------------------------------


def test_route_after_review_accept_ends():
    """AC2: an ACCEPT verdict ends the graph without a supervisor round-trip."""
    state = AgentState(verdict="ACCEPT", revision_count=1)
    assert route_after_review(state) == END


def test_route_after_review_revise_goes_to_writer():
    """A REVISE verdict below the cap loops back to the writer."""
    state = AgentState(verdict="REVISE", revision_count=1)
    assert route_after_review(state) == "writer"


def test_route_after_review_max_revisions_is_hard_stop():
    """AC3: hitting the cap ends the graph even if the verdict says REVISE."""
    state = AgentState(verdict="REVISE", revision_count=MAX_REVISIONS)
    assert route_after_review(state) == END


# ---------------------------------------------------------------------------
# AC1 / AC4 -- reviewer_node persists a robustly-parsed verdict
# ---------------------------------------------------------------------------


def test_reviewer_persists_accept_verdict(monkeypatch):
    """AC1: verdict lands in state, not just a display message."""
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM("Solid work.\nACCEPT"))
    out = reviewer_node(AgentState(draft="d"))
    assert out["verdict"] == "ACCEPT"


def test_reviewer_persists_revise_verdict(monkeypatch):
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM("Needs sources.\nREVISE"))
    out = reviewer_node(AgentState(draft="d"))
    assert out["verdict"] == "REVISE"


def test_reviewer_verdict_robust_to_trailing_newline(monkeypatch):
    """AC4: a trailing blank line must not swallow the verdict."""
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM("Great.\nACCEPT\n\n"))
    out = reviewer_node(AgentState(draft="d"))
    assert out["verdict"] == "ACCEPT"


def test_reviewer_force_accepts_at_cap(monkeypatch):
    """AC3: at the last allowed revision the verdict is forced to ACCEPT."""
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM("Still weak.\nREVISE"))
    out = reviewer_node(AgentState(draft="d", revision_count=MAX_REVISIONS - 1))
    assert out["verdict"] == "ACCEPT"
    assert out["revision_count"] == MAX_REVISIONS


# ---------------------------------------------------------------------------
# P1 -- verdict parse must not false-positive or miss the real verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("last_line", ["This draft is UNACCEPTABLE.", "REVISE -- not accepted yet"])
def test_reviewer_rejects_inversion_words(monkeypatch, last_line):
    """'UNACCEPTABLE' / 'not accepted' must NOT be read as ACCEPT."""
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM(f"Feedback.\n{last_line}"))
    out = reviewer_node(AgentState(draft="d"))
    assert out["verdict"] == "REVISE"


def test_reviewer_accepts_with_trailing_commentary(monkeypatch):
    """A genuine ACCEPT followed by a chatty closing line is still ACCEPT."""
    reply = "ACCEPT\n\nGreat job on the intro!"
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM(reply))
    out = reviewer_node(AgentState(draft="d"))
    assert out["verdict"] == "ACCEPT"


def test_reviewer_empty_feedback_defaults_to_revise(monkeypatch):
    """No parseable verdict → REVISE (safe default), not a crash."""
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM("   \n\n"))
    out = reviewer_node(AgentState(draft="d"))
    assert out["verdict"] == "REVISE"


# ---------------------------------------------------------------------------
# D3 -- non-string message content (Anthropic/Gemini blocks) is handled
# ---------------------------------------------------------------------------


def test_as_text_flattens_content_blocks():
    assert _as_text("plain") == "plain"
    assert _as_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _as_text(["x", "y"]) == "xy"


def test_reviewer_handles_block_content(monkeypatch):
    """A provider returning content as blocks must not crash the parse."""
    blocks = [{"type": "text", "text": "Good draft.\nACCEPT"}]
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM(blocks))
    out = reviewer_node(AgentState(draft="d"))
    assert out["verdict"] == "ACCEPT"


# ---------------------------------------------------------------------------
# AC5 -- the compiled graph halts in bounded steps
# ---------------------------------------------------------------------------


def _role_aware_llm(messages):
    """Fake chat model that answers based on the system prompt's role."""
    system = messages[0].content
    human = messages[1].content if len(messages) > 1 else ""
    if "supervisor of a research team" in system:
        if "Current draft:" in human:
            return AIMessage(content="reviewer")
        if "Research notes:" in human:
            return AIMessage(content="writer")
        return AIMessage(content="researcher")
    if "meticulous editor" in system:
        return AIMessage(content="Not there yet.\nREVISE")  # never accepts on its own
    if "skilled technical writer" in system:
        return AIMessage(content="A draft paragraph.")
    return AIMessage(content="FINISH")


class _FakeTool:
    """Stand-in for a LangChain tool (the real ones are frozen models)."""

    def __init__(self, result: str) -> None:
        self._result = result

    def invoke(self, _arg):
        return self._result


def test_graph_terminates_without_recursion_error(monkeypatch):
    """AC5: even with a reviewer that never accepts, the cap ends the run."""
    # Route the fake LLM by system prompt rather than a single canned reply.
    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: type("LLM", (), {"invoke": staticmethod(_role_aware_llm)})(),
    )
    monkeypatch.setattr(graph_module, "web_search", _FakeTool("raw search results"))
    monkeypatch.setattr(graph_module, "summarize", _FakeTool("condensed notes"))

    app = build_graph()
    final = app.invoke({"messages": [HumanMessage(content="test query")]})

    assert final["revision_count"] == MAX_REVISIONS  # exact cap, no premature exit
    assert final["draft"]


def test_graph_ends_early_on_accept(monkeypatch):
    """The graph terminates via verdict==ACCEPT before the cap is reached."""

    def accepting_llm(messages):
        system = messages[0].content
        human = messages[1].content if len(messages) > 1 else ""
        if "supervisor of a research team" in system:
            if "Current draft:" in human:
                return AIMessage(content="reviewer")
            if "Research notes:" in human:
                return AIMessage(content="writer")
            return AIMessage(content="researcher")
        if "meticulous editor" in system:
            return AIMessage(content="Looks great.\nACCEPT")  # accepts first pass
        if "skilled technical writer" in system:
            return AIMessage(content="A draft paragraph.")
        return AIMessage(content="FINISH")

    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: type("LLM", (), {"invoke": staticmethod(accepting_llm)})(),
    )
    monkeypatch.setattr(graph_module, "web_search", _FakeTool("raw search results"))
    monkeypatch.setattr(graph_module, "summarize", _FakeTool("condensed notes"))

    final = build_graph().invoke({"messages": [HumanMessage(content="test query")]})

    assert final["verdict"] == "ACCEPT"
    assert final["revision_count"] == 1  # ended on first review, well below the cap
