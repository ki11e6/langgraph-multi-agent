"""Tests for Phase 3: human-in-the-loop review and tracing status.

HITL is opt-in (`build_graph(human_in_the_loop=True)`): a `human_review` node
interrupts the graph so a person makes the accept/revise call, resumed via
`Command(resume=...)`. These tests drive that interrupt/resume cycle with a
mocked LLM and search tool.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

import main
from agents import graph as graph_module
from agents.graph import build_graph


def _role_aware_llm(messages):
    system = messages[0].content
    human = messages[1].content if len(messages) > 1 else ""
    if "supervisor of a research team" in system:
        if "Current draft:" in human:
            return AIMessage(content="reviewer")
        if "Research notes:" in human:
            return AIMessage(content="writer")
        return AIMessage(content="researcher")
    if "meticulous editor" in system:
        return AIMessage(content="Needs work.\nREVISE")  # model wants revise; human overrides
    if "skilled technical writer" in system:
        return AIMessage(content="A grounded draft [^1].")
    return AIMessage(content="researcher")


class _Tool:
    @staticmethod
    def invoke(_q):
        return [{"title": "T", "url": "https://ex/1", "snippet": "s", "content": "c"}]


def _wire(monkeypatch):
    monkeypatch.setattr(
        graph_module, "get_llm", lambda: type("L", (), {"invoke": staticmethod(_role_aware_llm)})()
    )
    monkeypatch.setattr(graph_module, "web_search", _Tool())


def _start(thread_id: str):
    app = build_graph(human_in_the_loop=True)
    config = {"configurable": {"thread_id": thread_id}}
    state = app.invoke({"messages": [HumanMessage(content="q")]}, config)
    return app, config, state


# ---------------------------------------------------------------------------
# HITL interrupt / resume
# ---------------------------------------------------------------------------


def test_hitl_pauses_for_human_review(monkeypatch):
    _wire(monkeypatch)
    _app, _config, state = _start("t-pause")
    assert "__interrupt__" in state  # graph paused, waiting on the human


def test_hitl_human_approval_ends_the_run(monkeypatch):
    _wire(monkeypatch)
    app, config, _state = _start("t-approve")
    final = app.invoke(Command(resume={"action": "approve"}), config)
    assert "__interrupt__" not in final  # completed
    assert final["verdict"] == "ACCEPT"  # human's decision, overriding the model's REVISE


def test_hitl_human_revise_loops_then_approve(monkeypatch):
    _wire(monkeypatch)
    app, config, _state = _start("t-revise")
    # Human sends it back -> writer revises -> pauses for review again.
    again = app.invoke(Command(resume={"action": "revise", "feedback": "add detail"}), config)
    assert "__interrupt__" in again
    assert again["revision_count"] >= 2
    # Now approve -> ends.
    final = app.invoke(Command(resume={"action": "approve"}), config)
    assert final["verdict"] == "ACCEPT"


def test_autonomous_graph_needs_no_thread_and_has_no_human_node(monkeypatch):
    # Default (autonomous) build stays checkpointer-free and invokes without config.
    _wire(monkeypatch)
    final = build_graph().invoke({"messages": [HumanMessage(content="q")]})
    assert final["verdict"] == "ACCEPT"  # reviewer force-accepts at the cap
    assert "human_review" not in build_graph().get_graph().nodes


# ---------------------------------------------------------------------------
# Observability: tracing status reporting
# ---------------------------------------------------------------------------


def test_tracing_status_off_by_default(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert "off" in main._tracing_status()


def test_tracing_status_on_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "my-proj")
    status = main._tracing_status()
    assert "ON" in status
    assert "my-proj" in status
