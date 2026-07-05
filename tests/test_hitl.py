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


def test_tracing_status_on_when_enabled_with_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "my-proj")
    status = main._tracing_status()
    assert "ON" in status
    assert "my-proj" in status


def test_tracing_status_enabled_without_key_warns(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    status = main._tracing_status()
    assert "ON" not in status
    assert "won't upload" in status


# ---------------------------------------------------------------------------
# Review-fix coverage: human authority past the cap, non-dict resume, refusal,
# and the _prompt_human input handling.
# ---------------------------------------------------------------------------


def test_hitl_human_revise_is_honored_past_the_cap(monkeypatch):
    """route_after_human ignores MAX_REVISIONS — a human can keep revising."""
    from agents.graph import MAX_REVISIONS

    _wire(monkeypatch)
    app, config, state = _start("t-past-cap")
    # Revise more times than the autonomous cap would allow.
    for _ in range(MAX_REVISIONS + 2):
        assert "__interrupt__" in state  # still paused for the human, not force-ended
        state = app.invoke(Command(resume={"action": "revise"}), config)
    assert state["revision_count"] > MAX_REVISIONS  # cap did not force termination
    final = app.invoke(Command(resume={"action": "approve"}), config)
    assert "__interrupt__" not in final and final["verdict"] == "ACCEPT"


def test_hitl_non_dict_resume_defaults_to_approve(monkeypatch):
    _wire(monkeypatch)
    app, config, _state = _start("t-nondict")
    final = app.invoke(Command(resume="approve"), config)  # a string, not a dict
    assert final["verdict"] == "ACCEPT"  # no crash; treated as default (approve)


def test_hitl_refuses_ungrounded_without_reaching_human(monkeypatch):
    """An ungrounded draft is refused at `validate`; human_review never runs."""

    def llm(messages):
        system = messages[0].content
        human = messages[1].content if len(messages) > 1 else ""
        if "supervisor of a research team" in system:
            if "Research notes:" in human:
                return AIMessage(content="writer")
            return AIMessage(content="researcher")
        if "skilled technical writer" in system:
            return AIMessage(content="A confident claim with no citation.")
        return AIMessage(content="researcher")

    monkeypatch.setattr(
        graph_module, "get_llm", lambda: type("L", (), {"invoke": staticmethod(llm)})()
    )
    monkeypatch.setattr(graph_module, "web_search", _Tool())
    app = build_graph(human_in_the_loop=True)
    config = {"configurable": {"thread_id": "t-refuse"}}
    final = app.invoke({"messages": [HumanMessage(content="q")]}, config)
    assert final["validation_error"]  # refused at validate
    assert "__interrupt__" not in final  # never paused for a human
    assert final.get("verdict", "") == ""  # reviewer/human never ran


def test_prompt_human_reprompts_then_approves(monkeypatch):
    replies = iter(["huh?", "", "a"])  # garbage, empty, then approve
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    assert main._prompt_human({"reviewer_verdict": "REVISE"}) == {"action": "approve"}


def test_prompt_human_revise_captures_feedback(monkeypatch):
    replies = iter(["r", "add a section on setup"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    assert main._prompt_human({}) == {"action": "revise", "feedback": "add a section on setup"}


def test_prompt_human_eof_propagates(monkeypatch):
    def _raise(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    import pytest

    with pytest.raises(EOFError):
        main._prompt_human({})
