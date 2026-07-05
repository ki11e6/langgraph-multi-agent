"""LangGraph state-graph definition for the multi-agent research assistant.

Architecture
------------
A **Supervisor** node inspects the current state and routes work to one of
three specialist agents:

* **Researcher** -- gathers information via web search and summarisation.
* **Writer** -- turns research notes into a polished draft.
* **Reviewer** -- critiques the draft; decides *revise* or *accept*.

Conditional edges feed the reviewer's verdict back into the supervisor so
the loop can iterate until the output is accepted or a maximum number of
iterations is reached.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agents.config import get_llm
from agents.tools import summarize, web_search

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

MAX_REVISIONS = 3


def _as_text(content) -> str:
    """Flatten a chat message's ``content`` to plain text.

    Groq/OpenAI return a ``str``, but Anthropic and Gemini can return a list of
    content blocks; downstream code (``.strip()``, ``.splitlines()``, ``len()``)
    assumes a string, so normalise here.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(block.get("text", ""))
        else:
            parts.append(str(block))
    return "".join(parts)


class AgentState(BaseModel):
    """Shared state flowing through the graph."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    research_notes: str = ""
    draft: str = ""
    review_feedback: str = ""
    next_agent: str = "researcher"
    revision_count: int = 0
    verdict: str = ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def supervisor_node(state: AgentState) -> dict:
    """Decide which agent should act next based on the current state."""
    llm = get_llm()

    system = SystemMessage(
        content=(
            "You are the supervisor of a research team.  Based on the "
            "conversation so far, decide which agent should act next.\n\n"
            "Agents:\n"
            "  - researcher: gathers information from the web\n"
            "  - writer: drafts a report from research notes\n"
            "  - reviewer: critiques the draft\n\n"
            "Rules:\n"
            "  1. If there are no research notes yet, pick 'researcher'.\n"
            "  2. If there are research notes but no draft, pick 'writer'.\n"
            "  3. If there is a draft, pick 'reviewer'.\n\n"
            "The reviewer decides when the work is complete -- you never finish "
            "the task yourself.\n\n"
            "Respond with ONLY the agent name (one word)."
        )
    )

    context_parts: list[str] = []
    if state.research_notes:
        context_parts.append(f"Research notes:\n{state.research_notes[:2000]}")
    if state.draft:
        context_parts.append(f"Current draft:\n{state.draft[:2000]}")

    human = HumanMessage(
        content=(
            "Current state:\n"
            + ("\n---\n".join(context_parts) if context_parts else "(empty -- no work done yet)")
        )
    )

    response = llm.invoke([system, human])
    next_agent = _as_text(response.content).strip().lower().replace("'", "").replace('"', "")

    # Normalise common LLM responses
    if "finish" in next_agent:
        next_agent = "FINISH"
    elif "research" in next_agent:
        next_agent = "researcher"
    elif "writ" in next_agent:
        next_agent = "writer"
    elif "review" in next_agent:
        next_agent = "reviewer"

    return {
        "next_agent": next_agent,
        "messages": [AIMessage(content=f"[Supervisor] Routing to: {next_agent}")],
    }


def researcher_node(state: AgentState) -> dict:
    """Use tools to research the user's query."""
    query = state.messages[0].content if state.messages else "general research"

    # Step 1 -- web search
    search_results = web_search.invoke(query)

    # Step 2 -- summarise findings
    summary = summarize.invoke(search_results)

    return {
        "research_notes": summary,
        "messages": [AIMessage(content=f"[Researcher] Gathered notes:\n{summary}")],
    }


def writer_node(state: AgentState) -> dict:
    """Produce or revise a written draft from the research notes."""
    llm = get_llm()

    revision_context = ""
    if state.review_feedback:
        revision_context = (
            f"\n\nThe reviewer provided this feedback on your previous draft -- "
            f"address every point:\n{state.review_feedback}"
        )

    system = SystemMessage(
        content=(
            "You are a skilled technical writer.  Using the research notes "
            "provided, write a clear, well-structured report (3-5 paragraphs). "
            "Use markdown formatting."
            + revision_context
        )
    )
    human = HumanMessage(
        content=f"Research notes:\n{state.research_notes}\n\nPrevious draft:\n{state.draft}"
    )

    response = llm.invoke([system, human])
    draft = _as_text(response.content)

    # Clear the consumed feedback so the supervisor routes the fresh draft back
    # to the reviewer instead of looping straight to the writer again.
    return {
        "draft": draft,
        "review_feedback": "",
        "messages": [AIMessage(content=f"[Writer] Draft produced ({len(draft)} chars)")],
    }


def reviewer_node(state: AgentState) -> dict:
    """Critique the draft and decide whether to accept or request revision."""
    llm = get_llm()

    system = SystemMessage(
        content=(
            "You are a meticulous editor.  Review the draft below for accuracy, "
            "clarity, and completeness.  Provide brief, actionable feedback.\n\n"
            "End your review with exactly one of these verdicts on its own line:\n"
            "  ACCEPT -- the draft is ready for publication.\n"
            "  REVISE -- the draft needs further work."
        )
    )
    human = HumanMessage(content=f"Draft:\n{state.draft}")

    response = llm.invoke([system, human])
    feedback = _as_text(response.content)

    # Find the verdict by scanning lines from the end for a standalone ACCEPT /
    # REVISE token. Tokenising avoids substring false-positives ("UNACCEPTABLE",
    # "not accepted"), and scanning past trailing commentary avoids missing a
    # real verdict that isn't on the very last line.
    verdict = "REVISE"
    for line in reversed(feedback.upper().splitlines()):
        tokens = re.findall(r"[A-Z]+", line)
        if "REVISE" in tokens:
            break
        if "ACCEPT" in tokens:
            if "NOT" not in tokens:
                verdict = "ACCEPT"
            break

    # Enforce maximum revision count
    revision_count = state.revision_count + 1
    if revision_count >= MAX_REVISIONS:
        verdict = "ACCEPT"
        feedback += "\n\n[Auto-accepted after maximum revisions reached.]"

    return {
        "review_feedback": feedback,
        "revision_count": revision_count,
        "verdict": verdict,
        "messages": [AIMessage(content=f"[Reviewer] Verdict: {verdict}\n{feedback}")],
    }


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def route_supervisor(state: AgentState) -> Literal["researcher", "writer", "reviewer", "__end__"]:
    """Return the next node name based on the supervisor's decision."""
    agent = state.next_agent
    if agent == "FINISH":
        return END
    if agent in {"researcher", "writer", "reviewer"}:
        return agent
    # Fallback -- shouldn't happen but keeps the graph safe.
    return END


def route_after_review(state: AgentState) -> Literal["writer", "__end__"]:
    """Decide the loop's fate directly from the reviewer's verdict.

    Termination is deterministic here -- not delegated to the LLM supervisor --
    so an ACCEPT (or hitting ``MAX_REVISIONS``) reliably ends the run instead of
    depending on the supervisor to re-read the feedback prose correctly.
    """
    if state.verdict == "ACCEPT" or state.revision_count >= MAX_REVISIONS:
        return END
    return "writer"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Construct and compile the multi-agent research graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes conditionally
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            END: END,
        },
    )

    # Researcher and writer report back to the supervisor for sequencing.
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")

    # The reviewer decides termination deterministically -- accept ends the run,
    # revise loops back to the writer -- without a supervisor round-trip.
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "writer": "writer",
            END: END,
        },
    )

    return graph.compile()
