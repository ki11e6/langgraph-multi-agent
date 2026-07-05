"""CLI entry point for the multi-agent research assistant."""

from __future__ import annotations

import argparse
import os
import sys

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from agents.graph import build_graph


def _tracing_status() -> str:
    """Report whether LangSmith tracing is active (it auto-instruments via env)."""
    truthy = {"1", "true", "yes"}
    enabled = (
        os.getenv("LANGSMITH_TRACING", "").lower() in truthy
        or os.getenv("LANGCHAIN_TRACING_V2", "").lower() in truthy
    )
    if enabled:
        project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"
        return f"LangSmith tracing: ON (project={project!r})"
    return "LangSmith tracing: off (set LANGSMITH_TRACING=true + LANGSMITH_API_KEY to enable)"


def _prompt_human(payload: dict) -> dict:
    """Show the draft + AI critique and ask the human to approve or revise."""
    print(f"\n{'=' * 60}")
    print("  🧑 HUMAN REVIEW — graph paused for your decision")
    print(f"{'=' * 60}")
    print(
        f"AI reviewer verdict: {payload.get('reviewer_verdict', '?')} "
        f"(revision {payload.get('revision_count')}/{payload.get('max_revisions')})"
    )
    feedback = payload.get("reviewer_feedback") or ""
    if feedback:
        print(f"\nAI reviewer notes:\n{feedback[:800]}")
    choice = input("\n  [a]pprove / [r]evise: ").strip().lower()
    if choice.startswith("r"):
        note = input("  Revision instructions (optional): ").strip()
        return {"action": "revise", "feedback": note}
    return {"action": "approve"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant powered by LangGraph",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="The research query to investigate.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full message contents from every node.",
    )
    parser.add_argument(
        "--human-review",
        action="store_true",
        help="Pause for a human approve/revise decision before finishing (human-in-the-loop).",
    )
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Research query: {args.query}")
    print(f"  {_tracing_status()}")
    print(f"{'=' * 60}\n")

    graph = build_graph(human_in_the_loop=args.human_review)
    # HITL needs a checkpointer thread to pause/resume; autonomous runs don't.
    config = {"configurable": {"thread_id": "cli-session"}} if args.human_review else None

    # Stream node-by-node so the user sees progress, capturing the latest draft
    # and any grounding refusal as they flow by. When the graph interrupts for
    # human review, prompt and resume via Command(resume=...) until it completes.
    draft = ""
    sources: list[dict] = []
    validation_error = ""
    stream_input = {"messages": [HumanMessage(content=args.query)]}
    try:
        while True:
            interrupted = False
            for step in graph.stream(stream_input, config, stream_mode="updates"):
                if "__interrupt__" in step:
                    payload = step["__interrupt__"][0].value
                    stream_input = Command(resume=_prompt_human(payload))
                    interrupted = True
                    break

                for node_name, node_output in step.items():
                    print(f"\n--- [{node_name.upper()}] ---")

                    if node_output.get("draft"):
                        draft = node_output["draft"]
                    if node_output.get("sources"):
                        sources = node_output["sources"]
                    # Track the latest grounding verdict ("" once a draft passes).
                    if "validation_error" in node_output:
                        validation_error = node_output["validation_error"]

                    # Print latest message from this node
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        if args.verbose:
                            print(content)
                        else:
                            # Print first 500 chars for brevity
                            preview = content[:500]
                            if len(content) > 500:
                                preview += "..."
                            print(preview)

            if not interrupted:
                break
    except GraphRecursionError:
        # The agents failed to converge within LangGraph's step budget. The latest
        # draft may never have passed the grounding gate, so refuse rather than
        # print an unvalidated report.
        print(f"\n{'=' * 60}")
        print("  ⚠ DID NOT CONVERGE — no answer produced")
        print(f"{'=' * 60}\n")
        print(f"Query: {args.query!r}")
        print("The agents did not converge within the step limit.")
        sys.exit(1)

    # Refuse rather than ship an ungrounded answer.
    if validation_error:
        print(f"\n{'=' * 60}")
        print("  ⚠ INSUFFICIENT GROUNDING — no answer produced")
        print(f"{'=' * 60}\n")
        print(f"Query: {args.query!r}")
        print(f"Reason: {validation_error}\n")
        print("This tool will not present claims it cannot attribute to a source.")
        sys.exit(1)

    # Print final draft
    print(f"\n{'=' * 60}")
    print("  FINAL REPORT")
    print(f"{'=' * 60}\n")

    if draft:
        print(draft)
    else:
        print("No draft was produced.")

    if sources:
        print("\n## Sources")
        for i, s in enumerate(sources, 1):
            print(f"[{i}] {s.get('title', 'Untitled')} — {s.get('url', '')}")


if __name__ == "__main__":
    main()
