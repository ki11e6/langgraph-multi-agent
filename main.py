"""CLI entry point for the multi-agent research assistant."""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from agents.graph import build_graph


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
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Research query: {args.query}")
    print(f"{'=' * 60}\n")

    graph = build_graph()

    initial_state = {
        "messages": [HumanMessage(content=args.query)],
    }

    # Stream node-by-node so the user sees progress, capturing the latest draft
    # and any grounding refusal as they flow by.
    draft = ""
    sources: list[dict] = []
    validation_error = ""
    try:
        for step in graph.stream(initial_state, stream_mode="updates"):
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
    except GraphRecursionError:
        # The agents failed to converge within LangGraph's step budget. Surface
        # the best draft produced so far instead of crashing with a traceback.
        print(
            "\n[!] The agents did not converge within the step limit. "
            "Returning the latest draft produced so far."
        )

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
