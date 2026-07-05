# Design & Decision Log — Multi-Agent Research Assistant

> This is a **learning + portfolio** project. Its goal is to demonstrate understanding
> of multi-agent orchestration with LangGraph — not to be a production research product.
> This document is a first-class deliverable: it records what was built, how it failed,
> and why the fixes were chosen. If you're reviewing this repo, start here.

---

## 1. What it is

A CLI research assistant built on a LangGraph `StateGraph`. A **Supervisor** routes a
query through three specialists that share one typed state object:

```
          ┌──────────────┐
   ┌─────▶│  Supervisor  │──────┐  (routes on state)
   │      └──────────────┘      │
   │         ▲   ▲   ▲          ▼
   │  researcher writer      (reviewer owns termination)
   │      │     │              │
   └──────┴─────┴──────────────┘
                     writer ⇄ reviewer refinement loop
```

**Why a supervisor topology (not a fixed chain)?** It demonstrates dynamic, state-based
routing and a cyclic writer⇄reviewer refinement loop — patterns a linear chain can't show.
That is the *point* of the project.

---

## 2. The incident (the story worth telling)

Running `how to connect langfuse to langgraph` produced a confident, well-formatted
report **full of fabricated APIs** (`LangfuseTracer`, `graph.run(model=...)`, a `Command`
class "removed in 0.0.30", invented version numbers). The Reviewer *smelled* the
inaccuracy every round and demanded verification — but had no tools to verify — so it
emitted more plausible-but-wrong "corrections." After 3 rounds it force-accepted an
authoritative-looking hallucination.

**Capture the real transcript** (`docs/incident-hallucination.md`) — the ugly before is
evidence you actually debugged this.

### Root cause: ungrounded state, not a bad prompt

Only the Researcher touches reality — and it **summarizes search results into prose and
drops the source URLs**. The Writer and Reviewer then operate on a lossy paraphrase with
no source text, so the LLM fills gaps from parametric memory. That *is* hallucination.
Iterating a writer⇄reviewer loop with no new evidence doesn't increase accuracy — it
increases fluency (polish on the wrong answer).

---

## 3. Product contract (decision)

- **Scope stays broad** — "research assistant for anything."
- **The honesty bar is fixed, not the scope.** One sentence:
  > *For any query, return only claims traceable to a retrieved source, mark anything
  > it can't ground as uncertain, and never present ungrounded content as fact.*
- **Grounding is per-claim, refusal is a fallback:** ship grounded claims with citations,
  flag unverifiable ones as explicit uncertainty, and hard-refuse only when *nothing*
  grounds.

---

## 4. Phased plan

### Phase 0 — Deterministic termination ✅ (done)
The writer⇄reviewer loop previously delegated termination to the LLM supervisor and
never reliably stopped. Fixed: verdict persisted in state, `route_after_review` ends the
graph on ACCEPT or `MAX_REVISIONS`. (Committed on `fix/loop-termination`.)

### Phase 1 — Grounding (the core fix) ✅ (done)

**As built** (a few deliberate deviations from the original sketch below):
- Citations are inline `[n]` markers, not `(claim, [source_id])` tuples — deterministic
  to check with a regex, no fragile structured-output parsing.
- The `validate` node checks citation **presence + validity** (no sources / no citations /
  dangling index), not semantic entailment — that stays future work.
- Kept the Phase 0 force-accept terminal (didn't invert `MAX_REVISIONS`); the `validate`
  gate guarantees grounding regardless of the accept path.
- Dropped the lossy `summarize` step in the researcher — that was *where* the URLs died.

Original ordered plan (test-first):

| # | File | Change |
|---|------|--------|
| 1 | `agents/tools.py` | `web_search` returns structured results `{title, url, snippet, content}` where `content` is **fetched page text** (add `httpx` + `trafilatura`/readability), not just a snippet. |
| 2 | `agents/graph.py` | Add `sources: list[Source]` to graph state (`Source = {url, title, excerpt}`); Researcher appends, Writer/Reviewer read. |
| 3 | `agents/graph.py` | Writer emits claims as `(claim_text, [source_id])` and is forbidden (via prompt) from unattributed statements — no invented APIs/versions. |
| 4 | `agents/graph.py` | New **deterministic** `validate` node (`writer → validate → reviewer`): plain Python, no LLM. Checks (a) sources non-empty, (b) every claim cites ≥1 `source_id` that exists in state. Classifies each claim `supported`/`unsupported`. |
| 5 | `agents/graph.py` | Routing: `supported == 0` → **refuse** terminal node; some unsupported → answer **with a "⚠ Unverified" section**; all supported → clean. Invert the `MAX_REVISIONS` terminal from force-*accept* to force-*caveat*. |
| 6 | `main.py` | Render a `## Sources` section; exit non-zero on refusal. |

**Tests** (`tests/`, red first):
- `web_search` returns non-empty `url` + `content` (mock `ddgs`).
- state propagation: seed `sources`, assert Writer node receives them.
- **regression:** a claim citing a `source_id` absent from state → refuse/flag (the langfuse incident).
- empty sources → refuse; fully-grounded claims → pass.

**Key teaching point (say this in interviews):** the `validate` node is *deterministic
code*, deliberately separated from the LLM Reviewer. "LLM proposes, code disposes" — an
agent graph shouldn't be all model calls; some edges are boring, testable Python.

### Phase 2 — Make it legible (equal-weight deliverable) 📚
- `README.md` architecture section + the loop diagram, auto-generated:
  `app.get_graph().draw_mermaid()` → paste the mermaid block.
- "How it works" (5 bullets): supervisor routing, typed state passing, the validate gate,
  refuse-vs-answer decision, why deterministic not LLM-judged.
- `docs/incident-hallucination.md`: before (hallucinated) vs after (grounded) transcripts.
- **Scope & Limitations** section: this is a learning project; grounding ≠ source *truth*;
  future work below. Naming what you didn't do is a seniority signal.

### Phase 3 — Optional, if you want the advanced-primitives signal 🔭
Sequential, not required. Each is a documented "chapter":
- **Human-in-the-loop:** add a checkpointer + `interrupt()` so the graph pauses for human
  approve/revise instead of MAX_REVISIONS. Demonstrates persistence + checkpointing + HITL.
- **Observability:** LangSmith (or OTel) tracing — the "glass box" screenshot for the README.
- **Streaming:** `stream_mode="updates"` demo GIF.
- **Subgraph:** refactor Researcher into its own compiled subgraph (plan → search → dedupe → summarize).

---

## 5. Explicitly deferred (Future Work — say why)

- **Semantic entailment grading** ("does the source *actually support* this claim") — needs
  an LLM/NLI judge; non-deterministic; expands the diff. Phase 1 does structural
  citation-presence only (the 80/20). Documenting that you know the difference is the win.
- **Source-trust scoring** — grounding to a source ≠ the source being true. Separate epic.
- **Retrieval quality** for obscure topics (content-farm results) — real limit of the
  web-search architecture; acknowledged, not solved.

---

## 6. One-line résumé framing

> Built a LangGraph supervisor/worker research agent; caught it confidently hallucinating,
> diagnosed the cause as **ungrounded state** (source URLs dropped between nodes), and fixed
> it with evidence-carrying state, a **deterministic validation gate**, and a cite-or-refuse
> contract — keeping the LLM for judgment and plain code for invariants.
