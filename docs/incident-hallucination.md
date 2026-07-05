# Incident: confident hallucination, and the grounding fix

This is a real failure this project hit, and how it was fixed. It's kept in the
repo on purpose — the diagnosis is the point, not just the working demo. See
[`../DESIGN.md`](../DESIGN.md) for the full decision log.

## The query

```
uv run python main.py --verbose 'how to connect langfuse to langgraph'
```

## Before — a fluent, confident lie

The pipeline produced a polished, authoritative report **full of fabricated
APIs**. None of these exist as written:

- `LangfuseTracer` and `lf.get_callback()` returning it
- `Graph(callbacks=[lf.get_callback()])` and `graph.run(model=..., prompt=...)`
- a `Command` / `State` class "removed in LangGraph 0.0.30"
- invented version numbers (`v0.0.38`)

Worse: the **Reviewer smelled the inaccuracy every round** ("verify the exact
env var names", "the `completion` helper does not accept a `client` argument")
but had **no tools to verify**, so it emitted more plausible-but-wrong
corrections. After 3 rounds it force-accepted the hallucination. The output
looked more trustworthy the longer it ran — the worst possible failure mode.

## Root cause — ungrounded state, not a bad prompt

Only the Researcher touched reality, and it **summarised the search results into
prose, dropping the source URLs**. From that point on, the Writer and Reviewer
were reasoning from a lossy paraphrase with no source text in context, so the
LLM filled the gaps from parametric memory. That *is* hallucination. Iterating a
writer⇄reviewer loop with no new evidence didn't add accuracy — it added
fluency (polish on the wrong answer).

## Fix

1. `web_search` returns **structured sources** `{title, url, snippet, content}`
   and fetches the actual page text — provenance is preserved, not summarised away.
2. `sources` are carried in graph state; the Writer must **cite each claim `[n]`**
   and is instructed not to invent APIs/versions absent from the sources.
3. A **deterministic `validate` node** (plain Python, no LLM) sits between the
   Writer and Reviewer and refuses any draft with no sources, no citations, or a
   citation pointing at a source that doesn't exist.

## After — grounded, cited, or an honest refusal

Same query, same command. The run now retrieves **5 real sources** (the official
Langfuse LangGraph cookbook, the LangChain provider docs), passes the `validate`
gate, and cites them inline:

```
## Overview
Langfuse is an open-source LLM-engineering platform that lets teams trace API
calls, manage prompts, and run evaluations [4][5]. ...

## Observability for LangGraph
LangGraph ... Langfuse provides first-class observability for LangGraph
pipelines, and the official integration is documented on the LangChain
Providers → Langfuse page [4]. ...

## Sources
[1] Open Source Observability for LangGraph - Langfuse — https://langfuse.com/guides/cookbook/integration_langgraph
[4] Langfuse integrations - Docs by LangChain — https://docs.langchain.com/oss/python/integrations/providers/langfuse
[5] Get Started - Langfuse — https://langfuse.com/docs/observability/get-started
```

The fabricated `LangfuseTracer` / `graph.run(model=...)` / invented-version
strings are **gone** (0 occurrences); the real `CallbackHandler` API appears
instead. If retrieval had returned nothing usable, the tool would have exited
with a refusal rather than inventing an answer.

## What this deliberately does *not* do (future work)

- **Semantic entailment** — `validate` checks that citations *exist*, not that the
  cited source actually *supports* the sentence. That needs an NLI/LLM judge and
  is a separate, non-deterministic gate.
- **Source-trust scoring** — a claim can faithfully cite a low-quality page.
  Grounding ≠ truth; it means every claim is traceable to a source you can eyeball.
