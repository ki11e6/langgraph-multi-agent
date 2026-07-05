# 🧠 Multi-Agent Research Assistant

A multi-agent AI system built with **[LangGraph](https://github.com/langchain-ai/langgraph)** that turns a single question into a polished, well-researched report. A team of specialized agents — a **Researcher**, a **Writer**, and a **Reviewer** — collaborate under a central **Supervisor** that orchestrates the workflow as a stateful graph.

It demonstrates the **supervisor / orchestrator-worker pattern** for agentic systems, dynamic LLM-based routing via conditional edges, an iterative writer↔reviewer refinement loop, tool use for live web search, and a fully **provider-agnostic** LLM layer (Groq, Gemini, OpenAI, or Anthropic — switchable with one environment variable).

---

## ✨ Features

- **Multi-agent orchestration** — cooperating agents modeled as nodes in a LangGraph `StateGraph`.
- **Dynamic supervisor routing** — a coordinator node decides which agent runs next based on shared state, instead of a brittle hard-coded pipeline.
- **Grounded, cite-or-refuse output** — the Writer must cite retrieved sources with footnote markers (`[^n]`); a **deterministic `validate` node** refuses to ship a draft that has no sources, no citations, or citations pointing at sources that don't exist (or that were retrieved empty). See the [incident write-up](docs/incident-hallucination.md) for *why*.
- **Iterative refinement** — the Writer and Reviewer loop until the draft is accepted or `MAX_REVISIONS` is reached; termination is decided deterministically, so the loop provably halts.
- **Live web research** — the Researcher retrieves keyless **DuckDuckGo** results and fetches the page text, carrying structured sources (title, URL, content) through the graph.
- **Provider-agnostic LLM layer** — run on **Groq** (free, default), **Gemini**, **OpenAI**, or **Anthropic** by changing `LLM_PROVIDER`.
- **Streaming output** — node-by-node progress is streamed to the terminal as the graph executes.
- **Human-in-the-loop (opt-in)** — `--human-review` pauses the graph (LangGraph `interrupt` + a `MemorySaver` checkpointer) so a human makes the final accept/revise call, overriding the model's verdict. Resumes via `Command(resume=...)`.
- **Observability** — LangSmith tracing auto-instruments every LLM/graph step when env vars are set; startup reports whether it's on.
- **Zero-cost by default** — Groq's free tier + keyless search means it runs without any paid API or credit card.

---

## 🏗️ Architecture

```mermaid
graph TD;
    __start__([start]) --> supervisor
    supervisor -.->|no sources| researcher
    supervisor -.->|sources, no draft| writer
    researcher --> supervisor
    writer --> validate
    validate -.->|grounded| reviewer
    validate -.->|ungrounded → refuse| __end__([END])
    reviewer -.->|REVISE| writer
    reviewer -.->|ACCEPT / max revisions| __end__
```

> This diagram is generated from the compiled graph itself:
> `build_graph().get_graph().draw_mermaid()`.

Two design choices worth calling out:

- **The Supervisor sequences the early work** (research → first draft), but it does **not** own termination. Once a draft exists, control flows `writer → validate → reviewer`, and the **Reviewer's verdict deterministically ends the loop** (via `route_after_review`) rather than asking the LLM Supervisor to notice "ACCEPT." Delegating termination to the LLM was the original cause of a near-infinite loop.
- **`validate` is a deterministic gate, not an agent.** The LLM *proposes* a draft; plain Python *disposes* on whether it's grounded enough to ship. An agent graph shouldn't be all model calls — some edges are boring, testable code. This is what turns "confident hallucination" into "grounded answer or honest refusal."

### Agent roles

| Agent | Responsibility | Tools |
|---|---|---|
| **Supervisor** | Inspects shared state and routes the early work (research → first draft). | LLM-based routing |
| **Researcher** | Retrieves web results and fetches page text, keeping structured sources (title, URL, content) in state — provenance is preserved, not summarised away. | `web_search` (DuckDuckGo + page fetch) |
| **Writer** | Transforms sources into a markdown report, **citing each claim `[^n]`**; incorporates reviewer feedback on revisions. | LLM generation |
| **Validate** | **Deterministic (no LLM).** Refuses drafts with no sources, no citations, dangling citations, or citations to empty-content sources. | plain Python |
| **Reviewer** | Critiques the draft for accuracy, clarity, and completeness; issues an **ACCEPT** or **REVISE** verdict that deterministically ends or continues the loop. | LLM evaluation |

---

## ⚙️ How it works

The system is a **state machine**. A single typed `AgentState` object flows through the graph, and each node returns a partial update that LangGraph merges into it.

**Shared state (`AgentState`):**

| Field | Purpose |
|---|---|
| `messages` | Running conversation log (uses the `add_messages` reducer to append). |
| `sources` | Structured `{title, url, snippet, content}` retrieved by the Researcher — the evidence every claim must trace back to. |
| `research_notes` | Numbered, URL-preserving view of the sources handed to the Writer. |
| `draft` | Current report produced by the Writer, with footnote `[^n]` citations. |
| `review_feedback` | Reviewer's critique; consumed and cleared by the Writer on each revision. |
| `verdict` | Reviewer's `ACCEPT`/`REVISE` decision — persisted so routing is deterministic. |
| `validation_error` | Set by the `validate` node when a draft isn't grounded; triggers refusal. |
| `next_agent` | The Supervisor's routing decision. |
| `revision_count` | Counts revision cycles; caps the loop at `MAX_REVISIONS` (3). |

**Execution flow for one query:**

1. **Supervisor** sees no sources → routes to **Researcher**.
2. **Researcher** searches + fetches pages → writes `sources` and `research_notes`.
3. **Supervisor** sees sources but no draft → routes to **Writer**.
4. **Writer** drafts a report citing `[^n]` → writes `draft`.
5. **Validate** (deterministic) checks grounding → `ungrounded` ends the run with a refusal; `grounded` → **Reviewer**.
6. **Reviewer** critiques → `ACCEPT` or `REVISE` (verdict persisted to state).
7. `REVISE` → back to **Writer** (loop); `ACCEPT` or `MAX_REVISIONS` → **END**.

---

## 🧩 Tech stack

| Component | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph`, conditional edges) |
| LLM abstraction | [LangChain](https://github.com/langchain-ai/langchain) (`langchain-core`) |
| LLM — Groq (default) | Llama 3.3 70B via `langchain-groq` |
| LLM — Gemini | `gemini-2.5-flash-lite` via `langchain-google-genai` |
| LLM — OpenAI | GPT-4o via `langchain-openai` |
| LLM — Anthropic | Claude Sonnet 4.5 via `langchain-anthropic` |
| Web search | [DuckDuckGo](https://duckduckgo.com/) (`ddgs`) via `langchain-community` — no API key |
| Page fetching | `httpx` + stdlib HTML parsing (best-effort, falls back to snippets) |
| State & validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Configuration | `python-dotenv` |
| Build backend | [Hatchling](https://hatch.pypa.io/) |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| Linting | [Ruff](https://github.com/astral-sh/ruff) |
| Containerization | Docker (Python 3.12 slim) |

---

## 🚀 Getting started

### Prerequisites
- Python **3.12+**
- [uv](https://github.com/astral-sh/uv) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A free LLM API key (Groq recommended — no credit card)

### Install & run

```bash
# 1. Clone
git clone https://github.com/<your-username>/langgraph-multi-agent.git
cd langgraph-multi-agent

# 2. Configure environment
cp .env-template .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com/keys)

# 3. Install dependencies (creates .venv)
uv sync

# 4. Run a research query
uv run python main.py "What are the latest breakthroughs in quantum computing?"

# Verbose mode prints the full output of every agent
uv run python main.py --verbose "Explain the current state of nuclear fusion energy"

# Human-in-the-loop: pause for your approve/revise decision before finishing
uv run python main.py --human-review "How do I connect Langfuse to LangGraph?"
```

---

## 🔧 Configuration

Set the provider with `LLM_PROVIDER` in `.env` and supply the matching key:

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | No | `groq` (default), `gemini`, `openai`, or `anthropic` |
| `GROQ_API_KEY` | If `groq` | [Groq](https://console.groq.com/keys) — free tier, no credit card |
| `GOOGLE_API_KEY` | If `gemini` | [Google AI Studio](https://aistudio.google.com) |
| `OPENAI_API_KEY` | If `openai` | OpenAI |
| `ANTHROPIC_API_KEY` | If `anthropic` | Anthropic |
| `LLM_MODEL` | No | Override the provider's default model |
| `LANGSMITH_TRACING` | No | Set to `true` (with `LANGSMITH_API_KEY`) to trace every step in [LangSmith](https://smith.langchain.com) |

> Web search uses **DuckDuckGo** and requires **no API key**.

---

## 📤 Example output

```text
$ uv run python main.py "Compare React and Svelte for building modern web apps"

============================================================
  Research query: Compare React and Svelte for building modern web apps
============================================================

--- [SUPERVISOR] ---  [Supervisor] Routing to: researcher
--- [RESEARCHER] ---  [Researcher] Gathered notes: ...
--- [SUPERVISOR] ---  [Supervisor] Routing to: writer
--- [WRITER] ---      [Writer] Draft produced (2847 chars)
--- [SUPERVISOR] ---  [Supervisor] Routing to: reviewer
--- [REVIEWER] ---    [Reviewer] Verdict: ACCEPT
--- [SUPERVISOR] ---  [Supervisor] Routing to: FINISH

============================================================
  FINAL REPORT
============================================================

## React vs. Svelte: A Comparative Analysis
...
```

---

## 🐳 Docker

```bash
docker build -t research-assistant .
docker run --env-file .env research-assistant "Your research query here"
```

The image uses a multi-stage build with `uv` for fast, reproducible dependency installation on a `python:3.12-slim` base.

---

## 📁 Project structure

```
langgraph-multi-agent/
├── agents/
│   ├── __init__.py
│   ├── config.py      # Provider-agnostic LLM factory (Groq/Gemini/OpenAI/Anthropic)
│   ├── graph.py       # AgentState, agent nodes, routing, and StateGraph assembly
│   └── tools.py       # Custom tools: web_search (DuckDuckGo) + summarize
├── main.py            # CLI entry point — builds and streams the graph
├── pyproject.toml     # Project metadata, dependencies, tooling config
├── Dockerfile         # Multi-stage container build
├── .env-template      # Environment variable template
└── README.md
```

---

## 🧠 Design notes

- **Supervisor / orchestrator-worker pattern** — a central node routes to specialist workers, keeping the workflow flexible and extensible (adding an agent = add a node + a routing branch).
- **Conditional edges** — the Supervisor uses `add_conditional_edges` so the next step is decided at runtime from state, rather than fixed at graph-build time.
- **Reducers** — `messages` uses LangGraph's `add_messages` reducer to append history; other fields overwrite by default.
- **Loop safety** — `revision_count` + `MAX_REVISIONS` guarantees termination of the writer↔reviewer loop.
- **Factory + Strategy** — `get_llm()` constructs the configured provider behind a single interface, with lazy imports so only the chosen SDK is required.

## 🗺️ Possible extensions

- **Human-in-the-loop** — pause before the Reviewer using LangGraph's `interrupt_before` + a checkpointer to let a human inject feedback.
- **Persistence** — add a checkpointer (e.g. `MemorySaver` or a database) to pause/resume runs and support multi-turn sessions.
- **Structured routing** — replace prose-based verdict parsing with structured output (function calling / Pydantic schemas) for more reliable control flow.
- **Parallel research** — fan out multiple search queries concurrently and merge results.
- **Test suite** — unit tests for nodes/routing and an integration test for the full graph.

---

## 📄 License

MIT
