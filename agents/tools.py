"""Custom tools available to the research assistant agents."""

from __future__ import annotations

from html.parser import HTMLParser

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from agents.config import get_llm

# Cap fetched page text so a single source can't blow up the LLM context.
_MAX_CONTENT_CHARS = 2000
_FETCH_TIMEOUT_SECONDS = 5.0


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text: keep tag data, drop script/style and markup."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self._parts)


def _raw_search(query: str) -> list[dict]:
    """Return raw DuckDuckGo results (list of dicts with title/link/snippet)."""
    # Lazy import: the ddgs backend is only needed when the tool actually runs.
    from langchain_community.tools import DuckDuckGoSearchResults

    search = DuckDuckGoSearchResults(num_results=5, output_format="list")
    return search.invoke(query)


def _fetch_text(url: str) -> str:
    """Best-effort fetch of a page's visible text. Returns '' on any failure."""
    if not url:
        return ""
    try:
        import httpx

        # A browser-like UA avoids the blanket 403s many sites return to
        # default clients, which would otherwise silently gut our grounding.
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-assistant/1.0)"}
        resp = httpx.get(
            url, timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True, headers=headers
        )
        resp.raise_for_status()
        # Only parse HTML/text; binary (PDF, images) would decode to junk.
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return ""
        parser = _TextExtractor()
        parser.feed(resp.text)
        return parser.text()[:_MAX_CONTENT_CHARS]
    except Exception:
        # Network errors, timeouts, blocked pages, parse errors -- never fatal.
        # The caller falls back to the search snippet.
        return ""


@tool
def web_search(query: str) -> list[dict]:
    """Search the web (DuckDuckGo, no API key) and return grounded sources.

    Each source is a dict ``{title, url, snippet, content}`` where ``content`` is
    the fetched page text (falling back to the search snippet if the fetch fails).
    Returning structured sources — instead of a prose blob — lets downstream
    agents cite specific URLs and lets the pipeline refuse ungrounded answers.

    Parameters
    ----------
    query:
        The search query string.

    Returns
    -------
    list[dict]
        Up to five sources, each with ``title``, ``url``, ``snippet``, ``content``.
    """
    results = _raw_search(query)
    if not results:
        return []

    sources: list[dict] = []
    for result in results:
        url = result.get("link", "")
        snippet = result.get("snippet", "")
        content = _fetch_text(url) or snippet
        sources.append(
            {
                "title": result.get("title", "Untitled"),
                "url": url,
                "snippet": snippet,
                "content": content,
            }
        )
    return sources


@tool
def summarize(text: str) -> str:
    """Produce a concise summary of the provided text using the configured LLM.

    Parameters
    ----------
    text:
        The text to summarise.

    Returns
    -------
    str
        A concise summary.
    """
    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(
            content=(
                "You are a concise summariser. Distil the following text into "
                "its key points using no more than five bullet points."
            )
        ),
        HumanMessage(content=text),
    ]
    response = llm.invoke(messages)
    return response.content
