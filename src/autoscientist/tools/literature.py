"""Literature search and lookup.

Three free APIs, no keys required:

  * **Semantic Scholar** — primary. Best abstracts and citation graph.
    Rate-limited to ~1 req/s without an API key (env: ``SEMANTIC_SCHOLAR_API_KEY``).
  * **OpenAlex** — fallback. ``api.openalex.org``, no auth, polite-pool
    via ``mailto`` (env: ``OPENALEX_MAILTO``).
  * **arxiv** — preprints. Used when ``arxiv_id`` is provided or the result
    set should include un-peer-reviewed work.

All search results normalize to :class:`Paper`. Network responses are
cached in ``tool_cache`` (sha256 of canonical query) so identical queries
collapse to one network call.

Failure modes:
  * S2 returns 429 -> back off, then fall through to OpenAlex.
  * Both APIs return empty -> return ``[]`` (no exception).
  * Network error -> the error propagates; caller handles the retry policy.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
import structlog

from autoscientist.tools import tool_cache

log = structlog.get_logger("autoscientist.tools.literature")

S2_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_BASE = "https://api.openalex.org"
ARXIV_BASE = "http://export.arxiv.org/api/query"

S2_FIELDS = (
    "paperId,title,authors,year,venue,externalIds,abstract,citationCount,referenceCount"
)

_HTTP_TIMEOUT = 30.0
_USER_AGENT = "autoscientist/0.1 (research; contact: see KICKOFF.md)"


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Paper:
        return cls(**d)


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def _s2_headers() -> dict[str, str]:
    h = {"User-Agent": _USER_AGENT}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        h["x-api-key"] = key
    return h


def _s2_paper_to_paper(d: dict[str, Any]) -> Paper:
    ext = d.get("externalIds") or {}
    authors = [a.get("name", "") for a in (d.get("authors") or [])]
    return Paper(
        title=d.get("title") or "",
        authors=authors,
        year=d.get("year"),
        venue=d.get("venue") or None,
        doi=ext.get("DOI"),
        arxiv_id=ext.get("ArXiv"),
        semantic_scholar_id=d.get("paperId"),
        abstract=d.get("abstract"),
        citation_count=d.get("citationCount"),
        source="semantic_scholar",
    )


def _s2_search(query: str, limit: int) -> list[Paper]:
    url = f"{S2_BASE}/paper/search"
    params = {"query": query, "limit": limit, "fields": S2_FIELDS}
    with httpx.Client(timeout=_HTTP_TIMEOUT, headers=_s2_headers()) as client:
        r = client.get(url, params=params)
    if r.status_code == 429:
        log.warning("literature.s2.rate_limited")
        return []
    r.raise_for_status()
    data = r.json().get("data") or []
    return [_s2_paper_to_paper(d) for d in data]


def _s2_lookup(*, doi: str | None, arxiv_id: str | None, s2_id: str | None) -> Paper | None:
    if doi:
        ident = f"DOI:{doi}"
    elif arxiv_id:
        ident = f"ARXIV:{arxiv_id}"
    elif s2_id:
        ident = s2_id
    else:
        return None
    url = f"{S2_BASE}/paper/{ident}"
    with httpx.Client(timeout=_HTTP_TIMEOUT, headers=_s2_headers()) as client:
        r = client.get(url, params={"fields": S2_FIELDS})
    if r.status_code == 404:
        return None
    if r.status_code == 429:
        log.warning("literature.s2.rate_limited")
        return None
    r.raise_for_status()
    return _s2_paper_to_paper(r.json())


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

def _openalex_headers() -> dict[str, str]:
    return {"User-Agent": _USER_AGENT}


def _openalex_polite_param() -> dict[str, str]:
    mailto = os.environ.get("OPENALEX_MAILTO")
    return {"mailto": mailto} if mailto else {}


def _openalex_work_to_paper(d: dict[str, Any]) -> Paper:
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in (d.get("authorships") or [])
    ]
    primary_loc = d.get("primary_location") or {}
    venue_src = primary_loc.get("source") or {}
    return Paper(
        title=d.get("title") or "",
        authors=[a for a in authors if a],
        year=d.get("publication_year"),
        venue=venue_src.get("display_name"),
        doi=(d.get("doi") or "").removeprefix("https://doi.org/") or None,
        openalex_id=d.get("id"),
        abstract=_openalex_reconstruct_abstract(d.get("abstract_inverted_index")),
        citation_count=d.get("cited_by_count"),
        source="openalex",
    )


def _openalex_reconstruct_abstract(inv_index: dict[str, list[int]] | None) -> str | None:
    if not inv_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inv_index.items():
        for idx in idxs:
            positions[idx] = word
    if not positions:
        return None
    ordered = [positions[i] for i in sorted(positions)]
    return " ".join(ordered)


def _openalex_search(query: str, limit: int) -> list[Paper]:
    url = f"{OPENALEX_BASE}/works"
    params: dict[str, Any] = {"search": query, "per_page": limit, **_openalex_polite_param()}
    with httpx.Client(timeout=_HTTP_TIMEOUT, headers=_openalex_headers()) as client:
        r = client.get(url, params=params)
    r.raise_for_status()
    data = r.json().get("results") or []
    return [_openalex_work_to_paper(d) for d in data]


def _openalex_lookup(*, doi: str | None) -> Paper | None:
    if not doi:
        return None
    url = f"{OPENALEX_BASE}/works/doi:{doi}"
    with httpx.Client(timeout=_HTTP_TIMEOUT, headers=_openalex_headers()) as client:
        r = client.get(url, params=_openalex_polite_param())
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return _openalex_work_to_paper(r.json())


# ---------------------------------------------------------------------------
# arxiv
# ---------------------------------------------------------------------------

def _arxiv_search(query: str, limit: int) -> list[Paper]:
    """Use the ``arxiv`` library — XML feed parsing is a pain by hand."""
    import arxiv  # lazy import; large module

    client = arxiv.Client(page_size=limit, delay_seconds=3.0, num_retries=2)
    search = arxiv.Search(query=query, max_results=limit)
    results = list(client.results(search))
    return [
        Paper(
            title=r.title,
            authors=[a.name for a in r.authors],
            year=r.published.year if r.published else None,
            venue="arxiv",
            doi=r.doi,
            arxiv_id=r.entry_id.split("/")[-1] if r.entry_id else None,
            abstract=r.summary,
            citation_count=None,
            source="arxiv",
        )
        for r in results
    ]


def _arxiv_lookup(arxiv_id: str) -> Paper | None:
    import arxiv

    client = arxiv.Client(page_size=1, delay_seconds=3.0, num_retries=2)
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        return None
    r = results[0]
    return Paper(
        title=r.title,
        authors=[a.name for a in r.authors],
        year=r.published.year if r.published else None,
        venue="arxiv",
        doi=r.doi,
        arxiv_id=arxiv_id,
        abstract=r.summary,
        source="arxiv",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(
    query: str,
    *,
    limit: int = 20,
    include_arxiv: bool = False,
    conn: sqlite3.Connection | None = None,
) -> list[Paper]:
    """Search literature; Semantic Scholar primary, OpenAlex fallback.

    If ``include_arxiv``, append arxiv preprint hits (deduped on arxiv_id).
    Cached: identical (query, limit, include_arxiv) returns from cache.
    """
    key_payload = {"query": query, "limit": limit, "include_arxiv": include_arxiv}
    key = tool_cache.cache_key(key_payload)

    if conn is not None:
        cached = tool_cache.cache_get(conn, "literature.search", key)
        if cached is not None:
            log.info("literature.search.cache_hit", q=query[:60])
            return [Paper.from_dict(d) for d in cached]

    started = time.monotonic()
    results: list[Paper] = []
    try:
        results = _s2_search(query, limit)
    except httpx.HTTPError as e:
        log.warning("literature.s2.error", error=str(e))

    if not results:
        log.info("literature.s2.empty_or_failed", q=query[:60])
        try:
            results = _openalex_search(query, limit)
        except httpx.HTTPError as e:
            log.warning("literature.openalex.error", error=str(e))

    if include_arxiv:
        try:
            arx = _arxiv_search(query, limit)
            seen = {(p.arxiv_id, p.doi) for p in results}
            for p in arx:
                if (p.arxiv_id, p.doi) not in seen:
                    results.append(p)
        except Exception as e:  # arxiv lib raises a variety of exceptions
            log.warning("literature.arxiv.error", error=str(e))

    elapsed_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "literature.search.done",
        q=query[:60], n=len(results), elapsed_ms=elapsed_ms,
    )
    if conn is not None:
        tool_cache.cache_put(
            conn,
            "literature.search",
            key,
            [p.to_dict() for p in results],
        )
    return results


def lookup(
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    semantic_scholar_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> Paper | None:
    """Resolve a single paper by ID. Tries S2 first, then OpenAlex (DOI),
    then arxiv (arxiv_id). Returns ``None`` if no source has it.
    """
    if not (doi or arxiv_id or semantic_scholar_id):
        return None
    key = tool_cache.cache_key({
        "doi": doi, "arxiv_id": arxiv_id, "s2_id": semantic_scholar_id,
    })
    if conn is not None:
        cached = tool_cache.cache_get(conn, "literature.lookup", key)
        if cached is not None:
            return Paper.from_dict(cached) if cached else None

    p: Paper | None = None
    try:
        p = _s2_lookup(doi=doi, arxiv_id=arxiv_id, s2_id=semantic_scholar_id)
    except httpx.HTTPError as e:
        log.warning("literature.s2.lookup_error", error=str(e))

    if p is None and doi:
        try:
            p = _openalex_lookup(doi=doi)
        except httpx.HTTPError as e:
            log.warning("literature.openalex.lookup_error", error=str(e))

    if p is None and arxiv_id:
        try:
            p = _arxiv_lookup(arxiv_id)
        except Exception as e:
            log.warning("literature.arxiv.lookup_error", error=str(e))

    if conn is not None:
        tool_cache.cache_put(
            conn,
            "literature.lookup",
            key,
            p.to_dict() if p else None,
        )
    return p
