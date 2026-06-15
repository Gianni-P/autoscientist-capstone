"""Citation verification — round-trip every cited work through the
literature module to confirm the paper exists and the claimed metadata
is at least approximately correct.

Per KICKOFF.md §4 #8 and §10:
  *Hallucinated citations are the single fastest way for AI-generated
   work to get caught and rejected.*

Algorithm:
  1. If the citation has a DOI → ``literature.lookup(doi=...)``.
  2. Else if it has an arxiv_id → ``literature.lookup(arxiv_id=...)``.
  3. Else fall back to ``literature.search(title)`` and pick the top
     result with a high title-similarity score.
  4. Compare title (Jaccard on token sets), first author surname, and
     year against the resolved Paper. Mismatches are recorded but a
     citation is considered ``verified`` only if all three match within
     tolerance.

Citations that fail verification are emitted with ``verified=False``;
the paper_writer agent's hard rule replaces them with
``[CITATION NEEDED]`` before final output.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from autoscientist.tools import literature
from autoscientist.tools.literature import Paper

log = structlog.get_logger("autoscientist.tools.citation_check")


@dataclass
class CitationCheck:
    citation_key: str
    verified: bool
    matched_paper: dict[str, Any] | None = None
    mismatch_reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0..1, max(title_sim, exact_id_match)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TOKEN_RE = re.compile(r"\w+")


def _tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    return {t.lower() for t in _TOKEN_RE.findall(s) if len(t) > 2}


def _title_similarity(a: str | None, b: str | None) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _first_author_surname(authors: list[str] | None) -> str | None:
    if not authors:
        return None
    first = authors[0].strip()
    if not first:
        return None
    # "Smith, John" or "John Smith"
    if "," in first:
        return first.split(",", 1)[0].strip().lower()
    parts = first.split()
    return parts[-1].lower() if parts else None


def verify_citation(
    citation: dict[str, Any],
    *,
    title_threshold: float = 0.6,
    year_tolerance: int = 1,
    conn: sqlite3.Connection | None = None,
) -> CitationCheck:
    """Verify a single citation.

    Expected ``citation`` shape (matches paper_writer prompt output):

        {
          "key": "Rajpurkar2017",
          "title": "...",
          "authors": ["..."],
          "year": 2017,
          "venue": "...",
          "doi_or_arxiv": "10.xxxx/yyyy",  # optional
        }
    """
    key = citation.get("key", "<unkeyed>")
    title = citation.get("title")
    authors = citation.get("authors") or []
    year = citation.get("year")
    ident = citation.get("doi_or_arxiv") or ""

    doi = ident if "/" in ident and not ident.lower().startswith("arxiv:") else None
    arxiv_id = None
    if ident.lower().startswith("arxiv:"):
        arxiv_id = ident.split(":", 1)[1]
    elif re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", ident):
        arxiv_id = ident

    matched: Paper | None = None
    confidence = 0.0
    reasons: list[str] = []

    # 1. ID lookup if available.
    if doi or arxiv_id:
        matched = literature.lookup(doi=doi, arxiv_id=arxiv_id, conn=conn)
        if matched is not None:
            confidence = 1.0
        else:
            reasons.append(f"id_lookup_failed:{ident}")

    # 2. Title fallback.
    if matched is None and title:
        results = literature.search(title, limit=5, conn=conn)
        best: tuple[float, Paper] | None = None
        for p in results:
            sim = _title_similarity(title, p.title)
            if best is None or sim > best[0]:
                best = (sim, p)
        if best and best[0] >= title_threshold:
            matched = best[1]
            confidence = best[0]
        elif best:
            reasons.append(
                f"title_below_threshold:best_sim={best[0]:.2f}<{title_threshold}"
            )
        else:
            reasons.append("no_title_search_results")

    if matched is None:
        return CitationCheck(
            citation_key=key,
            verified=False,
            matched_paper=None,
            mismatch_reasons=reasons or ["no_match"],
            confidence=0.0,
        )

    # 3. Cross-check fields against the matched paper.
    if title and _title_similarity(title, matched.title) < title_threshold:
        reasons.append(
            f"title_mismatch:claimed={title!r},found={matched.title!r}"
        )
    claimed_surname = _first_author_surname(authors)
    found_surname = _first_author_surname(matched.authors)
    if (
        claimed_surname
        and found_surname
        and claimed_surname != found_surname
        # Allow surname substring (e.g. "Wang" vs "Wang-Chen")
        and claimed_surname not in found_surname
        and found_surname not in claimed_surname
    ):
        reasons.append(
            f"first_author_mismatch:claimed={claimed_surname},found={found_surname}"
        )
    if (
        year is not None
        and matched.year is not None
        and abs(int(year) - int(matched.year)) > year_tolerance
    ):
        reasons.append(
            f"year_mismatch:claimed={year},found={matched.year}"
        )

    verified = len(reasons) == 0
    return CitationCheck(
        citation_key=key,
        verified=verified,
        matched_paper=matched.to_dict(),
        mismatch_reasons=reasons,
        confidence=confidence,
    )


def verify_citations(
    citations: list[dict[str, Any]],
    *,
    conn: sqlite3.Connection | None = None,
) -> list[CitationCheck]:
    return [verify_citation(c, conn=conn) for c in citations]
