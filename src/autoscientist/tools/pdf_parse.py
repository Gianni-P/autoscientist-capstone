"""PDF parsing — pypdf-based extraction with sha256-keyed cache.

Tier 1: ``pypdf`` for simple layouts (most academic PDFs work fine).
Tier 2 (deferred to v1.x): ``marker-pdf`` for complex layouts. Marker
pulls torch and is heavyweight; we'll add it when we hit a paper that
pypdf garbles.

Cache: keyed on the SHA256 of the PDF file's bytes, so re-parsing the
same file is free even if the path changes.
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog
from pypdf import PdfReader

from autoscientist.tools import tool_cache

log = structlog.get_logger("autoscientist.tools.pdf_parse")


@dataclass
class PdfDocument:
    sha256: str
    page_count: int
    text: str
    pages: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PdfDocument:
        return cls(**d)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pdf(path: str | Path, *, conn: sqlite3.Connection | None = None) -> PdfDocument:
    """Parse a PDF file. Idempotent and cached.

    Raises FileNotFoundError if path does not exist; pypdf errors propagate.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    sha = _hash_file(p)
    if conn is not None:
        cached = tool_cache.cache_get(conn, "pdf_parse.parse", sha)
        if cached is not None:
            log.info("pdf_parse.cache_hit", sha=sha[:12], path=str(p))
            return PdfDocument.from_dict(cached)

    reader = PdfReader(str(p))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            log.warning("pdf_parse.page_failed", page=i, error=str(e))
            pages.append("")

    metadata: dict[str, Any] = {}
    if reader.metadata:
        for k, v in reader.metadata.items():
            with contextlib.suppress(Exception):
                metadata[str(k)] = str(v) if v is not None else None

    doc = PdfDocument(
        sha256=sha,
        page_count=len(reader.pages),
        text="\n\n".join(pages),
        pages=pages,
        metadata=metadata,
    )

    if conn is not None:
        tool_cache.cache_put(conn, "pdf_parse.parse", sha, doc.to_dict())

    log.info("pdf_parse.done", sha=sha[:12], pages=doc.page_count, chars=len(doc.text))
    return doc
