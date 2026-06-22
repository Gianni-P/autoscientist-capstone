"""LaTeX → PDF compilation via tectonic.

We invoke the ``tectonic`` binary as a subprocess. tectonic is single-file
and self-bootstrapping (it downloads needed packages on first use into a
local cache), so the operator doesn't have to install TeXLive.

Operator install:
  * cargo: ``cargo install tectonic``
  * apt:   ``sudo apt install tectonic`` (might be old; cargo is safer)
  * brew:  ``brew install tectonic``
  * binary: github.com/tectonic-typesetting/tectonic/releases

The compile is wrapped in a sha256-keyed cache: identical .tex sources
collapse to one compile, with the resulting PDF copied out of the cache
on subsequent calls. The cache stores the PDF as base64 in tool_cache.
"""

from __future__ import annotations

import base64
import hashlib
import shutil
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

from autoscientist.tools import tool_cache

log = structlog.get_logger("autoscientist.tools.latex")


class TectonicMissingError(RuntimeError):
    """Raised when tectonic is not on PATH."""


# Backwards-compatible alias.
TectonicMissing = TectonicMissingError


@dataclass
class LatexBuild:
    success: bool
    pdf_path: str | None
    log: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_available() -> bool:
    return shutil.which("tectonic") is not None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_sha(tex_source: str, cache_extra: str | None) -> str:
    """Cache key for a compile. Mixes ``cache_extra`` (e.g. a digest of the
    figure bytes) into the .tex hash so a figure-only change — byte-identical
    LaTeX referencing the same figure filenames but with different image bytes —
    invalidates the cached PDF instead of shipping a stale render."""
    material = tex_source if not cache_extra else f"{tex_source}\n\x00CACHE_EXTRA\x00{cache_extra}"
    return _hash_text(material)


def _parse_tectonic_log(log_text: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for line in log_text.splitlines():
        s = line.strip()
        # tectonic emits errors like "error: ..." and warnings as "warning: ..."
        # Underlying tex emits "! Undefined control sequence." etc.
        low = s.lower()
        if low.startswith("error:") or s.startswith("!"):
            errors.append(s)
        elif low.startswith("warning:") or "warning:" in low:
            warnings.append(s)
    return errors, warnings


def compile_latex(
    tex_source: str,
    *,
    output_dir: Path | str,
    job_name: str = "paper",
    conn: sqlite3.Connection | None = None,
    keep_intermediates: bool = False,
    cache_extra: str | None = None,
) -> LatexBuild:
    """Compile ``tex_source`` to PDF under ``output_dir``.

    Caches by sha256(tex_source [+ ``cache_extra``]). Pass ``cache_extra`` (a
    digest of any non-.tex inputs baked into the PDF, e.g. the figure image
    bytes) so a change there invalidates the cache. On hit, decodes and writes
    the cached PDF to ``output_dir / job_name.pdf`` and returns immediately.

    Raises :class:`TectonicMissing` if tectonic is not installed.
    """
    if not is_available():
        raise TectonicMissingError(
            "tectonic not on PATH. Install via "
            "`cargo install tectonic` or download from "
            "github.com/tectonic-typesetting/tectonic/releases"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = output_dir / f"{job_name}.pdf"

    sha = _cache_sha(tex_source, cache_extra)
    if conn is not None:
        cached = tool_cache.cache_get(conn, "latex.compile", sha)
        if cached is not None and cached.get("success"):
            pdf_b64 = cached.get("pdf_b64")
            if pdf_b64:
                pdf_out.write_bytes(base64.b64decode(pdf_b64))
                log.info("latex.cache_hit", sha=sha[:12])
                build = LatexBuild(**{k: v for k, v in cached.items() if k != "pdf_b64"})
                build.pdf_path = str(pdf_out)
                return build

    tex_path = output_dir / f"{job_name}.tex"
    tex_path.write_text(tex_source, encoding="utf-8")

    import time
    started = time.monotonic()

    cmd = [
        "tectonic",
        "--outdir", str(output_dir),
        "--keep-intermediates" if keep_intermediates else "--keep-logs",
        "--chatter", "minimal",
        str(tex_path),
    ]
    log.info("latex.compile.start", sha=sha[:12], cmd=cmd)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    log_text = (proc.stdout or "") + (proc.stderr or "")
    errors, warnings = _parse_tectonic_log(log_text)
    success = proc.returncode == 0 and pdf_out.exists()

    build = LatexBuild(
        success=success,
        pdf_path=str(pdf_out) if success else None,
        log=log_text,
        errors=errors,
        warnings=warnings,
        duration_ms=elapsed_ms,
    )

    if conn is not None and success:
        pdf_b64 = base64.b64encode(pdf_out.read_bytes()).decode("ascii")
        cache_payload = build.to_dict()
        cache_payload["pdf_b64"] = pdf_b64
        tool_cache.cache_put(conn, "latex.compile", sha, cache_payload)

    log.info(
        "latex.compile.done",
        sha=sha[:12],
        success=success,
        n_errors=len(errors),
        n_warnings=len(warnings),
        duration_ms=elapsed_ms,
    )
    return build
