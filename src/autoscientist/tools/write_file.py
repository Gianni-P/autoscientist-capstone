"""Write a file into the project sandbox.

Thin wrapper that ensures paths stay within the sandbox, creates
parent directories, and returns a confirmation dict for the tool loop.
Agents that synthesize code (``code_gen``, ``test_gen``) use this to
persist files one at a time instead of emitting them in a monolithic
JSON blob — breaking the task into manageable steps and ensuring each
file is durably written before the next is planned.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import structlog

log = structlog.get_logger("autoscientist.tools.write_file")


class SandboxEscape(RuntimeError):
    """Raised when a path would resolve outside the sandbox."""


def write_file(
    *,
    path: str,
    content: str,
    project_id: str,
    projects_root: Path | str,
) -> dict[str, object]:
    """Write ``content`` to ``path`` inside the project sandbox.

    ``path`` must be relative and must resolve within
    ``projects/<project_id>/sandbox/``. Parent directories are created.

    Returns a dict with ``written: True``, ``path``, and ``size_bytes``.
    """
    projects_root = Path(projects_root)
    sandbox = projects_root / project_id / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)

    # Normalise and reject escapes.
    rel = PurePosixPath(path)
    if rel.is_absolute():
        raise SandboxEscape(f"path must be relative, got: {path}")
    sandbox_root = sandbox.resolve()
    dest = (sandbox / rel).resolve()
    # Real path-boundary containment. A string-prefix check (startswith) would
    # also accept a SIBLING whose name extends the sandbox prefix — e.g.
    # ``../sandbox_evil/x`` resolves to ``…/sandbox_evil/x`` which *string*-starts
    # with ``…/sandbox`` yet is outside the sandbox dir. is_relative_to enforces
    # a true ancestor relationship on path components.
    if not dest.is_relative_to(sandbox_root):
        raise SandboxEscape(f"path escapes sandbox: {path}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    size = dest.stat().st_size

    log.info(
        "write_file.done",
        project_id=project_id,
        path=str(rel),
        size_bytes=size,
    )
    return {"written": True, "path": str(rel), "size_bytes": size}
