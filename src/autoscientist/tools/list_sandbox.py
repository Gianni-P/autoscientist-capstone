"""List files inside the project sandbox.

Cheap directory tool so ``repo_publisher`` does not need to spawn a
subprocess for ``find``. Returns relative paths sorted lexicographically;
hidden files and the conventional ``__pycache__`` / ``.venv`` / ``.git``
trees are filtered out so the LLM does not chase noise.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import structlog

log = structlog.get_logger("autoscientist.tools.list_sandbox")

_SKIP_DIRS = frozenset({"__pycache__", ".venv", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
DEFAULT_MAX_ENTRIES = 500


class SandboxEscape(RuntimeError):
    """Raised when a subdir would resolve outside the sandbox."""


def list_sandbox(
    *,
    project_id: str,
    projects_root: Path | str,
    subdir: str = "",
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> dict[str, object]:
    """Walk the sandbox (or a subdir of it) and return relative file paths.

    Entries are returned as POSIX-style relative paths from the sandbox
    root. Directories themselves are not returned; only files. If the
    walk hits ``max_entries``, the result is marked ``truncated: True``.
    """
    projects_root = Path(projects_root)
    sandbox = projects_root / project_id / "sandbox"
    if not sandbox.exists():
        return {"root": str(sandbox), "entries": [], "count": 0, "truncated": False}

    rel_subdir = PurePosixPath(subdir or "")
    if rel_subdir.is_absolute():
        raise SandboxEscape(f"subdir must be relative, got: {subdir}")
    start = (sandbox / rel_subdir).resolve()
    # Real path-boundary containment (not a string prefix, which would accept a
    # prefix-colliding sibling like ``../sandbox_x``).
    if not start.is_relative_to(sandbox.resolve()):
        raise SandboxEscape(f"subdir escapes sandbox: {subdir}")
    if not start.exists():
        return {"root": str(start), "entries": [], "count": 0, "truncated": False}

    entries: list[dict[str, object]] = []
    truncated = False
    sandbox_resolved = sandbox.resolve()
    for path in sorted(start.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(sandbox_resolved).parts):
            continue
        rel = path.relative_to(sandbox_resolved).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        entries.append({"path": rel, "size_bytes": size})
        if len(entries) >= max_entries:
            truncated = True
            break

    log.info(
        "list_sandbox.done",
        project_id=project_id,
        subdir=str(rel_subdir),
        count=len(entries),
        truncated=truncated,
    )
    return {
        "root": str(start),
        "entries": entries,
        "count": len(entries),
        "truncated": truncated,
    }
