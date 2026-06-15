"""Write a file into the project release directory.

The release directory lives at ``projects/<project_id>/release/`` — a sibling
of ``sandbox/``. This separation is intentional: the sandbox holds the
working dev tree (archives, partial runs, scratch scripts) while the release
holds the curated, publishable repository assembled by ``repo_publisher``.

Mirrors :func:`autoscientist.tools.write_file.write_file` but with a different
root and a different sandbox-escape check.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import structlog

log = structlog.get_logger("autoscientist.tools.write_release_file")


class ReleaseEscape(RuntimeError):
    """Raised when a path would resolve outside the release directory."""


def write_release_file(
    *,
    path: str,
    content: str,
    project_id: str,
    projects_root: Path | str,
) -> dict[str, object]:
    """Write ``content`` to ``path`` inside ``projects/<project_id>/release/``.

    ``path`` must be relative and must resolve within the release root.
    Parent directories are created. Returns a dict with ``written: True``,
    ``path``, and ``size_bytes``.
    """
    projects_root = Path(projects_root)
    release = projects_root / project_id / "release"
    release.mkdir(parents=True, exist_ok=True)

    rel = PurePosixPath(path)
    if rel.is_absolute():
        raise ReleaseEscape(f"path must be relative, got: {path}")
    dest = (release / rel).resolve()
    if not str(dest).startswith(str(release.resolve())):
        raise ReleaseEscape(f"path escapes release dir: {path}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    size = dest.stat().st_size

    log.info(
        "write_release_file.done",
        project_id=project_id,
        path=str(rel),
        size_bytes=size,
    )
    return {"written": True, "path": str(rel), "size_bytes": size}
