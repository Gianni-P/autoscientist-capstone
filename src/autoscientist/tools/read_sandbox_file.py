"""Read a file from the project sandbox.

Counterpart to :mod:`autoscientist.tools.write_file`. ``repo_publisher`` uses
this to read the working code so it can rewrite/curate it into the release
directory. Returned text is truncated past 30k chars to keep tokens bounded.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import structlog

log = structlog.get_logger("autoscientist.tools.read_sandbox_file")

MAX_TEXT_BYTES = 30_000


class SandboxEscape(RuntimeError):
    """Raised when a path would resolve outside the sandbox."""


def read_sandbox_file(
    *,
    path: str,
    project_id: str,
    projects_root: Path | str,
) -> dict[str, object]:
    """Read ``path`` from ``projects/<project_id>/sandbox/``.

    Returns a dict with ``path``, ``content`` (text, possibly truncated),
    ``truncated`` flag, ``size_bytes``, and ``encoding``. Binary files
    surface as ``encoding="binary"`` with an empty ``content``.
    """
    projects_root = Path(projects_root)
    sandbox = projects_root / project_id / "sandbox"

    rel = PurePosixPath(path)
    if rel.is_absolute():
        raise SandboxEscape(f"path must be relative, got: {path}")
    src = (sandbox / rel).resolve()
    # Real path-boundary containment (not a string prefix, which would accept a
    # prefix-colliding sibling like ``../sandbox_x/secret``).
    if not src.is_relative_to(sandbox.resolve()):
        raise SandboxEscape(f"path escapes sandbox: {path}")
    if not src.exists():
        raise FileNotFoundError(f"no such file in sandbox: {path}")
    if not src.is_file():
        raise IsADirectoryError(f"not a file: {path}")

    raw = src.read_bytes()
    size = len(raw)
    try:
        text = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        log.info("read_sandbox_file.binary", project_id=project_id, path=str(rel), size_bytes=size)
        return {
            "path": str(rel),
            "content": "",
            "truncated": False,
            "size_bytes": size,
            "encoding": "binary",
        }

    truncated = len(text) > MAX_TEXT_BYTES
    if truncated:
        text = text[:MAX_TEXT_BYTES] + "\n…[truncated]"

    log.info(
        "read_sandbox_file.done",
        project_id=project_id,
        path=str(rel),
        size_bytes=size,
        truncated=truncated,
    )
    return {
        "path": str(rel),
        "content": text,
        "truncated": truncated,
        "size_bytes": size,
        "encoding": encoding,
    }
