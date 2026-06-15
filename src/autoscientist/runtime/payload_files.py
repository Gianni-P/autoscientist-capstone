"""Safety-net file persister for agent handoff payloads.

Some agents (historically ``code_gen`` on Qwen) emit a
``files: [{path, content}, ...]`` JSON array in their handoff payload
instead of (or in addition to) calling the ``write_file`` tool per file.
The runner uses this module to persist those files to the sandbox
post-hoc, so the next agent sees them on disk.

The primary contract is still the ``write_file`` tool — see
``prompts/code_gen.md``. This module exists so a single regression in
an LLM's structured output doesn't stall the entire pipeline. Every
payload-write is logged with structlog under
``runtime.payload_files`` so operators can see when the fallback fires.

Design notes
~~~~~~~~~~~~

* **Best-effort.** Parse failures, missing keys, sandbox-escape errors,
  and write errors are captured and logged, never raised. The run loop
  must always make progress.
* **Targets a specific shape.** Only persists entries that are dicts
  with both ``path`` (non-empty string) and ``content`` (str-coercible).
  This deliberately excludes ``files_written: [...]`` (the path-only
  summary list that the contract *does* allow in payloads).
* **Idempotent overwrite.** If a file already exists with the same name,
  ``write_file`` overwrites it. That matches the agent's intent — the
  payload represents what the agent *wants* on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from autoscientist.tools import write_file as wf

log = structlog.get_logger("autoscientist.runtime.payload_files")


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Return the first balanced top-level JSON object in ``text`` as a dict.

    Returns ``None`` if no balanced object is present or the object fails
    to parse. Mirrors the logic in ``runtime/runner._maybe_parse_json``
    so callers can rely on the same parser for both file-extraction and
    checkpoint payload preview.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    parsed = json.loads(blob)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def persist_files_from_payload(
    *,
    payload: str,
    project_id: str,
    projects_root: Path | str,
    agent_name: str,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Persist any ``files: [{path, content}]`` array embedded in ``payload``.

    Returns a list of result dicts (one per attempted entry) so the runner
    can log a single summary line. Each result has ``path``, ``status``
    (``"ok"`` / ``"error"`` / ``"skipped"``), and either ``size_bytes`` or
    ``error``. Never raises.
    """
    parsed = _extract_first_json_object(payload)
    if parsed is None:
        return []
    files = parsed.get("files")
    if not isinstance(files, list) or not files:
        return []

    projects_root = Path(projects_root)
    written: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            written.append({"path": None, "status": "skipped",
                            "reason": "entry not a dict"})
            continue
        path = entry.get("path")
        content = entry.get("content")
        if not isinstance(path, str) or not path:
            written.append({"path": path, "status": "skipped",
                            "reason": "missing or non-string path"})
            continue
        if content is None:
            written.append({"path": path, "status": "skipped",
                            "reason": "missing content"})
            continue
        try:
            res = wf.write_file(
                path=path,
                content=str(content),
                project_id=project_id,
                projects_root=projects_root,
            )
            written.append({
                "path": path,
                "status": "ok",
                "size_bytes": res["size_bytes"],
            })
            log.warning(
                "payload_files.persisted",
                agent=agent_name,
                run_id=run_id,
                project_id=project_id,
                path=path,
                size_bytes=res["size_bytes"],
                note="fallback path - agent should call write_file directly",
            )
        except Exception as e:
            written.append({
                "path": path,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            })
            log.error(
                "payload_files.error",
                agent=agent_name,
                run_id=run_id,
                project_id=project_id,
                path=path,
                error=str(e),
                error_type=type(e).__name__,
            )
    return written


# ---------------------------------------------------------------------------
# Reverse direction: rebuild a code_review input payload FROM the sandbox.
# ---------------------------------------------------------------------------

# Skip pathologically large files so one stray artifact can't blow the
# review context; the head is kept with a truncation marker.
_MAX_REVIEW_FILE_BYTES = 100_000


def _collect_py_files(directory: Path, sandbox_root: Path) -> list[dict[str, str]]:
    """Return ``[{path, content}, ...]`` for every ``*.py`` under ``directory``.

    ``path`` is relative to ``sandbox_root`` (e.g. ``src/train.py``) to match
    the ``code_review`` input contract. ``__pycache__`` and unreadable files
    are skipped; over-large files are head-truncated with a marker. Sorted for
    deterministic output. Never raises.
    """
    if not directory.is_dir():
        return []
    out: list[dict[str, str]] = []
    for p in sorted(directory.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            rel = p.relative_to(sandbox_root).as_posix()
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(text.encode("utf-8", errors="replace")) > _MAX_REVIEW_FILE_BYTES:
            text = (
                text[:_MAX_REVIEW_FILE_BYTES]
                + f"\n# … truncated by runner (file > {_MAX_REVIEW_FILE_BYTES} bytes) …\n"
            )
        out.append({"path": rel, "content": text})
    return out


def build_code_review_payload_from_sandbox(
    *,
    project_id: str,
    projects_root: Path | str,
) -> str | None:
    """Reconstruct a ``code_review`` input payload from the sandbox on disk.

    ``code_review`` has no file-reading tool (its ``execute`` allowlist is
    pytest/python only), so it depends entirely on receiving the source and
    test file *contents* in its inbound payload. When an upstream agent
    (``code_gen``/``test_gen``) exhausts its tool rounds and hands off with
    empty content, the runner would otherwise feed ``code_review`` the
    ``"(no payload)"`` sentinel; the review then becomes a no-op (it just asks
    for the missing input) and a degenerate CP3 opens carrying that complaint
    (observed 2026-06-12, run_5273a6fe…). This helper rebuilds the
    ``{src_files, test_files, run_cmd_src, run_cmd_tests}`` envelope from the
    sandbox ``src/`` and ``tests/`` trees so the review can proceed against the
    real code/tests.

    Returns the JSON payload string, or ``None`` when the sandbox has no
    reviewable ``.py`` files (the caller keeps its existing fallback).
    Never raises.
    """
    try:
        sandbox = Path(projects_root) / project_id / "sandbox"
        src_files = _collect_py_files(sandbox / "src", sandbox)
        test_files = _collect_py_files(sandbox / "tests", sandbox)
    except Exception as e:  # pragma: no cover - defensive
        log.error(
            "payload_files.review_rebuild_error",
            project_id=project_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
    if not src_files and not test_files:
        return None
    payload = {
        "src_files": src_files,
        "test_files": test_files,
        "run_cmd_src": "python src/main.py",
        "run_cmd_tests": "pytest tests/ -x -q",
        "_reconstructed_by_runner": (
            "The upstream agent handed off with empty content; the runner "
            "rebuilt this payload from the sandbox on disk. Review the files "
            "below as usual and emit your verdict + HANDOFF."
        ),
    }
    return json.dumps(payload, indent=2)
