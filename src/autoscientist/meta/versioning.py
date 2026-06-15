"""Prompt version archive.

KICKOFF.md §9 Phase 6 (hard rule):
    *Version everything in git; never overwrite a prompt without saving
    the previous version.*

This module is the single seam through which prompts are mutated. Every
write archives the current ``prompts/<agent>.md`` to
``prompts/_versions/<agent>/<timestamp>_<slug>.md`` AND records a row in
the ``prompt_versions`` table before overwriting. The archive path lives
in the row so a future operator can locate the snapshot on disk even if
the DB is restored to a different filesystem.

Reads are still a plain ``Path.read_text()`` — only writes route through
``write_prompt``.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from autoscientist.state.db import new_id, now_iso

log = structlog.get_logger("autoscientist.meta.versioning")


@dataclass(frozen=True)
class PromptVersion:
    version_id: str
    agent_name: str
    prompt_text: str
    parent_version_id: str | None
    note: str | None
    archived_path: str | None
    created_at: str


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(s: str | None, default: str) -> str:
    base = (s or default).lower()
    base = _SLUG_RE.sub("-", base).strip("-")
    return (base or default)[:40]


def _ts_for_path() -> str:
    return now_iso().replace(":", "").replace("-", "").replace(".", "")[:15]


def archive_path(prompts_dir: Path, agent_name: str, note: str | None = None) -> Path:
    """Compute the archive path that ``write_prompt`` would use right now.

    Exposed so callers can pre-compute the location for logging without
    actually mutating anything.
    """
    return (
        prompts_dir / "_versions" / agent_name
        / f"{_ts_for_path()}_{_slug(note, 'archive')}.md"
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def latest_version(
    conn: sqlite3.Connection, agent_name: str,
) -> PromptVersion | None:
    # rowid tiebreaker — millisecond timestamps can collide under tight
    # write loops (tests, A/B harness) and pure created_at DESC then has
    # undefined order, returning a non-latest row.
    row = conn.execute(
        "SELECT * FROM prompt_versions WHERE agent_name = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (agent_name,),
    ).fetchone()
    return _row_to_version(row) if row else None


def list_versions(
    conn: sqlite3.Connection, agent_name: str,
) -> list[PromptVersion]:
    rows = conn.execute(
        "SELECT * FROM prompt_versions WHERE agent_name = ? "
        "ORDER BY created_at ASC, rowid ASC",
        (agent_name,),
    ).fetchall()
    return [_row_to_version(r) for r in rows]


def get_version(
    conn: sqlite3.Connection, version_id: str,
) -> PromptVersion | None:
    row = conn.execute(
        "SELECT * FROM prompt_versions WHERE version_id = ?", (version_id,),
    ).fetchone()
    return _row_to_version(row) if row else None


def write_prompt(
    conn: sqlite3.Connection,
    *,
    prompts_dir: Path,
    agent_name: str,
    new_text: str,
    note: str | None = None,
) -> PromptVersion:
    """Archive the current prompt, record the row, then overwrite.

    If no prompt exists yet at ``prompts/<agent>.md``, the very first
    write creates the file *and* records an initial ``prompt_versions``
    row with no parent and no archived snapshot (nothing to archive).

    No-op write detection: if ``new_text`` matches the current file
    byte-for-byte, this still records a new version row but does not
    write a duplicate archive snapshot. This keeps the DB authoritative
    on "I tried to write at ts=X" while avoiding archive churn.
    """
    target = prompts_dir / f"{agent_name}.md"
    parent: PromptVersion | None = latest_version(conn, agent_name)

    archived: Path | None = None
    if target.exists():
        current_text = target.read_text(encoding="utf-8")
        if current_text != new_text:
            archived = archive_path(prompts_dir, agent_name, note=note)
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_text(current_text, encoding="utf-8")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(new_text, encoding="utf-8")

    version_id = new_id("pv_")
    conn.execute(
        """INSERT INTO prompt_versions
           (version_id, agent_name, prompt_text, parent_version_id,
            note, archived_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            version_id, agent_name, new_text,
            parent.version_id if parent else None,
            note,
            str(archived) if archived else None,
            now_iso(),
        ),
    )
    log.info(
        "meta.versioning.wrote_prompt",
        agent=agent_name, version_id=version_id,
        parent=parent.version_id if parent else None,
        archived=str(archived) if archived else None,
        text_hash=_hash_text(new_text),
    )
    return PromptVersion(
        version_id=version_id,
        agent_name=agent_name,
        prompt_text=new_text,
        parent_version_id=parent.version_id if parent else None,
        note=note,
        archived_path=str(archived) if archived else None,
        created_at=now_iso(),
    )


def register_existing_prompt(
    conn: sqlite3.Connection,
    *,
    prompts_dir: Path,
    agent_name: str,
    note: str | None = "registered_existing",
) -> PromptVersion | None:
    """Record a row for the on-disk prompt without rewriting it.

    Used the first time the meta tooling is pointed at an agent whose
    prompt was authored outside the harness. Idempotent — if the latest
    version row already matches the file, returns it unchanged.
    """
    target = prompts_dir / f"{agent_name}.md"
    if not target.exists():
        return None
    text = target.read_text(encoding="utf-8")
    cur = latest_version(conn, agent_name)
    if cur is not None and cur.prompt_text == text:
        return cur
    version_id = new_id("pv_")
    conn.execute(
        """INSERT INTO prompt_versions
           (version_id, agent_name, prompt_text, parent_version_id,
            note, archived_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            version_id, agent_name, text,
            cur.version_id if cur else None,
            note, None, now_iso(),
        ),
    )
    return PromptVersion(
        version_id=version_id,
        agent_name=agent_name,
        prompt_text=text,
        parent_version_id=cur.version_id if cur else None,
        note=note,
        archived_path=None,
        created_at=now_iso(),
    )


def _row_to_version(row: sqlite3.Row) -> PromptVersion:
    return PromptVersion(
        version_id=row["version_id"],
        agent_name=row["agent_name"],
        prompt_text=row["prompt_text"],
        parent_version_id=row["parent_version_id"],
        note=row["note"],
        archived_path=row["archived_path"],
        created_at=row["created_at"],
    )


def to_dict(v: PromptVersion) -> dict[str, Any]:
    return {
        "version_id": v.version_id,
        "agent_name": v.agent_name,
        "parent_version_id": v.parent_version_id,
        "note": v.note,
        "archived_path": v.archived_path,
        "created_at": v.created_at,
        "text_hash": _hash_text(v.prompt_text),
    }
