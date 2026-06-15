"""Unit tests for autoscientist.meta.versioning."""

from __future__ import annotations

from autoscientist.meta import versioning
from autoscientist.state.db import open_db


def _setup(tmp_path):
    db_path = tmp_path / "test.db"
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    return open_db(db_path), prompts


def test_first_write_creates_file_and_row(tmp_path):
    conn, prompts = _setup(tmp_path)
    try:
        v = versioning.write_prompt(
            conn, prompts_dir=prompts, agent_name="x",
            new_text="hello\n", note="initial",
        )
        conn.commit()
        assert v.parent_version_id is None
        assert v.archived_path is None
        assert (prompts / "x.md").read_text(encoding="utf-8") == "hello\n"
        rows = versioning.list_versions(conn, "x")
        assert len(rows) == 1
        assert rows[0].version_id == v.version_id
    finally:
        conn.close()


def test_overwrite_archives_previous_text(tmp_path):
    conn, prompts = _setup(tmp_path)
    try:
        v1 = versioning.write_prompt(
            conn, prompts_dir=prompts, agent_name="x",
            new_text="v1 text\n", note="first",
        )
        v2 = versioning.write_prompt(
            conn, prompts_dir=prompts, agent_name="x",
            new_text="v2 text\n", note="second",
        )
        conn.commit()
        assert v2.parent_version_id == v1.version_id
        assert v2.archived_path is not None
        from pathlib import Path
        assert Path(v2.archived_path).read_text(encoding="utf-8") == "v1 text\n"
        assert (prompts / "x.md").read_text(encoding="utf-8") == "v2 text\n"
    finally:
        conn.close()


def test_identical_rewrite_skips_archive(tmp_path):
    conn, prompts = _setup(tmp_path)
    try:
        versioning.write_prompt(
            conn, prompts_dir=prompts, agent_name="x", new_text="same\n",
        )
        v = versioning.write_prompt(
            conn, prompts_dir=prompts, agent_name="x", new_text="same\n",
        )
        conn.commit()
        assert v.archived_path is None
        # Both versions still recorded.
        assert len(versioning.list_versions(conn, "x")) == 2
    finally:
        conn.close()


def test_register_existing_prompt_idempotent(tmp_path):
    conn, prompts = _setup(tmp_path)
    try:
        (prompts / "x.md").write_text("preexisting\n", encoding="utf-8")
        v1 = versioning.register_existing_prompt(
            conn, prompts_dir=prompts, agent_name="x",
        )
        assert v1 is not None
        v2 = versioning.register_existing_prompt(
            conn, prompts_dir=prompts, agent_name="x",
        )
        # Same content → same row returned, no new insertion.
        assert v2 is not None
        assert v2.version_id == v1.version_id
        assert len(versioning.list_versions(conn, "x")) == 1
    finally:
        conn.close()


def test_register_existing_returns_none_when_file_missing(tmp_path):
    conn, prompts = _setup(tmp_path)
    try:
        v = versioning.register_existing_prompt(
            conn, prompts_dir=prompts, agent_name="absent",
        )
        assert v is None
    finally:
        conn.close()


def test_latest_version_returns_most_recent(tmp_path):
    conn, prompts = _setup(tmp_path)
    try:
        versioning.write_prompt(conn, prompts_dir=prompts, agent_name="x", new_text="a\n")
        versioning.write_prompt(conn, prompts_dir=prompts, agent_name="x", new_text="b\n")
        v3 = versioning.write_prompt(conn, prompts_dir=prompts, agent_name="x", new_text="c\n")
        latest = versioning.latest_version(conn, "x")
        assert latest is not None
        assert latest.version_id == v3.version_id
        assert latest.prompt_text == "c\n"
    finally:
        conn.close()
