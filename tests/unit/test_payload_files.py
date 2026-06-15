"""Tests for the runner-level files: [...] safety-net persister.

Covers payload_files.persist_files_from_payload and the runner integration
shape (smoke-level: feed the mock code_gen fixture and confirm files land
in the sandbox).
"""

from __future__ import annotations

import json
from pathlib import Path

from autoscientist.clients import mock as mock_client
from autoscientist.runtime.payload_files import (
    _extract_first_json_object,
    persist_files_from_payload,
)


# ---------------------------------------------------------------------------
# _extract_first_json_object
# ---------------------------------------------------------------------------

def test_extract_first_json_object_simple() -> None:
    out = _extract_first_json_object('preface {"a": 1, "b": 2} trailing')
    assert out == {"a": 1, "b": 2}


def test_extract_first_json_object_nested_braces() -> None:
    text = 'noise {"outer": {"inner": [1, 2]}} more'
    out = _extract_first_json_object(text)
    assert out == {"outer": {"inner": [1, 2]}}


def test_extract_first_json_object_missing_returns_none() -> None:
    assert _extract_first_json_object("no json here") is None
    assert _extract_first_json_object("") is None


def test_extract_first_json_object_malformed_returns_none() -> None:
    # Unterminated quote
    assert _extract_first_json_object('{"a": "b') is None
    # Trailing comma is invalid JSON
    assert _extract_first_json_object('{"a": 1, }') is None


def test_extract_first_json_object_list_root_returns_none() -> None:
    # Function only returns dicts at root.
    assert _extract_first_json_object("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# persist_files_from_payload — happy path
# ---------------------------------------------------------------------------

def test_persist_files_writes_each_entry(tmp_path: Path) -> None:
    payload = json.dumps({
        "files": [
            {"path": "src/a.py", "content": "x = 1\n"},
            {"path": "src/sub/b.py", "content": "y = 2\n"},
        ],
        "entrypoint": "src/a.py",
    })
    results = persist_files_from_payload(
        payload=payload,
        project_id="p1",
        projects_root=tmp_path,
        agent_name="code_gen",
        run_id="run_test",
    )
    assert len(results) == 2
    assert all(r["status"] == "ok" for r in results)
    sandbox = tmp_path / "p1" / "sandbox"
    assert (sandbox / "src" / "a.py").read_text() == "x = 1\n"
    assert (sandbox / "src" / "sub" / "b.py").read_text() == "y = 2\n"


def test_persist_files_handles_prefix_text(tmp_path: Path) -> None:
    """The agent's content often has prose before the JSON body."""
    payload = (
        "Here is the implementation:\n\n"
        + json.dumps({"files": [{"path": "x.py", "content": "1"}]})
        + "\n\nHANDOFF: test_gen\n"
    )
    results = persist_files_from_payload(
        payload=payload,
        project_id="p1",
        projects_root=tmp_path,
        agent_name="code_gen",
    )
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert (tmp_path / "p1" / "sandbox" / "x.py").read_text() == "1"


# ---------------------------------------------------------------------------
# persist_files_from_payload — robustness
# ---------------------------------------------------------------------------

def test_persist_files_no_op_on_empty_payload(tmp_path: Path) -> None:
    assert persist_files_from_payload(
        payload="",
        project_id="p1",
        projects_root=tmp_path,
        agent_name="code_gen",
    ) == []


def test_persist_files_no_op_on_non_json(tmp_path: Path) -> None:
    assert persist_files_from_payload(
        payload="just some prose with HANDOFF: test_gen",
        project_id="p1",
        projects_root=tmp_path,
        agent_name="code_gen",
    ) == []


def test_persist_files_no_op_when_files_field_missing(tmp_path: Path) -> None:
    payload = json.dumps({"entrypoint": "foo.py", "notes": "no files here"})
    assert persist_files_from_payload(
        payload=payload, project_id="p1", projects_root=tmp_path,
        agent_name="code_gen",
    ) == []


def test_persist_files_ignores_files_written_string_list(tmp_path: Path) -> None:
    """`files_written: [str, str, ...]` is the contract output — must not be
    confused with `files: [{path, content}, ...]`."""
    payload = json.dumps({
        "files_written": ["src/a.py", "src/b.py"],
        "entrypoint": "src/a.py",
    })
    results = persist_files_from_payload(
        payload=payload, project_id="p1", projects_root=tmp_path,
        agent_name="code_gen",
    )
    assert results == []
    # Verify the sandbox stayed empty.
    sandbox = tmp_path / "p1" / "sandbox"
    assert not sandbox.exists() or not list(sandbox.iterdir())


def test_persist_files_skips_missing_keys(tmp_path: Path) -> None:
    payload = json.dumps({
        "files": [
            {"path": "good.py", "content": "ok"},
            {"path": "no_content.py"},                # missing content -> skip
            {"content": "no path"},                    # missing path -> skip
            "string instead of dict",                  # not a dict -> skip
            {"path": "", "content": "empty path"},     # empty path -> skip
        ],
    })
    results = persist_files_from_payload(
        payload=payload, project_id="p1", projects_root=tmp_path,
        agent_name="code_gen",
    )
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    assert len(ok) == 1
    assert ok[0]["path"] == "good.py"
    assert len(skipped) == 4


def test_persist_files_logs_error_on_sandbox_escape_does_not_raise(tmp_path: Path) -> None:
    payload = json.dumps({
        "files": [
            {"path": "../escape.py", "content": "x"},
            {"path": "ok.py", "content": "y"},
        ],
    })
    # Must NOT raise — the offending entry is logged and others proceed.
    results = persist_files_from_payload(
        payload=payload, project_id="p1", projects_root=tmp_path,
        agent_name="code_gen",
    )
    statuses = sorted(r["status"] for r in results)
    assert statuses == ["error", "ok"]
    err = next(r for r in results if r["status"] == "error")
    assert "SandboxEscape" in err["error"]
    # Verify the second file was still written.
    assert (tmp_path / "p1" / "sandbox" / "ok.py").exists()


def test_persist_files_coerces_non_string_content(tmp_path: Path) -> None:
    """str() coercion: numbers and bools are written as their str() form."""
    payload = json.dumps({
        "files": [
            {"path": "n.txt", "content": 42},
            {"path": "b.txt", "content": True},
        ],
    })
    results = persist_files_from_payload(
        payload=payload, project_id="p1", projects_root=tmp_path,
        agent_name="code_gen",
    )
    assert all(r["status"] == "ok" for r in results)
    assert (tmp_path / "p1" / "sandbox" / "n.txt").read_text() == "42"
    assert (tmp_path / "p1" / "sandbox" / "b.txt").read_text() == "True"


# ---------------------------------------------------------------------------
# Integration-shape: feed the mock code_gen fixture and confirm persistence.
# ---------------------------------------------------------------------------

def test_persist_files_against_mock_code_gen_fixture(tmp_path: Path) -> None:
    """The mock code_gen fixture emits a `files: [{path, content}]` body
    — this is exactly the failure-mode the safety net exists for. After
    persistence the sandbox must contain every file the fixture promised."""
    content = mock_client._fix_code_gen(inbound="")
    results = persist_files_from_payload(
        payload=content,
        project_id="p1",
        projects_root=tmp_path,
        agent_name="code_gen",
        run_id="run_test",
    )
    ok_paths = sorted(r["path"] for r in results if r["status"] == "ok")
    assert ok_paths == ["scripts/run.sh", "src/data.py", "src/train.py"]
    sandbox = tmp_path / "p1" / "sandbox"
    assert (sandbox / "src" / "data.py").read_text() == "# mock data loader\n"
    assert (sandbox / "src" / "train.py").read_text() == "# mock train loop\n"
    assert (sandbox / "scripts" / "run.sh").read_text().startswith("#!/usr/bin/env bash")
