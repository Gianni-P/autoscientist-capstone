"""Tests for build_code_review_payload_from_sandbox.

code_review has no file-reading tool, so when an upstream agent hands off
with empty content the runner must rebuild code_review's input from the
sandbox on disk (otherwise CP3 opens with a degenerate "(no payload)"
review — observed 2026-06-12, run_5273a6fe…). These tests cover the
reconstruction helper directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoscientist.runtime.payload_files import (
    _collect_py_files,
    build_code_review_payload_from_sandbox,
)


def _make_sandbox(root: Path, project_id: str = "p1") -> Path:
    sandbox = root / project_id / "sandbox"
    (sandbox / "src").mkdir(parents=True)
    (sandbox / "tests").mkdir(parents=True)
    return sandbox


# ---------------------------------------------------------------------------
# build_code_review_payload_from_sandbox
# ---------------------------------------------------------------------------

def test_rebuild_returns_none_when_no_sandbox(tmp_path: Path) -> None:
    assert build_code_review_payload_from_sandbox(
        project_id="missing", projects_root=tmp_path,
    ) is None


def test_rebuild_returns_none_when_sandbox_empty(tmp_path: Path) -> None:
    _make_sandbox(tmp_path)  # src/ and tests/ exist but contain no .py
    assert build_code_review_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path,
    ) is None


def test_rebuild_collects_src_and_tests(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    (sandbox / "src" / "train.py").write_text("def train():\n    return 1\n")
    (sandbox / "src" / "config.py").write_text("SEED = 0\n")
    (sandbox / "tests" / "test_train.py").write_text("def test_x():\n    assert True\n")

    raw = build_code_review_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path,
    )
    assert raw is not None
    payload = json.loads(raw)

    assert payload["run_cmd_tests"] == "pytest tests/ -x -q"
    assert "run_cmd_src" in payload
    assert "_reconstructed_by_runner" in payload

    src_paths = {f["path"] for f in payload["src_files"]}
    test_paths = {f["path"] for f in payload["test_files"]}
    assert src_paths == {"src/config.py", "src/train.py"}
    assert test_paths == {"tests/test_train.py"}

    # Contents are carried verbatim — this is the whole point (code_review
    # cannot read them itself).
    train = next(f for f in payload["src_files"] if f["path"] == "src/train.py")
    assert train["content"] == "def train():\n    return 1\n"


def test_rebuild_works_with_only_tests(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    (sandbox / "tests" / "test_only.py").write_text("def test_y():\n    assert 1\n")
    raw = build_code_review_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path,
    )
    assert raw is not None
    payload = json.loads(raw)
    assert payload["src_files"] == []
    assert [f["path"] for f in payload["test_files"]] == ["tests/test_only.py"]


def test_rebuild_nested_paths_relative_to_sandbox(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    (sandbox / "src" / "sub").mkdir()
    (sandbox / "src" / "sub" / "deep.py").write_text("x = 1\n")
    payload = json.loads(build_code_review_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path,
    ))
    assert [f["path"] for f in payload["src_files"]] == ["src/sub/deep.py"]


# ---------------------------------------------------------------------------
# _collect_py_files
# ---------------------------------------------------------------------------

def test_collect_skips_pycache_and_non_py(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    src = sandbox / "src"
    (src / "keep.py").write_text("a = 1\n")
    (src / "notes.txt").write_text("ignore me\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "keep.cpython-312.pyc").write_text("bytecode")

    collected = _collect_py_files(src, sandbox)
    assert [f["path"] for f in collected] == ["src/keep.py"]


def test_collect_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert _collect_py_files(tmp_path / "nope", tmp_path) == []


def test_collect_truncates_oversize_file(tmp_path: Path) -> None:
    from autoscientist.runtime.payload_files import _MAX_REVIEW_FILE_BYTES

    sandbox = _make_sandbox(tmp_path)
    big = "x = 1  # padding\n" * (_MAX_REVIEW_FILE_BYTES // 4)
    (sandbox / "src" / "big.py").write_text(big)
    collected = _collect_py_files(sandbox / "src", sandbox)
    assert len(collected) == 1
    assert "truncated by runner" in collected[0]["content"]
