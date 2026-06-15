"""Tests for the repo_publisher agent and its supporting filesystem tools.

Covers:
  * write_release_file: writes under release/, rejects sandbox escapes.
  * read_sandbox_file: reads back, truncates long content, refuses escapes.
  * list_sandbox: walks files, skips noise dirs, respects max_entries.
  * registry: the three new tools are registered and dispatch correctly.
  * agent registry: repo_publisher resolves and points at a real prompt.
  * topology: peer_reviewer can hand off to repo_publisher.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscientist.agents import AGENT_NAMES, get_agent
from autoscientist.runtime.agent import load_prompt
from autoscientist.runtime.config import load_config
from autoscientist.tools import (
    list_sandbox as ls_mod,
    read_sandbox_file as rsf_mod,
    write_release_file as wrf_mod,
)
from autoscientist.tools.registry import ToolContext, dispatch, get_spec


# ---------------------------------------------------------------------------
# write_release_file
# ---------------------------------------------------------------------------

def test_write_release_file_writes_under_release_dir(tmp_path: Path) -> None:
    result = wrf_mod.write_release_file(
        path="README.md",
        content="# hello\n",
        project_id="p1",
        projects_root=tmp_path,
    )
    assert result["written"] is True
    target = tmp_path / "p1" / "release" / "README.md"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "# hello\n"
    assert result["size_bytes"] == target.stat().st_size


def test_write_release_file_creates_nested_parents(tmp_path: Path) -> None:
    wrf_mod.write_release_file(
        path="src/pkg/mod.py",
        content="x = 1\n",
        project_id="p1",
        projects_root=tmp_path,
    )
    assert (tmp_path / "p1" / "release" / "src" / "pkg" / "mod.py").exists()


def test_write_release_file_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(wrf_mod.ReleaseEscape):
        wrf_mod.write_release_file(
            path="/etc/passwd",
            content="",
            project_id="p1",
            projects_root=tmp_path,
        )


def test_write_release_file_rejects_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(wrf_mod.ReleaseEscape):
        wrf_mod.write_release_file(
            path="../../escape.txt",
            content="",
            project_id="p1",
            projects_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# read_sandbox_file
# ---------------------------------------------------------------------------

def test_read_sandbox_file_reads_back(tmp_path: Path) -> None:
    sandbox = tmp_path / "p1" / "sandbox"
    sandbox.mkdir(parents=True)
    (sandbox / "a.py").write_text("print('hi')\n", encoding="utf-8")
    out = rsf_mod.read_sandbox_file(
        path="a.py",
        project_id="p1",
        projects_root=tmp_path,
    )
    assert out["content"] == "print('hi')\n"
    assert out["truncated"] is False
    assert out["encoding"] == "utf-8"


def test_read_sandbox_file_truncates_large(tmp_path: Path) -> None:
    sandbox = tmp_path / "p1" / "sandbox"
    sandbox.mkdir(parents=True)
    big = "x" * (rsf_mod.MAX_TEXT_BYTES + 1000)
    (sandbox / "big.txt").write_text(big, encoding="utf-8")
    out = rsf_mod.read_sandbox_file(
        path="big.txt",
        project_id="p1",
        projects_root=tmp_path,
    )
    assert out["truncated"] is True
    assert out["content"].endswith("[truncated]")
    assert len(out["content"]) < len(big)


def test_read_sandbox_file_missing(tmp_path: Path) -> None:
    (tmp_path / "p1" / "sandbox").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        rsf_mod.read_sandbox_file(
            path="nope.py",
            project_id="p1",
            projects_root=tmp_path,
        )


def test_read_sandbox_file_rejects_escape(tmp_path: Path) -> None:
    (tmp_path / "p1" / "sandbox").mkdir(parents=True)
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(rsf_mod.SandboxEscape):
        rsf_mod.read_sandbox_file(
            path="../../outside.txt",
            project_id="p1",
            projects_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# list_sandbox
# ---------------------------------------------------------------------------

def test_list_sandbox_returns_only_files_skipping_noise(tmp_path: Path) -> None:
    sandbox = tmp_path / "p1" / "sandbox"
    (sandbox / "src").mkdir(parents=True)
    (sandbox / "src" / "a.py").write_text("a", encoding="utf-8")
    (sandbox / "src" / "__pycache__").mkdir()
    (sandbox / "src" / "__pycache__" / "a.cpython-312.pyc").write_text("", encoding="utf-8")
    (sandbox / ".git").mkdir()
    (sandbox / ".git" / "HEAD").write_text("ref", encoding="utf-8")

    out = ls_mod.list_sandbox(project_id="p1", projects_root=tmp_path)
    paths = [e["path"] for e in out["entries"]]
    assert "src/a.py" in paths
    assert not any(".git" in p or "__pycache__" in p for p in paths)
    assert out["truncated"] is False


def test_list_sandbox_respects_max_entries(tmp_path: Path) -> None:
    sandbox = tmp_path / "p1" / "sandbox"
    sandbox.mkdir(parents=True)
    for i in range(10):
        (sandbox / f"f{i}.py").write_text("", encoding="utf-8")
    out = ls_mod.list_sandbox(project_id="p1", projects_root=tmp_path, max_entries=3)
    assert out["count"] == 3
    assert out["truncated"] is True


def test_list_sandbox_subdir_scoping(tmp_path: Path) -> None:
    sandbox = tmp_path / "p1" / "sandbox"
    (sandbox / "src").mkdir(parents=True)
    (sandbox / "scripts").mkdir()
    (sandbox / "src" / "a.py").write_text("", encoding="utf-8")
    (sandbox / "scripts" / "go.sh").write_text("", encoding="utf-8")
    out = ls_mod.list_sandbox(project_id="p1", projects_root=tmp_path, subdir="scripts")
    paths = [e["path"] for e in out["entries"]]
    assert paths == ["scripts/go.sh"]


def test_list_sandbox_rejects_escape(tmp_path: Path) -> None:
    (tmp_path / "p1" / "sandbox").mkdir(parents=True)
    with pytest.raises(ls_mod.SandboxEscape):
        ls_mod.list_sandbox(
            project_id="p1", projects_root=tmp_path, subdir="../../etc",
        )


# ---------------------------------------------------------------------------
# Registry: dispatch through the LLM tool-use layer
# ---------------------------------------------------------------------------

def test_registry_dispatch_write_release_file(tmp_path: Path) -> None:
    ctx = ToolContext(conn=None, project_id="p1", projects_root=tmp_path)
    dr = dispatch(
        "write_release_file",
        {"path": "hello.txt", "content": "world"},
        ctx,
    )
    assert dr.error is None
    assert dr.output["written"] is True
    assert (tmp_path / "p1" / "release" / "hello.txt").read_text(encoding="utf-8") == "world"


def test_registry_dispatch_read_sandbox_file(tmp_path: Path) -> None:
    (tmp_path / "p1" / "sandbox").mkdir(parents=True)
    (tmp_path / "p1" / "sandbox" / "x.py").write_text("payload", encoding="utf-8")
    ctx = ToolContext(conn=None, project_id="p1", projects_root=tmp_path)
    dr = dispatch("read_sandbox_file", {"path": "x.py"}, ctx)
    assert dr.error is None
    assert dr.output["content"] == "payload"


def test_registry_dispatch_list_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "p1" / "sandbox"
    sandbox.mkdir(parents=True)
    (sandbox / "a.py").write_text("", encoding="utf-8")
    ctx = ToolContext(conn=None, project_id="p1", projects_root=tmp_path)
    dr = dispatch("list_sandbox", {}, ctx)
    assert dr.error is None
    assert dr.output["count"] == 1


def test_registry_schemas_present() -> None:
    for name in ("write_release_file", "read_sandbox_file", "list_sandbox"):
        spec = get_spec(name)
        assert spec.name == name
        assert "type" in spec.input_schema


# ---------------------------------------------------------------------------
# Agent registry: repo_publisher resolves cleanly and the topology is wired.
# ---------------------------------------------------------------------------

def test_repo_publisher_listed_in_agent_names() -> None:
    assert "repo_publisher" in AGENT_NAMES


def test_repo_publisher_resolves_with_real_prompt() -> None:
    cfg = load_config()
    agent = get_agent("repo_publisher", cfg)
    assert agent is not None
    assert agent.name == "repo_publisher"
    assert agent.handoff_targets == ()  # terminal
    assert "write_release_file" in agent.tools
    assert "list_sandbox" in agent.tools
    assert "read_sandbox_file" in agent.tools
    # Prompt must exist and be loadable with frontmatter.
    assert agent.system_prompt_path.exists()
    prompt = load_prompt(agent.system_prompt_path)
    assert "repo_publisher" in prompt.system_text


def test_peer_reviewer_can_hand_off_to_repo_publisher() -> None:
    cfg = load_config()
    pr = get_agent("peer_reviewer", cfg)
    assert pr is not None
    assert "repo_publisher" in pr.handoff_targets
    assert "paper_writer" in pr.handoff_targets


def test_repo_publisher_routing_in_models_toml() -> None:
    cfg = load_config()
    agents = cfg.models.get("agents", {})
    assert "repo_publisher" in agents
    assert agents["repo_publisher"]["model"]
