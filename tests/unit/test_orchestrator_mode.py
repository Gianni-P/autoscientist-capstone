"""Opus-orchestrator mode for code_gen / test_gen.

Covers:
  * the ``delegate`` tool is registered with a usable schema;
  * in orchestrator mode the runner routes the agent to the manager model,
    adds the ``delegate`` tool, and appends the orchestrator playbook;
  * a plain-alias override does NOT trigger any of that;
  * ``delegate`` runs the local worker through the real agent loop, the worker
    writes a file, and delegate returns a compact (summary + check_imports) view.

Network-free: ``runner.route`` is monkeypatched; the worker's write_file tool
dispatches for real into a tmp sandbox so the summary reflects real disk state.
"""

from __future__ import annotations

import json

import pytest
import structlog

from autoscientist.clients.base import CompletionResult, ToolCall
from autoscientist.runtime import orchestration, runner
from autoscientist.runtime.config import load_config
from autoscientist.state.db import open_db, start_run
from autoscientist.tools import registry as tool_registry


def test_delegate_tool_registered():
    assert tool_registry.is_registered("delegate")
    spec = tool_registry.get_spec("delegate")
    assert "assignment" in spec.input_schema["properties"]
    assert spec.input_schema["required"] == ["assignment"]


def _drive_one_leg(tmp_path, monkeypatch, overrides):
    """Drive a single code_gen leg; return the captured route() kwargs."""
    seen: list[dict] = []

    def fake_route(**kw):
        seen.append({
            "agent": kw["agent_name"],
            "override": kw.get("model_override"),
            "system": kw.get("system") or "",
            "tools": [t["name"] for t in (kw.get("tools_anthropic") or [])],
        })
        return CompletionResult(
            content="HANDOFF: DONE", model="m", provider="claude",
            prompt_tokens=5, completion_tokens=5, finish_reason="end_turn",
        )

    monkeypatch.setattr(runner, "route", fake_route)
    cfg = load_config(reload=True)
    conn = open_db(str(tmp_path / "orch.db"))
    run_id = start_run(conn, project_id="orch", config_snapshot={})
    conn.commit()
    runner._drive_loop(
        conn=conn, cfg=cfg, log=structlog.get_logger("test"),
        run_id=run_id, project_id="orch",
        starting_agent="code_gen", starting_payload='{"plan":"x"}',
        max_handoffs=5, max_tool_rounds=3, enable_checkpoints=False,
        model_overrides=overrides,
    )
    conn.close()
    return seen


def test_orchestrator_mode_routes_to_manager_and_adds_delegate(tmp_path, monkeypatch):
    seen = _drive_one_leg(tmp_path, monkeypatch, {"code_gen": orchestration.ORCH_OVERRIDE})
    assert seen, "route() was never called"
    cg = seen[0]
    assert cg["agent"] == "code_gen"
    assert cg["override"] == "claude_opus_48"        # routed to the manager model
    assert "delegate" in cg["tools"]                  # delegate tool offered
    assert "ORCHESTRATOR MODE" in cg["system"]        # playbook appended


def test_plain_alias_override_has_no_orchestrator_augmentation(tmp_path, monkeypatch):
    seen = _drive_one_leg(tmp_path, monkeypatch, {"code_gen": "claude_haiku"})
    assert seen
    cg = seen[0]
    assert cg["override"] == "claude_haiku"           # plain model swap
    assert "delegate" not in cg["tools"]              # no delegate tool
    assert "ORCHESTRATOR MODE" not in cg["system"]    # no playbook


def test_delegate_runs_worker_and_summarizes(tmp_path, monkeypatch):
    """delegate drives the local worker: it writes a file (real dispatch) and
    delegate returns the sandbox listing + a passing check_imports."""

    def fake_route(**kw):
        msgs = kw["messages"]
        has_tool_result = any(
            m.get("role") == "tool"
            or (isinstance(m.get("content"), list)
                and any(isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in m["content"]))
            for m in msgs
        )
        if not has_tool_result:
            args = {"path": "src/foo.py", "content": "X = 1\n"}
            tc = ToolCall(id="w1", name="write_file", input=args)
            return CompletionResult(
                content="", model="qwen2.5-32b-64k", provider="ollama",
                prompt_tokens=5, completion_tokens=5, finish_reason="tool_calls",
                tool_calls=[tc],
                raw_content_blocks={
                    "role": "assistant", "content": "",
                    "tool_calls": [{"id": "w1", "type": "function",
                                    "function": {"name": "write_file",
                                                 "arguments": json.dumps(args)}}],
                },
            )
        return CompletionResult(
            content="Wrote src/foo.py defining X.", model="qwen2.5-32b-64k",
            provider="ollama", prompt_tokens=5, completion_tokens=5,
            finish_reason="stop", tool_calls=[],
        )

    monkeypatch.setattr(runner, "route", fake_route)
    cfg = load_config(reload=True)  # ensure delegate's cached load_config sees real config

    projects_root = tmp_path / "projects"
    (projects_root / "proj" / "sandbox").mkdir(parents=True)
    # In production the worker's write_file (rooted by _invoke_agent at the
    # cfg projects dir) and delegate's summary (rooted at ctx.projects_root)
    # are both cfg.root/projects. Point the cfg projects dir at the tmp sandbox
    # so they agree here too (absolute path wins the cfg.root join).
    monkeypatch.setitem(cfg.default["paths"], "projects_dir", str(projects_root))
    conn = open_db(str(tmp_path / "deleg.db"))
    run_id = start_run(conn, project_id="proj", config_snapshot={})
    conn.commit()
    ctx = tool_registry.ToolContext(
        conn=conn, project_id="proj", projects_root=projects_root, run_id=run_id,
    )
    out = orchestration.delegate_assignment(
        ctx, assignment="Write src/foo.py defining X = 1", files=["src/foo.py"],
    )
    conn.close()

    assert out["worker_model"] == "qwen2.5-32b-64k"
    assert "src/foo.py" in out["files_in_sandbox"]
    assert out["check_imports"]["ok"] is True
    assert "verify it yourself" in out["note"]
    # the file really hit disk
    assert (projects_root / "proj" / "sandbox" / "src" / "foo.py").read_text() == "X = 1\n"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
