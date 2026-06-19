"""Tests for the tool-loop verdict-emission safety net (2026-06-18).

A thorough agent (observed: code_review on Sonnet) can spend its entire
tool-round budget investigating and never emit a tool-free final message — the
loop then exits at the cap with an empty-content tool-call result, which gets
force-forwarded as an empty payload and opens a degenerate checkpoint
(run_fbd5651…). The fix: (1a) nudge the agent to finalize a few rounds before
the cap, and (1b) if it still ends on tool calls, force one tools-disabled
completion so a real verdict is always produced.

Network-free and DB-isolated: route() is monkeypatched; tool dispatch is a
no-op (mirrors test_runner_invocation_cap.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog

from autoscientist.clients.base import CompletionResult, ToolCall
from autoscientist.runtime import runner
from autoscientist.runtime.agent import Agent, LoadedPrompt
from autoscientist.runtime.config import load_config
from autoscientist.state.db import open_db, start_run
from autoscientist.tools import registry as tool_registry


def _drive(tmp_path, monkeypatch, *, max_tool_rounds, terminal_on_call=None):
    """Drive one _invoke_agent. Returns (result, info).

    ``terminal_on_call`` (1-based) makes that route() call return a tool-free
    verdict; otherwise every call returns a tool call with empty content (the
    failure mode). The forced extraction call (tools disabled) always returns a
    verdict so we can detect it.
    """
    info = {"n": 0, "tools_off_calls": [], "nudge_seen": False, "forced_seen": False}

    def fake_route(**kw):
        info["n"] += 1
        n = info["n"]
        tools_off = kw.get("tools_anthropic") is None and kw.get("tools_openai") is None
        info["tools_off_calls"].append(tools_off)
        # Detect the (1a) nudge having been injected into the message history.
        blob = str(kw.get("messages"))
        if "tool-use round(s) left" in blob:
            info["nudge_seen"] = True
        if tools_off:  # the (1b) forced extraction call
            info["forced_seen"] = True
            return CompletionResult(
                content="VERDICT: block\nHANDOFF: results_validator",
                model="fake", provider="claude",
                prompt_tokens=10, completion_tokens=10,
                finish_reason="end_turn", tool_calls=[], cost_usd=0.0,
            )
        if terminal_on_call is not None and n == terminal_on_call:
            return CompletionResult(
                content="VERDICT: approve\nHANDOFF: results_validator",
                model="fake", provider="claude",
                prompt_tokens=10, completion_tokens=10,
                finish_reason="end_turn", tool_calls=[], cost_usd=0.0,
            )
        return CompletionResult(
            content="", model="fake", provider="claude",
            prompt_tokens=10, completion_tokens=10,
            finish_reason="tool_use",
            tool_calls=[ToolCall(id=f"t{n}", name="list_sandbox", input={})],
            cost_usd=0.0,
        )

    monkeypatch.setattr(runner, "route", fake_route)
    monkeypatch.setattr(
        tool_registry, "dispatch",
        lambda name, inp, ctx: tool_registry.DispatchResult(
            name=name, input=inp, output={"ok": True}, error=None, duration_ms=0
        ),
    )
    # Keep the per-invocation cost cap out of the way.
    monkeypatch.setenv("AUTOSCIENTIST_INVOCATION_CEILING_USD", "0")

    cfg = load_config(reload=True)
    conn = open_db(str(tmp_path / "verdict.db"))
    run_id = start_run(conn, project_id="vt", config_snapshot={})
    conn.commit()
    agent = Agent(
        name="code_review", role="x",
        system_prompt_path=Path("code_review.md"), tools=("list_sandbox",),
    )
    prompt = LoadedPrompt(system_text="sys", temperature=0.0, max_tokens=128)
    result = runner._invoke_agent(
        conn=conn, agent=agent, prompt=prompt, inbound_text="go",
        run_id=run_id, cfg=cfg, log=structlog.get_logger("test"),
        project_id="vt", max_tool_rounds=max_tool_rounds,
    )
    conn.close()
    return result, info


def test_forced_verdict_when_loop_exhausts_rounds(tmp_path, monkeypatch):
    """Agent that never stops calling tools gets one tools-disabled completion,
    so the returned content is a real verdict — never empty."""
    result, info = _drive(tmp_path, monkeypatch, max_tool_rounds=5)
    assert info["forced_seen"] is True
    assert result.content == "VERDICT: block\nHANDOFF: results_validator"
    assert not result.tool_calls
    # 6 loop calls (max_tool_rounds + 1) + 1 forced extraction call.
    assert info["n"] == 7, info["n"]
    # Only the final (forced) call had tools disabled.
    assert info["tools_off_calls"] == [False] * 6 + [True]


def test_finalize_nudge_injected_before_cap(tmp_path, monkeypatch):
    """The (1a) nudge is appended to the message history before the cap."""
    _result, info = _drive(tmp_path, monkeypatch, max_tool_rounds=5)
    assert info["nudge_seen"] is True


def test_terminal_verdict_returns_without_forcing(tmp_path, monkeypatch):
    """When the agent emits a tool-free verdict normally, the loop returns it
    immediately — no nudge, no forced extraction call."""
    result, info = _drive(
        tmp_path, monkeypatch, max_tool_rounds=40, terminal_on_call=2,
    )
    assert result.content == "VERDICT: approve\nHANDOFF: results_validator"
    assert info["forced_seen"] is False
    assert info["nudge_seen"] is False
    assert info["n"] == 2, info["n"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
