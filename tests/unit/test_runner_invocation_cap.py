"""Adversarial test for the per-invocation budget cap (2026-05-31 audit, item 2).

The hole: a loop of individually-sub-ceiling calls summed to $16+. We construct
exactly that — a tool-loop where every call is cheap and would otherwise run to
max_tool_rounds — and confirm the cumulative cap stops it early. The control
case (high ceiling) confirms the cap is what's bounding the loop, not some other
limit.

Network-free and DB-isolated: route() is monkeypatched to a counter that returns
a fixed per-call cost, and tool dispatch is a no-op.
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


def _run_invocation(tmp_path, monkeypatch, *, per_call_cost, max_tool_rounds,
                    always_tool_call=True):
    """Drive one _invoke_agent with a counting fake route(). Returns n_calls."""
    calls = {"n": 0}

    def fake_route(**kw):
        calls["n"] += 1
        tcs = (
            [ToolCall(id=f"t{calls['n']}", name="list_sandbox", input={})]
            if always_tool_call else []
        )
        return CompletionResult(
            content="" if always_tool_call else "HANDOFF: DONE",
            model="fake", provider="claude",
            prompt_tokens=1000, completion_tokens=1000,
            finish_reason="tool_use" if always_tool_call else "end_turn",
            tool_calls=tcs, cost_usd=per_call_cost,
        )

    monkeypatch.setattr(runner, "route", fake_route)
    # Isolate from real tools: dispatch is a cheap no-op.
    monkeypatch.setattr(
        tool_registry, "dispatch",
        lambda name, inp, ctx: tool_registry.DispatchResult(
            name=name, input=inp, output={"ok": True}, error=None, duration_ms=0
        ),
    )

    cfg = load_config(reload=True)
    conn = open_db(str(tmp_path / "cap.db"))
    run_id = start_run(conn, project_id="capt", config_snapshot={})
    conn.commit()
    agent = Agent(
        name="test_gen", role="x",
        system_prompt_path=Path("test_gen.md"), tools=("list_sandbox",),
    )
    prompt = LoadedPrompt(system_text="sys", temperature=0.0, max_tokens=128)
    runner._invoke_agent(
        conn=conn, agent=agent, prompt=prompt, inbound_text="go",
        run_id=run_id, cfg=cfg, log=structlog.get_logger("test"),
        project_id="capt", max_tool_rounds=max_tool_rounds,
    )
    conn.close()
    return calls["n"]


def test_subceiling_loop_is_bounded_by_invocation_cap(tmp_path, monkeypatch):
    """$0.20/call, $0.50 ceiling: stop after the 3rd call (cum 0.60 >= 0.50),
    NOT after 51 rounds."""
    monkeypatch.setenv("AUTOSCIENTIST_INVOCATION_CEILING_USD", "0.50")
    n = _run_invocation(tmp_path, monkeypatch, per_call_cost=0.20, max_tool_rounds=50)
    assert n == 3, f"expected cap to stop at 3 calls, got {n}"


def test_high_ceiling_lets_loop_reach_max_rounds(tmp_path, monkeypatch):
    """Control: with the cap effectively off, the loop runs to max_tool_rounds+1
    route calls. Proves the cap (not another limit) is what bounds the case above."""
    monkeypatch.setenv("AUTOSCIENTIST_INVOCATION_CEILING_USD", "100000")
    n = _run_invocation(tmp_path, monkeypatch, per_call_cost=0.20, max_tool_rounds=3)
    assert n == 4, f"expected max_tool_rounds+1 = 4 calls, got {n}"


def test_disabled_ceiling_does_not_break_loop(tmp_path, monkeypatch):
    """Ceiling <= 0 disables the cap entirely."""
    monkeypatch.setenv("AUTOSCIENTIST_INVOCATION_CEILING_USD", "0")
    n = _run_invocation(tmp_path, monkeypatch, per_call_cost=5.0, max_tool_rounds=3)
    assert n == 4, f"expected disabled cap to let loop run to 4, got {n}"


def test_no_tool_calls_returns_after_one_call(tmp_path, monkeypatch):
    """A terminal (no-tool) response returns immediately; cap never engages."""
    monkeypatch.setenv("AUTOSCIENTIST_INVOCATION_CEILING_USD", "0.01")
    n = _run_invocation(
        tmp_path, monkeypatch, per_call_cost=1.0, max_tool_rounds=50,
        always_tool_call=False,
    )
    assert n == 1, f"expected single call for terminal response, got {n}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
