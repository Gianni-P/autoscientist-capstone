"""Tests for the structured `handoff` tool interception in _invoke_agent.

qwen3-coder reliably emits tool calls but routinely failed to emit the bare-line
`HANDOFF: <target>` directive the runner parses — so test_gen/code_gen got
force-forwarded with empty payloads and CP3 degenerated (run_358912…, 2026-06-12).
The fix lets an agent call a `handoff` tool; the runner validates the target and
synthesizes the canonical directive into the content so the rest of the pipeline
is unchanged. These tests drive _invoke_agent with a fake route() and assert:

  * a valid handoff target ends the turn with a parseable `HANDOFF:` directive;
  * an invalid target does NOT end the turn — it is fed back as a tool error so
    the model can retry within the loop.

Network-free and DB-isolated (route() is monkeypatched; the sole real tool call
is the special-cased `handoff`, which is never dispatched to a handler).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog

from autoscientist.clients.base import CompletionResult, ToolCall
from autoscientist.runtime import runner
from autoscientist.runtime.agent import Agent, LoadedPrompt
from autoscientist.runtime.config import load_config
from autoscientist.runtime.handoff import parse_handoff
from autoscientist.state.db import open_db, start_run


def _drive(tmp_path, monkeypatch, route_results):
    """Run _invoke_agent, returning (result, n_route_calls). ``route_results`` is
    a list of CompletionResult returned in order, one per route() call."""
    seq = {"i": 0}

    def fake_route(**kw):
        i = seq["i"]
        seq["i"] += 1
        return route_results[min(i, len(route_results) - 1)]

    monkeypatch.setattr(runner, "route", fake_route)

    cfg = load_config(reload=True)
    conn = open_db(str(tmp_path / "ho.db"))
    run_id = start_run(conn, project_id="hot", config_snapshot={})
    conn.commit()
    agent = Agent(
        name="test_gen", role="x",
        system_prompt_path=Path("test_gen.md"),
        handoff_targets=("code_review",),
        tools=("handoff",),
    )
    prompt = LoadedPrompt(system_text="sys", temperature=0.0, max_tokens=128)
    result = runner._invoke_agent(
        conn=conn, agent=agent, prompt=prompt, inbound_text="go",
        run_id=run_id, cfg=cfg, log=structlog.get_logger("test"),
        project_id="hot", max_tool_rounds=10,
    )
    conn.close()
    return result, seq["i"]


def _handoff_call(target, summary=""):
    return CompletionResult(
        content="", model="fake", provider="claude",
        prompt_tokens=10, completion_tokens=10, finish_reason="tool_use",
        tool_calls=[ToolCall(id="h1", name="handoff",
                             input={"target": target, "summary": summary})],
        cost_usd=0.0,
    )


def test_valid_handoff_tool_ends_turn_with_parseable_directive(tmp_path, monkeypatch):
    summary = '{"test_files": ["tests/test_core.py"]}'
    result, n = _drive(tmp_path, monkeypatch, [_handoff_call("code_review", summary)])

    assert n == 1, f"valid handoff should terminate after one route call, got {n}"
    # The synthesized content must parse to the right target via the existing regex.
    ho = parse_handoff(result.content, from_agent="test_gen")
    assert ho is not None and ho.to_agent == "code_review"
    assert summary in result.content  # the summary became the forwarded payload


def test_invalid_target_is_fed_back_then_model_retries(tmp_path, monkeypatch):
    # First call asks for a bogus target (rejected, loop continues); second call
    # uses a valid one (terminates).
    result, n = _drive(
        tmp_path, monkeypatch,
        [_handoff_call("nonsense_agent"), _handoff_call("code_review", "ok")],
    )
    assert n == 2, f"invalid target should not terminate; expected retry, got n={n}"
    ho = parse_handoff(result.content, from_agent="test_gen")
    assert ho is not None and ho.to_agent == "code_review"


def test_handoff_to_DONE_is_terminal(tmp_path, monkeypatch):
    result, n = _drive(tmp_path, monkeypatch, [_handoff_call("DONE")])
    assert n == 1
    ho = parse_handoff(result.content, from_agent="test_gen")
    assert ho is not None and ho.to_agent == "DONE" and ho.is_terminal


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
