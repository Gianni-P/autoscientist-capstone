"""Orchestrator worker — writes a single delegated assignment to the sandbox.

NOT part of the normal handoff pipeline. Invoked only by the ``delegate`` tool
(see ``runtime/orchestration.py``) when code_gen/test_gen run in Opus-orchestrator
mode. The Opus manager decomposes the task and hands one file-level assignment
here; this agent (a local model — $0) writes the file(s) and stops.

Write-only toolset on purpose: no ``execute`` (can't debug-spin), no ``handoff``
(the worker never routes — control returns to the orchestrator when it stops
calling tools), and no ``delegate`` (no recursion). The orchestrator owns
running tests, routing, and the final correctness review.
"""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="code_worker",
    role="local file-writer for delegated orchestrator assignments",
    system_prompt_path=Path("code_worker.md"),
    handoff_targets=(),
    tools=("write_file", "read_sandbox_file", "list_sandbox", "check_imports"),
)
