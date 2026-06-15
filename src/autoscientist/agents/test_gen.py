"""Test generation agent — writes tests targeting plan pitfalls."""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="test_gen",
    role="test synthesis targeting methodology pitfalls",
    system_prompt_path=Path("test_gen.md"),
    handoff_targets=("code_review",),
    tools=("execute", "write_file"),
)
