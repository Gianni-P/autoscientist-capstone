"""Idea generation agent — proposes 5 concrete research ideas."""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="idea_gen",
    role="research idea generation",
    system_prompt_path=Path("idea_gen.md"),
    handoff_targets=("idea_critic",),
    tools=("literature_search",),
)
