"""Methodology agent — turns top idea into an experimental plan."""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="methodology",
    role="experimental design and statistical plan",
    system_prompt_path=Path("methodology.md"),
    handoff_targets=("code_gen",),
    tools=("literature_search", "literature_lookup", "dataset_info"),
)
