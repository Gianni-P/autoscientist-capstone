"""Code review agent — Claude-tier review of Qwen-generated code.

Loops back to code_gen on revise/block; advances to results_validator on pass.
"""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="code_review",
    role="code review and methodology-violation gate",
    system_prompt_path=Path("code_review.md"),
    handoff_targets=("code_gen", "results_validator"),
    tools=("execute", "literature_lookup"),
)
