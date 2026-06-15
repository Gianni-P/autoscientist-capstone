"""Idea critic agent — adversarial review and ranking.

Forwards the top-ranked idea to methodology. The operator's checkpoint #1
sees this output before methodology runs (Phase 4 wires that gate).
"""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="idea_critic",
    role="adversarial idea critique",
    system_prompt_path=Path("idea_critic.md"),
    handoff_targets=("methodology",),
    tools=("literature_search", "literature_lookup"),
)
