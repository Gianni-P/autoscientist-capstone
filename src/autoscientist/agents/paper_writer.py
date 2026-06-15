"""Paper writer agent — drafts academic paper + supplementary materials."""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="paper_writer",
    role="academic paper drafting from validated results",
    system_prompt_path=Path("paper_writer.md"),
    handoff_targets=("peer_reviewer",),
    tools=("literature_lookup", "citation_check", "latex_compile"),
)
