"""Simulated peer reviewer agent.

Hands off to paper_writer for revision (``minor_revise`` / ``major_revise``)
or to repo_publisher when the draft is acceptable (``accept``). CP5 fires
either way; the operator approves CP5 and resume continues at whichever
agent peer_reviewer routed to. On ``reject`` the operator rejects CP5 and
the run is cancelled — peer_reviewer still routes its handoff to
paper_writer so the rejection is logged on a real edge.
"""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="peer_reviewer",
    role="simulated peer review of draft + supplementary",
    system_prompt_path=Path("peer_reviewer.md"),
    handoff_targets=("paper_writer", "repo_publisher"),
    tools=("literature_lookup", "pdf_parse", "citation_check"),
)
