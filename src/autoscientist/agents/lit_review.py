"""Literature review agent — pulls structured digest of cited works.

Phase 2: free-text/JSON only; tools (Semantic Scholar, OpenAlex, arxiv)
arrive in Phase 3. Until then, the prompt instructs the agent to mark
unverified citations with ``[CITATION NEEDED]``.
"""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="lit_review",
    role="literature review and citation grounding",
    system_prompt_path=Path("lit_review.md"),
    handoff_targets=("idea_gen",),
    tools=("literature_search", "literature_lookup", "pdf_parse"),
)
