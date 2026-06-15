"""Phase 6 — autoresearch / prompt optimization.

Public surface:

* :mod:`autoscientist.meta.versioning` — archive-on-write prompt store
* :mod:`autoscientist.meta.anchors` — anchor example files (prompts/anchors/<agent>/)
* :mod:`autoscientist.meta.rubrics` — per-agent rubric definitions
* :mod:`autoscientist.meta.eval_rubrics` — judge call + scoring
* :mod:`autoscientist.meta.meta_prompter` — propose prompt variants
* :mod:`autoscientist.meta.ab_harness` — run variants x anchors, pick winners

KICKOFF.md §9 Phase 6 (deferred until Phases 1-5 are stable). The
substrate ships with: idea_gen rubric (KICKOFF example), methodology
rubric, paper_writer rubric, plus a generic-rubric fallback so any
agent can be evaluated. Anchors live as JSON files under
``prompts/anchors/<agent>/`` so they version-control with the prompts.
"""

from __future__ import annotations

from autoscientist.meta import (
    ab_harness,
    anchors,
    eval_rubrics,
    meta_prompter,
    rubrics,
    versioning,
)

__all__ = (
    "ab_harness",
    "anchors",
    "eval_rubrics",
    "meta_prompter",
    "rubrics",
    "versioning",
)
