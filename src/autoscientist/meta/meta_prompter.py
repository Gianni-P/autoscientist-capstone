"""Meta-prompter — propose prompt variants from low-scoring eval traces.

KICKOFF.md §9 Phase 6:
    *Meta-prompter: Claude instance that reads eval traces and proposes
    prompt variations.*

Inputs:

  * ``baseline_prompt`` — the current ``prompts/<agent>.md`` text.
  * ``low_score_traces`` — recent eval rows (from
    :func:`eval_rubrics.score_output`) where the candidate scored poorly.
    Each trace exposes the per-dim scores and the candidate output so the
    meta-prompter can target the actual deficits.

Output: a list of :class:`PromptVariant` records, each with a candidate
prompt rewrite and a one-sentence hypothesis. The harness
(``ab_harness.py``) then evaluates them against the same anchor set.

The mock fixture returns deterministic variants so smoke tests run
offline; the real call goes through the ``meta_prompter`` agent route.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import structlog

from autoscientist.clients.router import route
from autoscientist.meta.eval_rubrics import RubricScore
from autoscientist.runtime.config import Config, load_config

log = structlog.get_logger("autoscientist.meta.meta_prompter")


@dataclass(frozen=True)
class PromptVariant:
    prompt_text: str
    hypothesis: str

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_text": self.prompt_text, "hypothesis": self.hypothesis}


@dataclass(frozen=True)
class TraceSlice:
    """A compressed eval trace the meta-prompter can ingest cheaply."""
    anchor_id: str
    total_score: float
    dim_scores: dict[str, float]
    candidate_output_excerpt: str  # truncated to keep tokens bounded
    judge_summary: str = ""

    @classmethod
    def from_score(
        cls, score: RubricScore, candidate_output: str,
        *, output_chars: int = 1500,
    ) -> TraceSlice:
        return cls(
            anchor_id=score.anchor_id,
            total_score=score.total,
            dim_scores=dict(score.scores),
            candidate_output_excerpt=candidate_output[:output_chars],
            judge_summary=score.summary,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "total_score": self.total_score,
            "dim_scores": self.dim_scores,
            "candidate_output_excerpt": self.candidate_output_excerpt,
            "judge_summary": self.judge_summary,
        }


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_prompt(
    *,
    agent: str,
    baseline_prompt: str,
    traces: list[TraceSlice],
    n_variants: int,
    rubric_dims: tuple[str, ...],
    n_baseline_chars: int = 6000,
) -> tuple[str, list[dict[str, Any]]]:
    system = (
        "You are a prompt engineer for a multi-agent research pipeline. "
        "Given a baseline system prompt for one agent plus a set of low-"
        "scoring eval traces, propose targeted rewrites that should "
        "score higher on the rubric dimensions. Each variant must keep "
        "the core contract (handoff schema, output format) but may "
        "tighten instructions, add examples, or restructure sections. "
        "Return STRICT JSON: {'variants': [{'prompt_text': str, "
        "'hypothesis': str}, ...]}. Do not include anything outside the "
        "JSON object."
    )
    envelope = {
        "agent_name": agent,
        "n_variants": n_variants,
        "rubric_dims": list(rubric_dims),
        "baseline_prompt": baseline_prompt[:n_baseline_chars],
        "low_score_traces": [t.to_dict() for t in traces],
    }
    user_text = (
        "Propose prompt variants tailored to the deficits visible in the "
        "low-score traces. Each hypothesis should name which dimension(s) "
        "you expect to improve and why.\n\n"
        f"{json.dumps(envelope, indent=2)}"
    )
    return system, [{"role": "user", "content": user_text}]


def _parse_variants(text: str) -> list[PromptVariant]:
    candidate = text.strip()
    if not candidate.startswith("{"):
        m = _JSON_RE.search(text)
        if m:
            candidate = m.group(0)
    try:
        blob = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    raw = blob.get("variants") if isinstance(blob, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[PromptVariant] = []
    for v in raw:
        if not isinstance(v, dict):
            continue
        text = v.get("prompt_text")
        hyp = v.get("hypothesis")
        if not isinstance(text, str) or not text.strip():
            continue
        out.append(PromptVariant(
            prompt_text=text,
            hypothesis=str(hyp or ""),
        ))
    return out


def propose_variants(
    *,
    conn: sqlite3.Connection,
    agent: str,
    baseline_prompt: str,
    low_score_traces: list[TraceSlice],
    n_variants: int = 3,
    rubric_dims: tuple[str, ...] = (),
    cfg: Config | None = None,
    project_id: str | None = None,
    run_id: str | None = None,
) -> list[PromptVariant]:
    """Run one meta-prompter call.

    Returns an empty list if the response cannot be parsed — the caller
    decides whether to retry, fall back, or surface to the operator.
    """
    cfg = cfg or load_config()
    system, messages = _build_prompt(
        agent=agent,
        baseline_prompt=baseline_prompt,
        traces=low_score_traces,
        n_variants=n_variants,
        rubric_dims=rubric_dims,
    )
    result = route(
        conn=conn,
        agent_name="meta_prompter",
        system=system,
        messages=messages,
        cfg=cfg,
        project_id=project_id,
        run_id=run_id,
    )
    variants = _parse_variants(result.content)
    log.info(
        "meta.meta_prompter.proposed",
        agent=agent, n_variants=len(variants),
        n_traces=len(low_score_traces),
    )
    return variants
