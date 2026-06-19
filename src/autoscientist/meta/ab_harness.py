"""A/B harness — run prompt variants against an anchor set, score, pick a winner.

KICKOFF.md §9 Phase 6: "A/B harness: run variants on anchor examples,
score, keep winners."

For each (variant, anchor) pair the harness:

  1. Builds an in-memory ``Agent`` whose ``system_prompt_path`` points at
     a temp file containing the variant's prompt text. This lets the
     existing ``router.route`` / cache / budget path drive the call
     unchanged — no special "variant agent" wiring required.
  2. Calls the agent with the anchor's ``input_payload`` as the user
     message. The model alias is whatever the agent already maps to in
     ``models.toml`` so variants compete on equal footing.
  3. Hands the candidate output to the judge and records the rubric
     score.
  4. Persists ``prompt_versions`` rows for each variant tried (note:
     "ab_harness:<run_label>") plus one ``eval_runs`` row per
     (variant, anchor) judging.

Aggregation: per-variant mean total score across all anchors. Higher
wins ties broken by lower max_dim_variance (preferring variants that are
consistent across anchors).

The smoke test pipes deterministic mock outputs through this harness so
the picking-the-winner behavior can be exercised offline.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from autoscientist.clients.router import route
from autoscientist.meta import eval_rubrics, versioning
from autoscientist.meta.anchors import Anchor, AnchorSet
from autoscientist.meta.meta_prompter import PromptVariant
from autoscientist.meta.rubrics import Rubric
from autoscientist.runtime.config import Config, load_config

log = structlog.get_logger("autoscientist.meta.ab_harness")


@dataclass(frozen=True)
class VariantResult:
    variant_index: int
    prompt_version_id: str
    hypothesis: str
    per_anchor_totals: dict[str, float]
    mean_score: float
    score_variance: float
    n_parse_errors: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_index": self.variant_index,
            "prompt_version_id": self.prompt_version_id,
            "hypothesis": self.hypothesis,
            "per_anchor_totals": dict(self.per_anchor_totals),
            "mean_score": self.mean_score,
            "score_variance": self.score_variance,
            "n_parse_errors": self.n_parse_errors,
        }


@dataclass(frozen=True)
class ABResult:
    agent: str
    rubric: str
    anchor_count: int
    variants: tuple[VariantResult, ...] = field(default_factory=tuple)
    winner_index: int = -1

    @property
    def winner(self) -> VariantResult | None:
        if 0 <= self.winner_index < len(self.variants):
            return self.variants[self.winner_index]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "rubric": self.rubric,
            "anchor_count": self.anchor_count,
            "variants": [v.to_dict() for v in self.variants],
            "winner_index": self.winner_index,
        }


def _variance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _pick_winner(variants: list[VariantResult]) -> int:
    """Highest mean score; ties broken by lowest variance, then lowest parse errors."""
    if not variants:
        return -1
    best = 0
    for i in range(1, len(variants)):
        a, b = variants[best], variants[i]
        if _strictly_better(b, a):
            best = i
    return best


def _strictly_better(b: VariantResult, a: VariantResult) -> bool:
    if b.mean_score != a.mean_score:
        return b.mean_score > a.mean_score
    if b.score_variance != a.score_variance:
        return b.score_variance < a.score_variance
    return b.n_parse_errors < a.n_parse_errors


def _record_variant_version(
    conn: sqlite3.Connection,
    *,
    agent: str,
    variant: PromptVariant,
    run_label: str,
) -> str:
    """Insert a ``prompt_versions`` row for the variant without touching disk.

    A/B variants are *candidates* — they should not overwrite the canonical
    ``prompts/<agent>.md`` until the operator promotes a winner. We persist
    the row so eval_runs has a foreign key to point at, but ``archived_path``
    is null and the canonical file is unchanged.
    """
    from autoscientist.state.db import new_id, now_iso

    parent = versioning.latest_version(conn, agent)
    version_id = new_id("pv_")
    conn.execute(
        """INSERT INTO prompt_versions
           (version_id, agent_name, prompt_text, parent_version_id,
            note, archived_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            version_id, agent, variant.prompt_text,
            parent.version_id if parent else None,
            f"ab_harness:{run_label}:{variant.hypothesis[:80]}",
            None, now_iso(),
        ),
    )
    return version_id


def _run_candidate(
    *,
    conn: sqlite3.Connection,
    agent: str,
    variant_prompt: str,
    anchor: Anchor,
    cfg: Config,
    project_id: str | None = None,
    run_id: str | None = None,
) -> str:
    """Invoke the agent with the variant prompt as system + anchor as user.

    Returns the raw assistant content. We don't parse it here — the
    judge sees it verbatim.
    """
    # Route via ``router.route`` directly with the variant text as the system
    # prompt — no need to materialize a full Agent or a temp prompt file.
    result = route(
        conn=conn,
        agent_name=agent,
        system=variant_prompt,
        messages=[{"role": "user", "content": anchor.input_payload}],
        cfg=cfg,
        project_id=project_id,
        run_id=run_id,
    )
    return result.content


def run_ab(
    *,
    conn: sqlite3.Connection,
    agent: str,
    rubric: Rubric,
    anchors: AnchorSet,
    variants: list[PromptVariant],
    run_label: str = "ab",
    cfg: Config | None = None,
    judge_envelope_per_anchor: dict[str, dict[str, Any]] | None = None,
    project_id: str | None = None,
    run_id: str | None = None,
) -> ABResult:
    """Evaluate ``variants`` x ``anchors``; persist eval_runs rows; pick a winner.

    ``judge_envelope_per_anchor`` lets smoke tests inject deterministic
    mock scores keyed by anchor id (passed through to
    :func:`eval_rubrics.score_output` as ``extra_envelope``).

    Pass ``project_id`` (and optionally ``run_id``) so the candidate and judge
    calls are attributed to the project budget ledger and gated by its
    per-project soft cap — without it an A/B sweep (n_variants x n_anchors
    candidate + judge calls) spends uncapped and lands in the ledger unattributed.
    """
    cfg = cfg or load_config()
    if not variants:
        raise ValueError("run_ab needs at least one variant")
    if not anchors.anchors:
        raise ValueError(f"no anchors for agent {agent}")
    log.info(
        "meta.ab_harness.start",
        agent=agent, n_variants=len(variants), n_anchors=len(anchors),
        run_label=run_label,
    )

    variant_results: list[VariantResult] = []
    for vi, variant in enumerate(variants):
        version_id = _record_variant_version(
            conn, agent=agent, variant=variant, run_label=run_label,
        )
        per_anchor: dict[str, float] = {}
        n_parse_errors = 0
        for anchor in anchors:
            output = _run_candidate(
                conn=conn,
                agent=agent,
                variant_prompt=variant.prompt_text,
                anchor=anchor,
                cfg=cfg,
                project_id=project_id,
                run_id=run_id,
            )
            extra = None
            if judge_envelope_per_anchor:
                extra = judge_envelope_per_anchor.get(anchor.anchor_id)
            score = eval_rubrics.score_output(
                conn=conn,
                rubric=rubric,
                anchor=anchor,
                candidate_prompt=variant.prompt_text,
                candidate_output=output,
                cfg=cfg,
                extra_envelope=extra,
                project_id=project_id,
                run_id=run_id,
            )
            eval_rubrics.persist_eval_run(
                conn,
                agent=agent,
                prompt_version_id=version_id,
                anchor_id=anchor.anchor_id,
                candidate_output=output,
                score=score,
                judge_cost_usd=score.judge_cost_usd,
                note=run_label,
            )
            per_anchor[anchor.anchor_id] = score.total
            n_parse_errors += len(score.parse_errors)
        means = list(per_anchor.values()) or [0.0]
        variant_results.append(VariantResult(
            variant_index=vi,
            prompt_version_id=version_id,
            hypothesis=variant.hypothesis,
            per_anchor_totals=per_anchor,
            mean_score=sum(means) / len(means),
            score_variance=_variance(means),
            n_parse_errors=n_parse_errors,
        ))

    winner_idx = _pick_winner(variant_results)
    result = ABResult(
        agent=agent,
        rubric=rubric.agent,
        anchor_count=len(anchors),
        variants=tuple(variant_results),
        winner_index=winner_idx,
    )
    log.info(
        "meta.ab_harness.completed",
        agent=agent, run_label=run_label,
        winner_index=winner_idx,
        winner_score=variant_results[winner_idx].mean_score if winner_idx >= 0 else None,
    )
    return result


def write_winner_to_canonical(
    conn: sqlite3.Connection,
    *,
    prompts_dir: Path,
    ab: ABResult,
    variants: list[PromptVariant],
    note: str | None = None,
) -> versioning.PromptVersion | None:
    """Promote the winning variant to ``prompts/<agent>.md``.

    Routes through :func:`versioning.write_prompt` so the previous
    canonical file is archived first. Returns the new ``PromptVersion``
    or ``None`` if no winner was found.
    """
    if ab.winner is None:
        return None
    variant = variants[ab.winner.variant_index]
    return versioning.write_prompt(
        conn,
        prompts_dir=prompts_dir,
        agent_name=ab.agent,
        new_text=variant.prompt_text,
        note=note or f"ab_harness_winner:{ab.winner.variant_index}",
    )


def variants_summary_json(ab: ABResult) -> str:
    """Compact JSON summary; useful for log records and operator UI."""
    return json.dumps(ab.to_dict(), indent=2)
