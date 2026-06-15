"""Judge calls and scoring.

KICKOFF.md §9 Phase 6: rubrics are "scored by a separate Claude judge".

The judge is wired as the ``judge`` agent in ``config/models.toml`` so
it routes through the same client / cache / budget plumbing every other
agent uses. ``score_output`` builds a strict-JSON request and parses the
response into a :class:`RubricScore`. When a single dimension cannot be
parsed it is recorded with score 0 and a parse-error rationale so the
total score reflects the deficit (the judge "missed" that dim).

Persistence
-----------
``persist_eval_run`` writes one row to ``eval_runs`` per (variant,
anchor) judged. The A/B harness (``ab_harness.py``) is the normal
caller, but the function is exposed so ad-hoc evals can persist too.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import structlog

from autoscientist.clients.router import route
from autoscientist.meta.anchors import Anchor
from autoscientist.meta.rubrics import Rubric, RubricDimension
from autoscientist.runtime.config import Config, load_config
from autoscientist.state.db import new_id, now_iso

log = structlog.get_logger("autoscientist.meta.eval_rubrics")


@dataclass(frozen=True)
class RubricScore:
    agent: str
    anchor_id: str
    scores: dict[str, float]
    rationales: dict[str, str]
    summary: str
    total: float
    judge_model: str
    parse_errors: tuple[str, ...] = field(default_factory=tuple)
    raw_judge_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "anchor_id": self.anchor_id,
            "scores": dict(self.scores),
            "rationales": dict(self.rationales),
            "summary": self.summary,
            "total": self.total,
            "judge_model": self.judge_model,
            "parse_errors": list(self.parse_errors),
        }


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_judge_prompt(
    rubric: Rubric,
    anchor: Anchor,
    *,
    candidate_prompt: str,
    candidate_output: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Render the judge's system prompt + the single user message.

    The user message embeds a JSON envelope with the fields the mock
    fixture and the real judge both consume; that means the smoke test
    can drive the same code path as the real judge by inserting
    ``__mock_scores`` into the envelope.
    """
    system = (
        "You are an evaluation judge for autoscientist. "
        "You score the output of a target agent against a rubric, on a "
        "1-5 scale per dimension. Return STRICT JSON with a 'scores' "
        "object whose keys are exactly the dimension names; each value "
        "must be {'score': int 1-5, 'rationale': string <= 240 chars}. "
        "Also include a 'summary' string. Do not include anything outside "
        "the JSON object."
    )
    envelope: dict[str, Any] = {
        "agent_name": rubric.agent,
        "anchor_id": anchor.anchor_id,
        "rubric_dims": list(rubric.dim_names),
        "rubric": rubric.to_dict(),
        "candidate_prompt_excerpt": candidate_prompt[:2000],
        "anchor_input": anchor.input_payload,
        "anchor_expected_summary": anchor.expected_summary,
        "candidate_output": candidate_output[:6000],
    }
    user_text = (
        "Score the candidate output below. Each rubric dimension carries "
        "1-5 anchor descriptions; align your score to whichever anchor "
        "best matches the candidate.\n\n"
        f"{json.dumps(envelope, indent=2)}"
    )
    return system, [{"role": "user", "content": user_text}]


def _parse_judge_response(
    text: str,
    rubric: Rubric,
) -> tuple[dict[str, float], dict[str, str], str, list[str]]:
    """Pull dim scores, rationales, and summary out of the raw judge text.

    Returns ``(scores, rationales, summary, parse_errors)``.
    """
    parse_errors: list[str] = []
    scores: dict[str, float] = {}
    rationales: dict[str, str] = {}
    summary = ""
    blob: dict[str, Any] | None = None

    candidate = text.strip()
    if not candidate.startswith("{"):
        m = _JSON_RE.search(text)
        if m:
            candidate = m.group(0)
    try:
        blob = json.loads(candidate)
    except json.JSONDecodeError as e:
        parse_errors.append(f"top_level_json:{e}")
        blob = None
    if isinstance(blob, dict):
        summary = str(blob.get("summary") or "")
        raw_scores = blob.get("scores")
        if isinstance(raw_scores, dict):
            for dim in rubric.dimensions:
                cell = raw_scores.get(dim.name)
                if cell is None:
                    parse_errors.append(f"missing_dim:{dim.name}")
                    continue
                score, rationale, err = _coerce_dim(cell, dim)
                if err:
                    parse_errors.append(err)
                    continue
                scores[dim.name] = score
                rationales[dim.name] = rationale
        else:
            parse_errors.append("scores_not_object")
    return scores, rationales, summary, parse_errors


def _coerce_dim(cell: Any, dim: RubricDimension) -> tuple[float, str, str | None]:
    if isinstance(cell, dict):
        raw_score = cell.get("score")
        rationale = str(cell.get("rationale") or "")
    else:
        raw_score = cell
        rationale = ""
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0, "", f"unparseable_score:{dim.name}={raw_score!r}"
    if not (1.0 <= score <= 5.0):
        return 0.0, rationale, f"score_out_of_range:{dim.name}={score}"
    return score, rationale, None


def score_output(
    *,
    conn: sqlite3.Connection,
    rubric: Rubric,
    anchor: Anchor,
    candidate_prompt: str,
    candidate_output: str,
    cfg: Config | None = None,
    extra_envelope: dict[str, Any] | None = None,
) -> RubricScore:
    """Run the judge on a single (candidate, anchor) pair.

    ``extra_envelope`` is merged into the user-message JSON envelope —
    used by smoke tests to inject ``__mock_scores`` so the offline mock
    judge can return predetermined scores.
    """
    cfg = cfg or load_config()
    system, messages = _build_judge_prompt(
        rubric, anchor,
        candidate_prompt=candidate_prompt,
        candidate_output=candidate_output,
    )
    if extra_envelope:
        # Patch the user message in place — keep the LLM-visible content
        # otherwise unchanged so the cache key is stable across smoke runs.
        u = messages[0]
        # Reparse the embedded JSON, merge, re-serialize.
        prefix, _, blob = u["content"].partition("\n\n")
        try:
            payload = json.loads(blob)
            payload.update(extra_envelope)
            u["content"] = f"{prefix}\n\n{json.dumps(payload, indent=2)}"
        except json.JSONDecodeError:
            u["content"] = u["content"] + "\n\n" + json.dumps(extra_envelope)

    result = route(
        conn=conn,
        agent_name="judge",
        system=system,
        messages=messages,
        cfg=cfg,
    )
    scores, rationales, summary, parse_errors = _parse_judge_response(
        result.content, rubric,
    )
    total = rubric.weighted_total(scores)
    rs = RubricScore(
        agent=rubric.agent,
        anchor_id=anchor.anchor_id,
        scores=scores,
        rationales=rationales,
        summary=summary,
        total=total,
        judge_model=result.model,
        parse_errors=tuple(parse_errors),
        raw_judge_text=result.content,
    )
    log.info(
        "meta.eval.scored",
        agent=rubric.agent, anchor_id=anchor.anchor_id,
        total=round(total, 4),
        n_parse_errors=len(parse_errors),
    )
    return rs


def persist_eval_run(
    conn: sqlite3.Connection,
    *,
    agent: str,
    prompt_version_id: str,
    anchor_id: str,
    candidate_output: str,
    score: RubricScore,
    judge_cost_usd: float = 0.0,
    note: str | None = None,
) -> str:
    """Insert one row into ``eval_runs``. Returns the eval_run_id."""
    eval_run_id = new_id("er_")
    conn.execute(
        """INSERT INTO eval_runs
           (eval_run_id, agent_name, prompt_version_id, anchor_id,
            raw_output, rubric_scores, total_score, judge_model,
            judge_cost_usd, judge_summary, note, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            eval_run_id, agent, prompt_version_id, anchor_id,
            candidate_output,
            json.dumps({
                "scores": score.scores,
                "rationales": score.rationales,
                "parse_errors": list(score.parse_errors),
            }),
            float(score.total),
            score.judge_model,
            float(judge_cost_usd),
            score.summary,
            note,
            now_iso(),
        ),
    )
    return eval_run_id
