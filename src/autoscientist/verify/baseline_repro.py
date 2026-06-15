"""Baseline reproduction harness.

KICKOFF.md §4 #7 (hard rule):
    *No "novel result" claim is allowed in any output until the pipeline
    has reproduced a published baseline within configured tolerance on
    the same dataset.*

A "baseline claim" pairs a published number against the pipeline's
observed number for the same model + dataset + metric. The harness
returns one verdict per claim (pass/fail) plus an aggregate verdict
keyed ``baseline_reproduced_within_tolerance`` that the pitfall library
can route through.

Tolerance can be specified either absolutely (``tolerance_abs``) or as
a relative fraction (``tolerance_rel``); the harness uses whichever is
present, preferring ``tolerance_abs`` when both are. Either-or design
matches how ML papers state tolerances (sometimes ``±0.5 AUROC``,
sometimes ``±2%``).

For metrics where higher-is-worse (loss, RMSE, error rate), set
``higher_is_better=False`` on the claim. The tolerance is symmetric
either way; the flag only affects the ``direction`` evidence field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import structlog

from autoscientist.verify.types import Verdict, make_skipped

log = structlog.get_logger("autoscientist.verify.baseline_repro")

_DEFAULT_TOLERANCE_ABS = 0.02  # 2 absolute points (e.g. AUROC 0.847 -> 0.827-0.867)


@dataclass(frozen=True)
class BaselineClaim:
    name: str               # e.g. "CheXNet-Rajpurkar2017"
    dataset: str            # e.g. "NIH-ChestX-ray14"
    metric: str             # e.g. "AUROC-pneumonia"
    published_value: float
    observed_value: float
    tolerance_abs: float | None = None
    tolerance_rel: float | None = None
    higher_is_better: bool = True
    citation_key: str | None = None  # links back to the bib entry

    def effective_tolerance_abs(self) -> float:
        if self.tolerance_abs is not None:
            return float(self.tolerance_abs)
        if self.tolerance_rel is not None:
            return abs(float(self.published_value)) * float(self.tolerance_rel)
        return _DEFAULT_TOLERANCE_ABS


# ---------------------------------------------------------------------------
# Per-claim check
# ---------------------------------------------------------------------------

def check_baseline(claim: BaselineClaim, *, severity: str = "fail") -> Verdict:
    title = f"Baseline reproduction — {claim.name} / {claim.metric}"
    delta = float(claim.observed_value) - float(claim.published_value)
    abs_delta = abs(delta)
    tol = claim.effective_tolerance_abs()
    direction = "above" if delta > 0 else ("below" if delta < 0 else "equal")
    evidence: dict[str, Any] = {
        "name": claim.name,
        "dataset": claim.dataset,
        "metric": claim.metric,
        "published": claim.published_value,
        "observed": claim.observed_value,
        "delta": delta,
        "abs_delta": abs_delta,
        "tolerance_abs": tol,
        "direction": direction,
        "higher_is_better": claim.higher_is_better,
    }
    if claim.citation_key is not None:
        evidence["citation_key"] = claim.citation_key
    if abs_delta <= tol:
        return Verdict(
            check_id=f"baseline_repro::{claim.name}::{claim.metric}",
            title=title,
            status="pass",
            severity=severity,  # type: ignore[arg-type]
            detail=(
                f"observed {claim.observed_value:.4f} within ±{tol:.4f} of "
                f"published {claim.published_value:.4f}"
            ),
            evidence=evidence,
            category="baseline_repro",
        )
    return Verdict(
        check_id=f"baseline_repro::{claim.name}::{claim.metric}",
        title=title,
        status="fail",
        severity=severity,  # type: ignore[arg-type]
        detail=(
            f"observed {claim.observed_value:.4f} differs from published "
            f"{claim.published_value:.4f} by {abs_delta:.4f} > tol {tol:.4f}"
        ),
        evidence=evidence,
        category="baseline_repro",
    )


# ---------------------------------------------------------------------------
# Aggregate verdict
# ---------------------------------------------------------------------------

def _aggregate(verdicts: list[Verdict], *, severity: str = "fail") -> Verdict:
    """Roll all per-claim verdicts into the canonical pitfall id.

    KICKOFF.md §4 #7 hinges on a single boolean: did *any* baseline land
    within tolerance? If no claims were submitted, this is ``skipped`` —
    the orchestrator (or the pitfall library, with novelty-claim context)
    decides whether absence is itself a fail.
    """
    title = "Published baseline reproduced within tolerance"
    if not verdicts:
        return make_skipped(
            "baseline_reproduced_within_tolerance", title,
            severity=severity, reason="no baseline claims submitted",
            category="baseline_repro",
        )
    fails = [v for v in verdicts if v.status == "fail"]
    passes = [v for v in verdicts if v.status == "pass"]
    if passes:
        detail = (
            f"{len(passes)}/{len(verdicts)} baseline(s) reproduced within tolerance; "
            f"{len(fails)} outside"
        )
        # KICKOFF: "reproduced *a* baseline" is sufficient. Any pass → overall pass,
        # but we still expose the per-claim fails as evidence.
        return Verdict(
            check_id="baseline_reproduced_within_tolerance",
            title=title,
            status="pass",
            severity=severity,  # type: ignore[arg-type]
            detail=detail,
            evidence={
                "n_claims": len(verdicts),
                "n_pass": len(passes),
                "n_fail": len(fails),
                "fails": [v.to_dict() for v in fails],
            },
            category="baseline_repro",
        )
    return Verdict(
        check_id="baseline_reproduced_within_tolerance",
        title=title,
        status="fail",
        severity=severity,  # type: ignore[arg-type]
        detail=f"0/{len(verdicts)} baseline(s) reproduced within tolerance",
        evidence={
            "n_claims": len(verdicts),
            "fails": [v.to_dict() for v in verdicts],
        },
        category="baseline_repro",
    )


# ---------------------------------------------------------------------------
# State-driven runner
# ---------------------------------------------------------------------------

def _coerce_claim(d: Mapping[str, Any]) -> BaselineClaim:
    return BaselineClaim(
        name=str(d["name"]),
        dataset=str(d["dataset"]),
        metric=str(d["metric"]),
        published_value=float(d["published_value"]),
        observed_value=float(d["observed_value"]),
        tolerance_abs=(float(d["tolerance_abs"]) if "tolerance_abs" in d and d["tolerance_abs"] is not None else None),
        tolerance_rel=(float(d["tolerance_rel"]) if "tolerance_rel" in d and d["tolerance_rel"] is not None else None),
        higher_is_better=bool(d.get("higher_is_better", True)),
        citation_key=d.get("citation_key"),
    )


def run_baseline_repro(state: Mapping[str, Any]) -> list[Verdict]:
    """Run baseline-reproduction checks against the pipeline state.

    Expected keys:
      * ``baseline_claims`` — iterable of dicts (see :class:`BaselineClaim`)

    The returned list is per-claim verdicts followed by the aggregate
    verdict that the pitfall library reads under
    ``baseline_reproduced_within_tolerance``. If no claims are present
    *and* the pipeline state declares ``claims_novelty=True``, the
    aggregate becomes a fail — novelty claims without a reproduced
    baseline violate KICKOFF §4 #7.
    """
    raw: Iterable[Mapping[str, Any]] | None = state.get("baseline_claims")
    per_claim: list[Verdict] = []
    if raw:
        for d in raw:
            try:
                claim = _coerce_claim(d)
            except (KeyError, TypeError, ValueError) as e:
                per_claim.append(Verdict(
                    check_id="baseline_repro::malformed",
                    title="Baseline reproduction — malformed claim",
                    status="error",
                    severity="fail",
                    detail=f"could not parse baseline claim: {e}",
                    evidence={"raw": dict(d) if isinstance(d, Mapping) else repr(d)},
                    category="baseline_repro",
                ))
                continue
            per_claim.append(check_baseline(claim))

    aggregate = _aggregate(per_claim)

    # Hard rule: novelty claim without reproduced baseline = fail, even if no
    # baseline_claims were submitted (the pipeline simply skipped its job).
    if (
        state.get("claims_novelty")
        and aggregate.status != "pass"
    ):
        aggregate = Verdict(
            check_id=aggregate.check_id,
            title=aggregate.title,
            status="fail",
            severity="fail",
            detail=(
                "novelty claim present but no published baseline reproduced "
                "within tolerance (KICKOFF.md §4 #7)"
            ),
            evidence={
                **aggregate.evidence,
                "novelty_claimed": True,
                "underlying_aggregate_status": aggregate.status,
            },
            category="baseline_repro",
        )

    log.info(
        "verify.baseline_repro.completed",
        n_claims=len(per_claim),
        aggregate_status=aggregate.status,
    )
    return [*per_claim, aggregate]
