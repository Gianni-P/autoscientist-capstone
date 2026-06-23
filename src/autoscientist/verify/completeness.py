"""Experiment-completeness & baseline-presence verification.

Closes the "wrote a paper about experiments it never ran" gap. In an earlier
math693a run only the E1 baseline executed, while E2–E5 — the
rotation-vs-Dijkstra comparison that *was* the research question — never
produced results, yet the draft still advanced toward CP5. A deterministic
gate that cross-checks the methodology ``plan`` against the materialised
``results`` would have caught that before paper_writer ran.

Two checks, both reading ``state["plan"]`` and ``state["results"]``:

* :func:`check_experiment_completeness` — every experiment id declared in
  ``plan.experiments`` must have a result artifact. A declared experiment with
  no results is a hard ``fail`` (severity ``fail``): the promised comparison
  was never run.
* :func:`check_baselines_present` — if the plan declares baselines, the results
  must contain baseline-comparison evidence. Absence is ``needs_human``
  (detecting "a baseline was compared against" deterministically is fuzzy; we
  surface for the operator rather than block).

Both return ``skipped`` when their inputs are absent (e.g. ``run_all`` called
before any results exist), so they are safe to include in the standard
:func:`autoscientist.verify.run_all` sweep at any stage.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

import structlog

from autoscientist.verify.types import Verdict, make_skipped

log = structlog.get_logger("autoscientist.verify.completeness")

_CATEGORY = "completeness"

# Experiment ids as they show up in artifact keys/filenames: e1, E2, _e3_,
# /e4_summary.json. Require the E not to be glued to a surrounding letter so we
# don't match "size12" or "phase1stage".
_EXP_ID_RE = re.compile(r"(?<![A-Za-z])[eE]\d+(?![A-Za-z])")

# Tokens that, if present anywhere in the serialized results, are evidence that
# *some* baseline / reference comparison was computed.
_BASELINE_MARKERS = (
    "baseline", "reference", "ground_truth", "ground truth", "optimal",
    "optimality", "gap", "cog", "reference_optimum",
)


def _unwrap_plan(plan: Any) -> Mapping[str, Any] | None:
    """Payloads sometimes nest the real plan under a ``plan`` key."""
    if not isinstance(plan, Mapping):
        return None
    if (
        "experiments" not in plan
        and "baselines" not in plan
        and isinstance(plan.get("plan"), Mapping)
    ):
        return plan["plan"]  # type: ignore[return-value]
    return plan


def _declared_experiment_ids(plan: Mapping[str, Any]) -> list[str]:
    exps = plan.get("experiments")
    ids: list[str] = []
    if isinstance(exps, Iterable) and not isinstance(exps, (str, bytes)):
        for e in exps:
            if isinstance(e, Mapping):
                eid = e.get("id") or e.get("experiment") or e.get("name")
                if eid:
                    ids.append(str(eid).strip().upper())
    return ids


def _iter_result_records(results: Any) -> Iterable[tuple[str, Any]]:
    """Yield ``(key, value)`` over whatever shape ``results`` takes.

    Accepts a dict keyed by filename/path → record, a list of records, or a
    single record.
    """
    if isinstance(results, Mapping):
        for k, v in results.items():
            yield str(k), v
    elif isinstance(results, Iterable) and not isinstance(results, (str, bytes)):
        for i, v in enumerate(results):
            yield str(i), v
    elif results is not None:
        yield "", results


def _present_experiment_ids(results: Any) -> set[str]:
    present: set[str] = set()
    for key, rec in _iter_result_records(results):
        # 1) explicit experiment field on the record
        if isinstance(rec, Mapping):
            ev = rec.get("experiment") or rec.get("experiment_id") or rec.get("id")
            if ev:
                present.add(str(ev).strip().upper())
        # 2) experiment ids embedded in the artifact key/filename
        for m in _EXP_ID_RE.findall(key):
            present.add(m.upper())
    return present


def check_experiment_completeness(
    plan: Any, results: Any, *, severity: str = "fail",
) -> Verdict:
    title = "All declared experiments produced results"
    p = _unwrap_plan(plan)
    if p is None:
        return make_skipped(
            "experiment_completeness", title,
            severity=severity, reason="no plan available", category=_CATEGORY,
        )
    declared = _declared_experiment_ids(p)
    if not declared:
        return make_skipped(
            "experiment_completeness", title,
            severity=severity, reason="plan declares no experiments",
            category=_CATEGORY,
        )
    # No results yet (e.g. run_all called before code_gen) → nothing to check,
    # not a failure.
    if results is None:
        return make_skipped(
            "experiment_completeness", title,
            severity=severity, reason="no results materialised yet",
            category=_CATEGORY,
        )
    present = _present_experiment_ids(results)
    missing = [d for d in declared if d not in present]
    evidence = {
        "declared": declared,
        "present": sorted(present),
        "missing": missing,
    }
    if missing:
        return Verdict(
            check_id="experiment_completeness", title=title,
            status="fail", severity=severity,  # type: ignore[arg-type]
            detail=(
                f"{len(missing)} declared experiment(s) have no result artifact: "
                f"{', '.join(missing)} — the comparison was never run"
            ),
            evidence=evidence, category=_CATEGORY,
        )
    return Verdict(
        check_id="experiment_completeness", title=title,
        status="pass", severity=severity,  # type: ignore[arg-type]
        detail=f"all {len(declared)} declared experiment(s) produced results",
        evidence=evidence, category=_CATEGORY,
    )


def check_baselines_present(
    plan: Any, results: Any, *, severity: str = "needs_human",
) -> Verdict:
    title = "Declared baselines have comparison evidence in results"
    p = _unwrap_plan(plan)
    if p is None:
        return make_skipped(
            "baselines_present", title,
            severity=severity, reason="no plan available", category=_CATEGORY,
        )
    baselines = p.get("baselines")
    names: list[str] = []
    if isinstance(baselines, Iterable) and not isinstance(baselines, (str, bytes)):
        names = [
            str(b.get("name")) for b in baselines
            if isinstance(b, Mapping) and b.get("name")
        ]
    if not names:
        return make_skipped(
            "baselines_present", title,
            severity=severity, reason="plan declares no baselines",
            category=_CATEGORY,
        )
    if results is None:
        return make_skipped(
            "baselines_present", title,
            severity=severity, reason="no results materialised yet",
            category=_CATEGORY,
        )
    blob = json.dumps(results, default=str).lower()
    found = sorted({m for m in _BASELINE_MARKERS if m in blob})
    evidence = {
        "declared_baselines": names,
        "markers_found": found,
        "markers_searched": list(_BASELINE_MARKERS),
    }
    if found:
        return Verdict(
            check_id="baselines_present", title=title,
            status="pass", severity=severity,  # type: ignore[arg-type]
            detail=(
                "baseline-comparison evidence present in results: "
                f"{', '.join(found)}"
            ),
            evidence=evidence, category=_CATEGORY,
        )
    return Verdict(
        check_id="baselines_present", title=title,
        status="needs_human", severity=severity,  # type: ignore[arg-type]
        detail=(
            f"{len(names)} baseline(s) declared but no baseline-comparison metric "
            "found in results — confirm the comparison was actually run"
        ),
        evidence=evidence, category=_CATEGORY,
    )


def run_completeness(state: Mapping[str, Any]) -> list[Verdict]:
    """Run experiment-completeness checks against the pipeline state.

    Expected keys:
      * ``plan`` — the methodology plan (or a payload wrapping it under ``plan``)
      * ``results`` — materialised results (dict keyed by artifact, a list, or
        a single record)
    """
    plan = state.get("plan")
    results = state.get("results")
    verdicts = [
        check_experiment_completeness(plan, results),
        check_baselines_present(plan, results),
    ]
    log.info(
        "verify.completeness.completed",
        statuses=[v.status for v in verdicts],
    )
    return verdicts
