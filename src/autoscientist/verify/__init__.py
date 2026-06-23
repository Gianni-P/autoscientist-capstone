"""Phase 5 verification harness.

KICKOFF.md §4 #2 — every check that *can* be deterministic *must* be
deterministic. LLM "review" is last-resort.

Modules:

* :mod:`autoscientist.verify.leakage` — train/test ID overlap, target leakage
* :mod:`autoscientist.verify.baseline_repro` — published-baseline reproduction
* :mod:`autoscientist.verify.stats` — multicollinearity, normality, sample size
* :mod:`autoscientist.verify.pitfalls` — domain pitfall TOML + handlers
* :mod:`autoscientist.verify.completeness` — declared-experiment & baseline presence
* :mod:`autoscientist.verify.provenance` — paper-number → results traceability

Public surface:

* :class:`Verdict`, :class:`VerifyReport` — result types (verify.types)
* :func:`run_all` — runs every module against a pipeline-state dict
* :func:`open_verify_checkpoint` — convenience: turn a non-clean
  report into a Phase 4 checkpoint row that the operator UI can show

The "pipeline state" passed to :func:`run_all` is a plain dict — each
module reads only the keys it needs, returning ``skipped`` verdicts for
missing data. This keeps the verify package decoupled from upstream
agent output schemas; the runner (or smoke tests) hands in whatever it
has, and the harness fills in the rest with skips.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any

import structlog

from autoscientist.verify import (
    baseline_repro,
    completeness,
    leakage,
    pitfalls,
    provenance,
    stats,
)
from autoscientist.verify.types import Verdict, VerifyReport, make_skipped

log = structlog.get_logger("autoscientist.verify")

__all__ = (
    "Verdict",
    "VerifyReport",
    "baseline_repro",
    "completeness",
    "leakage",
    "make_skipped",
    "open_verify_checkpoint",
    "pitfalls",
    "provenance",
    "run_all",
    "stats",
)

# When pitfalls.py and baseline_repro.py both produce
# ``baseline_reproduced_within_tolerance``, the pitfall version carries
# the TOML-declared severity and is the canonical aggregate. Keep that
# one and drop the duplicate from baseline_repro; the per-claim verdicts
# from baseline_repro are unique and stay.
_PITFALL_OWNS = {"baseline_reproduced_within_tolerance"}


def run_all(
    state: Mapping[str, Any],
    *,
    domain: str = "medical_imaging",
    pitfall_config_path=None,
) -> VerifyReport:
    """Run every verification module against ``state`` and aggregate.

    ``pitfall_config_path`` is forwarded to
    :func:`pitfalls.run_pitfalls` for tests that inject a fixture TOML.
    """
    leakage_v = leakage.run_leakage(state)
    baseline_v = baseline_repro.run_baseline_repro(state)
    stats_v = stats.run_stats(state)
    completeness_v = completeness.run_completeness(state)
    provenance_v = provenance.run_provenance(state)
    pitfalls_v = pitfalls.run_pitfalls(
        state, domain=domain, config_path=pitfall_config_path,
    )

    pitfall_ids = {v.check_id for v in pitfalls_v}
    baseline_dedup = [
        v for v in baseline_v
        if not (v.check_id in _PITFALL_OWNS and v.check_id in pitfall_ids)
    ]

    all_verdicts = [
        *leakage_v, *baseline_dedup, *stats_v,
        *completeness_v, *provenance_v, *pitfalls_v,
    ]
    report = VerifyReport.from_verdicts(all_verdicts)
    log.info(
        "verify.run_all.completed",
        domain=domain,
        outcome=report.outcome,
        n_verdicts=len(report.verdicts),
    )
    return report


def open_verify_checkpoint(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    from_agent: str,
    to_agent: str,
    report: VerifyReport,
    stage: int = 4,
) -> str | None:
    """If ``report`` is non-clean, open a checkpoint and return its id.

    Stage defaults to 4 (Full results validation) — that's where the
    pipeline calls the verify harness in the v1 design. Caller can pass
    stage=3 for the preliminary-results variant.

    Returns ``None`` when the report is clean — no checkpoint needed
    and the caller should advance.
    """
    if report.outcome == "clean":
        return None
    from autoscientist.checkpoints.manager import open_checkpoint

    fail_verdicts = report.by_status("fail")
    human_verdicts = report.by_status("needs_human")
    summary_lines = [report.summary]
    if fail_verdicts:
        summary_lines.append(f"FAIL: {len(fail_verdicts)} verdict(s)")
        for v in fail_verdicts[:5]:
            summary_lines.append(f"  - {v.check_id}: {v.detail}")
    if human_verdicts:
        summary_lines.append(f"NEEDS HUMAN: {len(human_verdicts)} verdict(s)")
        for v in human_verdicts[:5]:
            summary_lines.append(f"  - {v.check_id}: {v.detail}")
    summary = "\n".join(summary_lines)

    parsed = report.to_dict()
    default_payload = json.dumps({
        "verify_outcome": report.outcome,
        "verify_summary": report.summary,
    })
    return open_checkpoint(
        conn,
        run_id=run_id,
        stage=stage,
        from_agent=from_agent,
        to_agent=to_agent,
        agent_output_raw=summary,
        default_payload=default_payload,
        parsed=parsed,
        summary=summary,
        extra={"verify_outcome": report.outcome},
    )
