"""Shared types for the verification harness.

KICKOFF.md §4 #2 — every check that *can* be deterministic *must* be
deterministic. Each verify module produces :class:`Verdict` records;
:class:`VerifyReport` aggregates them into a single outcome the runner
uses to decide whether to halt, open a checkpoint, or continue.

Status vs. severity
-------------------
A check produces a *status* (``pass`` / ``fail`` / ``needs_human`` /
``skipped`` / ``error``). The check's *severity* (declared up-front,
e.g. in ``config/domains/<domain>.toml``) is the consequence policy
that fires when status==``fail``:

  * severity ``fail``         → contributes to the report's ``block`` outcome
  * severity ``needs_human``  → contributes to ``needs_human`` outcome
  * severity ``warn``         → recorded but never escalates

A status of ``needs_human`` always contributes to ``needs_human``
regardless of severity (a check that explicitly punts to the operator
is more cautious than its declared escalation).

The aggregated outcome ranks: ``block`` > ``needs_human`` > ``clean``.
``skipped`` and ``error`` verdicts never escalate; an ``error`` verdict
is logged so the operator can investigate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

VerdictStatus = Literal["pass", "fail", "needs_human", "skipped", "error"]
Severity = Literal["fail", "needs_human", "warn"]
Outcome = Literal["clean", "needs_human", "block"]

_VALID_STATUS = frozenset({"pass", "fail", "needs_human", "skipped", "error"})
_VALID_SEVERITY = frozenset({"fail", "needs_human", "warn"})


@dataclass(frozen=True)
class Verdict:
    check_id: str
    title: str
    status: VerdictStatus
    severity: Severity
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    category: str = "generic"  # leakage|baseline_repro|stats|pitfall|...

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUS:
            raise ValueError(f"invalid verdict status: {self.status}")
        if self.severity not in _VALID_SEVERITY:
            raise ValueError(f"invalid verdict severity: {self.severity}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def escalates_block(self) -> bool:
        return self.status == "fail" and self.severity == "fail"

    @property
    def escalates_human(self) -> bool:
        if self.status == "needs_human":
            return True
        return self.status == "fail" and self.severity == "needs_human"


@dataclass(frozen=True)
class VerifyReport:
    outcome: Outcome
    verdicts: tuple[Verdict, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "summary": self.summary,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }

    def by_status(self, status: VerdictStatus) -> tuple[Verdict, ...]:
        return tuple(v for v in self.verdicts if v.status == status)

    def by_category(self, category: str) -> tuple[Verdict, ...]:
        return tuple(v for v in self.verdicts if v.category == category)

    @classmethod
    def from_verdicts(cls, verdicts: list[Verdict] | tuple[Verdict, ...]) -> VerifyReport:
        verdicts = tuple(verdicts)
        outcome: Outcome = "clean"
        n_block = sum(1 for v in verdicts if v.escalates_block)
        n_human = sum(1 for v in verdicts if v.escalates_human and not v.escalates_block)
        n_pass = sum(1 for v in verdicts if v.status == "pass")
        n_skip = sum(1 for v in verdicts if v.status == "skipped")
        n_err = sum(1 for v in verdicts if v.status == "error")
        if n_block:
            outcome = "block"
        elif n_human:
            outcome = "needs_human"
        summary = (
            f"{outcome.upper()}: {len(verdicts)} checks "
            f"({n_pass} pass, {n_block} block, {n_human} needs_human, "
            f"{n_skip} skipped, {n_err} error)"
        )
        return cls(outcome=outcome, verdicts=verdicts, summary=summary)


def make_skipped(
    check_id: str, title: str, *, severity: Severity, reason: str, category: str = "generic"
) -> Verdict:
    """Helper: a check that lacks the required state returns ``skipped``.

    Skipping is preferable to ``error`` when the missing data simply
    isn't part of this run (e.g. no claimed baseline, so the baseline
    check has nothing to compare).
    """
    return Verdict(
        check_id=check_id,
        title=title,
        status="skipped",
        severity=severity,
        detail=f"skipped: {reason}",
        evidence={"reason": reason},
        category=category,
    )
