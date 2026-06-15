"""Per-agent rubric definitions.

KICKOFF.md §9 Phase 6 (example):
    *for idea_gen: novelty + grounding + feasibility + counter-arg quality,
    scored by a separate Claude judge*

Each :class:`Rubric` declares the dimensions on which the judge scores
an output. Every dimension carries a 1-to-5 anchor description so the
judge has stable definitions of what each score means; without them,
"3" drifts.

Rubrics live in code (not TOML) because the scoring prompt embeds the
dimension descriptions verbatim and the wording is part of the contract.
A new rubric requires a code change *and* an evaluation pass on existing
anchors so old eval traces remain comparable — the operator has to
notice when scoring semantics shift.

Phase 6 ships rubrics for the three highest-leverage agents:
``idea_gen`` (KICKOFF example), ``methodology`` (KICKOFF §4 #1 framework
quality), and ``paper_writer`` (final-output gate). Other agents inherit
the generic rubric until they accumulate enough eval traces to warrant a
dedicated one — the smoke harness still works on the generic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RubricDimension:
    name: str
    description: str
    anchors: dict[int, str]  # 1..5 score descriptions
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "anchors": dict(self.anchors),
            "weight": self.weight,
        }


@dataclass(frozen=True)
class Rubric:
    agent: str
    dimensions: tuple[RubricDimension, ...] = field(default_factory=tuple)

    @property
    def dim_names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dimensions)

    def total_weight(self) -> float:
        return sum(d.weight for d in self.dimensions)

    def weighted_total(self, scores: dict[str, float]) -> float:
        """Compute the weighted average score across declared dimensions.

        Missing scores contribute 0 — the judge is expected to score
        every dimension; if it doesn't, the missing dim drags the total
        down so the harness flags the failure.
        """
        if not self.dimensions:
            return 0.0
        weight = self.total_weight()
        if weight <= 0:
            return 0.0
        acc = 0.0
        for d in self.dimensions:
            acc += d.weight * float(scores.get(d.name, 0.0))
        return acc / weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }


# ---------------------------------------------------------------------------
# Concrete rubrics
# ---------------------------------------------------------------------------

_IDEA_GEN_NOVELTY = RubricDimension(
    name="novelty",
    description="Does the idea offer a non-obvious contribution beyond restating the literature?",
    anchors={
        1: "Restates a known result; no contribution.",
        2: "Minor parameter tweak on an existing study; outcome predictable.",
        3: "Useful confirmation or replication with a new dataset; modest delta.",
        4: "Specific hypothesis that the existing literature has not directly tested.",
        5: "Cross-cuts methods or domains in a way that opens a clear new question.",
    },
)
_IDEA_GEN_GROUNDING = RubricDimension(
    name="grounding",
    description="Is the idea anchored in the supplied literature digest with specific citations?",
    anchors={
        1: "No citations or invented references.",
        2: "Vague nods to a field without naming work.",
        3: "Names at least one relevant paper; connection to the idea is loose.",
        4: "Connects to two or more cited gaps and references them by title/author.",
        5: "Builds the idea on a specific gap or contradiction in cited work, with the gap quoted.",
    },
)
_IDEA_GEN_FEASIBILITY = RubricDimension(
    name="feasibility",
    description="Can this be executed on the operator's stated compute and public datasets?",
    anchors={
        1: "Requires private data or compute the operator does not have.",
        2: "Theoretically possible but resource estimate is hand-wavy.",
        3: "Plausible on a single 5090 with minor compromises; estimate within 2x.",
        4: "Concrete dataset, model size, and seed budget that fit; estimate is grounded.",
        5: "Detailed compute plan with checkpoints and an early-abort criterion.",
    },
)
_IDEA_GEN_COUNTER_ARG = RubricDimension(
    name="counter_arg_quality",
    description="Are failure modes specific, falsifiable, and methodologically substantive?",
    anchors={
        1: "No failure modes, or only 'might not work'.",
        2: "Generic risks (data noisy, model hard to train).",
        3: "Names one substantive risk (e.g. domain shift) but not a kill criterion.",
        4: "Multiple specific risks with measurable kill criteria.",
        5: "Risks tied to identifiable confounders in the cited literature with mitigation hooks.",
    },
)

IDEA_GEN_RUBRIC = Rubric(
    agent="idea_gen",
    dimensions=(
        _IDEA_GEN_NOVELTY,
        _IDEA_GEN_GROUNDING,
        _IDEA_GEN_FEASIBILITY,
        _IDEA_GEN_COUNTER_ARG,
    ),
)


METHODOLOGY_RUBRIC = Rubric(
    agent="methodology",
    dimensions=(
        RubricDimension(
            name="experimental_completeness",
            description="Does the plan cover hypothesis, datasets, baselines, metrics, and seeds?",
            anchors={
                1: "Major sections missing.",
                2: "Some sections present but high-level only.",
                3: "All sections present; some lack actionable detail.",
                4: "All sections actionable; baselines named with sources.",
                5: "All sections rigorous; statistical assumptions called out and justified.",
            },
        ),
        RubricDimension(
            name="statistical_rigor",
            description="Are statistical tests appropriate, multiple-comparison corrected, and powered?",
            anchors={
                1: "No stats plan.",
                2: "Mentions a metric without significance treatment.",
                3: "Names a test but does not address assumptions.",
                4: "Test + correction + minimum effect size declared.",
                5: "All four: test, assumptions, correction, power calc — and the power calc is sane.",
            },
        ),
        RubricDimension(
            name="pitfall_awareness",
            description="Does the plan name and mitigate domain pitfalls (leakage, site shift, TTA, etc.)?",
            anchors={
                1: "No mitigations.",
                2: "Lists pitfalls but does not mitigate them.",
                3: "Mitigates the most obvious pitfall (e.g. patient-level split).",
                4: "Mitigates several with specific mechanisms.",
                5: "Mitigations enumerated against the domain pitfall TOML, with verify hooks.",
            },
        ),
    ),
)


PAPER_WRITER_RUBRIC = Rubric(
    agent="paper_writer",
    dimensions=(
        RubricDimension(
            name="structure",
            description="All standard sections present, properly sequenced, sized appropriately?",
            anchors={
                1: "Sections missing or out of order.",
                2: "All sections present but several are skeletal.",
                3: "Sections present and proportionate; one or two skeletal.",
                4: "All sections substantive and well-sequenced.",
                5: "Section flow drives the argument; transitions explicit.",
            },
        ),
        RubricDimension(
            name="citation_grounding",
            description="Every citation is verifiable; claims are backed by named work.",
            anchors={
                1: "Many unverifiable or hallucinated citations.",
                2: "Some unverified citations remain.",
                3: "All citations point to real work; some claims uncited.",
                4: "All claims cited; cites are tightly bound to the claim wording.",
                5: "Citations include specific results from the cited work, not just titles.",
            },
        ),
        RubricDimension(
            name="clarity",
            description="Methods reproducible from text alone; results clearly stated; limitations honest.",
            anchors={
                1: "Reader cannot reconstruct what was done.",
                2: "Methods sketchy; results partially conveyed.",
                3: "Methods reproducible with effort; results clear; limitations general.",
                4: "Methods directly reproducible; results unambiguous; limitations specific.",
                5: "All three plus crisp framing of contribution and scope.",
            },
        ),
        RubricDimension(
            name="methodology_match",
            description="Does the paper faithfully describe what the methodology agent specified?",
            anchors={
                1: "Paper describes a different study.",
                2: "Major divergences (different datasets, different baselines).",
                3: "Minor divergences clearly justified.",
                4: "Paper matches plan; deviations explicitly explained.",
                5: "Paper matches plan exactly; pre-registration-quality fidelity.",
            },
        ),
    ),
)


_GENERIC_RUBRIC = Rubric(
    agent="__generic__",
    dimensions=(
        RubricDimension(
            name="schema_conformance",
            description="Does the output adhere to the agent's declared output schema?",
            anchors={
                1: "Output is malformed or missing required keys.",
                2: "Several keys missing or wrong types.",
                3: "Schema mostly correct; minor key omissions.",
                4: "Schema correct.",
                5: "Schema correct and uses optional fields appropriately.",
            },
        ),
        RubricDimension(
            name="completeness",
            description="Does the output address every part of the inbound payload?",
            anchors={
                1: "Ignores most of the inbound payload.",
                2: "Addresses only one aspect.",
                3: "Addresses most aspects superficially.",
                4: "Addresses every aspect substantively.",
                5: "Addresses every aspect and surfaces gaps the operator may have missed.",
            },
        ),
        RubricDimension(
            name="grounding",
            description="Are claims supported by named evidence (citations, observed values, prior outputs)?",
            anchors={
                1: "Unsupported claims throughout.",
                2: "Mix of supported and unsupported claims.",
                3: "Most claims supported; some hand-wavy.",
                4: "All claims supported.",
                5: "All claims supported with specific quantitative or citation backing.",
            },
        ),
    ),
)


_REGISTRY: dict[str, Rubric] = {
    "idea_gen": IDEA_GEN_RUBRIC,
    "methodology": METHODOLOGY_RUBRIC,
    "paper_writer": PAPER_WRITER_RUBRIC,
}


def get_rubric(agent: str) -> Rubric:
    """Return the rubric for ``agent``, falling back to the generic rubric."""
    return _REGISTRY.get(agent, Rubric(
        agent=agent, dimensions=_GENERIC_RUBRIC.dimensions,
    ))


def register_rubric(rubric: Rubric, *, overwrite: bool = False) -> None:
    if rubric.agent in _REGISTRY and not overwrite:
        raise ValueError(f"rubric already registered for {rubric.agent}")
    _REGISTRY[rubric.agent] = rubric


def known_agents() -> tuple[str, ...]:
    """Agents that have a dedicated (non-generic) rubric."""
    return tuple(sorted(_REGISTRY.keys()))
