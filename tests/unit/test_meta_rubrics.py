"""Unit tests for autoscientist.meta.rubrics."""

from __future__ import annotations

import pytest

from autoscientist.meta import rubrics


def test_idea_gen_rubric_has_kickoff_dimensions():
    r = rubrics.get_rubric("idea_gen")
    assert set(r.dim_names) == {
        "novelty", "grounding", "feasibility", "counter_arg_quality",
    }


def test_methodology_rubric_dimensions():
    r = rubrics.get_rubric("methodology")
    assert set(r.dim_names) == {
        "experimental_completeness", "statistical_rigor", "pitfall_awareness",
    }


def test_paper_writer_rubric_dimensions():
    r = rubrics.get_rubric("paper_writer")
    assert set(r.dim_names) == {
        "structure", "citation_grounding", "clarity", "methodology_match",
    }


def test_unknown_agent_falls_back_to_generic():
    r = rubrics.get_rubric("unregistered")
    assert r.dim_names == ("schema_conformance", "completeness", "grounding")
    # Agent name still records the requested one (so callers can identify).
    assert r.agent == "unregistered"


def test_weighted_total_uniform_weights():
    r = rubrics.get_rubric("idea_gen")
    t = r.weighted_total({"novelty": 5, "grounding": 5,
                          "feasibility": 5, "counter_arg_quality": 5})
    assert t == 5.0


def test_weighted_total_partial_dims_scored():
    r = rubrics.get_rubric("idea_gen")
    # Missing dims contribute 0; total = (4+0+0+0)/4 = 1.0
    t = r.weighted_total({"novelty": 4})
    assert t == 1.0


def test_weighted_total_zero_when_no_dims():
    empty = rubrics.Rubric(agent="x", dimensions=())
    assert empty.weighted_total({"novelty": 5}) == 0.0


def test_register_rubric_disallows_overwrite_by_default():
    r = rubrics.Rubric(agent="idea_gen", dimensions=())
    with pytest.raises(ValueError, match="already registered"):
        rubrics.register_rubric(r)


def test_dimension_anchors_cover_one_to_five():
    for agent in ("idea_gen", "methodology", "paper_writer"):
        r = rubrics.get_rubric(agent)
        for d in r.dimensions:
            assert set(d.anchors.keys()) == {1, 2, 3, 4, 5}, (
                f"{agent}/{d.name} anchors missing scores"
            )
