"""Unit tests for autoscientist.meta.eval_rubrics."""

from __future__ import annotations

import json

from autoscientist.meta import eval_rubrics, rubrics
from autoscientist.meta.anchors import Anchor


def _anchor() -> Anchor:
    return Anchor(
        anchor_id="t1", agent="idea_gen",
        input_payload="i", expected_summary="s",
    )


def _r() -> rubrics.Rubric:
    return rubrics.get_rubric("idea_gen")


def test_parse_well_formed_response():
    r = _r()
    text = json.dumps({
        "scores": {
            "novelty": {"score": 4, "rationale": "r1"},
            "grounding": {"score": 5, "rationale": "r2"},
            "feasibility": {"score": 3, "rationale": "r3"},
            "counter_arg_quality": {"score": 4, "rationale": "r4"},
        },
        "summary": "ok",
    })
    scores, _rationales, summary, errs = eval_rubrics._parse_judge_response(text, r)
    assert scores == {"novelty": 4, "grounding": 5,
                       "feasibility": 3, "counter_arg_quality": 4}
    assert summary == "ok"
    assert errs == []


def test_parse_extracts_embedded_json():
    r = _r()
    text = (
        "Some preamble.\n"
        + json.dumps({
            "scores": {
                "novelty": {"score": 3},
                "grounding": {"score": 3},
                "feasibility": {"score": 3},
                "counter_arg_quality": {"score": 3},
            },
            "summary": "",
        })
        + "\nSome trailing."
    )
    scores, _, _, errs = eval_rubrics._parse_judge_response(text, r)
    assert len(scores) == 4
    # Even though we extracted JSON from a longer text, no parse errors.
    assert errs == []


def test_parse_missing_dimension_recorded():
    r = _r()
    text = json.dumps({
        "scores": {
            "novelty": {"score": 4},
            # grounding missing
            "feasibility": {"score": 3},
            "counter_arg_quality": {"score": 4},
        },
        "summary": "",
    })
    scores, _, _, errs = eval_rubrics._parse_judge_response(text, r)
    assert "grounding" not in scores
    assert any("missing_dim:grounding" in e for e in errs)


def test_parse_score_out_of_range_dropped():
    r = _r()
    text = json.dumps({
        "scores": {
            "novelty": {"score": 9},   # invalid
            "grounding": {"score": 5},
            "feasibility": {"score": 3},
            "counter_arg_quality": {"score": 4},
        },
        "summary": "",
    })
    scores, _, _, errs = eval_rubrics._parse_judge_response(text, r)
    assert "novelty" not in scores
    assert any("score_out_of_range" in e for e in errs)


def test_parse_unparseable_response():
    r = _r()
    scores, _, _, errs = eval_rubrics._parse_judge_response("no json here", r)
    assert scores == {}
    assert errs and any("top_level_json" in e for e in errs)


def test_parse_bare_int_per_dim():
    """Judges that emit ``{dim: 4}`` rather than ``{dim: {score: 4}}``."""
    r = _r()
    text = json.dumps({
        "scores": {
            "novelty": 4, "grounding": 5,
            "feasibility": 3, "counter_arg_quality": 4,
        },
        "summary": "",
    })
    scores, _, _, errs = eval_rubrics._parse_judge_response(text, r)
    assert scores == {"novelty": 4, "grounding": 5,
                       "feasibility": 3, "counter_arg_quality": 4}
    assert errs == []
