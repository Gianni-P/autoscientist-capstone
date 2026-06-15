"""Unit tests for autoscientist.meta.meta_prompter parsing logic."""

from __future__ import annotations

import json

from autoscientist.meta.meta_prompter import TraceSlice, _parse_variants


def test_parse_variants_basic():
    text = json.dumps({"variants": [
        {"prompt_text": "v1", "hypothesis": "h1"},
        {"prompt_text": "v2", "hypothesis": "h2"},
    ]})
    out = _parse_variants(text)
    assert [v.prompt_text for v in out] == ["v1", "v2"]
    assert [v.hypothesis for v in out] == ["h1", "h2"]


def test_parse_variants_extracts_embedded_json():
    text = "preamble\n" + json.dumps({
        "variants": [{"prompt_text": "v", "hypothesis": "h"}],
    })
    out = _parse_variants(text)
    assert len(out) == 1
    assert out[0].prompt_text == "v"


def test_parse_variants_invalid_json():
    assert _parse_variants("not json") == []


def test_parse_variants_skips_malformed_entries():
    text = json.dumps({"variants": [
        {"prompt_text": "ok", "hypothesis": "h"},
        {"hypothesis": "missing prompt"},   # skipped
        "string entry",                      # skipped
        {"prompt_text": "", "hypothesis": "empty"},   # skipped
    ]})
    out = _parse_variants(text)
    assert [v.prompt_text for v in out] == ["ok"]


def test_parse_variants_missing_hypothesis_defaults_to_empty():
    text = json.dumps({"variants": [
        {"prompt_text": "v", "hypothesis": None},
    ]})
    out = _parse_variants(text)
    assert out[0].hypothesis == ""


def test_trace_slice_truncates_output():
    from autoscientist.meta.eval_rubrics import RubricScore

    long_output = "x" * 5000
    rs = RubricScore(
        agent="a", anchor_id="aid",
        scores={"d": 1.0}, rationales={}, summary="", total=1.0,
        judge_model="m",
    )
    ts = TraceSlice.from_score(rs, long_output, output_chars=200)
    assert len(ts.candidate_output_excerpt) == 200
