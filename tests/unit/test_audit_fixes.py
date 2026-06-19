"""Regression tests for the 2026-06-18 full-audit fixes.

Each test pins a specific defect the audit found so it can't silently come back:
sandbox path-containment escapes, secret-env leakage, the brace-counting JSON
parser, the handoff regex, the verification-gate false-negatives, and the
citation/year/leakage edge cases.
"""

from __future__ import annotations

import pytest

from autoscientist.runtime.handoff import parse_handoff
from autoscientist.runtime.payload_files import _extract_first_json_object
from autoscientist.tools import list_sandbox as ls_mod
from autoscientist.tools import read_sandbox_file as rsf_mod
from autoscientist.tools import write_file as wf_mod
from autoscientist.tools import write_release_file as wrf_mod
from autoscientist.tools.citation_check import _coerce_year
from autoscientist.tools.datasets import _safe_download_name
from autoscientist.tools.execute import _scrub_secret_env
from autoscientist.verify.leakage import check_target_leakage
from autoscientist.verify.pitfalls import run_pitfalls
from autoscientist.verify.stats import check_discrimination_floor

# --- Theme A: sandbox path containment (prefix-collision escape) -------------

def test_write_file_blocks_prefix_colliding_sibling(tmp_path):
    # `<id>/sandbox_evil` string-prefix-matches `<id>/sandbox` but is OUTSIDE it.
    with pytest.raises(wf_mod.SandboxEscape):
        wf_mod.write_file(
            path="../sandbox_evil/x.py", content="pwn",
            project_id="p1", projects_root=tmp_path,
        )
    # The sibling must not have been created.
    assert not (tmp_path / "p1" / "sandbox_evil").exists()


def test_write_file_allows_legitimate_nested_path(tmp_path):
    out = wf_mod.write_file(
        path="pkg/mod.py", content="x = 1",
        project_id="p1", projects_root=tmp_path,
    )
    assert out["written"] is True
    assert (tmp_path / "p1" / "sandbox" / "pkg" / "mod.py").read_text() == "x = 1"


def test_read_sandbox_file_blocks_prefix_colliding_sibling(tmp_path):
    with pytest.raises(rsf_mod.SandboxEscape):
        rsf_mod.read_sandbox_file(
            path="../sandbox_x/secret", project_id="p1", projects_root=tmp_path,
        )


def test_write_release_file_blocks_prefix_colliding_sibling(tmp_path):
    with pytest.raises(wrf_mod.ReleaseEscape):
        wrf_mod.write_release_file(
            path="../release_x/f", content="x",
            project_id="p1", projects_root=tmp_path,
        )


def test_list_sandbox_blocks_prefix_colliding_sibling(tmp_path):
    (tmp_path / "p1" / "sandbox").mkdir(parents=True)
    with pytest.raises(ls_mod.SandboxEscape):
        ls_mod.list_sandbox(project_id="p1", projects_root=tmp_path, subdir="../sandbox_x")


# --- Theme B: secret-env scrubbing -------------------------------------------

def test_scrub_secret_env_drops_keys_keeps_benign():
    env = {
        "PATH": "/usr/bin", "HOME": "/home/x", "CUDA_VISIBLE_DEVICES": "0",
        "ANTHROPIC_API_KEY": "sk-ant-xxx",
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx",
        "BIMCV_TOKEN": "t", "DB_PASSWORD": "p", "aws_secret_access_key": "s",
    }
    out = _scrub_secret_env(env)
    assert out == {"PATH": "/usr/bin", "HOME": "/home/x", "CUDA_VISIBLE_DEVICES": "0"}


# --- Theme C: JSON extractor respects string literals ------------------------

def test_extract_first_json_object_with_brace_inside_string():
    # A '}' inside a string value used to drive the depth counter to 0 early.
    text = 'noise {"files": [{"path": "a.py", "content": "s = {1, 2}\\nd = {}"}]} tail'
    obj = _extract_first_json_object(text)
    assert obj is not None
    assert obj["files"][0]["path"] == "a.py"
    assert obj["files"][0]["content"] == "s = {1, 2}\nd = {}"


# --- Theme C: handoff regex tolerates decorations, prefers last directive ----

def test_parse_handoff_tolerates_trailing_comment():
    ho = parse_handoff("verdict text\nHANDOFF: code_gen   # if verdict in {revise, block}", "code_review")
    assert ho is not None and ho.to_agent == "code_gen"


def test_parse_handoff_strips_placeholder_angle_bracket():
    ho = parse_handoff("HANDOFF: <repo_publisher if accept, else paper_writer>", "peer_reviewer")
    assert ho is not None and ho.to_agent == "repo_publisher"


def test_parse_handoff_uses_last_directive():
    ho = parse_handoff("HANDOFF: code_gen\nbody\nHANDOFF: test_gen\npayload", "test_gen")
    assert ho is not None and ho.to_agent == "test_gen"
    assert ho.payload == "payload"


def test_parse_handoff_done_is_terminal():
    ho = parse_handoff("all set\nHANDOFF: DONE", "peer_reviewer")
    assert ho is not None and ho.is_terminal


# --- Theme E: discrimination floor (below-chance point, no CI) ----------------

def test_discrimination_floor_below_chance_point_no_ci_is_not_pass():
    v = check_discrimination_floor([{"point_estimate": 0.30, "primary": True}])
    assert v.status == "fail"  # worse-than-chance, must not pass


def test_discrimination_floor_clear_signal_passes():
    v = check_discrimination_floor([{"point_estimate": 0.85, "primary": True}])
    assert v.status == "pass"


def test_discrimination_floor_near_chance_needs_human():
    v = check_discrimination_floor([{"point_estimate": 0.49, "primary": True}])
    assert v.status == "needs_human"


# --- Theme E: domain-configurable seed minimum --------------------------------

def _by_id(verdicts, cid):
    return next(v for v in verdicts if v.check_id == cid)


def test_multi_seed_clinical_tabular_requires_five():
    four = run_pitfalls(
        {"seeds": [1, 2, 3, 4], "report_seed_variance": True},
        domain="clinical_tabular",
    )
    assert _by_id(four, "multi_seed_reporting").status == "fail"
    five = run_pitfalls(
        {"seeds": [1, 2, 3, 4, 5], "report_seed_variance": True},
        domain="clinical_tabular",
    )
    assert _by_id(five, "multi_seed_reporting").status == "pass"


def test_multi_seed_medical_imaging_still_three():
    three = run_pitfalls(
        {"seeds": [1, 2, 3], "report_seed_variance": True},
        domain="medical_imaging",
    )
    assert _by_id(three, "multi_seed_reporting").status == "pass"


# --- Theme E: explicit infeasibility is authoritative -------------------------

def test_constraint_feasibility_explicit_false_with_zero_violations_fails():
    out = run_pitfalls(
        {"constraint_satisfied": False, "constraint_violations": 0},
        domain="numerical_optimization",
    )
    assert _by_id(out, "constraint_feasibility_verified").status == "fail"


# --- Theme E: two-class (non-0/1) leakage path --------------------------------

def test_leakage_runs_for_non_binary_encoded_two_class_target():
    # {-1, 1} target with a feature an outlier makes threshold-separable but
    # only weakly linearly correlated — the single-feature path must still fire.
    target = [-1, -1, -1, 1, 1, 1]
    features = {"leaky": [0.0, 1.0, 2.0, 3.0, 4.0, 1000.0]}
    v = check_target_leakage(features=features, target=target)
    assert v.status == "fail"


# --- Theme E: citation year coercion never crashes ----------------------------

@pytest.mark.parametrize(
    "value,expected",
    [(2017, 2017), ("2017", 2017), ("2017a", 2017), ("n.d.", None),
     ("in press", None), ("", None), (None, None), (True, None)],
)
def test_coerce_year(value, expected):
    assert _coerce_year(value) == expected


# --- Theme A (datasets): download filename can't traverse ---------------------

@pytest.mark.parametrize(
    "url,expected",
    [("http://h/data.zip", "data.zip"),
     ("http://h/a/b/file.tar?x=1", "file.tar"),
     ("http://h/dir/", "dir"),       # trailing slash stripped -> last segment
     ("http://h/..", "ds.bin"),      # traversal -> fallback, never writes to parent
     ("http://h/a/../..", "ds.bin")],
)
def test_safe_download_name(url, expected):
    assert _safe_download_name(url, "ds") == expected
