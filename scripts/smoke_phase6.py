"""Phase 6 smoke test — autoresearch / prompt optimization.

KICKOFF.md §9 Phase 6 — meta/* exists end-to-end:

  * Anchor JSON files load.
  * Rubrics define dimensions per agent (idea_gen carries the 4 KICKOFF dims).
  * Judge call routes through the ``judge`` agent (mock) and parses scored JSON.
  * A/B harness runs N variants x M anchors, persists eval_runs rows,
    picks a winner by mean score (variance-tiebreak).
  * Promoting the winner archives the previous canonical prompt and
    records a new ``prompt_versions`` row.
  * Meta-prompter round-trip via the mock fixture.

Zero LLM spend — every call routes through the mock provider.

    uv run python scripts/smoke_phase6.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_DB = _REPO / "smoke_phase6.db"
_RUNS = _REPO / "runs_smoke_phase6"
if _DB.exists():
    _DB.unlink()
if _RUNS.exists():
    shutil.rmtree(_RUNS)

os.environ["AUTOSCIENTIST_DB_PATH"] = str(_DB)


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def passed(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    raise AssertionError(msg)


def expect(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)
    passed(msg)


def main() -> int:
    from autoscientist.meta import ab_harness, anchors, eval_rubrics, rubrics, versioning
    from autoscientist.meta.meta_prompter import (
        PromptVariant,
        TraceSlice,
        propose_variants,
    )
    from autoscientist.runtime.config import load_config
    from autoscientist.state.db import open_db

    cfg = load_config(reload=True)

    # Force every Phase 6 LLM call through the mock provider.
    cfg.models["agents"]["judge"]["model"] = "mock_stub"
    cfg.models["agents"]["meta_prompter"]["model"] = "mock_stub"
    cfg.models["agents"]["idea_gen"]["model"] = "mock_stub"
    cfg.models["agents"]["paper_writer"]["model"] = "mock_stub"

    # Use a temp prompts dir so the smoke doesn't churn the real prompts/.
    smoke_prompts = _RUNS / "prompts"
    smoke_prompts.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_REPO / "prompts", smoke_prompts, dirs_exist_ok=True)
    cfg.default.setdefault("paths", {})["prompts_dir"] = str(
        smoke_prompts.relative_to(_REPO)
    )

    # ------------------------------------------------------------------
    # 1) Anchor files load.
    # ------------------------------------------------------------------
    section("Anchors: idea_gen + paper_writer load with valid schema")
    idea_anchors = anchors.load_anchor_set(smoke_prompts, "idea_gen")
    expect(len(idea_anchors) >= 3,
           f"idea_gen anchors >= 3 (got {len(idea_anchors)})")
    for a in idea_anchors:
        expect(a.agent == "idea_gen", f"anchor agent matches dir ({a.anchor_id})")

    paper_anchors = anchors.load_anchor_set(smoke_prompts, "paper_writer")
    expect(len(paper_anchors) >= 3,
           f"paper_writer anchors >= 3 (got {len(paper_anchors)})")

    # has_expected_keys structural pre-check
    a01 = idea_anchors.by_id("idea_gen_01_strong_lit_grounded")
    assert a01 is not None
    ok, missing = anchors.has_expected_keys(
        a01, {"ideas": [{"title": "x", "literature_gap": "y", "novelty": "med",
                          "feasibility": "high", "expected_experiments": ["e"],
                          "failure_modes": ["f"], "grounding": "strong"}]},
    )
    expect(ok, f"has_expected_keys finds all required keys (missing={missing})")
    ok2, missing2 = anchors.has_expected_keys(a01, {"ideas": []})
    expect(not ok2 and missing2, "empty ideas list flagged as missing keys")

    # ------------------------------------------------------------------
    # 2) Rubrics: idea_gen has the 4 KICKOFF dims.
    # ------------------------------------------------------------------
    section("Rubrics: idea_gen carries the 4 KICKOFF dimensions")
    r = rubrics.get_rubric("idea_gen")
    expect(r.agent == "idea_gen", "rubric agent matches")
    expect(set(r.dim_names) == {
        "novelty", "grounding", "feasibility", "counter_arg_quality",
    }, f"idea_gen dims (got {r.dim_names})")
    # Generic fallback for unknown agent
    rg = rubrics.get_rubric("nonexistent_agent")
    expect(rg.dim_names == ("schema_conformance", "completeness", "grounding"),
           f"generic rubric dims (got {rg.dim_names})")

    # weighted_total sanity
    t = r.weighted_total({"novelty": 4, "grounding": 5, "feasibility": 3, "counter_arg_quality": 4})
    expect(abs(t - 4.0) < 1e-9, f"weighted_total = 4.0 (got {t})")

    # ------------------------------------------------------------------
    # 3) Judge round-trip with deterministic mock scores.
    # ------------------------------------------------------------------
    section("Judge: returns parsed scores for one anchor (mock)")
    conn = open_db(_DB)
    try:
        # Register the existing on-disk idea_gen prompt so we have a
        # version_id to attach eval_runs to.
        existing = versioning.register_existing_prompt(
            conn, prompts_dir=smoke_prompts, agent_name="idea_gen",
        )
        assert existing is not None
        expect(existing.parent_version_id is None,
               "first registration has no parent")

        # Score with forced mock scores
        score = eval_rubrics.score_output(
            conn=conn,
            rubric=r,
            anchor=a01,
            candidate_prompt="(mock baseline)",
            candidate_output='{"ideas": [{"title": "mock"}]}',
            cfg=cfg,
            extra_envelope={"__mock_scores": {
                "novelty": 4, "grounding": 5, "feasibility": 3, "counter_arg_quality": 4,
            }},
        )
        expect(set(score.scores.keys()) == set(r.dim_names),
               f"all dims scored (got {set(score.scores.keys())})")
        expect(abs(score.total - 4.0) < 1e-9,
               f"weighted total = 4.0 (got {score.total})")
        expect(len(score.parse_errors) == 0,
               f"no parse errors (got {score.parse_errors})")

        eval_run_id = eval_rubrics.persist_eval_run(
            conn,
            agent="idea_gen",
            prompt_version_id=existing.version_id,
            anchor_id=a01.anchor_id,
            candidate_output='{"ideas": [{"title": "mock"}]}',
            score=score,
            note="phase6_smoke_judge",
        )
        expect(eval_run_id.startswith("er_"), f"eval_run_id format ({eval_run_id})")

        row = conn.execute(
            "SELECT total_score, agent_name FROM eval_runs WHERE eval_run_id = ?",
            (eval_run_id,),
        ).fetchone()
        expect(row is not None and abs(row["total_score"] - 4.0) < 1e-9,
               "eval_runs row persisted with the right total")
        conn.commit()
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # 4) A/B harness: two variants x two anchors -> winner picked by mean.
    # ------------------------------------------------------------------
    section("A/B harness: picks higher-mean variant")
    conn = open_db(_DB)
    try:
        v_strong = PromptVariant(
            prompt_text="# Variant STRONG\nYou are idea_gen. Tight schema.",
            hypothesis="raise grounding via tighter schema",
        )
        v_weak = PromptVariant(
            prompt_text="# Variant WEAK\nbe creative",
            hypothesis="loosen and see what happens",
        )
        # Pick two anchors so the harness has more than one row.
        sub_anchors = anchors.AnchorSet(
            agent="idea_gen",
            anchors=(
                idea_anchors.anchors[0],
                idea_anchors.anchors[1],
            ),
        )
        # Force scores: STRONG averages 4.5, WEAK averages 2.5.
        envelope = {
            sub_anchors.anchors[0].anchor_id: {
                "__mock_scores": {
                    "novelty": 4, "grounding": 5,
                    "feasibility": 5, "counter_arg_quality": 4,
                },
            },
            sub_anchors.anchors[1].anchor_id: {
                "__mock_scores": {
                    "novelty": 5, "grounding": 4,
                    "feasibility": 4, "counter_arg_quality": 5,
                },
            },
        }
        envelope_weak = {
            sub_anchors.anchors[0].anchor_id: {
                "__mock_scores": {
                    "novelty": 2, "grounding": 3,
                    "feasibility": 3, "counter_arg_quality": 2,
                },
            },
            sub_anchors.anchors[1].anchor_id: {
                "__mock_scores": {
                    "novelty": 3, "grounding": 2,
                    "feasibility": 2, "counter_arg_quality": 3,
                },
            },
        }

        # The harness invokes one variant at a time, but we need different
        # mock scores per variant. Run twice and merge results manually
        # so each variant lives under its own forced envelope.
        ab_strong = ab_harness.run_ab(
            conn=conn, agent="idea_gen", rubric=r,
            anchors=sub_anchors,
            variants=[v_strong],
            run_label="strong_only",
            cfg=cfg,
            judge_envelope_per_anchor=envelope,
        )
        ab_weak = ab_harness.run_ab(
            conn=conn, agent="idea_gen", rubric=r,
            anchors=sub_anchors,
            variants=[v_weak],
            run_label="weak_only",
            cfg=cfg,
            judge_envelope_per_anchor=envelope_weak,
        )
        expect(abs(ab_strong.variants[0].mean_score - 4.5) < 1e-9,
               f"strong variant mean = 4.5 (got {ab_strong.variants[0].mean_score})")
        expect(abs(ab_weak.variants[0].mean_score - 2.5) < 1e-9,
               f"weak variant mean = 2.5 (got {ab_weak.variants[0].mean_score})")

        # Combined run with both variants under the same anchor envelope.
        # The mock judge sees the same forced scores for both variants
        # under this envelope, so they tie. The harness should still
        # return a deterministic winner (lowest variance / parse errors).
        ab_combined = ab_harness.run_ab(
            conn=conn, agent="idea_gen", rubric=r,
            anchors=sub_anchors,
            variants=[v_weak, v_strong],
            run_label="combined",
            cfg=cfg,
            judge_envelope_per_anchor=envelope,
        )
        expect(ab_combined.winner_index >= 0, "winner picked")
        expect(len(ab_combined.variants) == 2, "two variant results")

        # Persistence: every (variant, anchor) → one eval_runs row,
        # plus one prompt_versions row per variant.
        n_eval = conn.execute(
            "SELECT COUNT(*) AS n FROM eval_runs WHERE note = 'combined'"
        ).fetchone()["n"]
        expect(n_eval == 4,
               f"combined run produced 2*2=4 eval_runs rows (got {n_eval})")

        n_versions = conn.execute(
            "SELECT COUNT(*) AS n FROM prompt_versions "
            "WHERE agent_name = 'idea_gen' AND note LIKE 'ab_harness:combined%'"
        ).fetchone()["n"]
        expect(n_versions == 2,
               f"combined run recorded 2 variant versions (got {n_versions})")
        conn.commit()
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # 5) Versioning: archive-on-write keeps the previous prompt.
    # ------------------------------------------------------------------
    section("Versioning: write_prompt archives the previous file")
    conn = open_db(_DB)
    try:
        original_text = (smoke_prompts / "idea_gen.md").read_text(encoding="utf-8")
        new_text = original_text + "\n\n# appended-by-phase6-smoke\n"

        before = versioning.list_versions(conn, "idea_gen")
        n_before = len(before)
        v_new = versioning.write_prompt(
            conn, prompts_dir=smoke_prompts,
            agent_name="idea_gen", new_text=new_text,
            note="phase6_smoke_promotion",
        )
        conn.commit()
        expect(v_new.parent_version_id is not None,
               "new version has a parent")
        expect(v_new.archived_path is not None,
               "previous prompt archived")
        archived = Path(v_new.archived_path)
        expect(archived.exists(), f"archive file exists at {archived}")
        expect(archived.read_text(encoding="utf-8") == original_text,
               "archive contents == original")
        expect((smoke_prompts / "idea_gen.md").read_text(encoding="utf-8") == new_text,
               "canonical file overwritten")

        # Idempotent re-write to the same content: still records a row,
        # but does not duplicate the archive.
        v_same = versioning.write_prompt(
            conn, prompts_dir=smoke_prompts,
            agent_name="idea_gen", new_text=new_text,
            note="phase6_smoke_noop",
        )
        conn.commit()
        expect(v_same.archived_path is None,
               "re-writing identical content skips the archive")

        after = versioning.list_versions(conn, "idea_gen")
        expect(len(after) == n_before + 2,
               f"two new versions recorded (got {len(after) - n_before})")
        conn.commit()
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # 6) Meta-prompter: returns parsed variants from mock fixture.
    # ------------------------------------------------------------------
    section("Meta-prompter: round-trip via mock returns parsed variants")
    conn = open_db(_DB)
    try:
        # Build a TraceSlice from the eval row above so the request
        # envelope embeds at least one trace.
        traces = [TraceSlice(
            anchor_id=a01.anchor_id,
            total_score=2.0,
            dim_scores={"novelty": 2, "grounding": 2,
                        "feasibility": 3, "counter_arg_quality": 1},
            candidate_output_excerpt='{"ideas": []}',
            judge_summary="No grounding to literature.",
        )]
        variants = propose_variants(
            conn=conn,
            agent="idea_gen",
            baseline_prompt="(baseline)",
            low_score_traces=traces,
            n_variants=3,
            rubric_dims=r.dim_names,
            cfg=cfg,
        )
        expect(len(variants) == 3,
               f"meta_prompter returned 3 variants (got {len(variants)})")
        expect(all(v.prompt_text and v.hypothesis for v in variants),
               "every variant has prompt_text and hypothesis")
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # 7) Spend invariant.
    # ------------------------------------------------------------------
    section("Spend invariant: zero spend (every call mocked)")
    conn = open_db(_DB)
    try:
        spend = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM budget_ledger"
        ).fetchone()["s"]
        expect(spend == 0.0, f"zero spend (got ${spend})")
    finally:
        conn.close()

    print("\n*** All Phase 6 smoke checks passed. ***")
    print(f"  DB:    {_DB}")
    print(f"  Runs:  {_RUNS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
