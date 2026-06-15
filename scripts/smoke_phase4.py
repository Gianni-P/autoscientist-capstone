"""Phase 4 smoke test — checkpoint manager + runner pause/resume + Q&A.

KICKOFF.md §7: the operator gates the chain at five stages. Phase 4
ships the manager + Streamlit UI; this smoke exercises the runtime
half (the UI is rendered by Streamlit and not exercised here).

Scenarios
---------
  Run A (approve): chain pauses at stage 1 after idea_critic. Operator
    approves. Resume. Chain pauses at stage 2 after methodology (which
    only emits HANDOFF: DONE in the mock fixture, so the stage-2
    checkpoint is terminal). Operator approves the terminal checkpoint.
    Run completes.

  Run B (modify): same chain, first checkpoint operator modifies the
    payload (override). Resume. Methodology agent sees the override as
    its inbound user payload — verifiable in messages table.

  Run C (reject): same chain, first checkpoint operator rejects. Resume
    is called and marks the run cancelled without invoking any further
    agents.

  Q&A: on a freshly-paused run, the operator can add questions;
    questions persist; the checkpoint stays pending; resolving still
    works after Q&A.

All agents go through the mock provider — zero LLM spend.

    uv run python scripts/smoke_phase4.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_DB = _REPO / "smoke_phase4.db"
_RUNS = _REPO / "runs_smoke_phase4"
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


def _seed_lit_cache(db_path: Path, direction: str) -> None:
    """Pre-populate the literature_search tool cache so lit_review's tool
    round runs offline. Mirrors smoke_phase3_5."""
    from autoscientist.state.db import open_db
    from autoscientist.tools import literature, tool_cache

    expected_query = direction[:200]
    expected_key = tool_cache.cache_key(
        {"query": expected_query, "limit": 5, "include_arxiv": False}
    )
    fake_papers = [
        literature.Paper(
            title="CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays",
            authors=["Rajpurkar, P.", "Irvin, J.", "Zhu, K."],
            year=2017,
            venue="arxiv",
            arxiv_id="1711.05225",
            doi=None,
            abstract="CheXNet is a 121-layer convolutional neural network...",
            citation_count=999,
            source="seed_smoke",
        ),
    ]
    conn0 = open_db(db_path)
    try:
        tool_cache.cache_put(
            conn0, "literature.search", expected_key,
            [p.to_dict() for p in fake_papers],
        )
    finally:
        conn0.close()


def main() -> int:
    from autoscientist.checkpoints import manager as cp_manager
    from autoscientist.runtime.config import load_config
    from autoscientist.runtime.runner import resume_run, run
    from autoscientist.state.db import open_db

    cfg = load_config()
    cfg.default.setdefault("paths", {})["runs_dir"] = str(_RUNS.relative_to(_REPO))

    # Force the chain through the mock provider — zero spend, deterministic.
    for name in ("lit_review", "idea_gen", "idea_critic", "methodology"):
        cfg.models["agents"][name]["model"] = "mock_stub"

    direction = (
        "Cross-institutional generalization of CNN-based pneumonia "
        "detection in chest radiographs as a function of training-set size."
    )
    initial_payload = json.dumps({
        "direction": direction,
        "context": {"domain": "medical_imaging"},
    })
    _seed_lit_cache(_DB, direction)

    # ------------------------------------------------------------------
    # Run A — approve path. lit_review (no checkpoint) → idea_gen (none) →
    # idea_critic (PAUSE at stage 1) → resume → methodology (PAUSE at
    # stage 2 with terminal handoff) → resume → completed.
    # ------------------------------------------------------------------
    section("Run A: approve at stages 1 and 2")
    run_id_a = run(
        starting_agent="lit_review",
        project_id="smoke_phase4_A",
        initial_payload=initial_payload,
        cfg=cfg,
    )
    print(f"  run_id_a = {run_id_a}")

    conn = open_db(_DB)
    try:
        row = conn.execute(
            "SELECT status FROM runs WHERE run_id = ?", (run_id_a,)
        ).fetchone()
        expect(row is not None and row["status"] == "paused",
               f"run A paused after first checkpoint (got status={row['status'] if row else None})")

        cps = cp_manager.list_for_run(conn, run_id_a)
        expect(len(cps) == 1, f"run A has exactly one checkpoint open (got {len(cps)})")
        cp1 = cps[0]
        expect(cp1.stage == 1, f"first checkpoint is stage 1 (got {cp1.stage})")
        expect(cp1.from_agent == "idea_critic",
               f"checkpoint 1 emitted by idea_critic (got from_agent={cp1.from_agent})")
        expect(cp1.to_agent == "methodology",
               f"checkpoint 1 routes to methodology (got to_agent={cp1.to_agent})")
        expect(cp1.status == "pending", f"checkpoint 1 pending (got {cp1.status})")
        expect(isinstance(cp1.parsed, dict) and "ranked_indices" in cp1.parsed,
               "checkpoint 1 parsed JSON includes ranked_indices")

        # No methodology messages yet — the run should not have invoked it.
        n_meth = conn.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE run_id = ? AND agent_name = 'methodology'",
            (run_id_a,),
        ).fetchone()["n"]
        expect(n_meth == 0,
               f"methodology not invoked while paused at stage 1 (got {n_meth} msgs)")

        cp_manager.resolve(
            conn, checkpoint_id=cp1.checkpoint_id, decision="approve"
        )
        conn.commit()
    finally:
        conn.close()

    # Resume — should run methodology, then pause at stage 2 (terminal).
    resume_run(run_id_a, cfg=cfg)

    conn = open_db(_DB)
    try:
        row = conn.execute(
            "SELECT status FROM runs WHERE run_id = ?", (run_id_a,)
        ).fetchone()
        expect(row is not None and row["status"] == "paused",
               f"run A paused again at stage 2 (got status={row['status'] if row else None})")

        cps = cp_manager.list_for_run(conn, run_id_a)
        expect(len(cps) == 2,
               f"run A now has two checkpoints (got {len(cps)})")
        cp2 = cps[-1]
        expect(cp2.stage == 2, f"second checkpoint is stage 2 (got {cp2.stage})")
        expect(cp2.from_agent == "methodology",
               f"checkpoint 2 emitted by methodology (got from_agent={cp2.from_agent})")
        expect(cp2.to_agent in {"DONE", ""},
               f"checkpoint 2 is terminal (to_agent={cp2.to_agent!r})")
        expect(cp2.status == "pending", f"checkpoint 2 pending (got {cp2.status})")

        # methodology fired exactly once.
        n_meth = conn.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE run_id = ? AND agent_name = 'methodology' AND role = 'assistant'",
            (run_id_a,),
        ).fetchone()["n"]
        expect(n_meth >= 1,
               f"methodology produced an assistant message after resume (got {n_meth})")

        cp_manager.resolve(
            conn, checkpoint_id=cp2.checkpoint_id, decision="approve"
        )
        conn.commit()
    finally:
        conn.close()

    resume_run(run_id_a, cfg=cfg)

    conn = open_db(_DB)
    try:
        row = conn.execute(
            "SELECT status, note FROM runs WHERE run_id = ?", (run_id_a,)
        ).fetchone()
        expect(row is not None and row["status"] == "completed",
               f"run A completed after final approve (got status={row['status'] if row else None})")
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # Run B — modify path. Operator overrides the payload at stage 1;
    # the methodology agent must see the override on its inbound user msg.
    # ------------------------------------------------------------------
    section("Run B: modify at stage 1 with payload override")
    run_id_b = run(
        starting_agent="lit_review",
        project_id="smoke_phase4_B",
        initial_payload=initial_payload,
        cfg=cfg,
    )
    print(f"  run_id_b = {run_id_b}")

    override_marker = "OPERATOR_OVERRIDE_PAYLOAD_8d2a"
    override_payload = json.dumps({
        "top_idea": {"title": override_marker},
        "critique": {"recommendation": "advance", "rationale": "operator override"},
    })

    conn = open_db(_DB)
    try:
        cp1b = cp_manager.list_for_run(conn, run_id_b)[0]
        expect(cp1b.stage == 1, "run B paused at stage 1")
        cp_manager.resolve(
            conn,
            checkpoint_id=cp1b.checkpoint_id,
            decision="modify",
            modified_payload=override_payload,
        )
        conn.commit()
    finally:
        conn.close()

    resume_run(run_id_b, cfg=cfg)

    conn = open_db(_DB)
    try:
        # methodology's inbound user message should be the override.
        meth_user = conn.execute(
            "SELECT content FROM messages "
            "WHERE run_id = ? AND agent_name = 'methodology' AND role = 'user' "
            "ORDER BY created_at ASC LIMIT 1",
            (run_id_b,),
        ).fetchone()
        expect(meth_user is not None,
               "methodology was invoked after modify resume")
        expect(override_marker in meth_user["content"],
               "methodology saw the operator-overridden payload")

        # The handoff audit message also captures the modify decision.
        handoff_rows = conn.execute(
            "SELECT content FROM messages WHERE run_id = ? AND role = 'handoff'",
            (run_id_b,),
        ).fetchall()
        expect(len(handoff_rows) >= 1,
               f"resume recorded a handoff audit message (got {len(handoff_rows)})")
        audit = json.loads(handoff_rows[0]["content"])
        expect(audit.get("decision") == "modify",
               f"audit message decision == 'modify' (got {audit.get('decision')})")
        expect(audit.get("next_agent") == "methodology",
               f"audit message next_agent == 'methodology' (got {audit.get('next_agent')})")
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # Run C — reject path. Resume after reject must mark the run cancelled
    # without invoking methodology.
    # ------------------------------------------------------------------
    section("Run C: reject at stage 1 cancels the run")
    run_id_c = run(
        starting_agent="lit_review",
        project_id="smoke_phase4_C",
        initial_payload=initial_payload,
        cfg=cfg,
    )
    print(f"  run_id_c = {run_id_c}")

    conn = open_db(_DB)
    try:
        cp1c = cp_manager.list_for_run(conn, run_id_c)[0]
        cp_manager.resolve(
            conn, checkpoint_id=cp1c.checkpoint_id, decision="reject",
            instructions="ideas don't ground in clinical literature",
        )
        conn.commit()
    finally:
        conn.close()

    resume_run(run_id_c, cfg=cfg)

    conn = open_db(_DB)
    try:
        row = conn.execute(
            "SELECT status, note FROM runs WHERE run_id = ?", (run_id_c,)
        ).fetchone()
        expect(row is not None and row["status"] == "cancelled",
               f"run C cancelled after reject (got status={row['status'] if row else None})")

        n_meth = conn.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE run_id = ? AND agent_name = 'methodology'",
            (run_id_c,),
        ).fetchone()["n"]
        expect(n_meth == 0,
               f"methodology never invoked after reject (got {n_meth} msgs)")
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # Q&A — questions persist and do not resolve the checkpoint.
    # ------------------------------------------------------------------
    section("Q&A: questions persist without resolving the checkpoint")
    run_id_d = run(
        starting_agent="lit_review",
        project_id="smoke_phase4_D",
        initial_payload=initial_payload,
        cfg=cfg,
    )
    print(f"  run_id_d = {run_id_d}")

    conn = open_db(_DB)
    try:
        cp1d = cp_manager.list_for_run(conn, run_id_d)[0]
        cp_manager.add_question(
            conn,
            checkpoint_id=cp1d.checkpoint_id,
            role="operator",
            content="why was idea 0 ranked above idea 2?",
        )
        cp_manager.add_question(
            conn,
            checkpoint_id=cp1d.checkpoint_id,
            role="assistant",
            content="(mock) Idea 0 had stronger expected effect size; idea 2 was confound-prone.",
            agent_used="checkpoint_qa",
            cost_usd=0.0,
        )
        conn.commit()

        qs = cp_manager.list_questions(conn, cp1d.checkpoint_id)
        expect(len(qs) == 2, f"two Q&A entries persisted (got {len(qs)})")
        expect(qs[0].role == "operator", "first entry is operator role")
        expect(qs[1].role == "assistant", "second entry is assistant role")

        cp_after = cp_manager.get_checkpoint(conn, cp1d.checkpoint_id)
        assert cp_after is not None
        expect(cp_after.status == "pending",
               f"checkpoint still pending after Q&A (got {cp_after.status})")

        cp_manager.resolve(
            conn, checkpoint_id=cp1d.checkpoint_id, decision="approve"
        )
        conn.commit()
    finally:
        conn.close()

    resume_run(run_id_d, cfg=cfg)

    conn = open_db(_DB)
    try:
        # Run D should have moved on to stage 2 just like run A.
        cps = cp_manager.list_for_run(conn, run_id_d)
        expect(len(cps) == 2, f"run D opened a stage-2 checkpoint after approve (got {len(cps)})")
        expect(cps[-1].stage == 2, f"second checkpoint is stage 2 (got {cps[-1].stage})")
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # Spend invariant: every LLM call in this smoke ran through the mock
    # provider, so total spend is exactly $0 across all runs.
    # ------------------------------------------------------------------
    section("Budget invariant: zero spend across all runs (mock provider)")
    conn = open_db(_DB)
    try:
        spend = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM budget_ledger"
        ).fetchone()["s"]
        expect(spend == 0.0, f"total spend == $0 (got ${spend})")
    finally:
        conn.close()

    print("\n*** All Phase 4 smoke checks passed. ***")
    print(f"  DB:    {_DB}")
    print(f"  Runs:  {_RUNS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
