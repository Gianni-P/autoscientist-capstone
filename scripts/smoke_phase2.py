"""Phase 2 smoke test.

Per KICKOFF.md §9 Phase 2:

  *scripts/smoke_phase2.py: runs idea_gen → idea_critic → methodology on a
   hardcoded prompt, asserts output structure.*

Strategy:
  * Override the three agents' models to ``mock_stub`` so the chain runs
    cost-free against the in-process mock provider (``clients/mock.py``).
  * Drive the chain by starting at ``idea_gen`` with a hardcoded payload
    based on the v1 test project (KICKOFF.md §8).
  * Read each agent's assistant message back from SQLite and assert that
    the JSON body before the ``HANDOFF:`` directive contains the keys the
    agent's prompt documents.
  * Verify caching: a second run with identical inputs hits cache for all
    three steps (zero spend, three cache_hit=1 ledger rows).
  * Verify handoff topology: the registry-driven runner does not log
    ``run.handoff_off_topology`` for the canonical chain.

Self-contained: writes to dedicated DB + runs dir (cleared on each run).

    uv run python scripts/smoke_phase2.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_DB = _REPO / "smoke_phase2.db"
_RUNS = _REPO / "runs_smoke_phase2"
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


# Keys each agent's prompt documents in its JSON body.
_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "idea_gen": ("ideas",),
    "idea_critic": ("critiques", "ranked_indices", "top_pick"),
    "methodology": ("plan",),
}

# Plan sub-keys methodology must produce.
_PLAN_KEYS = (
    "research_question",
    "hypotheses",
    "datasets",
    "baselines",
    "metrics",
    "experiments",
    "stats_plan",
    "pitfall_acks",
    "stop_conditions",
)

_HANDOFF_RE = re.compile(r"^\s*HANDOFF:\s*\S+\s*$", re.MULTILINE)


def _split_body(content: str) -> str:
    """Return the JSON body emitted before the first HANDOFF directive."""
    m = _HANDOFF_RE.search(content)
    return content[: m.start()].strip() if m else content.strip()


def main() -> int:
    from autoscientist.runtime.config import load_config
    from autoscientist.runtime.runner import run
    from autoscientist.state.db import open_db

    cfg = load_config()
    cfg.default.setdefault("paths", {})["runs_dir"] = str(_RUNS.relative_to(_REPO))

    # Force the three smoke agents through the mock provider so no real
    # API spend happens and the chain is deterministic.
    for name in ("idea_gen", "idea_critic", "methodology"):
        cfg.models["agents"][name]["model"] = "mock_stub"

    initial_payload = json.dumps(
        {
            "direction": (
                "Cross-institutional generalization of CNN-based pneumonia "
                "detection in chest radiographs as a function of training-set size."
            ),
            "lit_digest": {
                "summary": "Hardcoded smoke digest; Phase 3 wires real lit_review.",
                "key_works": [],
                "gaps": ["effect of training-set size on external AUROC is under-studied"],
                "consensus": ["CheXNet-style ResNet-50 fine-tuning is competitive on NIH"],
                "disagreements": [],
                "tools_needed": True,
            },
        }
    )

    section("Run 1: idea_gen -> idea_critic -> methodology")
    run_id_1 = run(
        starting_agent="idea_gen",
        project_id="smoke_phase2",
        initial_payload=initial_payload,
        enable_checkpoints=False,  # Phase 4 gating tested in smoke_phase4
        cfg=cfg,
    )
    print(f"  run_id_1 = {run_id_1}")

    conn = open_db(_DB)
    try:
        row = conn.execute(
            "SELECT status FROM runs WHERE run_id = ?", (run_id_1,)
        ).fetchone()
        expect(row is not None and row["status"] == "completed",
               "run 1 status == 'completed'")

        n_msgs = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE run_id = ? AND role = 'assistant'",
            (run_id_1,),
        ).fetchone()["n"]
        expect(n_msgs == 3, f"run 1 assistant message count == 3 (got {n_msgs})")

        n_misses = conn.execute(
            "SELECT COUNT(*) AS n FROM budget_ledger "
            "WHERE run_id = ? AND cache_hit = 0",
            (run_id_1,),
        ).fetchone()["n"]
        expect(n_misses == 3, f"run 1 all 3 calls were cache misses (got {n_misses})")

        # Per-agent JSON body schema checks.
        for agent in ("idea_gen", "idea_critic", "methodology"):
            row = conn.execute(
                "SELECT content FROM messages "
                "WHERE run_id = ? AND agent_name = ? AND role = 'assistant' "
                "ORDER BY created_at ASC LIMIT 1",
                (run_id_1, agent),
            ).fetchone()
            expect(row is not None, f"run 1 has assistant message for {agent}")
            body_text = _split_body(row["content"])
            try:
                body = json.loads(body_text)
            except json.JSONDecodeError as e:
                fail(f"{agent} body is not valid JSON: {e}\n---\n{body_text}\n---")
            for key in _REQUIRED_KEYS[agent]:
                expect(key in body, f"{agent} body has key '{key}'")

            if agent == "idea_gen":
                expect(isinstance(body["ideas"], list) and len(body["ideas"]) >= 3,
                       f"idea_gen produced >= 3 ideas (got {len(body['ideas'])})")
                first = body["ideas"][0]
                for k in ("title", "novelty", "feasibility", "expected_experiments",
                         "compute_estimate", "failure_modes"):
                    expect(k in first, f"idea_gen first idea has '{k}'")

            if agent == "idea_critic":
                ranked = body["ranked_indices"]
                expect(isinstance(ranked, list) and len(ranked) == len(body["critiques"]),
                       "idea_critic ranked_indices len == critiques len")
                expect(body["top_pick"] == ranked[0],
                       "idea_critic top_pick == ranked_indices[0]")

            if agent == "methodology":
                plan = body["plan"]
                for k in _PLAN_KEYS:
                    expect(k in plan, f"methodology.plan has key '{k}'")
                # Pitfall: at least one ack must mention patient-level split.
                ack_text = json.dumps(plan["pitfall_acks"]).lower()
                expect("patient-level" in ack_text,
                       "methodology pitfall_acks mentions 'patient-level' split")
    finally:
        conn.close()

    # Verify the structured JSONL log captured all three agents and didn't
    # warn about off-topology handoffs (canonical chain should stay in-bounds).
    log_path = cfg.runs_dir() / run_id_1 / "logs" / "run.jsonl"
    expect(log_path.exists(), f"JSONL log exists at {log_path}")
    log_lines = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    agents_logged = {ev.get("agent") for ev in log_lines if ev.get("event") == "run.agent_done"}
    for agent in ("idea_gen", "idea_critic", "methodology"):
        expect(agent in agents_logged, f"JSONL run.agent_done event present for {agent}")
    off_topology = [ev for ev in log_lines if ev.get("event") == "run.handoff_off_topology"]
    expect(off_topology == [],
           f"no off-topology handoff warnings (got {len(off_topology)})")

    section("Run 2: identical input -> all cache hits, zero spend")
    run_id_2 = run(
        starting_agent="idea_gen",
        project_id="smoke_phase2",
        initial_payload=initial_payload,
        enable_checkpoints=False,
        cfg=cfg,
    )
    print(f"  run_id_2 = {run_id_2}")
    conn = open_db(_DB)
    try:
        n_hits = conn.execute(
            "SELECT COUNT(*) AS n FROM budget_ledger "
            "WHERE run_id = ? AND cache_hit = 1",
            (run_id_2,),
        ).fetchone()["n"]
        expect(n_hits == 3, f"run 2 all 3 calls were cache hits (got {n_hits})")

        spend = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM budget_ledger "
            "WHERE run_id = ?",
            (run_id_2,),
        ).fetchone()["s"]
        expect(spend == 0.0, f"run 2 total spend == $0 (got ${spend})")
    finally:
        conn.close()

    print("\n*** All Phase 2 smoke checks passed. ***")
    print(f"  DB:    {_DB}")
    print(f"  Logs:  {cfg.runs_dir() / run_id_1 / 'logs' / 'run.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
