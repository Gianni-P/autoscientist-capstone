"""Phase 1 smoke test.

Per KICKOFF.md §12 done-criteria:
  1. Stub agents run through >= 3 handoffs (we use 4 for clarity).
  2. JSONL logs exist in runs/<run_id>/logs/run.jsonl.
  3. Second run hits cache for every step (zero spend, all cache_hit=1).
  4. Budget circuit-breaker refuses a call when monthly cap = $0.01.

Self-contained: writes to a dedicated smoke DB (cleared on each run) and
a dedicated runs directory. Run with:

    uv run python scripts/smoke_phase1.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Dedicated DB + runs dir so smoke tests don't pollute real history.
_DB = _REPO / "smoke_phase1.db"
_RUNS = _REPO / "runs_smoke_phase1"
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
    from autoscientist.runtime.budget import (
        BudgetConfig,
        BudgetExceeded,
        assert_can_spend,
    )
    from autoscientist.runtime.config import load_config
    from autoscientist.runtime.runner import run
    from autoscientist.state.db import open_db

    cfg = load_config()
    # Redirect runs to a dedicated dir for isolation.
    cfg.default.setdefault("paths", {})["runs_dir"] = str(_RUNS.relative_to(_REPO))

    section("Run 1: stub-agent chain (echo -> handoff x4 from COUNT 2)")
    run_id_1 = run(
        starting_agent="echo",
        project_id="smoke_phase1",
        initial_payload="COUNT 2",
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
            "SELECT COUNT(*) AS n FROM messages WHERE run_id = ?", (run_id_1,)
        ).fetchone()["n"]
        # 4 invocations x (user + assistant) = 8
        expect(n_msgs == 8, f"run 1 message count == 8 (got {n_msgs})")

        n_ledger = conn.execute(
            "SELECT COUNT(*) AS n FROM budget_ledger WHERE run_id = ?",
            (run_id_1,),
        ).fetchone()["n"]
        expect(n_ledger == 4, f"run 1 ledger entries == 4 (got {n_ledger})")

        n_misses = conn.execute(
            "SELECT COUNT(*) AS n FROM budget_ledger "
            "WHERE run_id = ? AND cache_hit = 0",
            (run_id_1,),
        ).fetchone()["n"]
        expect(n_misses == 4, f"run 1 all 4 calls were cache misses (got {n_misses})")
    finally:
        conn.close()

    log_path = cfg.runs_dir() / run_id_1 / "logs" / "run.jsonl"
    expect(log_path.exists(), f"JSONL log exists at {log_path}")
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    expect(len(lines) > 0, f"JSONL log non-empty ({len(lines)} lines)")
    for i, ln in enumerate(lines):
        try:
            json.loads(ln)
        except json.JSONDecodeError as e:
            fail(f"line {i} of {log_path} not valid JSON: {ln!r} ({e})")
    passed(f"all {len(lines)} JSONL log lines are valid JSON")
    events = {json.loads(ln).get("event") for ln in lines}
    expect("run.start" in events, "run.start event present in JSONL")
    expect("run.end" in events, "run.end event present in JSONL")

    section("Run 2: identical input -> all cache hits, zero spend")
    run_id_2 = run(
        starting_agent="echo",
        project_id="smoke_phase1",
        initial_payload="COUNT 2",
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
        expect(n_hits == 4, f"run 2 all 4 calls were cache hits (got {n_hits})")

        spend_row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM budget_ledger "
            "WHERE run_id = ?",
            (run_id_2,),
        ).fetchone()
        expect(spend_row["s"] == 0.0,
               f"run 2 total spend == $0 (got ${spend_row['s']})")
    finally:
        conn.close()

    section("Budget circuit-breaker: cap=$0.01 refuses a costly call")
    bcfg = BudgetConfig(monthly_cap_usd=0.01, hard_floor_buffer_usd=0.005)
    conn = open_db(_DB)
    try:
        # ~1k prompt + ~4k output at Sonnet pricing ≈ $0.064 — far above
        # the cap (0.01) minus buffer (0.005), so the pre-check must refuse.
        try:
            assert_can_spend(conn, bcfg, estimated_cost_usd=0.064)
            fail("BudgetExceeded was NOT raised — the circuit breaker is broken")
        except BudgetExceeded as e:
            passed(f"BudgetExceeded raised: {e}")
    finally:
        conn.close()

    print("\n*** All Phase 1 smoke checks passed. ***")
    print(f"  DB:    {_DB}")
    print(f"  Logs:  {cfg.runs_dir() / run_id_1 / 'logs' / 'run.jsonl'}")
    print(f"         {cfg.runs_dir() / run_id_2 / 'logs' / 'run.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
