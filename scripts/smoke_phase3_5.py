"""Phase 3.5 smoke test — LLM tool-use loop end to end.

Exercises the runner's tool round-trip against the mock provider:
  1. ``lit_review`` is invoked with mock_stub provider and tools wired.
  2. Mock provider's first turn for ``lit_review`` emits a tool_use call
     for ``literature_search``.
  3. Runner dispatches the tool via the registry; the literature module
     returns cached results (we pre-populate the cache so the smoke does
     not need network).
  4. Runner appends a tool_result and calls the mock again.
  5. Mock returns the canonical lit_review fixture (HANDOFF: idea_gen).
  6. Chain continues idea_gen → idea_critic → methodology → DONE.

Asserts:
  * Exactly one ``role: tool`` message recorded for lit_review.
  * That tool message has ``name: literature_search`` and a non-error output.
  * ``lit_review`` produced 2+ assistant messages (tool round + final).
  * Chain completed with status ``completed``.
  * Run 2 of the same payload hits cache for every LLM call (zero spend).

    uv run python scripts/smoke_phase3_5.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_DB = _REPO / "smoke_phase3_5.db"
_RUNS = _REPO / "runs_smoke_phase3_5"
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
    from autoscientist.runtime.config import load_config
    from autoscientist.runtime.runner import run
    from autoscientist.state.db import open_db
    from autoscientist.tools import literature, tool_cache

    cfg = load_config()
    cfg.default.setdefault("paths", {})["runs_dir"] = str(_RUNS.relative_to(_REPO))

    # Force the chain through mock provider — no real Claude/Ollama spend.
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

    # Pre-populate literature_search cache so the smoke is offline-safe.
    # The mock builds {"query": <direction[:200]>, "limit": 5}.
    expected_query = direction[:200]
    expected_key = tool_cache.cache_key({
        "query": expected_query, "limit": 5, "include_arxiv": False,
    })
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
    conn0 = open_db(_DB)
    try:
        tool_cache.cache_put(
            conn0, "literature.search", expected_key,
            [p.to_dict() for p in fake_papers],
        )
    finally:
        conn0.close()

    section("Run 1: lit_review tool round + chain to methodology")
    run_id_1 = run(
        starting_agent="lit_review",
        project_id="smoke_phase3_5",
        initial_payload=initial_payload,
        enable_checkpoints=False,  # Phase 4 gating tested in smoke_phase4
        cfg=cfg,
    )
    print(f"  run_id_1 = {run_id_1}")

    conn = open_db(_DB)
    try:
        # Run completed cleanly.
        row = conn.execute(
            "SELECT status FROM runs WHERE run_id = ?", (run_id_1,)
        ).fetchone()
        expect(row is not None and row["status"] == "completed",
               f"run 1 status == 'completed' (got {row['status'] if row else None})")

        # lit_review fired a tool round: at least one role=tool message under
        # this run + agent.
        n_tool_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE run_id = ? AND agent_name = 'lit_review' AND role = 'tool'",
            (run_id_1,),
        ).fetchone()["n"]
        expect(n_tool_rows >= 1,
               f"lit_review recorded >=1 tool message (got {n_tool_rows})")

        # The tool message is for literature_search and has an output.
        tool_row = conn.execute(
            "SELECT content FROM messages "
            "WHERE run_id = ? AND agent_name = 'lit_review' AND role = 'tool' "
            "ORDER BY created_at ASC LIMIT 1",
            (run_id_1,),
        ).fetchone()
        tool_payload = json.loads(tool_row["content"])
        expect(tool_payload.get("name") == "literature_search",
               f"tool message name == 'literature_search' (got {tool_payload.get('name')})")
        expect(tool_payload.get("error") is None,
               f"tool dispatch had no error (got {tool_payload.get('error')})")
        expect(isinstance(tool_payload.get("output"), dict)
               and tool_payload["output"].get("count", 0) >= 1,
               f"tool output has >=1 result (got {tool_payload.get('output')})")

        # lit_review produced >=2 assistant messages (tool round + final answer).
        n_lit_assistant = conn.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE run_id = ? AND agent_name = 'lit_review' AND role = 'assistant'",
            (run_id_1,),
        ).fetchone()["n"]
        expect(n_lit_assistant >= 2,
               f"lit_review has >=2 assistant rows (tool + final), got {n_lit_assistant}")

        # The final lit_review assistant message contains HANDOFF: idea_gen.
        last_lit = conn.execute(
            "SELECT content FROM messages "
            "WHERE run_id = ? AND agent_name = 'lit_review' AND role = 'assistant' "
            "ORDER BY created_at DESC LIMIT 1",
            (run_id_1,),
        ).fetchone()
        expect("HANDOFF: idea_gen" in last_lit["content"],
               "lit_review final content routes to idea_gen")

        # The chain reached methodology and emitted HANDOFF: DONE.
        last_meth = conn.execute(
            "SELECT content FROM messages "
            "WHERE run_id = ? AND agent_name = 'methodology' AND role = 'assistant' "
            "ORDER BY created_at DESC LIMIT 1",
            (run_id_1,),
        ).fetchone()
        expect(last_meth is not None,
               "methodology was reached and produced an assistant message")
        expect("HANDOFF: DONE" in last_meth["content"],
               "methodology emits HANDOFF: DONE")

        # All four agents fired (one tool round on lit_review only).
        agents_fired = {
            r["agent_name"] for r in conn.execute(
                "SELECT DISTINCT agent_name FROM messages "
                "WHERE run_id = ? AND role = 'assistant'",
                (run_id_1,),
            ).fetchall()
        }
        for a in ("lit_review", "idea_gen", "idea_critic", "methodology"):
            expect(a in agents_fired, f"agent {a} fired in run 1")
    finally:
        conn.close()

    section("Run 2: identical input -> all LLM calls cached, zero spend")
    run_id_2 = run(
        starting_agent="lit_review",
        project_id="smoke_phase3_5",
        initial_payload=initial_payload,
        enable_checkpoints=False,  # Phase 4 gating tested in smoke_phase4
        cfg=cfg,
    )
    print(f"  run_id_2 = {run_id_2}")

    conn = open_db(_DB)
    try:
        row = conn.execute(
            "SELECT status FROM runs WHERE run_id = ?", (run_id_2,)
        ).fetchone()
        expect(row is not None and row["status"] == "completed",
               "run 2 status == 'completed'")

        n_misses = conn.execute(
            "SELECT COUNT(*) AS n FROM budget_ledger "
            "WHERE run_id = ? AND cache_hit = 0",
            (run_id_2,),
        ).fetchone()["n"]
        expect(n_misses == 0, f"run 2 LLM cache misses == 0 (got {n_misses})")

        spend = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM budget_ledger "
            "WHERE run_id = ?",
            (run_id_2,),
        ).fetchone()["s"]
        expect(spend == 0.0, f"run 2 total spend == $0 (got ${spend})")
    finally:
        conn.close()

    print("\n*** All Phase 3.5 smoke checks passed. ***")
    print(f"  DB:    {_DB}")
    print(f"  Logs:  {cfg.runs_dir() / run_id_1 / 'logs' / 'run.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
