"""Re-enter the pipeline at code_gen on qwen3-coder, reusing the approved
methodology from the cancelled run.

WHY THIS EXISTS
---------------
The run `run_249a1eb…` was *rejected* at the loop-cap CP3 (to_agent=code_gen),
which marks the run `cancelled` — and a cancelled run cannot be resumed
(runner.resume_run requires status='paused'). CP1 (idea) and CP2 (methodology)
were already approved, so rather than redo them we start a FRESH run beginning
at code_gen, feeding it the exact methodology payload code_gen received the
first time. Checkpoints stay ON, so the next CP3 (preliminary review) still
gates you.

MODEL: run this in a FRESH process so it loads the current models.toml
(code_gen/test_gen -> qwen3-coder:30b). The preflight block below prints the
resolved model so you can confirm qwen3 BEFORE any spend.

RUN IT (from WSL, fresh process, env sourced):
    cd /home/gdp/autoscientist
    set -a; source .env; set +a            # ANTHROPIC_API_KEY + OLLAMA_BASE_URL
    uv run python scripts/regen_code_qwen3.py
"""
from __future__ import annotations

import sqlite3

from autoscientist.runtime import runner
from autoscientist.runtime.config import load_config

PROJ = "math693a-limited-descent"
# Cancelled run whose approved methodology payload we reuse:
SRC_RUN = "run_249a1eb42a2641febc5dc7ec6b7f3daf"

cfg = load_config()

# --- Preflight: prove which model code_gen/test_gen resolve to in THIS process.
agents = cfg.models.get("agents", {})
models = cfg.models.get("models", {})
for a in ("code_gen", "test_gen"):
    alias = agents.get(a, {}).get("model")
    m = models.get(alias, {})
    print(f"[preflight] {a:9} -> alias={alias!r}  provider={m.get('provider')!r}  "
          f"model_id={m.get('model_id')!r}")
    if m.get("model_id") != "qwen3-coder:30b":
        print(f"[preflight] WARNING: {a} is NOT on qwen3-coder:30b — check models.toml / "
              f"that this is a fresh process.")

# --- Pull the exact payload code_gen got in the cancelled run (the approved plan).
db = str(cfg.db_path())
ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
ro.row_factory = sqlite3.Row
row = ro.execute(
    "SELECT content FROM messages "
    "WHERE run_id = ? AND agent_name = 'code_gen' AND role = 'user' "
    "ORDER BY created_at ASC LIMIT 1",
    (SRC_RUN,),
).fetchone()
ro.close()
if row is None:
    raise SystemExit(f"could not find code_gen input payload in source run {SRC_RUN}")
payload = row["content"]
print(f"[preflight] reusing methodology payload from {SRC_RUN}  ({len(payload)} chars)")
print("[preflight] starting a fresh run at code_gen (checkpoints ON) ...\n")

# --- Drive a fresh run. Synchronous: returns when it pauses at CP3 (or ends).
run_id = runner.run(
    starting_agent="code_gen",
    project_id=PROJ,
    initial_payload=payload,
    enable_checkpoints=True,
    cfg=cfg,
)
print("\nNEW_RUN_ID:", run_id)
print("Done — check status / resolve the new CP3 in the Streamlit console.")
