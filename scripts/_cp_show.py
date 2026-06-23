"""Dump a checkpoint's content for an operator decision (read-only).

Usage: python scripts/_cp_show.py <cp_id> [max_chars]
"""

import json
import sys

from autoscientist.checkpoints import manager
from autoscientist.runtime.config import load_config
from autoscientist.state.db import open_db

cp_id = sys.argv[1]
maxc = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

cfg = load_config()
conn = open_db(cfg.db_path())
cp = manager.get_checkpoint(conn, cp_id)
conn.close()
if cp is None:
    print("checkpoint not found:", cp_id)
    sys.exit(1)

p = cp.payload or {}
print(f"CP {cp.checkpoint_id} stage={cp.stage} status={cp.status} "
      f"{p.get('from_agent')}->{p.get('to_agent')}")
if p.get("summary"):
    print("=== SUMMARY ===")
    print(p["summary"])
if p.get("extra"):
    print("=== EXTRA ===")
    print(json.dumps(p["extra"]))
print("=== AGENT OUTPUT (raw) ===")
print((p.get("agent_output_raw") or "")[:maxc])
parsed = p.get("parsed")
if parsed is not None:
    print("=== PARSED ===")
    print(json.dumps(parsed, indent=2)[:maxc])
print("=== DEFAULT PAYLOAD -> next agent (snippet) ===")
print((p.get("default_payload") or "")[:1500])
