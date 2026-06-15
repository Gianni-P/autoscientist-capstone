"""Headless smoke for the new console helpers.

Calls the activity-stream helpers and message-card data path against the
real autoscientist.db, without booting Streamlit. Verifies the SQL is
well-formed and the rendering code paths execute without raising.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Avoid the Streamlit decorator side-effects at import. We just want the
# pure helpers.
import json

from autoscientist.state.db import open_db


def main() -> int:
    db = _REPO / "autoscientist.db"
    if not db.exists():
        print("no DB yet — nothing to smoke")
        return 0

    conn = open_db(db)
    try:
        # _latest_active_run + _recent_messages
        from autoscientist.checkpoints.ui import (
            _latest_active_run, _recent_messages, _shorten,
        )
        run = _latest_active_run(conn)
        if run is None:
            print("no runs yet — nothing more to smoke")
            return 0
        print(f"latest run: {run['run_id']}  project={run['project_id']}  status={run['status']}")

        rows = _recent_messages(conn, run_id=run["run_id"], limit=20)
        print(f"  fetched {len(rows)} messages for that run")

        # Exercise the role-specific rendering branches as data structures
        # (without writing to the actual st.* widgets — just parse content).
        for r in rows:
            role = r["role"]
            content = r["content"] or ""
            if role == "tool":
                try:
                    d = json.loads(content)
                    name = d.get("name", "?")
                    err = d.get("error")
                    dur = d.get("duration_ms")
                    print(f"  TOOL {name:24s} dur={dur} err={err}")
                except json.JSONDecodeError:
                    print(f"  TOOL (unparseable content {len(content)}c)")
            elif role == "assistant":
                print(
                    f"  AST  {r['agent_name']:14s} "
                    f"{r['model'] or '?':28s} "
                    f"{r['prompt_tokens'] or 0:5d}→{r['completion_tokens'] or 0:5d} "
                    f"${float(r['cost_usd'] or 0.0):.4f} "
                    f"{r['latency_ms'] or 0}ms "
                    f"| {_shorten(content, 60)}"
                )
            elif role == "user":
                print(f"  USR  → {r['agent_name']:14s} | {_shorten(content, 60)}")
            elif role == "handoff":
                print(f"  HND  {r['agent_name']:14s} | {_shorten(content, 60)}")
            else:
                print(f"  ??   role={role} agent={r['agent_name']}")

        print("OK — helpers execute, SQL is well-formed.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
