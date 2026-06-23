"""Dogfood the new verify trio (completeness + provenance) against a run's
real plan + on-disk results. Read-only.

Usage: python scripts/_check_trio.py <plan_checkpoint_id> <runs_dir> [paper_text_file] [provenance_json_file]
  plan_checkpoint_id: a checkpoint whose default_payload carries {"plan": ...}
                      (e.g. the CP2 methodology->code_gen checkpoint)
  runs_dir:           dir holding *_summary.json result artifacts
"""

import glob
import json
import os
import sys

from autoscientist.checkpoints import manager
from autoscientist.runtime.config import load_config
from autoscientist.state.db import open_db
from autoscientist.verify import completeness, provenance

plan_cp_id = sys.argv[1]
runs_dir = sys.argv[2]
paper_file = sys.argv[3] if len(sys.argv) > 3 else None
prov_file = sys.argv[4] if len(sys.argv) > 4 else None

cfg = load_config()
conn = open_db(cfg.db_path())
cp = manager.get_checkpoint(conn, plan_cp_id)
conn.close()
_raw = cp.payload["default_payload"]  # may have a ```json fence / trailing text
_i = _raw.find("{")
plan, _ = json.JSONDecoder().raw_decode(_raw[_i:] if _i >= 0 else _raw)

results = {}
for f in sorted(glob.glob(os.path.join(runs_dir, "*_summary.json"))):
    try:
        results[os.path.basename(f)] = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        print(f"  (could not parse {f}: {e})")

state = {"plan": plan, "results": results}
if paper_file and os.path.exists(paper_file):
    state["paper_text"] = open(paper_file, encoding="utf-8").read()
if prov_file and os.path.exists(prov_file):
    state["provenance"] = json.load(open(prov_file, encoding="utf-8"))

print(f"results artifacts: {sorted(results)}")
print("=== completeness ===")
for v in completeness.run_completeness(state):
    print(f"  {v.check_id}: {v.status} ({v.severity}) — {v.detail}")
    if v.evidence:
        print("     evidence:", json.dumps(v.evidence)[:400])
print("=== provenance ===")
for v in provenance.run_provenance(state):
    print(f"  {v.check_id}: {v.status} ({v.severity}) — {v.detail}")
    if v.evidence:
        print("     evidence:", json.dumps(v.evidence)[:400])
