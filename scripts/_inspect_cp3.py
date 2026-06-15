"""Read-only: inspect the pending CP3 for the limited-descent restart run."""
import json, sqlite3, textwrap

DB = "/home/gdp/autoscientist/autoscientist.db"
RUN = "run_bb0e896323f848ea81961a0d2852962f"
CP = "cp_2ba4b7c76f76465ab4371a01a7767372"

c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
c.row_factory = sqlite3.Row

print("=== checkpoint row ===")
row = c.execute(
    "SELECT checkpoint_id, stage, status, created_at, payload FROM checkpoints WHERE checkpoint_id=?",
    (CP,),
).fetchone()
pl = json.loads(row["payload"]) if row and row["payload"] else {}
print("stage", row["stage"], "status", row["status"])
print("payload keys:", list(pl.keys()))
for k, v in pl.items():
    s = v if isinstance(v, str) else json.dumps(v)
    print(f"\n--- payload[{k}] ({len(s)} chars) ---")
    print(textwrap.shorten(s.replace("\n", " "), width=1200))

print("\n\n=== substantive code_review verdicts (assistant, longest first) ===")
rows = c.execute(
    "SELECT created_at, length(content) AS n, content FROM messages "
    "WHERE run_id=? AND agent_name='code_review' AND role='assistant' "
    "ORDER BY n DESC LIMIT 3",
    (RUN,),
).fetchall()
for r in rows:
    print(f"\n--- {r['created_at']}  ({r['n']} chars) ---")
    print((r["content"] or "")[:1800])

c.close()
