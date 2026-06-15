import sqlite3, json, re
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
rid = "run_78f82b18daef406481c1d80c6c199550"

# All code_gen messages in this run, in order
print("=== code_gen messages ===")
for r in conn.execute(
    "SELECT rowid, role, length(content) AS clen, model, prompt_tokens, completion_tokens "
    "FROM messages WHERE run_id=? AND agent_name='code_gen' ORDER BY rowid",
    (rid,),
):
    print(f"  rowid={r['rowid']:5d} {r['role']:10s} len={r['clen']:6d}  model={r['model']} pt={r['prompt_tokens']} ct={r['completion_tokens']}")

# Show the final assistant output
print("\n--- LAST code_gen assistant content ---")
row = conn.execute(
    "SELECT content FROM messages WHERE run_id=? AND agent_name='code_gen' AND role='assistant' "
    "ORDER BY rowid DESC LIMIT 1",
    (rid,),
).fetchone()
text = row["content"]
print(f"length: {len(text)}")
print(f"ends with: {repr(text[-200:])}")
m = re.search(r"HANDOFF:\s*(\w+)\b", text)
print(f"HANDOFF found: {m.group() if m else 'NO'}")
print("\n--- first 2500 chars ---")
print(text[:2500])
print("\n--- last 2500 chars ---")
print(text[-2500:])
