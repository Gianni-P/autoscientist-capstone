import sqlite3, re
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
rid = "run_78f82b18daef406481c1d80c6c199550"

# Show schema then dump all code_gen messages
for r in conn.execute("PRAGMA table_info(messages)"):
    print(r["name"], r["type"])
print("---")

# Check reasoning column
cols = [r["name"] for r in conn.execute("PRAGMA table_info(messages)")]
print("cols:", cols)

# Get the actual rows
for r in conn.execute(
    "SELECT rowid, role, content, " + ("reasoning, " if "reasoning" in cols else "") +
    "completion_tokens FROM messages WHERE run_id=? AND agent_name='code_gen' ORDER BY rowid",
    (rid,),
):
    print(f"\n--- rowid={r['rowid']} role={r['role']} ct={r['completion_tokens']} ---")
    print(f"content len: {len(r['content']) if r['content'] else 0}")
    print(f"content head: {r['content'][:500] if r['content'] else '(empty)'}")
    if "reasoning" in cols and r["reasoning"]:
        print(f"reasoning len: {len(r['reasoning'])}")
        print(f"reasoning tail (last 800): {r['reasoning'][-800:]}")
