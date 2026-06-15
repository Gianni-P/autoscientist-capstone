import sqlite3
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
rid = "run_78f82b18daef406481c1d80c6c199550"

print("=== last 8 code_gen assistant messages (most recent first) ===")
for r in conn.execute(
    "SELECT rowid, length(content) AS cc, length(reasoning) AS rc, completion_tokens, model, created_at "
    "FROM messages WHERE run_id=? AND agent_name='code_gen' AND role='assistant' "
    "ORDER BY rowid DESC LIMIT 8",
    (rid,),
):
    print(f"  rowid={r['rowid']:4d} content={r['cc']:6d} reasoning={r['rc']!s:>5}  ct={r['completion_tokens']!s:>5}  at={r['created_at']}")

# Look at the very last assistant message's text and reasoning preview
print("\n=== most recent assistant message ===")
last = conn.execute(
    "SELECT content, reasoning, completion_tokens FROM messages "
    "WHERE run_id=? AND agent_name='code_gen' AND role='assistant' "
    "ORDER BY rowid DESC LIMIT 1",
    (rid,),
).fetchone()
print(f"completion_tokens: {last['completion_tokens']}")
print(f"content len: {len(last['content']) if last['content'] else 0}")
print("content preview (first 600):")
print((last['content'] or '')[:600])
print(f"\nreasoning len: {len(last['reasoning']) if last['reasoning'] else 0}")
print("reasoning tail (last 1200):")
print((last['reasoning'] or '')[-1200:])
