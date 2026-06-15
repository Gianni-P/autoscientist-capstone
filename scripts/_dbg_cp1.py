import sqlite3, json, sys
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
cp_id = "cp_fadf92b9cc1743cf965466e79731ab34"
rid = "run_2f9ec1d3172642ddbff80740757283be"

# Spend summary on this run
print("=== run spend ===")
for r in conn.execute(
    "SELECT agent_name, COUNT(*) AS n, ROUND(SUM(cost_usd),5) AS spent, "
    "SUM(CASE cache_hit WHEN 1 THEN 1 ELSE 0 END) AS hits "
    "FROM budget_ledger WHERE run_id=? GROUP BY agent_name ORDER BY spent DESC",
    (rid,),
):
    print(f"  {r['agent_name']:15s} ${r['spent']:.5f}  ({r['n']} calls, {r['hits']} cache hits)")
tot = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE run_id=? AND cache_hit=0", (rid,)).fetchone()['s']
print(f"TOTAL REAL SPEND THIS RUN: ${tot:.5f}")

# Cumulative monthly spend
mtot = conn.execute("SELECT COALESCE(SUM(cost_usd),0) AS s FROM budget_ledger WHERE cache_hit=0").fetchone()['s']
print(f"CUMULATIVE MONTHLY SPEND : ${mtot:.5f}")

# Checkpoint payload
print("\n=== checkpoint ===")
row = conn.execute(
    "SELECT stage, payload, status FROM checkpoints WHERE checkpoint_id=?",
    (cp_id,),
).fetchone()
print(f"stage={row['stage']} status={row['status']}")
payload = json.loads(row["payload"]) if row["payload"] else {}
print(f"payload keys: {list(payload.keys())}")
print(f"from_agent={payload.get('from_agent')} -> to_agent={payload.get('to_agent')}")

# Pull idea_gen's final assistant text (the 5 candidates)
print("\n=== idea_gen final output (5 candidates JSON) ===")
ig = conn.execute(
    "SELECT content FROM messages WHERE run_id=? AND agent_name='idea_gen' AND role='assistant' "
    "ORDER BY rowid DESC LIMIT 1",
    (rid,),
).fetchone()
ig_text = ig["content"]
# Try to parse out the JSON
try:
    start = ig_text.find("{")
    depth = 0
    end = -1
    for i in range(start, len(ig_text)):
        c = ig_text[i]
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end > 0:
        igj = json.loads(ig_text[start:end])
        for idx, idea in enumerate(igj.get("ideas", [])):
            print(f"\n--- Idea {idx+1}: {idea.get('title','?')} ---")
            print(f"  summary:   {idea.get('summary','?')}")
            print(f"  gap:       {idea.get('literature_gap','?')}")
            print(f"  novelty:   {idea.get('novelty','?')}   feasibility: {idea.get('feasibility','?')}")
            print(f"  compute:   {idea.get('compute_estimate','?')}")
            exps = idea.get("expected_experiments", [])
            print(f"  experiments ({len(exps)}):")
            for e in exps:
                print(f"    - {e[:240]}{'...' if len(e)>240 else ''}")
            fms = idea.get("failure_modes", [])
            if fms:
                print(f"  failure_modes:")
                for fm in fms:
                    print(f"    - {fm[:200]}{'...' if len(fm)>200 else ''}")
except Exception as e:
    print(f"parse error: {e}")
    print(ig_text[:3000])

# Critique
print("\n\n=== idea_critic output ===")
ic = conn.execute(
    "SELECT content FROM messages WHERE run_id=? AND agent_name='idea_critic' AND role='assistant' "
    "ORDER BY rowid DESC LIMIT 1",
    (rid,),
).fetchone()
ic_text = ic["content"] if ic else ""
try:
    start = ic_text.find("{")
    depth = 0
    end = -1
    for i in range(start, len(ic_text)):
        c = ic_text[i]
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1; break
    icj = json.loads(ic_text[start:end])
    ranked = icj.get("ranked_indices", [])
    top = icj.get("top_pick", None)
    print(f"top_pick: idea index {top} (0-based)")
    print(f"ranked_indices: {ranked}")
    for c in icj.get("critiques", []):
        idx = c.get("idea_index")
        rec = c.get("recommendation")
        print(f"\n  Critique of Idea {idx+1 if isinstance(idx,int) else '?'}  [{rec}]")
        for x in c.get("concerns", [])[:5]:
            print(f"    concern: {x[:200]}")
        for x in c.get("kill_criteria", [])[:3]:
            print(f"    kill:    {x[:200]}")
        if c.get("rationale"):
            print(f"    rationale: {c['rationale'][:300]}")
    if icj.get("operator_questions"):
        print("\n  operator_questions:")
        for q in icj["operator_questions"]:
            print(f"    - {q}")
except Exception as e:
    print(f"parse error: {e}")
    print(ic_text[:3000])
