import sqlite3, json, re
conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row
cp_id = "cp_59fb099fd9d24dc383727130964af40b"
row = conn.execute("SELECT payload FROM checkpoints WHERE checkpoint_id=?", (cp_id,)).fetchone()
payload = json.loads(row["payload"])
raw = payload["agent_output_raw"]

# Extract first balanced JSON object
start = raw.find("{")
depth = 0
end = -1
for i in range(start, len(raw)):
    c = raw[i]
    if c == "{": depth += 1
    elif c == "}":
        depth -= 1
        if depth == 0:
            end = i + 1; break
first_blob = raw[start:end]
print(f"first JSON: chars {start}-{end} (len={end-start})")
try:
    j = json.loads(first_blob)
    plan = j.get("plan", j)
    print(f"plan top-level keys: {list(plan.keys())}")
    print()
    print("=" * 70)
    print("Q:", plan.get("research_question", "?"))
    print("=" * 70)
    print("\nHYPOTHESES:")
    for h in plan.get("hypotheses", []):
        print(f"  [{h.get('id')}] ({h.get('predicted_direction')})")
        print(f"      {h.get('statement','')}")
    print("\nDATASETS:")
    for d in plan.get("datasets", []):
        print(f"  - {d.get('name')} [{d.get('role')}] split={d.get('split_strategy')}")
        if d.get("preprocessing"):
            print(f"      pp: {d['preprocessing'][:3]}{'...' if len(d['preprocessing'])>3 else ''}")
    print("\nBASELINES:")
    for b in plan.get("baselines", []):
        print(f"  - {b.get('name')}: {b.get('expected_metric')}  tol={b.get('tolerance')}")
    print("\nMETRICS:")
    for m in plan.get("metrics", []):
        primary = " [primary]" if m.get("primary") else ""
        print(f"  - {m.get('name')}{primary}  ci={m.get('ci_method')}")
    print("\nEXPERIMENTS:")
    for e in plan.get("experiments", []):
        print(f"  [{e.get('id')}] {e.get('describes','')}")
        intervention = e.get("intervention", "")
        if len(intervention) > 300:
            intervention = intervention[:300] + "..."
        print(f"      intervention: {intervention}")
        print(f"      seeds={e.get('n_seeds')}  compute={e.get('compute_budget','?')}")
    sp = plan.get("stats_plan", {})
    print(f"\nSTATS PLAN:")
    for k, v in sp.items():
        v_str = str(v)
        if len(v_str) > 250: v_str = v_str[:250] + "..."
        print(f"  {k}: {v_str}")
    print(f"\nPITFALL ACKS ({len(plan.get('pitfall_acks', []))}):")
    for p in plan.get("pitfall_acks", []):
        print(f"  - {p.get('pitfall','?')}")
        m = p.get('mitigation', '')
        if len(m) > 250: m = m[:250] + "..."
        print(f"      mitig: {m}")
    sc = plan.get("stop_conditions", {})
    print(f"\nSTOP CONDITIONS:")
    for k, v in sc.items():
        v_str = str(v)
        if len(v_str) > 250: v_str = v_str[:250] + "..."
        print(f"  {k}: {v_str}")
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
    print("--- first 2000 chars ---")
    print(first_blob[:2000])
