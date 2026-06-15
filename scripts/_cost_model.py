"""Read-only cost model: what does the local code_gen/test_gen + Sonnet
code_review loop cost, and what would moving code_gen/test_gen to a cloud
model cost? Prices per 1M tokens (input/output)."""
import sqlite3
import sys

DB = "/home/gdp/autoscientist/autoscientist.db"
RUN = sys.argv[1] if len(sys.argv) > 1 else "run_bb0e896323f848ea81961a0d2852962f"

PRICES = {  # $/1M tokens (input, output)
    "opus":   (5.0, 25.0),
    "sonnet": (3.0, 15.0),
    "haiku":  (1.0, 5.0),
}

c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
c.row_factory = sqlite3.Row

print(f"=== per-agent token + cost for {RUN} (one full 3-cycle run) ===")
rows = c.execute(
    "SELECT agent_name, provider, COUNT(*) n, "
    "  COALESCE(SUM(prompt_tokens),0) pin, COALESCE(SUM(completion_tokens),0) pout, "
    "  ROUND(COALESCE(SUM(cost_usd),0),4) cost "
    "FROM budget_ledger WHERE run_id=? AND cache_hit=0 "
    "GROUP BY agent_name, provider ORDER BY cost DESC, pin DESC",
    (RUN,),
).fetchall()

local_in = local_out = 0
review_cost = 0.0
for r in rows:
    print(f"  {r['agent_name']:14} {r['provider']:7} calls={r['n']:3}  "
          f"in={r['pin']:>8}  out={r['pout']:>7}  cost=${r['cost']:.4f}")
    if r["provider"] == "ollama":
        local_in += r["pin"]
        local_out += r["pout"]
    else:
        review_cost += r["cost"]

print(f"\n  LOCAL (code_gen+test_gen) tokens: in={local_in:,}  out={local_out:,}  (billed $0)")
print(f"  CLOUD code_review actual cost this run: ${review_cost:.4f}")

print("\n=== if code_gen+test_gen ran on a cloud model instead of local ===")
for name, (pin, pout) in PRICES.items():
    add = local_in / 1e6 * pin + local_out / 1e6 * pout
    print(f"  {name:6}: +${add:.4f} added to this run "
          f"(would make run ~${review_cost + add:.2f} vs ${review_cost:.2f} now)")

print("\n=== break-even framing ===")
print(f"  Current run buys: 3 Sonnet review cycles of non-converging local code = ${review_cost:.2f}")
print(f"  Per review cycle ≈ ${review_cost/3:.2f}. If a stronger code_gen converged in 1 cycle")
print(f"  instead of 3, you'd SAVE ~${review_cost*2/3:.2f} in review but ADD the code_gen cost above.")
c.close()
