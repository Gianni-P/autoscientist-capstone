"""Read-only: detect phantom-import findings in code_review for a run.

Usage: python scripts/_inspect_run_findings.py <run_id>
"""
import re
import sqlite3
import sys

DB = "/home/gdp/autoscientist/autoscientist.db"
RUN = sys.argv[1] if len(sys.argv) > 1 else "run_e331d729887c4aedbd12ae60185e80b7"

PHANTOM = re.compile(
    r"does not exist|not defined|ImportError|unimportable|never defines?|"
    r"are absent|is absent|do not exist|no such (?:name|attribute)|phantom|"
    r"undefined name|cannot be imported|not found in",
    re.I,
)

c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
c.row_factory = sqlite3.Row

rows = c.execute(
    "SELECT created_at, content FROM messages WHERE run_id=? AND agent_name='code_review' "
    "AND role='assistant' ORDER BY created_at ASC",
    (RUN,),
).fetchall()

print(f"code_review assistant turns: {len(rows)}")
for r in rows:
    t = r["content"] or ""
    if len(t) < 300:
        continue
    n_block = t.count('"blocker"')
    hits = PHANTOM.findall(t)
    verd = re.search(r'"verdict"\s*:\s*"(\w+)"', t)
    ho = re.search(r"HANDOFF:\s*(\w+)", t)
    ts = r["created_at"][11:19]
    vlabel = verd.group(1) if verd else "?"
    hlabel = ho.group(1) if ho else "?"
    print(f"[{ts}] len={len(t):6d} blockers~{n_block} phantom_hits={len(hits)} "
          f"verdict={vlabel} handoff={hlabel}")
    m = PHANTOM.search(t)
    if m:
        s = max(0, m.start() - 80)
        snippet = t[s:m.end() + 50].replace("\n", " ")
        print(f"      e.g. …{snippet}…")

cols = [d[1] for d in c.execute("PRAGMA table_info(budget_ledger)").fetchall()]
print(f"\nbudget_ledger columns: {cols}")
tot = c.execute(
    "SELECT COALESCE(SUM(cost_usd),0) s, COUNT(*) n FROM budget_ledger WHERE cache_hit=0"
).fetchone()
print(f"lifetime billed: ${tot['s']:.4f} ({tot['n']} calls)")
c.close()
