"""Extract Phase 8 artifacts from the DB into on-disk files for operator review."""
import sqlite3, json, re, os
from pathlib import Path

RID = "run_78f82b18daef406481c1d80c6c199550"
PROJECT = "pneumonia-data-efficiency"
SANDBOX = Path(f"projects/{PROJECT}/sandbox")
ARTIFACTS = Path(f"projects/{PROJECT}/phase8_artifacts")
ARTIFACTS.mkdir(exist_ok=True, parents=True)
(ARTIFACTS / "src_proposed").mkdir(exist_ok=True)
(ARTIFACTS / "src_revision_partial").mkdir(exist_ok=True)
(SANDBOX / "src").mkdir(exist_ok=True)
(SANDBOX / "scripts").mkdir(exist_ok=True)

conn = sqlite3.connect("autoscientist.db")
conn.row_factory = sqlite3.Row


def parse_files_json(text: str):
    """Pull the first balanced top-level JSON object from text; return its 'files' list."""
    start = text.find("{")
    if start < 0:
        return None, None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None, None
    blob = text[start:end]
    try:
        return json.loads(blob), blob
    except Exception as e:
        return {"_parse_error": str(e), "_raw_preview": blob[:500]}, blob


# Find all code_gen assistant messages with substantial content (>5000 chars)
code_gen_msgs = list(conn.execute(
    "SELECT rowid, content, completion_tokens FROM messages "
    "WHERE run_id=? AND agent_name='code_gen' AND role='assistant' "
    "AND length(content) > 5000 ORDER BY rowid",
    (RID,),
))
print(f"Found {len(code_gen_msgs)} substantial code_gen responses")
for i, r in enumerate(code_gen_msgs):
    print(f"  [{i}] rowid={r['rowid']} content_chars={len(r['content'])} ct={r['completion_tokens']}")

# Extract files from the SECOND code_gen output (the complete one before revisions)
# That's the "final" full-files JSON before code_review found issues.
print(f"\n=== Extracting files from code_gen response [-2] (latest complete) ===")
if len(code_gen_msgs) >= 2:
    second_to_last = code_gen_msgs[-2]
    parsed, raw = parse_files_json(second_to_last["content"])
    if parsed and "files" in parsed:
        print(f"Files in proposed source:")
        for f in parsed["files"]:
            path = f.get("path")
            content = f.get("content")
            if not path or not content:
                continue
            full = ARTIFACTS / "src_proposed" / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            print(f"  wrote {full} ({len(content)} chars)")
            # Also seed the sandbox so it has runnable code (the operator can edit later)
            sb_full = SANDBOX / path
            sb_full.parent.mkdir(parents=True, exist_ok=True)
            sb_full.write_text(content, encoding="utf-8")
        print(f"\nentrypoint: {parsed.get('entrypoint')}")
        print(f"run_cmd: {parsed.get('run_cmd')}")
        print(f"dependencies: {parsed.get('dependencies')}")
        print(f"notes: {(parsed.get('notes') or '')[:500]}")
    else:
        print(f"could not parse files JSON; parsed: {str(parsed)[:200]}")

# Extract partial revisions from the LAST (truncated) code_gen output
print(f"\n=== Extracting partial revisions from code_gen response [-1] (truncated) ===")
if code_gen_msgs:
    last = code_gen_msgs[-1]
    parsed, raw = parse_files_json(last["content"])
    rev_dir = ARTIFACTS / "src_revision_partial"
    if parsed and "files" in parsed:
        for f in parsed["files"]:
            path = f.get("path")
            content = f.get("content")
            if not path or content is None:
                continue
            full = rev_dir / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            print(f"  wrote {full} ({len(content)} chars)")
    else:
        # Save the raw text since JSON didn't parse
        (rev_dir / "RAW_TRUNCATED_RESPONSE.txt").write_text(last["content"], encoding="utf-8")
        print(f"  could not parse; wrote raw 49857 chars to RAW_TRUNCATED_RESPONSE.txt")

# Extract code_review's findings
print(f"\n=== Extracting code_review feedback ===")
cr = conn.execute(
    "SELECT content FROM messages WHERE run_id=? AND agent_name='code_review' AND role='assistant' "
    "AND length(content) > 1000 ORDER BY rowid DESC LIMIT 1",
    (RID,),
).fetchone()
if cr:
    (ARTIFACTS / "code_review_findings.md").write_text(cr["content"], encoding="utf-8")
    print(f"  wrote {ARTIFACTS}/code_review_findings.md ({len(cr['content'])} chars)")

# Extract methodology plan as reference
m = conn.execute(
    "SELECT content FROM messages WHERE run_id=? AND agent_name='methodology' AND role='assistant' "
    "AND length(content) > 5000 ORDER BY rowid DESC LIMIT 1",
    (RID,),
).fetchone()
if m:
    (ARTIFACTS / "methodology_plan.md").write_text(m["content"], encoding="utf-8")
    print(f"  wrote {ARTIFACTS}/methodology_plan.md ({len(m['content'])} chars)")

# Final summary
print(f"\n=== Done. Artifacts directory: {ARTIFACTS.resolve()} ===")
print("Files written:")
for p in sorted(ARTIFACTS.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(ARTIFACTS)}: {p.stat().st_size} bytes")
print(f"\nSandbox seeded with proposed source. Listing:")
for p in sorted(SANDBOX.rglob("*")):
    if p.is_file() and "__pycache__" not in str(p):
        print(f"  {p.relative_to(SANDBOX)}: {p.stat().st_size} bytes")
