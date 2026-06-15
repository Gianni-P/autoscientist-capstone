"""Scratch diagnostic: inspect autoscientist.db health from inside WSL.

Read-only probe first (won't create/modify WAL), then reproduce the exact op
the console fails on (connect + PRAGMA journal_mode=WAL) to capture the error.
Safe to delete.
"""
import glob
import os
import sqlite3

DB = "/home/gdp/autoscientist/autoscientist.db"
RUN = "run_249a1eb42a2641%"

print("=== sidecar files ===")
for p in sorted(glob.glob(DB + "*")):
    st = os.stat(p)
    print(f"{p}  size={st.st_size}  uid={st.st_uid} gid={st.st_gid} mode={oct(st.st_mode & 0o777)}")

print("\n=== read-only open (no WAL switch) ===")
try:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
    print("journal_mode:", c.execute("PRAGMA journal_mode").fetchone()[0])
    print("runs:", c.execute("SELECT count(1) FROM runs").fetchone()[0])
    print("this run:", c.execute(
        "SELECT status, note FROM runs WHERE run_id LIKE ?", (RUN,)).fetchone())
    last = c.execute(
        "SELECT agent_name, model, created_at FROM messages "
        "ORDER BY created_at DESC LIMIT 1").fetchone()
    print("latest message:", last)
    c.close()
except Exception as e:
    print("RO open FAILED:", type(e).__name__, e)

print("\n=== reproduce console op: connect + PRAGMA journal_mode=WAL ===")
try:
    c = sqlite3.connect(DB, timeout=5)
    print("WAL switch ->", c.execute("PRAGMA journal_mode = WAL").fetchone()[0])
    c.close()
    print("OK: WAL switch succeeded (DB openable read-write)")
except Exception as e:
    print("WAL open FAILED:", type(e).__name__, e)
