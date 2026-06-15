# PERMANENT FIX — Move autoscientist off `/mnt/d` into WSL-native ext4

> **STATUS: EXECUTED & FINALIZED 2026-06-11.** The project lives at `/home/gdp/autoscientist`
> (ext4). Verification passed (`_check_state.py`, package import, rsync completeness check).
> The old `/mnt/d/autoscientist` rollback copy has been **deleted** at the user's request.
> The 104 GB `data/` (nih_chestxray14, padchest) was preserved by moving it to
> **`D:\autoscientist_data`** (`/mnt/d/autoscientist_data`); `/home/gdp/autoscientist/data`
> is a symlink to it. The behavioral rules in §1 (never `wsl --shutdown`; DB is WSL-only)
> still apply. The ethernet-flap issue in §2 has fixes staged — see the STATUS note there.

> **How to use this file:** open a fresh chat, point Claude Code at this file, and say
> **"do it"**. This document is self-contained — it has the full diagnosis, the exact
> migration steps, verification, rollback, and the guardrails. Execute the **PLAN**
> section. Confirm with the user before anything destructive (see GUARDRAILS).

---

## 0. TL;DR — what "do it" means

Copy the whole `autoscientist` project from `/mnt/d/autoscientist` (a Windows drive that
WSL reaches over the slow, wedge-prone 9p/drvfs layer) into the WSL distro's **native
ext4 filesystem** at **`/home/gdp/autoscientist`**, recreate the virtualenv there, fix a
couple of hardcoded paths, smoke-test, and switch all future work to the new location.
**Do not delete the old `/mnt/d/autoscientist` copy** — it stays as the rollback until the
new location is verified.

This is the permanent fix for the **Windows shutdown hang** (and the `wsl --shutdown`
hang, the SQLite-WAL corruption risk, and slow runs).

---

## 1. Why this is the fix (diagnosis, established 2026-06-11)

Symptom the user cares about most: **a normal Windows shutdown hung for 5+ minutes on the
"Shutting down" screen and required a hard power-off.**

Root cause chain, proven from the Windows event log:
- A shutdown earlier the same day (1:43 PM, WSL healthy) completed **cleanly**.
- The shutdown that hung (3:21 PM) happened **right after** a `wsl --shutdown` that wedged
  the WSL2/Hyper-V virtual machine (`vmmemWSL`). Windows then could not terminate that
  stuck VM during shutdown, so it stalled in the late (post-event-log) phase until power
  was cut (Kernel-Power 41 dirty-shutdown logged on next boot).
- The VM wedges because of **uninterruptible kernel I/O on the cross-OS `/mnt/d` mount**
  (9p/drvfs). Heavy, fsync-y workloads (SQLite WAL on `autoscientist.db`, per-file
  `write_file`/exec-log writes during a run) put a process into uninterruptible **D-state**
  on the 9p transport; the VM can't be torn down while that's pending → `wsl --shutdown`
  hangs → the next Windows shutdown hangs.
- Fast Startup is already OFF (`HiberbootEnabled=0`), so that common cause is ruled out.

**Moving the project to ext4 removes the cross-OS layer entirely**, so the I/O can't wedge
the VM. Same change also fixes the SQLite-WAL corruption risk (no more Windows/WSL contention)
and makes runs *much* faster (ext4 ≫ 9p for small-file I/O).

Two behavioral rules that came out of the same investigation (keep them after migrating):
- **Never run `wsl --shutdown` on this machine** — it reliably hangs here and is what wedges
  the VM. To reset WSL, use **Start → Restart** while WSL is healthy/stopped.
- It is safe to shut down whenever `wsl -l -v` shows the distro **Stopped** (no VM to wedge).

---

## 2. Facts the executor must know before touching anything

- **Distro:** `Ubuntu` (WSL2). **User:** `gdp`. **Home:** `/home/gdp`.
- **Project now:** `/mnt/d/autoscientist`  ==  `D:\autoscientist` on Windows.
- **Target:** `/home/gdp/autoscientist`  (Windows view after move: `\\wsl.localhost\Ubuntu\home\gdp\autoscientist`).
- **Config paths are RELATIVE to the project root** (`config/default.toml` → `db_path="autoscientist.db"`,
  `runs_dir="runs"`, `projects_dir="projects"`, `prompts_dir="prompts"`). So moving the tree
  needs **no config path edits** — everything resolves to the new root automatically.
- **Ollama runs as a systemd service INSIDE WSL** (`enabled`+`active`, `/usr/local/bin/ollama serve`,
  port 11434 via `wslrelay`). It is **independent of the project location** — its models live in
  `~/.ollama`, so the custom model `qwen3-coder-30b-64k` survives the move untouched. Do **not**
  reinstall/restart Ollama as part of this.
- **`uv` is the package manager** (the app runs via `uv run python ...`). It is installed in WSL.
- **The `.venv` must be recreated, not copied** — it has absolute paths baked in. Recreate with
  `uv sync` in the new location.
- This is a **git repo** but (as of writing) had no commits; `.git/` is small — copy it as-is.

### Pipeline fixes already applied (DO NOT redo or revert these — they work)
The orchestration is already fixed; runs reach `code_review`. These live in the repo and will
move with it:
- `src/autoscientist/agents/code_gen.py` → `tools=("pdf_parse","write_file")` (removed `execute`
  so it can't debug-spin without handing off).
- `config/qwen3-coder-64k.Modelfile` + Ollama model `qwen3-coder-30b-64k` (num_ctx 65536, fits the
  RTX 5090 at ~25 GB/100% GPU). `config/models.toml` → `[models.qwen3_coder].model_id =
  "qwen3-coder-30b-64k"`; `code_gen` `max_tool_rounds=30`, `test_gen=15`, `default_max_tokens=16384`.
- `src/autoscientist/runtime/runner.py` → `_FORWARD_TARGET` map + a backstop in `_drive_loop` that
  **forces the forward handoff** when qwen3-coder omits the `HANDOFF:` directive (was the cause of
  the "empty operator console" — run ended `completed` with 0 handoffs).

### Separate open issues (NOT part of this migration — note, don't fix here)
- **Ethernet flap → WSL loses its internet route** (`EHOSTUNREACH`), which makes `code_review`
  (Claude/network) fail with `APIConnectionError: [Errno 101] Network is unreachable` even though
  the Windows host is online. Durable fixes to discuss later: disable Realtek 2.5GbE NIC
  power-management / Energy-Efficient-Ethernet, or set WSL `networkingMode=mirrored`. The ext4
  migration does **not** fix this — flag it to the user as the next item.
  **STATUS 2026-06-11: both fixes staged.** (a) `C:\Users\gdp\.wslconfig` now sets
  `networkingMode=mirrored` + `dnsTunneling=true` + `autoProxy=true` — takes effect at the next
  full Windows restart (Start → Restart; never `wsl --shutdown`). (b) NIC root-cause script at
  `C:\Users\gdp\fix_nic_power.ps1` — run once from an **elevated** PowerShell; it disables the
  Realtek EEE / Green Ethernet / Gigabit Lite / Power Saving Mode features (evidence: NetworkProfile
  event log shows repeated SETUP-A80C drop/reconnect cycles while the machine was running, and all
  four flap-prone features were Enabled). If mirrored mode misbehaves (e.g. with Docker-in-WSL),
  revert by deleting the three networking lines from `.wslconfig` and restarting Windows.
- `AMDRyzenMasterDriverV27` fails to start on every boot (cosmetic; uninstall Ryzen Master if unused).

---

## 3. GUARDRAILS (hard rules)

1. **Never run `wsl --shutdown`** (or `wsl --terminate`) on this machine as part of this — it hangs.
2. **Do not delete or overwrite `/mnt/d/autoscientist`.** It is the rollback. Leave it fully intact.
   (User standing rule: no destructive cleanup of operator artifacts — `projects/`, `runs/`,
   `autoscientist.db`. Propose, don't delete.)
3. **`autoscientist.db` is WSL-only.** Never open it from Windows-side tools (breaks WAL). Inspect it
   only from WSL, read-only: `sqlite3 "file:<path>?mode=ro"` style.
4. **Do the copy only when no run is active** and the DB has no live writer (so the copy is consistent).
5. **Config is process-cached** — any config change needs a fresh process to take effect (the launcher
   scripts are fresh processes, so they pick it up).
6. Run all WSL commands from **inside WSL** (`wsl bash -lc "..."` from Windows, or a WSL shell).
   The git-bash "Bash" tool on Windows does NOT see `/mnt/...` or `/home/...`.

---

## 4. PLAN — execute these steps

### Step A — Pre-flight (read-only; confirm safe state)
Run in WSL and report results to the user before proceeding:
```bash
# no run in progress? (expect the distro idle; no long-running python on the project)
wsl.exe -l -v                      # from Windows: distro should be Running or Stopped, not wedged
ps -eo pid,cmd | grep -i '[a]utoscientist\|[s]treamlit\|[r]egen_code' || echo "no app processes"

# sizes & the data/ question (decides whether data is copied or symlinked)
cd /mnt/d/autoscientist
ls -ld data 2>/dev/null; readlink -f data 2>/dev/null
du -sh --exclude=.venv . 2>/dev/null          # total to copy (excl venv)
du -sh .venv data .git autoscientist.db projects runs 2>/dev/null
ls -la .env                                   # confirm secrets file exists
```
**Decide on `data/`:**
- If `data/` is a **symlink** → `rsync -a` copies the *link* (not the target); big datasets are NOT
  duplicated. Fine to include. (If the link is **relative**, re-point it to an absolute path after.)
- If `data/` is a **real dir and small** (≤ a few GB) → copy it.
- If `data/` is a **real dir and huge** (10s of GB, e.g. chest-xray) → **exclude it** from the copy and
  symlink it back: `ln -s /mnt/d/autoscientist/data /home/gdp/autoscientist/data`. (Data reads are
  sequential and far less wedge-prone than the DB/WAL random writes; keeping big read-only data on
  `/mnt/d` is an acceptable compromise. The math693a project likely doesn't use the big ML datasets
  at all — verify.)

### Step B — Copy the tree to ext4 (excluding venv; data per decision above)
```bash
mkdir -p /home/gdp/autoscientist
rsync -aHAX --info=progress2 \
  --exclude='.venv' \
  /mnt/d/autoscientist/ /home/gdp/autoscientist/
# (add  --exclude='data'  ONLY if you chose to symlink huge data; then create the symlink)
```
This preserves dotfiles (`.env`, `.gitignore`, `.git/`), permissions, and symlinks.

### Step C — Recreate the virtualenv in the new location
```bash
cd /home/gdp/autoscientist
uv sync          # rebuilds .venv from pyproject.toml + uv.lock with correct absolute paths
```

### Step D — Fix hardcoded `/mnt/d` references
Config is relative, but a few **diagnostic scripts hardcode the DB path**. Find and fix:
```bash
cd /home/gdp/autoscientist
grep -rn '/mnt/d/autoscientist' --include='*.py' --include='*.toml' --include='*.sh' . || echo "none"
```
Known offenders to update (change `/mnt/d/autoscientist` → `/home/gdp/autoscientist`):
`scripts/_check_state.py`, `scripts/_resume_diag.py` (the `DB = "/mnt/d/autoscientist/autoscientist.db"`
line). The main launcher `scripts/regen_code_qwen3.py` uses **relative/`cfg`-derived** paths and the
`PROJ` constant — no change needed. (Comments mentioning `/mnt/d` are harmless.)

### Step E — Smoke test (in the new location)
```bash
cd /home/gdp/autoscientist
set -a; source .env; set +a
# 1) DB reads & latest run state (after fixing its hardcoded path in Step D):
.venv/bin/python scripts/_check_state.py
# 2) confirm the model still resolves to the 64k variant (preflight prints it, does not spend $):
#    (only do a full run if the user wants to; code_gen/test_gen are local/$0, code_review hits the network)
uv run python scripts/regen_code_qwen3.py     # OPTIONAL full run; needs WSL internet (see open issue)
```
Expect `_check_state.py` to list the same runs as before (the DB came over intact).

### Step F — Switch the workflow to the new location
- **Windows file access** is now: `\\wsl.localhost\Ubuntu\home\gdp\autoscientist`
  (or `\\wsl$\Ubuntu\home\gdp\autoscientist`). Pin it in Explorer if useful.
- **Recommended:** run future Claude Code / dev work **from inside WSL** (cd to
  `/home/gdp/autoscientist`) so everything is native ext4 — no cross-OS layer at all. If the user keeps
  using Windows-side Claude Code, it must target the `\\wsl.localhost\...` path; expect `Glob`/`Grep`
  over `\\wsl$` to be slower, and remember the DB is WSL-only.
- Update any shortcuts/aliases that pointed at `/mnt/d/autoscientist` or `D:\autoscientist`.

---

## 5. Verification (how we know it worked)
1. `_check_state.py` from `/home/gdp/autoscientist` lists the prior runs (DB migrated intact).
2. A run launched from the new path writes to `/home/gdp/autoscientist/{runs,projects,autoscientist.db}`
   (ext4) — confirm with `ls -la`.
3. **The real proof:** after using the new location for a run, **Start → Shut Down completes cleanly**
   (no black-screen hang). `wsl -l -v` shows `Stopped` after, and next boot has **no Kernel-Power 41**.
4. Runs are noticeably faster (ext4 vs 9p).

## 6. Rollback
Nothing is destroyed. If anything is wrong, just keep using `/mnt/d/autoscientist` as before — it is
untouched. Only after the user confirms the new location works for real should you *propose* (never
auto-delete) archiving the old copy.

---

## 7. Pointers / continuity
- Persistent memory (this machine): `C:\Users\gdp\.claude\projects\D--autoscientist\memory\`
  - `project_pipeline_and_wsl_2026-06-11.md` — the orchestration fixes + WSL networking gotchas.
  - `project_runtime_env_gotchas.md` — config caching, DB-is-WSL-only rule.
  - `project_pipeline_standing.md`, `feedback_no_destructive_cleanup.md`.
- Key source: `src/autoscientist/runtime/runner.py` (`_FORWARD_TARGET` backstop, `_drive_loop`),
  `src/autoscientist/agents/code_gen.py` (trimmed tools), `config/models.toml` (qwen3_coder →
  `qwen3-coder-30b-64k`), `config/qwen3-coder-64k.Modelfile`, `scripts/regen_code_qwen3.py` (launcher).
- After this migration, the **next** thing to tackle is the **ethernet-flap / WSL-loses-internet**
  issue (so `code_review` stops failing with `APIConnectionError`). That is environmental, not code.

---
*Authored 2026-06-11 as a handoff so the migration can be executed in a fresh session.*
