# autoscientist

Multi-agent research pipeline that produces academic papers, supplementary materials, and reproducible code repositories from high-level research directions in scoped domains.

See `KICKOFF.md` for the full project brief, architectural principles, build phases, and target ceiling.

## Quickstart (operator)

```bash
# 1. Set API key (in WSL)
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc && source ~/.bashrc

# 2. Sync deps
cd /mnt/d/autoscientist
uv sync

# 3. Smoke test the runtime (no API spend on second run — should hit cache)
uv run python scripts/smoke_phase1.py    # runtime + cache + budget circuit-breaker
uv run python scripts/smoke_phase2.py    # idea_gen -> idea_critic -> methodology, mock-driven
uv run python scripts/smoke_phase3.py    # tool integrations (literature, pdf, exec, datasets, latex, citations)
uv run python scripts/smoke_phase3_5.py  # full LLM tool-use loop (lit_review calls literature_search)
uv run python scripts/smoke_phase4.py    # checkpoint manager + Streamlit page rendering
uv run python scripts/smoke_phase5.py    # verify harness (leakage, baseline repro, stats, pitfalls)
uv run python scripts/smoke_phase6.py    # autoresearch / prompt optimization (anchors, rubrics, A/B, versioning)
uv run python scripts/smoke_phase7.py    # domain hardening (pneumonia-relevant medical-imaging pitfalls)
uv run python scripts/smoke_mcp_github.py # GitHub MCP integration (offline; fake server, no Docker/network)

# 4. Launch the operator console (auto-refreshing activity stream + Pause/Resume)
uv run streamlit run src/autoscientist/checkpoints/ui.py
```

For a live agent run rather than smoke testing, see
[Running a project end-to-end](#running-a-project-end-to-end) below.

## Running a project end-to-end

The smoke tests above exercise the substrate. A real run drives the
agent chain against a project (e.g. `pneumonia-data-efficiency`) and
pauses at five HITL checkpoints — see KICKOFF.md §7 and the per-project
runbook at `scripts/run_phase8.md` for the full pre-flight.

### Launch

Two WSL terminals. **Terminal A** runs the chain; **Terminal B** runs
the Streamlit console.

```bash
# Terminal A — the runner
cd /mnt/d/autoscientist
set -a; source .env; set +a   # picks up ANTHROPIC_API_KEY etc.
PAYLOAD=$(cat projects/<project-id>/kickoff_payload.json) # project-id = pnuemonia-data-efficiency-v2
uv run python -m autoscientist.runtime.runner \
    --agent lit_review \
    --project <project-id> \  
    --payload "$PAYLOAD"
# Note the printed run_id — you'll need it for resume.
```

```bash
# Terminal B — the operator console
cd /mnt/d/autoscientist
set -a; source .env; set +a
uv run streamlit run src/autoscientist/checkpoints/ui.py
# Open http://localhost:8501
```

If `uv` is not on PATH, `.venv/bin/python` and `.venv/bin/streamlit`
are direct equivalents.

### Pause and resume

The console's **Live activity** panel has a control row tied to the
most recent active run:

| Run state | Button | Behavior |
|---|---|---|
| 🟢 running, no pause pending | **⏸ Pause** | Sets the pause flag. Runner stops at the next agent boundary (after the current agent finishes its tool loop — can take minutes for long agents). |
| 🟢 running, pause pending | "⏸ Pause requested…" (disabled) + **Cancel pause** | Pause is queued; you can still cancel before the runner honors it. |
| 🟡 paused (manual) | **▶ Resume** | Reads saved state and resumes via a background thread. |
| 🟡 paused at a checkpoint | (no pause/resume buttons) | Open the pending checkpoint to approve / modify / reject — that resumes the chain. |

The activity panel and pending-checkpoints list auto-refresh every
2–3 seconds via `st.fragment(run_every=…)`; forms, expanders, and
scroll position survive the refresh.

CLI equivalents (use these if the UI is down or you're scripting):

```bash
# Request a pause on a running run
uv run python -c "
from autoscientist.state.db import open_db
from autoscientist.runtime import control
with open_db('autoscientist.db') as conn:
    control.request_pause(conn, '<run_id>')
    conn.commit()
"

# Resume any paused run (manual-pause or checkpoint-resolved)
uv run python -m autoscientist.runtime.runner --resume <run_id>
```

### Abort

Three ways, depending on how cleanly you need to stop:

1. **Reject a pending checkpoint** in the UI — runner marks the run
   `cancelled` on the next resume call and exits.
2. **`Ctrl-C` in Terminal A** — the runner catches `KeyboardInterrupt`,
   marks the run `cancelled` with note `operator interrupt`, and
   closes cleanly. The next agent's in-flight LLM call may still be
   billed.
3. **`uv run python scripts/_budget_status_v2.py`** in either terminal
   for a current-month spend snapshot; pair with rejecting at the next
   checkpoint to abort once the agent finishes.

### Checking spend mid-run

```bash
uv run python scripts/_budget_status_v2.py
# real spend lifetime: $X.XXXX
# per-agent (real $ only):
#   test_gen   $XX.XXXX  (NNN calls)
#   ...
# current month (YYYY-MM) spend: $X.XXXX / $150 cap
```

The runtime refuses new API calls within $5 of the monthly cap. That
limit is non-negotiable (KICKOFF.md §2) and lives in
`src/autoscientist/runtime/budget.py`.

## Publishing to GitHub (repo_publisher)

The terminal `repo_publisher` agent writes the curated release tree to
`projects/<project_id>/release/` **and** publishes it to a real GitHub
repository via the official [GitHub MCP server](https://github.com/github/github-mcp-server).
The bridge that connects autoscientist's synchronous tool loop to MCP servers
lives in `src/autoscientist/clients/mcp_bridge.py`; server definitions live in
`config/mcp.toml`.

**Setup (one-time):** create a GitHub PAT (fine-grained: *Administration*
read/write to create repos + *Contents* read/write to push; or a classic
`repo`-scoped token) and export it:

```bash
echo 'export GITHUB_PERSONAL_ACCESS_TOKEN=github_pat_...' >> ~/.bashrc && source ~/.bashrc
# Validate live wiring with one read-only call (creates nothing, costs $0):
uv run python scripts/dry_run_github_mcp.py
```

By default the integration uses GitHub's **remote** MCP server over Streamable
HTTP (`https://api.githubcopilot.com/mcp/`) — no Docker required. To use the
**local** Docker server instead, set `transport = "stdio"` in `config/mcp.toml`;
that path needs Docker reachable without sudo (`sudo usermod -aG docker $USER`
then `wsl.exe --shutdown`).

**Graceful degradation:** if the token is missing or the server is unreachable,
`repo_publisher` logs it, skips the GitHub push, and still writes the local
release tree — a failed publish never fails the run. Note the GitHub MCP server
has no release/tag tool, so tagging a release (e.g. `gh release create v1.0`)
remains a manual follow-up.

## Domain expertise disclaimer

This pipeline has no real medical/clinical knowledge baked in. The pitfall checklists in `config/domains/` substitute for expertise on common mistakes; they do not substitute for clinical judgment. Operators must inject domain taste at the five mandatory human checkpoints.

## Budget

The pipeline tracks every API call against a monthly cap (default $150) and refuses new calls within $5 of the cap. Budget enforcement is non-negotiable and lives in `src/autoscientist/runtime/budget.py`.

## Layout

See `KICKOFF.md` §5.

## Phase 8 — end-to-end on `pneumonia-data-efficiency`

When you are ready for the live run:

1. `uv run python scripts/dry_run_phase8.py` — one real Claude call (~$0.0001) that validates wiring + budget enforcement.
2. Follow `scripts/run_phase8.md` — pre-flight checklist, dataset prep, launch command, checkpoint expectations, spend monitoring.
3. Use the launch / pause / resume / abort commands from
   [Running a project end-to-end](#running-a-project-end-to-end) above.
4. The project lives under `projects/pneumonia-data-efficiency/`
   (kickoff payload, per-project config, sandbox). A second comparable
   attempt with the post-fix runtime is set up at
   `projects/pneumonia-data-efficiency-v2/`.
