# Phase 8 runbook -- pneumonia-data-efficiency end-to-end

KICKOFF.md Section 8: "Run the pipeline against the v1 test project.
Operator approves at all five checkpoints. Diff outputs against
expectations. Capture failures as new tests. This is when the pipeline
becomes real."

This runbook walks the operator through the launch sequence and the
checkpoint flow. Read it once before starting, then keep it open as a
checklist.

---

## 0. Pre-flight

Before launching anything that costs money:

```bash
cd ~/autoscientist
uv sync
uv run python scripts/dry_run_phase8.py            # ~$0.0001
uv run python scripts/dry_run_phase8.py --ollama   # validates Qwen too
```

The Claude pass should report `cost_usd ~ $0.00005` and a cache hit on
the second call. The Ollama pass takes ~25s on the first call (cold
load) and a few seconds thereafter. If either fails, do not proceed.

Then verify the prerequisites the dry-run does not check:

| Item                          | Check command                                  | If missing                                                |
|-------------------------------|------------------------------------------------|------------------------------------------------------------|
| `tectonic` on PATH            | `which tectonic && tectonic --version`         | Reinstall per `tools/latex.py` header                      |
| Kaggle creds                  | `ls -la ~/.kaggle/kaggle.json`                 | Generate at kaggle.com/settings; chmod 600                 |
| Ollama model loaded           | `curl -s localhost:11434/api/tags \| jq '.models[].name'` | `ollama pull qwen3.6:27b`                                  |
| GPU visible                   | `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` | Restart WSL or check driver                                |
| BIMCV access for PadChest     | `[ -n "$BIMCV_TOKEN" ] && echo set`            | Register at bimcv.cipf.es; export `BIMCV_TOKEN`; set `bimcv_url` in registry |
| Phase 1-7 smokes still green  | `for n in 1 2 3 3_5 4 5 6 7; do uv run python scripts/smoke_phase$n.py >/dev/null && echo "$n ok" || echo "$n FAIL"; done` | Diagnose before continuing                                |

If PadChest is not yet wired up, you can still launch -- the
methodology agent will declare external validation as required and the
verify harness will block at stage 4 until PadChest data is present.
Better to wait than to chew through API budget on a doomed run.

---

## 1. Pre-fetch the dataset

NIH ChestX-ray14 is ~45 GB and takes hours over residential bandwidth.
Run the fetch ahead of time so it is not on the critical path of a
multi-hour pipeline run:

```bash
uv run python -c "
from autoscientist.tools import datasets
from pathlib import Path
spec = datasets.DATASET_REGISTRY['nih_chestxray14']
dest = Path('projects/pneumonia-data-efficiency/sandbox/data/nih_chestxray14')
print('present:', datasets.is_present(spec, dest))
"
```

If `present` is False, kick off the Kaggle fetch (it is idempotent --
re-running will skip files already downloaded):

```bash
uv run python -c "
from autoscientist.tools import datasets
from pathlib import Path
spec = datasets.DATASET_REGISTRY['nih_chestxray14']
dest = Path('projects/pneumonia-data-efficiency/sandbox/data/nih_chestxray14')
datasets._fetch_kaggle(spec, dest)
"
```

For PadChest, set up the BIMCV bits first, then:

```bash
export BIMCV_TOKEN=<your token>
uv run python -c "
from autoscientist.tools import datasets
spec = datasets.DATASET_REGISTRY['padchest']
spec.bimcv_url = 'https://...your project URL...'
"
```

(Persisting the URL across runs requires editing `tools/datasets.py`
or wiring a per-project override.)

---

## 2. Launch the pipeline

Open two WSL terminals.

**Terminal A -- the runner.** Starts the pipeline at `lit_review`,
which hands off through the agent graph:

```
lit_review -> idea_gen -> idea_critic -> methodology -> code_gen ->
test_gen -> code_review -> results_validator -> paper_writer ->
peer_reviewer -> [accept ? repo_publisher : paper_writer (revise)] ->
... repo_publisher -> HANDOFF: DONE
```

Termination: peer_reviewer routes to `repo_publisher` when its
`recommendation` is `accept`, and back to `paper_writer` for any
revise/reject path. The accept branch is the only one that produces a
terminal `HANDOFF: DONE`, emitted by `repo_publisher` after it writes
the curated release tree to `projects/<id>/release/`.

```bash
cd ~/autoscientist
PAYLOAD=$(cat projects/pneumonia-data-efficiency/kickoff_payload.json)
uv run python -m autoscientist.runtime.runner \
    --agent lit_review \
    --project pneumonia-data-efficiency \
    --payload "$PAYLOAD"
# Note the printed run_id -- you'll need it for resume.
```

The runner will pause at each of the five HITL checkpoints and emit a
log line like `checkpoint.opened cp_id=cp_... stage=N`.

**Terminal B -- the checkpoint UI.** Operator-facing approve / reject /
modify / ask-questions pages:

```bash
cd ~/autoscientist
uv run streamlit run src/autoscientist/checkpoints/ui.py
```

---

## 3. Checkpoint expectations

| # | Stage                        | Watch for                                                                                       | Common reasons to reject       |
|---|------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------|
| 1 | Idea selection               | 5 candidates with literature gap, novelty, feasibility, expected experiments, failure modes     | Off-topic ideas, hallucinated citations, no candidate matches the data-efficiency framing |
| 2 | Methodology approval         | Patient-level split, training_sizes sweep, 3 seeds, ResNet-50/ImageNet, PadChest external       | Image-level split, missing seeds, missing external validation, hyperparameter tuning on test |
| 3 | Preliminary review           | `code_review` verdict (with findings table); fires on **forward** advance to `results_validator` OR on a forced loop-cap pause when `code_review` has fired `runtime.max_code_review_cycles` times (default 3) | Blocker finding still open; methodology violations; loop-cap exceeded with no path forward — `modify` to instruct code_gen explicitly, or `reject` |
| 4 | Full results validation      | Verify harness must report `outcome=clean` (or you will be asked to interpret needs_human items) | Counterintuitive sign without explanation; baseline outside tolerance; weak labels not disclosed |
| 5 | Draft review                 | LaTeX builds via tectonic; every citation round-trips through `tools/citation_check.py`         | `[CITATION NEEDED]` placeholders left in; tectonic build failure   |

The Phase 7 pitfalls that will fire if the run drifts:

- `multi_seed_reporting` (fail) -- methodology must declare >= 3 seeds and results must report variance.
- `hyperparameter_tuning_split` (fail) -- must be `validation`, not `test`.
- `weak_label_provenance_disclosed` (needs_human) -- NIH and PadChest are NLP-derived; the limitation must be acknowledged.
- `view_projection_documented` (warn) -- chest X-ray cohort needs PA/AP handling.
- `confidence_intervals_reported` (needs_human) -- comparisons need bootstrap CIs or equivalent.

---

## 4. Pause / resume / abort

**Pause.** The runner pauses automatically at each checkpoint. Logs are
flushed; you can shut down terminal A safely while the checkpoint sits
in the UI.

**Resume.** From the run_id printed at launch:

```bash
uv run python -m autoscientist.runtime.runner --resume <run_id>
```

This picks up from the most recently resolved checkpoint.

**Abort.** Reject the pending checkpoint in the Streamlit UI. The
runner will mark the run `cancelled` and exit. To resume from earlier
state, pick a prior approved checkpoint as the resume point (currently
requires manual SQLite edit -- see `state/db.py` schema).

---

## 5. Spend monitoring

The runtime enforces the global $150/month cap with a $5 buffer
(KICKOFF.md Section 2). Per-project soft cap is documented in
`projects/pneumonia-data-efficiency/config.toml` for operator
visibility but is not enforced by the runtime.

Inspect spend at any time:

```bash
uv run python -c "
import sqlite3, os
conn = sqlite3.connect(os.environ.get('AUTOSCIENTIST_DB_PATH', 'autoscientist.db'))
conn.row_factory = sqlite3.Row
print('--- per-agent monthly spend ---')
for r in conn.execute('SELECT agent_name, ROUND(SUM(cost_usd), 4) AS spent, COUNT(*) AS n_calls FROM budget_ledger WHERE cache_hit=0 GROUP BY agent_name ORDER BY spent DESC'):
    print(f'  {r[\"agent_name\"]:20s}  \${r[\"spent\"]:.4f}  ({r[\"n_calls\"]} calls)')
print()
total = conn.execute('SELECT COALESCE(SUM(cost_usd), 0) AS s FROM budget_ledger WHERE cache_hit=0').fetchone()['s']
hits  = conn.execute('SELECT COUNT(*) AS n FROM budget_ledger WHERE cache_hit=1').fetchone()['n']
print(f'TOTAL real spend: \${total:.4f}')
print(f'cache hits: {hits}')
"
```

If you are within $20 of the cap, stop and reassess before approving
the next checkpoint.

---

## 6. After the run

A successful run produces:

- `runs/<run_id>/logs/*.jsonl` -- structured logs of every prompt /
  response / tool call / cost / handoff
- `projects/pneumonia-data-efficiency/sandbox/` -- generated training
  scripts, model checkpoints, results
- `projects/pneumonia-data-efficiency/release/` -- the curated,
  publishable repository (README, LICENSE, requirements.txt,
  curated `src/` and `scripts/`, `reproduce.sh`, `CITATION.cff`).
  Inspect this before sharing externally — a successful run does not
  mean the release is ready to publish; it means an LLM thought it was.
- The final paper PDF (path printed by paper_writer's last output)

Capture any verification failures or pitfall surprises that emerged as
new entries for `config/domains/medical_imaging.toml` -- this is the
ongoing Phase 7 work that KICKOFF.md describes as "ongoing."

If you hit a real problem the smokes did not catch, write a regression
test for it in `tests/unit/` before fixing, so the next run cannot
silently regress on the same issue.
