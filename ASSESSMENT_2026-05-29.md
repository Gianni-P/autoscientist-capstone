# autoscientist — Setup Assessment & Recommendations

**Date:** 2026-05-29
**Reviewer:** Claude Code (max-effort review)
**Scope:** Full repository read — `KICKOFF.md`, `README.md`, `pyproject.toml`, the
complete `src/autoscientist/` tree (runtime, clients, agents, tools, verify, meta,
checkpoints, state), `config/`, both `projects/` configs and status docs, the
55 MB production `autoscientist.db`, and the generated paper draft.
**Method:** Static read of code + config, plus live queries against the production
database to ground "what actually happened" in data rather than docs.

> This is a **new, non-destructive** document. It does not modify `README.md`,
> `KICKOFF.md`, or the project `STATE_OF_PLAY.md` / `SESSION_SUMMARY.md`.

---

## 1. Executive summary

**The substrate is genuinely well-built; the central thesis is not yet proven.**

You set out (KICKOFF §1) to build a multi-agent pipeline that autonomously takes a
research direction to a paper + supplementary + reproducible repo, with five
mandatory human checkpoints. After ~3 weeks and **$36.88** of real API spend, you have:

- ✅ **A clean, legible, well-documented runtime substrate** covering all of Phases 1–7.
  Budget enforcement, SHA-256 caching, structured logging, SQLite state, the agent/tool
  loop, pause/resume, and the Streamlit console are all implemented and smoke-tested.
- ✅ **An honest scientific artifact** — a paper draft reporting a *negative/near-null*
  result (NIH→PadChest transfer collapses to near-chance AUROC). That intellectual
  honesty is exactly what KICKOFF asked for.
- ⚠️ **A pipeline that has never autonomously traversed past Checkpoint 2.** The
  database shows checkpoints only ever reached stages 1 (idea) and 2 (methodology).
  Stages 3, 4, 5 have **zero rows, ever.** `paper_writer`, `results_validator`,
  `peer_reviewer`, and `repo_publisher` have **zero messages** — they have never run on
  a real project. **The paper and the experiments were produced by hand**, using the
  pipeline's scaffolding but not its autonomous chain.

So the honest framing of current standing is: **you have built an excellent research
*workbench*, and used it (semi-manually) to produce one real negative-result study. You
have not yet demonstrated the autonomous end-to-end loop that is the project's reason
for existing.** That's a perfectly respectable place to be — but the README and status
docs read as if the pipeline is further along than the data supports, and there are
three risks that should be fixed before the next run.

### The three things to fix first

| # | Risk | Severity |
|---|---|---|
| **R1** | **Nothing is in version control.** Zero git commits; the entire project (including all hard-won config tuning and the only paper draft) lives in an untracked working tree. | 🔴 Critical |
| **R2** | **`test_gen` cost runaway.** One agent is **62% of all spend** ($22.82 / $36.88) via a high-volume small-call loop. Known since the first Phase 8 session; still unbounded. | 🔴 Critical |
| **R3** | **End-to-end never reached CP3+.** The autonomous chain stalls in the code↔test↔review loop; the verification harness (Phase 5, the "most underrated component") and the writing/publishing agents have never run on real results. | 🟠 High |

---

## 2. Current standing — grounded in the database

### 2.1 What's built (phase-by-phase)

| Phase | Component | Code | Smoke test | **Exercised on a real run?** |
|---|---|---|---|---|
| 1 | Runtime + clients + cache + budget | ✅ | ✅ | ✅ |
| 2 | 10 core agents + prompts | ✅ (11 agents — added `repo_publisher`) | ✅ | Partial — only agents 1–7 |
| 3 | Tools (lit, pdf, exec, datasets, latex, citation) | ✅ | ✅ | Partial (lit/exec yes; latex/citation no) |
| 4 | Streamlit checkpoint UI + pause/resume | ✅ | ✅ | ✅ (CP1/CP2 only) |
| 5 | Verify harness (leakage, baseline, stats, pitfalls) | ✅ | ✅ | ❌ **Never on real results** |
| 6 | Meta / autoresearch (anchors, rubrics, A/B, versioning) | ✅ | ✅ | ❌ `prompt_versions=0`, `eval_runs=0` |
| 7 | Domain hardening (`medical_imaging.toml`, 13 checks) | ✅ | ✅ | ❌ (depends on Phase 5 path) |
| 8 | End-to-end on `pneumonia-data-efficiency` | — | dry-run only | ❌ **Stalls at CP2** |

**Implementation quality is high.** The runner (`runtime/runner.py`) has clearly been
hardened against the failure modes documented in `SESSION_SUMMARY.md`:

- File-persistence safety net (`persist_files_from_payload`) for the "agent emits
  `files:[…]` but never writes them" bug → **fixed**.
- `code_review` revision-loop cap forcing a CP3 gate → **fixed**.
- Per-project soft-cap enforcement in `router.py` → **fixed** (though see §4.2 — a
  config comment still claims it isn't).
- `DEFAULT_MAX_TOOL_ROUNDS` 8 → 40 → **fixed**.

The code reads exactly as KICKOFF §13 asked: legible over clever, well-commented,
defensive. `budget.py`, `db.py`, `router.py`, and `cache.py` are clean and correct.

### 2.2 What actually ran (hard numbers)

From `autoscientist.db` (schema v4) as of this review:

**Spend — $36.88 lifetime, all in 2026-05, well under the $150 cap and $50 start balance:**

| Agent | Real spend | Calls | Share |
|---|---:|---:|---:|
| **`test_gen`** | **$22.82** | 171 | **62%** |
| `code_review` | $7.70 | 57 | 21% |
| `code_gen` | $2.75 | 124 | 7% |
| `idea_critic` | $1.39 | 11 | 4% |
| `methodology` | $1.08 | 12 | 3% |
| `idea_gen` | $0.70 | 9 | 2% |
| `lit_review` | $0.44 | 12 | 1% |

> The code loop (`test_gen`+`code_review`+`code_gen`) = **90% of all spend.** The actual
> "science" agents (idea/methodology/lit) cost **$3.61 combined.** Caching saved real
> money: **125 hits / 523 calls (~24% hit rate).**

**Checkpoints — only stages 1 and 2 ever existed:**

| Stage | Approved | Modified | Rejected | Pending |
|---|---:|---:|---:|---:|
| 1 — idea selection | 8 | — | — | — |
| 2 — methodology | 4 | 2 | 1 | **1** |
| 3 — preliminary | — | — | — | — |
| 4 — full results | — | — | — | — |
| 5 — draft review | — | — | — | — |

**Late-stage agents never ran:** `paper_writer = 0`, `results_validator = 0`,
`peer_reviewer = 0`, `repo_publisher = 0` messages.

**Runs:** 11 real runs (v1 ×5, v2 ×5, `cli_test` ×1) + 8 smoke DBs. 8 completed,
2 cancelled, 1 paused.

### 2.3 Where you are parked *right now*

The most recent run — **`run_3d5f5aa9c09c…` (project `pneumonia-data-efficiency-v2`,
started 2026-05-20)** — is **`paused` at CP2 (methodology approval)**, checkpoint
`cp_79257b4d…`, still `pending`. Every v2 run to date has stopped at stage 2. The v2
project has cycled methodology approval repeatedly but **never advanced into a
results-producing stage via the chain.**

To clear it: open the Streamlit console and resolve that checkpoint, or use the CLI
resume path in `README.md`. (No action taken here.)

### 2.4 The scientific outcome — and a doc that now misleads

The paper draft (`projects/pneumonia-data-efficiency/paper/draft.tex`) is good and
**honest**: it reports a pre-registered permutation test that **fails to reject the null**
($F=0.56$, $p=0.217$), all AUROCs clustering at **0.43–0.49 (near chance)**, and a
counterintuitive **direction reversal at $N=25\text{k}$** — and it openly attributes the
whole signal to NIH→PadChest generalization failure.

**This contradicts `STATE_OF_PLAY.md`** (2026-05-17), which claims E1 showed "robust,
significant learning" and a "massive data efficiency advantage" (matched NIH AUROCs
0.57–0.68). The optimistic in-domain narrative did **not** survive once external
PadChest validation was included. **`STATE_OF_PLAY.md` is now stale and overstates the
result** — anyone resuming from it will be misled. (See R-doc in §4.)

Two structural caveats about the paper itself:

- It describes a **5-agent framework** (Hypothesis/Data/Training/Evaluation/Interpretation)
  that **does not match the real 11-agent pipeline.** It's an idealized narrative, not
  the system that produced it — fine for a methods paper, but don't mistake it for an
  accurate system description.
- It was **written out-of-band** (0 `paper_writer` messages, 0 stage-5 checkpoints), so
  the "Interpretation Agent flagged the anomalies" claim was actually done by hand. The
  deterministic `verify/` harness — the thing meant to catch exactly the near-chance /
  sign-reversal anomalies — **was never run by the chain on these results.**

---

## 3. What's genuinely strong (keep doing this)

1. **Budget discipline is real and correct.** `budget.py` enforces `cap − buffer`
   (=$145) with an estimator-based pre-check and actual-usage accounting after. The
   non-negotiable from KICKOFF §2 is honored in code, and spend never approached the cap.
2. **Caching works and is principled.** Canonical SHA-256 over (provider, model, system,
   messages, temp, max_tokens, tools) — a 24% hit rate on a real workload is meaningful
   money saved, and the design is exactly what KICKOFF §4 #3 demanded.
3. **The runner is battle-tested.** It absorbed a dozen real failure modes (token caps,
   off-topology handoffs, file-persist gaps, runaway revision loops, manual pause/resume)
   and the fixes are documented inline. This is the hardest part of such a system and
   it's solid.
4. **Observability from day one.** Structured JSONL per run + a queryable SQLite state
   store meant this very review could be grounded in data. That paid off exactly as
   KICKOFF §4 #4 predicted.
5. **Intellectual honesty.** The pipeline (with operator help) produced a *negative*
   result and said so plainly. Most "AI scientist" demos manufacture a positive finding;
   this one didn't. That's the single most credible thing in the repo.
6. **The domain-authority disclaimer is handled well** — both README and KICKOFF state
   plainly that there's no real clinical knowledge baked in.

---

## 4. Risks & recommendations

### 🔴 R1 — Put the project under version control *today*

**Finding:** `git log` → "your current branch 'main' does not have any commits yet."
Everything is untracked. The `.gitignore` is well-designed and ready, but **not a single
commit exists.** All of your config tuning (the `max_tokens` bumps, the Qwen→Sonnet
reroute, the loop-cap), the only paper draft, and the runtime hardening exist only in a
working tree one bad `git clean`/`rm` away from gone. This also makes a mockery of
KICKOFF §6/§9's "version everything in git; never overwrite a prompt without saving the
previous version" — and explains why `prompt_versions` is empty.

**Recommendation (do this first, ~10 min):**
```bash
cd /mnt/d/autoscientist
git add -A
git status            # confirm .venv/, *.db, runs/, data/ are correctly ignored
git commit -m "Snapshot: substrate (Phases 1-7) + v1 negative-result study"
```
Then commit per working session. Consider a second commit tagging the current paused
state. **Do not** rely on `.gitignore` excluding `*.db` to mean the DB is safe — it's
not backed up at all; add a periodic copy of `autoscientist.db` to a backup location
(it's your entire experimental record: runs, spend ledger, cache, checkpoints).

> Per your standing preference, this is additive — committing does not delete or
> overwrite any operator artifact.

---

### 🔴 R2 — Bound `test_gen` (and any agentic loop) by *per-invocation* cost, not just per-call

**Finding:** `test_gen` is **62% of lifetime spend** ($22.82) across 171 small calls.
The `cost_ceiling_usd = $2.00` is a **per-call** ceiling; it does nothing against a loop
of 95 calls each under the ceiling that sums to $16+. This was flagged as "Pipeline
issue #5" in the very first Phase 8 session and is still listed as "Blocker #3 — still
open" in `pneumonia-data-efficiency-v2/config.toml`. Root cause is partly that
`code_gen`/`test_gen` were rerouted off local Qwen onto Sonnet (SESSION_SUMMARY #3),
so every iteration now costs money.

**Recommendations (pick at least the first two):**
1. **Add a per-agent-invocation budget cap** in `_invoke_agent` (`runtime/runner.py`):
   track cumulative real cost across the tool-loop rounds for a single invocation and
   break with a logged warning when it exceeds, e.g., `invocation_ceiling_usd`. This is
   the missing layer between per-call ceiling and per-month cap.
2. **Lower `test_gen`'s `max_tool_rounds`** (currently 8 in `models.toml`) and tighten
   the prompt's "stop iterating" instruction the way `code_gen` was tightened.
3. **Revisit the local-model leg.** The RTX 5090 is idle for inference. Either (a) invest
   once in getting a code-capable local model to close the agentic loop (the Qwen attempt
   failed per SESSION_SUMMARY #3 — try a current coding-tuned model and verify it can
   emit tool calls through the OpenAI-compat shim), or (b) formally accept that
   `code_gen`/`test_gen` run on Claude and bound them with #1/#2. Right now you're paying
   Claude prices for the highest-volume agents *and* leaving the GPU unused — the worst
   of both.

---

### 🟠 R3 — The autonomous loop has never crossed CP3; prove it on a *cheap* path before the next expensive run

**Finding:** Stages 3–5 never opened; `results_validator`/`paper_writer`/`peer_reviewer`/
`repo_publisher` never ran. The chain reliably stalls in code↔test↔review and the run
gets stopped (budget watchdog or operator). The science that exists was done by hand.

**Recommendation — build a "thin vertical slice" test that exercises CP3→CP4→CP5 without
the expensive code loop:**
- Feed `results_validator` a **canned small-results payload** (you already have real
  E1/E2 numbers) so you can drive CP4 → `paper_writer` → CP5 → `peer_reviewer` →
  `repo_publisher` and watch the back half of the pipeline work end-to-end for a few
  dollars. This de-risks the part that has *never* executed before you spend another $20+
  getting through the code loop.
- Only after the back half is proven, attempt a full v2 run with R2's caps in place.

This also surfaces whatever integration bugs are lurking in the four never-run agents —
right now their only evidence of working is unit/smoke tests with mocked I/O.

---

### 🟠 R4 — Run the verification harness on the *real* results (it's the whole point)

**Finding:** Phase 5 — described in KICKOFF §9 as "the most underrated component" — has
unit tests but **was never applied to the real run's outputs** (CP4 never opened). The
near-chance AUROCs and the $N=25\text{k}$ sign reversal — textbook triggers for
`verify/pitfalls.py` (`counterintuitive_signs_flagged`, `confidence_intervals_reported`)
and `baseline_repro` — were caught by a human, not the system.

**Recommendation:** As part of R3's slice, point `verify/leakage.py`, `verify/stats.py`,
`verify/baseline_repro.py`, and `verify/pitfalls.py` (with `medical_imaging.toml`) at the
real E1/E2 result JSON and the methodology plan. This is both a validation of Phase 5 and
a genuine second check on the paper's claims. If the harness *wouldn't* have flagged the
near-chance regime, that's a gap in the pitfall library worth closing (e.g., an
"AUROC ≈ 0.5 → results not interpretable" guard).

---

### 🟡 R-doc — Reconcile the status docs with reality

- **`STATE_OF_PLAY.md` overstates the result** (§2.4). Add a header note that the final
  paper supersedes it with a negative result, or update its "Key Scientific Discoveries"
  section. As written it will mislead the next session/agent — and it explicitly invites
  you to "point any new agent at this file to resume."
- **`pneumonia-data-efficiency-v2/config.toml` is stale**: its `[budget]` comment says
  "the runtime currently enforces only the monthly cap," but `router.py:207–216` *does*
  enforce the per-project soft cap via `assert_project_budget`. Update the comment so the
  soft cap isn't assumed to be inert.
- **`README.md`** presents Phase 8 as a launch-ready procedure; add a one-line current
  status ("autonomous chain proven through CP2; CP3–CP5 not yet exercised end-to-end")
  so expectations match the DB.

---

### 🟡 R5 — Paper deliverables are incomplete (and won't compile as-is)

The KICKOFF success criterion is paper **+ supplementary + reproducible repo**. Current state:
- `draft.tex` exists but **has no `.bib`** — it `\cite{}`s `rajpurkar2017chexnet`,
  `zech2018variable`, etc. with no bibliography file anywhere in the tree, so **it will
  not compile**, and **`citation_check.py` (KICKOFF §4 #8, "citations must be verified")
  appears never to have been run on it.** Hallucinated-citation risk is exactly what that
  rule exists to prevent.
- **No compiled PDF** (tectonic never invoked on it), **no supplementary document**, and
  **no `release/` tree** (`repo_publisher` never ran).
- The first line of `draft.tex` is a stray markdown fence (` ```latex `) — an LLM-output
  artifact that will break compilation.
- There's a **stray `projects/paper/draft.tex`** at the wrong path (should live under
  `projects/<project_id>/paper/`). Likely an accidental write; reconcile/remove the
  duplicate.

**Recommendation:** Treat "compile the paper with a verified `.bib` via `citation_check`
+ `latex.py`, generate the supplementary, and have `repo_publisher` emit a `release/`
tree" as the concrete definition of finishing the v1 study. It's close, and it's the
deliverable that makes the whole exercise legible to an outside reader.

---

### 🟡 R6 — Repo & tooling hygiene

- **`scripts/` is polluted with ~30 ad-hoc operator scripts** (`_dbg_*`, `_check_*`,
  `_approve_*`, `_resolve_*`, `_reject_*`) interleaved with the real `smoke_phaseN.py`
  and `run_phase8.md`. Move these under `scripts/scratch/` (or delete the dead ones after
  R1's commit captures them in history) so the entry points are discoverable.
- **`pyproject.toml`:** `pytest` is listed in **both** `dependencies` (`>=9.0.3`) and the
  `dev` group (`>=8`) — inconsistent and it shouldn't be a runtime dep. Remove it from
  `[project.dependencies]`.
- **`filterwarnings = ["error"]`** with torch / pandas 3 / numpy 2.4 will turn any
  third-party `DeprecationWarning` into a test failure — brittle. Scope it to the
  project's own modules (e.g. `error::DeprecationWarning:autoscientist`) or the suite
  will fail for reasons unrelated to your code.
- I **could not run the test suite from this Windows host** — the `.venv` is a Linux/WSL
  venv (`bin/`, `lib/python3.12`) and the only host Python is 3.13 without the deps.
  That's expected, but it means **there is no host-portable way to run tests and no CI.**
  Consider a tiny GitHub Actions (or local `uv run pytest`) gate so the 18 unit + 8 smoke
  tests actually guard regressions.

---

### 🟡 R7 — Latent correctness/maintenance traps (low urgency, worth a comment now)

- **Schema migrations are additive-only.** `db.py::_ensure_schema` runs
  `CREATE TABLE IF NOT EXISTS` then bumps `schema_version`. There is **no `ALTER TABLE`
  path** — the first time you add a *column* to an existing table, old DBs silently won't
  get it and the version bump will mask the divergence. Add a real migration step (or at
  least a comment + assertion) before that day comes.
- **`meta/` has both `eval_rubrics.py` and `rubrics.py`** (plus `meta_prompter.py`,
  `anchors.py`, `versioning.py`, `ab_harness.py`). KICKOFF §5 listed only `eval_rubrics`.
  Confirm these aren't two implementations of the same thing before the meta-layer is
  switched on for real.
- **`messages.cost_usd` is mostly NULL** — cost lives in `budget_ledger` (correct), but
  if any UI/report reads per-message cost it'll see nulls. Just be aware the ledger is the
  source of truth.

---

## 5. Recommended next actions (ordered)

**Track A — make the work safe (do this session):**
1. `git add -A && git commit` (R1). Back up `autoscientist.db`.
2. Reconcile `STATE_OF_PLAY.md` and the v2 config comment with reality (R-doc).
3. Resolve or formally park the pending CP2 on `run_3d5f5aa9…` so the v2 run isn't left
   dangling.

**Track B — prove the thesis cheaply (next session):**
4. Add the per-invocation cost cap + lower `test_gen` rounds (R2 #1–2).
5. Build the thin CP3→CP4→CP5 slice with canned results (R3) and run `verify/` on the
   real E1/E2 numbers (R4).
6. Finish the v1 deliverable: verified `.bib` + `citation_check` + compiled PDF +
   supplementary + `repo_publisher` release tree (R5).

**Track C — only after A & B:**
7. Decide the local-model question (R2 #3) before another full v2 run.
8. Turn on the dormant Phase 6 meta-layer once prompts are git-versioned and stable
   (it's pointless to A/B-optimize prompts that aren't yet under version control).

---

## 6. Appendix — evidence

**Database:** `autoscientist.db`, schema v4, 55 MB, last modified 2026-05-20.
**Lifetime spend:** $36.8767 over 398 paid calls; 125 cache hits (523 total calls).
**Runs:** 11 (`completed`×8, `cancelled`×2, `paused`×1) across `pneumonia-data-efficiency`,
`pneumonia-data-efficiency-v2`, `cli_test`.
**Paused run:** `run_3d5f5aa9c09c4849868548ae0f8d3139` → CP2 `cp_79257b4d9d644942bcfb20a7a34644cd` (pending).
**Checkpoints created, by stage:** {1: 8, 2: 8, 3: 0, 4: 0, 5: 0}.
**Agents with zero real messages:** `paper_writer`, `results_validator`, `peer_reviewer`, `repo_publisher`.
**Phase 6 tables:** `prompt_versions = 0`, `eval_runs = 0`.

**Source tree (non-`.venv`):** 11 agents, 12 tools, 5 verify modules, 6 meta modules,
runtime (agent/runner/handoff/budget/control/config/payload_files), clients
(claude/ollama/router/cache/base/mock), checkpoints (manager/ui), state (db). 18 unit
test files + 8 phase smoke tests + 1 dry-run.

*Generated by Claude Code on 2026-05-29. Non-destructive: no existing files were modified.*
