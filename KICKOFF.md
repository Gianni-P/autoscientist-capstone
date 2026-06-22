# autoscientist: Project Kickoff

This document is the starting point for building **autoscientist**, a multi-agent research pipeline that produces academic papers, supplementary documents, and reproducible code repositories from high-level research directions in scoped domains.

> **Status note (2026-06-20) — read me first.** This is the original Phase-0
> kickoff brief, preserved as the project's spec and north star. The system was
> since built through all phases, and a few things evolved past what this brief
> assumes:
> - It is now **domain-general** — domain facts come from each project's own
>   `config.toml`, not baked into prompts. The medical-imaging examples below
>   describe the *original* target, not a hard constraint.
> - The original `pneumonia-data-efficiency` example project (§8) has been
>   **retired and removed**. The current **featured end-to-end deliverable is
>   `math693a-limited-descent`** (a numerical-optimization study) — see
>   `README.md` and `projects/math693a-limited-descent/`.
> - **Model routing is now operator-selectable per leg** at each checkpoint (the
>   web console's model picker, incl. an Opus-orchestrator → local-worker mode);
>   current defaults live in `config/models.toml`. The monthly budget cap is
>   currently **$200**.
> - The primary operator console is now a push-based **web app** (Starlette/SSE);
>   the Streamlit UI in §3 remains as a fallback.
> - **The agent roster grew past the ten listed in §5/§6.** The built pipeline
>   now has twelve specialist agents: a new **`figure_gen`** (between
>   `results_validator` and `paper_writer`, renders the paper's figures from the
>   validated results) and **`repo_publisher`** were added alongside the original
>   ten. The §5/§6 lists below are the original spec, not the current roster.
>
> The principles (§4), checkpoints (§7), and verification (§5) describe the
> system as built and remain accurate.

You (Claude Code) are picking this up at Phase 0. Read this entire document before writing any code. Implement strictly in phase order — earlier phases are dependencies for later ones. When you finish a phase, run the smoke tests for it before moving on.

---

## 1. Mission and realistic ceiling

The system takes a high-level direction (e.g., "novel methods for lung disease prediction in CT images") and produces:

1. A draft academic paper (LaTeX → PDF)
2. A supplementary materials document
3. A reproducible code/data repository

It does this with a multi-agent pipeline running on a hybrid Claude API + local Ollama setup, with **five mandatory human-in-the-loop checkpoints** between idea generation and final paper.

**Realistic v1 ceiling:** mid-tier journal quality (think *Cancer Epidemiology, Biomarkers & Prevention*, *Medical Image Analysis* workshop tracks, *JMIR Cancer*) when the operator provides good direction at checkpoints. **Nature-tier work is explicitly not a target.** The system will produce workmanlike, methodologically sound, reproducible research — the operator's job at checkpoints is to inject the conceptual ambition and domain taste that elevates output to publishable.

If you ever feel pressure to claim more capability than this, push back in your status messages.

---

## 2. Operator environment

- **Host:** Windows with WSL2 / Ubuntu; project on the WSL ext4 filesystem
- **Working directory:** `~/autoscientist` (i.e. `/home/gdp/autoscientist` on ext4 — migrated off the old `/mnt/d/autoscientist` Windows-drive path, which must not be recreated)
- **GPU:** NVIDIA RTX 5090, 32 GB VRAM (verify with `nvidia-smi`)
- **Local LLM serving:** Ollama, accessible at `http://localhost:11434`
  - Operator referenced "Qwen3.6 27B" — this name is not a known official Qwen release. Run `ollama list` and use whatever 27B-class Qwen model is actually installed. Document the actual name in the config.
  - Ollama exposes an OpenAI-compatible endpoint at `http://localhost:11434/v1` — use that, not Open WebUI.
- **Anthropic API:** key in env var `ANTHROPIC_API_KEY`. Verify it's set before any API call.
- **Budget:** $50 starting balance, hard cap $150/month (later raised to **$200** — see the status note at the top). The system **must** track spend and refuse new API calls when within $5 of the monthly cap. This is non-negotiable.
- **Python:** 3.12. Use `uv` for package management (install with `curl -LsSf https://astral.sh/uv/install.sh | sh` if absent).

---

## 3. Prescribed tech stack

Do not deviate without flagging in a status update first.

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.12 | Standard for the ML ecosystem |
| Package manager | `uv` | Fast, modern, lockfile-based |
| Claude client | `anthropic` SDK | Official |
| Local LLM client | `openai` SDK pointed at Ollama's `/v1` | Avoids reinventing |
| UI | Streamlit | Ship the human-checkpoint UI in a day |
| State | SQLite | No separate DB to manage; resilient |
| Sandboxed execution | subprocess + resource limits + restricted CWD | v1 simplicity; Docker is post-v1 |
| Observability | Structured JSONL logs + Streamlit log viewer | No external dependencies |
| Literature APIs | Semantic Scholar, OpenAlex, arxiv | All free, no keys needed |
| PDF parsing | `pypdf` (simple), `marker-pdf` (complex layouts) | Tiered for cost |
| Embeddings cache | SQLite + `sentence-transformers` | No vector DB for v1 |
| LaTeX | `tectonic` (single-binary, no TeXLive setup) | Reproducible builds |

---

## 4. Architectural principles (non-negotiable)

These are the principles that make this system work versus making yet another "AI scientist" demo. Internalize them before writing code.

1. **Tier model usage by leverage, not by phase.** Do not hardcode "Claude does framework, Qwen does code." Make it configurable per-agent. Default routing is in §6.
2. **Verification > LLM review.** Every check that can be made deterministic (assertions, baseline-reproduction tolerances, leakage detectors, statistical assumption checks) must be deterministic. LLM "review" steps are last-resort, not primary defense.
3. **Cache aggressively.** Same paper retrieval, same prompt, same dataset — never re-run if the inputs are unchanged. Cache hits are free; cache misses cost real money. Implement caching in Phase 1 alongside the agent runtime, not later.
4. **Structured logs from day one.** Every prompt, response, tool call, cost, latency, and handoff goes to JSONL. You will spend more time reading logs than writing code; build observability before features.
5. **Five mandatory human checkpoints.** No autonomous run-to-completion. Checkpoint stages (detailed in §7): idea selection, methodology approval, preliminary results, full results validation, draft review.
6. **Counterintuitive findings get flagged automatically.** Any coefficient sign, effect direction, or metric outlier that contradicts dominant literature must trigger a hard checkpoint regardless of pipeline state. This is what would have caught the negative-PM₂.₅ ecological-confounding issue if a careless pipeline had produced that paper.
7. **Reproduce baselines before claiming novelty.** No "novel result" claim is allowed in any output until the pipeline has reproduced a published baseline within tolerance on the same dataset. Hard rule, enforced in code.
8. **Citations must be verified.** LLMs hallucinate references constantly. Every citation in any generated paper must round-trip through Semantic Scholar / OpenAlex / arxiv to confirm the paper exists and the metadata matches.

---

## 5. Repository structure

Create exactly this layout in Phase 1:

```
~/autoscientist/
├── pyproject.toml              # uv-managed
├── uv.lock
├── .env.example                # documents required env vars
├── .gitignore
├── README.md                   # operator-facing
├── KICKOFF.md                  # this document
├── config/
│   ├── default.toml            # default config
│   ├── models.toml             # model routing per agent
│   └── domains/                # domain-specific pitfall checklists
│       └── medical_imaging.toml
├── src/autoscientist/
│   ├── __init__.py
│   ├── runtime/                # agent runtime (Phase 1)
│   │   ├── agent.py
│   │   ├── runner.py
│   │   ├── handoff.py
│   │   └── budget.py           # spend tracking, hard cap enforcement
│   ├── clients/                # LLM and tool clients
│   │   ├── claude.py
│   │   ├── ollama.py
│   │   ├── router.py           # decides which model per agent call
│   │   └── cache.py
│   ├── agents/                 # one file per agent (Phase 2)
│   │   ├── lit_review.py
│   │   ├── idea_gen.py
│   │   ├── idea_critic.py
│   │   ├── methodology.py
│   │   ├── code_gen.py
│   │   ├── test_gen.py
│   │   ├── code_review.py
│   │   ├── results_validator.py
│   │   ├── paper_writer.py
│   │   └── peer_reviewer.py
│   ├── tools/                  # tool integrations (Phase 3)
│   │   ├── literature.py       # Semantic Scholar, OpenAlex, arxiv
│   │   ├── pdf_parse.py
│   │   ├── execute.py          # sandboxed subprocess runner
│   │   ├── datasets.py         # public dataset registry + fetchers
│   │   ├── latex.py
│   │   └── citation_check.py
│   ├── verify/                 # deterministic verification (Phase 5)
│   │   ├── leakage.py
│   │   ├── baseline_repro.py
│   │   ├── stats.py
│   │   └── pitfalls.py         # domain pitfall checks
│   ├── checkpoints/            # human-in-the-loop (Phase 4)
│   │   ├── manager.py
│   │   └── ui.py               # Streamlit pages
│   ├── meta/                   # autoresearch / prompt opt (Phase 6)
│   │   ├── eval_rubrics.py
│   │   ├── meta_prompter.py
│   │   └── ab_harness.py
│   └── state/
│       └── db.py               # SQLite schema + accessors
├── projects/                   # one subdir per research project run
│   └── .gitkeep
├── prompts/                    # all system prompts as .md files, version-controlled
├── tests/
│   ├── unit/
│   └── e2e/
└── scripts/
    ├── smoke_phase1.py
    ├── smoke_phase2.py
    └── ...
```

**Why this shape:** prompts as files (not embedded in code) so the Phase 6 meta-prompter can A/B test variations. Agents as separate files so adding/removing them is mechanical. Verification as a separate top-level module so it can be called independently of the agent loop.

---

## 6. Default model routing

In `config/models.toml`:

| Agent | Model | Why |
|---|---|---|
| `lit_review` | claude-haiku (latest) | High volume, low complexity per call |
| `idea_gen` | claude-sonnet (latest) | Quality matters; runs once per project |
| `idea_critic` | claude-sonnet | Adversarial reasoning |
| `methodology` | claude-sonnet | High-leverage decisions |
| `code_gen` | local Qwen 27B via Ollama | Volume; cost-sensitive |
| `test_gen` | local Qwen 27B | Same |
| `code_review` | claude-sonnet | Bug catches matter |
| `results_validator` | claude-sonnet | Catches counterintuitive findings |
| `paper_writer` | claude-sonnet | Writing quality matters |
| `peer_reviewer` | claude-sonnet | Different system prompt, simulates reviewer |

`claude-opus` is **not** in the default routing — use only when explicitly enabled per-agent for a hard problem. Operator can override per-project. Use whatever the latest `claude-sonnet` and `claude-haiku` aliases resolve to via the API.

The router (`clients/router.py`) reads this config and a per-agent `cost_ceiling` parameter. If a call would exceed the project's remaining budget, the router refuses and surfaces a checkpoint.

---

## 7. Human-in-the-loop checkpoints

All five are mandatory. The checkpoint manager (`checkpoints/manager.py`) writes a row to SQLite, halts the run, and the Streamlit UI surfaces the pending checkpoint to the operator. The run does not resume until the operator approves, modifies, or rejects.

| # | Stage | Payload to operator |
|---|---|---|
| 1 | Idea selection | 5 candidate directions, each with: cited literature gap, novelty assessment, feasibility, expected experiments, compute estimate, top reasons it could fail |
| 2 | Methodology approval | Detailed experimental plan, datasets, baselines, metrics, statistical analysis plan |
| 3 | Preliminary results | Output from a small-scale (subset) run, with sanity-check plots and baseline reproduction status |
| 4 | Full results validation | Final results, all flagged anomalies, all counterintuitive findings, all pitfall-check failures |
| 5 | Draft review | Full paper draft + supplementary, with diff view of what changed since last revision |

Approve / reject / modify (free-text instructions to the next agent) / ask-questions (operator can converse with the orchestrator).

---

## 8. The example v1 test project

> **Retired (2026-06-20).** This original v1 example project has been removed from
> the repository. The section is kept to document the initial target and design
> intent; the realized featured deliverable is `math693a-limited-descent` (see
> `README.md`). The pipeline is domain-general — the medical-imaging design below
> is illustrative of the *original* target, not the current featured project.

Build the pipeline against this concrete project. It's the regression test — every change to the pipeline re-runs against this and we diff against a known-good output.

**Project codename:** `pneumonia-data-efficiency`

**Research question:** *How does training data size affect cross-institutional generalization in CNN-based pneumonia detection from chest radiographs?*

**Datasets (all public):**
- Training pool: NIH ChestX-ray14 (~112k images, accessible via NIH Box or Kaggle mirror)
- External validation: PadChest (Spanish hospital, ~160k images, accessible via BIMCV)
- Optionally: CheXpert (Stanford) for second external validation

**Experimental design (the methodology agent should produce something close to this):**
- Fine-tune ResNet-50 (ImageNet pretrained) for binary pneumonia classification
- Train on NIH ChestX-ray14 subsets of size N ∈ {1k, 5k, 25k, 100k}
- For each N, evaluate on held-out NIH test split AND on PadChest pneumonia-labeled subset
- Plot generalization gap (in-domain AUROC − external AUROC) versus training size N
- Run 3 seeds per N, report mean ± SD
- Compare to a published reference point (Rajpurkar CheXNet or similar)

**Why this project for v1:**
- Tractable on a single 5090: ResNet-50 + chest X-rays + 100k samples max trains in <2 hours per run
- All datasets are genuinely public (NIH and BIMCV require free registration; document the steps)
- Real answerable question with a meaningful answer either way
- Plenty of literature to ground the lit-review agent (CheXNet, MIMIC-CXR work, domain shift literature)
- Clear baselines exist — pipeline can validate it isn't producing hallucinated numbers
- Output paper would target a workshop track or short letter, which is the right ceiling for v1 testing

**The first end-to-end success criterion:** the pipeline runs against this project end-to-end with operator approvals at each checkpoint, produces a draft paper, supplementary, and a code repo that re-runs from scratch and reproduces the headline numbers within seed variance.

---

## 9. Build phases — execution order

Implement in order. Each phase has a smoke test in `scripts/smoke_phaseN.py` that must pass before moving to the next phase.

### Phase 1 — Agent runtime + clients + caching + budget (target: 4–5 days)

Build the substrate. No domain logic yet.

- `pyproject.toml` with uv, all deps pinned
- `src/autoscientist/runtime/agent.py`: `Agent` dataclass (name, role, system_prompt_path, tools, model_key, handoff_targets)
- `src/autoscientist/runtime/runner.py`: main run loop with handoff, history management, error recovery
- `src/autoscientist/clients/claude.py`: Anthropic client with retry, structured logging, token counting
- `src/autoscientist/clients/ollama.py`: OpenAI-SDK client pointed at `localhost:11434/v1`
- `src/autoscientist/clients/router.py`: reads `config/models.toml`, routes by agent name
- `src/autoscientist/clients/cache.py`: SHA256-keyed cache of (system_prompt + messages + model + temp) → response, stored in SQLite
- `src/autoscientist/runtime/budget.py`: tracks cumulative cost, refuses calls when within $5 of monthly cap, logs every charge
- `src/autoscientist/state/db.py`: SQLite schema for `runs`, `messages`, `cache`, `budget_ledger`, `checkpoints`
- Two stub agents (echo agent + handoff agent) to test the runtime without spending money
- `scripts/smoke_phase1.py`: runs stub agents through 3 handoffs, asserts logs and state are correct, asserts cache hit on second run, asserts budget tracking works

**Done criterion:** smoke_phase1 passes; you can run `uv run python -m autoscientist.runtime.runner --agent echo` and see structured logs.

### Phase 2 — Core agents with prompts (target: 1 week)

Implement the 10 agents listed in §5, each as a separate file under `src/autoscientist/agents/`. Prompts live in `prompts/` as Markdown files with frontmatter (model, temperature, expected output schema).

For now, agents return free-text or simple JSON. Tools come in Phase 3.

`scripts/smoke_phase2.py`: runs idea_gen → idea_critic → methodology on a hardcoded prompt, asserts output structure.

### Phase 3 — Tool integrations (target: 1 week)

In order:
- Literature search (Semantic Scholar primary, OpenAlex fallback, arxiv for preprints)
- PDF parsing
- Sandboxed execution (this is the trickiest — see §10)
- Public dataset registry + fetchers (start with NIH ChestX-ray14 and PadChest for the v1 test project)
- LaTeX compilation (tectonic)
- Citation verification (round-trip every cited DOI/arxiv-id)

### Phase 4 — Streamlit checkpoint UI (target: 3–5 days)

Five pages in Streamlit, one per checkpoint stage. State backed by SQLite. Operator can approve / reject / modify / ask questions. Runs continue when operator approves.

### Phase 5 — Verification harness (target: 1 week, most underrated component)

In `src/autoscientist/verify/`:
- Leakage detector (train/test ID overlap, target leakage in features)
- Baseline reproduction harness (must match published baseline within configured tolerance)
- Statistical assumption checkers (multicollinearity, normality where assumed, sample size adequacy)
- Pitfall library — start with `domains/medical_imaging.toml` containing checks like:
  - "scanner/site stratification applied if multi-source"
  - "patient-level (not image-level) train/test split"
  - "no test-time augmentation in baseline comparison"
  - "external validation present if claiming generalization"
  - "counterintuitive coefficient signs flagged"

Pitfall checks return Pass / Fail / Needs-human. Fail blocks the pipeline; Needs-human triggers a checkpoint.

### Phase 6 — Autoresearch / prompt optimization (target: 2 weeks)

**Defer this until Phases 1–5 are stable.** Optimizing prompts for agents whose structure is still changing wastes cycles.

When ready:
- Per-agent eval rubric (e.g., for idea_gen: novelty + grounding + feasibility + counter-arg quality, scored by a separate Claude judge)
- Curate 10–30 anchor examples per agent (gold-standard outputs)
- Meta-prompter: Claude instance that reads eval traces and proposes prompt variations
- A/B harness: run variants on anchor examples, score, keep winners
- Version everything in git; never overwrite a prompt without saving the previous version

### Phase 7 — Domain hardening (start in parallel with Phase 5, ongoing)

Build out `config/domains/medical_imaging.toml` against the v1 test project. As pitfalls emerge during the v1 run, codify them.

### Phase 8 — End-to-end run on `pneumonia-data-efficiency` (target: 1 week)

Run the pipeline against the v1 test project. Operator approves at all five checkpoints. Diff outputs against expectations. Capture failures as new tests. This is when the pipeline becomes real.

---

## 10. Specific design decisions and gotchas

**Sandboxed execution (Phase 3):** subprocess approach for v1:
- All generated code runs in `projects/<project_id>/sandbox/` as CWD
- Subprocess limits via `resource` module: CPU time cap, memory cap, no network for the executor by default (whitelist for dataset fetches done outside the sandbox)
- Output captured to `stdout.log`, `stderr.log`; exit code logged
- 30-minute hard timeout per cell; kill the process group on timeout
- After Phase 8, revisit and migrate to Docker if needed for stronger isolation

**Why not Docker now:** an extra week of yak-shaving for marginal v1 benefit on a single-operator solo project. The risk is the agent overwriting files in the project directory, which the CWD restriction handles. Filesystem-wide damage is prevented by running the whole pipeline in WSL, which is itself isolated from the Windows host.

**Cost discipline (Phase 1, but applies always):**
- Every API call goes through `clients/router.py`, which checks `budget.py` first
- Cache check happens before the budget check (cache hits don't cost anything)
- Daily budget snapshot logged; weekly summary surfaced in the Streamlit UI
- Refuse new calls at $145/month spent (leaving $5 buffer for verification round-trips)

**Citation hallucination is not theoretical.** Every paper title, author list, year, and DOI emitted by paper_writer or lit_review must be verified through the citation_check tool before it survives into the final paper. If a citation cannot be verified, replace it with a `[CITATION NEEDED]` placeholder and surface to the checkpoint UI. Hallucinated citations are the single fastest way for AI-generated work to get caught and rejected.

**Counterintuitive findings.** Implement in `verify/pitfalls.py`. If the methodology agent stated a hypothesized direction (e.g., "expect higher PM2.5 to associate with higher cancer incidence") and the result agent produces the opposite sign, this is a hard halt with a checkpoint, regardless of statistical significance. The checkpoint payload should include candidate explanations (confounding, ecological fallacy, sample issues) and require operator interpretation before the writing agent proceeds.

**Domain expertise gap.** This pipeline has no real medical/clinical knowledge baked in. The pitfall checklist substitutes for expertise on common mistakes, but it does not substitute for clinical judgment. Operators must be told this in onboarding, and the README should say so plainly. Don't let the system claim more domain authority than it has.

---

## 11. What I (Claude Code) should flag back to the operator

You'll hit decisions where my (Anthropic-Claude-the-author-of-this-doc) guidance is ambiguous or where you genuinely need operator input. When that happens, use this format in a status message:

```
DECISION NEEDED: <short title>
Context: <2–3 sentences>
Options:
  A) <option>
  B) <option>
Default if no response: <which one and why>
```

Don't block. Pick the default and proceed; the operator can correct you.

Likely decisions you'll hit:
- Exact Qwen model name once `ollama list` shows what's actually installed
- Whether to use Kaggle vs. direct NIH download for ChestX-ray14 (depends on operator credentials)
- Specific Sonnet/Haiku version aliases at the time you build (use the latest stable)
- Specific Streamlit page layouts for checkpoints (you have license to design these reasonably)

---

## 12. First-week deliverables

By end of week 1, the operator should be able to:

1. `cd ~/autoscientist && uv sync && uv run streamlit run src/autoscientist/checkpoints/ui.py` — launches a stub UI
2. `uv run python scripts/smoke_phase1.py` — passes, exercising stub agents through handoffs with caching and budget tracking
3. See structured JSONL logs in `runs/<run_id>/logs/` after the smoke test
4. Verify cache hits on the second run of the smoke test (zero spend)
5. Verify the budget circuit breaker by setting the monthly cap to $0.01 and watching the runtime refuse a call

That's the Phase 1 done-criterion. Don't move to Phase 2 until all five hold.

---

## 13. Final notes

- This is a research tool, not a product. Code quality should be "good enough for a careful operator to debug," not "ready for external users." That bar gets raised later if the project goes anywhere.
- Optimize for legibility over cleverness. The operator and I (Claude, in future sessions) need to read this code to fix it.
- When in doubt, do the simpler thing and write a comment about why you didn't do the more complex thing. Future-you will thank present-you.
- If the budget runs out during your first week of work, halt and surface to the operator with a summary of what was built, what's left, and where the spend went. Don't keep grinding past the cap.

Good luck. Build it well.
