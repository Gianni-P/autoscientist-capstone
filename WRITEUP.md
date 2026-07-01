# autoscientist — A Human-Gated Harness for Trustworthy Autonomous Research

**Subtitle:** Twelve specialist agents turn a research direction into a paper and a reproducible repo — safety and correctness enforced by the harness

**Track:** Freestyle

---

## The problem

Autonomous "research agents" are easy to demo and easy to fool. Ask one to "study whether X improves Y" and it will happily produce a polished paper — built on a hallucinated citation, a train/test split that leaked, a baseline it never actually ran, and code that does not even import. The fluent prose hides the fact that nothing underneath is true.

The hard part of autonomous research is therefore *not* generating text. It is building the **safety envelope** that makes the output trustworthy enough to act on: the sandbox that contains the code, the budget gate that stops a runaway loop, the deterministic checks the model cannot talk its way past, and the human approvals at the points where being wrong is expensive. That envelope is an engineering problem, and it is exactly the "agentic engineering" discipline the course contrasts against casual vibe coding.

**autoscientist** is that envelope. It turns a one-line research direction into a finished academic paper and a self-contained, reproducible code repository by driving **twelve specialist LLM agents** through a constrained, human-gated pipeline. Every expensive or irreversible step is sandboxed, budget-capped, independently verified, and gated by a human at five mandatory checkpoints.

## Why agents (and why not one prompt)

A single prompt cannot do this safely, because the task is genuinely multi-stage and each stage needs a *different* capability, model, and form of verification:

- **literature review** needs web/PDF tools and a cheap, fast model;
- **idea generation and critique** need a strong reasoning model and an adversarial second opinion;
- **methodology** needs to commit to datasets, baselines, metrics, and a statistics plan;
- **code and test synthesis** need file-writing and a sandbox;
- **code review and results validation** need to *re-derive* claims, not trust them;
- **figure generation** writes and runs a plotting script to render the paper's figures from the validated results;
- **paper drafting and peer review** need citation verification and LaTeX;
- **release** needs to publish a curated repo.

Splitting the work into specialist agents on a **fixed handoff topology** gives each agent a small, auditable contract: it sees only its inbound payload, it has a narrow capability-scoped toolset, and it hands off to a known successor. That structure is what makes the human checkpoints possible — at each gate the operator sees exactly one agent's structured output and approves, edits, re-runs, or rejects before the pipeline spends money or GPU-hours on the next leg. The harness enforces the discipline; the agents supply the work; the human supplies the taste.

## Architecture

```
research direction
        │
        ▼
  lit_review → idea_gen → idea_critic ──①── methodology ──②──┐
                                                             │
  ┌──────────────────────────────────────────────────────────┘
  ▼
  code_gen → test_gen → code_review ──③── results_validator ──④──┐
       ▲          │                                              │
       └─ revise ─┘  (bounded loop, capped then escalated to ③)  │
  ┌──────────────────────────────────────────────────────────────┘
  ▼
  figure_gen → paper_writer → peer_reviewer ──⑤── repo_publisher → paper.pdf + release repo

  ①..⑤  five mandatory human-in-the-loop checkpoints
```

The pipeline is deterministic code (`runtime/runner.py`): it invokes an agent, runs that agent's tool-use loop, parses a `HANDOFF: <target>` directive, and routes to the next agent. State lives **outside the model** in SQLite — runs, every message, checkpoints, and an append-only budget ledger — so the system is fully observable and resumable. A detached runner process drives the chain; a Starlette web console pushes live state over Server-Sent Events.

## The operator console

The console is where the human-in-the-loop actually happens, and it is the most visible part of the system. A push-based web app (Starlette + Server-Sent Events) renders every run live — a five-stage checkpoint stepper, the streaming activity feed, per-agent handoffs with the exact prompt each agent received, a budget meter, and the agent roster — with no polling. At each of the five gates the operator can **approve, approve-with-changes, re-run with a nudge, or reject**; and — the course-week addition (below) — **pick the model for each agent in the next leg**, including the Opus-orchestrator option for `code_gen`/`test_gen`/`figure_gen`. This is the security/HITL concept made tangible: the place where a human supplies the judgment the agents lack, and where model routing becomes a per-decision choice rather than a config edit. Ten annotated screenshots are in `docs/screenshots/` (and the project README).

## What the course changed — the increment I built during the course week

The course crystallized three ideas for me: **intelligent model routing** (Day 1), the **conductor → orchestrator** shift to async multi-agent delegation (Day 1), and the **economics of AI development** — High CapEx / Low OpEx (Day 1). During the course week I applied all three by shipping a single feature into the harness:

**A per-leg model picker, plus an "Opus-orchestrator" mode for code generation.**

At each human approval gate, the operator now chooses *which model each agent in the next leg uses* — model routing exposed exactly where the course puts control: the human-in-the-loop checkpoint. The choice rides inside the checkpoint's operator decision (no schema change) and resets at the following gate.

For the two highest-volume agents (`code_gen`, `test_gen`), one option is an **orchestrator**: Claude Opus 4.8 *plans and spot-checks* while a local `qwen3.6:27b` worker *writes the files*, via a `delegate` tool. Opus decomposes the task, hands each file-level assignment to the worker, and gets back a compact summary (sandbox listing + a static import check) — so the expensive model reads spot-checks, not full files, and is explicitly instructed to re-read and verify any file with real math before handing off. This is the course's async multi-agent delegation and its cost thesis made concrete: pay frontier prices only for planning and verification; push the verbose work to a local model for ~$0. The worker is a **write-only, sandboxed sub-agent** with no `execute`, no `handoff`, and no `delegate` tool, so it cannot debug-spin, route, or recurse.

That one increment touches Day 1 (routing + delegation + economics) and Day 4 (it lives at a checkpoint, is metered by the budget circuit-breaker, and runs in a sandbox) at once. It is the clearest answer to "what did the course give me": a concrete capability, shipped, with tests.

## Course concepts demonstrated

The capstone asks for at least three. I demonstrate three core concepts plus a bonus, all in code.

### 1. Multi-agent system (Code)

Twelve specialist agents on a fixed handoff topology, each defined declaratively (`agents/`) with a system prompt, a capability-scoped toolset, and allowed handoff targets, and driven by a shared run loop (`runtime/runner.py`). The newest agent, `figure_gen`, renders the paper's figures from the validated results — it writes and runs a plotting script, then hands the figures to `paper_writer` to embed — and supports the same Opus-orchestrator mode as the code agents. On top of that sits the orchestrator-and-worker pattern above (`runtime/orchestration.py`) — a multi-agent system *within* an agent. Agents communicate only through structured handoff payloads, and off-topology or missing handoffs are corrected by the harness rather than blindly followed.

### 2. MCP server (Code)

The terminal `repo_publisher` agent publishes the curated release repository through the official **GitHub MCP server**, via a bridge that connects autoscientist's synchronous tool loop to MCP over **both stdio and remote HTTP/SSE** transports (`clients/mcp_bridge.py`, `config/mcp.toml`) — the two transports the course describes. It is **scoped** to one project's resources, runs token-gated, and **degrades gracefully**: if the server or token is missing, it logs the fact, skips the push, and still writes the local release tree, so a failed integration never fails the run. MCP tools are registered into the same registry as native tools, so an agent calls them identically.

### 3. Security features (Code)

This is where autoscientist is deepest, mapping directly onto the course's security pillars and zero-trust net:

- **Ephemeral sandboxing & egress governance** — the `execute` tool runs agent-written code under CPU, memory, and wall-clock limits with **outbound network blocked**, and an argv-allowlist (python/pytest only, no shell strings) that closes the command-injection path.
- **Mitigating hallucinated dependencies** — a static, read-only `check_imports` tool parses every file with the AST and resolves every intra-project import *without executing anything*, catching the "imports a name nothing defines" failure before it ever runs.
- **Checkpoints & a stateful circuit-breaker** — five mandatory HITL gates, plus a hard monthly **budget circuit-breaker** that refuses new spend within a buffer of the cap, enforced race-free by reserving the estimated charge under a single write lock *before* the call and reconciling to the real cost after. Per-project soft caps and per-invocation ceilings bound a loop of individually-cheap calls.
- **Observability** — every turn, tool call, cost, and handoff is written to a per-run JSONL trace and the SQLite message ledger, and streamed live to the console.

### Bonus — Deployability (Code/Video)

The system runs as a detached runner process plus a web console, with documented setup and a one-line config switch to run **cloud-only with no GPU** (point the code agents at a hosted model). A live public endpoint is not required for judging; the public repository and setup instructions stand in.

## Implementation highlights (clever tool use)

A few decisions that show *meaningful* use of agents and toolsets rather than tool-calling for its own sake:

- **Capability restriction as the security boundary.** Each agent is offered only its declared tools, enforced at the schema layer. `code_gen` deliberately has **no** `execute` tool — an earlier version with it burned every tool round in a debug-spin and never handed off. Removing the capability fixed the behavior; the toolset *is* the policy.
- **A parse-proof handoff tool.** Local models routinely failed to emit a bare `HANDOFF:` text line, so handoff is also a structured tool call the harness validates against the agent's allowed targets — with the bare line kept as a legacy fallback.
- **Input reconstruction backstops.** Several agents (code_review, paper_writer, peer_reviewer) detect a "thin" inbound payload and rebuild their real input from the sandbox and the run's history, so an upstream stumble degrades into a real review instead of a degenerate checkpoint.
- **A tool-result cache and a finalize-nudge** keep the cost and the verdict-emission of long tool loops bounded.

The code is heavily commented at the level of *why* — most non-obvious guards carry the run ID and date of the failure that motivated them.

## Verification & evaluation

The course splits trust into two axes — **security** (did the agent stay in bounds?) and **evaluation** (is the result worth shipping?) — and autoscientist implements both. Beyond an LM-judge agent scored against per-agent rubrics (`meta/eval_rubrics.py`) and prompt versioning / A-B harness, a dedicated `verify/` package runs **deterministic** gates the model cannot argue with: data-leakage detection, baseline-reproduction tolerance checks, statistical-validity checks, domain-specific pitfall handlers, **experiment-completeness** checks (a plan-declared experiment with no result artifact halts the pipeline — the failure mode that lets a system write a paper about experiments it never ran), and **claim-provenance / claim-verification** (every quantitative claim in the paper must trace to a value in a cited results artifact). A `test_gen` agent writes AI-generated test coverage *targeting the methodology's pitfalls* before code is accepted — programmatic confidence wired in as a gate, not an afterthought. The project ships **384 passing unit tests**.

## Result: a real end-to-end deliverable

The flagship run reworked an undergraduate study, *"Limited Descent: The Use of Gradient Descent in Mountain Rescue,"* into a well-posed **constrained-descent / safe-path-finding** study and carried it autonomously through all five checkpoints to a compiled paper and a reproducible repo (`projects/math693a-limited-descent/`). It compares a rotation heuristic, a feasible-cone projection, and unconstrained steepest descent against a Dijkstra/A\* shortest-safe-path ground truth — pure NumPy/SciPy, no external data, full sweep under fifteen minutes, with optimization-specific pitfall checks enforced throughout. The deliverable is a `paper.pdf` (its figures rendered by a dedicated `figure_gen` agent) plus a `src/` + ~30 tests + results + paper release tree. A second run, on **real** public-health data (`california-cancer-ocean`), drove the same pipeline through an observational question — whether distance to the Pacific Ocean predicts California cancer incidence after adjustment — and produced an honest negative-control falsification (the crude coastal gradient largely vanishes once pollution, SES, smoking, and screening are controlled for), showing the harness holds up outside a self-contained simulation.

## Limitations (stated honestly)

- It is **not built on ADK**. autoscientist is a bespoke harness on Anthropic Claude plus a local Ollama/Qwen worker; it demonstrates the course's *concepts* (harness, routing, MCP, security, evaluation, HITL), not Google's specific framework.
- It is a **substrate I extend**, not a weekend build. The harness predates the course; the model-router and orchestrator increment described above is the course-week work, and I am transparent about which is which.
- It **does not replace domain expertise**. The five human checkpoints exist precisely because the agents have no real scientific judgment — the harness enforces discipline; the human supplies taste.

## Reproduce & links

- **Code:** the public repository, with full setup in `README.md`, including a one-line **GPU-free** configuration (route the code agents to a hosted model).
- **Quick verification with no spend and no GPU:** the phase smoke tests run the whole substrate against a deterministic mock provider; `pytest` runs all 384 unit tests.
- **Concept mapping:** `WRITEUP.md` (this document) and the README's "Course concepts demonstrated" table point at the exact files.

autoscientist is the disciplined end of the vibe-coding spectrum: a system whose primary output is not code, but the harness that produces trustworthy science.
