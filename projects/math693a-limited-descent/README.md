# math693a-limited-descent

A reworking of the undergraduate project **"Limited Descent: The Use of Gradient
Descent in Mountain Rescue"** (Math 693A final, G. Pucillo) into a well-posed
**constrained-descent / safe-path-finding** study for the autoscientist pipeline.

Source: the author's original Math 693A final-project report.

## Why this project exists

It is the **cheapest and cleanest end-to-end candidate** the pipeline has had,
and it removes the two things that have blocked every prior run:

| Blocker on prior runs | Here |
|---|---|
| Expensive code loop / data plumbing | Pure NumPy/SciPy, **no external data**, seconds per run, < 15 min full sweep |
| No real domain expertise (clinical) | The subject **is** numerical optimization — the agents reason about it natively |

It became the **first project to traverse CP3–CP5 end-to-end** — see **Outcome** below.

## The original method, and its admitted flaw

The original takes fixed-step (l = 0.1) steepest descent over an analytic
terrain `f(x,y)` and, whenever a step would drop more steeply than a maximum
grade θ = 5°, **rotates the travel direction in 1° increments** (clockwise or
counter-clockwise) until the drop is ≤ `l·tan θ`. It compared cw / ccw / either
rotation by total 3-D path length on two terrains.

Its own Discussion identified the **fatal confound**: each rotation strategy
converges to a *different endpoint*, so comparing path lengths is ill-posed —
it measures distance to *a* minimum, not to a fixed destination. The appendix
code also has an **unbounded inner `while`** (rotate-until-walkable with no turn
cap) that can spin forever on a cliff face.

## Research question

> Once the objective is given a **single defined target** (a sink = the
> trailhead/road) so all strategies share an endpoint, does the original
> **rotation heuristic** approach the **constrained shortest safe path**, or
> merely a feasible-but-arbitrary one? And is the clockwise/counter-clockwise
> asymmetry a real effect or an artifact of the undefined endpoint?

A clean "feasible but far from optimal" result is a legitimate
negative/limitation finding at the workshop / short-methods-note ceiling — and
gives the verification harness a genuine correctness story to check.

## Outcome

The pipeline carried this project end-to-end through all five human checkpoints
to a compiled paper and a self-contained, reproducible release repo (in
[`release/`](release/)). A dedicated **`figure_gen`** agent rendered the paper's
four result figures from the validated runs — Corrected Optimality Gap by
strategy and terrain, grade-feasibility (unconstrained vs. the rotation
heuristic), the H1 rotation-COG test across terrains, and the grid-reference
quantisation-bias check — and embedded them in `release/paper/paper.pdf`;
`release/scripts/generate_figures.py` regenerates them from the result
summaries in `release/results/`.

The headline result is a clean **feasible-but-not-optimal** finding: the rotation
heuristic and the principled feasible-cone projection stay grade-feasible and
match or beat the Dijkstra shortest-safe-path reference on the smooth terrains,
while a ridge terrain exposes a large optimality gap for every method, and
unconstrained steepest descent violates the grade limit on a third or more of its
steps. Stated honestly: this is a **work-in-progress draft** — the simulated peer
review returned *major revise* (narrow experimental scope and a reference list to
clean up), which the pipeline surfaced at the final checkpoint rather than hid.

## Design

| Property | Choice | Why |
|---|---|---|
| Compute | NumPy/SciPy on CPU | Every evaluation is microseconds; full sweep < 15 min |
| Terrains | `mountain_one` (peaks (0,0),(1,1),(2,-1)), `mountain_two` (peaks (-1,-1.5),(0.5,3), a cliff face), + sink-augmented variants | Reuse the originals; add a defined sink so endpoints are comparable |
| Methods | rotation heuristic · **feasible-cone projection** (principled exact-θ step) · unconstrained steepest descent | (b) is the principled version of the heuristic the original gestured at |
| Ground truth | **grid shortest-safe-path** (Dijkstra/A*) under the same grade cap | The optimum the original lacked, to measure an optimality gap |
| Starts | ≥ 5 initial points (injured-hiker locations) per terrain | The original used one; behavior depends heavily on the start |
| Metrics | feasibility (grade ≤ θ at **every** step) · path length · **optimality gap** · convergence iters | Feasibility and gap are the point, not raw distance |

Terrains are reconstructed as closed-form analytic functions (e.g. sums of
Gaussians at the stated peak locations, plus a quadratic basin for the sink);
the methodology/code agents fix the exact forms at CP2/CP3.

## Pitfalls / verification

Verification uses `config/domains/numerical_optimization.toml` (set via
`[verify].domain`). It is the new optimization counterpart to the imaging and
tabular domains, with six optimization-specific handlers added to
`verify/pitfalls.py`:

- `constraint_feasibility_verified` — grade ≤ θ at **every** step (the central gate)
- `descent_terminates` — bounded rotation loop + max-iteration budget + convergence
- `well_defined_convergence_target` — shared defined target so endpoints are comparable
- `gradient_validated` — analytic gradient checked vs finite differences / autodiff
- `optimality_gap_reported` — optimality claims backed by the grid optimum + a gap
- `step_discretization_sensitivity` — sensitivity to step length / rotation increment

plus four reused generic checks: `baseline_reproduced_within_tolerance`,
`multi_seed_reporting` (multi-start), `counterintuitive_signs_flagged`
(the cw/ccw asymmetry), and `confidence_intervals_reported`.

## Launch

Two WSL terminals (see the repo `README.md` "Running a project end-to-end" for
the full pre-flight, pause/resume, and spend-monitoring details):

```bash
# Terminal A — the runner
cd ~/autoscientist
set -a; source .env; set +a
PAYLOAD=$(cat projects/math693a-limited-descent/kickoff_payload.json)
uv run python -m autoscientist.runtime.runner \
    --agent lit_review \
    --project math693a-limited-descent \
    --payload "$PAYLOAD"
# Note the printed run_id.

# Terminal B — the operator console
uv run autoscientist-web   # http://127.0.0.1:8650
# (Streamlit fallback: uv run streamlit run src/autoscientist/checkpoints/ui.py)
```

You will be paused at **CP1 (idea selection)** first, then **CP2
(methodology)**. With no data and a tiny code surface, CP3–CP5 should finally
arrive on a real run.

## Before a long unattended run

- Run the domain smoke first. It guards the six new pitfall handlers and the
  TOML wiring (clean state → all 10 pass; targeted mutations → each fires;
  outcome aggregation honors severities), with **zero spend**:

  ```bash
  uv run python scripts/smoke_numerical_optimization.py
  # *** All numerical_optimization smoke checks passed. ***
  ```

  This is the regression gate for the new handlers, since the WSL venv test
  suite can't be run from the Windows host.
- Model routing is **operator-selectable per leg** at each checkpoint: the
  `code_gen` / `test_gen` / `figure_gen` agents can run on a hosted model, a
  local `qwen2.5:32b` worker (≈ $0) via Ollama, or the **Opus-orchestrator**
  mode (Opus plans and spot-checks; the local worker writes the files). A
  per-project soft cap is the backstop. If you route to a local model, confirm
  Ollama is up (`scripts/smoke_local_toolcall.py`) before launching.
