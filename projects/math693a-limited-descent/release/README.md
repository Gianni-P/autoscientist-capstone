# math693a-limited-descent — autoscientist deliverable

This directory is a snapshot of what the **autoscientist** pipeline produced
for the `math693a-limited-descent` project: a self-contained code repository
plus a draft paper and the result summaries behind it. It is the system's
**output**, assembled from the run sandbox so it can be shown and reproduced
independently of the pipeline.

## The study

**Constrained-descent / safe-path-finding on analytic terrains.** Research
question: *how large is the optimality gap of a clockwise/counter-clockwise
rotation heuristic (and related greedy grade-constrained descent strategies)
relative to a Theta\*-corrected Dijkstra ground truth, on a suite of composite
analytic terrain functions with guaranteed unique minima, under a maximum-grade
constraint?* (A reworking of an undergraduate Math 693A project into a
well-posed numerical-optimization study.)

## Contents

| Path | What it is |
|---|---|
| `src/` | The implementation (16 modules): `terrains.py`, `strategies.py`, `graph.py`, `grid_search.py`, the Theta\*/Dijkstra reference, and the E0–E5 experiment drivers. Pure NumPy/SciPy, no external data. |
| `tests/` | 25 test modules — grade-constraint enforcement, reference-construction gates, determinism, gap-decomposition consistency, supercover/LOS checks, per-experiment pitfalls. |
| `paper/` | The draft paper (`paper.pdf`, `paper.tex`). |
| `results/` | The aggregate result summaries (`E1`–`E5_summary.json`) — the real per-terrain numbers the paper draws on. |
| `conftest.py` | pytest fixtures. |

## Reproduce

```bash
cd release
pip install numpy scipy            # the only dependencies
python -m pytest tests/ -q         # run the test suite
python -m src.main                 # run the experiment drivers (see src/experiment_e*.py)
```

## Status — honest

This is a **work-in-progress draft, not an accepted paper.** The pipeline
carried the project end-to-end through all five human checkpoints: idea
selection, methodology, code review, full-results validation (verdict:
*advance*), and draft review. At the final gate (draft review) the simulated
`peer_reviewer` returned **major-revise / reject** — the principal objections
were a **narrow experimental scope** (the effect is pronounced on only one of
the five terrains), an unproven sufficiency argument for the proposed fix, and
**citation-verification failures** on two references. The code and test suite
are substantive and pass; the scientific contribution is modest and the
write-up would need broader experiments and a cleaned-up reference list before
submission anywhere.

In other words: this is the first project autoscientist drove autonomously
across CP3–CP5 with real (verified) results and a real critical review — a
genuine end-to-end run, with the draft's limitations honestly surfaced by the
pipeline's own review gate rather than hidden.
