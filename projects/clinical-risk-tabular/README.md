# clinical-risk-tabular

A **fast** autoscientist test project: tabular clinical-risk prediction with
scikit-learn. Designed so the autonomous chain can run end-to-end in minutes,
not hours, and finally exercise checkpoints 3–5 (preliminary results, full
results validation, draft review) — which no prior run has reached
(see `../../ASSESSMENT_2026-05-29.md`).

## Research question

> Do flexible ML models (gradient-boosted trees, random forests) actually beat
> a well-regularized **logistic regression** for clinical risk prediction on
> tabular data, once you score **calibration** (ECE, Brier) and not just
> discrimination (AUROC) — and how does the answer depend on training-set size?

This is a real, contested question. Christodoulou et al. (2019, *J Clin
Epidemiol*) systematically found **no benefit of ML over logistic regression**
for clinical prediction; Van Calster et al. argue calibration is the routinely
ignored "Achilles heel." A clean negative or trade-off result here is
publishable at the workshop/short-letter ceiling — and gives the verification
harness a genuine counterintuitive-finding to catch.

## Why this shape

| Property | Choice | Why |
|---|---|---|
| Compute | scikit-learn on CPU | Every fit is seconds; full sweep < 45 min |
| Models | LogReg(L2), RandomForest, HistGradientBoosting | All sklearn — **no new deps** (no xgboost/lightgbm) |
| Datasets | `support2`, `diabetes_130`, `heart_disease_uci` | Public, tiny, no credentialing |
| Design | datasets × models × N∈{500,2k,8k,full} × 5 seeds × 5-fold | Cheap factorial with a data-efficiency axis |
| Eval | AUROC, AUPRC, **ECE, Brier, calibration slope** | Calibration is the whole point |

## Datasets — suggested sources (operator to confirm at CP2/CP3)

Fetched **in-sandbox** by the generated code (via `sklearn.datasets.fetch_openml`
or a direct CSV), **not** via the `dataset_fetch` tool. Suggested provenance —
verify the exact identifier when the code is generated:

- **support2** — SUPPORT study, ~9,105 rows, in-hospital/6-month mortality.
  Canonical source: Vanderbilt Biostatistics (`hbiostat.org/data/repo`), also on OpenML.
- **diabetes_130** — UCI "Diabetes 130-US hospitals 1999–2008", ~101,766 rows,
  30-day readmission. UCI ID 296; also on OpenML (`Diabetes130US`). **Note its
  repeated-admission structure → split by patient.**
- **heart_disease_uci** — UCI Cleveland heart disease, ~303 rows. Classic,
  well-baselined; on OpenML.

If you'd rather pre-stage data (as was done for the imaging project), drop CSVs
under `../../data/` and point the methodology/code at them; otherwise the chain
fetches them.

## Pitfalls

Verification uses `config/domains/clinical_tabular.toml` (set via
`[verify].domain` in `config.toml`). It targets the tabular failure modes:
preprocessing/imputation leakage, post-outcome target leakage, patient-level
splitting, calibration reporting, validation-only tuning, ≥5 seeds + CIs,
baseline reproduction, and counterintuitive-finding flags.

## Launch

Two WSL terminals (see the repo `README.md` "Running a project end-to-end" for
the full pre-flight, pause/resume, and spend-monitoring details):

```bash
# Terminal A — the runner
cd /mnt/d/autoscientist
set -a; source .env; set +a
PAYLOAD=$(cat projects/clinical-risk-tabular/kickoff_payload.json)
uv run python -m autoscientist.runtime.runner \
    --agent lit_review \
    --project clinical-risk-tabular \
    --payload "$PAYLOAD"
# Note the printed run_id.

# Terminal B — the operator console
uv run streamlit run src/autoscientist/checkpoints/ui.py
```

You will be paused at **CP1 (idea selection)** first; approve a direction, then
**CP2 (methodology)**. The fast compute is what should let CP3–CP5 actually
arrive this time.

## Recommended before a long unattended run

- Apply the **per-invocation cost cap for `test_gen`** (ASSESSMENT R2). The
  `$20` project soft cap here is only a backstop; the historical runaway burned
  ~$16 in a single `test_gen` invocation.
- Consider the **thin CP3→CP4→CP5 slice** (ASSESSMENT R3): feed
  `results_validator` a small canned results payload to prove the back half of
  the pipeline — none of `results_validator` / `paper_writer` / `peer_reviewer`
  / `repo_publisher` has ever run on a real project.
