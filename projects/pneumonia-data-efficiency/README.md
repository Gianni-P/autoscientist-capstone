# pneumonia-data-efficiency

The v1 end-to-end test project for autoscientist. Defined in
`KICKOFF.md` Section 8. This directory holds the per-project artifacts
the runner needs: kickoff payload, project-scoped config overrides,
and the sandbox that generated code is restricted to.

## Research question

How does training data size affect cross-institutional generalization
in CNN-based pneumonia detection from chest radiographs?

## Datasets

| Role               | Dataset           | Source       | Notes                            |
|--------------------|-------------------|--------------|----------------------------------|
| Training pool      | NIH ChestX-ray14  | Kaggle       | ~112k images, weak NLP labels    |
| External val       | PadChest          | BIMCV        | requires registration + token    |
| Optional 2nd val   | CheXpert          | Stanford AIMI| post-v1                          |

The `tools/datasets.py` registry already knows about both. NIH is
fetched via Kaggle API (`~/.kaggle/kaggle.json` is required); PadChest
requires `BIMCV_TOKEN` and a per-project `bimcv_url` set in the
registry after operator registration.

## Experimental design

Per KICKOFF Section 8, the methodology agent should land near:

- Fine-tune ResNet-50 (ImageNet pretrained) for binary pneumonia
  classification.
- Train on NIH ChestX-ray14 subsets of size N in {1k, 5k, 25k, 100k}.
- For each N, evaluate on the held-out NIH test split AND on the
  PadChest pneumonia-labeled subset.
- Plot generalization gap (in-domain AUROC minus external AUROC) versus
  training size N.
- Run 3 seeds per N, report mean +/- SD.
- Compare to a published reference point (Rajpurkar CheXNet or similar).

## Project files

- `config.toml` -- per-project overrides (budget, model routing,
  pitfall domain). The runtime merges this on top of `config/default.toml`
  and `config/models.toml`.
- `kickoff_payload.json` -- the JSON payload fed to `lit_review` as the
  first agent's input message.
- `sandbox/` -- restricted CWD for generated code per KICKOFF Section 10.
  All training scripts the agents emit run here, with stdout/stderr/exit
  captured by `tools/execute.py`.

## Checkpoint expectations

All five HITL checkpoints fire (KICKOFF Section 7). For this project the
operator should expect:

| # | Stage                        | What to look for                                                  |
|---|------------------------------|-------------------------------------------------------------------|
| 1 | Idea selection               | 5 candidates; pick the one matching the data-efficiency framing   |
| 2 | Methodology approval         | Confirms patient-level split, 3 seeds, N sweep, PadChest external |
| 3 | Preliminary results          | Tiny-subset run; sanity-check curves, baseline reproduction state |
| 4 | Full results validation      | Verify harness must clear; counterintuitive findings explained    |
| 5 | Draft review                 | LaTeX builds via tectonic; every citation round-trips             |

## Hard rules in scope

- **Patient-level splits** (NIH and PadChest both have patient ids; mixing
  images across the split is leakage).
- **3 seeds per N with reported variance** (Phase 7 pitfall enforces).
- **No hyperparameter tuning on test split** (Phase 7 pitfall enforces).
- **External validation present** since the research question explicitly
  claims generalization (Phase 5 pitfall enforces).
- **Baseline must reproduce within tolerance** before any "novel" claim
  (KICKOFF Section 4 principle 7; Phase 5 pitfall enforces).
- **Weak-label provenance disclosed** -- both NIH and PadChest labels
  are NLP-derived and must be acknowledged in limitations
  (Phase 7 pitfall surfaces to the operator).

## Launch

See `scripts/run_phase8.md` for the full runbook. The short version:

```bash
cd /mnt/d/autoscientist
uv run python scripts/dry_run_phase8.py            # ~1 cent, validates wiring
uv run python -m autoscientist.runtime.runner \
    --agent lit_review \
    --project pneumonia-data-efficiency \
    --payload "$(cat projects/pneumonia-data-efficiency/kickoff_payload.json)"
# In another terminal: uv run streamlit run src/autoscientist/checkpoints/ui.py
```
