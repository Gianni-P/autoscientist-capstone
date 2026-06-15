# Training-Set Size and Cross-Institutional Generalization of CNN Pneumonia Detection

**Paper:** Training-Set Size and Cross-Institutional Generalization of CNN Pneumonia Detection:
An NIH ChestX-ray14 → PadChest Study

## What the paper claims

A ResNet-50 pneumonia detector trained on NIH ChestX-ray14 subsets of size
N ∈ {1 k, 5 k, 25 k, 100 k} is evaluated both in-domain (held-out NIH test split)
and cross-institutionally (PadChest, Spain). Two central findings:

1. **In-domain performance saturates around N = 25 k.** The N = 100 k model does
   not improve over N = 25 k (saturation, not a sign flip).
2. **Cross-institutional AUROC collapses to near-chance (~0.43–0.45) regardless
   of training-set size.** This replicates the domain-shift failure mode documented
   by Zech et al. 2018 and Cohen et al. 2020.

The E0 baseline (full NIH dataset, 3 seeds) yields mean AUROC = 0.764, consistent
with the CheXNet reference value of 0.768 (Rajpurkar et al. 2017). Note that the
CheXNet number uses DenseNet-121 with a different split and is **not directly
comparable** to the ResNet-50 results in this study; it is cited as a sanity-check
reference only.

## Datasets

| Dataset | Source | Access |
|---------|--------|--------|
| NIH ChestX-ray14 | Wang et al. 2017 | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| PadChest | Bustos et al. 2020 | https://bimcv.cipf.es/bimcv-projects/padchest/ |

Both datasets require registration and acceptance of a data-use agreement.
They are **not** included in this repository.

## Runtime

| Experiment | Hardware | Approximate wall time |
|------------|----------|-----------------------|
| E0 (baseline, 3 seeds) | Single GPU (e.g. RTX 3090) | ~15 min |
| E1 (N sweep, 2 conditions × 4 sizes × 3 seeds) | Single GPU | ~3–4 h |
| E2 (prevalence-controlled sweep, 4 sizes × 3 seeds) | Single GPU | ~4–6 h |

Elapsed times recorded in `runs/E1_results.json` and `runs/E2_results.json`
reflect actual sandbox execution times and can be used as a guide.

## Repository layout

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── reproduce.sh
├── src/
│   └── data.py            # patient-level splitting (single source of truth)
├── scripts/
│   └── run_experiments.py # entry point: E0 / E1 / E2 training + evaluation
├── tests/
│   └── test_patient_split.py  # methodology guard tests
├── runs/                  # pre-computed result JSON files
│   ├── E0_summary.json
│   ├── E1_results.json
│   └── E2_results.json
└── figures/
    └── README.md          # figure manifest (figures regenerated from runs/)
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is required (uses `from __future__ import annotations` and
`dataclass(frozen=True)` with PEP 604 union syntax in type hints).

## Reproduce

```bash
bash reproduce.sh
```

Or step by step:

```bash
# 1. Preflight: verify patient-level split correctness
python -m pytest tests/ -v

# 2. E0 baseline (full NIH, 3 seeds)
python scripts/run_experiments.py --experiment E0 --seed 42
python scripts/run_experiments.py --experiment E0 --seed 123
python scripts/run_experiments.py --experiment E0 --seed 2024

# 3. E1 training-size sweep (unmatched prevalence)
for N in 1000 5000 25000 100000; do
  for SEED in 42 123 2024; do
    python scripts/run_experiments.py --experiment E1 --n $N --seed $SEED
  done
done

# 4. E2 prevalence-controlled sweep
for N in 5000 25000 100000; do
  for SEED in 42 123 2024; do
    python scripts/run_experiments.py --experiment E2 --n $N --seed $SEED
  done
done
```

Pre-computed results are already present in `runs/` and can be used to
regenerate figures without re-running training.

## Key results (from pre-computed runs)

| Experiment | N | In-domain AUROC (mean ± std) | External AUROC (mean ± std) |
|------------|---|------------------------------|------------------------------|
| E0 baseline | ~112 k | 0.764 ± 0.015 | — |
| E1 matched | 1 k | ~0.545 | ~0.486 |
| E1 matched | 5 k | ~0.598 | ~0.491 |
| E1 matched | 25 k | ~0.648 | ~0.477 |
| E1 matched | 100 k | ~0.588 | ~0.428 |

External (PadChest) AUROC values near or below 0.5 indicate near-chance or
inverted performance under domain shift. Values below 0.5 are consistent with
systematic inversion of learned features across institutions; see Limitations
section of the paper for discussion of label-definition mismatch between
NIH (English NLP) and PadChest (Spanish NLP) annotations.

## Figures

See `figures/README.md` for the mapping of each paper figure to the script
that produces it.

## Tests

```bash
python -m pytest tests/ -v
```

Two methodology guard tests are included:
- `test_no_patient_in_both_folds` — asserts zero patient-ID overlap between
  train and test splits (guards against image-level leakage).
- `test_split_is_deterministic_per_seed` — asserts that the same seed always
  produces the same split.

## Citation

If you use this code or data, please cite:

```bibtex
@misc{pneumonia_transfer_2024,
  title  = {Training-Set Size and Cross-Institutional Generalization of CNN
             Pneumonia Detection: An NIH ChestX-ray14 → PadChest Study},
  year   = {2024},
  note   = {See CITATION.cff for full metadata}
}
```

## Verified references

- Rajpurkar et al. 2017 — CheXNet. arXiv:1711.05225.
- Wang et al. 2017 — ChestX-Ray8. DOI:10.1109/CVPR.2017.369.
- He et al. 2016 — Deep Residual Learning. DOI:10.1109/CVPR.2016.90.
- Bustos et al. 2020 — PadChest. DOI:10.1016/j.media.2020.101797.
- Pooch et al. 2020 — Domain shift in chest radiograph classification.
  DOI:10.1007/978-3-030-62469-9_7.
- Cohen et al. 2020 — Limits of cross-domain generalization in X-ray prediction.
  arXiv:2002.02497.

## Pending citations

The following reference could **not** be verified by automated DOI round-trip
and must be manually confirmed before publication:

- **Zech et al. 2018** — "Variable generalization performance of a deep learning
  model to detect pneumonia from chest radiographs." PLOS Medicine 15(11):e1002686.
  The DOI 10.1371/journal.pmed.1002686 currently resolves to a different paper
  (Rajpurkar et al. CheXNeXt, 2018) in the literature APIs used. Authors must
  manually verify the correct DOI via the PLOS Medicine website and update
  CITATION.cff and any reference manager accordingly.
  Candidate correct DOI: 10.1371/journal.pmed.1002686 — confirm at
  https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686
