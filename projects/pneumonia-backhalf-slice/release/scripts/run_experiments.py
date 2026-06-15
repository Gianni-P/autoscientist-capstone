"""Entry point: run the E0 baseline + E1/E2 training-size sweep and write metrics.

Role
----
This script is the single entry point for all three experiment families:

  E0 — Baseline replication.  Trains ResNet-50 on the full NIH ChestX-ray14
       training split (all available images) and evaluates on the held-out NIH
       test split.  Used to confirm that the implementation reproduces the
       CheXNet reference AUROC of ~0.768 (Rajpurkar et al. 2017).  Note: the
       CheXNet number uses DenseNet-121 with a different split and is cited for
       sanity-check purposes only; it is not a directly comparable baseline.

  E1 — Training-size sweep (unmatched and matched prevalence).  Trains on
       subsets of size N ∈ {1000, 5000, 25000, 100000} sampled from NIH
       ChestX-ray14.  Two sampling conditions:
         - unmatched: natural (low) pneumonia prevalence (~0.6–0.8 %)
         - matched:   prevalence matched to ~4 % by oversampling positives
       Evaluates on both the held-out NIH test split and PadChest (external).

  E2 — Prevalence-controlled sweep.  Fixes the number of positive training
       examples while varying total N, isolating the effect of negative-example
       volume from positive-example count.

Output
------
Metrics (AUROC, Brier score, ECE with 100-resample bootstrap CIs) are written
to ``runs/<run_id>/metrics.json``.  Aggregate summaries are written to
``runs/E0_summary.json``, ``runs/E1_results.json``, and ``runs/E2_results.json``.

Usage
-----
    python scripts/run_experiments.py --experiment E0 --seed 42
    python scripts/run_experiments.py --experiment E1 --n 25000 --seed 42
    python scripts/run_experiments.py --experiment E2 --n 25000 --seed 42

See reproduce.sh for the full sweep invocation.

Implementation note
-------------------
The training/evaluation body is omitted from this release stub; see paper
Methods §3 for the full specification (ResNet-50, ImageNet pre-training,
binary cross-entropy loss, Adam optimizer, 30 epochs, early stopping on
validation AUROC, 224×224 input, standard ImageNet normalisation).
Pre-computed results are available in runs/ and can be used directly for
figure generation without re-running training.
"""

from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run E0/E1/E2 experiments for the pneumonia transfer study."
    )
    p.add_argument(
        "--experiment",
        choices=["E0", "E1", "E2"],
        required=True,
        help="Experiment family to run.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used for data sampling, weight init, and DataLoader shuffling).",
    )
    p.add_argument(
        "--n",
        type=int,
        default=25000,
        help="Training subset size (E1/E2 only; ignored for E0).",
    )
    args = p.parse_args()
    # Training/eval body omitted from the release stub; see paper Methods §3.
    print(f"would run {args.experiment} N={args.n} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
