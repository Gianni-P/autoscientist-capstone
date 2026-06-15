#!/usr/bin/env bash
# reproduce.sh — Full reproduction pipeline for the pneumonia transfer study.
#
# Steps:
#   1. Methodology guard tests (patient-level split, seed determinism)
#   2. E0 baseline (full NIH dataset, 3 seeds)
#   3. E1 training-size sweep (unmatched and matched prevalence, 4 sizes × 3 seeds)
#   4. E2 prevalence-controlled sweep (3 sizes × 3 seeds)
#
# Prerequisites:
#   - Python virtual environment activated with requirements.txt installed.
#   - NIH ChestX-ray14 and PadChest datasets downloaded and paths configured
#     in scripts/run_experiments.py (see paper Methods §3 for path conventions).
#
# Approximate total wall time on a single GPU: 8–12 hours.
# Pre-computed results are already present in runs/ and can be used to
# regenerate figures without re-running training.

set -euo pipefail

echo "=== Step 1: Methodology guard tests ==="
python -m pytest tests/ -v

echo "=== Step 2: E0 baseline ==="
for SEED in 42 123 2024; do
    python scripts/run_experiments.py --experiment E0 --seed "$SEED"
done

echo "=== Step 3: E1 training-size sweep ==="
for N in 1000 5000 25000 100000; do
    for SEED in 42 123 2024; do
        python scripts/run_experiments.py --experiment E1 --n "$N" --seed "$SEED"
    done
done

echo "=== Step 4: E2 prevalence-controlled sweep ==="
for N in 5000 25000 100000; do
    for SEED in 42 123 2024; do
        python scripts/run_experiments.py --experiment E2 --n "$N" --seed "$SEED"
    done
done

echo "=== Reproduction complete. Results written to runs/. ==="
