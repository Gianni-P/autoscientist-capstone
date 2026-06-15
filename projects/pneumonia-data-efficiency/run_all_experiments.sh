#!/usr/bin/env bash
export PATH="/home/gdp/.local/bin:$PATH"
cd /home/gdp/autoscientist/projects/pneumonia-data-efficiency/sandbox

PYTHON="/home/gdp/autoscientist/.venv/bin/python"

echo "=== E0 Baseline Reproduction ==="
$PYTHON scripts/run_e0.py --epochs 15 --target-pos-ratio 0.09

echo "=== E1 Ladder Experiment ==="
$PYTHON scripts/run_e1.py --epochs 20 --target-pos-ratio 0.09

echo "=== E2 Control Arm ==="
$PYTHON scripts/run_e2.py --epochs 20 --target-pos-ratio 0.09

echo "=== All experiments launched successfully! ==="
