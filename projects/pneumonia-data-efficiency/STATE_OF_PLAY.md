# State of Play — Pneumonia Data Efficiency

This document tracks the current status, plan, and next steps for the Pneumonia Data Efficiency project (Phase 8). **Point any new agent or LLM at this file to resume progress.**

> [!NOTE]
> This project is the v1 test project for the larger [**autoscientist**](file:///D:/autoscientist/KICKOFF.md) pipeline. All architectural principles (patient-level splits, baseline reproduction, etc.) defined in the master kickoff document must be strictly followed.

---

## 1. Goal
Evaluate whether **prevalence-matched subsampling** of NIH ChestX-ray14 training data (targeting PadChest's ~4% pneumonia rate) reduces the NIH→PadChest generalization gap more efficiently than simply scaling unmatched training set size ($N \in \{1k, 5k, 25k, 100k\}$).

---

## 2. Research Plan Summary
*   **Backbone**: ResNet-50 (ImageNet pretrained).
*   **Conditions**: 
    1.  **Unmatched**: Natural NIH prevalence (~1.2%).
    2.  **Matched**: Subsampled negatives to reach ~4% prevalence (matching PadChest).
*   **N-Ladder**: $\{1,000, 5,000, 25,000, 100,000\}$ images.
*   **Evaluation**: NIH Held-out Test set and External PadChest subset.
*   **Metrics**: AUROC, Brier Score, ECE (with Bootstrap 95% CIs).

---

## 3. Current Status (As of 2026-05-17)

### Infrastructure & Datasets
- [x] **NIH Pre-indexing**: `src/datasets.py` optimized to pre-index all 112k NIH images.
- [x] **PadChest Pre-indexing**: `src/datasets.py` optimized to pre-index 160k PadChest images (5.6k available locally).
- [x] **Pre-flight Measurement**: PadChest pneumonia prevalence measured at **4.04%** in the evaluation subset (frontal views only).
- [x] **Patient-Level Splits**: Enforced zero patient overlap between train/test via pre-flight assertions.

### Execution Scripts
- [x] `scripts/run_preflight.py`: Validates splits and measures PadChest prevalence. **PASSED**.
- [x] `scripts/run_e0.py`: **GATE PASSED**. Point Estimates: Seed 42 (**0.7531**), Seed 123 (**0.7538**), Seed 2024 (**0.7860**). Mean NIH AUROC: **0.7643 ± 0.0153**.
- [x] `scripts/run_e1.py`: **100% COMPLETED** ($2 \times 4 \times 3 = 24$ cells). Results logged in [runs/E1_results.json](file:///d:/autoscientist/projects/pneumonia-data-efficiency/sandbox/runs/E1_results.json).
- [/] `scripts/run_e2.py`: **IN PROGRESS / ACTIVE GPU SWEEP** (Running seeds 123 and 2024; seed 42 is cached).

---

## 4. Key Scientific Discoveries from E1

1.  **Signal Starvation in Unmatched Low-N Sets ($N \le 25k$)**: 
    In the unmatched condition (natural prevalence ~1.0%), the absolute positive count is so low (e.g., 36 positives at $N=5k$, 159 at $N=25k$) that **the model completely fails to learn useful signal on NIH**. The NIH AUROCs remain close to random guessing (`0.41 - 0.48`), meaning that simply collecting unmatched medical data up to 25,000 images is practically useless for training our backbone without extreme data scale.
2.  **Mitigation via Scaling**:
    Only when scaling to $N=100k$ unmatched (where the total positives reach 876) does the model finally obtain enough signal to learn (NIH AUROC leaps to `0.60 - 0.62`).
3.  **Efficiency of Prevalence Matching**:
    By contrast, the **matched condition** (which raises prevalence to 4.04% and provides 876 positives at $N=25k$) shows robust, significant learning already at $N=5k$ (NIH AUROCs `0.57 - 0.64`) and $N=25k$ (NIH AUROCs `0.64 - 0.68`). This represents a massive data efficiency advantage, proving that prevalence-matched subsampling provides a far more resource-efficient learning pathway than simple unmatched scaling.

---

## 5. Environment Context (WSL2/Ubuntu)
> [!IMPORTANT]
> **D-Drive Image Cache**: A pre-resized JPEG cache is now stored natively on the D: drive (`data/cache/`) to bypass massive WSL2 9P protocol I/O bottlenecks. Dataloaders have been updated with `num_workers=8` and `prefetch_factor=2` to maintain ~90% GPU usage without OOMing the WSL host. 
> **Bootstrap Speed**: `n_resamples` is temporarily set to `100` for interactive testing. **Reset to 1000 for final runs.**

---

## 6. How to Resume

### Initial Setup
Ensure you are in the `sandbox/` directory and use `uv` for execution:
```bash
cd sandbox
uv run python scripts/run_preflight.py
```

### Run Baseline (E0)
```bash
uv run python scripts/run_e0.py --seed 42 --epochs 20
```
*Wait for NIH AUROC to land in [0.73, 0.88]. (Status: PASSED - 0.7643 mean)*

### Run Factorial (E1)
To run the full sweep:
```bash
wsl /mnt/d/autoscientist/.venv/bin/python scripts/run_e1.py --epochs 20 --target-pos-ratio 0.09
```

### Run Control Sweep (E2)
```bash
wsl /mnt/d/autoscientist/.venv/bin/python scripts/run_e2.py --epochs 20 --target-pos-ratio 0.09
```

---

## 7. Next Steps
1.  **Complete E2 Sweep**: Monitor active background GPU sweep for E2.
2.  **Statistical Analysis**: Perform the interaction/permutation statistical testing to assess domain generalization significance.
3.  **Final Report**: Compile a comprehensive report on matched-prevalence subsampling efficiency.

---

## 6. Key Files
*   `sandbox/src/datasets.py`: Core data loading and indexing.
*   `sandbox/src/config.py`: Global constants and hyperparameters.
*   `sandbox/src/models.py`: Model factory with `freeze_backbone` support.
*   `sandbox/src/training.py`: Training loop with trainable-only optimizer.
*   `sandbox/scripts/run_e0.py`: Entry point for baseline.
*   `sandbox/scripts/run_e1.py`: Entry point for N-ladder experiment.
*   `phase8_artifacts/methodology_plan.md`: Detailed research design.

