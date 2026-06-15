"""Methodology guard tests: patient-level split correctness.

These tests validate two critical properties of the splitting function used
throughout the study:

1. No patient appears in both the train and test folds (guards against
   image-level identity leakage, a well-known pitfall in chest X-ray
   classification literature).

2. The split is deterministic for a given seed (required for reproducibility
   across machines and runs).

Run with:
    python -m pytest tests/test_patient_split.py -v
"""

from __future__ import annotations

from src.data import patient_level_split


def test_no_patient_in_both_folds():
    """Train and test sets must be disjoint at the patient level.

    Simulates a realistic scenario where each patient has multiple images
    (3 images per patient here).  The split must operate on unique patient IDs,
    not on individual images.
    """
    ids = [f"p{i}" for i in range(500)] * 3  # 3 images per patient
    sp = patient_level_split(ids, test_frac=0.2, seed=42)
    assert not (set(sp.train_ids) & set(sp.test_ids))


def test_split_is_deterministic_per_seed():
    """Calling patient_level_split twice with the same seed must return identical splits.

    This guards against any non-determinism in the hash-based assignment
    (e.g., set iteration order, hash randomisation).
    """
    ids = [f"p{i}" for i in range(200)]
    a = patient_level_split(ids, test_frac=0.2, seed=7)
    b = patient_level_split(ids, test_frac=0.2, seed=7)
    assert a == b
