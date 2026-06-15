"""Dataset loading + patient-level splitting for the pneumonia-transfer study.

Single source of truth for how NIH ChestX-ray14 and PadChest are loaded and
split. Splits are made at the PATIENT level (never image level) to avoid
identity leakage across train/test.

Design notes
------------
- ``patient_level_split`` uses a SHA-256 hash keyed by ``(seed, patient_id)``
  so that the assignment of each patient to train or test is deterministic and
  independent of list ordering.  The same patient will always land in the same
  fold for a given seed, regardless of how the input list is constructed.
- The ``Split`` dataclass is frozen (immutable) to prevent accidental mutation
  after construction.

Usage example
-------------
>>> from src.data import patient_level_split
>>> ids = ["p001", "p001", "p002", "p003"]  # multiple images per patient OK
>>> sp = patient_level_split(ids, test_frac=0.2, seed=42)
>>> set(sp.train_ids) & set(sp.test_ids)  # must be empty
set()
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    """Immutable container for a patient-level train/test partition.

    Attributes
    ----------
    train_ids:
        Tuple of unique patient IDs assigned to the training fold.
    test_ids:
        Tuple of unique patient IDs assigned to the test fold.
    """

    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def patient_level_split(patient_ids: list[str], *, test_frac: float, seed: int) -> Split:
    """Deterministically partition unique patient ids into train/test.

    Hash-based assignment keyed by (seed, patient_id) so the same patient always
    lands in the same fold for a given seed — no image-level leakage.

    Parameters
    ----------
    patient_ids:
        List of patient ID strings, possibly containing duplicates (one entry
        per image is fine; deduplication is performed internally).
    test_frac:
        Fraction of unique patients to assign to the test fold.  Must be in
        (0, 1).
    seed:
        Integer seed mixed into the hash to allow multiple independent splits
        of the same patient population.

    Returns
    -------
    Split
        Frozen dataclass with ``train_ids`` and ``test_ids`` tuples.

    Notes
    -----
    The hash bucket is computed as::

        bucket = int(sha256(f"{seed}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    A patient is assigned to the test fold if ``bucket < test_frac``, otherwise
    to the train fold.  The expected test fraction will match ``test_frac``
    asymptotically; small datasets may show slight deviation.
    """
    uniq = sorted(set(patient_ids))
    test: list[str] = []
    train: list[str] = []
    for pid in uniq:
        h = hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        (test if bucket < test_frac else train).append(pid)
    return Split(train_ids=tuple(train), test_ids=tuple(test))
