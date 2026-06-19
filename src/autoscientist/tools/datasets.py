"""Public dataset registry + fetchers.

Datasets are defined declaratively in ``DATASET_REGISTRY``. Each entry
specifies a fetch method (``kaggle`` | ``http`` | ``bimcv``) and optional
SHA256 checksums for verification.

Phase 3 v1: NIH ChestX-ray14 (Kaggle) and PadChest (BIMCV stub).

The actual file data is large (NIH: ~45 GB, PadChest: ~1 TB raw / much
smaller with the pneumonia-labeled subset). The fetcher is idempotent —
if files already exist on disk and pass checksum (or no checksum is
configured), the fetch is a no-op. This keeps the smoke test fast and
prevents accidentally re-downloading multi-GB archives.

BIMCV note: PadChest requires manual registration at bimcv.cipf.es and a
project-specific access URL. The ``bimcv`` fetcher accepts a URL +
``BIMCV_TOKEN`` env var; the operator wires this once they have access.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import httpx
import structlog

log = structlog.get_logger("autoscientist.tools.datasets")


@dataclass
class DatasetSpec:
    name: str
    source: str  # "kaggle" | "http" | "bimcv"
    description: str
    license_note: str
    citation: str
    # Source-specific:
    kaggle_dataset: str | None = None  # owner/dataset-slug for Kaggle
    http_url: str | None = None  # direct URL for "http"
    bimcv_url: str | None = None  # operator-supplied URL for "bimcv"
    # Verification:
    expected_files: list[str] = field(default_factory=list)
    sha256_files: dict[str, str] = field(default_factory=dict)


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "nih_chestxray14": DatasetSpec(
        name="nih_chestxray14",
        source="kaggle",
        description=(
            "NIH Clinical Center Chest X-Ray14 — 112,120 frontal-view X-rays "
            "from 30,805 patients with 14 disease labels."
        ),
        license_note=(
            "CC0 (public domain) per the dataset card. Cite Wang et al. 2017."
        ),
        citation=(
            "Wang X, Peng Y, Lu L, et al. ChestX-ray8: Hospital-scale Chest "
            "X-ray Database... CVPR 2017."
        ),
        kaggle_dataset="nih-chest-xrays/data",
        # The full archive is ~45 GB; expected_files lists the canonical labels file.
        expected_files=["Data_Entry_2017.csv"],
    ),
    "padchest": DatasetSpec(
        name="padchest",
        source="bimcv",
        description=(
            "PadChest — 160k chest radiographs from Hospital San Juan, "
            "Spain. Used as external validation for cross-institutional "
            "generalization studies."
        ),
        license_note=(
            "Free for academic use; requires BIMCV registration. "
            "Cite Bustos et al. 2020."
        ),
        citation=(
            "Bustos A, Pertusa A, Salinas J-M, de la Iglesia-Vayá M. "
            "PadChest: A large chest x-ray image dataset with multi-label "
            "annotated reports. Medical Image Analysis 2020."
        ),
        bimcv_url=None,  # operator sets via fetch_dataset(..., bimcv_url=...)
        expected_files=["padchest_meta.csv"],
    ),
}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(spec: DatasetSpec, dest_dir: Path) -> dict[str, str]:
    """Return a {file: status} map. status in {ok, missing, sha_mismatch, no_check}."""
    out: dict[str, str] = {}
    for fname in spec.expected_files:
        path = dest_dir / fname
        if not path.exists():
            out[fname] = "missing"
            continue
        if fname in spec.sha256_files:
            actual = _sha256_file(path)
            out[fname] = "ok" if actual == spec.sha256_files[fname] else "sha_mismatch"
        else:
            out[fname] = "no_check"
    return out


def is_present(spec: DatasetSpec, dest_dir: Path) -> bool:
    """True if all expected files exist (and pass checksum if configured)."""
    statuses = verify(spec, dest_dir)
    if not statuses:
        return False
    return all(s in {"ok", "no_check"} for s in statuses.values())


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

class FetchSkippedError(RuntimeError):
    """Raised when the operator hasn't configured the credentials this fetcher needs."""


# Backwards-compatible alias (the original name reads more naturally at call sites).
FetchSkipped = FetchSkippedError


def _fetch_kaggle(spec: DatasetSpec, dest_dir: Path) -> Path:
    if not spec.kaggle_dataset:
        raise ValueError(f"dataset {spec.name} has no kaggle_dataset slug")
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        raise FetchSkippedError(
            "~/.kaggle/kaggle.json missing; create it with "
            "{\"username\":..., \"key\":...} per Kaggle API docs"
        )
    # Lazy import: Kaggle SDK pulls a lot. Also it authenticates on import.
    import kaggle  # noqa: F401  (triggers auth)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    log.info("datasets.kaggle.fetch.start", dataset=spec.kaggle_dataset, dest=str(dest_dir))
    dest_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(spec.kaggle_dataset, path=str(dest_dir), unzip=True, quiet=False)
    log.info("datasets.kaggle.fetch.done", dataset=spec.kaggle_dataset)
    return dest_dir


def _safe_download_name(target: str, fallback_stem: str) -> str:
    """Derive a safe single-segment filename from a URL.

    The last path segment of an arbitrary URL is untrusted: a URL ending in
    ``/..`` or containing path separators must not be written into a parent of
    ``dest_dir``. We strip query/fragment, take the final segment, and reduce it
    to a bare name component, falling back to ``<stem>.bin`` for empty/``.``/``..``.
    """
    raw = target.split("?", 1)[0].split("#", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    candidate = PurePosixPath(raw).name
    if not candidate or candidate in (".", ".."):
        return f"{fallback_stem}.bin"
    return candidate


def _stream_to_file(target: str, out: Path, *, headers: dict[str, str] | None = None, timeout: float) -> None:
    """Stream ``target`` to ``out`` atomically: download to ``.part`` then rename.

    An interruption mid-download leaves only the ``.part`` temp file, never a
    truncated file at the final path (which ``is_present`` would later accept as
    a complete, valid download).
    """
    tmp = out.with_name(out.name + ".part")
    try:
        with httpx.stream("GET", target, headers=headers, follow_redirects=True, timeout=timeout) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _fetch_http(spec: DatasetSpec, dest_dir: Path, *, url: str | None = None) -> Path:
    target = url or spec.http_url
    if not target:
        raise ValueError(f"dataset {spec.name} has no http_url")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / _safe_download_name(target, spec.name)
    log.info("datasets.http.fetch.start", url=target, dest=str(out))
    _stream_to_file(target, out, timeout=300.0)
    log.info("datasets.http.fetch.done", path=str(out), bytes=out.stat().st_size)
    return dest_dir


def _fetch_bimcv(spec: DatasetSpec, dest_dir: Path, *, url: str | None = None) -> Path:
    """Stub for BIMCV-hosted datasets (PadChest etc.).

    BIMCV serves data over authenticated HTTPS. Operator supplies the URL
    and an env var ``BIMCV_TOKEN`` (Bearer token). We treat this as a
    plain HTTP GET with auth header. Per-dataset URL discovery is left to
    the operator after BIMCV registration.
    """
    target = url or spec.bimcv_url
    if not target:
        raise FetchSkippedError(
            f"{spec.name}: BIMCV URL not configured. Pass bimcv_url= or set "
            f"DATASET_REGISTRY[{spec.name!r}].bimcv_url after BIMCV registration."
        )
    token = os.environ.get("BIMCV_TOKEN")
    if not token:
        raise FetchSkippedError(f"{spec.name}: BIMCV_TOKEN env var not set")

    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / _safe_download_name(target, spec.name)
    headers = {"Authorization": f"Bearer {token}"}
    log.info("datasets.bimcv.fetch.start", url=target, dest=str(out))
    _stream_to_file(target, out, headers=headers, timeout=600.0)
    log.info("datasets.bimcv.fetch.done", path=str(out))
    return dest_dir


def fetch_dataset(
    name: str,
    *,
    dest_dir: Path | str,
    bimcv_url: str | None = None,
    force: bool = False,
) -> Path:
    """Fetch ``name`` into ``dest_dir`` if not already present.

    Idempotent: if ``is_present(spec, dest_dir)`` and not ``force``, returns
    immediately without touching the network.

    Raises ``FetchSkipped`` if credentials are missing for the configured
    source — caller decides whether to skip the smoke or fail loudly.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset: {name}; known: {list(DATASET_REGISTRY)}")
    spec = DATASET_REGISTRY[name]
    dest = Path(dest_dir)

    if not force and is_present(spec, dest):
        log.info("datasets.fetch.cached", name=name, dest=str(dest))
        return dest

    if spec.source == "kaggle":
        return _fetch_kaggle(spec, dest)
    if spec.source == "http":
        return _fetch_http(spec, dest)
    if spec.source == "bimcv":
        return _fetch_bimcv(spec, dest, url=bimcv_url)
    raise ValueError(f"unknown source for dataset {name}: {spec.source}")
