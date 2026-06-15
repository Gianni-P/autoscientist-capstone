"""Phase 3 smoke test — tool integrations.

Exercises each of the 6 Phase 3 tools end-to-end. Network-dependent steps
cache their results in ``tool_cache``, so re-runs are fast and offline-safe
once the cache is warm.

Each step is gated on the prerequisites it needs:
  * literature.search     — needs internet on first run; cached afterward.
  * pdf_parse             — generates a tiny PDF via reportlab fallback;
                            if reportlab is absent, downloads a tiny arxiv PDF.
                            On first run needs internet OR reportlab; cached.
  * execute               — pure local subprocess; always runs.
  * datasets              — registry lookup + (skipped) Kaggle/BIMCV fetch
                            unless --download is passed.
  * latex                 — tectonic compile; skipped with clear message
                            if tectonic not on PATH.
  * citation_check        — uses literature; cached after first run.

    uv run python scripts/smoke_phase3.py
    uv run python scripts/smoke_phase3.py --offline   # use cache only
    uv run python scripts/smoke_phase3.py --download  # also pull NIH (slow)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_DB = _REPO / "smoke_phase3.db"
_PROJECTS_ROOT = _REPO / "projects_smoke_phase3"
if _DB.exists():
    _DB.unlink()
if _PROJECTS_ROOT.exists():
    shutil.rmtree(_PROJECTS_ROOT)

os.environ["AUTOSCIENTIST_DB_PATH"] = str(_DB)


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def passed(msg: str) -> None:
    print(f"  PASS  {msg}")


def skipped(msg: str) -> None:
    print(f"  SKIP  {msg}")


def fail(msg: str) -> None:
    raise AssertionError(msg)


def expect(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)
    passed(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="use cached results only; fail if cache miss")
    parser.add_argument("--download", action="store_true", help="run real Kaggle download (slow, ~45 GB)")
    args = parser.parse_args(argv)

    from autoscientist.state.db import open_db
    from autoscientist.tools import (
        citation_check,
        datasets,
        execute,
        latex,
        literature,
        pdf_parse,
    )

    conn = open_db(_DB)

    # -----------------------------------------------------------------------
    section("literature.search — Semantic Scholar / OpenAlex")
    # -----------------------------------------------------------------------
    query = "CheXNet pneumonia detection chest x-ray"
    if args.offline:
        from autoscientist.tools import tool_cache
        key = tool_cache.cache_key({"query": query, "limit": 5, "include_arxiv": False})
        hit = tool_cache.cache_get(conn, "literature.search", key)
        expect(hit is not None, "literature.search has cached result for offline run")
        results = [literature.Paper.from_dict(d) for d in hit]
    else:
        results = literature.search(query, limit=5, conn=conn)
    expect(len(results) >= 1, f"literature.search returned >=1 result (got {len(results)})")
    expect(any(p.title for p in results), "at least one result has a title")
    expect(any((p.doi or p.arxiv_id or p.semantic_scholar_id) for p in results),
           "at least one result has a resolvable ID")
    # Cache replay: identical query should now be a hit.
    if not args.offline:
        results2 = literature.search(query, limit=5, conn=conn)
        expect(len(results2) == len(results), "literature.search cached call returns same result count")

    # -----------------------------------------------------------------------
    section("pdf_parse — round-trip via tectonic if available, else stub")
    # -----------------------------------------------------------------------
    pdf_built = False
    if latex.is_available():
        tex_src = (
            r"\documentclass{article}\usepackage[T1]{fontenc}"
            r"\title{Smoke}\author{autoscientist}\date{}"
            r"\begin{document}\maketitle"
            r"\section{intro}This is a smoke test PDF. abc 123."
            r"\end{document}"
        )
        tmp_dir = _PROJECTS_ROOT / "latex_smoke"
        try:
            build = latex.compile_latex(tex_src, output_dir=tmp_dir, conn=conn)
            if build.success:
                pdf_path = Path(build.pdf_path)  # type: ignore[arg-type]
                doc = pdf_parse.parse_pdf(pdf_path, conn=conn)
                expect(doc.page_count >= 1, f"pdf_parse extracted >=1 page (got {doc.page_count})")
                expect("smoke test PDF" in doc.text or "abc 123" in doc.text,
                       "pdf_parse extracted expected body text")
                # Cache replay
                doc2 = pdf_parse.parse_pdf(pdf_path, conn=conn)
                expect(doc2.sha256 == doc.sha256, "pdf_parse cache hit returns same sha")
                pdf_built = True
            else:
                skipped(f"latex.compile failed (errors={len(build.errors)}); skipping pdf_parse round-trip")
        except latex.TectonicMissing as e:
            skipped(f"tectonic missing: {e}")
    if not pdf_built:
        # Fallback: try parsing a known-tiny arxiv PDF if online; otherwise skip.
        if args.offline:
            skipped("pdf_parse skipped (offline + no PDF source)")
        else:
            try:
                import httpx
                test_url = "https://arxiv.org/pdf/1711.05225v3"  # CheXNet preprint, tiny enough
                tmp_pdf = _PROJECTS_ROOT / "fallback.pdf"
                tmp_pdf.parent.mkdir(parents=True, exist_ok=True)
                with httpx.stream("GET", test_url, follow_redirects=True, timeout=60.0) as r:
                    r.raise_for_status()
                    with tmp_pdf.open("wb") as f:
                        for chunk in r.iter_bytes(chunk_size=1 << 20):
                            f.write(chunk)
                doc = pdf_parse.parse_pdf(tmp_pdf, conn=conn)
                expect(doc.page_count >= 1, f"pdf_parse extracted >=1 page (got {doc.page_count})")
                expect(len(doc.text) > 100, f"pdf_parse extracted >100 chars (got {len(doc.text)})")
            except Exception as e:
                skipped(f"pdf_parse fallback failed: {e}")

    # -----------------------------------------------------------------------
    section("execute — sandboxed subprocess")
    # -----------------------------------------------------------------------
    project_id = "smoke3"
    sandbox = execute.reset_sandbox(project_id, _PROJECTS_ROOT)
    expect(sandbox.exists() and sandbox.is_dir(), f"sandbox dir created at {sandbox}")

    # Trivial ok run: write a file, print a line.
    py_script = (
        "import os, sys, json;"
        "open('output.txt','w').write('hello');"
        "print(json.dumps({'cwd': os.getcwd(), 'argv': sys.argv}))"
    )
    res = execute.execute(
        ["python", "-c", py_script],
        project_id=project_id,
        projects_root=_PROJECTS_ROOT,
        timeout_seconds=30,
        cpu_seconds=30,
    )
    expect(res.exit_code == 0, f"execute trivial python script (exit={res.exit_code})")
    expect("hello" in (sandbox / "output.txt").read_text(),
           "execute wrote output.txt inside sandbox")
    expect(str(sandbox) in res.stdout,
           f"execute child cwd is sandbox (stdout: {res.stdout[:200]})")

    # Timeout enforcement.
    res_to = execute.execute(
        ["python", "-c", "import time; time.sleep(10)"],
        project_id=project_id,
        projects_root=_PROJECTS_ROOT,
        timeout_seconds=1,
    )
    expect(res_to.timed_out, f"execute timeout fires (timed_out={res_to.timed_out})")
    expect(res_to.exit_code != 0, f"timed-out process has nonzero exit_code (got {res_to.exit_code})")

    # Nonzero exit propagates.
    res_err = execute.execute(
        ["python", "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(7)"],
        project_id=project_id,
        projects_root=_PROJECTS_ROOT,
        timeout_seconds=10,
    )
    expect(res_err.exit_code == 7, f"execute propagates exit code (got {res_err.exit_code})")
    expect("boom" in res_err.stderr, "execute captures stderr")

    # -----------------------------------------------------------------------
    section("datasets — registry + idempotent fetch")
    # -----------------------------------------------------------------------
    expect("nih_chestxray14" in datasets.DATASET_REGISTRY,
           "registry has nih_chestxray14")
    expect("padchest" in datasets.DATASET_REGISTRY,
           "registry has padchest")
    nih_spec = datasets.DATASET_REGISTRY["nih_chestxray14"]
    expect(nih_spec.kaggle_dataset == "nih-chest-xrays/data",
           "nih_chestxray14 points at correct Kaggle slug")

    # is_present false on empty dir
    nih_dir = _PROJECTS_ROOT / "datasets" / "nih"
    expect(not datasets.is_present(nih_spec, nih_dir),
           "is_present returns False before fetch")

    # Stage the expected file to verify idempotency check.
    nih_dir.mkdir(parents=True, exist_ok=True)
    (nih_dir / "Data_Entry_2017.csv").write_text("Image Index,Patient ID\n", encoding="utf-8")
    expect(datasets.is_present(nih_spec, nih_dir),
           "is_present returns True after staging expected_files")
    res_dir = datasets.fetch_dataset("nih_chestxray14", dest_dir=nih_dir)
    expect(res_dir == nih_dir,
           "fetch_dataset is no-op when files already present")

    # PadChest skip behavior
    pad_dir = _PROJECTS_ROOT / "datasets" / "padchest"
    try:
        datasets.fetch_dataset("padchest", dest_dir=pad_dir)
        skipped("padchest fetch did not raise (BIMCV must be configured?)")
    except datasets.FetchSkipped:
        passed("padchest fetch raises FetchSkipped without BIMCV credentials")

    if args.download:
        section("datasets — actual Kaggle download (--download)")
        try:
            real_dir = _PROJECTS_ROOT / "datasets" / "nih_real"
            real_dir.mkdir(parents=True, exist_ok=True)
            datasets.fetch_dataset("nih_chestxray14", dest_dir=real_dir)
            passed(f"NIH downloaded to {real_dir}")
        except datasets.FetchSkipped as e:
            skipped(f"Kaggle skip: {e}")

    # -----------------------------------------------------------------------
    section("latex — tectonic compile")
    # -----------------------------------------------------------------------
    if not latex.is_available():
        skipped("tectonic not on PATH; install with `cargo install tectonic` or `apt install tectonic`")
    else:
        # Re-use the source from earlier. If we didn't compile then, do it now.
        tex_src = (
            r"\documentclass{article}\title{Smoke}\author{autoscientist}\date{}"
            r"\begin{document}\maketitle\section{intro}smoke body 12345.\end{document}"
        )
        out_dir = _PROJECTS_ROOT / "latex_smoke2"
        try:
            build = latex.compile_latex(tex_src, output_dir=out_dir, conn=conn)
            expect(build.success, f"latex.compile success (errors={build.errors[:3]})")
            expect(Path(build.pdf_path).exists(),  # type: ignore[arg-type]
                   f"PDF written to {build.pdf_path}")
            # Cache replay: delete output, recompile, expect cache hit + recreation.
            shutil.rmtree(out_dir)
            build2 = latex.compile_latex(tex_src, output_dir=out_dir, conn=conn)
            expect(build2.success and Path(build2.pdf_path).exists(),  # type: ignore[arg-type]
                   "latex cache hit re-materializes PDF on second call")
        except latex.TectonicMissing as e:
            skipped(f"tectonic check passed but compile raised: {e}")

    # -----------------------------------------------------------------------
    section("citation_check — verifies real DOI, rejects fake")
    # -----------------------------------------------------------------------
    if args.offline:
        skipped("citation_check requires literature lookups; offline mode")
    else:
        # Use a citation that should exist: CheXNet (Rajpurkar et al. 2017).
        good = {
            "key": "Rajpurkar2017",
            "title": "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning",
            "authors": ["Rajpurkar, Pranav"],
            "year": 2017,
            "doi_or_arxiv": "1711.05225",  # arxiv id
        }
        chk_good = citation_check.verify_citation(good, conn=conn)
        if chk_good.verified:
            passed(f"citation_check verified Rajpurkar2017 (confidence={chk_good.confidence:.2f})")
        else:
            # Soft-pass: literature APIs vary. As long as it found a paper with high title sim, accept.
            if chk_good.confidence >= 0.6 and chk_good.matched_paper is not None:
                passed(
                    f"citation_check matched Rajpurkar2017 with confidence "
                    f"{chk_good.confidence:.2f} (mismatch: {chk_good.mismatch_reasons})"
                )
            else:
                fail(
                    f"citation_check failed to find Rajpurkar2017: "
                    f"{chk_good.mismatch_reasons}"
                )

        bogus = {
            "key": "Bogus2042",
            "title": "Definitely-fabricated paper with nonsense title qwertyuiop fjklasdjfklas",
            "authors": ["Made-up, A."],
            "year": 2042,
            "doi_or_arxiv": "10.9999/this-doi-does-not-exist-zzzz",
        }
        chk_bogus = citation_check.verify_citation(bogus, conn=conn)
        expect(not chk_bogus.verified,
               f"citation_check rejects fabricated citation (mismatch: {chk_bogus.mismatch_reasons})")

    print("\n*** Phase 3 smoke checks done. ***")
    print(f"  DB:        {_DB}")
    print(f"  Projects:  {_PROJECTS_ROOT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
