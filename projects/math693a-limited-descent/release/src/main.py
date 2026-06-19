"""CLI entrypoint for the limited-descent experiments.

Implements plan steps E1..E5 plus an 'all' option that runs E1->E5 in order.
E1 behaviour is unchanged from the original implementation.
"""
import argparse
import json
import sys

from src.config import GRID_N, START_SEED
from src.experiment_e1 import run_e1
from src.experiment_e2 import run_e2
from src.experiment_e3 import run_e3
from src.experiment_e4 import run_e4
from src.experiment_e5 import run_e5

CHOICES = ["E1", "E2", "E3", "E4", "E5", "all"]


def build_parser():
    p = argparse.ArgumentParser(description="Limited-descent experiments")
    p.add_argument("--experiment", default="E1", choices=CHOICES,
                   help="which experiment to run")
    p.add_argument("--run-id", default=None, help="output run id under runs/")
    p.add_argument("--seed", type=int, default=START_SEED)
    p.add_argument("--grid-n", type=int, default=GRID_N,
                   help="grid points per axis (smaller => faster)")
    return p


def _run_one(exp, run_id, seed, grid_n):
    """Dispatch a single experiment, returning its summary dict."""
    if exp == "E1":
        return run_e1(run_id=run_id, seed=seed, grid_n=grid_n)
    if exp == "E2":
        return run_e2(run_id=run_id, seed=seed, grid_n=grid_n)
    if exp == "E3":
        return run_e3(run_id=run_id, seed=seed, grid_n=grid_n)
    if exp == "E4":
        return run_e4(run_id=run_id, seed=seed, grid_n=grid_n)
    if exp == "E5":
        return run_e5(run_id=run_id, seed=seed, grid_n=grid_n)
    raise ValueError(f"unknown experiment {exp}")


def _report(exp, run_id, summary):
    rec = {
        "experiment": exp,
        "run_id": run_id,
        "n_trials": summary.get("n_trials"),
    }
    if exp == "E1":
        rec["internal_validity_passed"] = summary["internal_validity_passed"]
        rec["n_validity_failures"] = summary["n_validity_failures"]
    return rec


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.experiment == "all":
        exps = ["E1", "E2", "E3", "E4", "E5"]
        reports = []
        exit_code = 0
        for exp in exps:
            run_id = (args.run_id + "_" + exp.lower()) if args.run_id else exp.lower()
            summary = _run_one(exp, run_id, args.seed, args.grid_n)
            reports.append(_report(exp, run_id, summary))
        print(json.dumps({"experiment": "all", "reports": reports}, indent=2))
        return exit_code

    run_id = args.run_id or args.experiment.lower()
    summary = _run_one(args.experiment, run_id, args.seed, args.grid_n)
    print(json.dumps(_report(args.experiment, run_id, summary), indent=2))

    if args.experiment == "E1":
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
