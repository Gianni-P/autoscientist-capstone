"""E5 - aggregation + statistics and gap decomposition (plan experiment E5).

For every (strategy, terrain, start, seed) we compute, against the
Theta*-smoothed grade-constrained Dijkstra reference (the optimum from E1/graph):

    OG  = (strategy_length - theta_star_length) / theta_star_length
    QB  = (raw_dijkstra_length - theta_star_length) / theta_star_length
    COG = OG - QB                       (corrected optimality gap)

We then bootstrap 10000 resamples over the (terrain, start) pairs to obtain
95% CIs and run the planned hypothesis tests:

    H1: COG_rotation > 0 (one-sided bootstrap, alpha=0.05), per terrain and
        pooled across the non-trivial terrains (T2+T3+T4).
    H2: COG_projection < COG_rotation (paired bootstrap).
    H3: CW vs CCW asymmetry per terrain (Holm-Bonferroni across 4 terrains).

A consolidated summary json is written to runs/<run_id>/.

Public API
----------
N_BOOTSTRAP, ALPHA, NONTRIVIAL_TERRAINS
run_e5(run_id, seed, grid_n) -> dict
"""
import json
import os

import numpy as np

from src.config import GRID_N, MAX_GRADE_DEGREES, MAX_GRADE_TAN
from src.common import prepare_terrain, write_jsonl, ensure_run_dir, RUN_TERRAINS
from src.strategies import run_strategy, DS_DEFAULT, MAX_ITERS

N_BOOTSTRAP = 10000
ALPHA = 0.05
SEEDS = [0, 1, 2]
# Canonical ids for the non-trivial terrains (T2 == rosenbrock_ridge,
# T3 == sinusoidal_valley, T4 == monkey_saddle). MUST match the keys the
# terrains module exposes (RUN_TERRAINS = T1..T5), otherwise the H1 pooled loop
# never matches any terrain and the core hypothesis is structurally untestable.
NONTRIVIAL_TERRAINS = ["T2", "T3", "T4"]

# Strategies evaluated for the gap analysis.
_E5_STRATEGIES = [
    "unconstrained_steepest_descent",
    "rotation_cw",
    "rotation_ccw",
    "gradient_projection",
]


def _gaps_for_trial(setup, sp, res):
    theta = setup.theta_len[sp]
    raw = setup.raw_len[sp]
    if theta <= 0:
        return None
    length = res["path_length_3d"]
    og = (length - theta) / theta
    qb = (raw - theta) / theta
    cog = og - qb
    return og, qb, cog


def _bootstrap_mean_ci(values, rng, n=N_BOOTSTRAP):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "ci_lo_one_sided": float(np.percentile(means, 5.0)),
        "n": int(arr.size),
    }


def _paired_bootstrap_diff(a, b, rng, n=N_BOOTSTRAP):
    """Bootstrap CI of mean(a - b) over paired samples (a,b aligned)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    if diff.size == 0:
        return None
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    return {
        "mean_diff": float(diff.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "p_one_sided_lt0": float(np.mean(means >= 0.0)),  # P(diff>=0) for H:diff<0
        "n_pairs": int(diff.size),
    }


def _holm_bonferroni(pvals, alpha=ALPHA):
    """Holm-Bonferroni step-down correction.

    Returns a list (in the ORIGINAL p-value order) of dicts with keys
    {"p", "threshold", "reject"}. Step-down propagation: process p-values in
    ascending order; the first time a hypothesis fails to reject, every
    subsequent (larger) p-value is also non-rejected regardless of its own
    comparison to its threshold.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda k: pvals[k])
    results = [None] * m
    stop_rejecting = False
    for rank, k in enumerate(order):
        thresh = alpha / (m - rank)
        if stop_rejecting:
            reject = False
        else:
            reject = pvals[k] <= thresh
            if not reject:
                stop_rejecting = True
        results[k] = {"p": float(pvals[k]), "threshold": float(thresh),
                      "reject": bool(reject)}
    return results


def run_e5(run_id="e5", seed=0, grid_n=GRID_N):
    out_dir = ensure_run_dir(run_id)
    rng = np.random.default_rng(seed)

    trial_records = []
    # cog[strategy][terrain] -> {(seed,start): cog}; keyed for pairing
    cog = {s: {t: {} for t in RUN_TERRAINS} for s in _E5_STRATEGIES}
    og = {s: {t: {} for t in RUN_TERRAINS} for s in _E5_STRATEGIES}
    qb_by = {t: {} for t in RUN_TERRAINS}

    setups = {}

    def get_setup(name, sd):
        key = (name, sd)
        if key not in setups:
            setups[key] = prepare_terrain(name, grid_n, sd)
        return setups[key]

    for name in RUN_TERRAINS:
        for sd in SEEDS:
            setup = get_setup(name, sd)
            for sp in setup.starts:
                pair_key = (sd, sp[0], sp[1])
                qb_pair = (setup.raw_len[sp] - setup.theta_len[sp]) / setup.theta_len[sp] \
                    if setup.theta_len[sp] > 0 else 0.0
                qb_by[name][pair_key] = float(qb_pair)
                for strat in _E5_STRATEGIES:
                    res = run_strategy(setup.terrain, sp, strat, ds=DS_DEFAULT,
                                       max_iters=MAX_ITERS, seed=sd)
                    gaps = _gaps_for_trial(setup, sp, res)
                    if gaps is None:
                        continue
                    og_v, qb_v, cog_v = gaps
                    # Exclude non-converged trials from OG/COG (plan pitfall ack).
                    if not res["converged"]:
                        excluded = True
                    else:
                        excluded = False
                        cog[strat][name][pair_key] = cog_v
                        og[strat][name][pair_key] = og_v
                    trial_records.append({
                        "experiment": "E5",
                        "run_id": run_id,
                        "seed": sd,
                        "grid_n": grid_n,
                        "ds": DS_DEFAULT,
                        "max_grade_degrees": MAX_GRADE_DEGREES,
                        "max_grade_tan": MAX_GRADE_TAN,
                        "terrain": name,
                        "strategy": strat,
                        "start_ij": list(sp),
                        "converged": res["converged"],
                        "excluded_from_gap": excluded,
                        "path_length_3d": res["path_length_3d"],
                        "theta_star_length": setup.theta_len[sp],
                        "raw_dijkstra_length": setup.raw_len[sp],
                        "optimality_gap": og_v,
                        "quantisation_bias": qb_v,
                        "corrected_optimality_gap": cog_v,
                        "feasibility_rate": res["feasibility_rate"],
                        "iterations": res["iterations"],
                    })

    # --- per (strategy, terrain) OG/QB/COG means + CIs ---
    # The QB CI uses exactly the (seed,start) pairs that survived into the COG
    # set for this (strategy, terrain) cell, so COG = OG - QB holds at the
    # aggregate level and all three CIs share the same sample size. Because
    # qb_by[name] is populated for EVERY (seed,start) pair before any strategy
    # runs, every cog key is guaranteed present in qb_by[name]; a missing key
    # would be a logic error, so we look it up directly (no silent-drop guard).
    per_cell = []
    for strat in _E5_STRATEGIES:
        for name in RUN_TERRAINS:
            cog_keys = list(cog[strat][name].keys())
            cog_vals = [cog[strat][name][k] for k in cog_keys]
            og_vals = [og[strat][name][k] for k in cog_keys]
            qb_vals = [qb_by[name][k] for k in cog_keys]
            per_cell.append({
                "strategy": strat,
                "terrain": name,
                "og": _bootstrap_mean_ci(og_vals, rng),
                "qb": _bootstrap_mean_ci(qb_vals, rng),
                "cog": _bootstrap_mean_ci(cog_vals, rng),
            })

    # --- H1: COG_rotation > 0 (pool CW+CCW per terrain and pooled nontrivial) ---
    h1 = {}
    rotation_pooled_nontrivial = []
    for name in RUN_TERRAINS:
        vals = list(cog["rotation_cw"][name].values()) + \
            list(cog["rotation_ccw"][name].values())
        ci = _bootstrap_mean_ci(vals, rng)
        supported = bool(ci is not None and ci["ci_lo_one_sided"] > 0.0)
        h1[name] = {"ci": ci, "cog_gt_zero": supported}
        if name in NONTRIVIAL_TERRAINS:
            rotation_pooled_nontrivial.extend(vals)
    pooled_ci = _bootstrap_mean_ci(rotation_pooled_nontrivial, rng)
    h1["pooled_nontrivial"] = {
        "ci": pooled_ci,
        "cog_gt_zero": bool(pooled_ci is not None and pooled_ci["ci_lo_one_sided"] > 0.0),
        "terrains": NONTRIVIAL_TERRAINS,
    }

    # --- H2: COG_projection < COG_rotation (paired over shared pairs) ---
    proj_vals, rot_vals = [], []
    for name in RUN_TERRAINS:
        for key, pv in cog["gradient_projection"][name].items():
            rc = cog["rotation_cw"][name].get(key)
            cc = cog["rotation_ccw"][name].get(key)
            rv = [v for v in (rc, cc) if v is not None]
            if not rv:
                continue
            proj_vals.append(pv)
            rot_vals.append(float(np.mean(rv)))
    h2_diff = _paired_bootstrap_diff(proj_vals, rot_vals, rng)
    h2 = {
        "diff_proj_minus_rotation": h2_diff,
        "projection_lt_rotation": bool(
            h2_diff is not None and h2_diff["ci_hi"] < 0.0),
    }

    # --- H3: CW vs CCW asymmetry per terrain, Holm-Bonferroni across terrains ---
    h3_per_terrain = []
    h3_pvals = []
    for name in RUN_TERRAINS:
        keys = set(cog["rotation_cw"][name]) & set(cog["rotation_ccw"][name])
        cw = [cog["rotation_cw"][name][k] for k in keys]
        ccw = [cog["rotation_ccw"][name][k] for k in keys]
        if len(keys) == 0:
            diff = None
            p = 1.0
        else:
            diff = _paired_bootstrap_diff(cw, ccw, rng)
            # two-sided p: 2 * min(P(>=0), P(<=0))
            d = np.asarray(cw, dtype=np.float64) - np.asarray(ccw, dtype=np.float64)
            idx = rng.integers(0, d.size, size=(N_BOOTSTRAP, d.size))
            means = d[idx].mean(axis=1)
            p = float(2.0 * min(np.mean(means >= 0.0), np.mean(means <= 0.0)))
            p = min(1.0, p)
        h3_pvals.append(p)
        h3_per_terrain.append({"terrain": name, "diff_cw_minus_ccw": diff,
                               "p_two_sided": p, "n_pairs": len(keys)})
    holm = _holm_bonferroni(h3_pvals, alpha=ALPHA)
    for cell, hres in zip(h3_per_terrain, holm):
        cell["holm_threshold"] = hres["threshold"]
        cell["asymmetry_significant"] = hres["reject"]

    write_jsonl(os.path.join(out_dir, "e5_trials.jsonl"), trial_records)
    summary = {
        "experiment": "E5",
        "run_id": run_id,
        "seed": seed,
        "grid_n": grid_n,
        "n_bootstrap": N_BOOTSTRAP,
        "alpha": ALPHA,
        "seeds": SEEDS,
        "n_trials": len(trial_records),
        "per_strategy_terrain": per_cell,
        "H1_rotation_cog_gt_zero": h1,
        "H2_projection_lt_rotation": h2,
        "H3_cw_ccw_asymmetry": h3_per_terrain,
    }
    with open(os.path.join(out_dir, "e5_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
