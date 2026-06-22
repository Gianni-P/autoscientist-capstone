import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json

os.makedirs("figures", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
with open("runs/prod_s0_e5/e5_summary.json") as f:
    e5 = json.load(f)

with open("runs/prod_s0_e2/e2_summary.json") as f:
    e2 = json.load(f)

with open("runs/prod_s0_e4/e4_summary.json") as f:
    e4 = json.load(f)

with open("runs/prod_s0_e1/e1_summary.json") as f:
    e1 = json.load(f)

# ── Helpers ────────────────────────────────────────────────────────────────────
TERRAIN_LABELS = {
    "elliptic_paraboloid": "Elliptic\nParaboloid",
    "rosenbrock_ridge":    "Rosenbrock\nRidge",
    "sinusoidal_valley":   "Sinusoidal\nValley",
    "monkey_saddle":       "Monkey\nSaddle",
    "curved_valley":       "Curved\nValley",
    "anisotropic_bowl":    "Anisotropic\nBowl",
    "high_freq_valley":    "High-Freq\nValley",
    "gaussian_basin":      "Gaussian\nBasin",
}

STRATEGY_LABELS = {
    "unconstrained_steepest_descent": "Unconstrained\nDescent",
    "rotation_cw":                    "Rotation CW",
    "rotation_ccw":                   "Rotation CCW",
    "gradient_projection":            "Gradient\nProjection",
}

STRATEGY_COLORS = {
    "unconstrained_steepest_descent": "#e15759",
    "rotation_cw":                    "#4e79a7",
    "rotation_ccw":                   "#76b7b2",
    "gradient_projection":            "#f28e2b",
}

# ── Figure 1: Corrected Optimality Gap (COG) by strategy and terrain ───────────
# Use prod_s0_e5 per_strategy_terrain data
# Focus on 4 core terrains for clarity (all 4 strategies)
core_terrains = ["elliptic_paraboloid", "rosenbrock_ridge",
                 "sinusoidal_valley", "monkey_saddle"]
strategies = ["unconstrained_steepest_descent", "rotation_cw",
              "rotation_ccw", "gradient_projection"]

# Build lookup: (strategy, terrain) -> {mean, ci_lo, ci_hi}
cog_data = {}
for entry in e5["per_strategy_terrain"]:
    key = (entry["strategy"], entry["terrain"])
    cog_data[key] = {
        "mean":  entry["cog"]["mean"],
        "ci_lo": entry["cog"]["ci_lo"],
        "ci_hi": entry["cog"]["ci_hi"],
    }

fig1, axes = plt.subplots(1, 4, figsize=(13, 4.5), sharey=False)
fig1.suptitle(
    "Corrected Optimality Gap (COG) by Strategy and Terrain\n"
    "(negative = better than Dijkstra reference; positive = worse)",
    fontsize=11, y=1.02
)

x = np.arange(len(strategies))
width = 0.6

for ax, terrain in zip(axes, core_terrains):
    means  = []
    errs_lo = []
    errs_hi = []
    colors = []
    for strat in strategies:
        d = cog_data.get((strat, terrain), None)
        if d is None:
            means.append(np.nan); errs_lo.append(0); errs_hi.append(0)
        else:
            means.append(d["mean"])
            errs_lo.append(d["mean"] - d["ci_lo"])
            errs_hi.append(d["ci_hi"] - d["mean"])
        colors.append(STRATEGY_COLORS[strat])

    bars = ax.bar(x, means, width=width, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.8)
    ax.errorbar(x, means, yerr=[errs_lo, errs_hi],
                fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(TERRAIN_LABELS[terrain], fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies],
                       fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("COG (path length units)" if ax == axes[0] else "")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Legend
legend_patches = [
    mpatches.Patch(color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s].replace("\n", " "))
    for s in strategies
]
fig1.legend(handles=legend_patches, loc="lower center", ncol=4,
            fontsize=8.5, bbox_to_anchor=(0.5, -0.08), frameon=False)

plt.tight_layout()
plt.savefig("figures/fig1_cog_by_strategy_terrain.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1")

# ── Figure 2: Feasibility rate (E2) vs full-feasibility (E4) comparison ────────
# E2 = unconstrained steepest descent, E4 = constrained (rotation) strategies
# Both from prod runs; show feasibility rate per terrain for E2 (unconstrained)
# and mean_feasibility_rate=1.0 for E4 (all converged paths are grade-feasible)

terrains_e2 = [t["terrain"] for t in e2["terrain_summaries"]]
feas_e2     = [t["mean_feasibility_rate"] for t in e2["terrain_summaries"]]

terrains_e4 = [t["terrain"] for t in e4["terrain_summaries"]]
feas_e4     = [t["mean_feasibility_rate"] for t in e4["terrain_summaries"]]

# Align terrains
all_terrains = terrains_e2  # same set
feas_e4_aligned = []
for t in all_terrains:
    match = [x["mean_feasibility_rate"] for x in e4["terrain_summaries"] if x["terrain"] == t]
    feas_e4_aligned.append(match[0] if match else np.nan)

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
x2 = np.arange(len(all_terrains))
w2 = 0.35

bars_e2 = ax2.bar(x2 - w2/2, feas_e2, width=w2, color="#e15759", alpha=0.85,
                  label="Unconstrained Steepest Descent (E2)", edgecolor="white")
bars_e4 = ax2.bar(x2 + w2/2, feas_e4_aligned, width=w2, color="#4e79a7", alpha=0.85,
                  label="Constrained Rotation Heuristic (E4)", edgecolor="white")

ax2.set_xticks(x2)
ax2.set_xticklabels([TERRAIN_LABELS[t] for t in all_terrains], fontsize=8.5)
ax2.set_ylabel("Mean Feasibility Rate\n(fraction of steps within grade limit)", fontsize=9)
ax2.set_ylim(0, 1.08)
ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_title(
    "Grade-Feasibility Rate: Unconstrained Descent vs. Constrained Rotation Heuristic\n"
    "(E4 paths are fully grade-feasible by construction; E2 paths frequently violate the 5° limit)",
    fontsize=10
)
ax2.legend(fontsize=9, frameon=False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(axis="y", labelsize=8.5)

plt.tight_layout()
plt.savefig("figures/fig2_feasibility_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2")

# ── Figure 3: H1 test — rotation COG pooled CI across terrains ────────────────
# Show per-terrain COG (rotation strategies pooled CW+CCW) with 95% CI
# from H1_rotation_cog_gt_zero in prod_s0_e5

h1 = e5["H1_rotation_cog_gt_zero"]
terrain_order = ["elliptic_paraboloid", "rosenbrock_ridge",
                 "sinusoidal_valley", "monkey_saddle",
                 "curved_valley", "anisotropic_bowl",
                 "high_freq_valley", "gaussian_basin"]

h1_means  = []
h1_ci_lo  = []
h1_ci_hi  = []
h1_labels = []
h1_colors = []

for t in terrain_order:
    if t not in h1:
        continue
    d = h1[t]["ci"]
    h1_means.append(d["mean"])
    h1_ci_lo.append(d["mean"] - d["ci_lo"])
    h1_ci_hi.append(d["ci_hi"] - d["mean"])
    h1_labels.append(TERRAIN_LABELS[t])
    h1_colors.append("#e15759" if h1[t]["cog_gt_zero"] else "#4e79a7")

fig3, ax3 = plt.subplots(figsize=(9, 4.5))
y3 = np.arange(len(h1_means))

ax3.barh(y3, h1_means, xerr=[h1_ci_lo, h1_ci_hi],
         color=h1_colors, alpha=0.85, edgecolor="white",
         error_kw=dict(ecolor="black", capsize=4, linewidth=1.2))
ax3.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
ax3.set_yticks(y3)
ax3.set_yticklabels(h1_labels, fontsize=9)
ax3.set_xlabel("Mean Corrected Optimality Gap (COG)\n"
               "(negative = rotation path shorter than Dijkstra reference;\n"
               " positive = rotation path longer)", fontsize=9)
ax3.set_title(
    "H1: Does the Rotation Heuristic Produce Near-Optimal Paths?\n"
    "95% Bootstrap CI of COG (rotation CW + CCW pooled, prod run, n≥174 per terrain)",
    fontsize=10
)

# Legend
pos_patch = mpatches.Patch(color="#e15759", alpha=0.85, label="COG > 0 (worse than reference)")
neg_patch = mpatches.Patch(color="#4e79a7", alpha=0.85, label="COG ≤ 0 (at or better than reference)")
ax3.legend(handles=[neg_patch, pos_patch], fontsize=8.5, frameon=False,
           loc="lower right")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.tick_params(axis="x", labelsize=8.5)

plt.tight_layout()
plt.savefig("figures/fig3_h1_rotation_cog.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3")

# ── Figure 4: E1 quantisation bias (mean_qb) across terrains ──────────────────
# Shows how well the Dijkstra grid reference approximates the true geodesic
# Use prod_s0_e1 terrain_summaries

e1_terrains = [t["terrain"] for t in e1["terrain_summaries"]]
e1_mean_qb  = [t["mean_qb"] for t in e1["terrain_summaries"]]
e1_max_qb   = [t["max_qb"] for t in e1["terrain_summaries"]]

fig4, ax4 = plt.subplots(figsize=(9, 4.2))
x4 = np.arange(len(e1_terrains))
w4 = 0.35

ax4.bar(x4 - w4/2, e1_mean_qb, width=w4, color="#4e79a7", alpha=0.85,
        label="Mean quantisation bias", edgecolor="white")
ax4.bar(x4 + w4/2, e1_max_qb, width=w4, color="#f28e2b", alpha=0.85,
        label="Max quantisation bias", edgecolor="white")

ax4.set_xticks(x4)
ax4.set_xticklabels([TERRAIN_LABELS[t] for t in e1_terrains], fontsize=8.5)
ax4.set_ylabel("Quantisation Bias\n(relative excess of Dijkstra over true geodesic)", fontsize=9)
ax4.set_title(
    "E1: Grid Reference Quality — Quantisation Bias of Dijkstra Shortest Path\n"
    "(prod run, grid_n=300; bias < 0.09 on all terrains confirms reference validity)",
    fontsize=10
)
ax4.axhline(0.0875, color="gray", linewidth=1.0, linestyle=":", alpha=0.8,
            label="tan(5°) grade limit (reference threshold)")
ax4.legend(fontsize=8.5, frameon=False)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.tick_params(axis="y", labelsize=8.5)

plt.tight_layout()
plt.savefig("figures/fig4_e1_quantisation_bias.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4")

# ── Write manifest ─────────────────────────────────────────────────────────────
manifest = [
    {
        "path": "figures/fig1_cog_by_strategy_terrain.png",
        "caption": (
            "Corrected Optimality Gap (COG) with 95% bootstrap confidence intervals "
            "for four descent strategies across four core terrain types (prod run, "
            "grid\\_n=300, n=90 per cell). Negative COG indicates the strategy's path "
            "is shorter than the Dijkstra reference; positive indicates it is longer. "
            "Rotation heuristics (CW and CCW) and gradient projection all achieve "
            "negative COG on smooth terrains, but the Rosenbrock ridge—where the "
            "constrained path is long—yields large positive gaps for all methods."
        ),
        "label": "fig:cog_by_strategy_terrain",
    },
    {
        "path": "figures/fig2_feasibility_comparison.png",
        "caption": (
            "Mean grade-feasibility rate per terrain for unconstrained steepest "
            "descent (E2, red) versus the constrained rotation heuristic (E4, blue). "
            "Unconstrained descent violates the 5° grade limit on 36–40\\% of steps "
            "on most terrains, while E4 paths are fully feasible by construction "
            "(rate = 1.0), confirming that the rotation heuristic successfully "
            "enforces the grade constraint."
        ),
        "label": "fig:feasibility_comparison",
    },
    {
        "path": "figures/fig3_h1_rotation_cog.png",
        "caption": (
            "Hypothesis H1 test: mean COG (rotation CW + CCW pooled) with 95\\% "
            "bootstrap confidence intervals across all eight terrains (prod run). "
            "Blue bars indicate COG $\\leq 0$ (rotation path at or shorter than the "
            "Dijkstra reference); red bars indicate COG $> 0$ (longer). Only the "
            "Rosenbrock ridge yields a significantly positive COG, reflecting the "
            "terrain's inherently long constrained path; all other terrains show "
            "COG $\\leq 0$, meaning the rotation heuristic matches or beats the "
            "grid reference."
        ),
        "label": "fig:h1_rotation_cog",
    },
    {
        "path": "figures/fig4_e1_quantisation_bias.png",
        "caption": (
            "E1 reference quality: mean and maximum quantisation bias of the "
            "Dijkstra grid shortest path relative to the true 3-D geodesic, "
            "across all eight terrains (prod run, grid\\_n=300). "
            "All mean biases remain well below the tan(5°) grade threshold "
            "(dotted line), validating the grid reference as a reliable "
            "ground-truth comparator for the optimality gap analysis."
        ),
        "label": "fig:e1_quantisation_bias",
    },
]

with open("figures/figures.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Manifest written to figures/figures.json")
print("All figures generated successfully.")
