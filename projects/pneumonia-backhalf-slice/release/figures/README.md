# figures/ — Paper figure manifest

Figures are generated from the pre-computed results in `runs/` by
`scripts/run_experiments.py` (or a dedicated plotting script once added).
Binary PNG files are not stored in this repository; run the pipeline to
produce them.

## Figure inventory

| Figure file | Caption summary | Producing script / data source |
|-------------|-----------------|-------------------------------|
| `fig1_e0_baseline_auroc.png` | E0 baseline AUROC across 3 seeds vs. CheXNet reference (0.768). Bar chart with individual seed points. | `runs/E0_summary.json` |
| `fig2_e1_indomain_auroc_vs_n.png` | In-domain (NIH test) AUROC as a function of N for matched and unmatched conditions. Shows saturation at N=25k. | `runs/E1_results.json` |
| `fig3_e1_external_auroc_vs_n.png` | External (PadChest) AUROC as a function of N. Shows near-chance performance (~0.43–0.45) across all N. | `runs/E1_results.json` |
| `fig4_e1_generalization_gap.png` | Generalization gap (NIH AUROC − PadChest AUROC) vs. N, matched condition. | `runs/E1_results.json` |
| `fig5_e2_prevalence_control.png` | E2 prevalence-controlled sweep: in-domain and external AUROC vs. N_base with fixed positive count. | `runs/E2_results.json` |

## Notes

- All figures use bootstrap 95% CIs (100 resamples) as error bars.
  See paper Limitations for discussion of CI stability at 100 resamples.
- The CheXNet reference line in fig1 (AUROC = 0.768) is from Rajpurkar et al.
  2017 (DenseNet-121, different split) and is shown for orientation only;
  it is not a directly comparable baseline.
- External AUROC values below 0.5 in fig3 indicate systematic inversion of
  learned features under domain shift, not merely random performance.
  See paper Limitations §Domain shift interpretation.
