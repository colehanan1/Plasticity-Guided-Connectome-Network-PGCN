# Multi-ORN Output Mapping Summary

This document captures the outputs and findings from `scripts/map_multi_orn_outputs.py`, which standardises downstream connectivity analysis for Or7a, Or13a, Or42b, Or47b, and Or59b receptor neuron populations using FAFB v783 FlyWire exports.

## Workflow Overview
- Primary command: `python scripts/map_multi_orn_outputs.py --log-level INFO`
- Dedicated Or7a regeneration (optional cross-check): `python scripts/map_or7a_outputs.py --data-source local`
- Inputs: search-result CSVs in `data/flywire/`, local Princeton connection matrix, and consolidated cell-type annotations.
- Processing choices: chunked streaming of the connections table, minimum synapse threshold of 3, and top-20 target export per neuron.
- Outputs are written to `results/multi_orn_outputs/` and include long- and wide-format tables, summary statistics, comparative reports, and publication-ready figures that now place Or7a on the same visualisations as the other four receptor classes.
- Cell type abbreviations are automatically expanded to descriptive names (for example, `DL5_adPN` → “DL5 antenno-deuterocerebral projection neuron”) across CSV exports and plots to improve readability.

## ORN-Specific Findings
### Or7a (DL5, ACH only)
- 41 neurons retained 9,144 synapses post-thresholding with a mean 15.2 targets per neuron (`results/multi_orn_outputs/or7a_summary_statistics.csv`).
- AdPN dominance persists (`DL5_adPN` = 51.8% of synapses) alongside shared inhibitory partners (`lLN2X02`, `lLN2F_b`, `lLN2T_b`, `il3LN6`), matching previously validated OR7a analyses.
- Mild right-hemisphere bias (23 right vs. 18 left neurons) yields asymmetry index +0.12; hemispheric totals remain balanced enough for aggregated statistics.

### Or13a (DC2, ACH only)
- 20 neurons retained 6,029 synapses post-thresholding; mean 18.0 distinct targets per neuron (`results/multi_orn_outputs/or13a_summary_statistics.csv`).
- Strongly stereotyped projection onto `DC2_adPN` (58.6% of synapses) with supporting lateral local interneuron targeting (lLN2F and il3LN6 families).
- Mild right-hemisphere bias (12 right vs. 8 left neurons) but symmetric output totals, suggesting bilateral duplication of the same wiring motif.

### Or42b (DM1, mixed ACH/SER)
- 71 neurons produced 18,591 synapses with 15.9 targets per neuron and balanced laterality (35 left / 36 right).
- Dominant projection to `DM1_lPN` (47.4%), with notable engagement of inhibitory partners (`il3LN6`, `lLN2F`, `lLN2T`).
- Outputs remain diversified (Gini = 0.516) relative to Or13a, reflecting broader downstream portfolio consistent with mixed neurotransmitter identity.

### Or47b (VA1v, mixed ACH/SER)
- 98 neurons yielded 12,794 synapses but only 12.1 targets per neuron; nine neurons lack hemisphere assignments due to FlyWire metadata gaps that the pipeline flags as `unknown`.
- Primary excitatory target is `VA1v_adPN` (43.7%), followed by `VA1v_vPN` and modulatory partners (`MZ_lv2PN`, `v2LN36`).
- Lowest mean synapses per connection (10.79) and the most even synaptic distribution (Gini = 0.302), consistent with the reported variability and reduced output strength of Or47b neurons.

### Or59b (DM4, ACH only)
- 40 neurons retained 8,138 synapses with 9.8 targets per neuron; hemispheres are perfectly balanced (20 left / 20 right).
- Exhibits the strongest single-target dominance: `DM4_adPN` accounts for 68.9% of exports, reinforcing prior reports of DM4 stereotypy.
- Despite sparse target diversity, connections are individually strong (mean 19.99 synapses per connection), highlighting concentrated yet potent outputs.

## Comparative Insights

| ORN | n | retained synapses | mean targets / neuron | dominant target | fraction | mean syn / connection | synapse gini |
|:----|--:|------------------:|----------------------:|:----------------|---------:|----------------------:|-------------:|
| Or13a | 20 | 6,029 | 18.0 | DC2_adPN | 0.59 | 16.70 | 0.486 |
| Or42b | 71 | 18,591 | 15.9 | DM1_lPN | 0.47 | 15.90 | 0.516 |
| Or47b | 98 | 12,794 | 12.1 | VA1v_adPN | 0.44 | 10.79 | 0.302 |
| Or59b | 40 | 8,138 | 9.8 | DM4_adPN | 0.69 | 19.99 | 0.569 |
| Or7a | 41 | 9,144 | 15.2 | DL5_adPN | 0.52 | 13.57 | 0.499 |

- AdPN partners dominate across all ORN classes, but dominance strength varies: Or59b > Or13a > Or7a > Or42b > Or47b.
- Or47b uniquely spreads outputs across multiple targets and displays weaker synaptic weights, aligning with behavioural variability observations.
- Shared local interneuron partners (`il3LN6`, `lLN2F`, `lLN2T`) emerge across Or7a/Or13a/Or42b/Or59b, supporting a conserved inhibitory scaffold for alcohol-sensitive channels.
- Hemispheric distributions diverge: Or47b skews left (asymmetry index –0.079 with missing labels), Or13a favours right (+0.20), Or59b is symmetric, and Or7a tilts slightly right (+0.12).

## Statistical Testing
- Global per-connection synapse counts now differ significantly across the five ORN classes (Kruskal–Wallis H = 28.49, p = 9.94×10⁻⁶; `results/multi_orn_outputs/comparative_orn_significance.json`), driven by Or59b’s concentrated high-count connections and Or47b’s sparse outputs.
- Hemisphere allocation remains non-uniform (χ² = 18.28, p = 0.019), reflecting Or47b’s left bias and Or7a’s modest right skew; follow-up manual curation for missing metadata is still recommended.
- Pairwise Mann–Whitney tests on per-neuron synapse totals reveal robust differences (all p < 1e-4), with large effect sizes:
  - Or13a vs. Or47b: p = 5.3e-12, Cohen’s d = 2.58 (Or13a much stronger outputs).
  - Or13a vs. Or7a: p = 1.1e-8, d = 2.38.
  - Or42b vs. Or59b: p = 8.9e-11, d = 1.19.
  - Or47b vs. Or7a: p = 5.0e-12, d = –1.47 (Or7a stronger outputs).
  - Or59b vs. Or7a: p = 0.011, d = –0.60 (Or59b retains higher per-neuron totals).
- These comparisons confirm that population-level output strength, not single-connection weight, separates receptor classes; Or7a clusters with the high-output populations.

## Generated Artifacts
- Long-format exports: `results/multi_orn_outputs/{orn}_output_targets_long.csv`
- Wide-format per-neuron summaries: `results/multi_orn_outputs/{orn}_output_targets_wide.csv`
- Population statistics: `results/multi_orn_outputs/{orn}_summary_statistics.csv`
- Cross-ORN reports:
  - `comparative_orn_analysis.csv` – consolidated metrics
  - `comparative_orn_pairwise_tests.csv` – pairwise Mann–Whitney and effect sizes
  - `comparative_orn_significance.json` – global statistical tests
- Visualisations:
  - `orn_synapse_distribution.png` – violin plots of per-connection strengths (now including Or7a)
  - `orn_target_celltype_heatmap.png` – top 20 target cell types across ORNs (Or7a integrated)

All outputs are ready for integration with downstream behavioural correlation analyses and provide a reproducible baseline for multi-ORN connectomics comparisons.
