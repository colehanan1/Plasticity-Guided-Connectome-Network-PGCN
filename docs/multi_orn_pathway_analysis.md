# Multi-ORN Pathway Analysis

This note summarises the new multi-receptor pathway tracing workflow implemented in `scripts/map_multi_orn_pathways.py`. The pipeline reproduces the validated OR7a pathway analysis and extends it to Or13a, Or42b, Or47b, and Or59b so that all five receptor classes can be compared on identical PN→KC→MBON→behavior metrics.

## How to Run

- Combined multi-ORN export (all five receptors):
  ```
  python scripts/map_multi_orn_pathways.py --log-level INFO
  ```
- Optional single-ORN regeneration for OR7a baseline parity:
  ```
  python scripts/map_or7a_complete_pathway.py --data-source local
  ```

The multi-ORN command writes:
- Per-ORN artefacts in `results/{orn}_complete_pathway/`
- Comparative summaries and figures in `results/multi_orn_pathways/comparative_analysis/`
- All pathway outputs now expand abbreviated cell type labels to their descriptive names (e.g., `lLN2F_b` becomes “lateral local neuron 2Fb”), ensuring the figures and CSV exports are publication-ready.

## Per-ORN Highlights

- **Or7a (DL5, ACH only)**  
  - 29 DL5 projection neurons downstream of 41 ORNs (detected via cell-type labels).  
  - 575 Kenyon cells and 69 MBONs replicate the single-ORN analysis, with behavioural projections to 145 downstream partners.  
  - Mean synapses per connection: PN→KC ≈33.7, KC→MBON ≈16.8, MBON→behavior ≈6.1 / 11.3, mirroring the dedicated OR7a workflow.

- **Or13a (DC2, ACH only)**  
  - 21 DC2 projection neurons feeding 348 KCs (convergence ratio ≈0.06).  
  - 63 MBONs funnel activity to 143 behavioural targets split across Central-Complex and motor pathways.  
  - PN→KC synapses average ≈30.4, highlighting a robust, stereotyped bottleneck.

- **Or42b (DM1, mixed ACH/SER)**  
  - 51 DM1 projection neurons deliver 1,602 KCs—the widest recruitment in the cohort.  
  - 69 MBONs and 148 behavioural targets reflect the mixed-behaviour character of DM1.  
  - PN→KC synapses average ≈42.8, sustaining a highly distributed downstream drive.

- **Or47b (VA1v, mixed ACH/SER)**  
  - 25 VA1v projection neurons connect to 1,096 KCs (low convergence, high variability).  
  - 70 MBONs and 148 behavioural targets underline context-dependent outputs.  
  - PN→KC synapses remain weaker (≈11.6 mean), reinforcing the observed behavioural variability.

- **Or59b (DM4, ACH only)**  
  - 7 DM4 projection neurons (tight bottleneck) feed 611 KCs.  
  - 62 MBONs and 131 behavioural targets emphasise attractive bias with compact circuitry.  
  - PN→KC synapses average ≈65.6, the strongest among the cohort.

## Comparative Metrics

- **Convergence/Divergence** (`results/multi_orn_pathways/comparative_analysis/multi_orn_convergence_comparison.csv`)  
  - PN bottlenecks: Or59b (7 PNs) < Or13a (21) < Or47b (25) < Or7a (29) < Or42b (51).  
  - KC recruitment ratios mirror pathway breadth: Or59b ≈0.01, Or47b ≈0.02, Or42b ≈0.03, Or7a ≈0.05, Or13a ≈0.06.

- **Bottlenecks** (`receptor_specific_bottlenecks.csv`)  
  - Or13a/Or59b bottleneck at MBON level (focused outputs).  
  - Or42b/Or47b bottleneck earlier at PN level, consistent with mixed neurotransmission and weaker convergence.

- **KC Recruitment** (`cross_orn_kc_recruitment.csv`)  
  - KC pool sizes: Or42b 1,602 > Or47b 1,096 > Or7a 575 > Or59b 611 > Or13a 348.  
  - Mean KC synapses remain within 16–19 across receptors, aligning with hemibrain KC fan-in statistics.

- **Behavioural Output** (`behavioral_output_specialization.csv`)  
  - All pathways ultimately channel through MBON populations (MBON share = 1.0).  
  - Downstream allocation diverges: Or13a balances Central Complex and motor targets, Or7a/Or59b skew motor-heavy, while Or42b/Or47b distribute across multiple behavioural hubs.

- **Statistical Significance** (`pathway_significance_tests.json`)  
  - Kruskal–Wallis tests indicate significant synapse-count differences across ORNs at PN, KC, and MBON transitions (p < 1e-10), but not at the behaviour level (p ≈ 0.93), implying convergence differences diminish after MBON integration.

## Visualisation

- `results/multi_orn_pathways/comparative_analysis/multi_orn_pathway_overview.png` provides a five-panel log-scale comparison of neuron counts per level, highlighting the PN bottleneck in Or59b and the broad KC recruitment for Or42b/Or47b.

## Integration Notes

- All per-ORN outputs follow the same schema established by `map_or7a_complete_pathway.py`, enabling downstream scripts (blocking analyses, behavioural correlation) to consume either single-ORN or multi-ORN datasets without modification.
- Running `map_multi_orn_pathways.py` now regenerates all five receptor directories (`results/{orn}_complete_pathway/`) including Or7a, so historical single-ORN scripts remain optional for cross-checks only.
- Multi-ORN outputs complement the previously generated Or7a heatmaps and CSVs (`scripts/map_multi_orn_outputs.py`); together they provide both direct target statistics and whole-pathway context for multi-receptor blocking experiments.
