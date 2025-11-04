# GRN Connectivity & Visualization Pipeline

This document summarizes how to run and interpret the production-grade gustatory
receptor neuron (GRN) connectivity pipeline implemented in
`scripts/grn_downstream_pipeline.py`.

## Overview

- **Purpose:** Load verified GRN cohorts, fetch downstream connectivity from
  FlyWire (or synthetic demo data), and generate four interactive, shareable
  visualizations plus a JSON summary for downstream modelling.
- **GRN Selection:** Always relies on structured classification
  (`sub_class == "sugar/water"`) to prevent fragile text-mining failures.
- **Outputs:** 3D network, connectivity heatmap + synapse scatter, degree
  dashboard, composition summary with top synaptic partners, and a machine-readable
  statistics JSON file.
- **Data Integrity Checks:** Confirms expected GRN counts (131 sugar, 343 total),
  ensures classification consistency, and validates generated HTML files (>100 KB).

## Quick Start

```bash
# 1. Activate environment and install dependencies (if needed)
conda activate pgcn           # or source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the pipeline with synthetic demo data (safe offline default)
python scripts/grn_downstream_pipeline.py --include-demo true

# 3. View artifacts
ls reports/grn_network_html
```

## Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--grn-type {sugar,all}` | Cohort to analyze (`sugar` = 131 neurons, `all` = 343) | `sugar` |
| `--include-demo {true,false}` | Force synthetic connectivity (no API calls) | `false` |
| `--include-brain-mesh {true,false}` | Overlay scaled FAFB14 mesh (requires `flybrains`) | `false` |
| `--output-dir PATH` | Destination for HTML + JSON outputs | `reports/grn_network_html` |
| `--connectivity-cache PATH` | JSON cache for downstream partners | `data/flywire/downstream_connectivity.json` |
| `--max-retries INT` | FlyWire retry attempts per neuron | `3` |
| `--retry-backoff FLOAT` | Base seconds for exponential backoff | `2.0` |
| `--verbosity {0,1,2}` | Logging verbosity (0=warnings, 1=info, 2=debug) | `1` |

### Example Runs

```bash
# Production run (uses FlyWire when fafbseg is installed)
python scripts/grn_downstream_pipeline.py

# Analyze all gustatory neurons and embed FAFB14 brain mesh overlay
python scripts/grn_downstream_pipeline.py --grn-type all --include-brain-mesh true

# Increase logging verbosity for debugging FlyWire interactions
python scripts/grn_downstream_pipeline.py --verbosity 2
```

## Outputs

All artifacts are written to the output directory (default:
`reports/grn_network_html/`):

- `grn_downstream_network.html` – 3D network with GRN inputs (red) and coloured downstream classes.
- `grn_connectivity_heatmap.html` – Heatmap of GRN sub-class vs downstream class plus synapse scatter view.
- `grn_degree_distribution_dashboard.html` – Degree histograms and in/out degree scatter plot.
- `grn_connectivity_summary_dashboard.html` – Class/neuropil bar charts with top 2,000 synaptic partners.
- `grn_connectivity_summary.json` – Machine-readable statistics (counts, means, class distributions, metadata).

Each HTML is responsive, includes Plotly via CDN, and validated to exceed 100 KB
(target size 0.1–3 MB).

## Validation Pipeline

1. **GRN Integrity:** Checks sugar GRNs count (131) and sub-class uniformity, verifies all GRNs (343) remain gustatory.
2. **Connectivity Fetch:** Automatic retry with exponential backoff, safe fallback to synthetic data; cache reused across runs.
3. **Graph Build:** Limits downstream partners to the top 75 synapses per GRN for tractable layouts (full counts preserved in reports).
4. **Statistical Summary:** Computes totals, unique targets, mean/median synapses, class and neuropil distributions.
5. **Artifact Validation:** Confirms output HTML files exist and are >100 KB; raises descriptive errors otherwise.

## Tips & Troubleshooting

- **FlyWire API unavailable?** Set `--include-demo true` for deterministic synthetic connectivity.
- **Brain mesh overlay missing?** Install `flybrains` (`pip install flybrains`) and rerun with `--include-brain-mesh true`.
- **Large cohorts (grn-type all):** Expect longer layout computation; results stay within the 10-minute target on modern laptops.
- **Regeneration:** Delete `data/flywire/downstream_connectivity.json` to force a fresh FlyWire fetch.

## Downstream Integration

The generated `grn_connectivity_summary.json` and cached connectivity JSON can be
consumed by:

- PGCN model initialization (root IDs and synapse weights).
- Hypothesis testing notebooks (import JSON for reproducible statistics).
- Publication figures (HTML exports embed all data for reviewers).

For extended analysis patterns, see `scripts/navis_morphology_visualizer.py`
and `scripts/analyze-neuropils-connectivity.py` for connectome exploration
examples.
