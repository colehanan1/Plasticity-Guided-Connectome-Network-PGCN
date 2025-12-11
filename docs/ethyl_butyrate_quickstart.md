# Ethyl Butyrate Circuit Analysis - Quick Start

## Overview

Production-ready pipeline for mapping the complete ethyl butyrate → PER circuit from FlyWire FAFB v783.

**Target:** Or42a, Or43b, Or42b → PNs → KCs → MBONs
**Prediction:** ~50% PER probability
**Dataset:** Virgin adult female Drosophila (FAFB)

## Quick Start

```bash
# 1. Run complete analysis
python scripts/analysis/ethyl_butyrate_circuit_analysis.py

# 2. View results
jupyter notebook notebooks/ethyl_butyrate_visualization.ipynb

# 3. Run tests
pytest tests/test_ethyl_butyrate_extraction.py -v
```

## Key Files

| File | Description |
|------|-------------|
| `src/circuit_analysis/ethyl_butyrate_mapper.py` | Core extraction functions |
| `scripts/analysis/ethyl_butyrate_circuit_analysis.py` | Main pipeline script |
| `notebooks/ethyl_butyrate_visualization.ipynb` | Interactive visualizations |
| `tests/test_ethyl_butyrate_extraction.py` | Unit tests |
| `reports/ethyl_butyrate_pipeline_guide.md` | Complete documentation |

## Output Structure

```
data/cache/ethyl_butyrate_circuit/
├── or42a_or43b_or42b_neurons.csv        # 141 ORNs
├── dm3_vm2_dm1_pns.csv                   # PNs
├── appetitive_mbons.csv                  # MBONs
├── circuit_topology.json                 # Graph
├── connectivity_matrices/*.npz           # Sparse matrices
└── analysis/*.json                       # Metrics
```

## Expected Results

### Neuron Counts

- **ORNs:** ~141 (Or42a: 33, Or43b: 37, Or42b: 71)
- **PNs:** 5-10 per glomerulus
- **KCs:** ~1000-2000
- **MBONs:** 2-5 appetitive types

### Stochasticity Metrics

- **Mean ORN→PN synapses:** 50-150 (signal strength)
- **CV (coefficient of variation):** 0.3-0.6 (reliability)
- **Bottleneck score:** CV / mean (transmission failure risk)

### PER Prediction

- **Target:** 50% (behavioral data)
- **Model output:** Should match ±20%
- **Noise sources:** APL inhibition, KC sparsity, MBON integration

## Full Documentation

See [reports/ethyl_butyrate_pipeline_guide.md](../reports/ethyl_butyrate_pipeline_guide.md) for:

- Detailed biological context
- Function API documentation
- Validation procedures
- Troubleshooting guide
- PGCN integration instructions
- References and citations

## Troubleshooting

### "No ORNs found"
→ Check glomerulus labels in `data/flywire/processed_labels.csv.gz`

### "No connections found"
→ Lower `--min-synapses` threshold (try 3 instead of 5)

### "Memory error"
→ Use chunked loading or filter by neuropil early

### "DoOR odorant not found"
→ Check exact spelling in `data/door_cache/door_response_matrix.csv`

## Citation

```bibtex
@software{ethyl_butyrate_pipeline_2025,
  author = {Claude Code (Anthropic)},
  title = {Ethyl Butyrate Appetitive Circuit Extraction Pipeline},
  year = {2025},
  version = {1.0.0},
  note = {FlyWire FAFB v783 connectome analysis}
}
```
