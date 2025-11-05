# OR7a Analysis Module Usage Guide

This document outlines the recommended workflow for processing FlyWire FAFB v783 OR7a (ORN_DL5) neurons with `or7a_analysis.py`. The module supports quality control, connectivity profiling, experiment-oriented ranking, and production of publication-ready figures for optogenetic suppression studies.

## Prerequisites
- Activate the project virtual environment and install dependencies from `requirements.txt`.
- Ensure the FlyWire export is present at `data/flywire/search_results_or7a.csv` (41 neurons expected).
- Optional: set `PYTHONPATH` to the project root or run commands from the repository root so imports resolve.

## Quick Start
```python
from pathlib import Path
from or7a_analysis import OR7aDataProcessor, OR7aAnalysis

processor = OR7aDataProcessor(Path("data/flywire/search_results_or7a.csv"))
or7a_df = processor.load_data()

analysis = OR7aAnalysis(or7a_df)
summary = analysis.synaptic_summary()
hemispheres = analysis.hemispheric_summary()
ranking = analysis.rank_neurons(top_n=15)
analysis.export_target_list(Path("artifacts/or7a_analysis/top_targets.csv"), top_n=20)
```

## Processing Workflow
1. **Load & Validate**  
   `OR7aDataProcessor.load_data()` enforces required columns, 41-neuron count, hemisphere labels, and positive synapse counts. Set `strict=False` to emit warnings instead of raising errors.
2. **Connectivity Summaries**  
   - `synaptic_summary()` returns descriptive statistics for input/output synapses.  
   - `hemispheric_summary()` runs Welch’s t-tests, Cohen’s *d*, and 95% CIs for left vs right populations.  
     ```python
     hemispheres = analysis.hemispheric_summary()
     hemispheres.to_markdown()
     ```
3. **Target Ranking**  
   `rank_neurons()` computes a z-scored connectivity composite. Adjust weights (`input_weight`, `output_weight`) or apply minimum synapse filters. Use `export_target_list()` to generate CSVs for optogenetic targeting.

## Visualization
- `plot_synapse_distribution(metric="input_synapses")` – histogram + KDE.
- `plot_hemispheric_comparison(metric="output_synapses")` – boxplot with swarm overlay.
- `plot_input_output_relationship()` – scatter of input vs output synapses colored by hemisphere.

Each plotting method accepts `save_path` to persist 300 dpi figures and returns the `(fig, ax)` for notebook embedding:
```python
fig, ax = analysis.plot_synapse_distribution(metric="output_synapses")
ax.axvline(200, color="red", linestyle="--", label="Target threshold")
ax.legend()
```

## Automated Example
Run the built-in example to reproduce summary tables, ranked targets, and figures under `artifacts/or7a_analysis/`:
```bash
python or7a_analysis.py
```

## Integration Tips
- Merge ranked outputs with behavioural cohorts to link structural connectivity to cross-odor performance.
- Store exported CSVs (e.g., `or7a_targets_week07.csv`) under `artifacts/` with clear versioning.
- For manuscript figures, load the saved PNGs into the existing plotting pipelines in `app/visualization_utils.py`.

For questions or updates to this workflow, coordinate with the connectome integration lead and keep `docs/model_integration_status.md` in sync with any new FlyWire exports.
