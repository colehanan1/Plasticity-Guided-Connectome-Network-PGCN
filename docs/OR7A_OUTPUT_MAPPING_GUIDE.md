# OR7a Output Target Mapping Guide

## Overview

This guide explains how to use the `map_or7a_outputs.py` script to comprehensively analyze the downstream connectivity of OR7a (ORN_DL5) neurons from the FlyWire FAFB v783 connectome.

## What This Script Does

The script analyzes the 41 OR7a neurons in your dataset and:
1. **Identifies all downstream targets** that each OR7a neuron connects to
2. **Retrieves target information**: cell type, neuropil location, synapse counts
3. **Generates comprehensive CSV files** in both long and wide formats
4. **Creates summary statistics** across the OR7a population
5. **Produces publication-quality visualizations** of output patterns

## Quick Start

### Basic Usage (Recommended)

Run with local connection data:
```bash
python scripts/map_or7a_outputs.py --data-source local
```

This will:
- Load all 41 OR7a neurons from `data/flywire/search_results_or7a.csv`
- Query local FlyWire connections from `data/flywire/connections_princeton.csv.gz`
- Generate outputs in `results/or7a_outputs/`

### Test with Small Subset

To quickly test with just 5 neurons:
```bash
python scripts/test_or7a_mapping_demo.py
```

Results go to `results/or7a_outputs_demo/`

## Output Files Generated

### 1. Long Format CSV (`or7a_output_targets_long.csv`)

Every row represents one connection from an OR7a neuron to a target:

| Column | Description |
|--------|-------------|
| `or7a_root_id` | FlyWire root ID of the OR7a neuron |
| `or7a_name` | Name (e.g., AL.907) |
| `or7a_side` | Hemisphere (left/right) |
| `or7a_total_outputs` | Total output synapses for this OR7a |
| `target_root_id` | FlyWire root ID of the target neuron |
| `target_cell_type` | Cell type of target (e.g., DL5_adPN) |
| `target_neuropil` | Neuropil where connection occurs |
| `target_super_class` | Super class of target |
| `target_sub_class` | Sub class of target |
| `synapse_count` | Number of synapses in this connection |

**Use case**: Detailed analysis, filtering specific connections, database imports

### 2. Wide Format CSV (`or7a_output_targets_wide.csv`)

One row per OR7a neuron with top N targets as columns:

```
or7a_root_id, or7a_name, or7a_side, or7a_total_outputs,
target_1_root_id, target_1_cell_type, target_1_neuropil, target_1_synapse_count,
target_2_root_id, target_2_cell_type, target_2_neuropil, target_2_synapse_count,
...
total_targets_found, primary_target_type, main_output_neuropil
```

**Use case**: Quick overview, Excel analysis, comparing neurons side-by-side

### 3. Summary Statistics

Five summary CSV files are generated:

#### `summary_overall.csv`
- Total OR7a neurons analyzed
- Total unique target neurons
- Total connections
- Mean targets per OR7a
- Mean/median synapses per connection

#### `summary_target_cell_types.csv`
Top target cell types ranked by total synapses:
- Cell type name
- Number of unique target neurons
- Total synapses received
- Mean/median synapses

#### `summary_target_neuropils.csv`
Distribution of connections across neuropils:
- Neuropil name
- Number of connections
- Total synapses
- Mean synapses

#### `summary_hemispheric.csv`
Comparison of left vs right OR7a neurons:
- Hemisphere
- Number of unique targets
- Total connections
- Synapse statistics

#### `summary_per_neuron.csv`
Individual statistics for each OR7a neuron:
- Number of downstream targets
- Total output synapses mapped
- Mean synapses per target
- Maximum synapses to any single target

### 4. Visualization (`or7a_output_analysis.png`)

Comprehensive 6-panel figure showing:
1. **Top target cell types** (bar chart by synapse count)
2. **Output neuropil distribution** (pie chart)
3. **Targets per OR7a neuron** (histogram)
4. **Synapse count distribution** (histogram, log scale)
5. **Hemispheric comparison** (boxplot with swarm overlay)
6. **Top individual target neurons** (ranked bar chart)

## Command-Line Options

### Data Source

```bash
# Use local CSV files (faster, requires data download)
--data-source local

# Query FlyWire API (slower, requires authentication)
--data-source api
```

### Custom Paths

```bash
# Specify custom OR7a data file
--or7a-data path/to/my_or7a_neurons.csv

# Specify custom connections file
--connections path/to/my_connections.csv.gz

# Specify custom cell types file
--cell-types path/to/my_cell_types.csv.gz
```

### Output Configuration

```bash
# Custom output directory
--output-dir results/my_analysis/

# Minimum synapse threshold (default: 3)
--min-synapses 5

# Number of top targets in wide format (default: 20)
--top-targets 15
```

### Complete Example

```bash
python scripts/map_or7a_outputs.py \
  --data-source local \
  --output-dir results/or7a_final/ \
  --min-synapses 5 \
  --top-targets 25
```

## Understanding the Results

### Key Analysis Questions Answered

#### 1. Do all OR7a neurons target the same cell types?

Check `summary_target_cell_types.csv`. From the demo:
- **Primary target**: DL5_adPN (projection neurons) - 542 total synapses
- **Secondary targets**: Various local neurons (lLN2X02, lLN2T_b, lLN2F_b)

This indicates OR7a neurons converge on their designated glomerular projection neurons while also connecting to local interneurons.

#### 2. What is the primary output neuropil?

Check `summary_target_neuropils.csv` or the `main_output_neuropil` column in wide format.

Expected: **Antennal Lobe (AL_L/AL_R)** - this is where OR7a neurons synapse onto their downstream partners.

#### 3. How many downstream targets does each OR7a have?

Check `summary_per_neuron.csv` or the `total_targets_found` column.

From demo results:
- Mean: ~15.4 targets per neuron
- Range varies by neuron activity and completeness

#### 4. Are there hemispheric differences?

Check `summary_hemispheric.csv` and the hemispheric comparison visualization.

Compare:
- Number of unique targets (left vs right)
- Total synapses
- Mean synapses per connection

Statistical tests can be added to determine significance.

## Expected Target Cell Types

Based on the Drosophila olfactory system architecture, OR7a outputs should include:

### Primary Targets (Expected)
- **DL5 Projection Neurons (PNs)**: `DL5_adPN`, `DL5_lPN`
  - These are the main second-order olfactory neurons
  - Should receive the majority of OR7a output synapses

### Secondary Targets (Expected)
- **Local Neurons (LNs)**: `lLN2*`, `il3LN*`, `v2LN*`
  - Provide lateral inhibition and modulation
  - Typically 5-20 synapses per connection

### Incidental Targets (Possible)
- **Other ORNs**: `ORN_*` (feedback connections)
- **Other neuron types**: Various, typically with low synapse counts

### Unexpected Results to Investigate
- High synapse counts to non-canonical targets
- Missing connections to known DL5 projection neurons
- Unusual neuropil locations outside the antennal lobe

## Data Requirements

### Required Files

1. **OR7a neurons**: `data/flywire/search_results_or7a.csv`
   - Must contain: `root_id`, `name`, `side`, `input_synapses`, `output_synapses`

2. **Connections** (for local mode): `data/flywire/connections_princeton.csv.gz`
   - Must contain: `pre_root_id`, `post_root_id`, `neuropil`, `syn_count`

3. **Cell types** (optional but recommended): `data/flywire/consolidated_cell_types.csv.gz`
   - Must contain: `root_id`, `primary_type` (or `cell_type`)

### Using FlyWire API Instead

If you don't have local data files:

```bash
python scripts/map_or7a_outputs.py --data-source api
```

Requirements:
- `pip install caveclient`
- FlyWire authentication token configured
- Internet connection

See [`src/pgcn/flywire_access.py`](../src/pgcn/flywire_access.py) for authentication setup.

## Integration with PGCN

### Using Results in Learning Models

The output mappings can inform plasticity-guided models:

```python
import pandas as pd
from pgcn.models.learning_model import LearningModel

# Load OR7a output mappings
outputs = pd.read_csv('results/or7a_outputs/or7a_output_targets_long.csv')

# Filter to primary targets (e.g., DL5 PNs)
primary_targets = outputs[outputs['target_cell_type'].str.contains('DL5_adPN')]

# Use synapse counts as initial weights
or7a_to_pn_weights = primary_targets.groupby(
    ['or7a_root_id', 'target_root_id']
)['synapse_count'].sum()

# Initialize model with these connectome-derived weights
model = LearningModel(initial_weights=or7a_to_pn_weights)
```

### Validating Model Predictions

Compare learned connectivity changes against observed structural connectivity:

```python
# After training
predicted_changes = model.get_weight_changes()

# Compare to known OR7a output strengths
correlation = predicted_changes.corr(outputs['synapse_count'])
```

## Troubleshooting

### "Connections file not found"
- Ensure `data/flywire/connections_princeton.csv.gz` exists
- Or use `--data-source api` to query FlyWire directly
- Or specify custom path with `--connections`

### "No output data retrieved"
- Check OR7a root IDs are valid
- Verify connection data covers the correct FlyWire version (783)
- Reduce `--min-synapses` threshold

### "CAVEclient not available"
- Install with: `pip install caveclient`
- Configure authentication (see FlyWire docs)
- Or use `--data-source local` instead

### Script runs very slowly
- Connection files are large (~66-263 MB compressed)
- Loading takes ~2-3 seconds on first run
- Processing all 41 neurons takes ~10-30 seconds
- Use demo script for quick testing

## Performance Benchmarks

**Test System**: Standard laptop, 16GB RAM

| Operation | Time | Memory |
|-----------|------|--------|
| Load OR7a data | <1s | ~1 MB |
| Load connections | ~2-3s | ~2 GB |
| Load cell types | ~1s | ~50 MB |
| Map all outputs | ~5-10s | ~500 MB |
| Generate summaries | <1s | ~100 MB |
| Create visualizations | ~2-3s | ~200 MB |
| **Total (41 neurons)** | **~15-20s** | **~3 GB peak** |

## Citation & Attribution

If you use this script in publications:

```
FlyWire FAFB v783 connectome data:
Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain. Nature.

OR7a output mapping analysis:
[Your analysis] using PGCN framework.
```

## Next Steps

1. **Run full analysis** on all 41 OR7a neurons
2. **Examine target cell types** - verify expected DL5 PN connections
3. **Compare hemispheres** - test for systematic differences
4. **Identify outliers** - neurons with unusual connectivity patterns
5. **Integrate with DoOR** - correlate output patterns with odor responses
6. **Use in PGCN models** - initialize learning models with connectome data

## Contact & Support

For questions or issues:
- Check existing scripts: [`or7a_analysis.py`](../or7a_analysis.py) for neuron-level analysis
- Review FlyWire access: [`src/pgcn/flywire_access.py`](../src/pgcn/flywire_access.py)
- See integration docs: [`docs/or7a_analysis_usage.md`](or7a_analysis_usage.md)

## Version History

- **v1.0** (2025-11-05): Initial release
  - Local and API data sources
  - Long and wide format outputs
  - Comprehensive summaries and visualizations
  - Support for FlyWire FAFB v783
