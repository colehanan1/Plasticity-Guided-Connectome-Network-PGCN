## OR7a Complete Circuit Pathway Mapping Guide

## Overview

This guide explains how to use `map_or7a_complete_pathway.py` to trace the complete OR7a olfactory pathway from receptor neurons through the mushroom body circuit to behavioral outputs.

### Circuit Architecture

The script maps the multi-level circuit:

```
OR7a ORNs (41)
    ↓
DL5 Projection Neurons (~2)
    ↓
Kenyon Cells (200-500)
    ↓
MBONs (20-50)
    ↓
Behavioral Outputs (motor/descending neurons)
```

## What This Script Does

The complete pathway mapper:

1. **Traces multi-level connectivity** from OR7a through 5 circuit levels
2. **Identifies neuron populations** at each level (PNs, KCs, MBONs, etc.)
3. **Quantifies connection patterns** including convergence/divergence ratios
4. **Detects circuit bottlenecks** and critical chokepoints
5. **Generates connectivity matrices** for each level transition
6. **Identifies critical targets** ranked by importance for experiments
7. **Creates comprehensive visualizations** of the complete pathway

## Quick Start

### Basic Usage

Run complete pathway analysis (all 5 levels):

```bash
python scripts/map_or7a_complete_pathway.py --data-source local
```

### Test with Demo

Quick test with first 3 levels (OR7a → PN → KC):

```bash
python scripts/test_pathway_mapping_demo.py
```

### Custom Analysis

```bash
# Trace specific levels
python scripts/map_or7a_complete_pathway.py --max-levels 3

# Higher synapse threshold
python scripts/map_or7a_complete_pathway.py --min-synapses 10

# Custom output
python scripts/map_or7a_complete_pathway.py --output-dir results/pathway_final/
```

## Output Files

### 1. Complete Pathway (`or7a_complete_pathway.csv`)

Every connection across all circuit levels:

| Column | Description |
|--------|-------------|
| `pre_root_id` | Source neuron root ID |
| `post_root_id` | Target neuron root ID |
| `neuropil` | Neuropil where connection occurs |
| `syn_count` | Number of synapses |
| `cell_type` | Target cell type |
| `target_category` | Functional category (PN, KC, MBON, etc.) |
| `source_level` | Source circuit level (0-4) |
| `source_level_name` | Source level name |
| `target_level` | Target circuit level (1-5) |
| `target_level_name` | Target level name |

**Use case**: Complete circuit reconstruction, network modeling, connectivity analysis

### 2. Pathway Summary Files

#### `pathway_summary_by_level.csv`

Statistics for each circuit level:

- Level number and name
- Neuron count
- Convergence/divergence ratio
- Mean synapses per connection

#### `pathway_summary_connections.csv`

Connection statistics between levels:

- Source and target levels
- Number of source neurons
- Number of target neurons
- Total, mean, median, std synapses

#### `pathway_summary_categories.csv`

Target cell category distributions:

- Source level
- Target category (PN, KC, MBON, Motor, etc.)
- Number of neurons
- Total and mean synapses

#### `pathway_summary_bottlenecks.csv`

Circuit bottleneck analysis:

- Level transition (e.g., "OR7a_ORN → DL5_PN")
- Source/target neuron counts
- Expansion ratio (>1 = divergence, <1 = convergence)
- Bottleneck severity (High/Medium/Low)

### 3. Target Priorities (`target_priorities.csv`)

Ranked list of critical neurons for experimental targeting:

| Column | Description |
|--------|-------------|
| `root_id` | Neuron root ID |
| `cell_type` | Cell type |
| `category` | Functional category |
| `total_synapses` | Total synapses received |
| `num_connections` | Number of input connections |
| `num_levels` | Number of circuit levels involved |
| `importance_score` | Overall importance (synapses × connections) |

**Sorted by importance score** - top entries are the most critical neurons to target.

**Use case**: Prioritizing neurons for optogenetic suppression experiments

### 4. Visualization (`or7a_complete_pathway_analysis.png`)

Comprehensive 6-panel figure:

1. **Pathway Diagram** - Simplified circuit schematic with neuron counts
2. **Neuron Counts by Level** - Bar chart (log scale)
3. **Connection Strengths** - Mean synapses per connection at each level
4. **Category Distribution** - Target cell categories (horizontal bars)
5. **Convergence/Divergence** - Expansion ratios between levels
6. **Synapse Distribution** - Histogram of synapse counts by level

## Understanding the Circuit Levels

### Level 0: OR7a ORNs (Olfactory Receptor Neurons)

**Description**: Peripheral sensory neurons expressing Or7a receptor

**Count**: 41 neurons (from your dataset)

**Function**: Detect benzaldehyde and related aromatics

**Neuropil**: Antenna → Antennal Lobe

**Expected outputs**: Strong connections to DL5 projection neurons

### Level 1: DL5 Projection Neurons (PNs)

**Description**: Second-order olfactory neurons in DL5 glomerulus

**Expected count**: 2-5 neurons (adPN and lPN subtypes)

**Known root IDs**: 720575940639080700, 720575940617207200

**Function**: Relay OR7a signals to mushroom body and lateral horn

**Neuropil**: Antennal Lobe → Mushroom Body calyx

**Expected characteristics**:
- **High convergence**: 41 ORNs → ~2 PNs (ratio ~0.05)
- **Strong synapses**: 40-60 synapses per ORN→PN connection
- **Stereotyped connectivity**: All OR7a converge on same PNs

### Level 2: Kenyon Cells (KCs)

**Description**: Intrinsic neurons of mushroom body

**Expected count**: 200-500 neurons receiving DL5 input

**Subtypes**: KCα/β, KCα'/β', KCγ (different lobes)

**Function**: Sparse encoding of odor combinations, associative learning

**Neuropil**: Mushroom Body calyx → lobes

**Expected characteristics**:
- **High divergence**: ~2 PNs → 200-500 KCs (ratio 100-250)
- **Sparse connections**: 3-10 synapses per PN→KC connection
- **Random sampling**: Each KC receives from ~5-15 different PNs
- **Parallel processing**: Multiple KC subtypes

### Level 3: MBONs (Mushroom Body Output Neurons)

**Description**: Output neurons reading KC activity patterns

**Expected count**: 20-50 MBONs receiving input from OR7a-activated KCs

**Subtypes**: Approach-promoting vs avoidance-promoting

**Function**: Convert KC patterns into behavioral commands

**Neuropil**: Mushroom Body lobes → various brain regions

**Expected characteristics**:
- **Moderate convergence**: 200-500 KCs → 20-50 MBONs (ratio ~0.1)
- **Variable strength**: 5-50 synapses depending on compartment
- **Functional specificity**: Different MBONs control different behaviors

**Key MBONs to look for**:
- **Approach**: MBON-γ1pedc>α/β (M4/M6)
- **Avoidance**: MBON-α2sc, MBON-α3
- **Memory**: MBON-γ2α'1

### Level 4: Behavioral Outputs

**Description**: Motor and descending neurons controlling behavior

**Expected count**: 10-100 neurons

**Types**: Descending neurons (DNs), motor neurons, central complex neurons

**Function**: Execute approach/avoidance decisions

**Expected characteristics**:
- **Variable patterns**: Depends on specific MBON types
- **Weak connections**: 3-15 synapses per MBON→DN
- **Integration**: Multiple MBONs converge on behavior

## Key Analysis Questions Answered

### 1. How many neurons at each level?

**Check**: `pathway_summary_by_level.csv`

**Expected pattern**:
```
Level 0 (OR7a):    41 neurons
Level 1 (PN):      2-5 neurons      [Major convergence]
Level 2 (KC):      200-500 neurons  [Major divergence]
Level 3 (MBON):    20-50 neurons    [Moderate convergence]
Level 4 (Behavior): 10-100 neurons  [Variable]
```

### 2. What are the major bottlenecks?

**Check**: `pathway_summary_bottlenecks.csv`

**Expected bottleneck**: OR7a → DL5_PN

- **Why**: 41 ORNs funnel through only 2 PNs
- **Severity**: High (expansion ratio ~0.05)
- **Implication**: DL5 PNs are critical chokepoints
- **Experimental impact**: Suppressing these 2 PNs should block most OR7a signals

### 3. Which MBONs control approach/avoidance?

**Check**: `or7a_complete_pathway.csv` filtered for level 3→4

**Expected patterns**:
- **Approach MBONs** (γ1, γ2): Should connect to approach-promoting DNs
- **Avoidance MBONs** (α2, α3): Should connect to avoidance DNs
- **Check neuropil**: γ-lobe MBONs vs α-lobe MBONs have different targets

**Analysis code**:
```python
import pandas as pd

pathway = pd.read_csv('results/or7a_complete_pathway/or7a_complete_pathway.csv')
mbons = pathway[pathway['target_category'] == 'MBON']

# Group by MBON type
mbon_summary = mbons.groupby('cell_type').agg({
    'post_root_id': 'first',
    'syn_count': ['sum', 'mean', 'count']
})
print(mbon_summary)
```

### 4. What are optimal targets for experiments?

**Check**: `target_priorities.csv`

**Top priorities by category**:

1. **DL5 projection neurons** (Level 1)
   - **Why**: Essential bottleneck
   - **Effect**: Block most OR7a→KC transmission
   - **Specificity**: High - specific to OR7a pathway

2. **High-connectivity KCs** (Level 2)
   - **Why**: Receive strong DL5 input
   - **Effect**: Reduce specific odor encoding
   - **Specificity**: Medium - KCs integrate multiple inputs

3. **Behavioral MBONs** (Level 3)
   - **Why**: Direct behavioral control
   - **Effect**: Modulate specific behaviors
   - **Specificity**: Variable by MBON type

**Selection criteria**:
- `importance_score` > 1000: Critical targets
- `importance_score` 500-1000: Important targets
- `importance_score` < 500: Minor targets

### 5. How strong is the complete pathway?

**Calculate total pathway strength**:

```python
# Load complete pathway
pathway = pd.read_csv('results/or7a_complete_pathway/or7a_complete_pathway.csv')

# Sum synapses at each level
level_strength = pathway.groupby('source_level')['syn_count'].sum()

print("Total synapses by level:")
print(level_strength)

# Calculate transmission probability
# Assume ~40 synapses needed for reliable transmission
reliable_connections = pathway[pathway['syn_count'] >= 40]
print(f"\nReliable connections: {len(reliable_connections)} / {len(pathway)}")
```

**Expected strength pattern**:
- **OR7a→PN**: Strong (>40 syn) - reliable transmission
- **PN→KC**: Weak (5-10 syn) - sparse, probabilistic
- **KC→MBON**: Medium (10-30 syn) - integrative
- **MBON→Behavior**: Variable (5-50 syn) - context-dependent

## Circuit Validation

### Expected vs Unexpected Results

#### ✅ Expected (Validates Pathway)

1. **DL5 PNs found**: ~2 projection neurons
2. **High OR7a→PN convergence**: Ratio < 0.1
3. **KC divergence**: >100 KCs receive DL5 input
4. **MB compartment organization**: KCs organized by subtype
5. **MBON diversity**: Multiple MBON types

#### ⚠️ Investigate If Found

1. **No DL5 PNs**: Check cell type annotations, try lower synapse threshold
2. **Low KC count** (<50): May indicate incomplete tracing or high threshold
3. **No MBON connections**: KCs might terminate in unmapped regions
4. **Very high MBON→Motor connections**: May include indirect pathways

### Quality Checks

```python
import pandas as pd

pathway = pd.read_csv('results/or7a_complete_pathway/or7a_complete_pathway.csv')

# Check 1: PN bottleneck
pn_count = pathway[pathway['target_category'] == 'PN']['post_root_id'].nunique()
print(f"Projection neurons: {pn_count} (expect 2-5)")
assert pn_count >= 2, "Too few PNs found!"

# Check 2: KC divergence
kc_count = pathway[pathway['target_category'] == 'KC']['post_root_id'].nunique()
print(f"Kenyon cells: {kc_count} (expect 200-500)")
assert kc_count >= 50, "Too few KCs found!"

# Check 3: Connection strengths
level_0_mean = pathway[pathway['source_level'] == 0]['syn_count'].mean()
level_1_mean = pathway[pathway['source_level'] == 1]['syn_count'].mean()
print(f"OR7a→PN mean synapses: {level_0_mean:.1f} (expect 40-60)")
print(f"PN→KC mean synapses: {level_1_mean:.1f} (expect 5-15)")
```

## Integration with Experiments

### OR7a Suppression Experiments

The pathway analysis identifies targets for optogenetic suppression:

**Experimental design**: Test benzaldehyde→hexanol learning with OR7a pathway suppressed

**Target selection from `target_priorities.csv`**:

```python
import pandas as pd

targets = pd.read_csv('results/or7a_complete_pathway/target_priorities.csv')

# Select top DL5 projection neurons
top_pns = targets[targets['category'] == 'PN'].head(3)
print("Priority PN targets:")
print(top_pns[['root_id', 'cell_type', 'importance_score']])

# Select top KCs receiving DL5 input
top_kcs = targets[targets['category'] == 'KC'].head(20)
print("\nPriority KC targets:")
print(top_kcs[['root_id', 'cell_type', 'importance_score']])

# Select behavioral MBONs
top_mbons = targets[targets['category'] == 'MBON'].head(10)
print("\nPriority MBON targets:")
print(top_mbons[['root_id', 'cell_type', 'importance_score']])
```

**Experimental predictions**:

1. **Suppress DL5 PNs** → No benzaldehyde learning (complete block)
2. **Suppress OR7a-responsive KCs** → Impaired but not abolished learning
3. **Suppress specific MBONs** → Selective deficits in approach vs avoidance

### Integration with PGCN Models

Use pathway data to initialize learning models:

```python
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# Load complete pathway
pathway = pd.read_csv('results/or7a_complete_pathway/or7a_complete_pathway.csv')

# Build PN→KC connectivity matrix
pn_to_kc = pathway[(pathway['source_level'] == 1) & (pathway['target_category'] == 'KC')]

pn_ids = pn_to_kc['pre_root_id'].unique()
kc_ids = pn_to_kc['post_root_id'].unique()

# Create ID mappings
pn_map = {id: idx for idx, id in enumerate(pn_ids)}
kc_map = {id: idx for idx, id in enumerate(kc_ids)}

# Build sparse matrix
rows = [pn_map[id] for id in pn_to_kc['pre_root_id']]
cols = [kc_map[id] for id in pn_to_kc['post_root_id']]
data = pn_to_kc['syn_count'].values

W_pn_kc = csr_matrix((data, (rows, cols)), shape=(len(pn_ids), len(kc_ids)))

print(f"PN→KC matrix: {W_pn_kc.shape}")
print(f"Sparsity: {W_pn_kc.nnz / np.prod(W_pn_kc.shape):.3%}")
print(f"Mean synapses: {W_pn_kc.data.mean():.1f}")

# Use in PGCN learning model
from src.pgcn.models.learning_model import LearningModel

model = LearningModel(
    initial_weights=W_pn_kc,
    plasticity_rule='bcm',
    learning_rate=0.001
)
```

## Command-Line Options

### Data Source

```bash
# Use local CSV files (default, faster)
--data-source local

# Query FlyWire API (requires authentication)
--data-source api
```

### Circuit Depth

```bash
# Trace specific number of levels
--max-levels 3  # OR7a → PN → KC → MBON

# Options: 1, 2, 3, 4, or 5
```

### Connection Filtering

```bash
# Minimum synapse threshold
--min-synapses 5  # More stringent (fewer connections)
--min-synapses 1  # More permissive (more connections)
```

### Custom Paths

```bash
--or7a-data path/to/or7a_neurons.csv
--connections path/to/connections.csv.gz
--cell-types path/to/cell_types.csv.gz
--output-dir results/custom_pathway/
```

## Performance Expectations

**Test System**: Standard laptop, 16GB RAM

| Operation | Time | Memory |
|-----------|------|--------|
| Load connections | ~2-3s | ~2 GB |
| Load cell types | ~1s | ~50 MB |
| Level 1 (PN) | ~5s | ~500 MB |
| Level 2 (KC) | ~10s | ~800 MB |
| Level 3 (MBON) | ~15s | ~1 GB |
| Level 4 (Behavior) | ~20s | ~1.2 GB |
| **Total (5 levels)** | **~60s** | **~3 GB peak** |

**Tips for faster analysis**:
- Use `--max-levels 3` to stop at MBONs
- Increase `--min-synapses` to reduce connection count
- Process on machine with ≥8GB RAM

## Troubleshooting

### "Too few PNs found"

**Possible causes**:
- Synapse threshold too high
- Cell type annotations incomplete
- OR7a→PN connections weak

**Solutions**:
```bash
# Lower threshold
--min-synapses 1

# Check raw outputs
python -c "
import pandas as pd
pathway = pd.read_csv('results/or7a_complete_pathway/or7a_complete_pathway.csv')
print(pathway[pathway['target_category'] == 'PN'])
"
```

### "No KC connections found"

**Possible causes**:
- PNs not correctly identified at level 1
- PN→KC connections very sparse

**Solutions**:
- Check level 1 completed successfully
- Use `--min-synapses 1` for PN→KC level
- Verify PNs exist in connections file

### "Memory error"

**Cause**: Loading full connection table (5.3M connections) exceeds RAM

**Solutions**:
- Close other applications
- Use machine with ≥16GB RAM
- Process fewer levels at once

## Next Steps

1. **Run complete analysis** on all 5 levels
2. **Validate circuit structure** against literature
3. **Identify critical targets** for experiments
4. **Compare with other ORN pathways** (OR47a, OR47b)
5. **Use in PGCN modeling** for learning predictions
6. **Design suppression experiments** based on target priorities

## Citation

If using this analysis in publications:

```
FlyWire FAFB v783 connectome:
Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain. Nature.

OR7a pathway analysis:
[Your analysis] using PGCN complete pathway mapping framework.
```

## Contact & Support

For questions:
- Check output mapping guide: [`docs/OR7A_OUTPUT_MAPPING_GUIDE.md`](OR7A_OUTPUT_MAPPING_GUIDE.md)
- Review FlyWire access: [`src/pgcn/flywire_access.py`](../src/pgcn/flywire_access.py)
- See related scripts: [`scripts/map_or7a_outputs.py`](../scripts/map_or7a_outputs.py)

## Version History

- **v1.0** (2025-11-05): Initial release
  - Multi-level pathway tracing (5 levels)
  - Bottleneck detection
  - Target prioritization
  - Comprehensive visualization
  - Circuit validation protocols
