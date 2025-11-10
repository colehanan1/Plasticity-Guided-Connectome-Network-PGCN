# LN-Glomerulus Mapping Guide

## Overview

This guide explains how to use the **LN-glomerulus mapping script** to identify which glomeruli each Local Neuron (LN) is associated with by analyzing their connectivity patterns with Projection Neurons (PNs).

## Why This Approach?

Unlike Projection Neurons, **Local Neurons in FlyWire datasets typically do NOT have explicit glomerulus labels** in their cell type annotations. For example:
- PNs have labels like: `PN_DL5`, `DA1_adPN`, `uPN_VM2`
- LNs have labels like: `lLN2_F`, `LN_broad`, `ALLN` (no glomerulus info!)

We solve this by **inferring glomeruli from connectivity**:
- **Source glomerulus**: Which PN glomerulus provides INPUT to the LN
- **Target glomerulus**: Which PN glomerulus receives OUTPUT from the LN

## Quick Start

### Basic Usage

```bash
# Activate your conda environment
conda activate PGCN

# Run with default settings
python scripts/map_ln_glomeruli.py \
  --data-dir data/flywire \
  --output-dir results/ln_mapping
```

### Custom Parameters

```bash
# With custom synapse threshold and neuropil filter
python scripts/map_ln_glomeruli.py \
  --data-dir data/flywire \
  --output-dir results/ln_mapping \
  --min-synapses 5 \
  --neuropil AL
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | Path | `data/flywire` | Directory containing FlyWire CSV files |
| `--output-dir` | Path | `results/ln_mapping` | Directory for output files |
| `--min-synapses` | int | `3` | Minimum synapse threshold for connections |
| `--neuropil` | str | `None` | Neuropil to focus on (e.g., "AL" for antennal lobe) |

## Required Input Files

The script expects the following files in `--data-dir`:

```
data/flywire/
├── connections_princeton.csv.gz       # Synaptic connections
├── consolidated_cell_types.csv.gz     # Cell type annotations
└── classification.csv.gz               # Hierarchical classifications (optional)
```

### File Schemas

**connections_princeton.csv.gz:**
```
pre_pt_root_id, post_pt_root_id, syn_count, neuropil
720575940612345678, 720575940698765432, 12, AL(R)
```

**consolidated_cell_types.csv.gz:**
```
root_id, cell_type, hemibrain_type, flywire_type
720575940612345678, lLN2_F, local_LN, ALLN
```

## Output Files

The script generates **4 CSV files** in the output directory:

### 1. `ln_glomerulus_associations.csv`

**Complete list of all LN-glomerulus associations.**

```csv
ln_id,cell_type,glomerulus,ln_category,num_glomeruli,input_synapses,output_synapses,total_synapses,num_input_pns,num_output_pns,connection_direction
720575940632145678,lLN2_F,DL5,multiglomerular,5,234,189,423,12,8,bidirectional
720575940632145678,lLN2_F,DA1,multiglomerular,5,145,201,346,8,10,bidirectional
720575940698234567,LN_broad,VM2,broad,25,89,92,181,4,5,bidirectional
```

**Key Fields:**
- `ln_id`: FlyWire root ID of the LN
- `glomerulus`: Inferred glomerulus association
- `ln_category`: `uniglomerular` | `oligoglomerular` | `multiglomerular` | `broad`
- `total_synapses`: Sum of input + output synapses with this glomerulus
- `connection_direction`: `bidirectional` | `input_only` | `output_only`
- `num_glomeruli`: How many total glomeruli this LN connects to

**Use Cases:**
- Find all LNs associated with a specific glomerulus (e.g., DL5)
- Identify cross-glomerular LN pathways
- Analyze connection strength patterns

### 2. `ln_primary_glomerulus.csv`

**One row per LN, showing strongest glomerulus association.**

```csv
ln_id,cell_type,glomerulus,total_synapses,ln_category,num_glomeruli
720575940632145678,lLN2_F,DL5,423,multiglomerular,5
720575940698234567,LN_broad,VM2,181,broad,25
```

**Use Cases:**
- Assign a single "primary" glomerulus to each LN
- Count LNs per glomerulus
- Categorize LNs by their main association

### 3. `glomerulus_ln_summary.csv`

**Summary statistics per glomerulus.**

```csv
glomerulus,num_lns,total_synapses,input_synapses,output_synapses
DL5,45,12456,6234,6222
DA1,38,10234,5123,5111
VM2,42,9876,4988,4888
```

**Use Cases:**
- Compare LN innervation across glomeruli
- Identify glomeruli with most LN input/output
- Calculate LN density per glomerulus

### 4. `ln_categories.csv`

**Categorization of each LN by glomerular breadth.**

```csv
ln_id,ln_category,num_glomeruli,cell_type
720575940632145678,multiglomerular,5,lLN2_F
720575940698234567,broad,25,LN_broad
```

**LN Categories:**
- **Uniglomerular** (1 glomerulus): Highly specific, rare (~1-5% of LNs)
- **Oligoglomerular** (2-3 glomeruli): Pairwise interactions (~5-10%)
- **Multiglomerular** (4-10 glomeruli): Local processing (~20-30%)
- **Broad** (11+ glomeruli): Global gain control/inhibition (~60-70%)

**Use Cases:**
- Understand LN functional diversity
- Compare with literature classifications
- Filter by specificity for downstream analyses

## Example Analysis Workflows

### Workflow 1: Find LNs for a Specific Glomerulus

```python
import pandas as pd

# Load complete associations
df = pd.read_csv('results/ln_mapping/ln_glomerulus_associations.csv')

# Filter for DL5 glomerulus (Or7a)
dl5_lns = df[df['glomerulus'] == 'DL5'].copy()

# Sort by connection strength
dl5_lns = dl5_lns.sort_values('total_synapses', ascending=False)

print(f"LNs associated with DL5: {dl5_lns['ln_id'].nunique()}")
print(f"\nTop 10 by synapse strength:")
print(dl5_lns[['ln_id', 'cell_type', 'total_synapses', 'connection_direction']].head(10))
```

### Workflow 2: Identify Cross-Glomerular Pathways

```python
# Load complete associations
df = pd.read_csv('results/ln_mapping/ln_glomerulus_associations.csv')

# Find LNs that connect DL5 to DM1-DM4 (potential blocking pathway)
source_glom = 'DL5'
target_gloms = ['DM1', 'DM2', 'DM3', 'DM4']

# Get LNs associated with source
source_lns = set(df[df['glomerulus'] == source_glom]['ln_id'])

# Get LNs associated with targets
target_lns = set(df[df['glomerulus'].isin(target_gloms)]['ln_id'])

# Find cross-glomerular LNs
cross_glom_lns = source_lns & target_lns

print(f"LNs connecting {source_glom} to {target_gloms}: {len(cross_glom_lns)}")

# Get details
cross_glom_df = df[df['ln_id'].isin(cross_glom_lns)]
print(cross_glom_df.pivot_table(
    index='ln_id',
    columns='glomerulus',
    values='total_synapses',
    fill_value=0
))
```

### Workflow 3: Analyze LN Categories

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load categories
df = pd.read_csv('results/ln_mapping/ln_categories.csv')

# Plot distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='ln_category', order=['uniglomerular', 'oligoglomerular', 'multiglomerular', 'broad'])
plt.title('LN Glomerular Breadth Distribution')
plt.ylabel('Number of LNs')
plt.xlabel('LN Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/ln_mapping/ln_category_distribution.png', dpi=300)
plt.close()

# Print statistics
print("LN Category Distribution:")
for cat in ['uniglomerular', 'oligoglomerular', 'multiglomerular', 'broad']:
    count = (df['ln_category'] == cat).sum()
    pct = 100 * count / len(df)
    print(f"  {cat}: {count} ({pct:.1f}%)")
```

### Workflow 4: Compare Input vs Output Connectivity

```python
# Load complete associations
df = pd.read_csv('results/ln_mapping/ln_glomerulus_associations.csv')

# Calculate input/output balance per glomerulus
glom_balance = df.groupby('glomerulus').agg({
    'input_synapses': 'sum',
    'output_synapses': 'sum'
}).reset_index()

glom_balance['input_output_ratio'] = (
    glom_balance['input_synapses'] / glom_balance['output_synapses']
)

# Find glomeruli with asymmetric LN connectivity
asymmetric = glom_balance[
    (glom_balance['input_output_ratio'] > 1.5) |
    (glom_balance['input_output_ratio'] < 0.67)
].sort_values('input_output_ratio')

print("Glomeruli with asymmetric LN connectivity:")
print(asymmetric)
```

## Validation & Quality Checks

The script automatically runs 5 validation checks:

### Check 1: LN Detection
- **Expected**: 800-1000 LNs
- **Warning if**: < 800 LNs found

### Check 2: Glomerulus Coverage
- **Expected**: 30-60 unique glomeruli
- **Warning if**: Outside this range

### Check 3: LN Glomerular Breadth
- **Expected**: Median 5-10 glomeruli per LN
- **Warning if**: Median < 3

### Check 4: Connection Direction Balance
- **Expected**: >40% bidirectional connections
- **Warning if**: < 40% bidirectional

### Check 5: LN Mapping Coverage
- **Expected**: >90% of LNs mapped to at least one glomerulus
- **Warning if**: < 90% coverage

## Running Validation Tests

After running the main script, validate outputs:

```bash
python scripts/test_ln_mapping.py
```

This will:
- Check all output files exist
- Verify file structures
- Show summary statistics
- Test specific glomerulus associations (e.g., DL5)

## Integration with DoOR Database

Combine LN connectivity with odorant responses:

```python
from door_toolkit.encoder import DoOREncoder
from door_toolkit.integration.integrator import DoORFlyWireIntegrator

# Get activated glomeruli for an odorant
integrator = DoORFlyWireIntegrator()
odorant = 'benzaldehyde'
activated_glomeruli = integrator.get_activated_glomeruli(odorant, threshold=0.1)

# Load LN associations
ln_df = pd.read_csv('results/ln_mapping/ln_glomerulus_associations.csv')

# Find LNs involved in benzaldehyde processing
benz_lns = ln_df[ln_df['glomerulus'].isin(activated_glomeruli)]

print(f"LNs involved in benzaldehyde processing:")
print(f"  Total: {benz_lns['ln_id'].nunique()} LNs")
print(f"  Across: {len(activated_glomeruli)} glomeruli")
```

## Troubleshooting

### Issue: "No PNs with glomerulus labels found"

**Cause**: PN glomerulus extraction regex patterns don't match your data.

**Solution**: Check PN cell type labels and adjust patterns in the script:

```python
# Check what PN labels look like
import pandas as pd
ct = pd.read_csv('data/flywire/consolidated_cell_types.csv.gz')
pn_labels = ct[ct['cell_type'].str.contains('PN', na=False)]['cell_type'].head(20)
print(pn_labels)
```

### Issue: "Very few LN-glomerulus associations found"

**Possible causes**:
1. Neuropil filter is too strict
2. Min synapse threshold is too high
3. PN detection is missing many PNs

**Diagnostic**:
```bash
# Try without neuropil filter
python scripts/map_ln_glomeruli.py --min-synapses 1

# Check connection counts
import pandas as pd
conn = pd.read_csv('data/flywire/connections_princeton.csv.gz')
print(f"Total connections: {len(conn):,}")
print(f"AL connections: {conn['neuropil'].str.contains('AL', na=False).sum():,}")
```

### Issue: "Most LNs categorized as 'broad'"

**This is expected!** Broad LNs are the most common type (~60-70%). They provide global gain control across many glomeruli. This is normal biology, not a bug.

If >90% are broad, check:
- Are you filtering weak connections too aggressively?
- Are there false positive LN classifications?

## Expected Runtime

- **Time**: 30-90 seconds for full analysis
- **Memory**: 4-6 GB peak (loading 5M+ connections)
- **Disk**: 10-50 MB for output files

## Biological Interpretation

### LN Functional Classes

**Uniglomerular LNs** (1 glomerulus):
- Rare, highly specific
- May mediate direct PN-PN interactions within one glomerulus
- Examples: Some patchy LNs in literature

**Oligoglomerular LNs** (2-3 glomeruli):
- Pairwise or triplet interactions
- May create specific cross-talk between related odor channels
- Examples: Keystone LNs in certain pathways

**Multiglomerular LNs** (4-10 glomeruli):
- Local processing across related glomeruli
- May group functionally related channels
- Examples: Many patchy and keystone LNs

**Broad LNs** (11+ glomeruli):
- Global gain control and normalization
- Provide widespread lateral inhibition
- Examples: Broad LNs in Drosophila literature (~60-70% of LNs)

### Connection Directions

**Bidirectional** (most common):
- LN receives input from PNs in glomerulus A
- LN sends output to PNs in glomerulus A
- Suggests local feedback/gain control

**Input only**:
- LN receives from PNs but doesn't project back
- May integrate information without local feedback

**Output only**:
- LN projects to PNs without receiving input
- May relay information from other sources (e.g., other LNs)

## Citations & References

- Olsen SR, Wilson RI. (2008) "Lateral presynaptic inhibition mediates gain control in an olfactory circuit." *Nature* 452:956-960.
- Chou YH, et al. (2010) "Diversity and wiring variability of olfactory local interneurons in the Drosophila antennal lobe." *Nat Neurosci* 13:439-449.
- Schlegel P, et al. (2021) "Whole-brain annotation and multi-connectome cell typing of Drosophila." *bioRxiv*

## Support

For issues or questions:
- Check validation output for warnings
- Run `test_ln_mapping.py` to diagnose problems
- Review example workflows above
- Ensure FlyWire data is up-to-date

---

**Last Updated**: 2025-11-10
