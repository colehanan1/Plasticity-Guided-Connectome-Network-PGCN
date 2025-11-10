# Local Neuron (LN) to Glomerulus Mapping

## Quick Start

```bash
# Activate conda environment
conda activate PGCN

# Run LN-glomerulus mapping
python scripts/map_ln_glomeruli.py \
  --data-dir data/flywire \
  --output-dir results/ln_mapping \
  --min-synapses 3

# Validate outputs
python scripts/test_ln_mapping.py
```

## What This Does

Maps Local Neurons to glomeruli by inferring associations from connectivity patterns with Projection Neurons.

**Key Innovation**: Since LNs don't have glomerulus labels in their metadata (unlike PNs), we infer them from:
- **PN→LN connections**: Which PN glomerulus provides INPUT
- **LN→PN connections**: Which PN glomerulus receives OUTPUT

## Outputs

Four CSV files in `results/ln_mapping/`:

1. **`ln_glomerulus_associations.csv`** - Complete LN-glomerulus mapping
2. **`ln_primary_glomerulus.csv`** - Primary glomerulus per LN
3. **`glomerulus_ln_summary.csv`** - Statistics per glomerulus
4. **`ln_categories.csv`** - LN categorization (uni/oligo/multi/broad)

## Expected Results

- **LNs mapped**: 800-1,000 Local Neurons
- **Glomeruli**: 30-60 unique glomeruli
- **Categories**: ~60-70% broad, ~20-30% multiglomerular, ~10% oligo/uniglomerular

## Example: Find LNs for DL5 Glomerulus

```python
import pandas as pd

df = pd.read_csv('results/ln_mapping/ln_glomerulus_associations.csv')
dl5_lns = df[df['glomerulus'] == 'DL5']

print(f"LNs associated with DL5: {dl5_lns['ln_id'].nunique()}")
print(dl5_lns.nlargest(10, 'total_synapses'))
```

## Documentation

- **Full Guide**: `docs/LN_GLOMERULUS_MAPPING_GUIDE.md`
- **Validation Script**: `scripts/test_ln_mapping.py`
- **Main Script**: `scripts/map_ln_glomeruli.py`

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-synapses` | 3 | Minimum synapses for a connection |
| `--neuropil` | None | Filter to specific neuropil (e.g., "AL") |

## LN Categories

- **Uniglomerular** (1 glom): Highly specific
- **Oligoglomerular** (2-3 gloms): Pairwise interactions
- **Multiglomerular** (4-10 gloms): Local processing
- **Broad** (11+ gloms): Global gain control

## Integration with Existing Analysis

This complements `analyze_ln_pn_connectivity.py`:
- **map_ln_glomeruli.py**: Focuses on LN→glomerulus mapping (this script)
- **analyze_ln_pn_connectivity.py**: Analyzes cross-glomerular pathways using those mappings

Run both for comprehensive LN analysis!

---

**Created**: 2025-11-10
**Status**: ✅ Production Ready
