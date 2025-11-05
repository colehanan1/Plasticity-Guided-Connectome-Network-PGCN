# OR7a Connectomics - Quick Reference Card

## 🚀 Quick Start Commands

```bash
# System 1: Output Mapping (41 OR7a → targets)
python scripts/map_or7a_outputs.py --data-source local

# System 2: Complete Pathway (OR7a → PN → KC → MBON → Behavior)
python scripts/map_or7a_complete_pathway.py --data-source local

# Quick demos (test with smaller datasets)
python scripts/test_or7a_mapping_demo.py
python scripts/test_pathway_mapping_demo.py
```

## 📊 Output Files Location

```
results/or7a_outputs/              # Output mapping results
results/or7a_complete_pathway/     # Pathway tracing results
```

## 🔑 Key Findings (Expected)

| Circuit Level | Neurons | Convergence | Synapses | Significance |
|--------------|---------|-------------|----------|--------------|
| OR7a ORN | 41 | - | - | Benzaldehyde sensors |
| DL5_PN | 2 | **0.05** | 40-60 | **CRITICAL BOTTLENECK** |
| KC | 200-500 | 100-250 | 5-10 | Sparse odor encoding |
| MBON | 20-50 | 0.1 | 10-30 | Behavioral control |
| Behavior | 10-100 | Variable | 5-15 | Motor output |

## 🎯 Experimental Targets

### Priority 1: DL5 Projection Neurons ⭐⭐⭐
- **Count**: 2 neurons
- **Root IDs**: 720575940639080700, 720575940617207200
- **Effect**: Complete OR7a pathway block
- **Specificity**: Very high

### Priority 2: High-Connectivity KCs ⭐⭐
- **Count**: Top 20 from `target_priorities.csv`
- **Effect**: Partial learning deficit
- **Specificity**: Medium

### Priority 3: Behavioral MBONs ⭐
- **Count**: 5-10 specific MBONs
- **Effect**: Behavioral modulation
- **Specificity**: Variable by type

## 📁 Important Files

| File | Description |
|------|-------------|
| `or7a_output_targets_long.csv` | All OR7a connections |
| `or7a_output_targets_wide.csv` | Top 20 targets per OR7a |
| `or7a_complete_pathway.csv` | Multi-level circuit |
| `pathway_summary_by_level.csv` | Level statistics |
| `target_priorities.csv` | **Ranked experimental targets** |
| `pathway_summary_bottlenecks.csv` | **Circuit chokepoints** |

## 🔍 Quick Analysis Code

```python
import pandas as pd

# Load pathway
pathway = pd.read_csv('results/or7a_complete_pathway/or7a_complete_pathway.csv')

# Check bottleneck
pn = pathway[pathway['target_category']=='PN']['post_root_id'].nunique()
kc = pathway[pathway['target_category']=='KC']['post_root_id'].nunique()
print(f"OR7a(41) → PN({pn}) → KC({kc})")

# Get top targets
targets = pd.read_csv('results/or7a_complete_pathway/target_priorities.csv')
print(targets[targets['category']=='PN'].head(3))
```

## 🛠️ Common Options

```bash
# Different synapse thresholds
--min-synapses 1    # Permissive (more connections)
--min-synapses 10   # Stringent (fewer, stronger)

# Trace fewer levels (faster)
--max-levels 3      # Stop at MBON level

# Custom output directory
--output-dir results/my_analysis/
```

## 📚 Documentation

- **Complete Guide**: [`OR7A_CONNECTOMICS_COMPLETE.md`](OR7A_CONNECTOMICS_COMPLETE.md)
- **Output Mapping**: [`docs/OR7A_OUTPUT_MAPPING_GUIDE.md`](docs/OR7A_OUTPUT_MAPPING_GUIDE.md)
- **Pathway Tracing**: [`docs/OR7A_COMPLETE_PATHWAY_GUIDE.md`](docs/OR7A_COMPLETE_PATHWAY_GUIDE.md)

## ⚡ Performance

- **Output mapping**: ~20 seconds, ~3 GB RAM
- **Pathway tracing**: ~60 seconds, ~3 GB RAM
- **Total outputs**: 15 files, ~5 MB

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No PNs found" | Use `--min-synapses 1` |
| "Too few KCs" | Lower threshold for level 2 |
| "Memory error" | Close apps, use ≥16GB RAM |
| "No MBON connections" | Verify KC level completed |

## 🎓 What This Enables

1. ✅ Map complete OR7a circuits (receptors → behavior)
2. ✅ Identify 2 critical DL5 PNs (bottleneck)
3. ✅ Prioritize experimental targets
4. ✅ Predict suppression effects
5. ✅ Initialize PGCN learning models

---

**Need help?** See [`OR7A_CONNECTOMICS_COMPLETE.md`](OR7A_CONNECTOMICS_COMPLETE.md) for detailed workflows.
