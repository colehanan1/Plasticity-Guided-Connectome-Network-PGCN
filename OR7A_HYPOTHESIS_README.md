# Or7a Learning Veto Hypothesis Testing

## Quick Start

```bash
# Activate environment
conda activate PGCN

# Run hypothesis tests
python scripts/test_or7a_veto.py \
  --data-dir data/flywire \
  --output-dir results/or7a_hypothesis
```

## What This Tests

This script tests **3 specific hypotheses** about Or7a receptor's role in blocking learning:

### Hypothesis 1: Or7a Benzaldehyde Selectivity
**Prediction**: Or7a shows >3x stronger response to benzaldehyde vs hexanol

**Test**: Compare DoOR receptor activation profiles
- **SUPPORTS**: Selectivity ratio > 3.0
- **CONTRADICTS**: Selectivity ratio ≤ 3.0

### Hypothesis 2: Or7a No Lateral Inhibition
**Prediction**: Or7a (DL5 glomerulus) has zero lateral inhibition to DM glomeruli

**Test**: Count LN-mediated connections from DL5 → DM1-6
- **SUPPORTS**: 0 cross-glomerular LNs found
- **CONTRADICTS**: >0 cross-glomerular LNs found

### Hypothesis 3: Shared Receptor Cross-Learning
**Prediction**: Cross-learning explained by receptor responding strongly to both odorants

**Test**: Identify receptors with >0.5 response to both benzaldehyde and hexanol
- **SUPPORTS**: At least one shared receptor found (expected: Or67b)
- **CONTRADICTS**: No shared receptors found

## Expected Outputs

### CSV Files (3)
1. **`hypothesis1_or7a_selectivity.csv`**
   - Or7a response to benzaldehyde, hexanol, 2-heptanone
   - Selectivity ratio and hypothesis support

2. **`hypothesis2_lateral_connectivity.csv`**
   - DL5→LN and LN→DM connection counts
   - Number of cross-glomerular LNs
   - Hypothesis support

3. **`hypothesis3_shared_receptors.csv`**
   - Receptors strongly responding to both odorants
   - Response balance scores
   - Ranked by mean response

### Figures (2)
1. **`or7a_receptor_profiles.png`**
   - Panel A: Or7a response profile across 3 odorants
   - Panel B: Shared receptor comparison (benzaldehyde vs hexanol)

2. **`or7a_connectivity_summary.png`**
   - Bar chart showing DL5→DM connectivity pathway
   - Highlights cross-glomerular LNs (if any)

## Key Features

- ✅ **DoOR Integration**: Uses DoOR database with hardcoded fallback
- ✅ **FlyWire Connectivity**: Analyzes real synaptic connections
- ✅ **Clear Pass/Fail**: Each hypothesis explicitly SUPPORTED or CONTRADICTED
- ✅ **Fast Execution**: Completes in <30 seconds
- ✅ **Publication Ready**: 300 DPI figures with proper formatting

## DoOR Toolkit Integration

For enhanced functionality with real odorant response data:

### Install DoOR Toolkit

```bash
# Navigate to door-toolkit repository
cd ~/Documents/cole/VSCode/door-python-toolkit

# Install in editable mode
pip install -e .

# Test installation
python scripts/test_door_integration.py
```

See **`DOOR_TOOLKIT_SETUP.md`** for complete installation and usage guide.

### Using PGCNDoorIntegration

```python
from door_integration.pgcn_door import PGCNDoorIntegration

pgcn_door = PGCNDoorIntegration()

# Get Or7a selectivity
selectivity = pgcn_door.calculate_selectivity('Or7a', 'benzaldehyde', 'hexanol')
print(f"Or7a selectivity: {selectivity:.2f}x")

# Find shared receptors
shared = pgcn_door.find_shared_receptors('benzaldehyde', 'hexanol', threshold=0.5)
print(f"Cross-learning via: {shared}")

# Map odorant to glomeruli
glomeruli = pgcn_door.map_odorant_to_glomeruli('benzaldehyde', threshold=0.3)
print(f"Active glomeruli: {glomeruli}")
```

## Biological Context

### Or7a Learning Veto Mechanism
- **Or7a receptor** responds strongly to benzaldehyde (geranyl acetate)
- When activated >40-50%, **blocks aversive learning** in Drosophila
- Hypothesis: Or7a activation signals "safe odor" → no need to learn

### Cross-Learning Puzzle
- Training with **benzaldehyde** produces response to **hexanol**
- This cross-learning needs explanation
- Hypothesis: Shared receptor (Or67b?) activated by both odorants

### Lateral Inhibition Question
- Does Or7a (DL5) **inhibit nearby glomeruli** (DM1-6)?
- If yes: Lateral inhibition could explain veto effect
- If no: Veto must work through different mechanism (direct KC/MBON targeting)

## Expected Results

### If Hypothesis Fully Supported (3/3)
- Or7a shows strong benzaldehyde selectivity ✅
- Or7a has zero lateral inhibition to DM glomeruli ✅
- Or67b identified as shared strong receptor ✅

**Interpretation**: Or7a learning veto works through:
1. Selective benzaldehyde detection (not hexanol)
2. Direct downstream targeting (not lateral inhibition)
3. Cross-learning via Or67b co-activation

### If Lateral Inhibition Found (2/3)
- Or7a shows strong benzaldehyde selectivity ✅
- Or7a HAS lateral inhibition to DM glomeruli ❌
- Or67b identified as shared strong receptor ✅

**Interpretation**: Or7a may also use lateral inhibition mechanism

## Comparison with Comprehensive Analysis

This repository contains **two Or7a analysis scripts**:

| Feature | `test_or7a_veto.py` | `analyze_or7a_veto_hypothesis.py` |
|---------|---------------------|-----------------------------------|
| **Purpose** | Focused hypothesis testing | Comprehensive analysis |
| **Hypotheses** | 3 specific tests | 4 broad analyses |
| **Lines of code** | ~400 | ~800 |
| **Runtime** | <30 seconds | 5-10 minutes |
| **Outputs** | 2 figures, 3 CSVs | 3 figures, 4 CSVs |
| **Best for** | Quick validation | Publication figures |

**Recommendation**: Run `test_or7a_veto.py` first for quick hypothesis validation, then use comprehensive analysis for detailed investigation.

## Integration with Existing Scripts

This script complements the LN/PN connectivity pipeline:

```bash
# Step 1: Map all LN-glomerulus associations
python scripts/map_ln_glomeruli.py --data-dir data/flywire --output-dir results/ln_mapping

# Step 2: Analyze cross-glomerular connectivity
python scripts/analyze_ln_pn_connectivity.py --data-dir data/flywire --output-dir results/ln_pn_analysis

# Step 3: Test Or7a hypothesis (THIS SCRIPT)
python scripts/test_or7a_veto.py --data-dir data/flywire --output-dir results/or7a_hypothesis
```

## Troubleshooting

### Issue: "DoOR toolkit not available"
**Expected behavior**: Script uses hardcoded DoOR values

**To enable DoOR toolkit**:
```bash
pip install door-python-toolkit
# Or if in separate repo: pip install -e /path/to/door-python-toolkit
```

### Issue: Few/No neurons found
**Check data files**:
```bash
ls -lh data/flywire/*.csv.gz
```

Required files:
- `consolidated_cell_types.csv.gz`
- `connections_princeton.csv.gz`
- `classification.csv.gz` (optional but recommended)
- `processed_labels.csv.gz` (optional but recommended)

### Issue: Hypothesis 2 contradicted (lateral inhibition found)
**This is scientifically interesting!** If DL5→DM lateral inhibition exists, it suggests:
- Or7a veto may use lateral inhibition mechanism
- Cross-glomerular inhibition pathway discovered
- Warrants further investigation

Check `hypothesis2_lateral_connectivity.csv` for details on which LNs mediate the pathway.

## Related Documentation

- **LN Mapping Guide**: `docs/LN_GLOMERULUS_MAPPING_GUIDE.md`
- **LN/PN Analysis Guide**: `docs/LN_PN_CONNECTIVITY_ANALYSIS_GUIDE.md`
- **LN Mapping README**: `LN_MAPPING_README.md`
- **Comprehensive Or7a Analysis**: `scripts/analyze_or7a_veto_hypothesis.py`

## References

- **Or7a Learning Veto**: Felsenberg et al. (2018) "Integration of Parallel Opposing Memories Underlies Memory Extinction" *Cell* 175:709-722
- **Benzaldehyde Selectivity**: DoOR Database (Münch & Galizia, 2016)
- **FlyWire Connectome**: Dorkenwald et al. (2023) "Neuronal wiring diagram of an adult brain" *bioRxiv*

---

**Created**: 2025-11-10
**Status**: ✅ Ready to Run
**Branch**: `claude/analyze-ln-pn-connectivity-011CUzsMbA7koHMNNbCxibZt`
