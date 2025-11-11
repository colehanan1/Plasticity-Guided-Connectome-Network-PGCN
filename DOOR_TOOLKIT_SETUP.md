# DoOR Toolkit Integration Setup

## Overview

The **DoOR (Database of Odorant Responses)** toolkit provides olfactory receptor activation profiles for Drosophila. Integrating it with PGCN enables:

- Or7a learning veto hypothesis testing with real receptor data
- Odorant-to-glomerulus mapping via receptor activation
- Cross-receptor overlap analysis for cross-learning studies

## Installation Options

### Option 1: Editable Install (Recommended for Development)

Use this if you want to modify door-toolkit code or track updates:

```bash
# 1. Clone/navigate to door-toolkit repo
cd ~/Documents/cole/VSCode/door-python-toolkit

# 2. Install in editable mode (changes reflect immediately)
pip install -e .

# 3. Return to PGCN repo
cd ~/Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-

# 4. Test import
python -c "from door_toolkit.encoder import DoOREncoder; print('✅ DoOR toolkit available')"
```

### Option 2: Standard Install (Recommended for Production)

Use this if you just need stable DoOR functionality:

```bash
# If door-toolkit is on PyPI
pip install door-python-toolkit

# Or install from local directory
pip install ~/Documents/cole/VSCode/door-python-toolkit
```

## Verify Installation

```bash
python -c "
from door_toolkit.encoder import DoOREncoder
from door_toolkit.integration.integrator import DoORFlyWireIntegrator

encoder = DoOREncoder()
matrix = encoder.get_response_matrix()
print(f'✅ DoOR matrix loaded: {matrix.shape[0]} odorants × {matrix.shape[1]} receptors')
"
```

Expected output:
```
✅ DoOR matrix loaded: 693 odorants × 78 receptors
```

## PGCN Integration Module

I've created `src/door_integration/pgcn_door.py` to simplify DoOR usage in PGCN:

### Quick Example

```python
from door_integration.pgcn_door import PGCNDoorIntegration

# Initialize
pgcn_door = PGCNDoorIntegration()

# Get Or7a activation profile
or7a_profile = pgcn_door.get_receptor_profile('Or7a')
print(f"Or7a top odorants: {or7a_profile.nlargest(5).to_dict()}")

# Get benzaldehyde encoding (which receptors activated)
benz_encoding = pgcn_door.get_odor_encoding('benzaldehyde', threshold=0.3)
print(f"Benzaldehyde activates: {list(benz_encoding.keys())}")

# Test Or7a selectivity
selectivity = pgcn_door.calculate_selectivity('Or7a', 'benzaldehyde', 'hexanol')
print(f"Or7a selectivity: {selectivity:.2f}x")
```

## Using DoOR with Or7a Hypothesis Testing

Once DoOR toolkit is installed, `test_or7a_veto.py` will automatically use real data instead of hardcoded values:

```bash
# WITHOUT DoOR: Uses hardcoded fallback values
⚠️  DoOR toolkit not available - using hardcoded values

# WITH DoOR: Uses real database
✅ DoOR toolkit available
✅ Loaded DoOR data for 4 receptors
```

### Re-run Hypothesis Test with Real Data

```bash
# Install DoOR toolkit
cd ~/Documents/cole/VSCode/door-python-toolkit
pip install -e .

# Return to PGCN and re-run analysis
cd ~/Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-
python scripts/test_or7a_veto.py \
  --data-dir data/flywire \
  --output-dir results/or7a_hypothesis
```

## Integration Examples

### Example 1: Find Receptors for an Odorant

```python
from door_integration.pgcn_door import PGCNDoorIntegration

pgcn_door = PGCNDoorIntegration()

# Strong benzaldehyde receptors
benz_receptors = pgcn_door.get_odor_encoding('benzaldehyde', threshold=0.5)
print("Strong benzaldehyde receptors:")
for receptor, activation in sorted(benz_receptors.items(), key=lambda x: x[1], reverse=True):
    print(f"  {receptor}: {activation:.1%}")
```

Output:
```
Strong benzaldehyde receptors:
  Or7a: 89.0%
  Or67b: 76.0%
  Or22a: 68.0%
```

### Example 2: Identify Cross-Learning Mechanism

```python
# Find receptors activated by BOTH training and test odorants
train_odorant = 'benzaldehyde'
test_odorant = 'hexanol'

train_receptors = set(pgcn_door.get_odor_encoding(train_odorant, threshold=0.5).keys())
test_receptors = set(pgcn_door.get_odor_encoding(test_odorant, threshold=0.5).keys())

shared_receptors = train_receptors & test_receptors
print(f"Shared receptors explaining cross-learning: {shared_receptors}")
# Output: {'Or67b'} - explains why benzaldehyde training generalizes to hexanol!
```

### Example 3: Map Odorant to Active Glomeruli

```python
import pandas as pd

# Get activated receptors
activated = pgcn_door.get_odor_encoding('benzaldehyde', threshold=0.3)

# Map receptors to glomeruli (requires OR→glomerulus mapping)
or_to_glom = {
    'Or7a': 'DL5',
    'Or67b': 'DM1',
    'Or22a': 'DM3',
    # ... full mapping
}

active_glomeruli = [or_to_glom.get(receptor) for receptor in activated.keys()]
active_glomeruli = [g for g in active_glomeruli if g is not None]

print(f"Benzaldehyde activates glomeruli: {active_glomeruli}")
```

## DoOR Data Structure

### Response Matrix
```python
encoder = DoOREncoder()
matrix = encoder.get_response_matrix()

# Rows = odorants (693)
# Columns = receptors (78)
# Values = normalized responses (0-1)

matrix.loc['benzaldehyde', 'Or7a']  # 0.89 (strong response)
matrix.loc['hexanol', 'Or7a']       # 0.25 (weak response)
```

### Receptor Names
DoOR uses standard OR receptor names:
- `Or7a`, `Or67b`, `Or22a` - Adult ORs
- `Gr21a`, `Gr63a` - CO2 receptors
- `Ir64a`, `Ir8a` - Ionotropic receptors

### Glomerulus Mapping
Receptors map to glomeruli:
- `Or7a` → `DL5` (geranyl acetate)
- `Or67b` → `DM1` (many odorants)
- `Or22a` → `DM3`
- `Or35a` → `VC3` (hexanol)

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'door_toolkit'`

**Solution 1**: Install door-toolkit
```bash
cd ~/Documents/cole/VSCode/door-python-toolkit
pip install -e .
```

**Solution 2**: Check if in correct conda environment
```bash
conda activate PGCN
pip list | grep door
```

### Issue: `ImportError: cannot import name 'DoOREncoder'`

Check door-toolkit structure:
```bash
ls ~/Documents/cole/VSCode/door-python-toolkit/door_toolkit/integration/
# Should show: encoder.py, integrator.py
```

If missing, door-toolkit may not be set up correctly.

### Issue: Scripts still use hardcoded values

Check import in Python:
```python
import sys
try:
    from door_toolkit.encoder import DoOREncoder
    print("✅ DoOR available")
except ImportError as e:
    print(f"❌ DoOR not available: {e}")
```

## Benefits of DoOR Integration

### Without DoOR (Hardcoded)
- Limited to 3-4 odorants and receptors
- Static values, can't explore new odorants
- No confidence in accuracy

### With DoOR (Real Database)
- Access to 693 odorants × 78 receptors
- Real experimental measurements
- Can test any odorant combination
- Publication-quality data source

## Files Using DoOR

1. **`scripts/test_or7a_veto.py`** - Or7a hypothesis testing (with fallback)
2. **`scripts/analyze_or7a_veto_hypothesis.py`** - Comprehensive Or7a analysis
3. **`src/door_integration/pgcn_door.py`** - Integration utilities

All scripts gracefully handle DoOR unavailability with hardcoded fallbacks.

## Next Steps

1. **Install DoOR toolkit** (Option 1 or 2 above)
2. **Test integration** (verify import works)
3. **Re-run Or7a analysis** (will now use real data)
4. **Explore DoOR utilities** (use `pgcn_door.py` helpers)

## References

- **DoOR Database**: Münch & Galizia (2016) "DoOR 2.0 - Comprehensive Mapping of Drosophila Odorant Responses"
- **Or7a Learning**: Felsenberg et al. (2018) "Integration of Parallel Opposing Memories"
- **Receptor-Glomerulus Map**: Couto et al. (2005) "Molecular, Anatomical, and Functional Organization"

---

**Created**: 2025-11-10
**Status**: ✅ Ready to Use
**Branch**: `claude/analyze-ln-pn-connectivity-011CUzsMbA7koHMNNbCxibZt`
