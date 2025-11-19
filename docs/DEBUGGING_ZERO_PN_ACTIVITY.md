# Debugging Zero PN Activity Issue

**Problem**: All experimental odors return 0 active projection neurons, preventing model training.

**Symptom**:
```
DoOR coverage statistics:
  ✗ hexanol              :   0 active PNs
  ✗ ethyl_butyrate       :   0 active PNs
  ✗ benzaldehyde         :   0 active PNs
  ...
Mean active PNs per trial: 0.0
```

---

## Quick Diagnosis

**Step 1**: Pull the latest fix
```bash
git pull origin claude/connectome-constrained-behavior-prediction-014UV3FWTFdXYAttqMaTBEoh
```

**Step 2**: Run diagnostic scripts
```bash
cd ~/Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-
python diagnose_door_integration.py
python test_orn_pn_mapping.py
```

The scripts will identify which of these issues you have:

---

## Likely Root Causes

### 1. **Malformed DoOR Data from GitHub** (Current Issue - Fixed in Latest Code)

**Symptoms**:
- `test_orn_pn_mapping.py` shows "ORN types (columns): 0"
- DoOR odor names look like: `'sfr;0.0627144154948233;0.06972846128059;...'`
- Diagnostic shows "Matched: 0" ORN types

**Root Cause**: The `door_response_matrix.csv` downloaded from GitHub has InChIKeys with concatenated data instead of proper columns. You need to use door-toolkit formatted data instead.

**Fix**: Pull latest code - it will automatically detect and use your door-toolkit data:
```bash
git pull origin claude/connectome-constrained-behavior-prediction-014UV3FWTFdXYAttqMaTBEoh
```

The updated code now:
- Checks for door-toolkit formatted data (`response_matrix_norm.csv`) before GitHub download
- Automatically finds your data at `data/door_cache/response_matrix_norm.csv`
- Validates DoOR structure (detects 0 columns) and provides helpful error messages

**Verification**: Run `test_orn_pn_mapping.py` and check for:
```
[1/4] DoOR Data Structure
  Odorants (rows): 693
  ORN types (columns): 110  ← FIXED! (was 0)

[3/4] Glomerulus→ORN Mapping
  Matched: 35  ← FIXED! (was 0)
```

**See**: [docs/DOOR_MALFORMED_DATA_FIX.md](DOOR_MALFORMED_DATA_FIX.md) for complete details.

---

### 2. **Haven't Pulled Latest Odor Name Fix**

**Symptoms**:
- `diagnose_door_integration.py` shows "ODOR_NAME_MAP NOT FOUND"
- All odor name resolutions fail

**Fix**:
```bash
git pull origin claude/connectome-constrained-behavior-prediction-014UV3FWTFdXYAttqMaTBEoh
```

**Verification**: Run `diagnose_door_integration.py` and check for:
```
✓ ODOR_NAME_MAP found with 7 mappings
  Mappings:
    'hexanol' → '1-hexanol'
    'ethyl_butyrate' → 'ethyl butyrate'
    ...
```

---

### 3. **PN→Glomerulus Mapping Missing**

**Symptoms**:
- `test_orn_pn_mapping.py` shows "Total PNs mapped: 0"
- "NO PNs MAPPED TO GLOMERULI!"

**Root Cause**: The file `nodes.parquet` is missing or doesn't contain PN glomerulus assignments.

**Fix Options**:

**Option A**: If you have FlyWire data, regenerate nodes.parquet:
```bash
python src/scripts/extract_flywire_data.py --cache_dir data/cache
```

**Option B**: Create a minimal mock nodes.parquet for testing:
```python
import pandas as pd
import numpy as np

# Create mock PN data with glomerulus assignments
pn_data = []
glomeruli = ['DA1', 'DA2', 'DL3', 'DL5', 'DM1', 'DM2', 'DM5', 'VA1d', 'VA2', 'VC2']
for i in range(150):
    pn_data.append({
        'type': 'PN',
        'glomerulus': glomeruli[i % len(glomeruli)],
        'neuron_id': f'PN_{i}',
    })

df = pd.DataFrame(pn_data)
df.to_parquet('data/cache/nodes.parquet', index=True)
print(f"Created mock nodes.parquet with {len(df)} PNs")
```

**Verification**: Run `test_orn_pn_mapping.py` and check for:
```
[2/4] PN→Glomerulus Mapping
  Total PNs mapped: 150
  Glomeruli represented: ['DA1', 'DA2', 'DL3', ...]
```

---

### 4. **DoOR Column Names Don't Match ORN Names**

**Symptoms**:
- `test_orn_pn_mapping.py` shows "Matched: 0" ORN types
- "NO ORN TYPES MATCH!"
- Lists expected vs. actual column names

**Root Cause**: DoOR database column names don't match `GLOMERULUS_TO_ORN_MAPPING` expectations.

**Example Mismatch**:
```
Expected: Or7a, Or42b, Or59b
Actual:   or7a, or42b, or59b  (lowercase)
OR
Actual:   7a, 42b, 59b  (missing "Or" prefix)
```

**Fix**: Update `GLOMERULUS_TO_ORN_MAPPING` in `src/pgcn/data/door_integration.py` to match actual DoOR column names.

First, check what DoOR actually has:
```python
from pathlib import Path
import pandas as pd

cache_dir = Path("data/cache")
door_csv = cache_dir / "door_response_matrix.csv"

if door_csv.exists():
    door_data = pd.read_csv(door_csv, index_col=0)
    print("DoOR columns (ORN types):")
    for col in sorted(door_data.columns[:20]):
        print(f"  {col}")
```

Then update the mapping. For example, if DoOR uses lowercase:
```python
# In src/pgcn/data/door_integration.py, around line 50
GLOMERULUS_TO_ORN_MAPPING = {
    'DA1': 'or67d',  # Changed from 'Or67d' to 'or67d'
    'DA2': 'or56a',  # Changed from 'Or56a' to 'or56a'
    # ... update all mappings
}
```

Or if DoOR uses a different naming scheme entirely, you may need to map:
```python
GLOMERULUS_TO_ORN_MAPPING = {
    'DA1': '67d',  # If DoOR columns are just the number part
    'DA2': '56a',
    # ...
}
```

**Verification**: Run `test_orn_pn_mapping.py` and check for:
```
Matched: 35  (or however many glomeruli you have)
```

---

### 5. **Wrong Cache Directory Path**

**Symptoms**:
- DoOR data files exist but aren't being found
- "No cache directory found" message

**Root Cause**: `train_ccbpn.py` is looking in the wrong location.

**Common Paths**:
```
Correct:   ~/Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-/data/cache
Your code: data/cache (relative path)
```

**Fix A**: Use absolute path in training command:
```bash
python src/scripts/train_ccbpn.py \
    --cache_dir ~/Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-/data/cache \
    ...
```

**Fix B**: Create symlink:
```bash
cd ~/Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-
ln -s data/door_cache data/cache
```

**Fix C**: Copy DoOR files to expected location:
```bash
mkdir -p data/cache
cp data/door_cache/* data/cache/
```

---

### 6. **DoOR Data Format Mismatch** (Deprecated - See Issue #1)

**Note**: This issue is now handled automatically by the fix in Issue #1. The code will detect and use door-toolkit formatted data.

**Symptoms**:
- You have `response_matrix_norm.parquet` but code expects `door_response_matrix.csv`
- You have `odorant_index.csv`, `odor_metadata.parquet` from door-toolkit

**Root Cause**: You're using door-toolkit's data format, but `DoORIntegration` expects R-based DoOR format.

**Fix**: Convert parquet to CSV format that `DoORIntegration` expects:

```python
import pandas as pd
from pathlib import Path

# Load door-toolkit parquet
cache_dir = Path("data/door_cache")
response_matrix = pd.read_parquet(cache_dir / "response_matrix_norm.parquet")

# Save in format DoORIntegration expects
output_path = Path("data/cache") / "door_response_matrix.csv"
output_path.parent.mkdir(exist_ok=True)
response_matrix.to_csv(output_path)

print(f"Converted DoOR data saved to: {output_path}")
print(f"  Shape: {response_matrix.shape}")
print(f"  Odorants: {len(response_matrix)}")
print(f"  ORN types: {len(response_matrix.columns)}")
```

**Verification**: Check that file exists:
```bash
ls -lh data/cache/door_response_matrix.csv
```

---

## Step-by-Step Troubleshooting

### Step 1: Run Diagnostics

```bash
python diagnose_door_integration.py > door_diag.txt 2>&1
python test_orn_pn_mapping.py > orn_mapping_diag.txt 2>&1
```

### Step 2: Identify Issue from Output

Look for these key lines:

**In `door_diag.txt`**:
```
✗ ODOR_NAME_MAP NOT FOUND          → Need to pull latest code
✗ No cache directory found         → Wrong cache_dir path
✗ No DoOR data files found         → Missing DoOR database
✗ 'hexanol' → NOT FOUND            → Odor name resolution failing
✗ hexanol: 0 active PNs (ZERO!)    → PN mapping issue
```

**In `orn_mapping_diag.txt`**:
```
Total PNs mapped: 0                → nodes.parquet missing
Matched: 0                         → ORN column name mismatch
0 ORN types matched                → GLOMERULUS_TO_ORN_MAPPING wrong
```

### Step 3: Apply Appropriate Fix

Based on the error messages above, apply the corresponding fix from sections 1-5.

### Step 4: Verify Fix

```bash
python diagnose_door_integration.py
```

Expected SUCCESS output:
```
✓ ODOR_NAME_MAP found with 7 mappings
✓ DoORIntegration initialized successfully
  DoOR odorants loaded: 693
  PNs mapped to glomeruli: 150

✓ 'hexanol' → '1-hexanol'
✓ 'ethyl_butyrate' → 'ethyl butyrate'
...

✓ hexanol              :  17 active PNs
✓ ethyl_butyrate       :  14 active PNs
✓ benzaldehyde         :  12 active PNs
...

✅ All odors produce non-zero PN activity - DoOR integration is working!
```

### Step 5: Retrain

```bash
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --epochs 100 \
    --cache_dir data/cache \
    --behavioral_data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --dataset_mapping configs/dataset_to_odor_mapping.yaml \
    --output_dir results/ccbpn_fixed
```

Expected initialization output:
```
DoOR Integration Validation
============================================================
  ✓ hexanol              →  17 active PNs (DoOR: '1-hexanol')
  ✓ ethyl_butyrate       →  14 active PNs (DoOR: 'ethyl butyrate')
  ...
✅ All experimental odors successfully mapped to DoOR!

DoOR coverage statistics:
  ✓ hexanol              :  17 active PNs
  ...
Mean active PNs per trial: 13.0
```

---

## Emergency Bypass (For Testing Only)

If you can't fix the DoOR integration immediately but want to test other parts of the code, you can temporarily use synthetic patterns:

**In `src/scripts/train_ccbpn.py`, around line 307**, replace:
```python
odor_sequence = door.create_odor_sequence(...)
```

With:
```python
# TEMPORARY: Use synthetic patterns for testing
n_active_pns = np.random.randint(10, 30)
active_pns = np.random.choice(n_pn, size=n_active_pns, replace=False)
odor_sequence = np.zeros((sequence_length, n_pn))
odor_sequence[:40, active_pns] = 1.0  # Odor ON for first 40 timesteps
```

**WARNING**: This defeats the purpose of using DoOR! Only use for debugging other parts of the pipeline.

---

## Getting Help

If diagnostics don't identify the issue, provide:

1. **Output from both diagnostic scripts**:
   ```bash
   python diagnose_door_integration.py > door_diag.txt 2>&1
   python test_orn_pn_mapping.py > orn_mapping_diag.txt 2>&1
   ```

2. **Directory structure**:
   ```bash
   tree -L 3 data/
   ```

3. **DoOR file info**:
   ```bash
   ls -lh data/cache/
   ls -lh data/door_cache/
   ```

4. **First 10 lines of training output**:
   ```bash
   python src/scripts/train_ccbpn.py ... 2>&1 | head -50
   ```

This information will help pinpoint the exact issue.
