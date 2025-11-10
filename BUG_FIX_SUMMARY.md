# Bug Fix Summary: PN Identification in Multi-ORN Pathway Script

## Problem

The multi-ORN pathway script ([scripts/map_multi_orn_pathways.py](scripts/map_multi_orn_pathways.py)) was **incorrectly identifying 0 PNs for Or7a** (and potentially incorrect counts for other ORNs), while the OR7a-specific script correctly identified 2 PNs.

### Symptoms
- Running `map_or7a_complete_pathway.py`: **2 DL5 PNs** ✓
- Running `map_multi_orn_pathways.py`: **0 PNs for Or7a** ✗

## Root Cause

**Incorrect PN root IDs in the `ORNPathwayConfig`** for all ORN types.

The config had outdated/incorrect PN root IDs that:
1. Do not exist in the connections data
2. Do not exist in the cell types annotations
3. Were likely from an older data version or were incorrect from the start

### Example for Or7a:
- **Old (incorrect) IDs**: `720575940639080700`, `720575940617207200`
- **Correct IDs**: `720575940639080765`, `720575940617207185`
- **Cell type**: `DL5_adPN`

The filtering logic was actually **correct**, but the whitelist had the wrong IDs, so:
1. Pattern matching found PNs with cell_type containing "adPN"
2. Whitelist filtering attempted to narrow down to specific IDs
3. No neurons matched BOTH conditions → 0 PNs

## Solution

### 1. Created Test Script to Find Correct PNs ([tests/test_pathway_pn_counts.py](tests/test_pathway_pn_counts.py))

This script:
- Loads actual connection and cell type data
- Finds all PNs for each ORN using pattern matching
- Filters by glomerulus-specific patterns (DL5, DC2, DM1, VA1v, DM4)
- Reports correct PN root IDs

### 2. Updated PN Root IDs in Config

**Fixed [scripts/map_multi_orn_pathways.py](scripts/map_multi_orn_pathways.py) lines 1080-1133:**

```python
ORN_CONFIGS: Tuple[ORNPathwayConfig, ...] = (
    ORNPathwayConfig(
        name="Or7a",
        pn_root_ids=(720575940617207185, 720575940639080765),  # Fixed: correct DL5_adPN IDs
        ...
    ),
    ORNPathwayConfig(
        name="Or13a",
        pn_root_ids=(720575940631193052, 720575940627160322, 720575940630493818, 720575940616824588),
        ...
    ),
    ORNPathwayConfig(
        name="Or42b",
        pn_root_ids=(720575940619071005, 720575940630770042),
        ...
    ),
    ORNPathwayConfig(
        name="Or47b",
        pn_root_ids=(720575940629733626, 720575940623739076, 720575940620199962, 720575940621696747,
                     720575940628283560, 720575940630989354, 720575940625014928, 720575940633165025,
                     720575940629097922),
        ...
    ),
    ORNPathwayConfig(
        name="Or59b",
        pn_root_ids=(720575940615366055, 720575940623528925),
        ...
    ),
)
```

### 3. Created Regression Test ([tests/test_pn_counts_regression.py](tests/test_pn_counts_regression.py))

Automated test that:
- Runs the multi-ORN pathway script
- Verifies correct PN counts for all ORNs
- Compares Or7a results with the OR7a-specific script
- **Prevents future regressions**

## Results

### Before Fix:
```
Or7a:  0 PNs ✗
Or13a: ? PNs (incorrect)
Or42b: ? PNs (incorrect)
Or47b: ? PNs (incorrect)
Or59b: ? PNs (incorrect)
```

### After Fix:
```
✓ Or7a: 2 PNs (expected: 2)   ← DL5_adPN neurons
✓ Or13a: 4 PNs (expected: 4)  ← DC2_adPN neurons
✓ Or42b: 2 PNs (expected: 2)  ← DM1_lPN neurons
✓ Or47b: 9 PNs (expected: 9)  ← VA1v_adPN neurons
✓ Or59b: 2 PNs (expected: 2)  ← DM4_adPN neurons
```

### Test Results:
```bash
$ python tests/test_pn_counts_regression.py

======================================================================
PN Identification Regression Tests
======================================================================

PN Count Verification:
  ✓ Or7a: 2 PNs (expected 2)
  ✓ Or13a: 4 PNs (expected 4)
  ✓ Or42b: 2 PNs (expected 2)
  ✓ Or47b: 9 PNs (expected 9)
  ✓ Or59b: 2 PNs (expected 2)

✓ All PN counts match expected values!

OR7a-specific script: 2 PNs
Multi-ORN script:     2 PNs

✓ Both scripts agree: 2 PNs for Or7a

======================================================================
✓ ALL TESTS PASSED
======================================================================
```

## Files Modified

1. **[scripts/map_multi_orn_pathways.py](scripts/map_multi_orn_pathways.py)**
   - Lines 1088, 1098, 1108, 1118, 1130: Updated `pn_root_ids` with correct IDs

2. **[tests/test_pathway_pn_counts.py](tests/test_pathway_pn_counts.py)** (new)
   - Script to discover correct PN IDs for all ORNs
   - Can be run to verify PN identification logic

3. **[tests/test_pn_counts_regression.py](tests/test_pn_counts_regression.py)** (new)
   - Automated regression test
   - Verifies PN counts for all ORNs
   - Compares multi-ORN script with OR7a-specific script

## Testing

### Run All Tests:
```bash
# Test PN identification for all ORNs
python tests/test_pathway_pn_counts.py

# Run regression test
python tests/test_pn_counts_regression.py
```

### Verify Fix Manually:
```bash
# OR7a-specific script (reference)
python scripts/map_or7a_complete_pathway.py --max-levels 2

# Multi-ORN script (should match)
python scripts/map_multi_orn_pathways.py --max-levels 2
```

## Lessons Learned

1. **Always validate data IDs**: Root IDs can change between data versions
2. **Test with actual data**: The filtering logic was correct, but data IDs were wrong
3. **Create regression tests**: Prevents future breakage when data is updated
4. **Document expected counts**: Makes it easy to verify correctness

## Future Recommendations

1. **Regenerate PN whitelists** if data is updated to a new FlyWire version
2. **Run regression tests** before any release
3. **Consider removing whitelists**: Pattern matching alone may be more robust to data updates
4. **Add data version tracking**: Document which FlyWire version the IDs are from
