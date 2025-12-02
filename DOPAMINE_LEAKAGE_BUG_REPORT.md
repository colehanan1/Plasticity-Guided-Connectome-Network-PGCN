# 🐛 CRITICAL BUG REPORT: Dopamine Signal Label Leakage

## Executive Summary

**Status:** ✅ **FIXED** (commit e3a292d)
**Severity:** CRITICAL - Complete model invalidation
**Discovery Method:** Scrambled labels diagnostic test

Your model was achieving 100% accuracy because it was **cheating** - the behavioral label was directly encoded in the dopamine input signal. The model didn't need to learn from odor patterns at all.

---

## The Bug

### Location
`src/scripts/train_ccbpn_recurrent.py` line 221

### Original Code (BUGGY)
```python
def _generate_dopamine_signal(self, row: pd.Series) -> np.ndarray:
    """Generate dopamine signal for one trial."""
    dopamine = np.zeros(self.odor_duration, dtype=np.float32)

    # BUG: Using the label to generate dopamine!
    is_cs_plus = row.get('prediction', 0) > 0.5

    if is_cs_plus:
        dopamine[dopamine_start:dopamine_end] = 1.0

    return dopamine
```

### What Was Happening
- **Label = 1 (approach)** → Dopamine pulse present in input
- **Label = 0 (avoid)** → No dopamine in input

The model's "forward pass" literally received the answer:
```
Input to model:
  - Odor pattern: [0.2, 0.5, 0.1, ...]
  - Dopamine: [0, 0, 0, 1, 1, 1, 0]  ← THIS ENCODES THE LABEL!
  - Previous outcome: 0.0

Model output:
  - Prediction: 1.0  ← Just copies whether dopamine is present!
```

### Why This Caused 100% Accuracy

**On Original Data:**
- Model saw dopamine → predicted approach (100% correct)
- Model saw no dopamine → predicted avoid (100% correct)

**On Scrambled Data (the smoking gun!):**
- Even though labels were random, dopamine was generated FROM those random labels
- Trial with scrambled label=1 → dopamine present → model predicts 1 ✓
- Trial with scrambled label=0 → no dopamine → model predicts 0 ✓
- Result: 100% accuracy on random labels!

This is why the scrambled labels test was so critical - it proved the model wasn't learning from odors.

---

## The Fix

### New Code (FIXED)
```python
def _generate_dopamine_signal(self, row: pd.Series) -> np.ndarray:
    """Generate dopamine signal for one trial.

    CRITICAL FIX: Previously used row['prediction'] to determine dopamine,
    which leaked the label directly to the model!

    For now, we disable dopamine signals entirely (all zeros).
    """
    # No dopamine signal (all zeros) to prevent label leakage
    dopamine = np.zeros(self.odor_duration, dtype=np.float32)
    return dopamine
```

### Rationale
- We don't have reliable CS+/CS- information independent of behavioral outcomes
- Setting dopamine to zero is biologically reasonable for this dataset
- Forces model to learn from odor patterns and previous outcomes only

---

## Expected Results After Fix

### Test 1: Scrambled Labels (Sanity Check)
**Run this first to verify the fix:**

```bash
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions_SCRAMBLED.csv \
    --cache-dir data/cache \
    --output-dir results/ccbpn_FIXED_SCRAMBLED \
    --epochs 50 \
    --n-folds 5
```

**Expected validation accuracy:** 45-55% (random guessing)

✅ **If you get ~50%:** Fix is working! Model can't predict random labels.
❌ **If you get >70%:** There's still label leakage somewhere.

### Test 2: Real Data (Actual Performance)
**After verifying fix on scrambled data:**

```bash
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --cache-dir data/cache \
    --output-dir results/ccbpn_FIXED_REAL \
    --epochs 50 \
    --n-folds 5
```

**Expected validation accuracy:** 74-78%

This would be realistic performance showing the model actually learns from:
- Odor-specific PN activation patterns
- Previous trial outcomes (recurrent context)
- LSTM-based memory across trials

---

## Timeline of Bugs

### Bug 1: `previous_outcome` Leakage (FIXED in commit 57a3f8a)
- **Issue:** Model saw true labels from previous trials instead of its own predictions
- **Impact:** Moderate - flies are consistent, so copying previous label works well
- **Status:** ✅ Fixed

### Bug 2: Dopamine Signal Leakage (FIXED in commit e3a292d)
- **Issue:** Model saw current trial's label encoded in dopamine input
- **Impact:** CRITICAL - model doesn't need to learn anything from odors
- **Status:** ✅ Fixed

### Remaining Diagnostics
From your test results:
- ✅ **Test 1 (Class Distribution):** PASSED - Data has good variance
- ❌ **Test 2 (CV Splits):** INCONCLUSIVE - Fly tracking not implemented
- ❌ **Test 3 (Val Size):** FAILED - Cannot verify sizes without fly tracking

**Note:** Tests 2 and 3 are now less critical since we found the root cause (dopamine leakage). However, adding fly tracking would still be good practice for future debugging.

---

## What This Means Biologically

### What We Thought Was Happening
"The model learns that benzaldehyde → approach, hexanol → avoid based on PN activation patterns and LSTM context memory."

### What Was Actually Happening
"The model learns that dopamine present → approach, no dopamine → avoid. Odor patterns irrelevant."

### What Should Happen Now
"The model must learn odor→behavior mappings from PN patterns, using LSTM to maintain context across trials. No shortcuts!"

---

## Action Items

### 1. ✅ DONE: Fix Applied
- Dopamine signals set to zero (all trials)
- Commit: e3a292d
- Branch: claude/recurrent-context-memory-ccbpn-01N5CLHBQSgJBXtoDVTj86Ki

### 2. ⏳ YOUR TURN: Verify Fix on Scrambled Labels

```bash
cd ~/Plasticity-Guided-Connectome-Network-PGCN

python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions_SCRAMBLED.csv \
    --cache-dir data/cache \
    --output-dir results/ccbpn_FIXED_SCRAMBLED \
    --epochs 50 \
    --n-folds 5
```

**Report back the validation accuracy!**

### 3. ⏳ YOUR TURN: Train on Real Data (If Test Passes)

```bash
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --cache-dir data/cache \
    --output-dir results/ccbpn_FIXED_REAL \
    --epochs 50 \
    --n-folds 5
```

**Expected:** 74-78% validation accuracy (realistic learning)

---

## Future Improvements

### Proper Dopamine Signals (Optional)
If you want to include dopamine modeling in the future:

1. **Add CS+/CS- column to behavioral CSV**
   - Based on experimental design (not behavioral outcome!)
   - Example: `is_cs_plus = True` for reward-paired odors

2. **Update dopamine generation:**
   ```python
   is_cs_plus = row.get('is_cs_plus', False)  # From experiment design
   if is_cs_plus:
       dopamine[dopamine_start:dopamine_end] = 1.0
   ```

3. **Verify with scrambled labels:**
   - Model should still get ~50% on scrambled data
   - Because CS+ status is independent of (scrambled) outcomes

### Add Fly Tracking (Optional)
Useful for verifying CV splits:
```python
fold_results.append({
    'fold': fold + 1,
    'train_acc': train_acc,
    'val_acc': val_acc,
    'train_flies': train_flies,  # Add this
    'val_flies': val_flies,      # Add this
})
```

---

## Key Takeaway

**The scrambled labels test saved your research!** Without it, you might have published results based on a model that was "cheating" via label leakage. This is why systematic diagnostic testing is critical in machine learning.

Now that the bug is fixed, the model must learn the hard way - from actual odor patterns. Let's see if it can achieve realistic 74-78% accuracy! 🧪

---

## Quick Reference

| Test | Command | Expected Result |
|------|---------|----------------|
| **Scrambled labels** | `python src/scripts/train_ccbpn_recurrent.py --behavioral-data ...SCRAMBLED.csv --output-dir results/ccbpn_FIXED_SCRAMBLED --epochs 50 --n-folds 5` | ~50% accuracy |
| **Real data** | `python src/scripts/train_ccbpn_recurrent.py --behavioral-data ...model_predictions.csv --output-dir results/ccbpn_FIXED_REAL --epochs 50 --n-folds 5` | 74-78% accuracy |

**Report back after running the scrambled labels test!** 🚀
