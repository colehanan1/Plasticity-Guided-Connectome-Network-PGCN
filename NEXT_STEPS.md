# 🔬 Next Steps: Run Diagnostics to Find Root Cause

## ✅ What Was Just Created

I've created a comprehensive diagnostic test suite to systematically identify why your model still achieves 100% accuracy even after fixing the `previous_outcome` bug:

### Diagnostic Test Files:
1. **`tests/test_1_class_distribution.py`** - Checks if task is trivially easy (all labels same class)
2. **`tests/test_2_cv_splits.py`** - Verifies no data leakage in cross-validation splits
3. **`tests/test_3_val_set_size.py`** - Checks if validation sets are large enough
4. **`tests/test_4_scrambled_labels.py`** - Creates scrambled dataset to test metadata memorization

### Master Scripts:
- **`tests/run_all_diagnostics.py`** - Runs all 4 tests in sequence with comprehensive report
- **`DIAGNOSTIC_GUIDE.md`** - 460-line guide with detailed explanations, decision trees, and fixes

### All files committed and pushed to:
```
Branch: claude/recurrent-context-memory-ccbpn-01N5CLHBQSgJBXtoDVTj86Ki
Commit: f7908c8
```

---

## 🎯 What You Need to Do RIGHT NOW

### Step 1: Run the Diagnostic Suite (5 minutes)

Open terminal and run:

```bash
cd ~/Plasticity-Guided-Connectome-Network-PGCN

python tests/run_all_diagnostics.py
```

This will:
- ✓ Check class distribution in your behavioral data
- ✓ Verify cross-validation splits from your training results
- ✓ Check validation set sizes
- ✓ Create a scrambled labels dataset for Test 4

### Step 2: Report Back the Results

Tell me which tests PASSED or FAILED:

```
Test 1 (Class Distribution): [✅ PASS or ❌ FAIL]
Test 2 (CV Splits):           [✅ PASS or ❌ FAIL]
Test 3 (Val Size):            [✅ PASS or ❌ FAIL]
```

### Step 3: If Tests 1-3 Pass, Run Test 4 (1 hour)

Train on the scrambled dataset that was created:

```bash
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions_SCRAMBLED.csv \
    --cache-dir data/cache \
    --output-dir results/ccbpn_SCRAMBLED \
    --epochs 50 \
    --context-dim 64 \
    --lr 0.001 \
    --n-folds 5
```

**Then report the validation accuracy:**
- If **~50%** → Model learns from odors (GOOD!)
- If **>70%** → Model memorizes metadata (BAD - needs fix!)

---

## 📊 What Each Test Means

### Test 1: Class Distribution
**Checks:** Are all labels the same class?
**If it fails:** Task is trivially easy, verify behavioral data source

### Test 2: CV Splits
**Checks:** Do same flies appear in train AND validation?
**If it fails:** Data leakage in cross-validation, need to fix GroupKFold

### Test 3: Validation Size
**Checks:** Are validation sets large enough?
**If it fails:** Sample size too small, use 3-fold CV or collect more data

### Test 4: Scrambled Labels (MOST IMPORTANT!)
**Checks:** Does model learn from odors or memorize metadata?
**If it fails:** Model memorizes fly_id/dataset_id, need to strip metadata

---

## 🔍 Decision Tree

```
Run diagnostics
    ↓
Test 1 fails? → Verify behavioral data
    ↓
Test 2 fails? → Fix GroupKFold implementation
    ↓
Test 3 fails? → Use fewer folds or more data
    ↓
All pass? → Run Test 4 (scrambled labels)
    ↓
Test 4: >70%? → Strip metadata from model
    ↓
Test 4: ~50%? → Try baseline CCBPN (no LSTM)
```

---

## 📁 Where to Find Help

- **Comprehensive Guide**: `DIAGNOSTIC_GUIDE.md` (read this for detailed explanations!)
- **Quick Reference**: `BUG_FIX_SUMMARY.md` (previous fix summary)
- **Your Results**: `YOUR_RESULTS_EXPLAINED.md` (high school explanation)

---

## ⚡ Quick Commands Reference

```bash
# Run all diagnostics
python tests/run_all_diagnostics.py

# Run individual tests
python tests/test_1_class_distribution.py
python tests/test_2_cv_splits.py results/ccbpn_recurrent_FIXED
python tests/test_3_val_set_size.py results/ccbpn_recurrent_FIXED
python tests/test_4_scrambled_labels.py

# Train on scrambled data (if Tests 1-3 pass)
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions_SCRAMBLED.csv \
    --output-dir results/ccbpn_SCRAMBLED \
    --epochs 50 --n-folds 5
```

---

## 🚨 CRITICAL: Don't Re-Train Until You Know the Root Cause!

Running diagnostics first will save you hours of wasted training time. Once we identify the root cause, I'll implement the appropriate fix and THEN you can re-train with confidence.

---

## 💬 What to Tell Me Next

After running diagnostics, reply with:

```
Test 1: [PASS/FAIL] - [any warnings or findings]
Test 2: [PASS/FAIL] - [any warnings or findings]
Test 3: [PASS/FAIL] - [any warnings or findings]
Test 4: [if applicable, validation accuracy on scrambled labels]
```

Then I'll know exactly which fix to implement! 🔬

---

**Ready? Run those diagnostics and report back! We're close to solving this! 🚀**
