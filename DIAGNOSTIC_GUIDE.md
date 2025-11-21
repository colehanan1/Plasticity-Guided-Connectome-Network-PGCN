# 🔬 Deep Diagnostic Guide - Debugging 100% Accuracy

## TL;DR - What to Do Now

**Your 100% accuracy persists even after the previous_outcome fix. Run these diagnostic tests to find the root cause:**

```bash
cd ~/Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-

# Run all diagnostics at once:
python tests/run_all_diagnostics.py

# Or run individually:
python tests/test_1_class_distribution.py
python tests/test_2_cv_splits.py results/ccbpn_recurrent_FIXED
python tests/test_3_val_set_size.py results/ccbpn_recurrent_FIXED
python tests/test_4_scrambled_labels.py
```

**Then report back the results!**

---

## 📋 The 4 Critical Tests

### Test 1: Class Distribution ⭐

**What it checks**: Are all labels the same class? (trivial task)

**Why it matters**: If 100% of trials are "approach", model just learns "always predict approach" → 100%

**Expected result**:
- ✅ PASS: Variance > 0.3, mix of approach/avoid trials
- ❌ FAIL: Variance < 0.1, nearly all one class

**If it fails**:
- **Root cause**: Task is trivially easy
- **Action**: Verify behavioral data is correct
- **Explanation**: Your `prediction` column might all be 1s or 0s

---

### Test 2: Cross-Validation Splits ⭐⭐

**What it checks**: Do same flies appear in train AND validation?

**Why it matters**: If yes, model memorizes fly-specific patterns → 100%

**Expected result**:
- ✅ PASS: Zero overlap between train_flies and val_flies
- ❌ FAIL: Some flies in both train and validation
- ⚠️ INCONCLUSIVE: Fly tracking not implemented

**If it fails**:
- **Root cause**: Data leakage in cross-validation
- **Action**: Fix GroupKFold to use flies as groups, not trials
- **Explanation**: CV must split by FLIES, not randomly by trials

---

### Test 3: Validation Set Size

**What it checks**: Are validation sets large enough?

**Why it matters**: Getting 100% on 20 samples is easier than on 200

**Expected result**:
- ✅ PASS: >100 validation trials per fold
- ❌ FAIL: <50 validation trials per fold
- ⚠️ WARNING: 50-100 validation trials

**If it fails**:
- **Root cause**: Sample size too small
- **Action**: Use fewer folds (3-fold) or collect more data
- **Explanation**: Small samples → high variance, 100% is possible by chance

---

### Test 4: Scrambled Labels ⭐⭐⭐ MOST IMPORTANT!

**What it checks**: Does model learn from odor patterns or memorize metadata?

**Why it matters**: If model gets >70% on RANDOM labels, it's cheating!

**How it works**:
1. Test creates `model_predictions_SCRAMBLED.csv` with random labels
2. You train on scrambled data
3. Check validation accuracy

**Expected results**:
- ✅ GOOD: Accuracy ~50% (random guessing) → Model learns from odors
- ❌ BAD: Accuracy >70% → Model memorizes metadata (fly ID, dataset ID)
- ⚠️ CONCERNING: Accuracy 55-70% → Partial metadata leakage

**If it fails** (accuracy > 70% on scrambled):
- **Root cause**: Model has access to informative metadata
- **Action**: Strip fly_id, dataset_id, trial_order from inputs
- **Explanation**: Model is "cheating" by using non-odor information

---

## 🎯 Decision Tree Based on Results

### Scenario 1: Test 1 Fails (No Label Variance)
```
Test 1: ❌ FAIL - All labels same class
Test 2: N/A
Test 3: N/A
Test 4: N/A

ROOT CAUSE: Task is trivial
ACTION: Verify behavioral data source
FIX: Check data collection, ensure mix of CS+ and CS- trials
```

### Scenario 2: Test 2 Fails (CV Leakage)
```
Test 1: ✅ PASS
Test 2: ❌ FAIL - Flies overlap between train/val
Test 3: Any
Test 4: N/A

ROOT CAUSE: Cross-validation leakage
ACTION: Fix GroupKFold implementation
FIX: See "Fix 2: CV Leakage" below
```

### Scenario 3: Test 3 Fails (Small Sample)
```
Test 1: ✅ PASS
Test 2: ✅ PASS
Test 3: ❌ FAIL - Val set < 50 trials
Test 4: N/A

ROOT CAUSE: Validation set too small
ACTION: Use 3-fold CV or collect more data
FIX: Accept noisy results or get more flies
```

### Scenario 4: All Pass, But Test 4 Fails (Metadata Memorization)
```
Test 1: ✅ PASS
Test 2: ✅ PASS
Test 3: ✅ PASS
Test 4: ❌ FAIL - >70% accuracy on scrambled labels

ROOT CAUSE: Model memorizes metadata (fly ID, dataset ID)
ACTION: Strip metadata from inputs
FIX: See "Fix 3: Metadata Leakage" below
```

### Scenario 5: ALL Tests Pass (Including Test 4)
```
Test 1: ✅ PASS
Test 2: ✅ PASS
Test 3: ✅ PASS
Test 4: ✅ PASS - ~50% on scrambled labels

This is the MOST PUZZLING case!
Data is good, CV is proper, model learns from odors, but still 100% on real data.

POSSIBLE CAUSES:
1. Subtle bug in model architecture
2. LSTM not actually being used despite fix
3. Task is genuinely too easy (odors very distinct)

NEXT STEPS:
- Try baseline CCBPN without LSTM (should get 70-75%)
- If baseline also gets 100%, problem is in base CCBPN
- If baseline gets 70-75%, problem is in LSTM integration
```

---

## 🔧 Fixes for Each Scenario

### Fix 1: If Test 1 Fails (No Label Variance)

**Problem**: All labels are 0 or all labels are 1

**Diagnostic**:
```python
import pandas as pd
df = pd.read_csv('model_predictions.csv')
print(df['prediction'].value_counts())
# If you see only one value → problem confirmed
```

**Solution**: Check data source
1. Verify `prediction` column is approach/avoid (not something else)
2. Ensure data includes BOTH CS+ and CS- trials
3. Contact data provider if labels are wrong

---

### Fix 2: If Test 2 Fails (CV Leakage)

**Problem**: Same flies in train and validation

**Current code** (probably buggy):
```python
# WRONG: Splits trials randomly, not flies
kfold = GroupKFold(n_splits=5)
for train_idx, val_idx in kfold.split(trials, groups=flies):
    # Same fly might have trials in both train and val!
```

**Fixed code**:
```python
# CORRECT: Split at fly level
from sklearn.model_selection import GroupKFold

# Get unique fly IDs
fly_ids = list(sequences_by_fly.keys())  # e.g., ['fly_001', 'fly_002', ...]
n_flies = len(fly_ids)

# Create fly-level splits
kfold = GroupKFold(n_splits=5)
X_dummy = np.arange(n_flies)  # One entry per fly
groups = np.array(fly_ids)    # Fly IDs as groups

for fold_idx, (train_fly_idx, val_fly_idx) in enumerate(kfold.split(X_dummy, groups=groups)):
    # Get fly IDs for this fold
    train_flies = [fly_ids[i] for i in train_fly_idx]
    val_flies = [fly_ids[i] for i in val_fly_idx]

    # CRITICAL: Verify no overlap
    assert len(set(train_flies) & set(val_flies)) == 0, "Flies overlap!"

    # Create datasets
    train_dataset = dataset.get_subset(train_flies)
    val_dataset = dataset.get_subset(val_flies)

    # IMPORTANT: Track which flies are in each split
    fold_results = {
        'train_flies': train_flies,  # Save for verification
        'val_flies': val_flies,      # Save for verification
        # ... rest of results ...
    }
```

**Key points**:
- Split FLIES, not trials
- Each fly's all trials stay together
- No fly appears in both train and validation
- Track train_flies and val_flies for verification

---

### Fix 3: If Test 4 Fails (Metadata Leakage)

**Problem**: Model learns from fly_id, dataset_id, trial_order instead of odors

**Current code** (might be passing metadata):
```python
# In data loader or model forward:
outputs = model(
    odor_sequences=odor_tensor,
    dopamine_signals=dopamine_tensor,
    fly_id=fly_id,              # ← REMOVE THIS!
    dataset_id=dataset_id,      # ← REMOVE THIS!
    trial_index=trial_idx,      # ← REMOVE THIS!
    ...
)
```

**Fixed code**:
```python
# Model should ONLY see odor patterns and dopamine
# NO metadata about which fly, which dataset, which trial

# In SequentialBehavioralDataset.__getitem__:
for trial in fly_sequence:
    yield {
        'odor': odor_pattern,      # ✅ OK: sensory input
        'dopamine': dopamine_signal,  # ✅ OK: modulatory signal
        # ❌ REMOVE: fly_id, dataset_id, trial_index, session_id
    }

# In model forward pass:
def forward(self, odor_sequences, dopamine_signals, hidden_state, previous_outcome):
    """
    Inputs should ONLY be:
    - odor_sequences: PN activity patterns
    - dopamine_signals: DA neuron activity
    - hidden_state: recurrent memory (computed internally)
    - previous_outcome: model's own prediction (not metadata)

    NO: fly_id, dataset_id, trial_index, or any identifying info
    """
    # Verify input shapes
    assert odor_sequences.shape == (batch, time, n_pn)
    assert dopamine_signals.shape == (batch, time)
    # Good! No extra dimensions for metadata
```

**How to check**:
1. Print model inputs during training
2. Verify only odor and dopamine tensors
3. No fly IDs or dataset IDs passed to model
4. Context comes from previous_outcome (model's prediction), not metadata

---

## 📊 Example Outputs

### Test 1: PASS (Good)
```
TEST 1: CLASS DISTRIBUTION ANALYSIS
===================================

OVERALL:
  Total trials: 1200
  Approach: 720 (60.0%)
  Avoid: 480 (40.0%)
  Variance: 0.490

✅ PASS: Labels have good variance (0.490)
   Class distribution: 60.0% / 40.0%

PER-DATASET:
  opto_hex            :  300 trials,  210 approach (70.0%),   90 avoid (30.0%)
  opto_benz           :  300 trials,  150 approach (50.0%),  150 avoid (50.0%)
  ...

✅ Flies show natural variation (0.234)

✅ TEST 1 PASSED: Data has sufficient variance
```

### Test 2: FAIL (Bad)
```
TEST 2: CROSS-VALIDATION SPLIT ANALYSIS
=======================================

Fold 1:
  Train flies: 96
  Val flies: 24
  Overlap: 5
  ❌ DATA LEAKAGE! Overlapping flies: ['fly_012', 'fly_045', ...]

❌ TEST 2 FAILED: Data leakage detected!
   ROOT CAUSE: Same flies appear in both train and validation
   RECOMMENDATION: Fix GroupKFold implementation
```

### Test 4: FAIL (Metadata leakage)
```
After training on SCRAMBLED labels:

Mean Val Acc: 82.3% ± 3.1%  ← Should be ~50%!

❌ CRITICAL: Model achieves 82% on RANDOM labels!
   This is way above chance (50%)

   ROOT CAUSE: Model is memorizing metadata
   - Not learning from odor patterns
   - Using fly ID, dataset ID, or trial order

   ACTION: Strip all metadata from model inputs
```

---

## 🎯 What to Do Right Now

### Step 1: Run All Diagnostics (5 minutes)

```bash
cd ~/Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-

python tests/run_all_diagnostics.py
```

This will:
1. Check class distribution
2. Verify CV splits
3. Check validation set sizes
4. Create scrambled dataset

### Step 2: If Tests 1-3 Pass, Train on Scrambled Data (1 hour)

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

### Step 3: Report Results

Tell me:
1. Which tests passed/failed
2. If you trained on scrambled data, what accuracy did you get?
3. Any error messages or warnings

---

## 🔍 Nuclear Option: Baseline Test

If ALL tests pass (including scrambled labels ~50%), try baseline without LSTM:

```bash
# Quick test: Train just CCBPN core (no recurrence)
# We know this worked before (70-75% accuracy)

# If this also gets 100%:
# → Problem is in base CCBPN or data preprocessing

# If this gets 70-75%:
# → Problem is specifically in LSTM integration
```

---

## 📝 Commit Message Template

After running diagnostics:

```
docs: diagnostic results for persistent 100% accuracy

Test 1 (Class Distribution): [PASS/FAIL]
- Finding: [variance, distribution]

Test 2 (CV Splits): [PASS/FAIL]
- Finding: [overlap counts]

Test 3 (Val Size): [PASS/FAIL]
- Finding: [samples per fold]

Test 4 (Scrambled Labels): [PASS/FAIL]
- Finding: [accuracy on random labels]

Root Cause: [identified issue]

Next Action: [specific fix to implement]
```

---

## 🎉 Success Criteria

**Diagnostics complete when**:
- ✅ All 4 tests run successfully
- ✅ Root cause identified
- ✅ Appropriate fix implemented
- ✅ Model re-trained with fix
- ✅ Validation accuracy in realistic range (70-80%)

**If all tests pass but 100% persists**:
- Try baseline CCBPN (no LSTM)
- Compare single-dataset vs multi-dataset
- Consider dataset embeddings as alternative

---

**Run the diagnostics now and report back! We'll find the root cause systematically. 🔬**
