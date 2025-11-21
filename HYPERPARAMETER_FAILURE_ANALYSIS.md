# Why Hyperparameters Are Failing - Critical Analysis

## 🚨 The Core Problem

**Despite 4 hyperparameter tuning attempts spanning the full regularization spectrum, validation accuracy remains stuck at ~70-73% (only 2-5 percentage points above the 67.7% majority class baseline).**

More concerning: **Training accuracy is also only 68-70%**, meaning the model can BARELY fit the training data itself.

---

## 📊 The Evidence: Hyperparameter Journey Failed

| Attempt | Dropout | LR | Weight Decay | Noise | Train Acc | Val Acc | Issue |
|---------|---------|----|--------------| ------|-----------|---------|-------|
| Baseline | 30% | 0.01 | 0 | 15% (4 types) | **100%** | 70.5% | Overfitting to noise |
| #1 | 50% | 0.003 | 0.01 | 8% (additive) | **67%** | 70.3% | Underfitting |
| #2 | 35% | 0.005 | 0.005 | 8% | **66-68%** | 70-75% | Still underfitting |
| #3 | 20% | 0.01 | 0 | 8% | **84%** | 72% | Overfitting (val loss ↑) |
| #4 "Balanced" | 30% | 0.01 | 0.002 | 8% | **68-70%** | 70.5% | **BARELY LEARNING** |

### Recent Training Run (5-Fold CV, "Optimal" Settings):
```
Fold 1: Train=68%, Val=69.2% (stopped epoch 23)
Fold 2: Train=69%, Val=73.3% (stopped epoch 49)
Fold 3: Train=70%, Val=68.8% (stopped epoch 22)
Fold 4: Train=68%, Val=67.9% (stopped epoch 46)
Fold 5: Train=69%, Val=73.3% (stopped epoch 23)

Average validation accuracy: 70.5% ± 2.3%
```

---

## 🔍 The Critical Insight: This Is NOT a Hyperparameter Problem

### What We Expected:
```
Too much regularization → 67% train (can't learn)
OPTIMAL regularization → 75-80% train, 74-78% val (learns well, generalizes)
Too little regularization → 85-95% train, 70-75% val (overfits)
```

### What We Actually See:
```
Too much regularization → 67% train, 70% val
"OPTIMAL" regularization → 68-70% train, 70% val  ⚠️ ONLY 1-3pp BETTER!
Too little regularization → 84% train, 72% val (but val loss increases)
```

**The "sweet spot" is barely above the 67.7% majority class baseline!**

This means the model is fundamentally struggling to learn odor-specific patterns, regardless of hyperparameter settings.

---

## 🎯 Root Cause Analysis: Why Can't the Model Learn?

### Hypothesis 1: Input Representations Are Not Discriminative ⚠️ MOST LIKELY

**Issue**: DoOR-based PN response patterns may not provide enough information to distinguish odors in context-dependent manner across 8 different datasets.

**Evidence**:
- Model can memorize training data with 15% noise (100% train accuracy)
- But with realistic noise (8%, correlation 0.94), it can barely exceed baseline
- Suggests signal-to-noise ratio is marginal even at biological noise levels

**Test**:
```bash
# Verify PN patterns are actually different between odors
python src/scripts/analyze_pn_patterns.py \
    --behavioral_data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --output_dir results/pn_analysis
```

**Expected findings**:
- Are PN patterns for hexanol vs 3-octanol sufficiently different?
- Are patterns consistent across datasets?
- Is DoOR providing realistic responses?

---

### Hypothesis 2: KC Sparsity Too Low (10% = 200 KCs) ⚠️ LIKELY

**Issue**: With only 200 active KCs per trial (out of 2000), the model may have insufficient representational capacity to:
1. Encode 8 different odors
2. Across 8 different datasets (context-dependent learning)
3. With trial-to-trial noise (8%)
4. And distinguish CS+ from CS- in each context

**Current capacity**:
- 200 active KCs per trial
- 44 MBONs trying to decode behavior
- 8 odors × 8 datasets = 64 odor-context combinations
- 200 KCs / 64 combinations = **~3 KCs per odor-context** ⚠️

**Biological comparison**:
- Real flies have ~2000 KCs, but use 5-20% sparsity (100-400 active KCs)
- Our 10% (200 KCs) may be at the lower bound
- Increasing to 15% (300 KCs) or 20% (400 KCs) could help

**Test**:
```bash
# Try higher KC sparsity
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --kc_sparsity 0.15 \
    --use_class_weights \
    --use_lr_scheduler \
    --output_dir results/ccbpn_sparsity15
```

---

### Hypothesis 3: Dopamine Signal May Be Hurting ⚠️ POSSIBLE

**Issue**: Dopamine-gated plasticity assigns dopamine=1.0 only when odor matches CS+ for that dataset. This creates a very sparse learning signal:
- CS+ trials (hexanol, benzaldehyde, ethyl_butyrate, 3-octanol in respective datasets): dopamine=1.0
- CS- trials (hexanol in blocking trials): dopamine=0.0
- Novel odors: dopamine=0.0
- Control datasets: dopamine=0.0

**Result**: Model must learn approach/avoid behavior with minimal reward signal.

**Test**:
```bash
# Try training WITHOUT dopamine modulation (pure supervised learning)
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --use_class_weights \
    --use_lr_scheduler \
    --output_dir results/ccbpn_no_dopamine \
    --disable_dopamine  # Need to add this flag
```

---

### Hypothesis 4: FlyWire Connectivity Is Too Constrained 🤔 LESS LIKELY

**Issue**: Using biologically-realistic FlyWire connectivity (150 PNs → 2000 KCs → 44 MBONs) may limit model capacity compared to fully-connected networks.

**Counter-evidence**:
- Model CAN reach 100% train accuracy with excessive noise (Attempt #0)
- Model CAN reach 84% train accuracy with low regularization (Attempt #3)
- This suggests architecture has capacity, but learning is unstable

**Unlikely to be the main issue**, but could contribute.

---

### Hypothesis 5: Task Is Too Difficult 🤔 POSSIBLE

**Issue**: Learning approach/avoid behavior across 8 datasets with context-dependent CS+ assignments may be too complex:
- opto_hex: approach hexanol (CS+), avoid others
- opto_benz_1: approach benzaldehyde (CS+), avoid others
- opto_EB: approach ethyl_butyrate (CS+), avoid others
- opto_3-oct: approach 3-octanol (CS+), avoid others
- 4 control datasets: innate preferences only

**This requires**:
1. Identifying the odor (PN → KC encoding)
2. Identifying the dataset context (what was CS+ in training?)
3. Deciding approach/avoid based on odor + context
4. Handling blocking trials (CS- during training)
5. Generalizing to novel odors

**Test**:
```bash
# Try training on SINGLE dataset only
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --dataset opto_hex \
    --use_class_weights \
    --use_lr_scheduler \
    --output_dir results/ccbpn_single_dataset
```

If single-dataset training reaches 80%+ accuracy, then the issue is context-dependent learning across 8 datasets.

---

### Hypothesis 6: Data Quality Issues 🤔 LESS LIKELY

**Issue**: Behavioral data itself may have low signal-to-noise ratio (flies behaving inconsistently).

**Evidence against**:
- Dataset has 1200 trials from 68 flies
- Class imbalance is 67.7% avoid, 32.3% approach (not extreme)
- Data is from published experiments

**Unlikely**, but worth checking for outliers or mislabeled trials.

---

## 🚀 Recommended Action Plan (Priority Order)

### 1. **IMMEDIATE: Increase KC Sparsity** ⭐ HIGHEST PRIORITY

**Why**: Most tractable fix, directly addresses representational capacity issue.

**Action**:
```bash
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --kc_sparsity 0.15 \
    --dropout 0.30 \
    --learning_rate 0.01 \
    --use_class_weights \
    --use_lr_scheduler \
    --output_dir results/ccbpn_sparsity15 \
    --verbose
```

**Expected**: If this works, training accuracy should reach 75-80% (not 68-70%).

---

### 2. **DIAGNOSTIC: Analyze PN Input Patterns** ⭐ HIGH PRIORITY

**Why**: Need to verify input representations are actually discriminative.

**Action**:
```bash
# Create a diagnostic script to analyze PN patterns
python -c "
import numpy as np
import pandas as pd
from src.pgcn.data.door_integration import DoORIntegration

door = DoORIntegration(cache_dir='data/cache')

odors = ['hexanol', 'benzaldehyde', 'ethyl_butyrate', '3-octanol']
for odor in odors:
    pn_pattern = door.get_pn_response_pattern(odor, duration_ms=1000, add_noise=False)
    print(f'{odor}: mean={pn_pattern.mean():.3f}, std={pn_pattern.std():.3f}, nonzero={np.sum(pn_pattern > 0.1)}')

# Check pairwise correlations
patterns = [door.get_pn_response_pattern(o, 1000, False) for o in odors]
for i, o1 in enumerate(odors):
    for j, o2 in enumerate(odors):
        if i < j:
            corr = np.corrcoef(patterns[i], patterns[j])[0,1]
            print(f'{o1} vs {o2}: correlation = {corr:.3f}')
"
```

**Expected**:
- Correlations between different odors should be < 0.7 (if higher, odors are too similar)
- Each odor should activate different subsets of PNs

---

### 3. **TEST: Single Dataset Training** ⭐ HIGH PRIORITY

**Why**: Determine if problem is context-dependent learning across 8 datasets.

**Action**: Need to modify training script to support single-dataset mode, then:
```bash
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --datasets opto_hex \
    --use_class_weights \
    --use_lr_scheduler \
    --output_dir results/ccbpn_single_dataset
```

**Expected**: If single-dataset training reaches 80%+, problem is multi-dataset context learning.

---

### 4. **TEST: Train Without Dopamine Modulation** 🔬 MEDIUM PRIORITY

**Why**: Check if dopamine-gated plasticity is hindering learning.

**Action**: Modify model to use standard supervised learning (no dopamine signal).

**Expected**: If accuracy improves significantly, dopamine signal may be too sparse or incorrectly assigned.

---

### 5. **TEST: Reduce Trial-to-Trial Noise** 🔬 LOW PRIORITY

**Why**: Current noise (8%, correlation 0.94) may still be too high for this task.

**Action**:
```bash
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --noise_std 0.04 \
    --use_class_weights \
    --use_lr_scheduler \
    --output_dir results/ccbpn_noise4
```

**Expected**: If accuracy improves, biological realism (noise) is conflicting with task difficulty.

---

## 📊 Success Metrics for Each Test

### Test 1: Higher KC Sparsity (15%)
- ✅ **Success**: Training accuracy 75-80% (not 68-70%), validation 74-78%
- ❌ **Failure**: Still stuck at 68-70% train, 70-73% val

### Test 2: PN Pattern Analysis
- ✅ **Good sign**: Odor correlations < 0.7, distinct activation patterns
- ⚠️ **Warning sign**: Odor correlations > 0.8, very similar patterns
- ❌ **Problem**: Odor correlations > 0.9, nearly identical patterns

### Test 3: Single Dataset
- ✅ **Success**: Single dataset reaches 80%+ accuracy
  - **Interpretation**: Problem is multi-dataset context learning
  - **Solution**: Need stronger context representation (MBON state, recurrent connections)
- ❌ **Failure**: Single dataset still stuck at 70%
  - **Interpretation**: Fundamental issue with odor encoding or architecture

### Test 4: No Dopamine
- ✅ **Success**: Accuracy improves to 75-80%
  - **Interpretation**: Dopamine signal assignment is incorrect or too sparse
  - **Solution**: Revise dopamine assignment logic or remove dopamine gating
- ❌ **Failure**: No improvement
  - **Interpretation**: Dopamine is not the issue

### Test 5: Lower Noise
- ✅ **Success**: Accuracy improves to 75-80%
  - **Interpretation**: Biological realism (8% noise) is too high for task
  - **Solution**: Trade off biological realism for performance (use 4% noise)
- ❌ **Failure**: No improvement
  - **Interpretation**: Noise is not limiting factor

---

## 🎓 Key Lessons from This Journey

### Lesson 1: Hyperparameters Have Limits
**When training accuracy is barely above baseline (68-70% vs 67.7%), no amount of hyperparameter tuning will help.** The model fundamentally cannot learn the task with current:
- Input representations (DoOR PN patterns)
- Architecture (FlyWire connectivity)
- Capacity (10% KC sparsity)
- Learning signal (dopamine-gated plasticity)

### Lesson 2: Training Accuracy Is Your Diagnostic
The fact that training accuracy plateaus at 68-70% (barely above 67.7% baseline) tells us:
- Model has capacity (can reach 100% with wrong noise, 84% with no weight decay)
- Model cannot find stable solution with current settings
- Problem is NOT generalization (train and val both low)
- Problem is LEARNING CAPACITY with these specific conditions

### Lesson 3: Biological Realism vs Performance Tradeoff
We successfully achieved:
- ✅ Realistic trial-to-trial variability (8% noise, correlation 0.94)
- ✅ Biologically-motivated dopamine-gated plasticity
- ✅ FlyWire anatomical connectivity
- ✅ KC sparsity matching biology (10%)

But this biological realism may be **limiting performance**. Sometimes you need to relax biological constraints to achieve task performance.

### Lesson 4: Start Simple, Add Complexity
In hindsight, we should have:
1. First trained on single dataset (simplest task)
2. Verified model can reach 80%+ accuracy
3. Then added multi-dataset complexity
4. Then added biological noise
5. Then added dopamine-gated plasticity

Instead, we tried to solve everything at once and got stuck.

---

## 📈 Expected Timeline for Fixes

### Quick Wins (1-2 hours):
1. ✅ Fix opto_3-oct testing trials (DONE)
2. 🔬 Increase KC sparsity to 15% and retrain
3. 🔬 Analyze PN input patterns
4. 🔬 Train with 4% noise instead of 8%

### Medium Effort (1 day):
1. Implement single-dataset training mode
2. Train without dopamine modulation
3. Compare results across all tests
4. Determine root cause

### Longer Term (2-3 days):
1. If KC sparsity helps: find optimal sparsity level (10%, 15%, 20%, 25%)
2. If single-dataset works: implement proper context encoding for multi-dataset
3. If dopamine is issue: redesign dopamine assignment logic
4. If PN patterns poor: try alternative odor encoding (random projections, learned embeddings)

---

## 🎯 My Recommendation: Start with KC Sparsity

**Based on the evidence, I believe the most likely issue is insufficient representational capacity (200 active KCs trying to encode 64 odor-context combinations with noise).**

**Immediate next step**:
```bash
# Train with 15% KC sparsity (300 active KCs instead of 200)
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --kc_sparsity 0.15 \
    --dropout 0.30 \
    --learning_rate 0.01 \
    --use_class_weights \
    --use_lr_scheduler \
    --output_dir results/ccbpn_sparsity15 \
    --behavioral_data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --verbose
```

**If this reaches 75-80% training accuracy**, we've found the issue.

**If this still plateaus at 68-70%**, then we need to investigate PN patterns or simplify the task to single-dataset.

---

## ✅ Summary

**What we know**:
- ✅ Biological noise is correct (8%, correlation 0.94)
- ✅ Dopamine assignment logic is correct (CS+ based)
- ✅ Class weights are correct (handles 67.7%/32.3% imbalance)
- ✅ Dataset configuration is correct (1200 trials, 8 datasets)
- ✅ Hyperparameters span full regularization spectrum

**What's wrong**:
- ❌ Model can only reach 68-70% training accuracy (barely above 67.7% baseline)
- ❌ Validation accuracy stuck at 70-73% (only 2-5 points above baseline)
- ❌ Early stopping triggers immediately (no learning progression)

**Most likely causes** (in priority order):
1. ⚠️ **Insufficient KC capacity** (10% = 200 KCs → try 15% = 300 KCs)
2. ⚠️ **Poor input representations** (DoOR PN patterns not discriminative enough)
3. 🤔 **Task too complex** (8 datasets with context-dependent learning)
4. 🤔 **Dopamine signal too sparse** (try supervised learning without dopamine)

**Next steps**:
1. Increase KC sparsity to 15% and retrain ⭐
2. Analyze PN pattern discriminability ⭐
3. Test single-dataset training ⭐
4. Consider relaxing biological constraints for performance

---

**Last Updated**: 2025-11-21 (after 4 failed hyperparameter attempts)
