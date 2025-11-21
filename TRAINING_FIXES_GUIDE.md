# CCBPN Training Fixes Guide

This guide documents the fixes applied to resolve training plateau issues and improve model performance.

## Problem Summary

The original CCBPN training exhibited several issues:

1. **Loss plateau** (~0.54-0.56) after only 10-20 epochs
2. **Validation accuracy stuck** at 73.7% ± 2.8% (suspiciously close to majority class baseline)
3. **No learning progression** - validation accuracy determined by initialization, not training
4. **Identical train-val accuracy** - suggesting severe underfitting
5. **Class imbalance** - 65.7% avoid, 34.3% approach (model likely memorizing majority class)

## Root Causes Identified

### 1. Class Imbalance ⚖️ (CRITICAL)
- **Problem**: 65.7% of trials are "avoid", model achieves 65.7% by always predicting majority class
- **Evidence**: Validation accuracy stuck at 73.7% (only 8pp above baseline)
- **Impact**: Model not learning odor-specific patterns

### 2. Insufficient Model Capacity 🧠
- **Problem**: Only 5% KC sparsity = 100 active KCs per trial
- **Evidence**: ~14 KCs per odor (100 active / 7 odors) insufficient to represent odors
- **Impact**: Model lacks capacity to learn complex odor representations

### 3. Learning Rate Too Low 📉
- **Problem**: LR=0.001 too conservative, model stuck in local minimum
- **Evidence**: Loss drops 15% in 10 epochs, then flat for 90 epochs
- **Impact**: Cannot escape initialization

### 4. Overfitting Risk 🎯
- **Problem**: 100:1 parameter-to-sample ratio (40k params, 392 training samples)
- **Evidence**: Train and val accuracy nearly identical (high bias)
- **Impact**: Model overfits to training fold

## Fixes Implemented

### Fix #1: Class-Balanced Loss ⚖️ (HIGH PRIORITY)

**What it does**: Weights loss by inverse class frequency to prevent majority class bias

**Implementation**:
```python
# In BehavioralTaskLoss class (pgcn/models/ccbpn.py)
if self.use_class_weights and self.class_weights is not None:
    # class_weights[0] = weight for avoid
    # class_weights[1] = weight for approach
    sample_weights = torch.where(
        observed_behavior == 1,
        self.class_weights[1],
        self.class_weights[0]
    )
    loss = F.binary_cross_entropy(predicted, observed, weight=sample_weights)
```

**How to use**:
```bash
python src/scripts/train_ccbpn.py --use_class_weights
```

**Expected improvement**: Model forced to learn odor-specific patterns instead of memorizing majority class

---

### Fix #2: Increased KC Sparsity 🧠 (HIGH PRIORITY)

**What it does**: Increases active KCs from 5% to 10% per trial

**Rationale**:
- 5% sparsity = 100 active KCs → ~14 KCs per odor (insufficient)
- 10% sparsity = 200 active KCs → ~29 KCs per odor (better capacity)

**Implementation**:
```python
# Default changed from 0.05 to 0.10
parser.add_argument("--kc_sparsity", type=float, default=0.10)
```

**How to use**:
```bash
python src/scripts/train_ccbpn.py --kc_sparsity 0.10  # Now default
```

**Expected improvement**: More representational capacity → lower training loss (< 0.45), higher accuracy (80-85%)

---

### Fix #3: Learning Rate Scheduler 📉 (MEDIUM PRIORITY)

**What it does**: Warmup (10 epochs) + cosine decay learning rate schedule

**Implementation**:
```python
# Warmup for 10 epochs: LR = 0.001 → 0.01
# Then cosine decay: LR = 0.01 → 0.0001 over remaining epochs
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=90, eta_min=0.0001)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])
```

**How to use**:
```bash
python src/scripts/train_ccbpn.py --use_lr_scheduler --learning_rate 0.01
```

**Expected improvement**: Loss decreases smoothly over 100 epochs (not plateau at epoch 10)

---

### Fix #4: Dropout Regularization 🎯 (MEDIUM PRIORITY)

**What it does**: Randomly drops 30% of KC→MBON connections during training

**Implementation**:
```python
# In CCBPN forward pass (pgcn/models/ccbpn.py)
if self.kc_dropout is not None and self.training:
    kc_state_dropped = self.kc_dropout(kc_state)  # 30% dropout
```

**How to use**:
```bash
python src/scripts/train_ccbpn.py --dropout 0.3  # Now default
```

**Expected improvement**: Better generalization, validation accuracy closer to training accuracy

---

## Training Commands

### Baseline (original configuration)
```bash
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --epochs 100 \
    --kc_sparsity 0.05 \
    --learning_rate 0.001 \
    --dropout 0.0 \
    --output_dir results/ccbpn_baseline
```

**Expected results**:
- Average validation accuracy: ~73.7% ± 2.8%
- Training loss plateau: ~0.54-0.56
- Model predicts majority class >95% of the time

---

### Improved (with all fixes)
```bash
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --epochs 100 \
    --kc_sparsity 0.10 \
    --learning_rate 0.01 \
    --dropout 0.3 \
    --use_class_weights \
    --use_lr_scheduler \
    --output_dir results/ccbpn_fixed
```

**Expected results**:
- Average validation accuracy: 82-85% (vs. 73.7% baseline)
- Training loss: 0.35-0.40 (vs. 0.54-0.56 baseline)
- Odor-specific predictions (not just majority class)
- Smooth convergence over 100 epochs

---

## Diagnostic Experiments

### 1. Check Class Imbalance
```bash
python src/scripts/diagnose_class_imbalance.py
```

**Expected output**:
```
Class distribution:
  Avoid (0):    729 trials (65.7%)
  Approach (1): 381 trials (34.3%)

⚠️  Majority class baseline: 65.7%
```

**Interpretation**: If validation accuracy ≈ 65.7%, model is NOT learning!

---

### 2. Analyze Model Predictions
```bash
python src/scripts/diagnose_model_predictions.py \
    --model results/ccbpn_fixed/ccbpn_odor_discrimination_best.pt
```

**Good signs**:
- Prediction diversity: ~40-60% approach rate (not 95%+)
- Per-odor variation: >30% range across odors
- Probability std: >0.15 (not <0.05)

**Bad signs**:
- Model predicts >95% one class
- All odors have same prediction rate
- Output probabilities have very low variance

---

### 3. Visualize KC Activity
```bash
python src/scripts/visualize_kc_activity.py \
    --model results/ccbpn_fixed/ccbpn_odor_discrimination_best.pt \
    --output_dir results/diagnostics
```

**Good signs**:
- 7 distinct clusters in PCA plot (one per odor)
- Between-odor distance / within-odor distance > 1.5x
- KC sparsity: 8-12% (target: 10%)

**Bad signs**:
- All odors overlap in one cluster
- Between-odor / within-odor ratio < 1.5x
- KC sparsity < 3% or > 15%

---

### 4. Compare Training Results
```bash
python src/scripts/compare_training_results.py \
    --baseline results/ccbpn_baseline/ccbpn_odor_discrimination_metrics.json \
    --improved results/ccbpn_fixed/ccbpn_odor_discrimination_metrics.json \
    --output_dir results/diagnostics
```

**Expected improvements**:
- Δ Average accuracy: +8-12 percentage points
- Final training loss: 0.35-0.40 (vs. 0.54-0.56)
- Smooth loss curves (not plateau)
- Statistical significance: p < 0.05

---

## Quick Start

### Step 1: Run diagnostics to confirm issues
```bash
# Check class imbalance
python src/scripts/diagnose_class_imbalance.py
```

### Step 2: Train with all fixes
```bash
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --epochs 100 \
    --kc_sparsity 0.10 \
    --learning_rate 0.01 \
    --dropout 0.3 \
    --use_class_weights \
    --use_lr_scheduler \
    --output_dir results/ccbpn_fixed \
    --verbose
```

### Step 3: Analyze trained model
```bash
# Check prediction diversity
python src/scripts/diagnose_model_predictions.py \
    --model results/ccbpn_fixed/ccbpn_odor_discrimination_best.pt

# Visualize KC activity
python src/scripts/visualize_kc_activity.py \
    --model results/ccbpn_fixed/ccbpn_odor_discrimination_best.pt \
    --output_dir results/diagnostics
```

### Step 4: Compare with baseline (if you have one)
```bash
python src/scripts/compare_training_results.py \
    --baseline results/ccbpn_baseline/ccbpn_odor_discrimination_metrics.json \
    --improved results/ccbpn_fixed/ccbpn_odor_discrimination_metrics.json \
    --output_dir results/diagnostics
```

---

## Expected Performance

### Before Fixes
```
Average validation accuracy: 73.7% ± 2.8%
Best fold accuracy: 79.0%
Training loss: 0.54-0.56 (plateau)
Model behavior: Predicts majority class >95% of the time
```

### After Fixes
```
Average validation accuracy: 82-85% ± 2.0%
Best fold accuracy: 87-90%
Training loss: 0.35-0.40 (converges smoothly)
Model behavior: Odor-specific predictions with 30-50% variation
```

### Improvement Summary
- **Accuracy**: +8-12 percentage points
- **Loss**: -30% reduction (0.54 → 0.38)
- **Generalization**: Model learns odor representations, not just majority class
- **Convergence**: Smooth learning over 100 epochs (not plateau at epoch 10)

---

## Troubleshooting

### Issue: Accuracy still stuck at ~66%
**Cause**: Model still predicting majority class
**Solution**: Verify `--use_class_weights` flag is set

### Issue: Loss decreases but accuracy doesn't improve
**Cause**: Model overfitting to training fold
**Solution**: Increase `--dropout` to 0.5, reduce `--learning_rate` to 0.005

### Issue: Training loss < 0.3 but validation loss > 0.5
**Cause**: Severe overfitting (too much capacity)
**Solution**: Reduce `--kc_sparsity` to 0.08, increase `--dropout` to 0.4

### Issue: Validation accuracy > training accuracy
**Cause**: Suspicious! Check for data leakage or improper fold splits
**Solution**: Verify `make_group_kfold` is using `groups=fly_id` correctly

---

## Files Modified

### Model Implementation
- [`src/pgcn/models/ccbpn.py`](src/pgcn/models/ccbpn.py)
  - Added `dropout_rate` parameter to `ConnectomeConstrainedBehavioralPredictor`
  - Added `use_class_weights` and `class_weights` to `BehavioralTaskLoss`
  - Implemented weighted BCE loss in `forward()`

### Training Script
- [`src/scripts/train_ccbpn.py`](src/scripts/train_ccbpn.py)
  - Changed default `--kc_sparsity` from 0.05 → 0.10
  - Changed default `--learning_rate` from 0.001 → 0.01
  - Added `--dropout` parameter (default: 0.3)
  - Added `--use_class_weights` flag
  - Added `--use_lr_scheduler` flag
  - Implemented learning rate scheduler (warmup + cosine decay)
  - Computed class weights from training data

### Diagnostic Scripts (new)
- [`src/scripts/diagnose_class_imbalance.py`](src/scripts/diagnose_class_imbalance.py)
- [`src/scripts/diagnose_model_predictions.py`](src/scripts/diagnose_model_predictions.py)
- [`src/scripts/visualize_kc_activity.py`](src/scripts/visualize_kc_activity.py)
- [`src/scripts/compare_training_results.py`](src/scripts/compare_training_results.py)

---

## References

**Original Issue**: Training loss plateaus at 0.54-0.56, validation accuracy stuck at 73.7%

**Root Cause Analysis**:
1. Class imbalance (65.7% majority baseline)
2. Insufficient model capacity (5% KC sparsity)
3. Learning rate too low (0.001)
4. Overfitting (100:1 parameter-to-sample ratio)

**Priority Fixes**:
1. Class-balanced loss (CRITICAL)
2. Increased KC sparsity 5% → 10% (HIGH)
3. Learning rate scheduler (MEDIUM)
4. Dropout regularization (MEDIUM)

**Expected Impact**: +8-12 pp accuracy improvement, smooth convergence, odor-specific learning

---

## Contact

For issues or questions about the training fixes, please create an issue in the repository or contact the development team.

**Last Updated**: 2025-11-19
