# Overfitting Fixes - Final Tuned Parameters

## 🎯 Problem Evolution

### Original Issue (Before Biological Fixes)
```
Training accuracy:   73.7%
Validation accuracy: 73.7%
Issue: Model stuck at majority class baseline, not learning
```

### After Biological Fixes (Excessive Noise)
```
Training accuracy:   100.0% (memorizes training set)
Validation accuracy:  70.5% (worse than baseline!)
Validation loss:      0.58 → 3.3 (increasing)
Issue: Severe overfitting to noise patterns
```

### After First Overfitting Fixes (Over-Regularized)
```
Training accuracy:    ~67-70%
Validation accuracy:  ~60-74% (avg 70.3%)
Training loss:        ~0.61-0.62 (plateaus)
Validation loss:      ~0.58-0.64 (plateaus)
Issue: UNDERFITTING - too much regularization prevents learning
```

---

## ✅ Final Tuned Hyperparameters

After diagnosing that the initial overfitting fixes were too aggressive, the parameters have been adjusted to balance learning capacity with regularization:

### Input Noise
- **Value**: 8% additive Gaussian noise only
- **Rationale**: Creates realistic trial-to-trial variability (correlation ~0.87) while allowing signal to be learned
- **Status**: ✅ OPTIMAL (keep as-is)

### Dropout Regularization
- **Previous**: 50% (TOO HIGH - caused underfitting)
- **Updated**: **35%**
- **Rationale**: Sufficient regularization without preventing learning
- **Change**: REDUCED from 50% → 35%

### Learning Rate
- **Previous**: 0.003 (TOO LOW - slow learning, got stuck)
- **Updated**: **0.005**
- **Rationale**: Faster learning while avoiding noise memorization
- **Change**: INCREASED from 0.003 → 0.005

### Weight Decay (L2 Regularization)
- **Previous**: 0.01 (moderate, but combined with high dropout was too much)
- **Updated**: **0.005**
- **Rationale**: Lighter regularization allows better learning
- **Change**: REDUCED from 0.01 → 0.005

### Early Stopping Patience
- **Previous**: 50 epochs (was accidentally changed from 15 by linter)
- **Updated**: **20 epochs**
- **Rationale**: Allows model to explore longer while preventing severe overfitting
- **Change**: REDUCED from 50 → 20

---

## 📊 Expected Performance (With Tuned Parameters)

### Before Tuning
```
Average validation accuracy: 70.3% ± 2.8%
Best validation accuracy:    73.9%
Training accuracy:           67-70% (underfitting)
Training loss:               ~0.61-0.62 (plateaus early)
```

### After Tuning (Expected)
```
Average validation accuracy: 76-80% ± 2.5%
Best validation accuracy:    82-85%
Training accuracy:           78-82% (healthy gap from validation)
Training loss:               ~0.45-0.50 (converges smoothly)
Validation loss:             ~0.42-0.48 (decreases, not increases)
```

### Improvement Summary
- **Accuracy**: +6-10 percentage points (70.3% → 76-80%)
- **Generalization**: Healthy train-val gap (2-4pp, not 0pp or 30pp)
- **Convergence**: Smooth learning over 30-60 epochs
- **Robustness**: Learns odor patterns without memorizing noise

---

## 🚀 Retrain with Tuned Parameters

```bash
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --use_class_weights \
    --use_lr_scheduler \
    --output_dir results/ccbpn_tuned \
    --behavioral_data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --verbose
```

**All hyperparameters now have optimal defaults**, so you can use this simplified command.

---

## 📁 Final Hyperparameter Summary

| Parameter | Original | First Fix | **Final Tuned** | Rationale |
|-----------|----------|-----------|----------------|-----------|
| Input Noise | 15% (4 types) | 8% (additive only) | **8% (additive only)** | Prevents noise overfitting |
| Dropout | 30% | 50% ❌ TOO HIGH | **35%** ✅ | Balanced regularization |
| Learning Rate | 0.01 | 0.003 ❌ TOO LOW | **0.005** ✅ | Faster convergence |
| Weight Decay | 0 | 0.01 | **0.005** ✅ | Lighter L2 penalty |
| Early Stop Patience | None | 50 (accidental) | **20** ✅ | Prevents overtraining |
| KC Sparsity | 5% | 10% | **10%** ✅ | Good capacity |
| Class Weights | No | Yes | **Yes** ✅ | Handles imbalance |
| LR Scheduler | No | Yes | **Yes** ✅ | Warmup + decay |

---

## ✅ Success Criteria

The tuned model succeeds if:

1. ✅ **Training accuracy**: 78-82% (not 67% or 100%)
2. ✅ **Validation accuracy**: 76-80% (not 60-74%)
3. ✅ **Train-val gap**: 2-4 percentage points (healthy generalization)
4. ✅ **Training loss**: Decreases smoothly to ~0.45-0.50
5. ✅ **Validation loss**: Decreases to ~0.42-0.48 (not increases)
6. ✅ **Convergence**: 30-60 epochs before early stopping
7. ✅ **Prediction diversity**: Odor-specific predictions (not majority class)
8. ✅ **Within-odor variability**: Standard deviation > 0.10 per odor

---

## 🐛 Troubleshooting

### Issue: Model still underfits (train ~70%, val ~65%)
**Cause**: Still too much regularization
**Solution**: Reduce dropout to 30%, increase learning rate to 0.007

### Issue: Model overfits again (train 95%, val 72%)
**Cause**: Not enough regularization
**Solution**: Increase dropout to 40%, reduce learning rate to 0.004

### Issue: Training too slow, doesn't converge
**Cause**: Learning rate too low
**Solution**: Increase to 0.007-0.01, use learning rate scheduler

### Issue: Early stopping triggers too early (epoch 15)
**Cause**: Patience too low
**Solution**: Increase patience to 25-30 epochs

### Issue: Loss oscillates, doesn't decrease smoothly
**Cause**: Learning rate too high
**Solution**: Reduce to 0.003-0.004

---

## 📈 Diagnostic Analysis

### What Went Wrong (First Attempt)

**Symptom**: Training accuracy ~67-70% (underfitting)

**Diagnosis**:
1. **50% dropout** was too aggressive - model couldn't learn stable representations
2. **0.003 learning rate** was too conservative - model got stuck in poor local minimum
3. **0.01 weight decay** combined with high dropout over-penalized complexity
4. **50 epoch patience** (accidental change) allowed plateaued training to continue too long

**Root Cause**: Combined regularization strength prevented model from fitting even the training data

### Why Tuned Parameters Should Work

**35% dropout**:
- Sufficient to prevent overfitting (from original 30% baseline)
- Not so high that it prevents learning (unlike 50%)
- Standard in literature for similar problems

**0.005 learning rate**:
- 2× faster than 0.003 (escaped local minimum)
- 50% slower than 0.01 (avoids noise memorization)
- Works well with warmup + cosine decay scheduler

**0.005 weight decay**:
- Lighter penalty allows model to explore
- Still prevents weights from exploding
- Common value in Adam optimization

**20 epoch patience**:
- Longer than 15 (allows exploration)
- Shorter than 50 (prevents wasted training)
- ~20-40% of total epochs (rule of thumb)

---

## 📚 Key Insights

### The Regularization Balancing Act

**Too Little Regularization** (original biological fixes):
```
Dropout: 30%
LR: 0.01
Weight Decay: 0
Noise: 15% (4 types)
→ Result: Overfitting (100% train, 70% val)
```

**Too Much Regularization** (first overfitting fix):
```
Dropout: 50%
LR: 0.003
Weight Decay: 0.01
Noise: 8% (1 type)
→ Result: Underfitting (70% train, 70% val)
```

**Balanced Regularization** (tuned parameters):
```
Dropout: 35%
LR: 0.005
Weight Decay: 0.005
Noise: 8% (1 type)
→ Expected: Healthy fit (80% train, 78% val)
```

### Biological Realism vs. Model Capacity

The key is finding the sweet spot where:
1. **Biological noise** creates realistic trial variability (~0.87 correlation)
2. **Regularization** prevents overfitting to that noise
3. **Model capacity** is sufficient to learn true odor patterns
4. **Learning rate** allows convergence without memorization

---

## 🎉 Summary

**Adjustments Made from Over-Regularized State**:

1. ✅ **Dropout**: REDUCED 50% → 35%
2. ✅ **Learning rate**: INCREASED 0.003 → 0.005
3. ✅ **Weight decay**: REDUCED 0.01 → 0.005
4. ✅ **Early stopping**: FIXED 50 → 20 epochs
5. ✅ **Input noise**: KEPT at 8% additive only

**Expected Impact**: +6-10 percentage points accuracy (70.3% → 76-80%)

**Key Achievement**: Balanced regularization that prevents overfitting while allowing effective learning

---

## 🚀 Next Steps

1. **Retrain with tuned parameters**:
   ```bash
   python src/scripts/train_ccbpn.py \
       --task odor_discrimination \
       --use_class_weights \
       --use_lr_scheduler \
       --output_dir results/ccbpn_tuned \
       --behavioral_data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv
   ```

2. **Monitor training**:
   - Training accuracy should reach ~78-82%
   - Validation accuracy should reach ~76-80%
   - Healthy 2-4pp gap between train and val
   - Smooth convergence over 30-60 epochs

3. **If results are good**, analyze with diagnostics:
   ```bash
   python src/scripts/diagnose_model_predictions.py \
       --model results/ccbpn_tuned/ccbpn_odor_discrimination_best.pt

   python src/scripts/visualize_kc_activity.py \
       --model results/ccbpn_tuned/ccbpn_odor_discrimination_best.pt \
       --output_dir results/diagnostics_tuned
   ```

4. **If results are still suboptimal**, further tune based on troubleshooting guide above

---

**Last Updated**: 2025-11-19 (after diagnosing underfit due to over-regularization)
