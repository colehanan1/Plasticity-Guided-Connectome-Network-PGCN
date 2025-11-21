# 🚨 URGENT: Test the Dopamine Leakage Fix

## What We Just Found

Your model was **cheating** - it was seeing the behavioral label encoded as a dopamine signal. That's why it got 100% accuracy even on random labels!

**The bug:** Line 221 in `train_ccbpn_recurrent.py`
```python
is_cs_plus = row.get('prediction', 0) > 0.5  # Using the label itself!
```

**The fix:** Dopamine signals now set to zero (no leakage possible)

---

## What You Need to Do RIGHT NOW

### Step 1: Test on Scrambled Labels (30 min)

This verifies the fix is working:

```bash
cd ~/Plasticity-Guided-Connectome-Network-PGCN

python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions_SCRAMBLED.csv \
    --cache-dir data/cache \
    --output-dir results/ccbpn_FIXED_SCRAMBLED \
    --epochs 50 \
    --n-folds 5
```

**CRITICAL: Watch the validation accuracy!**

#### ✅ SUCCESS: ~50% accuracy
- Fix is working! Model can't predict random labels anymore
- Proceed to Step 2

#### ❌ FAILURE: >70% accuracy
- There's still label leakage somewhere
- Report back immediately - we need to find the remaining bug

---

### Step 2: Train on Real Data (1 hour)

Only do this if Step 1 gives ~50% accuracy:

```bash
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --cache-dir data/cache \
    --output-dir results/ccbpn_FIXED_REAL \
    --epochs 50 \
    --n-folds 5
```

**Expected:** 74-78% validation accuracy

This would show the model is actually learning from odor patterns!

---

### Step 3: Report Back

Tell me:
```
Scrambled labels accuracy: [XX%]
Real data accuracy: [XX%]
```

Then we'll know if the model is truly fixed! 🔬

---

## Files to Read

- `DOPAMINE_LEAKAGE_BUG_REPORT.md` - Full technical explanation
- `DIAGNOSTIC_GUIDE.md` - Original diagnostic tests
- `NEXT_STEPS.md` - Previous instructions (now superseded)

---

## Why This Matters

Before fix:
- Model: "I see dopamine → I predict approach" (cheating)
- Accuracy: 100% (meaningless)

After fix:
- Model: "I see benzaldehyde pattern → I predict approach based on past trials" (learning)
- Accuracy: 74-78% (realistic)

**The scrambled labels test caught this bug before you published bad results!** This is why systematic testing is so important. 🎯

---

## Quick Commands

```bash
# Activate environment
conda activate PGCN

# Test fix on scrambled data
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions_SCRAMBLED.csv \
    --output-dir results/ccbpn_FIXED_SCRAMBLED \
    --epochs 50 --n-folds 5

# If successful, train on real data
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --output-dir results/ccbpn_FIXED_REAL \
    --epochs 50 --n-folds 5
```

**GO! Run the scrambled labels test and report back!** ⚡
