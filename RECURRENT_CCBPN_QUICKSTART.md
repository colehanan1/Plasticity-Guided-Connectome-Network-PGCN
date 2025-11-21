# Recurrent CCBPN - Quick Start Guide

## What Was Implemented

A **recurrent context memory system** for the CCBPN that enables trial-to-trial learning across multiple experimental datasets. The model uses an LSTM to maintain context across trials, allowing it to learn different odor-outcome associations in different contexts (e.g., hexanol=CS+ in opto_hex but CS- in opto_benz).

## Files Created

1. **`src/pgcn/models/ccbpn_recurrent.py`** - Main model with LSTM context memory
2. **`src/scripts/train_ccbpn_recurrent.py`** - Training script with sequential data loading
3. **`tests/test_ccbpn_recurrent.py`** - Sanity check tests
4. **`docs/CCBPN_RECURRENT_IMPLEMENTATION.md`** - Comprehensive documentation

## Quick Test (5 minutes)

```bash
# 1. Install dependencies (if needed)
pip install torch numpy pandas scikit-learn tqdm

# 2. Run sanity checks
python tests/test_ccbpn_recurrent.py

# 3. Quick training test with 10 flies
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --cache-dir data/cache \
    --output-dir results/ccbpn_recurrent_test \
    --epochs 10 \
    --context-dim 32 \
    --max-flies 10
```

## Full Training (2-4 hours)

```bash
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --cache-dir data/cache \
    --output-dir results/ccbpn_recurrent_final \
    --epochs 100 \
    --context-dim 64 \
    --lr 0.001 \
    --use-class-weights \
    --use-lr-scheduler \
    --n-folds 5 \
    --patience 20
```

## Expected Results

- **Baseline (no context)**: ~70% accuracy
- **With recurrent context**: 74-80% accuracy
- **Improvement**: +4 to +10 percentage points

## Architecture Overview

```
PN → KC → MBON (base CCBPN)
        ↓
   LSTM context memory ← [previous trial outcome]
        ↓
   Context modulation
        ↓
   Behavioral output
```

**Key Innovation**: LSTM maintains hidden state across trials within each fly, enabling context-dependent learning without explicit context labels.

## Troubleshooting

### If tests fail:
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check imports
python -c "from pgcn.models.ccbpn_recurrent import CCBPNWithRecurrentContext; print('OK')"
```

### If training crashes:
- **GPU OOM**: Reduce batch size or context_dim
- **NaN loss**: Lower learning rate or stronger gradient clipping
- **No improvement**: Check class weights, try higher LSTM learning rate

## Next Steps

1. ✅ **Phase 1-2**: Architecture design and implementation (DONE)
2. ✅ **Phase 3.1**: Sanity check tests (DONE)
3. ⏳ **Phase 3.2**: Run tests (READY)
4. ⏳ **Phase 4**: Full training and evaluation
5. ⏳ **Analysis**: Compare with baselines, visualize context evolution

## Documentation

See `docs/CCBPN_RECURRENT_IMPLEMENTATION.md` for:
- Detailed architecture description
- Biological motivation
- Training tips and hyperparameters
- Visualization and analysis code
- Common issues and solutions

## Contact

For questions:
- Review the comprehensive documentation in `docs/`
- Check code comments in model and training script
- Run sanity checks to verify installation
