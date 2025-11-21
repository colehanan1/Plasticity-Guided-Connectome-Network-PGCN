# Setup Guide: Realistic Behavioral Training Protocol

## Quick Start

To run the realistic training protocol, you need to set up the environment and generate cache data first.

---

## Step 1: Install PGCN Package

```bash
# Make sure you're in the PGCN conda environment
conda activate PGCN

# Install the package in editable mode
pip install -e .[dev]
```

This installs all dependencies including:
- numpy
- pandas
- scipy
- matplotlib
- torch

---

## Step 2: Generate Sample Cache (Option A - Quick Test)

For quick testing without FlyWire data:

```bash
# Generate synthetic connectome cache
python scripts/setup_sample_cache.py --cache-dir data/cache

# This creates:
#   data/cache/nodes.parquet (neurons)
#   data/cache/edges.parquet (PN→KC and KC→MBON connections)
#   data/cache/dan_edges.parquet (DAN connections)
#   data/cache/meta.json (metadata)
```

---

## Step 2: Generate Real Cache (Option B - Production)

For real FlyWire data:

### A. If you have FlyWire credentials:

```bash
# Set up FlyWire token
pgcn-auth --token "your-flywire-token-here"

# Generate cache from FlyWire
pgcn-cache --datastack flywire_fafb_production --mv 783 --out data/cache/
```

### B. If you have local FlyWire CSV exports:

```bash
# Place CSVs in data/flywire/:
#   - connections_princeton.csv.gz
#   - consolidated_cell_types.csv.gz
#   - classification.csv.gz
#   - neurons.csv.gz
#   - names.csv.gz
#   - processed_labels.csv.gz

# Generate cache from local data
pgcn-cache --local-data data/flywire --out data/cache/
```

### C. If you have Codex exports:

```bash
# Download from https://codex.flywire-daf.com
# Place in ~/Downloads/fafb_codex_783/

# Convert to PGCN cache
pgcn-codex-import \
  --neurons ~/Downloads/fafb_codex_783/neurons.csv.gz \
  --synapses ~/Downloads/fafb_codex_783/synapses.csv.gz \
  --out data/cache/
```

---

## Step 3: Run Realistic Training Protocol

```bash
# Run complete 3-phase training
python scripts/realistic_behavioral_training.py \
    --cs-odor benzaldehyde \
    --test-odor 1-hexanol \
    --cache-dir data/cache \
    --output-dir results/realistic_training

# Expected runtime: 2-5 minutes
```

**Expected output:**
```
======================================================================
REALISTIC FLY BEHAVIORAL TRAINING PROTOCOL
======================================================================
CS Odor (rewarded): benzaldehyde
Test Odor (unrewarded): 1-hexanol

🔧 SETUP: Loading FlyWire connectome...
  ✓ Loaded 150 PNs
  ✓ Loaded 2000 KCs
  ✓ Loaded 44 MBONs
  ✓ Loaded 100 DANs

======================================================================
PHASE 1: CLASSICAL CONDITIONING (3 trials)
======================================================================
...
```

---

## Step 4: Generate Figure 4

```bash
# Generate behavioral validation figure
python scripts/generate_figure4_predictions.py \
    --results-dir results/realistic_training \
    --output-dir results/figure4_validation

# Outputs:
#   - fig4_behavioral_validation_realistic.png
#   - fig4_behavioral_validation_realistic.pdf
#   - observed_vs_predicted.csv
#   - behavioral_validation_report.txt
```

---

## Step 5: Validate Timing (Optional)

```bash
# Verify temporal trial implementation
python scripts/validate_temporal_trial.py

# Expected: ✅ ALL TESTS PASSED
```

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'numpy'`

**Solution:** Install the package first
```bash
pip install -e .[dev]
```

### Error: `FileNotFoundError: nodes.parquet not found`

**Solution:** Generate cache data (see Step 2)
```bash
python scripts/setup_sample_cache.py
```

### Error: `FileNotFoundError: test_results.csv not found`

**Solution:** Run training first before generating Figure 4
```bash
python scripts/realistic_behavioral_training.py
```

### Low R² (<0.70)

**Possible causes:**
1. Learning rate too high/low
2. Insufficient training trials
3. Random initialization variance

**Solutions:**
```bash
# Increase Phase 1 trials (edit script line ~590)
# Or adjust learning rate
python scripts/realistic_behavioral_training.py --learning-rate 0.01
```

### No learning (MBON stays flat)

**Check:**
1. Reward profile is not all zeros
2. KC activation is ~5% of n_kc
3. Plasticity updates are being applied

**Debug:**
```python
# Add debug prints in realistic_behavioral_training.py
print(f"Reward profile sum: {reward_profile.sum()}")
print(f"KC activation: {kc_activation.sum()} / {len(kc_ids)}")
```

---

## File Structure After Setup

```
Plasticity-Guided-Connectome-Network-PGCN/
├── data/
│   └── cache/               # Generated cache
│       ├── nodes.parquet
│       ├── edges.parquet
│       ├── dan_edges.parquet
│       └── meta.json
├── results/
│   ├── realistic_training/  # Training results
│   │   ├── test_results.csv
│   │   ├── response_summary.csv
│   │   ├── phase1_classical.csv
│   │   └── phase2_operant.csv
│   └── figure4_validation/  # Figure 4 outputs
│       ├── fig4_behavioral_validation_realistic.png
│       ├── fig4_behavioral_validation_realistic.pdf
│       ├── observed_vs_predicted.csv
│       └── behavioral_validation_report.txt
├── scripts/
│   ├── realistic_behavioral_training.py
│   ├── generate_figure4_predictions.py
│   ├── validate_temporal_trial.py
│   └── setup_sample_cache.py
└── docs/
    ├── REALISTIC_TRAINING_PROTOCOL.md
    ├── FIGURE4_BEHAVIORAL_VALIDATION_EXPLAINED.md
    └── SETUP_REALISTIC_TRAINING.md (this file)
```

---

## Next Steps

1. **Run training** with sample cache
2. **Validate results** - Check R² > 0.85
3. **Generate figures** for publication
4. **Iterate** - Adjust parameters if needed
5. **Document** - Save results for paper

---

## Expected Results

After successful run:

**Response rates (with sample cache):**
```
Benzaldehyde (CS):      ~0.60-0.70
Ethyl butyrate:         ~0.45-0.55
3-Octanol:              ~0.40-0.50
Linalool:               ~0.25-0.35
Hexanol (discrimination): ~0.15-0.25
```

**Validation metrics:**
```
R² = 0.85-0.95  (excellent)
Pearson r = 0.92-0.97
RMSE = 0.05-0.10
MAE = 0.03-0.08
```

---

## Additional Resources

- **Full Protocol Documentation:** `docs/REALISTIC_TRAINING_PROTOCOL.md`
- **Figure 4 Explanation:** `docs/FIGURE4_BEHAVIORAL_VALIDATION_EXPLAINED.md`
- **Main README:** `README.md`
- **Model Integration:** `docs/model_integration_status.md`

---

## Support

If you encounter issues:

1. **Check this setup guide** first
2. **Review error messages** - they usually indicate the problem
3. **Verify environment:** `conda list | grep numpy` should show numpy installed
4. **Check cache exists:** `ls data/cache/*.parquet` should list 3 files
5. **Open GitHub issue** with full error traceback

---

**Last updated:** 2025-11-11
**Version:** 1.0.0
