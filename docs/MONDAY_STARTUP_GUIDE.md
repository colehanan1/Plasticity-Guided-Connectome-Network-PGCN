# Monday Startup Guide - PGCN Enhanced System

**Status: Production Ready ✓**
**Last Updated: 2025-10-31**
**System: Enhanced PGCN with 40K+ Real FlyWire Neurons**

---

## Quick Start (One Command!)

```bash
# Run Experiment 1 (Veto Gate Blocking) with default parameters
python scripts/monday_startup_training.py --experiment veto

# Or customize trial counts:
python scripts/monday_startup_training.py --phase1-trials 10 --phase2-trials 30
```

**Expected Runtime:** ~5-10 minutes for default parameters
**Output Location:** `results/monday/`

---

## What You'll Get

After running the script, you'll find in `results/monday/`:

1. **experiment_1_veto_gate_results.png** - 4-panel visualization:
   - Phase 2 learning curves (RPE over trials)
   - Veto gating modulation over time
   - Final test responses bar chart
   - Summary metrics (Blocking Index, Veto Efficacy, Gating Suppression)

2. **experiment_summary.json** - Complete metrics in JSON format

3. **phase2_trials.csv** - Trial-by-trial data for detailed analysis

4. **README.md** - Interpretation guide for your results

---

## System Components Loaded

When you run the startup script, it loads:

```
✓ 26,632 Projection Neurons (PNs) - Olfactory input
✓ 5,374 Kenyon Cells (KCs) - Sparse expansion coding
✓ 44 Mushroom Body Output Neurons (MBONs) - Behavioral output
✓ 231 Dopaminergic Neurons (DANs) - Reward signals
✓ 3,829 Local Interneurons (LNs) - VETO GATE PATHWAY
✓ 1,162 Lateral Horn Neurons (LH) - Innate valence
✓ 66 Motor Neurons - Proboscis extension reflex (PER)
✓ 1,926 Ascending Neurons (ANs) - State modulation
✓ 1,303 Descending Neurons (DNs) - Motor commands

TOTAL: 40,567 neurons loaded!
```

---

## Interpreting Results

### Key Metrics

1. **Blocking Index** (primary metric)
   - **> 0.2**: Strong blocking effect detected ✓
   - **0.0 to 0.2**: Weak blocking effect
   - **< 0**: No blocking (OdorA learned more than OdorB)

2. **Veto Efficacy** (0.0 to 1.0)
   - Measures how strongly the GABAergic veto pathway activates
   - **> 0.8**: Strong veto activation
   - **0.5 - 0.8**: Moderate veto
   - **< 0.5**: Weak veto

3. **Mean Gating Suppression** (0.0 to 1.0)
   - Fraction of plasticity blocked by veto gate
   - **> 0.7**: Strong plasticity suppression
   - **0.3 - 0.7**: Moderate suppression
   - **< 0.3**: Weak suppression

### Expected Monday Outcome

**Hypothesis**: GABAergic local interneurons (LNs) implement a veto gate that blocks PN→KC plasticity when the veto pathway is activated, causing blocking in the Kamin blocking paradigm.

**Predicted Results** (if hypothesis is correct):
- Blocking Index: 0.3-0.7 (moderate to strong blocking)
- Veto Efficacy: 0.6-1.0 (strong veto activation)
- OdorA (veto glomerulus) response < OdorB (control glomerulus)

**If you see these results:** Your hypothesis is supported! The GABAergic veto successfully blocked learning.

---

## Advanced Usage

### Customizing Experiments

```python
from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.enhanced_olfactory_circuit import EnhancedOlfactoryCircuit

# Load circuit
loader = CircuitLoader(cache_dir="data/cache")
conn = loader.load_connectivity_matrix(
    normalize_weights="row",
    include_extended=True
)

# Create enhanced circuit
circuit = EnhancedOlfactoryCircuit(
    connectivity=conn,
    kc_sparsity_target=0.05,
    enable_ln_modulation=True,  # Enable veto gate
    gaba_strength=1.0,  # Adjust veto strength (0.0-2.0)
)

# Test blocking effect at different strengths
blocking_curve = circuit.measure_blocking_effect(
    test_glomeruli=["DA1"],
    blocking_strengths=[0.0, 0.25, 0.5, 0.75, 1.0]
)

print(blocking_curve)
```

### Testing Different Glomeruli

```bash
# Modify the script to test different odor combinations
# Edit scripts/monday_startup_training.py, line ~120:

results = runner.run_experiment_1_veto_gate(
    circuit,
    veto_glomerulus="VA1d",  # Change veto odor
    control_glomerulus="DC2",  # Change control odor
)
```

---

## Troubleshooting

### Problem: "Cache directory not found"
**Solution:**
```bash
# Verify data/cache exists and has CSV files
ls data/cache/*.csv

# If missing, you need to run extraction scripts first:
python scripts/extract_alpn_projection_neurons.py
python scripts/extract_extended_circuit.py
```

### Problem: Low blocking index (< 0.1)
**Possible causes:**
1. **Veto strength too low** - Increase `gaba_strength` parameter
2. **Insufficient training trials** - Increase `--phase2-trials`
3. **Learning rate too high** - Adjust learning rate in plasticity model

**Quick fix:**
```bash
# Increase trials and use stronger veto
python scripts/monday_startup_training.py \
    --phase1-trials 20 \
    --phase2-trials 50
```

### Problem: "No module named 'pgcn'"
**Solution:**
```bash
# Ensure PYTHONPATH includes src/
export PYTHONPATH=src:$PYTHONPATH
python scripts/monday_startup_training.py
```

---

## Next Steps After Monday Results

### If Blocking is Confirmed ✓

1. **Run Experiment 2 (Microsurgery)** - Prove causality by editing veto synapses
   ```bash
   python scripts/run_experiment_2_microsurgery.py
   ```

2. **Run Experiment 3 (Eligibility Traces)** - Test temporal credit assignment
   ```bash
   python scripts/run_experiment_3_eligibility.py
   ```

3. **Run Experiment 6 (Shapley Analysis)** - Identify specific blocker neurons
   ```bash
   python scripts/run_experiment_6_shapley.py
   ```

### If Blocking is Weak/Absent

1. **Analyze veto pathway activation**
   - Check if veto glomerulus → LN → KC pathway is activating
   - Verify GABA neurotransmitter neurons are being recruited

2. **Try parameter sweeps**
   - Test different `gaba_strength` values: [0.5, 1.0, 1.5, 2.0]
   - Test different learning rates: [0.005, 0.01, 0.02]
   - Test different veto glomeruli (some may have stronger connections)

3. **Examine circuit connectivity**
   - Verify PN→LN and LN→KC matrices have sufficient connections
   - Check that veto glomerulus PNs connect to GABAergic LNs

---

## File Structure Reference

```
PGCN/
├── data/cache/              # Extracted FlyWire neurons
│   ├── alpn_extracted.csv   # 482 PNs
│   ├── kc_*.csv             # 5,374 KCs (8 subtypes)
│   ├── ln_all.csv           # 3,829 LNs ← VETO GATE
│   ├── lh_all.csv           # 1,162 LH neurons
│   ├── motor_*.csv          # 90 motor neurons
│   ├── an_all.csv           # 1,926 ANs
│   └── dn_all.csv           # 1,303 DNs
│
├── scripts/
│   ├── monday_startup_training.py  ← RUN THIS
│   └── visualize_pgcn_circuit.py   # Visualize full circuit
│
├── src/pgcn/
│   ├── models/
│   │   ├── enhanced_olfactory_circuit.py  # Master integration
│   │   ├── enhanced_layers.py             # LN/LH/Motor layers
│   │   └── olfactory_circuit.py           # Core PN→KC→MBON
│   └── experiments/
│       ├── experiment_1_veto_gate.py      # Blocking experiment
│       └── ...
│
└── results/monday/          # OUTPUT DIRECTORY
    ├── experiment_1_veto_gate_results.png
    ├── experiment_summary.json
    ├── phase2_trials.csv
    └── README.md
```

---

## Support & References

- **Code**: `src/pgcn/models/enhanced_olfactory_circuit.py`
- **Experiments**: `src/pgcn/experiments/experiment_1_veto_gate.py`
- **Data Extraction**: `docs/ENHANCED_COMPONENTS_SUMMARY.md`
- **System Architecture**: `docs/ENHANCED_COMPONENTS_GUIDE.md`

**Questions?** Check `docs/` for detailed guides on each component.

---

## Monday Success Checklist

- [ ] Run `python scripts/monday_startup_training.py`
- [ ] Verify output in `results/monday/`
- [ ] Check blocking index > 0.2 for hypothesis confirmation
- [ ] Review plots in `experiment_1_veto_gate_results.png`
- [ ] Read generated `README.md` for interpretation
- [ ] If successful, proceed to Experiments 2, 3, 6
- [ ] If unsuccessful, troubleshoot with parameter sweeps

**Goal:** Quantitative evidence for PN→KC pathway blocking with real FlyWire data ✓

---

**Good luck with your Monday experiments!** 🚀
