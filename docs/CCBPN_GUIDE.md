# Connectome-Constrained Behavioral Prediction Network (CCBPN) Guide

## Table of Contents
1. [Overview](#overview)
2. [Biological Motivation](#biological-motivation)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Installation and Setup](#installation-and-setup)
5. [Quick Start](#quick-start)
6. [Training Pipeline](#training-pipeline)
7. [Validation and Analysis](#validation-and-analysis)
8. [Expected Performance](#expected-performance)
9. [Differences from Standard PGCN](#differences-from-standard-pgcn)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Overview

The **Connectome-Constrained Behavioral Prediction Network (CCBPN)** adapts the landmark 2024 Nature study methodology (Lappalainen et al.) from visual motion detection to olfactory learning and discrimination. CCBPN predicts both neural activity and behavioral outcomes from FlyWire connectome data alone, without recording neural activity.

### Key Features

- **Fixed connectivity topology**: Network nodes correspond to real FlyWire neurons, connected only if synaptic connections exist
- **Task-driven optimization**: End-to-end training on behavioral conditioning tasks
- **Recurrent dynamics**: Biologically-constrained neuronal dynamics with dopamine modulation
- **Dual predictions**: Generates both single-neuron responses and behavioral outputs

### What Makes CCBPN Unique

Unlike the original 2024 Nature study (which predicted motion selectivity in T4/T5 neurons), CCBPN predicts:
- **Odor discrimination performance**: Binary classification accuracy curves
- **Memory decay timecourses**: Retention over 24-48 hours
- **State-dependent decision-making**: Context-dependent generalization

directly from ORN→PN→KC→MBON connectivity + behavioral conditioning task constraints.

---

## Biological Motivation

### Vision → Olfaction Adaptation

The 2024 Nature study demonstrated that connectome data alone can predict neural responses across the fly visual system by:

1. Building **Deep Mechanistic Networks (DMNs)**: RNNs where each node is a real neuron
2. **Task-driven optimization**: Training on optic flow prediction
3. **Validating predictions**: Comparing to 24 experimental studies

**Key insight**: Sparse, structured connectivity creates a tight structure-function relationship that makes connectome+task constraints sufficient.

### CCBPN Application to Olfaction

The mushroom body (MB) olfactory learning circuit is ideal for this approach because:

1. **Well-characterized connectivity**: FlyWire provides complete ORN→PN→KC→MBON wiring
2. **Rich behavioral data**: Hundreds of conditioning experiments (Tully-Quinn paradigm)
3. **Sparse coding**: ~5% KC sparsity creates structure-function constraint
4. **Dopamine-gated plasticity**: Clear learning rules for associative memory

---

## Mathematical Formulation

### Network Architecture

CCBPN implements a recurrent neural network with biologically-constrained dynamics:

```
PN → KC → MBON → Behavioral Output
         ↑
      Dopamine
```

### Recurrent Dynamics Equations

**PN layer** (projection neurons):
```
τ_PN · dPN/dt = -PN + I_odor
```

**KC layer** (Kenyon cells with sparsity):
```
τ_KC · dKC/dt = -KC + ReLU(W_PN_KC @ PN)
KC_sparse = top_k(KC, k=0.05·N_KC)  # k-winners-take-all
```

**MBON layer** (mushroom body output neurons):
```
τ_MBON · dMBON/dt = -MBON + ReLU(W_KC_MBON @ KC_sparse)
```

**Behavioral output** (approach/avoid decision):
```
P(approach) = σ(W_readout @ MBON)
```

### Connectivity Constraints

**Critical design constraint**: Connectivity masks remain **FIXED** during training:

```python
W_PN_KC = W_PN_KC_trainable ⊙ M_PN_KC_fixed
W_KC_MBON = W_KC_MBON_trainable ⊙ M_KC_MBON_fixed
```

Where `M` are binary masks from FlyWire connectome, `⊙` is element-wise multiplication.

### Task Loss Function

Composite loss for behavioral conditioning:

```
L_total = L_discrimination + λ₁·L_retention + λ₂·L_generalization
```

- **L_discrimination**: Binary cross-entropy on approach/avoid trials
- **L_retention**: RMSE between predicted and observed memory decay curves
- **L_generalization**: Correlation between chemical similarity and behavioral transfer

---

## Installation and Setup

### Prerequisites

Ensure you have the PGCN repository installed with:
- Python 3.8+
- PyTorch 1.10+
- FlyWire connectivity cache (run extraction scripts)

### FlyWire Data Preparation

Before training CCBPN, ensure FlyWire cache exists:

```bash
# Check cache directory
ls data/cache/
# Should contain: nodes.parquet, edges.parquet, dan_edges.parquet, kc_*.csv, etc.

# If cache missing, run extraction:
python src/scripts/extract_flywire_connectivity.py --output_dir data/cache
```

### Verify Installation

```python
from pgcn.models.ccbpn import ConnectomeConstrainedBehavioralPredictor

# Should not raise errors
model = ConnectomeConstrainedBehavioralPredictor(
    cache_dir="data/cache",
    behavioral_task="odor_discrimination"
)
print(f"Loaded circuit: {model.n_pn} PNs → {model.n_kc} KCs → {model.n_mbon} MBONs")
```

---

## Quick Start

### Basic Usage: Train CCBPN on Odor Discrimination

```bash
# Train with default hyperparameters
python src/scripts/train_ccbpn.py \
    --task odor_discrimination \
    --epochs 100 \
    --cache_dir data/cache

# Training output:
# Loading FlyWire connectivity from data/cache...
# Loaded circuit: 150 PNs → 2500 KCs → 34 MBONs
# Training CCBPN (5-fold cross-validation)...
# Fold 1/5: Best val acc=0.742
# ...
# Best model saved to results/ccbpn_odor_discrimination_best.pt
```

### Validate Predictions

```bash
# Validate against behavioral data
python src/scripts/validate_ccbpn.py \
    --checkpoint results/ccbpn_odor_discrimination_best.pt \
    --behavioral_data data/model_predictions.csv

# Output:
# Behavioral validation:
#   Accuracy: 0.742
#   Precision: 0.768
#   F1 Score: 0.755
```

### Generate Neuron-Level Predictions

```python
from pgcn.analysis.ccbpn_validation import CCBPNValidator
import torch

# Load trained model
validator = CCBPNValidator(
    model_checkpoint="results/ccbpn_odor_discrimination_best.pt",
    cache_dir="data/cache"
)

# Create test odor stimuli (50 odors)
test_odors = torch.randn(50, validator.model.n_pn)

# Predict KC odor tuning curves
kc_predictions = validator.predict_neural_selectivity(
    test_odors=test_odors,
    neuron_type='KC'
)

print(kc_predictions.head())
#    neuron_id  preferred_odor  response_magnitude  sparsity  tuning_width
# 0          0              23                0.85      0.12             6
# 1          1               7                0.92      0.08             4
# ...
```

---

## Training Pipeline

### Hyperparameter Configuration

Key hyperparameters and their biological interpretation:

| Parameter | Default | Biological Meaning |
|-----------|---------|-------------------|
| `--kc_sparsity` | 0.05 | Target KC sparsity (~5% active per odor) |
| `--tau_pn` | 10.0 ms | PN membrane time constant |
| `--tau_kc` | 20.0 ms | KC membrane time constant |
| `--tau_mbon` | 15.0 ms | MBON membrane time constant |
| `--learning_rate` | 0.001 | Optimizer learning rate |
| `--sequence_length` | 50 | Temporal sequence length (time steps) |

### Advanced Training: Memory Retention Task

```bash
python src/scripts/train_ccbpn.py \
    --task memory_retention \
    --epochs 150 \
    --learning_rate 0.0005 \
    --sequence_length 100  # Longer for retention dynamics
```

### Custom Hyperparameter Sweep

```bash
# Example: Test different KC sparsity levels
for sparsity in 0.03 0.05 0.07 0.10; do
    python src/scripts/train_ccbpn.py \
        --task odor_discrimination \
        --kc_sparsity $sparsity \
        --epochs 100 \
        --output_dir results/sparsity_${sparsity}
done
```

### Monitoring Training

Training metrics are saved to `results/ccbpn_<task>_metrics.json`:

```python
import json

# Load training metrics
with open("results/ccbpn_odor_discrimination_metrics.json", 'r') as f:
    metrics = json.load(f)

# Plot learning curves
import matplotlib.pyplot as plt

epochs = [m['epoch'] for m in metrics['fold_results'][0]['metrics']]
train_acc = [m['train_acc'] for m in metrics['fold_results'][0]['metrics']]
val_acc = [m['val_acc'] for m in metrics['fold_results'][0]['metrics']]

plt.plot(epochs, train_acc, label='Train')
plt.plot(epochs, val_acc, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('learning_curve.png')
```

---

## Validation and Analysis

### 1. Behavioral Performance Metrics

Compare model predictions to real fly behavior:

```python
from pgcn.analysis.ccbpn_validation import CCBPNValidator

validator = CCBPNValidator("results/ccbpn_best.pt", "data/cache")

# Validate behavioral performance
metrics = validator.validate_behavioral_performance(
    behavioral_csv="data/model_predictions.csv"
)

print(f"Discrimination accuracy: {metrics['accuracy']:.3f}")
print(f"F1 score: {metrics['f1_score']:.3f}")
```

### 2. Neuron-Level Predictions

Generate KC/MBON odor tuning curves:

```python
# Predict KC selectivity
kc_df = validator.predict_neural_selectivity(
    test_odors=test_odor_matrix,
    neuron_type='KC'
)

# Find broadly-tuned vs. narrowly-tuned KCs
broad_kcs = kc_df[kc_df['tuning_width'] > 10]
narrow_kcs = kc_df[kc_df['tuning_width'] < 3]

print(f"Broadly-tuned KCs: {len(broad_kcs)}")
print(f"Narrowly-tuned KCs: {len(narrow_kcs)}")
```

### 3. Mechanistic Insights: Shapley Analysis

Identify discrimination-critical neurons:

```python
# Compute Shapley values for KC importance
shapley_df = validator.compute_neuron_importance(
    test_odors=test_odors,
    test_labels=test_labels,
    neuron_type='KC',
    n_samples=100  # Monte Carlo samples
)

# Top 20 critical KCs
critical_kcs = shapley_df.nlargest(20, 'shapley_value')
print(critical_kcs[['neuron_id', 'shapley_value', 'rank']])
```

**Biological interpretation**: Analogous to "Why are only 12/19 neurons motion-selective?" in Nature study, this answers: "Which KCs are discrimination-critical despite sparse connectivity?"

---

## Expected Performance

### Behavioral Prediction Benchmarks

Based on initial experiments (to be updated with real results):

| Task | Expected Accuracy | Notes |
|------|------------------|-------|
| Odor Discrimination | ≥70% | Binary classification (approach vs. avoid) |
| Memory Retention | RMSE < 0.15 | 24-hour retention curve fit |
| Cross-Generalization | r > 0.6 | Correlation with chemical similarity |

### Neuron-Level Predictions

- **KC odor selectivity**: 80-90% of KCs should show sparse tuning (< 20% odors)
- **MBON valence coding**: Approach MBONs respond to rewarded odors, avoidance MBONs to punished odors

### Computational Requirements

- **Training time**: ~2-4 hours on single GPU (NVIDIA V100) for 100 epochs
- **Memory**: ~8GB GPU RAM for typical FlyWire circuit (150 PNs, 2500 KCs, 34 MBONs)
- **Inference**: ~10 ms per trial (batch_size=32)

---

## Differences from Standard PGCN Experiments

### Unique Aspects of CCBPN

| Feature | Standard PGCN | CCBPN |
|---------|--------------|-------|
| **Connectivity** | Trainable weights | Fixed topology from FlyWire |
| **Training data** | Synthetic features | Real behavioral conditioning curves |
| **Optimization** | Task-agnostic | Task-driven (behavioral prediction) |
| **Dynamics** | Feedforward or reservoir | Recurrent with temporal integration |
| **Validation** | Against synthetic ground truth | Against real fly behavioral data |
| **Predictions** | Feature classification | Neural activity + behavior |

### Relationship to Other PGCN Modules

- **OlfactoryCircuit**: CCBPN uses similar sparse PN→KC→MBON architecture, but adds recurrent dynamics and dopamine modulation
- **DrosophilaReservoir**: CCBPN enforces FlyWire connectivity masks (not random sparse), and trains on behavioral tasks
- **MultiTaskDrosophilaModel**: CCBPN is task-specific (behavioral conditioning), not multi-task

**Backward compatibility**: CCBPN can coexist with all existing PGCN modules without conflicts.

---

## Troubleshooting

### Common Issues

#### 1. `FileNotFoundError: Cache directory not found`

**Solution**: Run FlyWire extraction scripts before training:
```bash
python src/scripts/extract_flywire_connectivity.py --output_dir data/cache
```

#### 2. Low Training Accuracy (< 60%)

**Possible causes**:
- Insufficient training epochs (try --epochs 200)
- Learning rate too high/low (try 0.0001 to 0.01)
- Behavioral data quality (verify labels in CSV)

**Debugging**:
```bash
# Check behavioral data distribution
python -c "
import pandas as pd
df = pd.read_csv('data/model_predictions.csv')
print('Approach trials:', (df['prediction'] > 0.5).sum())
print('Avoid trials:', (df['prediction'] < 0.5).sum())
"
```

#### 3. `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size or use CPU:
```bash
python src/scripts/train_ccbpn.py \
    --batch_size 8  # Reduce from default 32
    --device cpu    # Use CPU if GPU insufficient
```

#### 4. KC Sparsity Not Enforced

**Check**:
```python
model = ConnectomeConstrainedBehavioralPredictor(cache_dir="data/cache")
outputs = model(odor_seq, dopa_sig, return_intermediates=True)
print(f"Mean KC sparsity: {outputs['sparsity_fraction'].mean():.3f}")
```

If sparsity >> 0.05, check `kc_sparsity_target` parameter.

#### 5. Connectivity Constraints Violated After Training

**Solution**: Always call `model.enforce_connectivity_constraints()` after `optimizer.step()`:
```python
optimizer.step()
model.enforce_connectivity_constraints()  # Critical!
```

---

## References

### Primary Reference

**Lappalainen et al. (2024)** "Connectome-constrained networks predict neural activity across the fly visual system" *Nature* 634:1132-1140
DOI: [10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3)

### PGCN Repository

**PGCN**: Plasticity-Guided Connectome Network
- GitHub: [colehanan1/Plasticity-Guided-Connectome-Network-PGCN](https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN)
- README: [Project overview](../README.md)

### FlyWire Connectome

**Dorkenwald et al. (2023)** "Neuronal wiring diagram of an adult brain" *bioRxiv*
DOI: [10.1101/2023.06.27.546656](https://doi.org/10.1101/2023.06.27.546656)

### Behavioral Conditioning

**Tully & Quinn (1985)** "Classical conditioning and retention in normal and mutant Drosophila melanogaster" *J Comp Physiol A* 157:263-277

---

## Citation

If you use CCBPN in your research, please cite:

```bibtex
@software{ccbpn2025,
  title = {Connectome-Constrained Behavioral Prediction Network (CCBPN)},
  author = {PGCN Contributors},
  year = {2025},
  url = {https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN},
  note = {Part of the Plasticity-Guided Connectome Network (PGCN) project}
}

@article{lappalainen2024connectome,
  title={Connectome-constrained networks predict neural activity across the fly visual system},
  author={Lappalainen, Janne K and others},
  journal={Nature},
  volume={634},
  pages={1132--1140},
  year={2024}
}
```

---

## Contributing

Contributions to CCBPN are welcome! Please see the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Suggested Improvements

- [ ] Add real DoOR database integration for odor stimuli
- [ ] Implement online plasticity during inference (meta-learning)
- [ ] Add attention mechanisms for state-dependent modulation
- [ ] Extend to other behavioral tasks (spatial learning, courtship)
- [ ] Integrate with experimental neural recordings for validation

---

## Support

For questions or issues:
- Open a GitHub issue: [PGCN Issues](https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN/issues)
- Check existing documentation: [docs/](../docs/)
- Review troubleshooting section above

---

**Last updated**: 2025-01-18
**CCBPN version**: 1.0.0
**PGCN version**: Compatible with latest main branch
