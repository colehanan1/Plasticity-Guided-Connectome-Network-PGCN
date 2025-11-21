# CCBPN with Recurrent Context Memory - Implementation Guide

## Overview

This document describes the implementation of **recurrent context memory** for the Connectome-Constrained Behavioral Predictor Network (CCBPN). This enhancement enables the model to maintain trial-to-trial memory, allowing it to learn context-dependent associations across multiple experimental datasets.

## Biological Motivation

Real *Drosophila* maintain trial-to-trial memory through:
- **Synaptic tags**: Short-term molecular markers at KC→MBON synapses
- **Dopaminergic plasticity**: Reward/punishment signals that accumulate across trials
- **Context learning**: Flies learn different associations in different contexts

This implementation mimics these biological mechanisms using an LSTM that:
1. Accumulates context from previous trial outcomes
2. Modulates current trial processing based on learned context
3. Enables context-specific odor-outcome associations

### Example Use Case

In multi-dataset training:
- **opto_hex**: hexanol = CS+ (rewarded) → approach
- **opto_benz**: hexanol = CS- (unrewarded) → avoid

Without context memory, the model cannot learn both associations simultaneously. With context memory, the LSTM learns to detect the experimental context and adjust its predictions accordingly.

## Architecture

### Model Structure

```
Input: Odor Sequence (batch, time, n_pn)
   ↓
Base CCBPN (PN → KC → MBON)
   ↓
MBON Activity (batch, time, n_mbon)
   ↓
Aggregate over time (mean)
   ↓
Context LSTM ← [MBON | Dopamine | Previous Outcome]
   ↓
Context Vector (batch, context_dim)
   ↓
Context Gate (learns when to use memory)
   ↓
Context Modulation of MBON
   ↓
Behavioral Output (batch,)
```

### Key Components

#### 1. Base CCBPN (`DrosophilaReservoir`)
- **PN → KC**: Sparse connectivity from connectome (~6-8 PNs per KC)
- **KC sparsity**: k-winners-take-all (~5% active)
- **KC → MBON**: Linear readout with learnable weights
- **Frozen PN→KC weights**: Preserves connectome structure

#### 2. Context Memory (`nn.LSTM`)
- **Input**: MBON activity + dopamine + previous outcome (n_mbon + 2)
- **Hidden state**: 64-dimensional context embedding
- **Purpose**: Accumulates trial history to detect context

#### 3. Context Gate (`nn.Sequential`)
- **Input**: Context + current MBON activity
- **Output**: Gate value ∈ [0, 1]
- **Purpose**: Learns when to rely on memory vs. current input

#### 4. Context Modulation (`nn.Sequential`)
- **Input**: Context vector
- **Output**: Modulation signal (same dimension as MBON)
- **Purpose**: Adjusts MBON activity based on learned context

## Implementation Files

### 1. Model: `src/pgcn/models/ccbpn_recurrent.py`

Main model class: `CCBPNWithRecurrentContext`

**Key Methods:**
- `forward(odor_sequences, dopamine_signals, hidden_state, previous_outcome)`
  - Processes one trial with recurrent context
  - Returns: behavioral_output, hidden_state, context, gate_value, etc.

- `reset_context(batch_size, device)`
  - Initializes hidden state for new fly/session
  - Returns: (h_0, c_0) for LSTM

- `freeze_ccbpn_core()` / `unfreeze_ccbpn_core()`
  - Control whether to train base CCBPN or only context layers

**Parameters:**
- `n_pn`: 150 (projection neurons)
- `n_kc`: 2000 (Kenyon cells)
- `n_mbon`: 44 (mushroom body output neurons)
- `cache_dir`: Path to FlyWire connectivity cache
- `kc_sparsity`: 0.05 (5% KC activity)
- `context_dim`: 64 (context embedding size)
- `use_gate`: True (enable learned gating)
- `dropout`: 0.2 (regularization)

### 2. Training Script: `src/scripts/train_ccbpn_recurrent.py`

**Features:**
- Sequential data loading (preserves trial order within flies)
- Truncated BPTT (detaches hidden state after each trial)
- Cross-validation with fly-level splits
- Early stopping with patience
- Learning rate scheduling (optional)
- Class weight balancing (optional)
- Gradient clipping (prevents exploding gradients)

**Key Classes:**
- `SequentialBehavioralDataset`
  - Loads behavioral CSV
  - Groups trials by fly
  - Preserves temporal order
  - Generates odor sequences and dopamine signals

**Training Loop:**
```python
for fly in shuffle(flies):
    hidden_state = None
    previous_outcome = None

    for trial in fly_trials:
        outputs = model(odor, dopamine, hidden_state, previous_outcome)
        loss = criterion(outputs['behavioral_output'], label)
        loss.backward()
        clip_gradients(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update for next trial
        previous_outcome = label.detach()
        hidden_state = tuple(h.detach() for h in outputs['hidden_state'])
```

### 3. Tests: `tests/test_ccbpn_recurrent.py`

**Sanity Checks:**
1. **Output Shape Verification**: Ensures all outputs have correct dimensions
2. **Context Reset**: Verifies hidden state initialization
3. **Hidden State Propagation**: Confirms context affects predictions
4. **Gradient Flow**: Checks gradients reach LSTM and context layers
5. **Context Learning**: Tests model can learn simple context-dependent task

## Usage

### Quick Test (10 flies, 10 epochs)

```bash
python src/scripts/train_ccbpn_recurrent.py \
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
    --cache-dir data/cache \
    --output-dir results/ccbpn_recurrent_test \
    --epochs 10 \
    --context-dim 32 \
    --max-flies 10
```

**Expected runtime**: ~5-10 minutes
**Expected result**: Code runs without errors, some improvement in accuracy

### Full Training

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

**Expected runtime**: 2-4 hours (depending on hardware)
**Expected result**: 74-80% validation accuracy (compared to ~70% baseline)

### Running Sanity Checks

```bash
# Install dependencies (if needed)
pip install torch numpy pandas scikit-learn tqdm

# Run tests
python tests/test_ccbpn_recurrent.py
```

**Expected output**: All 5 tests should pass

## Expected Results

### Performance Targets

| Scenario | Baseline (No Context) | With Recurrent Context | Improvement |
|----------|------------------------|------------------------|-------------|
| **Minimum viable** | 70% | 74% | +4pp |
| **Good** | 70% | 76-78% | +6-8pp |
| **Excellent** | 70% | 78-80% | +8-10pp |

### Why Not Higher?

Several factors limit maximum achievable accuracy:

1. **Marginal PN discriminability**: Worst odor pair correlation r=0.783
2. **Noise**: 8% PN noise reduces signal-to-noise ratio to ~6
3. **Class imbalance**: 67.7% baseline (need class weights)
4. **Limited data**: 1110 trials across 64 flies
5. **Biological realism**: Real flies don't achieve 100% accuracy either

### Success Criteria

**Minimum Viable:**
- ✅ Code runs without errors
- ✅ Validation accuracy ≥ 74% (+4pp over baseline)
- ✅ Context demonstrably affects predictions

**Good:**
- ✅ Validation accuracy 76-78% (+6-8pp)
- ✅ Outperforms single-dataset by 1-2pp
- ✅ Context evolves meaningfully across trials

**Excellent:**
- ✅ Validation accuracy 78-80% (+8-10pp)
- ✅ Matches single-dataset performance on multi-dataset task
- ✅ Biological analysis shows context tracks reward history

## Training Tips

### Hyperparameter Tuning

**Context Dimension:**
- Start with 64 (good balance)
- Try 32 if overfitting
- Try 128 if underfitting and have enough data

**Learning Rate:**
- Default: 0.001 (works well for most cases)
- LSTM may benefit from 3× higher LR: use separate parameter groups
```python
optimizer = Adam([
    {'params': model.ccbpn_core.parameters(), 'lr': 0.001},
    {'params': model.context_memory.parameters(), 'lr': 0.003},
])
```

**Gradient Clipping:**
- Default: max_norm=1.0 (prevents exploding gradients)
- Reduce to 0.5 if training is unstable
- Increase to 2.0 if gradients are too small

### Common Issues & Solutions

#### 1. Exploding Gradients
**Symptom**: Loss becomes NaN after a few iterations

**Solutions:**
```python
# Stronger gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# Lower learning rate
optimizer = Adam(model.parameters(), lr=0.0003)

# Add layer normalization
self.layer_norm = nn.LayerNorm(context_dim)
```

#### 2. Context Not Learning
**Symptom**: Gate values always near 0 or 1, no context effect

**Solutions:**
```python
# Higher LR for LSTM
optimizer = Adam([
    {'params': model.context_memory.parameters(), 'lr': 0.003},
    {'params': model.behavior_head.parameters(), 'lr': 0.001},
])

# Reduce context dimension (simpler model)
context_dim = 32

# Try without gating
model = CCBPNWithRecurrentContext(..., use_gate=False)
```

#### 3. Memory Leak (GPU OOM)
**Symptom**: GPU memory grows over time, eventually crashes

**Solutions:**
```python
# Detach hidden state after each trial (already implemented)
hidden_state = tuple(h.detach() for h in hidden_state)

# Use gradient checkpointing for very long sequences
# (not implemented yet, but can be added if needed)
```

#### 4. Vanishing Gradients
**Symptom**: Loss doesn't decrease, gradients near zero

**Solutions:**
```python
# Layer normalization
self.layer_norm = nn.LayerNorm(context_dim)

# LSTM dropout (if using multi-layer LSTM)
nn.LSTM(..., num_layers=2, dropout=0.1)

# Skip connections (more complex architecture)
```

## Visualization and Analysis

### Context Evolution Plots

After training, analyze how context changes across trials:

```python
from pgcn.models.ccbpn_recurrent import CCBPNWithRecurrentContext
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load trained model
model = CCBPNWithRecurrentContext(...)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Process one fly's sequence
contexts = []
predictions = []
labels = []

hidden_state = None
previous_outcome = None

for trial in fly_sequence:
    outputs = model(odor, dopamine, hidden_state, previous_outcome)
    contexts.append(outputs['context'].cpu().numpy())
    predictions.append(outputs['behavioral_output'].item())
    labels.append(trial_label)

    hidden_state = outputs['hidden_state']
    previous_outcome = torch.tensor([trial_label])

# PCA visualization
pca = PCA(n_components=2)
contexts_2d = pca.fit_transform(np.vstack(contexts))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(contexts_2d[:, 0], contexts_2d[:, 1], 'o-', alpha=0.6)
plt.xlabel('Context PC1')
plt.ylabel('Context PC2')
plt.title('Context Evolution Across Trials')

plt.subplot(1, 2, 2)
plt.plot(predictions, label='Predictions', marker='o')
plt.plot(labels, label='True labels', marker='x')
plt.xlabel('Trial')
plt.ylabel('Approach probability')
plt.legend()
plt.title('Predictions Over Time')
plt.show()
```

### Comparison with Baselines

```python
import json
import numpy as np

# Load results from different approaches
approaches = {
    'Multi-dataset baseline': 'results/ccbpn_baseline/results.json',
    'Single-dataset (opto_hex)': 'results/ccbpn_single/results.json',
    'Recurrent context': 'results/ccbpn_recurrent/results.json',
}

for name, path in approaches.items():
    results = json.load(open(path))
    mean_acc = results['summary']['mean_val_acc']
    std_acc = results['summary']['std_val_acc']
    print(f"{name:30s}: {mean_acc:.1%} ± {std_acc:.1%}")
```

## Next Steps

### Phase 4: Full Training & Evaluation

1. **Run full training** (2-4 hours)
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
       --n-folds 5
   ```

2. **Compare with baselines**
   - Multi-dataset baseline (~70%)
   - Single-dataset models (~78%)
   - Dataset embeddings approach (if implemented)

3. **Create visualizations**
   - Context evolution across trials
   - Learning curves
   - Performance by dataset
   - Confusion matrices

4. **Biological analysis**
   - Does context track reward history?
   - Do gate values correlate with context switches?
   - Are CS+ and CS- trials separable in context space?

### Future Enhancements

1. **DoOR Integration**: Replace simplified odor encoding with realistic PN activation patterns from DoOR database

2. **Attention Mechanism**: Add attention over trial history (instead of just LSTM)

3. **Meta-Learning**: Use MAML or similar for fast adaptation to new contexts

4. **Bidirectional Context**: Process trials forward and backward for better context inference

5. **Hierarchical Context**: Multi-level context (session → block → trial)

## References

### Biological Background

1. **Aso et al. (2014)**: "The neuronal architecture of the mushroom body provides a logic for associative learning"
   - Describes MB circuit organization and learning rules

2. **Cohn et al. (2015)**: "Coordinated and compartmentalized neuromodulation shapes sensory processing in Drosophila"
   - Shows how dopamine modulates KC→MBON plasticity

3. **Hige et al. (2015)**: "Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila"
   - Demonstrates trial-to-trial plasticity in MB

### Machine Learning

4. **Hochreiter & Schmidhuber (1997)**: "Long Short-Term Memory"
   - Original LSTM paper

5. **Merity et al. (2018)**: "An Analysis of Neural Language Modeling at Multiple Scales"
   - Discusses truncated BPTT and gradient clipping

6. **Finn et al. (2017)**: "Model-Agnostic Meta-Learning"
   - Meta-learning for fast adaptation (future work)

## Contact

For questions or issues:
- Check the implementation code for detailed comments
- Run sanity checks to verify installation
- Review training logs for debugging information
- Consult biological papers for context on design decisions

## License

This implementation is part of the Plasticity-Guided Connectome Network (PGCN) project.
