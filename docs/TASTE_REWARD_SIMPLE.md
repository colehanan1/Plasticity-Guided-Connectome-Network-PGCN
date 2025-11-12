# Simplified Taste Reward Circuit Integration

**Purpose**: Provide dopamine (reward) signals for OR7a PN blocking experiments
**Status**: ✅ Ready for use
**Use Cases**: Experiments 1, 2, 3, 6 (OR7a blocking and Shapley analysis)

---

## Overview

The `TasteRewardCircuit` is a simplified version of the taste pathway that provides reward signals for olfactory learning experiments. It does **NOT** include GABA-LN inhibition or veto gate mechanisms.

### Architecture

```
Sugar Stimulus (scalar: 0-1)
    ↓
Sugar GRNs (90 neurons)
    ├→ Direct pathway → SEZ-PNs (21)
    └→ Relay pathway → ACh-LNs (60) → SEZ-PNs (21)
        ↓
    Reward Signal (scalar: 0-1)
        ↓
    Dopamine = Actual Reward - Predicted Reward
```

**Key Features:**
- ✅ Loads real connectome data from Shen et al. (2025)
- ✅ Uses synapse-weighted connectivity (actual synapse counts)
- ✅ Simple scalar input/output (easy integration)
- ✅ Computes dopamine (RPE) for plasticity
- ❌ NO GABA-LN pathway (removed for simplicity)
- ❌ NO veto gate mechanism (separate project)

---

## Installation & Setup

### 1. Ensure Data is Extracted

You should already have these files from running the extraction script:

```bash
data/cache/
├── shen2025_appetitive_grn.csv                  # 90 sugar GRNs
├── shen2025_appetitive_sez_pn.csv               # 21 SEZ-PNs
├── shen2025_appetitive_sez_ln_ach.csv           # 60 ACh-LNs
├── shen2025_appetitive_connectivity_grn_ach.npz # GRN→ACh weights
└── shen2025_appetitive_connectivity_grn_pn.npz  # GRN→PN weights
```

If not, run:
```bash
python scripts/extract_from_paper_data.py --mode appetitive --output-dir data/cache
```

### 2. Test the Circuit

```bash
python test_taste_reward.py
```

**Expected output:**
```
✓ PASS   Import module
✓ PASS   Instantiate circuit
✓ PASS   Forward (scalar)
✓ PASS   Forward (tensor)
✓ PASS   Dopamine computation
✓ PASS   Detailed output
✓ PASS   Statistics

✓ ALL TESTS PASSED
```

---

## Usage Examples

### Example 1: Basic Reward Signal

```python
from pgcn.models.taste_reward import TasteRewardCircuit
import torch

# Initialize circuit
taste = TasteRewardCircuit(
    data_dir=Path('data/cache'),
    use_synapse_weights=True
)

# Get reward signal (scalar input)
reward = taste(sugar_input=1.0)  # Full sugar
print(f"Reward: {reward.item():.3f}")
# Output: Reward: 0.487 (example)

# Try different sugar levels
reward_half = taste(sugar_input=0.5)  # Half sugar
reward_none = taste(sugar_input=0.0)  # No sugar
```

### Example 2: Dopamine (RPE) Computation

```python
# Scenario: Animal predicts low reward but gets high reward
predicted_reward = torch.tensor([0.2])  # Low expectation
actual_sugar = 1.0                       # High reward

dopamine = taste.compute_dopamine(
    sugar_input=actual_sugar,
    predicted_reward=predicted_reward
)

print(f"Dopamine (RPE): {dopamine.item():.3f}")
# Output: Dopamine (RPE): 0.287 (positive = better than expected)
```

### Example 3: Batch Processing

```python
# Process multiple trials at once
batch_size = 10
sugar_inputs = torch.rand(batch_size, taste.n_grns)  # Random sugar levels

rewards = taste(sugar_inputs)
print(f"Batch rewards shape: {rewards.shape}")  # (10,)
print(f"Mean reward: {rewards.mean().item():.3f}")
```

### Example 4: Detailed Activations

```python
# Get intermediate layer activations
output = taste(sugar_input=1.0, return_details=True)

print(f"Reward: {output['reward_signal'].item():.3f}")
print(f"GRN activity: {output['grn_activity'].mean().item():.3f}")
print(f"ACh-LN activity: {output['ach_ln_activity'].mean().item():.3f}")
print(f"SEZ-PN activity: {output['sez_pn_activity'].mean().item():.3f}")
```

---

## Integration with PGCN Model

### Add to Your Olfactory Circuit

```python
from pgcn.models.taste_reward import TasteRewardCircuit

class YourOlfactoryModel(nn.Module):
    def __init__(self, enable_taste_reward=True):
        super().__init__()

        # Your existing olfactory components
        self.orns = ORNLayer(...)
        self.pns = PNLayer(...)
        self.kcs = KCLayer(...)
        self.mbons = MBONLayer(...)

        # Add taste reward circuit
        if enable_taste_reward:
            self.taste = TasteRewardCircuit(
                data_dir=Path('data/cache')
            )
        else:
            self.taste = None

    def forward(self, odor_input, sugar_input=None):
        # Process odor through olfactory pathway
        orn_activity = self.orns(odor_input)
        pn_activity = self.pns(orn_activity)
        kc_activity = self.kcs(pn_activity)
        mbon_activity = self.mbons(kc_activity)

        # Get predicted reward from MBON
        predicted_reward = mbon_activity[:, 0]  # Approach neuron

        # Get actual reward from taste circuit
        if self.taste is not None and sugar_input is not None:
            actual_reward = self.taste(sugar_input)
            dopamine = actual_reward - predicted_reward
        else:
            # Default reward if taste circuit disabled
            actual_reward = torch.ones_like(predicted_reward) * 0.5
            dopamine = actual_reward - predicted_reward

        return {
            'mbon_activity': mbon_activity,
            'reward': actual_reward,
            'dopamine': dopamine
        }
```

### Training Loop

```python
def train_with_taste_reward(model, n_trials=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for trial in range(n_trials):
        # Generate stimuli
        odor = generate_or7a_odor()

        # Sugar reward (50% of trials)
        if trial % 2 == 0:
            sugar = 1.0  # Rewarded trial
        else:
            sugar = 0.0  # Unrewarded trial

        # Forward pass
        output = model(odor, sugar_input=sugar)

        # Dopamine-gated plasticity
        # (Update KC→MBON weights based on dopamine)
        dopamine = output['dopamine']
        # ... implement plasticity here ...

        print(f"Trial {trial}: Dopamine = {dopamine.mean().item():.3f}")
```

---

## Differences from Full Taste Circuit

| Feature | Simplified (`taste_reward.py`) | Full (`taste_circuit.py`) |
|---------|-------------------------------|---------------------------|
| **Purpose** | Reward signal only | Full taste processing + GABA testing |
| **GRNs** | ✅ 90 sugar GRNs | ✅ 90 sugar GRNs |
| **ACh-LNs** | ✅ 60 excitatory | ✅ 60 excitatory |
| **SEZ-PNs** | ✅ 21 projection | ✅ 21 projection |
| **GABA-LNs** | ❌ None | ✅ 36 inhibitory |
| **Veto Gate** | ❌ None | ✅ 3 modes |
| **GABA Gain** | ❌ None | ✅ Learnable parameter |
| **Output** | Scalar reward | Dict with veto signal |
| **Use Case** | OR7a blocking (Exp 1-3, 6) | Benzaldehyde failure (Exp 7) |
| **Complexity** | 🟢 Simple | 🔴 Complex |

**Recommendation:**
- Use `TasteRewardCircuit` for your core OR7a blocking experiments
- Use `TasteCircuit` only if testing GABA veto hypotheses

---

## Circuit Statistics

After loading, you can inspect the circuit:

```python
stats = taste.get_statistics()

# Example output:
{
    'n_grns': 90,
    'n_ach_lns': 60,
    'n_sez_pns': 21,
    'grn_to_ach_connections': 433,
    'grn_to_pn_connections': 44,
    'mean_grn_to_ach_weight': 0.396,
    'mean_grn_to_pn_weight': 0.412
}
```

---

## Troubleshooting

### Error: FileNotFoundError for CSV files

**Problem:** Data files not found in `data/cache/`

**Solution:**
```bash
# Run extraction script
python scripts/extract_from_paper_data.py --mode appetitive
```

### Error: Module not found

**Problem:** Can't import `TasteRewardCircuit`

**Solution:**
```bash
# Ensure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or add in script:
import sys
sys.path.insert(0, 'src')
```

### Issue: Reward signal always the same

**Problem:** Reward doesn't change with sugar input

**Check:**
1. Are you using synapse weights? (`use_synapse_weights=True`)
2. Is sugar input varying? (Not always 1.0 or 0.0)
3. Are GRN activations non-zero? (Check with `return_details=True`)

---

## API Reference

### TasteRewardCircuit

#### `__init__(data_dir, use_synapse_weights)`

Parameters:
- `data_dir` (Path): Directory with extracted data. Default: `Path('data/cache')`
- `use_synapse_weights` (bool): Use actual synapse counts vs binary. Default: `True`

#### `forward(sugar_input, return_details=False)`

Compute reward signal from sugar input.

Parameters:
- `sugar_input` (float or Tensor): Sugar stimulus, range [0, 1]
  - Scalar: Broadcast to all GRNs
  - Tensor (batch, n_grns): Explicit GRN activation
- `return_details` (bool): Return intermediate activations. Default: `False`

Returns:
- If `return_details=False`: Tensor (batch,) - Reward signal
- If `return_details=True`: Dict with keys:
  - `'reward_signal'`: Tensor (batch,) - Reward
  - `'grn_activity'`: Tensor (batch, 90) - GRN activations
  - `'ach_ln_activity'`: Tensor (batch, 60) - ACh-LN activations
  - `'sez_pn_activity'`: Tensor (batch, 21) - SEZ-PN activations

#### `compute_dopamine(sugar_input, predicted_reward)`

Compute dopamine (reward prediction error).

Parameters:
- `sugar_input` (float or Tensor): Actual sugar present
- `predicted_reward` (Tensor): Model's current prediction

Returns:
- Tensor (batch,) - Dopamine signal (RPE), range [-1, 1]

#### `get_statistics()`

Get circuit statistics.

Returns:
- Dict with neuron counts, connection counts, mean weights

---

## References

1. **Shen, K. et al. (2025).** Functional imaging and connectome analyses reveal organizing principles of taste circuits in Drosophila. *Current Biology* 35(9):1955-1970.e6. DOI: [10.1016/j.cub.2025.04.066](https://doi.org/10.1016/j.cub.2025.04.066)

2. **Data Extraction**: `scripts/extract_from_paper_data.py`

3. **Full Version**: `src/pgcn/models/taste_circuit.py` (includes GABA)

---

## Summary

✅ **Ready to Use**: Simplified taste reward circuit for OR7a experiments
✅ **Connectome-Constrained**: Uses real FlyWire data from Shen et al. (2025)
✅ **Simple API**: Scalar input → scalar reward output
✅ **Dopamine Computation**: Built-in RPE calculation for plasticity
✅ **Well-Tested**: 7 unit tests covering all functionality

This provides clean reward signals for your OR7a PN blocking experiments without the complexity of GABA veto gate testing.

---

*Last Updated: 2025-11-12*
