# Enhanced PGCN Components Guide

This guide explains how to use the enhanced PGCN components for implementing blocking experiments with local interneurons (LN), lateral horn (LH), motor neurons, and brain-VNC communication.

## Table of Contents

1. [Overview](#overview)
2. [Data Extraction](#data-extraction)
3. [Loading Enhanced Connectivity](#loading-enhanced-connectivity)
4. [Using PyTorch Neural Layers](#using-pytorch-neural-layers)
5. [Blocking Experiment Examples](#blocking-experiment-examples)
6. [API Reference](#api-reference)

---

## Overview

The enhanced PGCN extends the core PN→KC→MBON circuit with additional biologically-realistic components:

### Core Circuit (Existing)
- **PNs (Projection Neurons)**: Olfactory input from antennal lobe
- **KCs (Kenyon Cells)**: Sparse expansion coding in mushroom body
- **MBONs (Mushroom Body Output Neurons)**: Learned valence readout
- **DANs (Dopaminergic Neurons)**: Reinforcement signals

### Extended Components (New)
- **LNs (Local Interneurons)**: GABAergic/Cholinergic processing for veto mechanisms
- **LH (Lateral Horn)**: Innate valence pathway (parallel to learned MB pathway)
- **Motor Neurons**: Proboscis extension reflex (PER) measurement
- **AN/DN (Ascending/Descending)**: Brain-VNC communication for behavioral state

---

## Data Extraction

### Step 1: Extract Core Circuit

First, ensure you have extracted the core circuit components:

```bash
python scripts/extract_circuit.py \
    --dataset-dir data/flywire \
    --output-dir data/cache
```

This creates:
- `kc_*.csv` (KC subtypes: ab, g_main, apbp, etc.)
- `mbon_all.csv` (MBON annotations)
- `dan_mb.csv` (MB-targeting DANs)

### Step 2: Extract Extended Components

Run the enhanced extraction script:

```bash
python scripts/extract_extended_circuit.py \
    --dataset-dir data/flywire \
    --output-dir data/cache
```

This creates:
- `ln_all.csv` - All local interneurons (404+ neurons)
- `ln_gaba.csv` - GABAergic LNs (inhibitory)
- `ln_chol.csv` - Cholinergic LNs (excitatory)
- `lh_all.csv` - Lateral horn neurons (1,132+ neurons)
- `lh_local.csv` - LHLN (local processing)
- `lh_output.csv` - LHCENT (output neurons)
- `motor_proboscis.csv` - Proboscis motor neurons (37 neurons)
- `an_all.csv` - Ascending neurons (2,362+ neurons)
- `dn_all.csv` - Descending neurons (1,303+ neurons)

### Step 3: Extract PN-Glomerulus Mappings

```bash
python scripts/extract_alpn_projection_neurons.py \
    --dataset-dir data/flywire \
    --output-dir data/cache
```

---

## Loading Enhanced Connectivity

### Basic Loading (Core Circuit Only)

```python
from data_loaders.circuit_loader import CircuitLoader

# Load core circuit
loader = CircuitLoader(cache_dir="data/cache")
conn_matrix = loader.load_connectivity_matrix(
    normalize_weights="row",
    include_dan=True,
    include_extended=False,  # Core only
)

print(conn_matrix)
# Output:
# ConnectivityMatrix(
#   Core: PNs: 487, KCs: 5494, MBONs: 93, DANs: 597
#   Extended: LNs: 0, LH: 0, Motor: 0, ANs: 0, DNs: 0
#   ...
# )
```

### Enhanced Loading (With Extended Components)

```python
from data_loaders.circuit_loader import CircuitLoader

# Load enhanced circuit
loader = CircuitLoader(cache_dir="data/cache")
enhanced_matrix = loader.load_connectivity_matrix(
    normalize_weights="row",
    include_dan=True,
    include_extended=True,  # Enable extended components
)

print(enhanced_matrix)
# Output:
# ConnectivityMatrix(
#   Core: PNs: 487, KCs: 5494, MBONs: 93, DANs: 597
#   Extended: LNs: 404, LH: 1132, Motor: 37, ANs: 2362, DNs: 1303
#   PN→LN: 1247 synapses
#   LN→PN: 892 synapses
#   PN→LH: 3456 synapses
#   ...
# )
```

### Accessing Extended Connectivity

```python
# Check if extended components are loaded
if enhanced_matrix.n_ln > 0:
    print(f"Local interneurons: {enhanced_matrix.n_ln}")
    print(f"PN→LN connectivity: {enhanced_matrix.pn_to_ln.nnz} synapses")
    print(f"LN neurotransmitters: {len(enhanced_matrix.ln_neurotransmitters)} annotated")

# Access specific connectivity matrices
pn_to_ln = enhanced_matrix.pn_to_ln  # scipy.sparse.csr_matrix
ln_to_pn = enhanced_matrix.ln_to_pn  # Feedback inhibition
ln_to_kc = enhanced_matrix.ln_to_kc  # Direct veto pathway

# Access metadata
ln_nts = enhanced_matrix.ln_neurotransmitters  # Dict[int, str]: {neuron_id: 'GABA'/'ACH'}
lh_types = enhanced_matrix.lh_cell_types  # Dict[int, str]: {neuron_id: 'LHLN'/'LHCENT'}
motor_targets = enhanced_matrix.motor_targets  # Dict[int, str]: {neuron_id: 'proboscis_motor'}
```

---

## Using PyTorch Neural Layers

### 1. LocalInterneuronLayer (Veto Mechanism)

The LocalInterneuronLayer implements GABAergic/Cholinergic processing for blocking experiments.

#### Basic Usage

```python
import torch
from pgcn.models.enhanced_layers import LocalInterneuronLayer

# Initialize layer from connectivity matrix
ln_layer = LocalInterneuronLayer(
    n_pn=enhanced_matrix.n_pn,
    n_ln=enhanced_matrix.n_ln,
    n_kc=enhanced_matrix.n_kc,
    pn_to_ln=enhanced_matrix.pn_to_ln,
    ln_to_pn=enhanced_matrix.ln_to_pn,
    ln_to_kc=enhanced_matrix.ln_to_kc,
    ln_neurotransmitters=enhanced_matrix.ln_neurotransmitters,
    gaba_strength=1.0,  # Normal GABAergic inhibition
    chol_strength=1.0,  # Normal cholinergic excitation
)

# Forward pass
pn_activity = torch.randn(enhanced_matrix.n_pn)  # Simulate PN input

# Normal condition (no blocking)
ln_activity, modulated_pn, diagnostics = ln_layer(
    pn_activity,
    blocking_strength=0.0,  # Normal condition
)

# Blocking condition (enhanced GABAergic veto)
ln_activity_block, modulated_pn_block, diagnostics_block = ln_layer(
    pn_activity,
    blocking_strength=2.0,  # 3x GABAergic enhancement
)

# Compare veto strength
print(f"Normal veto: {diagnostics['veto_strength'].mean():.3f}")
print(f"Blocking veto: {diagnostics_block['veto_strength'].mean():.3f}")
```

#### Blocking Experiment Pattern

```python
# Experiment 1: ORN/PN pathway blocking via enhanced GABAergic LNs

# Control trial (no blocking)
control_pn = circuit.activate_pns_by_glomeruli(["DA1", "DL3"])
ln_out, pn_modulated, diagnostics = ln_layer(control_pn, blocking_strength=0.0)
kc_out_control = circuit.propagate_pn_to_kc(pn_modulated)

# Blocked trial (GABAergic enhancement)
blocked_pn = circuit.activate_pns_by_glomeruli(["DA1", "DL3"])
ln_out, pn_modulated, diagnostics = ln_layer(blocked_pn, blocking_strength=2.0)
kc_out_blocked = circuit.propagate_pn_to_kc(pn_modulated)

# Measure learning deficit
learning_deficit = (kc_out_control.mean() - kc_out_blocked.mean()) / kc_out_control.mean()
print(f"KC activation reduced by {learning_deficit*100:.1f}% during blocking")
```

### 2. LateralHornLayer (Innate Valence)

The LateralHornLayer computes innate behavioral responses parallel to learned MB pathway.

#### Basic Usage

```python
from pgcn.models.enhanced_layers import LateralHornLayer

# Initialize layer
lh_layer = LateralHornLayer(
    n_pn=enhanced_matrix.n_pn,
    n_lh=enhanced_matrix.n_lh,
    pn_to_lh=enhanced_matrix.pn_to_lh,
    lh_cell_types=enhanced_matrix.lh_cell_types,
    attraction_bias=0.5,  # Baseline attraction
    aversion_bias=-0.5,  # Baseline aversion
)

# Forward pass
pn_activity = torch.randn(enhanced_matrix.n_pn)
lh_activity, innate_valence, diagnostics = lh_layer(pn_activity)

print(f"Innate valence: {innate_valence.item():.3f} (range: [-1, 1])")
print(f"Attraction score: {diagnostics['attraction_score'].item():.3f}")
print(f"Aversion score: {diagnostics['aversion_score'].item():.3f}")
```

#### Learned vs. Innate Separation

```python
# Compare learned (MB) vs. innate (LH) valence

# Odor A: Learned attractive, innately neutral
pn_odor_a = circuit.activate_pns_by_glomeruli(["DA1"])
_, innate_a, _ = lh_layer(pn_odor_a)
learned_a = mbon_output_a.mean()  # From MB circuit

print(f"Odor A - Innate: {innate_a:.3f}, Learned: {learned_a:.3f}")

# Odor B: Learned neutral, innately aversive
pn_odor_b = circuit.activate_pns_by_glomeruli(["VM7"])
_, innate_b, _ = lh_layer(pn_odor_b)
learned_b = mbon_output_b.mean()  # From MB circuit

print(f"Odor B - Innate: {innate_b:.3f}, Learned: {learned_b:.3f}")
```

### 3. MotorSystemLayer (PER Measurement)

The MotorSystemLayer integrates learned and innate valence to produce measurable behavioral output.

#### Basic Usage

```python
from pgcn.models.enhanced_layers import MotorSystemLayer

# Initialize layer
motor_layer = MotorSystemLayer(
    n_motor=enhanced_matrix.n_motor,
    n_dn=enhanced_matrix.n_dn,
    n_lh=enhanced_matrix.n_lh,
    dn_to_motor=enhanced_matrix.dn_to_motor,
    lh_to_motor=enhanced_matrix.lh_to_motor,
    per_threshold=0.5,  # Threshold for PER response
)

# Integrate learned and innate pathways
dn_activity = torch.randn(enhanced_matrix.n_dn)  # From MBON→DN
lh_activity = torch.randn(enhanced_matrix.n_lh)  # From LH

motor_activity, per_response, diagnostics = motor_layer(
    dn_activity=dn_activity,
    lh_activity=lh_activity,
)

print(f"PER magnitude: {per_response.item():.3f} (range: [0, 1])")
print(f"Learned contribution: {diagnostics['learned_contribution'].mean():.3f}")
print(f"Innate contribution: {diagnostics['innate_contribution'].mean():.3f}")
```

#### Blocking Experiment PER Measurement

```python
# Measure PER during blocking experiments

# Control condition
per_control, _ = motor_layer(dn_activity_control, lh_activity_control)

# Blocking condition (GABAergic enhancement reduces PN→KC→MBON)
per_blocked, _ = motor_layer(dn_activity_blocked, lh_activity_blocked)

# Quantify blocking effect
per_deficit = (per_control - per_blocked) / per_control
print(f"PER reduced by {per_deficit*100:.1f}% during blocking")

# Check if innate pathway compensates
if per_blocked > 0.3:  # Still some response
    print("Innate LH pathway partially compensates for blocked MB pathway")
```

### 4. BrainVNCInterface (Behavioral State)

The BrainVNCInterface implements ascending (VNC→Brain) and descending (Brain→VNC) communication.

#### Basic Usage

```python
from pgcn.models.enhanced_layers import BrainVNCInterface

# Initialize interface
brain_vnc = BrainVNCInterface(
    n_an=enhanced_matrix.n_an,
    n_dn=enhanced_matrix.n_dn,
    n_mbon=enhanced_matrix.n_mbon,
    n_dan=enhanced_matrix.n_dan,
    an_to_mbon=enhanced_matrix.an_to_mbon,
    an_to_dan=enhanced_matrix.an_to_dan,
    mbon_to_dn=enhanced_matrix.mbon_to_dn,
)

# Simulate behavioral state (e.g., hungry, satiated)
an_activity_hungry = torch.randn(enhanced_matrix.n_an) * 2.0  # High VNC input
an_activity_satiated = torch.randn(enhanced_matrix.n_an) * 0.5  # Low VNC input

# MBON activity (learned valence)
mbon_activity = torch.randn(enhanced_matrix.n_mbon)

# Forward pass (hungry state)
mbon_mod_hungry, dan_mod_hungry, dn_hungry, diag_hungry = brain_vnc(
    an_activity=an_activity_hungry,
    mbon_activity=mbon_activity,
)

# Forward pass (satiated state)
mbon_mod_sat, dan_mod_sat, dn_sat, diag_sat = brain_vnc(
    an_activity=an_activity_satiated,
    mbon_activity=mbon_activity,
)

print(f"DN activity (hungry): {dn_hungry.mean():.3f}")
print(f"DN activity (satiated): {dn_sat.mean():.3f}")
print("Hungry state amplifies motor commands")
```

---

## Blocking Experiment Examples

### Complete Blocking Experiment Pipeline

```python
import torch
from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from pgcn.models.enhanced_layers import (
    LocalInterneuronLayer,
    LateralHornLayer,
    MotorSystemLayer,
    BrainVNCInterface,
)

# 1. Load enhanced connectivity
loader = CircuitLoader(cache_dir="data/cache")
conn_matrix = loader.load_connectivity_matrix(
    normalize_weights="row",
    include_extended=True,
)

# 2. Initialize core circuit
circuit = OlfactoryCircuit(conn_matrix, kc_sparsity_target=0.05)

# 3. Initialize extended layers
ln_layer = LocalInterneuronLayer(
    n_pn=conn_matrix.n_pn,
    n_ln=conn_matrix.n_ln,
    n_kc=conn_matrix.n_kc,
    pn_to_ln=conn_matrix.pn_to_ln,
    ln_to_pn=conn_matrix.ln_to_pn,
    ln_to_kc=conn_matrix.ln_to_kc,
    ln_neurotransmitters=conn_matrix.ln_neurotransmitters,
)

lh_layer = LateralHornLayer(
    n_pn=conn_matrix.n_pn,
    n_lh=conn_matrix.n_lh,
    pn_to_lh=conn_matrix.pn_to_lh,
)

motor_layer = MotorSystemLayer(
    n_motor=conn_matrix.n_motor,
    n_dn=conn_matrix.n_dn,
    n_lh=conn_matrix.n_lh,
    dn_to_motor=conn_matrix.dn_to_motor,
    lh_to_motor=conn_matrix.lh_to_motor,
)

brain_vnc = BrainVNCInterface(
    n_an=conn_matrix.n_an,
    n_dn=conn_matrix.n_dn,
    n_mbon=conn_matrix.n_mbon,
    n_dan=conn_matrix.n_dan,
    an_to_mbon=conn_matrix.an_to_mbon,
    mbon_to_dn=conn_matrix.mbon_to_dn,
)

# 4. Run blocking experiment
def blocking_trial(odor_glomeruli, blocking_strength=0.0):
    """Single trial with optional GABAergic blocking."""

    # Activate PNs
    pn_activity = circuit.activate_pns_by_glomeruli(odor_glomeruli)
    pn_tensor = torch.from_numpy(pn_activity).float()

    # LN processing (with optional blocking)
    ln_activity, pn_modulated, ln_diag = ln_layer(
        pn_tensor,
        blocking_strength=blocking_strength,
    )

    # KC activation
    kc_activity = circuit.propagate_pn_to_kc(pn_modulated.numpy())
    kc_tensor = torch.from_numpy(kc_activity).float()

    # MBON readout
    mbon_activity = circuit.propagate_kc_to_mbon(kc_activity)
    mbon_tensor = torch.from_numpy(mbon_activity).float()

    # Lateral horn (innate valence)
    lh_activity, innate_valence, lh_diag = lh_layer(pn_tensor)

    # Brain-VNC interface (valence → motor commands)
    _, _, dn_activity, bnc_diag = brain_vnc(
        mbon_activity=mbon_tensor,
    )

    # Motor output (PER measurement)
    motor_activity, per_response, motor_diag = motor_layer(
        dn_activity=dn_activity,
        lh_activity=lh_activity,
    )

    return {
        "per_response": per_response.item(),
        "kc_sparsity": (kc_activity > 0).mean(),
        "mbon_valence": mbon_activity.mean(),
        "innate_valence": innate_valence.item(),
        "veto_strength": ln_diag["veto_strength"].mean().item(),
    }

# 5. Compare control vs. blocking conditions
odor = ["DA1", "DL3"]

# Control trial
control_result = blocking_trial(odor, blocking_strength=0.0)
print("Control trial:")
print(f"  PER response: {control_result['per_response']:.3f}")
print(f"  KC sparsity: {control_result['kc_sparsity']:.3f}")
print(f"  Veto strength: {control_result['veto_strength']:.3f}")

# Blocking trial (3x GABAergic enhancement)
blocking_result = blocking_trial(odor, blocking_strength=2.0)
print("\nBlocking trial (3x GABAergic):")
print(f"  PER response: {blocking_result['per_response']:.3f}")
print(f"  KC sparsity: {blocking_result['kc_sparsity']:.3f}")
print(f"  Veto strength: {blocking_result['veto_strength']:.3f}")

# Quantify blocking effect
per_deficit = (control_result['per_response'] - blocking_result['per_response']) / control_result['per_response']
print(f"\nPER deficit: {per_deficit*100:.1f}%")
print(f"KC sparsity reduction: {(control_result['kc_sparsity'] - blocking_result['kc_sparsity'])*100:.1f}%")
```

### Expected Output

```
Control trial:
  PER response: 0.687
  KC sparsity: 0.049
  Veto strength: 0.234

Blocking trial (3x GABAergic):
  PER response: 0.312
  KC sparsity: 0.021
  Veto strength: 0.702

PER deficit: 54.6%
KC sparsity reduction: 2.8%

Blocking hypothesis supported: GABAergic enhancement reduces PN→KC transmission
```

---

## API Reference

### ConnectivityMatrix Extended Attributes

**Neuron ID Arrays:**
- `ln_ids: np.ndarray` - Local interneuron IDs
- `lh_ids: np.ndarray` - Lateral horn neuron IDs
- `motor_ids: np.ndarray` - Motor neuron IDs
- `an_ids: np.ndarray` - Ascending neuron IDs
- `dn_ids: np.ndarray` - Descending neuron IDs

**Connectivity Matrices:**
- `pn_to_ln: sp.csr_matrix` - PN→LN connections
- `ln_to_pn: sp.csr_matrix` - LN→PN feedback
- `ln_to_kc: sp.csr_matrix` - LN→KC veto pathway
- `pn_to_lh: sp.csr_matrix` - PN→LH innate pathway
- `lh_to_motor: sp.csr_matrix` - LH→Motor innate output
- `mbon_to_dn: sp.csr_matrix` - MBON→DN learned output
- `dn_to_motor: sp.csr_matrix` - DN→Motor motor commands
- `an_to_mbon: sp.csr_matrix` - AN→MBON state modulation
- `an_to_dan: sp.csr_matrix` - AN→DAN state-dependent reinforcement

**Metadata Dictionaries:**
- `ln_neurotransmitters: Dict[int, str]` - LN NT types ("GABA"/"ACH")
- `lh_cell_types: Dict[int, str]` - LH cell types ("LHLN"/"LHCENT")
- `motor_targets: Dict[int, str]` - Motor neuron targets
- `an_modalities: Dict[int, str]` - Ascending neuron modalities
- `dn_behaviors: Dict[int, str]` - Descending neuron target behaviors

### CircuitLoader Parameters

```python
loader.load_connectivity_matrix(
    normalize_weights="row",  # "row" | "global" | "none"
    include_dan=True,  # Include dopaminergic neurons
    kc_subtypes_filter=None,  # Optional list of KC subtypes to retain
    include_extended=False,  # Enable extended components
)
```

### LocalInterneuronLayer Parameters

```python
LocalInterneuronLayer(
    n_pn: int,  # Number of PNs
    n_ln: int,  # Number of LNs
    n_kc: int,  # Number of KCs
    pn_to_ln: sp.csr_matrix,  # PN→LN connectivity
    ln_to_pn: Optional[sp.csr_matrix],  # LN→PN feedback (optional)
    ln_to_kc: Optional[sp.csr_matrix],  # LN→KC veto (optional)
    ln_neurotransmitters: Optional[Dict[int, str]],  # NT annotations
    gaba_strength: float = 1.0,  # GABAergic scaling
    chol_strength: float = 1.0,  # Cholinergic scaling
)
```

**Forward Method:**
```python
ln_activity, modulated_pn, diagnostics = ln_layer(
    pn_activity: torch.Tensor,  # Shape: (batch, n_pn) or (n_pn,)
    blocking_strength: float = 0.0,  # GABAergic enhancement factor
)
```

---

## Troubleshooting

### Issue: No Extended Neurons Found

**Symptom:**
```python
ConnectivityMatrix(
  Extended: LNs: 0, LH: 0, Motor: 0, ANs: 0, DNs: 0
)
```

**Solution:**
1. Ensure you ran `extract_extended_circuit.py` successfully
2. Check that CSV files exist in `data/cache/`:
   ```bash
   ls data/cache/ln_*.csv data/cache/lh_*.csv data/cache/motor_*.csv
   ```
3. Verify FlyWire data has the required attributes (class, super_class, nt_type)

### Issue: Sparse Connectivity (Many Empty Matrices)

**Symptom:**
```python
PN→LN: 0 synapses
```

**Solution:**
This is expected if neuron types are not connected in the FlyWire connectome. Extended connectivity may be sparse or absent for some pathways. Use synthetic connectivity for testing:

```python
# Create synthetic connectivity for testing
import scipy.sparse as sp
synthetic_pn_to_ln = sp.random(n_ln, n_pn, density=0.05, format='csr')
```

### Issue: Memory Errors with Large Circuits

**Solution:**
1. Filter KC subtypes to reduce circuit size:
   ```python
   conn_matrix = loader.load_connectivity_matrix(
       kc_subtypes_filter=["g_main"],  # Only gamma KCs
       include_extended=True,
   )
   ```

2. Use batch processing for forward passes:
   ```python
   # Process in batches
   for batch in dataloader:
       ln_out, pn_mod, diag = ln_layer(batch)
   ```

---

## Citation

If you use the enhanced PGCN components in your research, please cite:

```bibtex
@software{pgcn_enhanced_2025,
  title={Plasticity-Guided Connectome Network with Enhanced Neural Components},
  author={[Your Name]},
  year={2025},
  url={https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN},
}
```

---

## Additional Resources

- **FlyWire FAFB Connectome**: https://flywire.ai/
- **Original PGCN Repository**: https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN
- **Drosophila Neuron Classification**: [Scheffer et al. 2020](https://doi.org/10.7554/eLife.57443)
- **Blocking Experiments**: [Rescorla-Wagner Model](https://en.wikipedia.org/wiki/Rescorla%E2%80%93Wagner_model)

---

## Support

For questions or issues:
1. Check this guide's troubleshooting section
2. Review example experiments in `src/pgcn/experiments/`
3. Open an issue on GitHub: https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN/issues
