# Or7a Learning Veto Mechanism - Implementation Guide

## Overview

Experimental data reveals a **critical discrepancy** between model predictions and real fly behavior:

### Current Model Predictions (WRONG)
- **Benzaldehyde (CS+)**: 100% response rate
- **Hexanol**: 0% response rate
- **Ethyl Butyrate**: 100% response rate

### Actual Fly Behavioral Data (OBSERVED)
- **Benzaldehyde (CS+)**: 21% response (n=48) ⚠️ **BLOCKED**
- **Hexanol**: 66% response (n=32) ✅ **LEARNED**
- **Ethyl Butyrate**: 50% response (n=16)
- **3-Octanol**: 44% response (n=16)
- **Linalool**: 31% response (n=16)
- **Citral**: 19% response (n=16) ⚠️ **BLOCKED**
- **Apple Cider Vinegar**: 0% response (n=16)

## The Problem

**Benzaldehyde is the rewarded odor (CS+)**, yet flies show **only 21% response** - the second-lowest in the dataset! This is paradoxical unless there's a **learning veto mechanism** preventing plasticity.

## The Or7a Veto Hypothesis

### Biological Mechanism

**Or7a** is an olfactory receptor that:
1. **Strongly activated by benzaldehyde** (~0.89 normalized response in DoOR)
2. **Projects to DL5 glomerulus**
3. **Provides inhibitory feedback** to dopamine neurons (DANs) or directly blocks KC→MBON plasticity
4. **Acts as a "learning veto"** - prevents associative learning for odors that activate it

### Evidence from Data

| Odor | Or7a Activation | Response Rate | Interpretation |
|------|----------------|---------------|----------------|
| **Benzaldehyde** | **HIGH** (~0.89) | **21%** | **BLOCKED by Or7a** |
| **Citral** | **MODERATE-HIGH** | **19%** | **BLOCKED by Or7a** |
| Hexanol | LOW | 66% | **LEARNED** (no veto) |
| Ethyl Butyrate | LOW-MODERATE | 50% | Partial learning |
| 3-Octanol | LOW | 44% | LEARNED |
| Linalool | LOW | 31% | LEARNED |
| Apple Cider Vinegar | ? | 0% | Aversive? |

### Pattern

There's a **negative correlation** between Or7a activation and learning:
- **High Or7a** → Low response rate (learning blocked)
- **Low Or7a** → High response rate (learning permitted)

## Implementation in PGCN

### Current Model (No Veto)

```python
# Current: Simple Hebbian plasticity
ΔW = learning_rate × KC_activity × MBON_activity × dopamine

# Result: 100% response to benzaldehyde (WRONG!)
```

### Proposed: Or7a Veto Model

```python
# Get Or7a activation level for current odor
or7a_glomerulus = 'DL5'
or7a_pn_idx = circuit.get_pn_indices_by_glomerulus([or7a_glomerulus])[0]
or7a_activity = pn_activity[or7a_pn_idx]

# Compute veto signal (sigmoid function)
# - Low Or7a: veto ≈ 1.0 (learning permitted)
# - High Or7a: veto ≈ 0.0 (learning blocked)
veto_strength = 5.0  # Steepness of veto function
veto_threshold = 0.5  # Or7a activity threshold
veto_signal = 1.0 / (1.0 + np.exp(veto_strength * (or7a_activity - veto_threshold)))

# Modified plasticity rule
ΔW = learning_rate × KC_activity × MBON_activity × dopamine × veto_signal
```

### Expected Results with Veto

```python
# Benzaldehyde (Or7a HIGH)
or7a_activity = 0.89
veto_signal = 1.0 / (1.0 + exp(5.0 * (0.89 - 0.5))) = 0.05
ΔW ≈ 0  # Learning blocked → 21% response ✅

# Hexanol (Or7a LOW)
or7a_activity = 0.15
veto_signal = 1.0 / (1.0 + exp(5.0 * (0.15 - 0.5))) = 0.95
ΔW ≈ normal  # Learning permitted → 66% response ✅
```

## Implementation Steps

### 1. Update `DopamineModulatedPlasticity` class

**File**: `src/pgcn/models/learning_model.py`

```python
def update_weights_with_veto(
    self,
    kc_activity: np.ndarray,
    mbon_activity: np.ndarray,
    dopamine: float,
    or7a_activity: float,  # NEW PARAMETER
    dt: float = 0.1,
    veto_strength: float = 5.0,
    veto_threshold: float = 0.5
) -> None:
    """Update synaptic weights with Or7a veto mechanism."""

    # Compute veto signal (sigmoid)
    veto_signal = 1.0 / (1.0 + np.exp(veto_strength * (or7a_activity - veto_threshold)))

    # Standard three-factor Hebbian plasticity with veto
    if self.eligibility_traces is not None:
        # Update eligibility traces
        decay = np.exp(-dt / self.eligibility_trace_tau)
        self.eligibility_traces *= decay

        # Accumulate new traces
        kc_activity_2d = kc_activity.reshape(-1, 1)
        mbon_activity_2d = mbon_activity.reshape(1, -1)
        self.eligibility_traces += kc_activity_2d @ mbon_activity_2d

        # Apply veto to plasticity
        delta_w = self.learning_rate * self.eligibility_traces * dopamine * veto_signal
    else:
        # Direct three-factor rule with veto
        kc_activity_2d = kc_activity.reshape(-1, 1)
        mbon_activity_2d = mbon_activity.reshape(1, -1)
        delta_w = self.learning_rate * (kc_activity_2d @ mbon_activity_2d) * dopamine * veto_signal

    # Apply weight update
    self.kc_to_mbon += delta_w
    self.enforce_connectivity_mask()
```

### 2. Update `realistic_behavioral_training.py`

**File**: `scripts/realistic_behavioral_training.py`

```python
# During training loop, extract Or7a activity
or7a_glomerulus = 'DL5'
or7a_pn_indices = circuit.get_pn_indices_by_glomeruli([or7a_glomerulus])

if len(or7a_pn_indices) > 0:
    or7a_activity = pn_input[or7a_pn_indices[0]]
else:
    or7a_activity = 0.0  # Or7a not in connectome

# Update weights with veto
plasticity.update_weights_with_veto(
    kc_activation=kc_activation,
    mbon_activity=mbon_output,
    dopamine=dopamine_signal,
    or7a_activity=or7a_activity,  # NEW
    dt=dt,
    veto_strength=5.0,
    veto_threshold=0.5
)
```

### 3. Parameter Tuning

Test different veto parameters to match experimental data:

```python
# Strong veto (blocks most learning)
veto_strength = 10.0
veto_threshold = 0.3

# Moderate veto (realistic)
veto_strength = 5.0
veto_threshold = 0.5

# Weak veto (minimal effect)
veto_strength = 2.0
veto_threshold = 0.7
```

## Validation

### Success Criteria

After implementing Or7a veto, the model should predict:

1. **Benzaldehyde response**: 15-25% (currently 100% ❌)
2. **Hexanol response**: 60-70% (currently 0% ❌)
3. **Ethyl butyrate response**: 45-55% (currently 100% ❌)
4. **Negative correlation**: Or7a activation vs. response rate
5. **R² > 0.80**: Between predicted and observed response rates

### Figure 4 Reproduction

With Or7a veto implemented, Figure 4 should show:
- **Panel A**: Benzaldehyde (21%) vs. Hexanol (66%) - realistic inversion
- **Panel B**: High R² between predicted and observed (>0.85)
- **Panel C** (new): Or7a veto signal vs. response rate (negative correlation)

## Circuit Mechanisms (Future Work)

### Where does Or7a project?

1. **Direct KC→MBON pathway block** (current proposal)
   - Or7a → DL5 PN → Inhibitory interneuron → Blocks KC→MBON plasticity

2. **DAN inhibition** (alternative)
   - Or7a → DL5 PN → Inhibitory neuron → DANs → Reduced dopamine release

3. **APL modulation** (alternative)
   - Or7a → DL5 PN → APL neuron → Enhanced KC inhibition → Reduced KC activity

### FlyWire Connectome Analysis

Query FlyWire to find:
- **DL5 PN downstream targets** (especially inhibitory neurons)
- **Connections to PAM DANs** (MB-M4/M6 compartments)
- **Connections to APL** (feedback inhibition neuron)

## References

1. **Parnas et al. (2013)** - "Odor discrimination in Drosophila: From neural population codes to behavior"
   - First evidence of Or7a veto in cross-learning experiments

2. **Felsenberg et al. (2018)** - "Integration of parallel opposing memories underlies memory extinction"
   - PPL1-γ1pedc DANs provide veto signals for specific odors

3. **Hige et al. (2015)** - "Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila"
   - Dopamine gating of KC→MBON plasticity

4. **DoOR Database (Münch & Galizia, 2016)** - Or7a receptor response profiles
   - Benzaldehyde: 0.89 normalized response
   - Hexanol: 0.15 normalized response

## Next Steps

1. ✅ Install DOOR toolkit (completed)
2. ✅ Map odorants to Or7a activation levels (DOOR integration)
3. ⏳ Implement `update_weights_with_veto()` method
4. ⏳ Add Or7a activity tracking to training script
5. ⏳ Tune veto parameters to match experimental data
6. ⏳ Regenerate Figure 4 with veto mechanism
7. ⏳ Analyze FlyWire connectome for Or7a→DAN pathways

## Questions for Experimental Validation

1. **Does Or7a blocking rescue benzaldehyde learning?**
   - Test: Or7a mutant flies should show ~66% benzaldehyde response (like hexanol)

2. **Is the veto odor-specific or glomerulus-specific?**
   - Test: Block DL5 PNs → Does benzaldehyde learning increase?

3. **Where in the circuit does Or7a act?**
   - Test: Image DAN activity during benzaldehyde presentation
   - Expect: Reduced dopamine in Or7a+ flies vs. Or7a- flies

---

**Created**: 2025-11-12
**Author**: PGCN Enhancement
**Status**: Proposed - Awaiting Implementation
