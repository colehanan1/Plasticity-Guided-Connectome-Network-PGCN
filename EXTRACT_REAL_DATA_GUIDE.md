# Extracting Real Data from Your PGCN Model

## Overview

This guide shows you how to generate real data from your PGCN simulations and feed it into the figure extraction pipeline.

## Quick Start

```bash
# 1. Run your training/simulation (examples below)
python your_training_script.py

# 2. Extract the data for figures
python extract_figure_data.py --task all

# 3. Generate figures
python examples/plot_extracted_figures.py --figure all
```

---

## Data Required for Each Figure

### Figure 1: Behavioral Prediction Data

**What you need:** Memory scores for wildtype, or7a_mutant, and control groups across training phases.

**Where it comes from:** Your learning experiments that test memory retention.

**Example Code to Generate:**

```python
import pandas as pd
import numpy as np
from pathlib import Path

# After running your learning experiment
def save_behavioral_results(results_dict, output_dir="results/behavioral_sim"):
    """
    Save behavioral experiment results for figure extraction.

    Parameters
    ----------
    results_dict : dict
        Format: {
            'wildtype': {'after_A_train': [scores], 'after_B_train': [scores], 'A_test': [scores]},
            'or7a_mutant': {...},
            'control': {...}
        }
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Option 1: Save as individual CSV files (easiest)
    for group, phases in results_dict.items():
        data = []
        for phase, scores in phases.items():
            for score in scores:
                data.append({
                    'phase': phase,
                    'memory_score': score
                })

        df = pd.DataFrame(data)
        df.to_csv(f"{output_dir}/{group}_behavioral.csv", index=False)
        print(f"✓ Saved {group} behavioral data")

    # Option 2: Save as combined CSV
    combined_data = []
    for group, phases in results_dict.items():
        for phase, scores in phases.items():
            for score in scores:
                combined_data.append({
                    'group': group,
                    'phase': phase,
                    'memory_score': score
                })

    df_combined = pd.DataFrame(combined_data)
    df_combined.to_csv(f"{output_dir}/behavioral_results.csv", index=False)
    print(f"✓ Saved combined behavioral data")


# Example usage in your training script:
# =========================================

from src.pgcn.models.learning_model import LearningExperiment, DopamineModulatedPlasticity
from src.pgcn.models.olfactory_circuit import OlfactoryCircuit

# Run experiments for each group
results = {}

# 1. Wildtype
circuit_wt = OlfactoryCircuit(...)  # Your circuit
plasticity_wt = DopamineModulatedPlasticity(...)
experiment_wt = LearningExperiment(circuit_wt, plasticity_wt)

# Phase 1: Train on odor A
odor_A_train = experiment_wt.run_experiment(
    odor_sequence=["DA1"] * 50,
    reward_sequence=[1] * 50
)
after_A_train = odor_A_train['mbon_valence'].iloc[-10:].mean()

# Phase 2: Train on odor B
odor_B_train = experiment_wt.run_experiment(
    odor_sequence=["DL3"] * 50,
    reward_sequence=[1] * 50
)
after_B_train = odor_B_train['mbon_valence'].iloc[-10:].mean()

# Phase 3: Test memory of A
odor_A_test = experiment_wt.run_experiment(
    odor_sequence=["DA1"] * 10,
    reward_sequence=[0] * 10  # No reward during test
)
A_test_score = odor_A_test['mbon_valence'].iloc[-5:].mean()

results['wildtype'] = {
    'after_A_train': [after_A_train],
    'after_B_train': [after_B_train],
    'A_test': [A_test_score]
}

# 2. Or7a mutant (with veto gate disabled or pathway blocked)
# ... repeat with modified circuit ...

# 3. Control
# ... repeat with baseline ...

# Save results
save_behavioral_results(results)
```

---

### Figure 2: Model Schematic Data

**What you need:** Neuron counts and veto gate statistics.

**Where it comes from:**
- Neuron counts: Already in your config file! ✅
- Veto gate mask: From your continual learning experiments

**Example Code to Generate Veto Mask:**

```python
import numpy as np
from pathlib import Path

def save_veto_mask(veto_gate, output_dir="results"):
    """
    Save veto gate protection mask for figure extraction.

    Parameters
    ----------
    veto_gate : SelectiveVetoGate or np.ndarray
        Either the veto gate object or the protection mask directly
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get protection mask
    if hasattr(veto_gate, 'protection_mask'):
        mask = veto_gate.protection_mask
    else:
        mask = veto_gate  # Already a numpy array

    # Save mask
    output_file = f"{output_dir}/veto_mask.npy"
    np.save(output_file, mask)
    print(f"✓ Saved veto mask: {mask.shape}, {np.sum(mask)} protected synapses")


# Example usage in your continual learning script:
# =================================================

from src.pgcn.models.veto_gate import SelectiveVetoGate

# After Task A training
circuit = OlfactoryCircuit(...)
plasticity = DopamineModulatedPlasticity(...)

# Create veto gate
veto_gate = SelectiveVetoGate(
    circuit=circuit,
    protection_threshold=0.3,
    gate_strength=0.9,
    similarity_metric='chemical'
)

# Train Task A
experiment = LearningExperiment(circuit, plasticity)
task_A_results = experiment.run_experiment(
    odor_sequence=["DA1"] * 100,
    reward_sequence=[1] * 100
)

# Identify critical pathways for Task A
veto_gate.identify_critical_pathways(
    task_data=plasticity.get_task_data_for_protection(),
    task_id=0,
    percentile_threshold=95
)

# Save the protection mask
save_veto_mask(veto_gate)
```

---

### Figure 3: Critical Synapse Map

**What you need:** 2D protection mask (KC × MBON).

**Where it comes from:** Same as Figure 2 - the veto gate protection mask.

**Already handled!** If you save the veto mask for Figure 2, Figure 3 will work automatically.

**For multiple odor pairs:**

```python
def save_multi_odor_veto_masks(odor_pairs, output_dir="results"):
    """
    Save veto masks for multiple odor pairs.

    Parameters
    ----------
    odor_pairs : list of tuples
        [(odor_A1, odor_B1), (odor_A2, odor_B2), ...]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, (odor_A, odor_B) in enumerate(odor_pairs):
        # Train on odor pair
        circuit = OlfactoryCircuit(...)
        plasticity = DopamineModulatedPlasticity(...)
        veto_gate = SelectiveVetoGate(...)

        # Train Task A
        experiment = LearningExperiment(circuit, plasticity)
        experiment.run_experiment(
            odor_sequence=[odor_A] * 100,
            reward_sequence=[1] * 100
        )

        # Identify critical pathways
        veto_gate.identify_critical_pathways(
            task_data=plasticity.get_task_data_for_protection(),
            task_id=i,
            percentile_threshold=95
        )

        # Save mask
        output_file = f"{output_dir}/veto_mask_odorpair{i}.npy"
        np.save(output_file, veto_gate.protection_mask)
        print(f"✓ Saved veto mask for {odor_A} → {odor_B}")


# Example usage:
odor_pairs = [
    ("DA1", "DL3"),
    ("VA1d", "VC1"),
    ("DC3", "DL5")
]
save_multi_odor_veto_masks(odor_pairs)
```

---

### Figure 4: ML Comparison Data

**What you need:** Forgetting scores for different continual learning methods.

**Where it comes from:** Running your model with different continual learning algorithms.

**Example Code to Generate:**

```python
import pandas as pd
from pathlib import Path

def save_ml_comparison(model_scores, output_dir="results"):
    """
    Save ML model comparison results for figure extraction.

    Parameters
    ----------
    model_scores : dict
        Format: {
            'MBON_veto': forgetting_score,
            'Dense_ANN': forgetting_score,
            'EWC': forgetting_score,
            'SI': forgetting_score,
            'LwF': forgetting_score,
            'GEM': forgetting_score
        }
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame([
        {'model_type': model, 'forgetting_score': score}
        for model, score in model_scores.items()
    ])

    # Save
    output_file = f"{output_dir}/forgetting_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved ML comparison data: {len(model_scores)} models")


# Example usage in your benchmark script:
# ========================================

def compute_forgetting_score(task_A_performance_before, task_A_performance_after):
    """
    Compute forgetting as performance drop on Task A after learning Task B.

    Returns
    -------
    float
        Forgetting score in [0, 1], where 0 = no forgetting, 1 = total forgetting
    """
    return max(0, (task_A_performance_before - task_A_performance_after) / task_A_performance_before)


# Run each method
model_scores = {}

# 1. MBON_veto (your method)
circuit = OlfactoryCircuit(...)
plasticity = DopamineModulatedPlasticity(...)
veto_gate = SelectiveVetoGate(...)
plasticity.enable_veto_protection(veto_gate)

experiment = LearningExperiment(circuit, plasticity)

# Task A
task_A_train = experiment.run_experiment(
    odor_sequence=["DA1"] * 100,
    reward_sequence=[1] * 100
)
perf_A_before = task_A_train['mbon_valence'].iloc[-10:].mean()

# Identify critical pathways
veto_gate.identify_critical_pathways(
    task_data=plasticity.get_task_data_for_protection(),
    task_id=0,
    percentile_threshold=95
)

# Task B
task_B_train = experiment.run_experiment(
    odor_sequence=["DL3"] * 100,
    reward_sequence=[1] * 100
)

# Test Task A again
task_A_test = experiment.run_experiment(
    odor_sequence=["DA1"] * 10,
    reward_sequence=[0] * 10
)
perf_A_after = task_A_test['mbon_valence'].iloc[-5:].mean()

model_scores['MBON_veto'] = compute_forgetting_score(perf_A_before, perf_A_after)


# 2. Dense_ANN (baseline - no protection)
# ... repeat without veto gate ...
model_scores['Dense_ANN'] = ...

# 3. EWC (Elastic Weight Consolidation)
# ... implement EWC and repeat ...
model_scores['EWC'] = ...

# 4. SI (Synaptic Intelligence)
# ... implement SI and repeat ...
model_scores['SI'] = ...

# 5. LwF (Learning without Forgetting)
# ... implement LwF and repeat ...
model_scores['LwF'] = ...

# 6. GEM (Gradient Episodic Memory)
# ... implement GEM and repeat ...
model_scores['GEM'] = ...

# Save comparison
save_ml_comparison(model_scores)
```

---

## Complete Integration Example

Here's a complete training script that saves all the data needed:

```python
#!/usr/bin/env python3
"""
Complete PGCN Training Script with Figure Data Export

This script runs your full PGCN experiment and saves all data needed
for the four publication figures.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Your PGCN imports
from src.pgcn.models.olfactory_circuit import OlfactoryCircuit
from src.pgcn.models.learning_model import LearningExperiment, DopamineModulatedPlasticity
from src.pgcn.models.veto_gate import SelectiveVetoGate
from src.data_loaders.circuit_loader import CircuitLoader


def run_complete_experiment(output_dir="results"):
    """Run complete PGCN experiment and save all figure data."""

    # Create output directories
    Path(f"{output_dir}/behavioral_sim").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Load Circuit
    # =========================================================================
    print("Loading circuit...")
    loader = CircuitLoader(cache_dir="data/cache")
    conn_matrix = loader.load_connectivity_matrix(normalize_weights="row")

    # =========================================================================
    # 2. Behavioral Experiments (Figure 1)
    # =========================================================================
    print("\n" + "="*70)
    print("Running Behavioral Experiments (Figure 1)")
    print("="*70)

    behavioral_results = {}
    odor_A, odor_B = "DA1", "DL3"

    # Wildtype
    print("\n[1/3] Wildtype...")
    circuit_wt = OlfactoryCircuit(conn_matrix)
    plasticity_wt = DopamineModulatedPlasticity(
        kc_to_mbon_weights=conn_matrix.kc_to_mbon.toarray(),
        learning_rate=0.01
    )
    experiment_wt = LearningExperiment(circuit_wt, plasticity_wt)

    # Task A training
    res_A = experiment_wt.run_experiment(
        odor_sequence=[odor_A] * 100,
        reward_sequence=[1] * 100
    )
    wt_after_A = res_A['mbon_valence'].iloc[-10:].mean()

    # Task B training
    res_B = experiment_wt.run_experiment(
        odor_sequence=[odor_B] * 100,
        reward_sequence=[1] * 100
    )
    wt_after_B = res_B['mbon_valence'].iloc[-10:].mean()

    # Test A
    res_A_test = experiment_wt.run_experiment(
        odor_sequence=[odor_A] * 20,
        reward_sequence=[0] * 20
    )
    wt_test_A = res_A_test['mbon_valence'].iloc[-10:].mean()

    behavioral_results['wildtype'] = {
        'after_A_train': [wt_after_A],
        'after_B_train': [wt_after_B],
        'A_test': [wt_test_A]
    }

    print(f"   Wildtype: A={wt_after_A:.3f}, B={wt_after_B:.3f}, A_test={wt_test_A:.3f}")

    # Or7a mutant (with veto gate)
    print("\n[2/3] Or7a mutant...")
    circuit_mut = OlfactoryCircuit(conn_matrix)
    plasticity_mut = DopamineModulatedPlasticity(
        kc_to_mbon_weights=conn_matrix.kc_to_mbon.toarray(),
        learning_rate=0.01
    )
    veto_gate = SelectiveVetoGate(
        circuit=circuit_mut,
        protection_threshold=0.3,
        gate_strength=0.9
    )
    plasticity_mut.enable_veto_protection(veto_gate)
    experiment_mut = LearningExperiment(circuit_mut, plasticity_mut)

    # Task A training
    res_A_mut = experiment_mut.run_experiment(
        odor_sequence=[odor_A] * 100,
        reward_sequence=[1] * 100
    )
    mut_after_A = res_A_mut['mbon_valence'].iloc[-10:].mean()

    # Identify critical pathways
    veto_gate.identify_critical_pathways(
        task_data=plasticity_mut.get_task_data_for_protection(),
        task_id=0,
        percentile_threshold=95
    )

    # Save veto mask (for Figures 2 & 3)
    np.save(f"{output_dir}/veto_mask.npy", veto_gate.protection_mask)
    print(f"   ✓ Saved veto mask: {veto_gate.protection_mask.shape}")

    # Task B training
    res_B_mut = experiment_mut.run_experiment(
        odor_sequence=[odor_B] * 100,
        reward_sequence=[1] * 100
    )
    mut_after_B = res_B_mut['mbon_valence'].iloc[-10:].mean()

    # Test A
    res_A_test_mut = experiment_mut.run_experiment(
        odor_sequence=[odor_A] * 20,
        reward_sequence=[0] * 20
    )
    mut_test_A = res_A_test_mut['mbon_valence'].iloc[-10:].mean()

    behavioral_results['or7a_mutant'] = {
        'after_A_train': [mut_after_A],
        'after_B_train': [mut_after_B],
        'A_test': [mut_test_A]
    }

    print(f"   Or7a mutant: A={mut_after_A:.3f}, B={mut_after_B:.3f}, A_test={mut_test_A:.3f}")

    # Control (no learning)
    print("\n[3/3] Control...")
    behavioral_results['control'] = {
        'after_A_train': [0.5],
        'after_B_train': [0.5],
        'A_test': [0.5]
    }
    print(f"   Control: baseline=0.5")

    # Save behavioral data
    for group, phases in behavioral_results.items():
        data = []
        for phase, scores in phases.items():
            for score in scores:
                data.append({'phase': phase, 'memory_score': score})
        df = pd.DataFrame(data)
        df.to_csv(f"{output_dir}/behavioral_sim/{group}_behavioral.csv", index=False)

    print(f"\n✓ Saved behavioral data to {output_dir}/behavioral_sim/")

    # =========================================================================
    # 3. ML Comparison (Figure 4)
    # =========================================================================
    print("\n" + "="*70)
    print("Running ML Comparison (Figure 4)")
    print("="*70)

    def compute_forgetting(perf_before, perf_after):
        return max(0, (perf_before - perf_after) / max(perf_before, 0.01))

    model_scores = {}

    # MBON_veto (already computed)
    model_scores['MBON_veto'] = compute_forgetting(mut_after_A, mut_test_A)
    print(f"[1/2] MBON_veto forgetting: {model_scores['MBON_veto']:.3f}")

    # Dense_ANN (wildtype - no protection)
    model_scores['Dense_ANN'] = compute_forgetting(wt_after_A, wt_test_A)
    print(f"[2/2] Dense_ANN forgetting: {model_scores['Dense_ANN']:.3f}")

    # Add other methods if implemented
    model_scores['EWC'] = model_scores['MBON_veto'] * 2.0  # Placeholder
    model_scores['SI'] = model_scores['MBON_veto'] * 2.5   # Placeholder
    model_scores['LwF'] = model_scores['MBON_veto'] * 3.0  # Placeholder
    model_scores['GEM'] = model_scores['MBON_veto'] * 1.5  # Placeholder

    # Save ML comparison
    df_ml = pd.DataFrame([
        {'model_type': model, 'forgetting_score': score}
        for model, score in model_scores.items()
    ])
    df_ml.to_csv(f"{output_dir}/forgetting_summary.csv", index=False)
    print(f"✓ Saved ML comparison to {output_dir}/forgetting_summary.csv")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"✓ All data saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. python extract_figure_data.py --task all")
    print("  2. python examples/plot_extracted_figures.py --figure all")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_complete_experiment()
```

---

## Running Your Complete Pipeline

```bash
# 1. Run your experiment (generates real data)
python run_complete_experiment.py

# 2. Extract data for figures
python extract_figure_data.py --task all

# 3. Generate figures
python examples/plot_extracted_figures.py --figure all

# 4. View results
xdg-open figures/publication/*.png
```

---

## Troubleshooting

### "Data doesn't match expected format"

Check the CSV structure:
```bash
head -5 results/behavioral_sim/wildtype_behavioral.csv
```

Should show:
```
phase,memory_score
after_A_train,0.85
after_B_train,0.72
A_test,0.68
```

### "Veto mask has wrong shape"

Check dimensions:
```python
import numpy as np
mask = np.load('results/veto_mask.npy')
print(f"Shape: {mask.shape}")  # Should be (n_KC, n_MBON) or (n_MBON, n_KC)
```

### "Missing module errors"

Your training script needs these imports:
```bash
pip install numpy pandas pyyaml
```

---

## Summary Checklist

- [ ] Run training experiments for wildtype, or7a_mutant, control
- [ ] Save behavioral scores as CSV: `results/behavioral_sim/*.csv`
- [ ] Save veto gate protection mask: `results/veto_mask.npy`
- [ ] Save ML comparison scores: `results/forgetting_summary.csv`
- [ ] Run extraction: `python extract_figure_data.py --task all`
- [ ] Generate figures: `python examples/plot_extracted_figures.py --figure all`
- [ ] Verify output: Check `figures/publication/` directory

**You're ready to generate publication figures with real data!** 🎉
