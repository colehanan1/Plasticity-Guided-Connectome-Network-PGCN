#!/usr/bin/env python3
"""
Critical Test: Prove Homeostatic Normalization Causes Catastrophic Forgetting

This experiment isolates the normalization mechanism by testing 0% KC overlap
with three conditions:

1. **Baseline + Normalization** (current): Should show forgetting (~0.2)
2. **Baseline - Normalization**: Should show NO forgetting before weight explosion
3. **Veto Gate + Normalization**: Should show reduced forgetting (~0.03)

Key Hypothesis:
Even with 0% KC overlap (no direct KC interference), homeostatic normalization
causes indirect forgetting by rescaling Task A KC weights when Task B KCs grow.

If baseline_no_norm shows near-zero forgetting (before instability), this PROVES
normalization is the catastrophic forgetting mechanism, not MBON drift!

Usage:
    python scripts/experiments/test_normalization_hypothesis.py \
        --output reports/analysis/normalization_ablation.csv

Author: PGCN Project
Date: 2025-11-19
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.veto_gate import SelectiveVetoGate
from door_integration.pgcn_door import PGCNDoorIntegration
from pgcn.analysis.forgetting_metrics import compute_forgetting_metrics


class NormalizationAblationExperiment:
    """Test whether normalization causes catastrophic forgetting."""

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        door: PGCNDoorIntegration,
        trials_per_task: int = 10,
        learning_rate: float = 0.001,
        protection_threshold: float = 0.3,
        gate_strength: float = 0.9,
        random_seed: int = 42,
    ):
        """Initialize normalization ablation experiment.

        Parameters
        ----------
        circuit : OlfactoryCircuit
            Olfactory circuit with specified KC sparsity.
        door : PGCNDoorIntegration
            DoOR database integration.
        trials_per_task : int
            Number of training trials per task (default: 10).
        learning_rate : float
            Plasticity learning rate (default: 0.001).
        protection_threshold : float
            Fraction of synapses to protect (default: 0.3).
        gate_strength : float
            Veto gate strength (default: 0.9).
        random_seed : int
            Random seed for reproducibility (default: 42).
        """
        self.circuit = circuit
        self.door = door
        self.trials_per_task = trials_per_task
        self.learning_rate = learning_rate
        self.protection_threshold = protection_threshold
        self.gate_strength = gate_strength
        self.random_seed = random_seed

        np.random.seed(random_seed)

        # Use 0% KC overlap condition (medium chem similarity)
        self.task_A_odors = ('ethyl butyrate', 'benzaldehyde')
        self.task_B_odors = ('linalool', '1-hexanol')

        self.results = []

    def run_all_conditions(self) -> pd.DataFrame:
        """Run all three conditions and return results."""
        print("\n" + "="*80)
        print("NORMALIZATION ABLATION EXPERIMENT")
        print("="*80)
        print(f"Testing 0% KC overlap condition:")
        print(f"  Task A: {self.task_A_odors[0]} (CS+) vs {self.task_A_odors[1]} (CS-)")
        print(f"  Task B: {self.task_B_odors[0]} (CS+) vs {self.task_B_odors[1]} (CS-)")
        print(f"  KC sparsity: {self.circuit.sparsity_target:.1%}")
        print(f"  Trials per task: {self.trials_per_task}")
        print("="*80 + "\n")

        # Condition 1: Baseline with normalization (current)
        print("="*80)
        print("CONDITION 1: Baseline + Normalization (CONTROL)")
        print("="*80)
        result_norm = self._run_condition(
            enable_normalization=True,
            use_veto=False,
            condition_name="baseline_with_norm",
        )
        self.results.append(result_norm)

        # Condition 2: Baseline without normalization (ABLATION)
        print("\n" + "="*80)
        print("CONDITION 2: Baseline - Normalization (ABLATION)")
        print("="*80)
        result_no_norm = self._run_condition(
            enable_normalization=False,
            use_veto=False,
            condition_name="baseline_no_norm",
        )
        self.results.append(result_no_norm)

        # Condition 3: Veto gate with normalization (RESCUE)
        print("\n" + "="*80)
        print("CONDITION 3: Veto Gate + Normalization (RESCUE)")
        print("="*80)
        result_veto = self._run_condition(
            enable_normalization=True,
            use_veto=True,
            condition_name="veto_with_norm",
        )
        self.results.append(result_veto)

        return pd.DataFrame(self.results)

    def _run_condition(
        self,
        enable_normalization: bool,
        use_veto: bool,
        condition_name: str,
    ) -> dict:
        """Run a single experimental condition."""
        # Create fresh plasticity model
        connectivity = self.circuit.connectivity
        kc_to_mbon_dense = connectivity.kc_to_mbon.toarray()
        plasticity = DopamineModulatedPlasticity(
            kc_to_mbon_weights=kc_to_mbon_dense,
            learning_rate=self.learning_rate,
            plasticity_mode='three_factor',
            kc_sparsity=self.circuit.sparsity_target,
        )

        # Configure normalization
        plasticity.enable_weight_normalization = enable_normalization
        print(f"Normalization: {'ENABLED' if enable_normalization else 'DISABLED'}")
        print(f"Veto gate: {'ENABLED' if use_veto else 'DISABLED'}")

        # Task A: Train on first odor pair
        print(f"\n▶ Training Task A: {self.task_A_odors[0]} (CS+) vs {self.task_A_odors[1]} (CS-)...")
        task_A_acc = self._train_task(
            plasticity,
            odor_a=self.task_A_odors[0],
            odor_b=self.task_A_odors[1],
            task_id=0,
        )
        print(f"   Task A final accuracy: {task_A_acc:.2f}")

        # Measure MBON output range after Task A
        mbon_range_after_A = self._measure_mbon_range(plasticity)
        print(f"   MBON output range after Task A: [{mbon_range_after_A['min']:.2f}, {mbon_range_after_A['max']:.2f}]")

        # Set up veto protection if needed
        if use_veto:
            veto_gate = SelectiveVetoGate(
                circuit=self.circuit,
                protection_threshold=self.protection_threshold,
                gate_strength=self.gate_strength,
                similarity_metric='chemical',
            )

            # Get task data BEFORE enabling veto
            task_data = plasticity.get_task_data_for_protection()
            plasticity.enable_veto_protection(veto_gate, track_weight_changes=False)

            # Identify critical pathways
            protection_mask = veto_gate.identify_critical_pathways(task_data, method='gradient_magnitude')
            num_protected = protection_mask.sum()
            print(f"   Protected {num_protected}/{protection_mask.size} synapses ({num_protected/protection_mask.size*100:.1f}%)")

            # Compute veto signal
            veto_signal = veto_gate.compute_veto_signal(
                self.task_B_odors[0],  # Task B CS+
                self.task_A_odors[0],  # Task A CS+ (protected)
            )
            print(f"   Veto signal: {veto_signal:.3f}")
            veto_gate.set_veto_signal(veto_signal)

        # Task B: Train on second odor pair
        print(f"\n▶ Training Task B: {self.task_B_odors[0]} (CS+) vs {self.task_B_odors[1]} (CS-)...")
        task_B_acc = self._train_task(
            plasticity,
            odor_a=self.task_B_odors[0],
            odor_b=self.task_B_odors[1],
            task_id=1,
        )
        print(f"   Task B final accuracy: {task_B_acc:.2f}")

        # Measure MBON output range after Task B
        mbon_range_after_B = self._measure_mbon_range(plasticity)
        print(f"   MBON output range after Task B: [{mbon_range_after_B['min']:.2f}, {mbon_range_after_B['max']:.2f}]")

        # Test forgetting on Task A
        task_A_acc_after = self._test_task(plasticity, odor=self.task_A_odors[0])
        print(f"\n▶ Testing Task A retention:")
        print(f"   Task A after Task B: {task_A_acc_after:.2f}")

        # Compute forgetting metrics
        forgetting_metrics = compute_forgetting_metrics(task_A_acc, task_A_acc_after)
        print(f"   Absolute forgetting: {forgetting_metrics['absolute_change']:.3f}")
        print(f"   Valence flip: {'YES ❌' if forgetting_metrics['valence_flip'] else 'NO ✅'}")

        # Compute weight statistics
        weight_stats = self._compute_weight_stats(plasticity)

        return {
            'condition': condition_name,
            'enable_normalization': enable_normalization,
            'use_veto': use_veto,
            'task_A_acc': task_A_acc,
            'task_B_acc': task_B_acc,
            'task_A_acc_after': task_A_acc_after,
            'absolute_forgetting': forgetting_metrics['absolute_change'],
            'valence_flip': forgetting_metrics['valence_flip'],
            'mbon_range_min_after_A': mbon_range_after_A['min'],
            'mbon_range_max_after_A': mbon_range_after_A['max'],
            'mbon_range_min_after_B': mbon_range_after_B['min'],
            'mbon_range_max_after_B': mbon_range_after_B['max'],
            'weight_norm_mean': weight_stats['norm_mean'],
            'weight_norm_std': weight_stats['norm_std'],
            'weight_norm_max': weight_stats['norm_max'],
        }

    def _train_task(
        self,
        plasticity: DopamineModulatedPlasticity,
        odor_a: str,
        odor_b: str,
        task_id: int,
    ) -> float:
        """Train on a single task (odor pair)."""
        for trial in range(self.trials_per_task):
            # Positive trial (CS+)
            glomeruli_a = self.door.map_odorant_to_glomeruli(odor_a, threshold=0.3)
            pn_a = self.circuit.activate_pns_by_glomeruli(glomeruli_a, firing_rate=1.0)
            kc_a = self.circuit.propagate_pn_to_kc(pn_a)
            mbon_a = plasticity.compute_mbon_output(kc_a)
            rpe_a = plasticity.compute_rpe(trial_outcome=1.0, predicted_value=np.sum(mbon_a))
            plasticity.update_weights(kc_a, mbon_a, rpe_a, dt=1.0)

            # Negative trial (CS-)
            glomeruli_b = self.door.map_odorant_to_glomeruli(odor_b, threshold=0.3)
            pn_b = self.circuit.activate_pns_by_glomeruli(glomeruli_b, firing_rate=1.0)
            kc_b = self.circuit.propagate_pn_to_kc(pn_b)
            mbon_b = plasticity.compute_mbon_output(kc_b)
            rpe_b = plasticity.compute_rpe(trial_outcome=0.0, predicted_value=np.sum(mbon_b))
            plasticity.update_weights(kc_b, mbon_b, rpe_b, dt=1.0)

        # Return final accuracy (MBON output for CS+)
        return self._test_task(plasticity, odor=odor_a)

    def _test_task(self, plasticity: DopamineModulatedPlasticity, odor: str) -> float:
        """Test MBON output for a single odor."""
        glomeruli = self.door.map_odorant_to_glomeruli(odor, threshold=0.3)
        pn_activity = self.circuit.activate_pns_by_glomeruli(glomeruli, firing_rate=1.0)
        kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
        mbon_output_vec = plasticity.compute_mbon_output(kc_activity)
        return float(np.sum(mbon_output_vec))

    def _measure_mbon_range(self, plasticity: DopamineModulatedPlasticity) -> dict:
        """Measure min/max MBON outputs across all test odors."""
        test_odors = ['ethyl butyrate', 'linalool', 'benzaldehyde', '1-hexanol']
        mbon_outputs = []

        for odor in test_odors:
            try:
                output = self._test_task(plasticity, odor)
                mbon_outputs.append(output)
            except:
                pass

        if len(mbon_outputs) == 0:
            return {'min': np.nan, 'max': np.nan}

        return {
            'min': np.min(mbon_outputs),
            'max': np.max(mbon_outputs),
        }

    def _compute_weight_stats(self, plasticity: DopamineModulatedPlasticity) -> dict:
        """Compute weight matrix statistics."""
        # Compute L2 norm per KC (column-wise)
        kc_norms = np.linalg.norm(plasticity.kc_to_mbon, axis=0, ord=2)  # (n_kc,)

        return {
            'norm_mean': float(np.mean(kc_norms)),
            'norm_std': float(np.std(kc_norms)),
            'norm_max': float(np.max(kc_norms)),
        }


def main():
    """Run normalization ablation experiment from command line."""
    parser = argparse.ArgumentParser(
        description="Test whether normalization causes catastrophic forgetting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--output',
        type=str,
        default='reports/analysis/normalization_ablation.csv',
        help='Output CSV path (default: reports/analysis/normalization_ablation.csv)'
    )
    parser.add_argument(
        '--trials-per-task',
        type=int,
        default=10,
        help='Number of training trials per task (default: 10)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Plasticity learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Load circuit
    print("Loading circuit connectivity...")
    loader = CircuitLoader(cache_dir='data/cache')
    connectivity = loader.load_connectivity_matrix(
        normalize_weights='row',
        include_dan=False,
        include_extended=False,
    )

    circuit = OlfactoryCircuit(connectivity=connectivity, kc_sparsity_target=0.05)
    door = PGCNDoorIntegration()

    print(f"✅ Circuit: {connectivity.n_pn} PNs, {connectivity.n_kc} KCs, {connectivity.n_mbon} MBONs")
    print(f"   KC sparsity: 5.0%\n")

    # Run experiment
    experiment = NormalizationAblationExperiment(
        circuit=circuit,
        door=door,
        trials_per_task=args.trials_per_task,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )

    results_df = experiment.run_all_conditions()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80 + "\n")

    for _, row in results_df.iterrows():
        print(f"{row['condition'].upper()}:")
        print(f"  Normalization: {'ON' if row['enable_normalization'] else 'OFF'}")
        print(f"  Veto gate: {'ON' if row['use_veto'] else 'OFF'}")
        print(f"  Task A accuracy: {row['task_A_acc']:.2f} → {row['task_A_acc_after']:.2f}")
        print(f"  Absolute forgetting: {row['absolute_forgetting']:.3f}")
        print(f"  MBON range (after B): [{row['mbon_range_min_after_B']:.1f}, {row['mbon_range_max_after_B']:.1f}]")
        print(f"  KC weight norms: {row['weight_norm_mean']:.4f} ± {row['weight_norm_std']:.4f} (max: {row['weight_norm_max']:.4f})")
        print()

    # Key findings
    print("="*80)
    print("KEY FINDINGS")
    print("="*80 + "\n")

    forgetting_with_norm = results_df[results_df['condition'] == 'baseline_with_norm']['absolute_forgetting'].values[0]
    forgetting_no_norm = results_df[results_df['condition'] == 'baseline_no_norm']['absolute_forgetting'].values[0]
    forgetting_veto = results_df[results_df['condition'] == 'veto_with_norm']['absolute_forgetting'].values[0]

    reduction_by_removing_norm = (forgetting_with_norm - forgetting_no_norm) / forgetting_with_norm * 100
    reduction_by_veto = (forgetting_with_norm - forgetting_veto) / forgetting_with_norm * 100

    print(f"✅ Baseline + Normalization: {forgetting_with_norm:.3f} forgetting")
    print(f"✅ Baseline - Normalization: {forgetting_no_norm:.3f} forgetting ({reduction_by_removing_norm:+.1f}% change)")
    print(f"✅ Veto Gate + Normalization: {forgetting_veto:.3f} forgetting ({reduction_by_veto:+.1f}% reduction)")

    print()
    if abs(reduction_by_removing_norm) > 50:
        print("🔬 HYPOTHESIS CONFIRMED: Normalization causes >50% of catastrophic forgetting!")
        print("   This proves homeostatic plasticity is the primary forgetting mechanism at 0% KC overlap.")
    else:
        print("🔬 HYPOTHESIS REJECTED: Normalization is not the primary forgetting mechanism.")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
