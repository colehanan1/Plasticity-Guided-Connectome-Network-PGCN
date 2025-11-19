#!/usr/bin/env python3
"""
KC Overlap vs Veto Efficacy Experiment

This script systematically tests the hypothesis that veto gate protection
efficacy depends on KC overlap between tasks. We expect:
- Low KC overlap (<10%): Veto provides ~0% benefit (sparse coding sufficient)
- Medium KC overlap (15-40%): Veto provides ~20-40% benefit
- High KC overlap (>50%): Veto provides ~50-70% benefit

Author: PGCN Project
Date: 2025-11-18

Usage:
    python scripts/experiments/kc_overlap_vs_veto_efficacy.py \\
        --output reports/experiments/overlap/results.csv \\
        --kc-sparsity 0.05 \\
        --trials-per-task 30
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.veto_gate import SelectiveVetoGate
from door_integration.pgcn_door import PGCNDoorIntegration
from pgcn.analysis.odor_similarity import measure_kc_overlap, compute_chemical_similarity
from pgcn.analysis.forgetting_metrics import compute_forgetting_metrics, format_forgetting_report


class KCOverlapExperiment:
    """Systematic experiment testing KC overlap vs veto gate efficacy."""

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        door: PGCNDoorIntegration,
        trials_per_task: int = 30,
        learning_rate: float = 0.005,
        protection_threshold: float = 0.3,
        gate_strength: float = 0.9,
        random_seed: int = 42,
        diagnostic_dir: str = None,
    ):
        """Initialize KC overlap experiment.

        Parameters
        ----------
        circuit : OlfactoryCircuit
            Olfactory circuit with specified KC sparsity.
        door : PGCNDoorIntegration
            DoOR database integration.
        trials_per_task : int
            Number of training trials per task.
        learning_rate : float
            Plasticity learning rate.
        protection_threshold : float
            Fraction of synapses to protect (0.3 = top 30%).
        gate_strength : float
            Veto gate strength (0.9 = 90% suppression).
        random_seed : int
            Random seed for reproducibility.
        diagnostic_dir : str, optional
            Directory to save diagnostic data (weights, KC indices) for MBON drift analysis.
            If None, diagnostic data is not saved.
        """
        self.circuit = circuit
        self.door = door
        self.trials_per_task = trials_per_task
        self.learning_rate = learning_rate
        self.protection_threshold = protection_threshold
        self.gate_strength = gate_strength
        self.random_seed = random_seed
        self.diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir else None

        np.random.seed(random_seed)

        # Define task pairs for KC overlap sweep
        # Based on comprehensive pairwise KC overlap analysis (45 pairs from 10 odors)
        # Selected to span 0-43% KC overlap with diverse chemical similarities
        # NOTE: citral excluded (Or83c not mapped to glomerulus)
        self.task_pairs = {
            # 0% KC overlap (sparse coding sufficient)
            'overlap_0_low_chem': {
                'task_A': ('1-hexanol', 'benzaldehyde'),  # Dummy (won't train Task A)
                'task_B': ('methyl salicylate', 'benzaldehyde'),  # Only train Task B
                'chemical_sim': 0.347,
                'expected_kc_overlap': 0.0,
                'description': '0% overlap, low chem sim (35%)',
            },
            'overlap_0_med_chem': {
                'task_A': ('ethyl butyrate', 'benzaldehyde'),
                'task_B': ('linalool', '1-hexanol'),
                'chemical_sim': 0.413,
                'expected_kc_overlap': 0.0,
                'description': '0% overlap, medium chem sim (41%)',
            },
            'overlap_0_high_chem': {
                'task_A': ('linalool', '1-hexanol'),
                'task_B': ('geranyl acetate', 'benzaldehyde'),
                'chemical_sim': 0.620,
                'expected_kc_overlap': 0.0,
                'description': '0% overlap, high chem sim (62%)',
            },
            # 7-10% KC overlap (low interference)
            'overlap_7': {
                'task_A': ('1-hexanol', 'linalool'),
                'task_B': ('ethyl acetate', 'benzaldehyde'),
                'chemical_sim': 0.503,
                'expected_kc_overlap': 7.0,
                'description': '7% overlap, medium chem sim (50%)',
            },
            'overlap_10': {
                'task_A': ('ethyl butyrate', 'benzaldehyde'),
                'task_B': ('ethyl acetate', '1-hexanol'),
                'chemical_sim': 0.772,
                'expected_kc_overlap': 10.5,
                'description': '10% overlap, high chem sim (77%)',
            },
            # 13-18% KC overlap (moderate interference)
            'overlap_14': {
                'task_A': ('1-hexanol', 'benzaldehyde'),
                'task_B': ('linalool', 'ethyl butyrate'),
                'chemical_sim': 0.448,
                'expected_kc_overlap': 13.6,
                'description': '14% overlap, medium chem sim (45%)',
            },
            'overlap_16': {
                'task_A': ('benzaldehyde', 'ethyl butyrate'),
                'task_B': ('2-heptanone', '1-hexanol'),
                'chemical_sim': 0.601,
                'expected_kc_overlap': 15.6,
                'description': '16% overlap, high chem sim (60%)',
            },
            'overlap_18': {
                'task_A': ('1-hexanol', 'linalool'),
                'task_B': ('ethyl butyrate', 'benzaldehyde'),
                'chemical_sim': 0.717,
                'expected_kc_overlap': 18.3,
                'description': '18% overlap, high chem sim (72%)',
            },
            # 40-45% KC overlap (high interference)
            'overlap_43': {
                'task_A': ('1-hexanol', 'ethyl butyrate'),
                'task_B': ('2-heptanone', 'linalool'),
                'chemical_sim': 0.804,
                'expected_kc_overlap': 42.9,
                'description': '43% overlap, very high chem sim (80%)',
            },
        }

        self.results = []

    def run_full_sweep(self) -> pd.DataFrame:
        """Run experiment across all similarity levels.

        Returns
        -------
        pd.DataFrame
            Results with columns: similarity_level, strategy, kc_overlap_pct,
            baseline_forgetting, veto_forgetting, protection_benefit, etc.
        """
        print("\n" + "="*80)
        print("KC OVERLAP VS VETO EFFICACY EXPERIMENT")
        print("="*80)
        print(f"KC sparsity: {self.circuit.sparsity_target:.1%}")
        print(f"Trials per task: {self.trials_per_task}")
        print(f"Protection threshold: {self.protection_threshold:.1%}")
        print(f"Gate strength: {self.gate_strength:.1%}")
        print("="*80 + "\n")

        # Test all task pairs in order of increasing KC overlap
        task_pair_order = [
            'overlap_0_low_chem', 'overlap_0_med_chem', 'overlap_0_high_chem',
            'overlap_7', 'overlap_10',
            'overlap_14', 'overlap_16', 'overlap_18',
            'overlap_43',
        ]

        for level in task_pair_order:
            print(f"\n{'='*80}")
            print(f"TESTING: {level.upper()}")
            print(f"{'='*80}")

            task_config = self.task_pairs[level]
            print(f"Description: {task_config['description']}")
            print(f"Task A: {task_config['task_A'][0]} (CS+) vs {task_config['task_A'][1]} (CS-)")
            print(f"Task B: {task_config['task_B'][0]} (CS+) vs {task_config['task_B'][1]} (CS-)")

            # Measure KC overlap BEFORE training
            kc_overlap = measure_kc_overlap(
                self.circuit,
                task_config['task_A'][0],  # Task A CS+
                task_config['task_B'][0],  # Task B CS+
                self.door,
            )

            print(f"\n📊 PRE-TRAINING KC OVERLAP:")
            print(f"   Active KCs (Task A): {kc_overlap['active_kcs_A']}")
            print(f"   Active KCs (Task B): {kc_overlap['active_kcs_B']}")
            print(f"   Overlap: {kc_overlap['kc_overlap_count']} KCs ({kc_overlap['kc_overlap_pct']:.1f}%)")
            print(f"   Jaccard index: {kc_overlap['jaccard_index']:.3f}")

            # Run baseline (no protection)
            print(f"\n▶ Running BASELINE (no protection)...")
            baseline_results = self._run_condition(
                level=level,
                task_config=task_config,
                strategy='baseline',
                kc_overlap_measured=kc_overlap,
            )

            # Run veto gate
            print(f"\n▶ Running VETO GATE protection...")
            veto_results = self._run_condition(
                level=level,
                task_config=task_config,
                strategy='veto_gate',
                kc_overlap_measured=kc_overlap,
            )

            # Compute protection benefit (using absolute change as primary metric)
            baseline_forgetting = baseline_results['forgetting_pct']
            veto_forgetting = veto_results['forgetting_pct']
            protection_benefit = baseline_forgetting - veto_forgetting

            print(f"\n✅ RESULTS:")
            print(f"   Baseline forgetting (absolute): {baseline_forgetting:.3f}")
            print(f"   Baseline valence flip: {'YES ❌' if baseline_results['valence_flip'] else 'NO ✅'}")
            print(f"   Veto forgetting (absolute): {veto_forgetting:.3f}")
            print(f"   Veto valence flip: {'YES ❌' if veto_results['valence_flip'] else 'NO ✅'}")
            relative_reduction = abs(protection_benefit) / baseline_forgetting * 100 if baseline_forgetting > 1e-6 else 0.0
            print(f"   Protection benefit: {protection_benefit:+.3f} ({relative_reduction:.1f}% reduction)")

            # Log summary
            summary = {
                'condition': level,
                'expected_kc_overlap': task_config.get('expected_kc_overlap', kc_overlap['kc_overlap_pct']),
                'measured_kc_overlap_pct': kc_overlap['kc_overlap_pct'],
                'kc_overlap_count': kc_overlap['kc_overlap_count'],
                'jaccard_index': kc_overlap['jaccard_index'],
                'chemical_similarity': task_config['chemical_sim'],
                'description': task_config['description'],
                'baseline_forgetting_absolute': baseline_forgetting,
                'baseline_valence_flip': baseline_results['valence_flip'],
                'veto_forgetting_absolute': veto_forgetting,
                'veto_valence_flip': veto_results['valence_flip'],
                'protection_benefit_absolute': protection_benefit,
                'relative_reduction_pct': abs(protection_benefit) / baseline_forgetting * 100 if baseline_forgetting > 1e-6 else 0.0,
                'task_A_odor': task_config['task_A'][0],
                'task_B_odor': task_config['task_B'][0],
                'kc_sparsity': self.circuit.sparsity_target,
            }
            self.results.append(summary)

        return pd.DataFrame(self.results)

    def _run_condition(
        self,
        level: str,
        task_config: Dict,
        strategy: str,
        kc_overlap_measured: Dict,
    ) -> Dict:
        """Run single condition (one similarity level, one strategy).

        Returns
        -------
        Dict
            Results including forgetting_pct, task_A_acc, task_B_acc, etc.
        """
        # Create fresh plasticity model
        connectivity = self.circuit.connectivity
        kc_to_mbon_dense = connectivity.kc_to_mbon.toarray()
        plasticity = DopamineModulatedPlasticity(
            kc_to_mbon_weights=kc_to_mbon_dense,
            learning_rate=self.learning_rate,
            plasticity_mode='three_factor',
            kc_sparsity=self.circuit.sparsity_target,
        )

        # Task A: Train on first odor pair
        task_A_odors = task_config['task_A']
        print(f"   Training Task A: {task_A_odors[0]} (CS+) vs {task_A_odors[1]} (CS-)...")

        # Get Task A active KCs for diagnostic analysis
        glom_A = self.door.map_odorant_to_glomeruli(task_A_odors[0], threshold=0.3)
        pn_A = self.circuit.activate_pns_by_glomeruli(glom_A, firing_rate=1.0)
        kc_A = self.circuit.propagate_pn_to_kc(pn_A)
        task_A_active_kcs = np.where(kc_A > 0)[0]

        task_A_acc = self._train_task(
            plasticity,
            odor_a=task_A_odors[0],
            odor_b=task_A_odors[1],
            task_id=0,
        )

        print(f"   Task A final accuracy: {task_A_acc:.2f}")

        # Save weights after Task A (before Task B) for diagnostic analysis
        if self.diagnostic_dir is not None:
            self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
            weights_after_task_A = plasticity.kc_to_mbon.copy()
            weights_file = self.diagnostic_dir / f"weights_{level}_{strategy}_after_taskA.npy"
            np.save(weights_file, weights_after_task_A)
            print(f"   💾 Saved weights to: {weights_file.name}")

        # Set up veto protection if needed
        if strategy == 'veto_gate':
            veto_gate = SelectiveVetoGate(
                circuit=self.circuit,
                protection_threshold=self.protection_threshold,
                gate_strength=self.gate_strength,
                similarity_metric='chemical',
            )

            # CRITICAL FIX: Get task data BEFORE enabling veto protection
            # (otherwise enable_veto clears weight_change_history!)
            task_data = plasticity.get_task_data_for_protection()

            # Now enable veto protection (without clearing history)
            plasticity.enable_veto_protection(veto_gate, track_weight_changes=False)

            # Identify critical pathways using Task A's weight changes
            protection_mask = veto_gate.identify_critical_pathways(task_data, method='gradient_magnitude')
            num_protected = protection_mask.sum()
            total_synapses = protection_mask.size
            print(f"   Protected synapses: {num_protected}/{total_synapses} ({num_protected/total_synapses*100:.1f}%)")

            # Compute veto signal
            veto_signal = veto_gate.compute_veto_signal(
                task_config['task_B'][0],  # Task B CS+
                task_config['task_A'][0],  # Task A CS+ (protected)
            )
            print(f"   Veto signal: {veto_signal:.3f}")
            veto_gate.set_veto_signal(veto_signal)
        else:
            veto_signal = 0.0

        # Task B: Train on second odor pair
        task_B_odors = task_config['task_B']
        print(f"   Training Task B: {task_B_odors[0]} (CS+) vs {task_B_odors[1]} (CS-)...")

        # Get Task B active KCs for diagnostic analysis
        glom_B = self.door.map_odorant_to_glomeruli(task_B_odors[0], threshold=0.3)
        pn_B = self.circuit.activate_pns_by_glomeruli(glom_B, firing_rate=1.0)
        kc_B = self.circuit.propagate_pn_to_kc(pn_B)
        task_B_active_kcs = np.where(kc_B > 0)[0]

        task_B_acc = self._train_task(
            plasticity,
            odor_a=task_B_odors[0],
            odor_b=task_B_odors[1],
            task_id=1,
        )

        print(f"   Task B final accuracy: {task_B_acc:.2f}")

        # Save weights after Task B and KC indices for diagnostic analysis
        if self.diagnostic_dir is not None:
            weights_after_task_B = plasticity.kc_to_mbon.copy()
            weights_file = self.diagnostic_dir / f"weights_{level}_{strategy}_after_taskB.npy"
            np.save(weights_file, weights_after_task_B)
            print(f"   💾 Saved weights to: {weights_file.name}")

            # Save KC indices
            kc_file = self.diagnostic_dir / f"active_kcs_{level}.npz"
            np.savez(kc_file, task_A_kcs=task_A_active_kcs, task_B_kcs=task_B_active_kcs)
            print(f"   💾 Saved KC indices to: {kc_file.name}")

        # Test forgetting on Task A
        task_A_acc_after = self._test_task(
            plasticity,
            odor=task_A_odors[0],
        )

        # Compute robust forgetting metrics
        forgetting_metrics = compute_forgetting_metrics(task_A_acc, task_A_acc_after)

        print(f"   Task A after Task B: {task_A_acc_after:.2f}")
        print(f"   {format_forgetting_report(forgetting_metrics)}")

        return {
            'task_A_acc': task_A_acc,
            'task_B_acc': task_B_acc,
            'task_A_acc_after': task_A_acc_after,
            'forgetting_pct': forgetting_metrics['absolute_change'],  # Use absolute change as primary
            'forgetting_relative_pct': forgetting_metrics['relative_change'] * 100.0,
            'valence_flip': forgetting_metrics['valence_flip'],
            'performance_drop': forgetting_metrics['performance_drop'],
            'veto_signal': veto_signal,
        }

    def _train_task(
        self,
        plasticity: DopamineModulatedPlasticity,
        odor_a: str,
        odor_b: str,
        task_id: int,
    ) -> float:
        """Train on a single discrimination task.

        Returns
        -------
        float
            Final MBON output for odor_a (CS+).
        """
        for trial in range(self.trials_per_task):
            # Alternate CS+ and CS-
            if trial % 2 == 0:
                odor, reward = odor_a, 1.0
            else:
                odor, reward = odor_b, 0.0

            # Forward pass
            glomeruli = self.door.map_odorant_to_glomeruli(odor, threshold=0.3)
            pn_activity = self.circuit.activate_pns_by_glomeruli(glomeruli, firing_rate=1.0)
            kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
            mbon_output_vec = plasticity.compute_mbon_output(kc_activity)
            mbon_output = float(np.sum(mbon_output_vec))

            # Update weights
            predicted_value = mbon_output / (plasticity.mbon_output_max * len(mbon_output_vec))
            predicted_value = np.clip(predicted_value, -1.0, 1.0)
            rpe = plasticity.compute_rpe(reward, predicted_value, learning_rate_rpe=0.1)
            plasticity.update_weights(kc_activity, mbon_output_vec, rpe, dt=1.0)

        # Return final accuracy for odor_a
        return self._test_task(plasticity, odor_a)

    def _test_task(
        self,
        plasticity: DopamineModulatedPlasticity,
        odor: str,
    ) -> float:
        """Test MBON output for a single odor.

        Returns
        -------
        float
            MBON output (proxy for accuracy).
        """
        glomeruli = self.door.map_odorant_to_glomeruli(odor, threshold=0.3)
        pn_activity = self.circuit.activate_pns_by_glomeruli(glomeruli, firing_rate=1.0)
        kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
        mbon_output_vec = plasticity.compute_mbon_output(kc_activity)
        return float(np.sum(mbon_output_vec))


def main():
    """Run KC overlap experiment from command line."""
    parser = argparse.ArgumentParser(
        description="KC Overlap vs Veto Efficacy Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--output',
        type=str,
        default='reports/experiments/overlap/kc_overlap_results.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--kc-sparsity',
        type=float,
        default=0.05,
        help='Fraction of KCs active per odor (default: 0.05)'
    )
    parser.add_argument(
        '--trials-per-task',
        type=int,
        default=30,
        help='Training trials per task (default: 30)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.005,
        help='Plasticity learning rate (default: 0.005)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--diagnostic-dir',
        type=str,
        default=None,
        help='Directory to save diagnostic data (weights, KC indices) for MBON drift analysis. If not specified, diagnostics are not saved.'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load circuit
    print("Loading circuit connectivity...")
    loader = CircuitLoader(cache_dir='data/cache')
    connectivity = loader.load_connectivity_matrix(
        normalize_weights='row',
        include_dan=False,
        include_extended=False,
    )

    circuit = OlfactoryCircuit(connectivity=connectivity, kc_sparsity_target=args.kc_sparsity)
    print(f"✅ Circuit: {connectivity.n_pn} PNs, {connectivity.n_kc} KCs, {connectivity.n_mbon} MBONs")
    print(f"   KC sparsity: {args.kc_sparsity:.1%} ({int(connectivity.n_kc * args.kc_sparsity)} active per odor)")

    # Load DoOR
    print("Loading DoOR database...")
    door = PGCNDoorIntegration()
    print("✅ DoOR loaded")

    # Run experiment
    experiment = KCOverlapExperiment(
        circuit=circuit,
        door=door,
        trials_per_task=args.trials_per_task,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        diagnostic_dir=args.diagnostic_dir,
    )

    if args.diagnostic_dir:
        print(f"\n📊 DIAGNOSTIC MODE ENABLED")
        print(f"   Saving weights and KC indices to: {args.diagnostic_dir}")

    results_df = experiment.run_full_sweep()

    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to: {args.output}")

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for _, row in results_df.iterrows():
        print(f"\n{row['description']}")
        print(f"  Condition: {row['condition']}")
        print(f"  Expected KC overlap: {row['expected_kc_overlap']:.1f}%")
        print(f"  Measured KC overlap: {row['measured_kc_overlap_pct']:.1f}%")
        print(f"  Chemical similarity: {row['chemical_similarity']:.3f}")
        print(f"  Baseline forgetting: {row['baseline_forgetting_absolute']:.3f} (valence flip: {'YES ❌' if row['baseline_valence_flip'] else 'NO ✅'})")
        print(f"  Veto forgetting: {row['veto_forgetting_absolute']:.3f} (valence flip: {'YES ❌' if row['veto_valence_flip'] else 'NO ✅'})")
        print(f"  Protection benefit: {row['protection_benefit_absolute']:+.3f} ({row['relative_reduction_pct']:.1f}% reduction)")


if __name__ == '__main__':
    main()
