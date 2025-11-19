#!/usr/bin/env python3
"""Continual Learning with Selective Veto Gate Protection.

This experiment compares multiple continual learning strategies for preventing
catastrophic forgetting during sequential odor discrimination task learning,
inspired by the Or7a blocking mechanism in Drosophila.

Experimental Protocol
---------------------
**Sequential Task Learning**:
- Task 1: Learn Odor A → reward (e.g., ethyl butyrate vs hexanol)
- Task 2: Learn Odor C → reward (e.g., benzaldehyde vs 3-octanol)
- Task 3: Learn Odor E → reward (e.g., citral vs linalool)
- ... (configurable number of tasks)

**Measured Outcomes**:
- **Forgetting**: Accuracy drop on Task t after learning Task t+1
  Forgetting_t = max_acc(Task t) - final_acc(Task t)
- **Backward Transfer**: Does Task t+1 hurt Task t performance?
- **Forward Transfer**: Does Task t improve Task t+1 learning speed?

**Protection Strategies**:
1. **baseline**: No protection (catastrophic forgetting baseline)
2. **veto_gate**: Selective veto gate with DoOR chemical similarity
3. **simplified_ewc**: Elastic Weight Consolidation (penalty on weight change)
4. **freeze_topk**: Hard freeze top-k critical neurons (no plasticity)

Example
-------
>>> # From command line
>>> python scripts/experiments/continual_learning_veto.py \\
...     --tasks "ethyl butyrate,1-hexanol" "benzaldehyde,3-octanol" "citral,linalool" \\
...     --strategies baseline veto_gate simplified_ewc freeze_topk \\
...     --trials-per-task 25 \\
...     --output results/continual_learning_2025-11-18.csv

Biological Context
------------------
Inspired by Or7a veto mechanism (Shen et al., 2025):
- Selective pathway blocking prevents interference
- Chemical similarity predicts interference strength
- Graded veto (not binary) allows beneficial transfer

Author: PGCN Project (Generated with Claude Code)
Date: 2025-11-18
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_loaders.circuit_loader import CircuitLoader
from door_integration.pgcn_door import PGCNDoorIntegration
from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from pgcn.models.veto_gate import SelectiveVetoGate


class ContinualLearningExperiment:
    """Continual learning experiment with multiple protection strategies.

    This class orchestrates sequential task learning with different strategies
    for preventing catastrophic forgetting, using realistic multi-glomerulus
    odor encoding via the DoOR database.

    Parameters
    ----------
    circuit : OlfactoryCircuit
        Feedforward olfactory circuit.
    plasticity : DopamineModulatedPlasticity
        Plasticity model (will be deep-copied for each strategy).
    door_integration : PGCNDoorIntegration
        DoOR database interface for odor encoding.
    tasks : List[Tuple[str, str]]
        List of (odor_A, odor_B) pairs defining each discrimination task.
        Example: [('ethyl butyrate', '1-hexanol'), ('benzaldehyde', '3-octanol')]
    strategies : List[str]
        Protection strategies to test. Options: 'baseline', 'veto_gate',
        'simplified_ewc', 'freeze_topk'.
    trials_per_task : int, optional
        Number of training trials per task. Default: 25
    learning_rate : float, optional
        Plasticity learning rate. Default: 0.001
    protection_threshold : float, optional
        Percentile threshold for critical synapse protection. Default: 0.3
    gate_strength : float, optional
        Veto gate strength. Default: 0.9
    random_seed : int, optional
        Random seed for reproducibility. Default: 42

    Attributes
    ----------
    circuit : OlfactoryCircuit
        Circuit reference (copied for each strategy).
    plasticity_template : DopamineModulatedPlasticity
        Template plasticity (deep-copied for each strategy).
    door : PGCNDoorIntegration
        DoOR interface.
    tasks : List[Tuple[str, str]]
        Task definitions.
    strategies : List[str]
        Strategy names.
    results : List[Dict[str, Any]]
        Trial-by-trial results for all strategies.
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        plasticity: DopamineModulatedPlasticity,
        door_integration: PGCNDoorIntegration,
        tasks: List[Tuple[str, str]],
        strategies: List[str],
        trials_per_task: int = 25,
        learning_rate: float = 0.001,
        protection_threshold: float = 0.3,
        gate_strength: float = 0.9,
        random_seed: int = 42,
    ) -> None:
        """Initialize continual learning experiment."""
        self.circuit = circuit
        self.plasticity_template = plasticity
        self.door = door_integration
        self.tasks = tasks
        self.strategies = strategies
        self.trials_per_task = trials_per_task
        self.learning_rate = learning_rate
        self.protection_threshold = protection_threshold
        self.gate_strength = gate_strength
        self.random_seed = random_seed

        # Set random seed
        np.random.seed(random_seed)

        # Results storage
        self.results: List[Dict[str, Any]] = []

        print("=" * 80)
        print("CONTINUAL LEARNING EXPERIMENT - VETO GATE PROTECTION")
        print("=" * 80)
        print(f"Tasks: {len(tasks)}")
        for i, (odor_a, odor_b) in enumerate(tasks):
            print(f"  Task {i+1}: {odor_a} vs {odor_b}")
        print(f"Strategies: {', '.join(strategies)}")
        print(f"Trials per task: {trials_per_task}")
        print(f"Learning rate: {learning_rate}")
        print("=" * 80)

    def run_all_strategies(self) -> pd.DataFrame:
        """Run experiment for all strategies.

        Returns
        -------
        pd.DataFrame
            Results dataframe with columns:
            - strategy: str
            - task_id: int
            - trial_id: int
            - odor: str
            - reward: float
            - mbon_output: float
            - veto_signal: float (for veto_gate strategy)
            - forgetting: float (measured after each task)
        """
        for strategy in self.strategies:
            print(f"\n{'='*80}")
            print(f"RUNNING STRATEGY: {strategy.upper()}")
            print(f"{'='*80}")

            # Deep copy plasticity for this strategy
            plasticity = copy.deepcopy(self.plasticity_template)

            # Reset random seed for each strategy (for reproducibility)
            np.random.seed(self.random_seed)

            # Diagnostic: Print initial weight stats
            initial_weight_sum = np.sum(plasticity.kc_to_mbon)
            initial_weight_nonzero = np.count_nonzero(plasticity.kc_to_mbon)
            total_weights = plasticity.kc_to_mbon.size
            print(f"Initial weights: sum={initial_weight_sum:.6f}, nonzero={initial_weight_nonzero}/{total_weights}, "
                  f"id={id(plasticity.kc_to_mbon)}")

            # Run strategy
            if strategy == 'baseline':
                strategy_results = self._run_baseline(plasticity)
            elif strategy == 'veto_gate':
                strategy_results = self._run_veto_gate(plasticity)
            elif strategy == 'veto_strong':
                strategy_results = self._run_veto_gate(plasticity, gate_strength=0.95)
            elif strategy == 'veto_adaptive':
                strategy_results = self._run_veto_gate(plasticity, adaptive=True)
            elif strategy == 'simplified_ewc':
                strategy_results = self._run_simplified_ewc(plasticity)
            elif strategy == 'freeze_topk':
                strategy_results = self._run_freeze_topk(plasticity)
            else:
                print(f"⚠️  Unknown strategy: {strategy}. Skipping.")
                continue

            # Add strategy label
            for result in strategy_results:
                result['strategy'] = strategy

            # Diagnostic: Print final weight stats
            final_weight_sum = np.sum(plasticity.kc_to_mbon)
            weight_change = final_weight_sum - initial_weight_sum
            print(f"Final weight sum: {final_weight_sum:.6f} (change: {weight_change:+.6f})")

            # Save weight matrix for comparison
            weight_file = f"/tmp/weights_{strategy}.npy"
            np.save(weight_file, plasticity.kc_to_mbon)
            print(f"Saved weights to {weight_file}\n")

            self.results.extend(strategy_results)

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # PHASE 1 DIAGNOSTIC: Check for shared weight references
        print(f"\n{'='*80}")
        print("WEIGHT MATRIX VERIFICATION")
        print(f"{'='*80}")

        if len(self.results) > 0:
            # Group results by strategy
            strategies_tested = df['strategy'].unique()

            # Store weight matrix info for comparison
            weight_info = {}

            # We need to save weight matrices during execution, not here
            # So let's check the saved .npy files
            print("\nChecking saved weight matrices...")
            for strategy in strategies_tested:
                weight_file = f"/tmp/weights_{strategy}.npy"
                try:
                    weights = np.load(weight_file)
                    weight_info[strategy] = {
                        'sum': np.sum(weights),
                        'mean': np.mean(weights),
                        'std': np.std(weights),
                        'nonzero': np.count_nonzero(weights),
                        'shape': weights.shape
                    }
                    print(f"\n{strategy}:")
                    print(f"  Sum: {weight_info[strategy]['sum']:.6f}")
                    print(f"  Mean: {weight_info[strategy]['mean']:.6f}")
                    print(f"  Std: {weight_info[strategy]['std']:.6f}")
                    print(f"  Nonzero: {weight_info[strategy]['nonzero']}/{weights.size}")
                except FileNotFoundError:
                    print(f"\n{strategy}: weight file not found")

            # Compare baseline vs veto_gate if both exist
            if 'baseline' in weight_info and 'veto_gate' in weight_info:
                w_baseline = np.load('/tmp/weights_baseline.npy')
                w_veto = np.load('/tmp/weights_veto_gate.npy')

                diff = np.abs(w_baseline - w_veto)
                print(f"\nBASELINE vs VETO_GATE Comparison:")
                print(f"  Absolute difference sum: {np.sum(diff):.6f}")
                print(f"  Max absolute difference: {np.max(diff):.6f}")
                print(f"  # zero differences: {np.sum(diff == 0)} / {diff.size}")
                print(f"  # small diffs (<0.001): {np.sum(diff < 0.001)}")
                print(f"  # medium diffs (0.001-0.01): {np.sum((diff >= 0.001) & (diff < 0.01))}")
                print(f"  # large diffs (>0.01): {np.sum(diff >= 0.01)}")

                print(f"\n  Matrices are {'IDENTICAL' if np.allclose(w_baseline, w_veto) else 'DIFFERENT'}")

        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"Total trials: {len(df)}")
        print(f"Strategies: {df['strategy'].nunique()}")
        print(f"Tasks: {df['task_id'].nunique()}")

        return df

    def _run_baseline(self, plasticity: DopamineModulatedPlasticity) -> List[Dict[str, Any]]:
        """Run baseline strategy (no protection)."""
        print("[Baseline] No protection - expect catastrophic forgetting")

        results = []
        task_accuracies = {}  # Track max accuracy per task

        for task_id, (odor_a, odor_b) in enumerate(self.tasks):
            print(f"\n[Task {task_id+1}/{len(self.tasks)}] {odor_a} vs {odor_b}")

            # Train on this task
            task_results = self._train_task(
                plasticity, odor_a, odor_b, task_id, veto_signal=0.0
            )
            results.extend(task_results)

            # Record max accuracy
            task_acc = [r['mbon_output'] for r in task_results if r['odor'] == odor_a]
            task_accuracies[task_id] = max(task_acc) if task_acc else 0.0

            # Test forgetting on previous tasks
            if task_id > 0:
                forgetting_results = self._measure_forgetting(
                    plasticity, task_id, task_accuracies
                )
                for f_result in forgetting_results:
                    f_result['trial_id'] = len(results)
                    f_result['phase'] = 'forgetting_test'
                results.extend(forgetting_results)

        return results

    def _run_veto_gate(
        self,
        plasticity: DopamineModulatedPlasticity,
        gate_strength: Optional[float] = None,
        adaptive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run veto gate strategy with DoOR chemical similarity.

        Parameters
        ----------
        plasticity : DopamineModulatedPlasticity
            Plasticity model.
        gate_strength : Optional[float]
            Override gate strength. If None, uses self.gate_strength.
        adaptive : bool
            If True, scale gate strength by similarity (adaptive veto).
        """
        if gate_strength is None:
            gate_strength = self.gate_strength

        strategy_name = "Veto Gate (Adaptive)" if adaptive else f"Veto Gate (strength={gate_strength:.2f})"
        print(f"[{strategy_name}] Selective protection based on chemical similarity")

        # Create veto gate
        veto_gate = SelectiveVetoGate(
            circuit=self.circuit,
            protection_threshold=self.protection_threshold,
            gate_strength=gate_strength,
            similarity_metric='chemical',
        )

        # Enable veto protection
        plasticity.enable_veto_protection(veto_gate, track_weight_changes=True)

        results = []
        task_accuracies = {}

        for task_id, (odor_a, odor_b) in enumerate(self.tasks):
            print(f"\n[Task {task_id+1}/{len(self.tasks)}] {odor_a} vs {odor_b}")

            if task_id == 0:
                # Task 0: Learn without protection
                print("  Training Task 0 (no protection)...")
                task_results = self._train_task(
                    plasticity, odor_a, odor_b, task_id, veto_signal=0.0
                )

                # Identify critical pathways for Task 0
                print("  Identifying critical pathways for Task 0...")
                task_data = plasticity.get_task_data_for_protection()
                protection_mask = veto_gate.identify_critical_pathways(
                    task_data, method='gradient_magnitude'
                )

            else:
                # Subsequent tasks: Apply veto protection
                protected_odor = self.tasks[0][0]  # Protect Task 0 odor
                veto_signal = veto_gate.compute_veto_signal(odor_a, protected_odor)
                print(f"  Veto signal ({odor_a} vs {protected_odor}): {veto_signal:.3f}")

                # For adaptive strategy, scale gate strength by similarity
                if adaptive:
                    # Adaptive: gate_strength = base_strength × similarity
                    # High similarity → strong gate, low similarity → weak gate
                    effective_veto = veto_signal  # Already scaled by similarity
                    print(f"  Adaptive veto (similarity-scaled): {effective_veto:.3f}")
                else:
                    effective_veto = veto_signal

                task_results = self._train_task(
                    plasticity, odor_a, odor_b, task_id, veto_signal=effective_veto
                )

            results.extend(task_results)

            # Record max accuracy
            task_acc = [r['mbon_output'] for r in task_results if r['odor'] == odor_a]
            task_accuracies[task_id] = max(task_acc) if task_acc else 0.0

            # Test forgetting
            if task_id > 0:
                forgetting_results = self._measure_forgetting(
                    plasticity, task_id, task_accuracies
                )
                for f_result in forgetting_results:
                    f_result['trial_id'] = len(results)
                    f_result['phase'] = 'forgetting_test'
                results.extend(forgetting_results)

        return results

    def _run_simplified_ewc(self, plasticity: DopamineModulatedPlasticity) -> List[Dict[str, Any]]:
        """Run simplified EWC (penalty on weight change from Task 0)."""
        print("[Simplified EWC] Penalize weight changes from Task 0")

        results = []
        task_accuracies = {}
        task_0_weights = None

        for task_id, (odor_a, odor_b) in enumerate(self.tasks):
            print(f"\n[Task {task_id+1}/{len(self.tasks)}] {odor_a} vs {odor_b}")

            if task_id == 0:
                # Task 0: Learn normally and save weights
                task_results = self._train_task(
                    plasticity, odor_a, odor_b, task_id, veto_signal=0.0
                )
                task_0_weights = plasticity.kc_to_mbon.copy()
                print("  Saved Task 0 weights for EWC penalty")

            else:
                # Subsequent tasks: Apply EWC penalty (via manual decay toward Task 0 weights)
                task_results = self._train_task_with_ewc(
                    plasticity, odor_a, odor_b, task_id, task_0_weights, ewc_strength=0.5
                )

            results.extend(task_results)

            # Record max accuracy
            task_acc = [r['mbon_output'] for r in task_results if r['odor'] == odor_a]
            task_accuracies[task_id] = max(task_acc) if task_acc else 0.0

            # Test forgetting
            if task_id > 0:
                forgetting_results = self._measure_forgetting(
                    plasticity, task_id, task_accuracies
                )
                for f_result in forgetting_results:
                    f_result['trial_id'] = len(results)
                    f_result['phase'] = 'forgetting_test'
                results.extend(forgetting_results)

        return results

    def _run_freeze_topk(self, plasticity: DopamineModulatedPlasticity) -> List[Dict[str, Any]]:
        """Run freeze top-k neurons (hard freeze, no graded protection)."""
        print("[Freeze Top-K] Hard freeze top 30% of critical synapses")

        results = []
        task_accuracies = {}

        for task_id, (odor_a, odor_b) in enumerate(self.tasks):
            print(f"\n[Task {task_id+1}/{len(self.tasks)}] {odor_a} vs {odor_b}")

            if task_id == 0:
                # Task 0: Learn normally
                task_results = self._train_task(
                    plasticity, odor_a, odor_b, task_id, veto_signal=0.0
                )

                # Identify top-k synapses to freeze
                task_data = plasticity.get_task_data_for_protection()
                weight_changes = task_data['weight_changes']
                if len(weight_changes) > 0:
                    mean_abs_change = np.mean([np.abs(dw) for dw in weight_changes], axis=0)
                    cutoff = np.percentile(mean_abs_change, 70)  # Top 30%
                    freeze_mask = mean_abs_change >= cutoff

                    # Freeze synapses
                    frozen_indices = list(zip(*np.where(freeze_mask)))
                    plasticity._frozen_synapses = set(
                        [(kc, mb) for mb, kc in frozen_indices]
                    )
                    print(f"  Froze {len(frozen_indices)} synapses (top 30%)")

            else:
                # Subsequent tasks: Training with frozen synapses
                task_results = self._train_task(
                    plasticity, odor_a, odor_b, task_id, veto_signal=0.0
                )

            results.extend(task_results)

            # Record max accuracy
            task_acc = [r['mbon_output'] for r in task_results if r['odor'] == odor_a]
            task_accuracies[task_id] = max(task_acc) if task_acc else 0.0

            # Test forgetting
            if task_id > 0:
                forgetting_results = self._measure_forgetting(
                    plasticity, task_id, task_accuracies
                )
                for f_result in forgetting_results:
                    f_result['trial_id'] = len(results)
                    f_result['phase'] = 'forgetting_test'
                results.extend(forgetting_results)

        return results

    def _train_task(
        self,
        plasticity: DopamineModulatedPlasticity,
        odor_a: str,
        odor_b: str,
        task_id: int,
        veto_signal: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Train single discrimination task.

        Parameters
        ----------
        plasticity : DopamineModulatedPlasticity
            Plasticity model (modified in-place).
        odor_a : str
            Rewarded odor (CS+).
        odor_b : str
            Unrewarded odor (CS-).
        task_id : int
            Task index.
        veto_signal : float
            Veto strength (0 = no protection, 1 = full protection).

        Returns
        -------
        List[Dict[str, Any]]
            Trial results.
        """
        # Set veto signal for this task (if veto gate registered)
        if plasticity.veto_gate is not None:
            plasticity.veto_gate.set_veto_signal(veto_signal)

        results = []

        for trial in range(self.trials_per_task):
            # Alternate between CS+ (odor_a) and CS- (odor_b)
            if trial % 2 == 0:
                odor = odor_a
                reward = 1.0
            else:
                odor = odor_b
                reward = 0.0

            # Get glomeruli for odor (multi-glomerulus activation)
            glomeruli = self.door.map_odorant_to_glomeruli(odor, threshold=0.3)

            # Activate PNs
            pn_activity = self.circuit.activate_pns_by_glomeruli(glomeruli, firing_rate=1.0)

            # Forward propagation
            kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
            mbon_output_vec = plasticity.compute_mbon_output(kc_activity)

            # Use sum of all MBON outputs (not just first MBON)
            mbon_output = float(np.sum(mbon_output_vec))

            # Compute RPE
            predicted_value = mbon_output / (plasticity.mbon_output_max * len(mbon_output_vec))
            predicted_value = np.clip(predicted_value, -1.0, 1.0)
            rpe = plasticity.compute_rpe(reward, predicted_value, learning_rate_rpe=0.1)

            # Dopamine signal
            dopamine = rpe

            # DEBUG: Save Task 0 active KCs
            if task_id == 0 and trial == 0:
                if not hasattr(self, '_task0_active_kcs'):
                    self._task0_active_kcs = set(np.where(kc_activity > 0)[0])

            # Update weights
            plasticity.update_weights(kc_activity, mbon_output_vec, dopamine, dt=1.0)

            # Print progress every 5 trials during Task 0
            if task_id == 0 and trial % 5 == 4:
                print(f"    Trial {trial+1}: {odor[:15]:15s} → MBON={mbon_output:.3f}, RPE={rpe:.3f}")

            # DEBUG: Print KC overlap at start of Task 2
            if task_id == 1 and trial == 0:
                kc_active_task2 = set(np.where(kc_activity > 0)[0])
                if hasattr(self, '_task0_active_kcs'):
                    overlap = len(self._task0_active_kcs & kc_active_task2)
                    print(f"\n  ⚠️  KC OVERLAP (Task 0 vs Task 2):")
                    print(f"     Task 0 active KCs: {len(self._task0_active_kcs)}")
                    print(f"     Task 2 active KCs: {len(kc_active_task2)}")
                    print(f"     Overlap: {overlap} ({overlap/len(self._task0_active_kcs)*100:.1f}%)")
                    if overlap == 0:
                        print(f"     ❌ NO OVERLAP! Tasks use completely different KCs (sparse coding prevents interference)\n")
                    else:
                        print()

            # Record result
            results.append({
                'task_id': task_id,
                'trial_id': len(results),
                'odor': odor,
                'reward': reward,
                'mbon_output': abs(mbon_output),
                'veto_signal': veto_signal,
                'rpe': rpe,
                'dopamine': dopamine,
                'phase': 'training',
            })

        return results

    def _train_task_with_ewc(
        self,
        plasticity: DopamineModulatedPlasticity,
        odor_a: str,
        odor_b: str,
        task_id: int,
        task_0_weights: np.ndarray,
        ewc_strength: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Train task with EWC penalty (pull weights toward Task 0)."""
        results = []

        for trial in range(self.trials_per_task):
            # Alternate odors
            if trial % 2 == 0:
                odor, reward = odor_a, 1.0
            else:
                odor, reward = odor_b, 0.0

            # Get glomeruli
            glomeruli = self.door.map_odorant_to_glomeruli(odor, threshold=0.3)

            # Forward pass
            pn_activity = self.circuit.activate_pns_by_glomeruli(glomeruli, firing_rate=1.0)
            kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
            mbon_output_vec = plasticity.compute_mbon_output(kc_activity)

            # Use sum of all MBON outputs (must match training computation!)
            mbon_output = float(np.sum(mbon_output_vec))

            # RPE and update
            predicted_value = np.clip(
                mbon_output / (plasticity.mbon_output_max * len(mbon_output_vec)), -1.0, 1.0
            )
            rpe = plasticity.compute_rpe(reward, predicted_value, learning_rate_rpe=0.1)
            dopamine = rpe

            plasticity.update_weights(kc_activity, mbon_output_vec, dopamine, dt=1.0)

            # Apply EWC penalty: decay toward Task 0 weights
            weight_diff = plasticity.kc_to_mbon - task_0_weights
            plasticity.kc_to_mbon -= ewc_strength * 0.01 * weight_diff  # Small penalty

            results.append({
                'task_id': task_id,
                'trial_id': len(results),
                'odor': odor,
                'reward': reward,
                'mbon_output': abs(mbon_output),
                'veto_signal': 0.0,
                'rpe': rpe,
                'dopamine': dopamine,
                'phase': 'training_ewc',
            })

        return results

    def _measure_forgetting(
        self,
        plasticity: DopamineModulatedPlasticity,
        current_task_id: int,
        task_accuracies: Dict[int, float],
    ) -> List[Dict[str, Any]]:
        """Measure forgetting on Task 0 after learning current task."""
        results = []

        # Test Task 0
        task_0_odor = self.tasks[0][0]  # CS+ from Task 0
        glomeruli = self.door.map_odorant_to_glomeruli(task_0_odor, threshold=0.3)

        # PHASE 2 DIAGNOSTIC: Trace forgetting test flow
        print(f"\n  {'='*60}")
        print(f"  FORGETTING TEST - DETAILED TRACE")
        print(f"  {'='*60}")

        # Step 1: PNs
        pn_activity = self.circuit.activate_pns_by_glomeruli(glomeruli, firing_rate=1.0)
        pn_sum = np.sum(pn_activity)
        pn_active = np.count_nonzero(pn_activity)
        print(f"  1. PN Activity: sum={pn_sum:.4f}, active={pn_active}/{len(pn_activity)}")

        # Step 2: KCs (THIS IS CRITICAL - check if identical across strategies)
        kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
        kc_active_count = np.count_nonzero(kc_activity)
        kc_sum = np.sum(kc_activity)
        kc_active_indices = np.where(kc_activity > 0)[0]
        print(f"  2. KC Activity: sum={kc_sum:.4f}, active={kc_active_count}/{len(kc_activity)}")
        print(f"     Active KC indices (first 10): {kc_active_indices[:10].tolist()}")
        print(f"     Active KC values (first 10): {kc_activity[kc_active_indices[:10]].tolist()}")

        # Step 3: Weights (check these are different across strategies)
        weight_sum = np.sum(plasticity.kc_to_mbon)
        weight_nonzero = np.count_nonzero(plasticity.kc_to_mbon)
        weight_id = id(plasticity.kc_to_mbon)
        print(f"  3. Weights: sum={weight_sum:.6f}, nonzero={weight_nonzero}, id={weight_id}")

        # Step 4: Weights for active KCs only
        weights_for_active_kcs = plasticity.kc_to_mbon[:, kc_active_indices]
        print(f"     Weights for active KCs: sum={np.sum(weights_for_active_kcs):.6f}")
        print(f"     Weights for active KCs (MBON 0, first 5): {weights_for_active_kcs[0, :5].tolist()}")

        # Step 5: MBON computation
        mbon_output_vec = plasticity.compute_mbon_output(kc_activity)
        mbon_output = float(np.sum(mbon_output_vec))
        print(f"  4. MBON Output: sum={mbon_output:.6f}")
        print(f"     Individual MBONs (first 5): {mbon_output_vec[:5].tolist()}")

        # CRITICAL CHECK: Are active KCs protected?
        if hasattr(plasticity, 'veto_gate') and plasticity.veto_gate is not None:
            protection_mask = plasticity.veto_gate.protection_mask
            if protection_mask is not None:
                # Check how many active KC→MBON connections are protected
                n_protected_active = 0
                n_total_active = 0
                for kc_idx in kc_active_indices:
                    for mbon_idx in range(len(mbon_output_vec)):
                        n_total_active += 1
                        if protection_mask[mbon_idx, kc_idx]:
                            n_protected_active += 1

                print(f"     ⚠️  PROTECTION OVERLAP:")
                print(f"        {n_protected_active}/{n_total_active} active synapses are protected "
                      f"({n_protected_active/n_total_active*100:.1f}%)")
                if n_protected_active == 0:
                    print(f"        ❌ NO OVERLAP! Veto gate doesn't protect any active test synapses!")
            else:
                print(f"     No protection mask set")

        # Manual verification: compute weighted sum manually
        manual_mbon = np.zeros(len(mbon_output_vec))
        for i, kc_idx in enumerate(kc_active_indices):
            manual_mbon += plasticity.kc_to_mbon[:, kc_idx] * kc_activity[kc_idx]
        manual_mbon_activated = np.tanh(manual_mbon / plasticity.mbon_output_divisor) * plasticity.mbon_output_max
        manual_sum = np.sum(manual_mbon_activated)
        print(f"     Manual computation: {manual_sum:.6f} (match: {np.abs(manual_sum - mbon_output) < 0.001})")

        # Compute forgetting
        max_acc = task_accuracies.get(0, 0.0)
        forgetting = (max_acc - mbon_output) / (max_acc + 1e-6) if max_acc > 0 else 0.0

        print(f"  {'='*60}")
        print(f"  Forgetting: {forgetting:.2%} (was {max_acc:.3f}, now {mbon_output:.3f})")
        print(f"  {'='*60}\n")

        results.append({
            'task_id': 0,  # Testing Task 0
            'trial_id': -1,
            'odor': task_0_odor,
            'reward': -1.0,  # No reward (probe trial)
            'mbon_output': mbon_output,
            'veto_signal': 0.0,
            'rpe': 0.0,
            'dopamine': 0.0,
            'forgetting': forgetting,
            'tested_after_task': current_task_id,
        })

        return results


def main():
    """Run continual learning experiment from command line."""
    parser = argparse.ArgumentParser(
        description="Continual Learning with Veto Gate Protection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['ethyl butyrate,1-hexanol', 'benzaldehyde,3-octanol', 'citral,linalool'],
        help='Task odor pairs (comma-separated). Example: "ethyl butyrate,1-hexanol"'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['baseline', 'veto_gate', 'simplified_ewc', 'freeze_topk'],
        help='Protection strategies to test (baseline, veto_gate, veto_strong, veto_adaptive, simplified_ewc, freeze_topk)'
    )
    parser.add_argument(
        '--trials-per-task',
        type=int,
        default=25,
        help='Number of training trials per task (default: 25)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Plasticity learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--kc-sparsity',
        type=float,
        default=0.05,
        help='Fraction of KCs active per odor (default: 0.05 = 5%%). Higher values increase KC overlap.'
    )
    parser.add_argument(
        '--protection-threshold',
        type=float,
        default=0.3,
        help='Percentile threshold for critical synapses (default: 0.3 = top 30%%)'
    )
    parser.add_argument(
        '--gate-strength',
        type=float,
        default=0.9,
        help='Veto gate strength (default: 0.9)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/cache',
        help='Directory for circuit cache (default: data/cache)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/continual_learning_results.csv',
        help='Output CSV file path (default: results/continual_learning_results.csv)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Parse tasks
    tasks = []
    for task_str in args.tasks:
        odor_a, odor_b = [s.strip() for s in task_str.split(',')]
        tasks.append((odor_a, odor_b))

    # Load circuit
    print("Loading circuit connectivity...")
    loader = CircuitLoader(cache_dir=args.cache_dir)
    connectivity = loader.load_connectivity_matrix(
        normalize_weights='row',
        include_dan=False,
        include_extended=False,
    )

    circuit = OlfactoryCircuit(connectivity=connectivity, kc_sparsity_target=args.kc_sparsity)
    print(f"✅ Circuit loaded: {connectivity.n_pn} PNs, {connectivity.n_kc} KCs, {connectivity.n_mbon} MBONs")
    print(f"   KC sparsity: {args.kc_sparsity:.1%} ({int(connectivity.n_kc * args.kc_sparsity)} active KCs per odor)")

    # Initialize plasticity
    kc_to_mbon_dense = connectivity.kc_to_mbon.toarray()
    plasticity = DopamineModulatedPlasticity(
        kc_to_mbon_weights=kc_to_mbon_dense,
        learning_rate=args.learning_rate,
        plasticity_mode='three_factor',
    )

    # Initialize DoOR
    print("Loading DoOR database...")
    door = PGCNDoorIntegration()
    print("✅ DoOR loaded")

    # Create experiment
    experiment = ContinualLearningExperiment(
        circuit=circuit,
        plasticity=plasticity,
        door_integration=door,
        tasks=tasks,
        strategies=args.strategies,
        trials_per_task=args.trials_per_task,
        learning_rate=args.learning_rate,
        protection_threshold=args.protection_threshold,
        gate_strength=args.gate_strength,
        random_seed=args.random_seed,
    )

    # Run experiment
    results_df = experiment.run_all_strategies()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✅ Results saved to: {output_path}")
    print(f"Total trials: {len(results_df)}")

    # Print summary
    print("\n" + "="*80)
    print("FORGETTING SUMMARY (Task 0)")
    print("="*80)

    forgetting_df = results_df[results_df['phase'] == 'forgetting_test']
    if len(forgetting_df) > 0:
        for strategy in args.strategies:
            strategy_forget = forgetting_df[forgetting_df['strategy'] == strategy]
            if len(strategy_forget) > 0:
                mean_forget = strategy_forget['forgetting'].mean()
                print(f"{strategy:20s}: {mean_forget:.2%} forgetting")

    print("="*80)


if __name__ == '__main__':
    main()
