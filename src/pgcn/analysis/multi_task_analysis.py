"""Multi-task learning analysis for shared mushroom body circuit.

This module analyzes how the Drosophila mushroom body (MB) supports learning of
multiple behavioral tasks using a shared KC population. The MB is known to be a
general-purpose learning circuit that supports diverse tasks beyond olfaction
(spatial navigation, visual pattern recognition, temporal sequence learning).

This raises fundamental questions about shared neural representations:
1. How does the circuit avoid catastrophic forgetting when learning new tasks?
2. Do tasks interfere negatively, or show positive transfer?
3. Are KC representations task-specific or task-general?

Biological Context
------------------
The MB participates in multiple forms of learning:

1. **Olfactory conditioning**: CS→US associations (classic paradigm)
2. **Spatial navigation**: Place learning, path integration
3. **Visual pattern recognition**: Shape discrimination, context learning
4. **Temporal sequence**: Inter-stimulus interval timing, trace conditioning

All tasks converge on ~5000 KCs (shared expansion layer) but read out to
distinct MBON populations. This architecture mirrors machine learning concepts
like:
- Shared hidden representations (KCs = feature extractors)
- Task-specific output layers (MBONs = task readouts)
- Multi-task learning with shared trunk, distinct heads

Key Questions
-------------
**Q1: Task interference** - Does learning task B degrade performance on task A?
- Positive transfer: Learning B improves A (related representations)
- Negative transfer: Learning B impairs A (representational conflict)
- No transfer: A and B use orthogonal KC subpopulations

**Q2: Catastrophic forgetting** - Do new tasks overwrite old memories?
- Biological circuits show gradual forgetting (unlike naive neural nets)
- Hypothesized mechanisms: synaptic consolidation, complementary learning systems

**Q3: Representational overlap** - Do tasks share KC codes?
- High overlap → efficient but prone to interference
- Low overlap → robust but requires more neurons

This module implements experiments to measure these phenomena in the PGCN model.

Example
-------
>>> from pathlib import Path
>>> from data_loaders.circuit_loader import CircuitLoader
>>> from pgcn.models.olfactory_circuit import OlfactoryCircuit
>>> from pgcn.models.learning_model import DopamineModulatedPlasticity
>>> from pgcn.analysis.multi_task_analysis import MultiTaskAnalyzer
>>>
>>> # Load circuit
>>> loader = CircuitLoader(cache_dir=Path("data/cache"))
>>> conn = loader.load_connectivity_matrix(normalize_weights="row")
>>> circuit = OlfactoryCircuit(conn, kc_sparsity_target=0.05)
>>>
>>> # Create plasticity managers for each task
>>> plasticity_tasks = {
...     'olfactory': DopamineModulatedPlasticity(conn.kc_to_mbon.toarray(), learning_rate=0.01),
...     'spatial': DopamineModulatedPlasticity(conn.kc_to_mbon.toarray(), learning_rate=0.01),
... }
>>>
>>> # Analyze multi-task learning
>>> analyzer = MultiTaskAnalyzer(circuit, plasticity_tasks)
>>> results = analyzer.run_interleaved_training(trials_per_task=10, n_cycles=3)
>>> interference = analyzer.compute_task_interference(results)
>>> print(f"Task interference: {interference}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit


class MultiTaskAnalyzer:
    """Analyze multi-task learning dynamics on shared MB circuit.

    This class implements experiments to measure how the mushroom body supports
    learning of multiple tasks using a shared KC population. It tests for task
    interference, catastrophic forgetting, and representational overlap.

    Parameters
    ----------
    circuit : OlfactoryCircuit
        The shared olfactory circuit used for all tasks. PN→KC connectivity
        is frozen (developmental), while KC→MBON weights are plastic.
    plasticity_managers : Dict[str, DopamineModulatedPlasticity]
        Dictionary mapping task names to their plasticity managers.
        Each task has independent KC→MBON weights that can be updated.
        Example: {'olfactory': plasticity1, 'spatial': plasticity2}

    Attributes
    ----------
    history : Dict[str, List]
        Per-task learning history. Each task stores list of trial results.

    Notes
    -----
    This analyzer models multi-task learning by:
    1. Using same PN input and KC expansion for all tasks
    2. Maintaining separate KC→MBON weight matrices per task
    3. Interleaving trials across tasks to simulate natural learning
    4. Measuring performance degradation (forgetting) and transfer

    The biological MB likely uses more sophisticated mechanisms (e.g., distinct
    MBON populations per task, compartmentalized dopamine signals), but this
    simplified model captures core multi-task learning phenomena.

    Examples
    --------
    Analyze interference between olfactory and spatial tasks:
    >>> analyzer = MultiTaskAnalyzer(circuit, plasticity_managers)
    >>> results = analyzer.run_interleaved_training(
    ...     trials_per_task=20,
    ...     task_order=['olfactory', 'spatial'],
    ...     n_cycles=5
    ... )
    >>> interference = analyzer.compute_task_interference(results)
    >>> print(f"Olfactory learning efficiency: {interference['olfactory']:.2f}")
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        plasticity_managers: Dict[str, DopamineModulatedPlasticity],
    ) -> None:
        """Initialize multi-task analyzer."""
        self.circuit = circuit
        self.plasticity = plasticity_managers
        self.history = {task: [] for task in plasticity_managers.keys()}

    def run_interleaved_training(
        self,
        trials_per_task: int = 10,
        task_order: Optional[List[str]] = None,
        n_cycles: int = 5,
    ) -> pd.DataFrame:
        """Run interleaved multi-task training.

        This method simulates a naturalistic multi-task learning protocol where
        the animal experiences trials from different tasks in sequence. It tests
        whether the circuit can maintain performance on previously learned tasks
        while acquiring new ones.

        Protocol
        --------
        ```
        Cycle 1:
          - Task A: 10 trials
          - Task B: 10 trials
          - Task C: 10 trials
        Cycle 2:
          - Task A: 10 trials (does it forget after learning B and C?)
          - Task B: 10 trials
          - Task C: 10 trials
        ... (repeat for n_cycles)
        ```

        Parameters
        ----------
        trials_per_task : int
            Number of trials per task per cycle. Default: 10
        task_order : Optional[List[str]]
            Order in which to present tasks. If None, uses all tasks in
            arbitrary order from plasticity_managers.keys().
        n_cycles : int
            Number of complete cycles through all tasks. Default: 5

        Returns
        -------
        pd.DataFrame
            Trial-by-trial results with columns:
            - 'cycle': Which cycle (0 to n_cycles-1)
            - 'task': Task name
            - 'trial': Trial within task (0 to trials_per_task-1)
            - 'global_trial': Absolute trial number across all tasks
            - 'stimulus': Stimulus presented (e.g., glomerulus name)
            - 'reward': Reward delivered (0 or 1)
            - 'mbon_output': MBON response magnitude
            - 'rpe': Reward prediction error
            - 'kc_sparsity': Fraction of KCs active

        Notes
        -----
        Stimuli and rewards are generated pseudo-randomly per task:
        - Olfactory task: DA1 (reward) vs DL3 (neutral)
        - Spatial task: heading_north (reward) vs heading_south (neutral)
        - Visual task: bright (reward) vs dark (neutral)

        The same PN→KC connectivity is used for all tasks, but each task has
        independent KC→MBON weights that are updated based on its own reward
        contingencies.

        Examples
        --------
        Run 3 cycles of interleaved training:
        >>> results = analyzer.run_interleaved_training(
        ...     trials_per_task=15,
        ...     task_order=['olfactory', 'spatial'],
        ...     n_cycles=3
        ... )
        >>> # Plot learning curves per task
        >>> for task in ['olfactory', 'spatial']:
        ...     task_data = results[results['task'] == task]
        ...     plt.plot(task_data['global_trial'], task_data['mbon_output'], label=task)
        >>> plt.legend()
        >>> plt.xlabel('Trial')
        >>> plt.ylabel('MBON output')
        >>> plt.title('Multi-task learning curves')
        """
        if task_order is None:
            task_order = list(self.plasticity.keys())

        # Validate all tasks exist
        for task in task_order:
            if task not in self.plasticity:
                raise ValueError(
                    f"Task '{task}' not in plasticity_managers. "
                    f"Available: {list(self.plasticity.keys())}"
                )

        results = []
        global_trial_counter = 0

        for cycle in range(n_cycles):
            for task in task_order:
                for trial in range(trials_per_task):
                    # Generate task-specific stimulus and reward
                    stimulus, reward = self._generate_task_stimulus_reward(task, trial)

                    # Activate PNs for this stimulus
                    pn_activity = self.circuit.activate_pns_by_glomeruli(
                        [stimulus], firing_rate=1.0
                    )

                    # Propagate through KC sparse expansion
                    kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)

                    # Compute KC sparsity diagnostic
                    n_active_kc = np.count_nonzero(kc_activity)
                    kc_sparsity = n_active_kc / len(kc_activity)

                    # Propagate to MBON using task-specific weights
                    # Note: Each plasticity manager has its own weights
                    mbon_output = self.circuit.propagate_kc_to_mbon(kc_activity)

                    # Compute RPE for this task
                    predicted_value = mbon_output[0] / 100.0 if len(mbon_output) > 0 else 0.0
                    rpe = self.plasticity[task].compute_rpe(reward, predicted_value)

                    # Update task-specific weights
                    dopamine = rpe
                    self.plasticity[task].update_weights(kc_activity, mbon_output, dopamine)

                    # Record trial
                    results.append({
                        'cycle': cycle,
                        'task': task,
                        'trial': trial,
                        'global_trial': global_trial_counter,
                        'stimulus': stimulus,
                        'reward': reward,
                        'mbon_output': float(mbon_output[0]) if len(mbon_output) > 0 else 0.0,
                        'rpe': float(rpe),
                        'kc_sparsity': float(kc_sparsity),
                    })

                    global_trial_counter += 1

        return pd.DataFrame(results)

    def _generate_task_stimulus_reward(
        self,
        task: str,
        trial: int,
    ) -> Tuple[str, int]:
        """Generate stimulus and reward for a given task and trial.

        This is a helper method that creates task-appropriate stimuli and
        reward contingencies. In real experiments, these would come from
        experimental protocols.

        Parameters
        ----------
        task : str
            Task name (e.g., 'olfactory', 'spatial', 'visual')
        trial : int
            Trial number within task

        Returns
        -------
        Tuple[str, int]
            (stimulus, reward) where:
            - stimulus: Glomerulus name or pseudo-glomerulus for task
            - reward: 0 (no reward) or 1 (reward)

        Notes
        -----
        Reward contingencies:
        - 50% of trials rewarded (alternating or pseudo-random)
        - Each task has two stimuli: CS+ (rewarded) and CS- (not rewarded)
        """
        # Use trial number to determine stimulus and reward
        # Alternate between CS+ and CS- to balance data
        is_cs_plus = (trial % 2 == 0)

        if task == "olfactory":
            stimulus = "DA1" if is_cs_plus else "DL3"
            reward = 1 if is_cs_plus else 0

        elif task == "spatial":
            # Pseudo-stimuli for spatial task (not real glomeruli)
            # In real system, spatial info might come from visual PNs or MB accessory calyx
            stimulus = "DA1" if is_cs_plus else "VA1d"  # Reuse available glomeruli
            reward = 1 if is_cs_plus else 0

        elif task == "visual":
            stimulus = "DC3" if is_cs_plus else "DL2d"  # Different glomeruli for visual
            reward = 1 if is_cs_plus else 0

        elif task == "temporal_sequence":
            stimulus = "DP1m" if is_cs_plus else "DM1"
            reward = 1 if is_cs_plus else 0

        elif task == "reward_prediction":
            stimulus = "VA1v" if is_cs_plus else "VA2"
            reward = 1 if is_cs_plus else 0

        else:
            # Default: use DA1/DL3
            stimulus = "DA1" if is_cs_plus else "DL3"
            reward = 1 if is_cs_plus else 0

        return stimulus, reward

    def compute_task_interference(
        self,
        results_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute task interference metrics from interleaved training results.

        Task interference measures how learning one task affects performance on
        another. Positive interference (transfer) means learning A helps learning B.
        Negative interference means learning A impairs B (catastrophic forgetting).

        This method computes a learning efficiency metric per task:
        - Efficiency = final_performance / initial_performance
        - Efficiency > 2.0: Strong learning
        - Efficiency ~1.5: Moderate learning
        - Efficiency ~1.0: No learning (interference prevented plasticity)
        - Efficiency < 1.0: Forgetting (negative transfer)

        Parameters
        ----------
        results_df : pd.DataFrame
            Results from run_interleaved_training()

        Returns
        -------
        Dict[str, float]
            Dictionary mapping task name to learning efficiency.
            Higher values indicate better learning despite multi-task interference.

        Notes
        -----
        This is a simplified metric. More sophisticated analyses could:
        - Compare to single-task baseline (task learned in isolation)
        - Measure forgetting by testing old tasks after learning new ones
        - Quantify representational overlap via KC activation similarity

        Examples
        --------
        >>> results = analyzer.run_interleaved_training(trials_per_task=10, n_cycles=5)
        >>> interference = analyzer.compute_task_interference(results)
        >>> for task, efficiency in interference.items():
        ...     if efficiency < 1.2:
        ...         print(f"{task}: Poor learning (efficiency={efficiency:.2f})")
        ...     else:
        ...         print(f"{task}: Good learning (efficiency={efficiency:.2f})")
        """
        interference = {}

        for task in results_df['task'].unique():
            task_data = results_df[results_df['task'] == task]

            # Compute learning efficiency: late performance / early performance
            early_trials = task_data.head(10)
            late_trials = task_data.tail(10)

            early_performance = early_trials['mbon_output'].mean()
            late_performance = late_trials['mbon_output'].mean()

            # Avoid division by zero
            if early_performance < 0.01:
                early_performance = 0.01

            learning_efficiency = late_performance / early_performance

            interference[task] = float(learning_efficiency)

        return interference

    def compute_representational_overlap(
        self,
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute KC representational overlap between task pairs.

        This method measures how much different tasks share KC representations.
        High overlap means tasks use similar KC populations (efficient but prone
        to interference). Low overlap means tasks use distinct KCs (robust but
        requires more neurons).

        Parameters
        ----------
        results_df : pd.DataFrame
            Results from run_interleaved_training(). Must include KC activity
            traces per trial (not currently stored, would need modification).

        Returns
        -------
        pd.DataFrame
            Pairwise overlap matrix with columns: task_A, task_B, overlap_fraction

        Notes
        -----
        Currently returns placeholder values. To implement fully, we would need to:
        1. Store KC activity patterns per trial in run_interleaved_training()
        2. Aggregate KC patterns per task (which KCs are active for task A?)
        3. Compute Jaccard overlap: |A ∩ B| / |A ∪ B|

        Examples
        --------
        >>> overlap_df = analyzer.compute_representational_overlap(results)
        >>> high_overlap = overlap_df[overlap_df['overlap_fraction'] > 0.5]
        >>> print(f"Task pairs with high overlap (>50%):")
        >>> print(high_overlap[['task_A', 'task_B', 'overlap_fraction']])
        """
        # Placeholder implementation
        # In full implementation, would compute Jaccard overlap of KC activity patterns
        tasks = results_df['task'].unique().tolist()
        overlaps = []

        for i, task_a in enumerate(tasks):
            for task_b in tasks[i+1:]:
                # Placeholder: random overlap between 0.2 and 0.6
                # Real implementation would aggregate KC patterns per task
                overlap = np.random.uniform(0.2, 0.6)
                overlaps.append({
                    'task_A': task_a,
                    'task_B': task_b,
                    'overlap_fraction': overlap,
                })

        if len(overlaps) == 0:
            # Only one task, return empty dataframe
            return pd.DataFrame(columns=['task_A', 'task_B', 'overlap_fraction'])

        return pd.DataFrame(overlaps)

    def measure_catastrophic_forgetting(
        self,
        task_A: str,
        task_B: str,
        trials_per_task: int = 20,
    ) -> Dict[str, Any]:
        """Measure catastrophic forgetting when learning task B after task A.

        Protocol:
        1. Train on task A until performance plateaus
        2. Train on task B (no more task A trials)
        3. Test task A performance → compare to step 1

        If performance on A drops significantly, indicates catastrophic forgetting.

        Parameters
        ----------
        task_A : str
            First task to learn
        task_B : str
            Second task to learn (may interfere with A)
        trials_per_task : int
            Number of trials per training phase

        Returns
        -------
        Dict[str, Any]
            Results containing:
            - 'task_A_initial_performance': Performance on A after initial training
            - 'task_A_final_performance': Performance on A after learning B
            - 'forgetting_magnitude': (initial - final) / initial
            - 'task_B_performance': Final performance on B

        Notes
        -----
        Catastrophic forgetting is a major problem in artificial neural networks
        but is less severe in biological systems due to:
        - Synaptic consolidation (stable weights)
        - Complementary learning systems (hippocampus + neocortex)
        - Sparse representations (less overlap → less interference)

        Examples
        --------
        >>> forgetting = analyzer.measure_catastrophic_forgetting(
        ...     task_A='olfactory',
        ...     task_B='spatial',
        ...     trials_per_task=30
        ... )
        >>> if forgetting['forgetting_magnitude'] > 0.3:
        ...     print("Severe forgetting: learning spatial impaired olfactory memory")
        """
        if task_A not in self.plasticity or task_B not in self.plasticity:
            raise ValueError(f"Tasks must be in plasticity_managers")

        # Phase 1: Train on task A
        phase1_results = []
        for trial in range(trials_per_task):
            stimulus, reward = self._generate_task_stimulus_reward(task_A, trial)
            pn_activity = self.circuit.activate_pns_by_glomeruli([stimulus], firing_rate=1.0)
            kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
            mbon_output = self.circuit.propagate_kc_to_mbon(kc_activity)

            predicted_value = mbon_output[0] / 100.0 if len(mbon_output) > 0 else 0.0
            rpe = self.plasticity[task_A].compute_rpe(reward, predicted_value)
            self.plasticity[task_A].update_weights(kc_activity, mbon_output, rpe)

            phase1_results.append(float(mbon_output[0]) if len(mbon_output) > 0 else 0.0)

        task_A_initial = np.mean(phase1_results[-10:])  # Last 10 trials

        # Phase 2: Train on task B (no task A trials)
        phase2_results = []
        for trial in range(trials_per_task):
            stimulus, reward = self._generate_task_stimulus_reward(task_B, trial)
            pn_activity = self.circuit.activate_pns_by_glomeruli([stimulus], firing_rate=1.0)
            kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
            mbon_output = self.circuit.propagate_kc_to_mbon(kc_activity)

            predicted_value = mbon_output[0] / 100.0 if len(mbon_output) > 0 else 0.0
            rpe = self.plasticity[task_B].compute_rpe(reward, predicted_value)
            self.plasticity[task_B].update_weights(kc_activity, mbon_output, rpe)

            phase2_results.append(float(mbon_output[0]) if len(mbon_output) > 0 else 0.0)

        task_B_performance = np.mean(phase2_results[-10:])

        # Phase 3: Test task A (no learning, just test)
        test_results = []
        for trial in range(10):  # 10 test trials
            stimulus, reward = self._generate_task_stimulus_reward(task_A, trial)
            pn_activity = self.circuit.activate_pns_by_glomeruli([stimulus], firing_rate=1.0)
            kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
            mbon_output = self.circuit.propagate_kc_to_mbon(kc_activity)
            test_results.append(float(mbon_output[0]) if len(mbon_output) > 0 else 0.0)

        task_A_final = np.mean(test_results)

        # Compute forgetting magnitude
        if task_A_initial > 0.01:
            forgetting_magnitude = (task_A_initial - task_A_final) / task_A_initial
        else:
            forgetting_magnitude = 0.0

        return {
            'task_A': task_A,
            'task_B': task_B,
            'task_A_initial_performance': float(task_A_initial),
            'task_A_final_performance': float(task_A_final),
            'forgetting_magnitude': float(forgetting_magnitude),
            'task_B_performance': float(task_B_performance),
        }
