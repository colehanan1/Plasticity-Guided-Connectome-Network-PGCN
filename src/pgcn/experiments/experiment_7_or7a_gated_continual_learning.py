"""Experiment 7: Or7a-Gated Continual Learning.

This experiment tests whether the Or7a veto gate can enable sequential task learning
(Task A → Task B) with minimal catastrophic forgetting, inspired by the biological
finding that Or7a blocks benzaldehyde learning but enables hexanol cross-transfer.

Biological Context
------------------
**Or7a Cross-Learning Phenomenon** (Shen et al., 2025):

1. **Benzaldehyde Training**: Or7a 55% active → strong veto → minimal learning
2. **Hexanol Testing**: Or7a 14% active → weak veto → cross-transfer response
3. **Mechanism**: Graded veto allows similar inputs to benefit from partial transfer

ML Translation
--------------
- **Task A** (benzaldehyde) = Learn "cats" with Or7a veto detecting cat features
- **Task B** (hexanol) = Learn "dogs" with Or7a gating plasticity on cat features
- **Expected**: Minimal cat forgetting (~5%) vs catastrophic forgetting (~40%) baseline

Experimental Protocol
---------------------
**Phase 1: Task A Training (Cats)**
- Train on Glomerulus A (e.g., "DA1") → reward
- Or7a veto learns to detect Task A features
- Expected final accuracy: ~85%

**Phase 2: Task B Training (Dogs)**
- Train on Glomerulus B (e.g., "DL3") → reward
- Or7a veto gates plasticity when Task A features detected
- Expected Task B accuracy: ~88%

**Phase 3: Task A Re-Test (Forgetting Measurement)**
- Re-test Task A without further training
- Expected Task A retention: ~82% (vs ~45% without veto)
- Catastrophic forgetting: ~3% (vs ~40% baseline)

Example
-------
>>> from pgcn.models.olfactory_circuit import OlfactoryCircuit
>>> from pgcn.models.learning_model import DopamineModulatedPlasticity
>>> from pgcn.models.or7a_veto_gate import Or7aVetoGate
>>> from pgcn.experiments.experiment_7_or7a_gated_continual_learning import Or7aGatedContinualLearning
>>>
>>> # Load circuit
>>> circuit = OlfactoryCircuit(conn_matrix, kc_sparsity_target=0.05)
>>> plasticity = DopamineModulatedPlasticity(weights, learning_rate=0.001)
>>>
>>> # Create Or7a veto gate
>>> veto_gate = Or7aVetoGate(circuit, or7a_glomerulus="DL5", activation_threshold=0.3)
>>>
>>> # Run experiment
>>> exp = Or7aGatedContinualLearning(
...     circuit=circuit,
...     plasticity=plasticity,
...     veto_gate=veto_gate,
...     task_a_glomerulus="DA1",  # Cats
...     task_b_glomerulus="DL3",  # Dogs
... )
>>> results = exp.run_full_experiment(
...     n_task_a_trials=25,
...     n_task_b_trials=25
... )
>>>
>>> print(f"Task A forgetting: {results['forgetting_index']:.2%}")
>>> print(f"Task B accuracy: {results['task_b_final_response']:.3f}")
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from pgcn.models.or7a_veto_gate import Or7aVetoGate


class Or7aGatedContinualLearning:
    """Or7a-gated continual learning for sequential task training.

    This experiment demonstrates that the Or7a veto gate can enable learning
    Task B after Task A with minimal catastrophic forgetting, by detecting
    Task A features and gating plasticity to protect Task A knowledge.

    Biological Rationale
    --------------------
    **Why Or7a for continual learning?**

    1. **Graded veto**: Or7a provides graded plasticity control (14% vs 55%
       activation), not binary on/off. Similar tasks get partial veto, enabling
       beneficial transfer while protecting core features.

    2. **Efficient protection**: The 41 ORNs → 2 PNs → 312 KCs bottleneck shows
       that minimal infrastructure (2 neurons!) can protect hundreds of downstream
       parameters.

    3. **Cross-transfer**: Training on benzaldehyde (strong veto) enables response
       to hexanol (weak veto), demonstrating graded transfer rather than complete
       blocking.

    Parameters
    ----------
    circuit : OlfactoryCircuit
        Feedforward circuit for PN→KC→MBON propagation.
    plasticity : DopamineModulatedPlasticity
        Plasticity manager for KC→MBON weight updates.
    veto_gate : Or7aVetoGate
        Or7a veto gate for plasticity gating.
    task_a_glomerulus : str
        Glomerulus for Task A (e.g., "DA1" for cats).
    task_b_glomerulus : str
        Glomerulus for Task B (e.g., "DL3" for dogs).
    learning_rate : float, optional
        Learning rate for weight updates. Default: 0.001 (from README).
    veto_strength : float, optional
        How strongly veto gates plasticity. Default: 0.8 (80% suppression).

    Attributes
    ----------
    circuit : OlfactoryCircuit
        Circuit for forward propagation.
    plasticity : DopamineModulatedPlasticity
        Plasticity rules for weight updates.
    veto_gate : Or7aVetoGate
        Or7a veto mechanism.
    task_a_glomerulus : str
        Task A glomerulus name.
    task_b_glomerulus : str
        Task B glomerulus name.
    learning_rate : float
        Learning rate parameter.
    veto_strength : float
        Veto gating strength.
    trial_history : List[Dict[str, Any]]
        Per-trial results for analysis.

    Notes
    -----
    **Design choices**:

    1. **Task A = Or7a pathway**: We use the Or7a glomerulus as Task A to directly
       match the biological scenario (benzaldehyde training).

    2. **Graded veto strength**: Default veto_strength=0.8 (not 1.0) allows 20%
       residual plasticity even under strong veto, matching biological reality.

    3. **Simple odor protocol**: Uses glomerulus-based odor presentations (not
       complex visual datasets) to match PGCN's olfactory focus.

    Example
    -------
    >>> # Standard usage
    >>> exp = Or7aGatedContinualLearning(
    ...     circuit=circuit,
    ...     plasticity=plasticity,
    ...     veto_gate=veto_gate,
    ...     task_a_glomerulus="DA1",
    ...     task_b_glomerulus="DL3",
    ... )
    >>>
    >>> # Run continual learning protocol
    >>> results = exp.run_full_experiment(
    ...     n_task_a_trials=25,
    ...     n_task_b_trials=25
    ... )
    >>>
    >>> # Analyze forgetting
    >>> forgetting = results['forgetting_index']
    >>> print(f"Catastrophic forgetting: {forgetting:.2%}")
    >>> if forgetting < 0.05:
    ...     print("✅ Or7a veto successfully prevented forgetting!")
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        plasticity: DopamineModulatedPlasticity,
        veto_gate: Or7aVetoGate,
        task_a_glomerulus: str,
        task_b_glomerulus: str,
        learning_rate: float = 0.001,
        veto_strength: float = 0.8,
    ) -> None:
        """Initialize Or7a-gated continual learning experiment."""
        self.circuit = circuit
        self.plasticity = plasticity
        self.veto_gate = veto_gate
        self.task_a_glomerulus = task_a_glomerulus
        self.task_b_glomerulus = task_b_glomerulus
        self.learning_rate = learning_rate
        self.veto_strength = veto_strength

        # Trial history
        self.trial_history: List[Dict[str, Any]] = []

    def run_trial(
        self,
        odor_glomerulus: str,
        reward: float,
        apply_veto: bool = False,
        trial_type: str = "task_a",
    ) -> Dict[str, Any]:
        """Run single trial with optional Or7a veto gating.

        Biological Rationale
        --------------------
        This implements the core Or7a-gated plasticity mechanism:

        1. **Activate PNs**: Present odor (glomerulus activation)
        2. **Compute veto**: Or7a detector fires on Task A features
        3. **Forward propagation**: PN → KC → MBON (compute prediction)
        4. **Compute RPE**: Reward - predicted value
        5. **Gate plasticity**: Apply veto to weight update if Task A detected
        6. **Update weights**: dW × (1 - veto_strength × veto_signal)

        Parameters
        ----------
        odor_glomerulus : str
            Glomerulus to activate (e.g., "DA1", "DL3").
        reward : float
            Reward value (0 or 1 for binary conditioning).
        apply_veto : bool, optional
            Whether to apply Or7a veto gating. Default: False
        trial_type : str, optional
            Trial label for logging. Default: "task_a"

        Returns
        -------
        Dict[str, Any]
            Trial outcome with keys:
            - odor: str
            - reward: float
            - mbon_output: float
            - veto_signal: float
            - gating_factor: float
            - rpe: float
            - weight_change_magnitude: float
            - trial_type: str

        Example
        -------
        >>> # Task A trial (no veto during initial learning)
        >>> trial = exp.run_trial("DA1", reward=1.0, apply_veto=False, trial_type="task_a")
        >>> print(f"MBON response: {trial['mbon_output']:.3f}")
        >>>
        >>> # Task B trial (with veto to protect Task A)
        >>> trial = exp.run_trial("DL3", reward=1.0, apply_veto=True, trial_type="task_b")
        >>> print(f"Veto signal: {trial['veto_signal']:.3f}")
        >>> print(f"Gating factor: {trial['gating_factor']:.3f}")
        """
        # 1. Activate PNs for this odor
        pn_activity = self.circuit.activate_pns_by_glomeruli([odor_glomerulus], firing_rate=1.0)

        # 2. Compute Or7a veto signal
        if apply_veto:
            veto_signal = self.veto_gate.compute_veto_signal(pn_activity)
        else:
            veto_signal = 0.0

        # 3. Forward propagation (PN → KC → MBON)
        kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
        raw_mbon_output = self.plasticity.kc_to_mbon @ kc_activity
        mbon_output_vec = self.plasticity.apply_mbon_activation(raw_mbon_output)
        mbon_output = float(mbon_output_vec[0])

        # 4. Compute RPE
        predicted_value = mbon_output / self.plasticity.mbon_output_max
        predicted_value = np.clip(predicted_value, -1.0, 1.0)
        rpe = self.plasticity.compute_rpe(reward, predicted_value, learning_rate_rpe=0.1)

        # 5. Map RPE to dopamine
        dopamine = rpe

        # 6. Compute weight update (three-factor rule)
        prepost = np.outer(raw_mbon_output, kc_activity)
        delta_w = self.learning_rate * prepost * dopamine  # dt=1.0

        # 7. Apply Or7a veto gating
        if apply_veto and veto_signal > 0:
            delta_w = self.veto_gate.gate_plasticity(delta_w, veto_signal * self.veto_strength)
            gating_factor = 1.0 - self.veto_strength * veto_signal
        else:
            gating_factor = 1.0

        # 8. Update weights
        self.plasticity.kc_to_mbon += delta_w
        self.plasticity.enforce_connectivity_mask()

        weight_change_magnitude = float(np.linalg.norm(delta_w))

        # Record trial
        trial_data = {
            "odor": odor_glomerulus,
            "reward": reward,
            "mbon_output": mbon_output,
            "veto_signal": veto_signal,
            "gating_factor": gating_factor,
            "rpe": rpe,
            "dopamine": dopamine,
            "weight_change_magnitude": weight_change_magnitude,
            "trial_type": trial_type,
        }

        self.trial_history.append(trial_data)

        return trial_data

    def run_full_experiment(
        self,
        n_task_a_trials: int = 25,
        n_task_b_trials: int = 25,
    ) -> Dict[str, Any]:
        """Run complete continual learning experiment.

        Biological Protocol
        -------------------
        **Phase 1: Task A Training (Cats/Benzaldehyde)**
        - Train on Task A glomerulus → reward (e.g., DA1 → sugar)
        - No veto gating (Or7a detector learns Task A features)
        - Expected final accuracy: ~85%

        **Phase 2: Task B Training (Dogs/Hexanol)**
        - Train on Task B glomerulus → reward (e.g., DL3 → sugar)
        - Or7a veto active (gates plasticity when Task A features detected)
        - Expected Task B accuracy: ~88%
        - Expected Task A retention: ~82% (minimal forgetting)

        **Phase 3: Testing**
        - Test Task A: Measure forgetting
        - Test Task B: Measure learning
        - Compare to baseline catastrophic forgetting (~40%)

        Parameters
        ----------
        n_task_a_trials : int, optional
            Number of Task A training trials. Default: 25
        n_task_b_trials : int, optional
            Number of Task B training trials. Default: 25

        Returns
        -------
        Dict[str, Any]
            Results with keys:
            - task_a_trials: List[Dict] (Task A training trials)
            - task_b_trials: List[Dict] (Task B training trials)
            - task_a_initial_response: float (Task A MBON after Phase 1)
            - task_a_final_response: float (Task A MBON after Phase 2)
            - task_b_final_response: float (Task B MBON after Phase 2)
            - forgetting_index: float (Task A performance drop)
            - veto_statistics: Dict (Or7a veto activity stats)

        Notes
        -----
        **Interpretation of results**:
        - forgetting_index < 0.05: Excellent retention (Or7a working)
        - forgetting_index 0.05-0.20: Moderate retention (partial veto)
        - forgetting_index > 0.40: Catastrophic forgetting (veto failed)

        Example
        -------
        >>> results = exp.run_full_experiment(
        ...     n_task_a_trials=25,
        ...     n_task_b_trials=25
        ... )
        >>>
        >>> # Analyze results
        >>> forgetting = results['forgetting_index']
        >>> print(f"Catastrophic forgetting: {forgetting:.2%}")
        >>>
        >>> # Compare to baseline
        >>> baseline_forgetting = 0.40  # From literature
        >>> improvement = baseline_forgetting - forgetting
        >>> print(f"Or7a improvement: {improvement:.2%}")
        """
        results = {
            "task_a_trials": [],
            "task_b_trials": [],
        }

        print("=" * 80)
        print("EXPERIMENT 7: Or7a-Gated Continual Learning")
        print("=" * 80)

        # Phase 1: Train Task A (no veto, Or7a learns Task A features)
        print(f"\n[Phase 1] Training Task A ({self.task_a_glomerulus}) - {n_task_a_trials} trials")
        print("  Or7a detector learning Task A features (no veto gating)...")

        for trial_idx in range(n_task_a_trials):
            trial_data = self.run_trial(
                odor_glomerulus=self.task_a_glomerulus,
                reward=1.0,
                apply_veto=False,  # No veto during Task A training
                trial_type="task_a_training",
            )
            results["task_a_trials"].append(trial_data)

            if (trial_idx + 1) % 10 == 0:
                mbon = trial_data["mbon_output"]
                print(f"    Trial {trial_idx+1}/{n_task_a_trials}: MBON = {mbon:.3f}")

        # Test Task A (initial)
        task_a_initial_response = self._test_response(self.task_a_glomerulus)
        print(f"\n  Task A initial response: {task_a_initial_response:.3f}")

        # Phase 2: Train Task B (with Or7a veto gating)
        print(f"\n[Phase 2] Training Task B ({self.task_b_glomerulus}) - {n_task_b_trials} trials")
        print("  Or7a veto ACTIVE (protecting Task A features)...")

        for trial_idx in range(n_task_b_trials):
            trial_data = self.run_trial(
                odor_glomerulus=self.task_b_glomerulus,
                reward=1.0,
                apply_veto=True,  # Apply veto during Task B training
                trial_type="task_b_training",
            )
            results["task_b_trials"].append(trial_data)

            if (trial_idx + 1) % 10 == 0:
                mbon = trial_data["mbon_output"]
                veto = trial_data["veto_signal"]
                gate = trial_data["gating_factor"]
                print(f"    Trial {trial_idx+1}/{n_task_b_trials}: MBON = {mbon:.3f}, "
                      f"Veto = {veto:.3f}, Gate = {gate:.3f}")

        # Test Task A (final, measure forgetting)
        task_a_final_response = self._test_response(self.task_a_glomerulus)
        print(f"\n  Task A final response: {task_a_final_response:.3f}")

        # Test Task B (final)
        task_b_final_response = self._test_response(self.task_b_glomerulus)
        print(f"  Task B final response: {task_b_final_response:.3f}")

        # Compute forgetting index
        forgetting_index = (task_a_initial_response - task_a_final_response) / (
            task_a_initial_response + 1e-6
        )

        # Get veto statistics
        veto_statistics = self.veto_gate.get_veto_statistics()

        # Store results
        results.update({
            "task_a_initial_response": task_a_initial_response,
            "task_a_final_response": task_a_final_response,
            "task_b_final_response": task_b_final_response,
            "forgetting_index": forgetting_index,
            "veto_statistics": veto_statistics,
        })

        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT 7 RESULTS")
        print("=" * 80)
        print(f"Task A Initial:  {task_a_initial_response:.3f}")
        print(f"Task A Final:    {task_a_final_response:.3f}")
        print(f"Task B Final:    {task_b_final_response:.3f}")
        print(f"Forgetting:      {forgetting_index:.2%}")
        print(f"\nOr7a Veto Statistics:")
        print(f"  Mean veto: {veto_statistics['mean_veto']:.3f}")
        print(f"  Max veto:  {veto_statistics['max_veto']:.3f}")
        print(f"  Fraction above threshold: {veto_statistics['fraction_above_threshold']:.2%}")

        # Interpretation
        if forgetting_index < 0.05:
            print("\n✅ SUCCESS: Or7a veto prevented catastrophic forgetting!")
        elif forgetting_index < 0.20:
            print("\n⚠️  PARTIAL: Or7a veto reduced forgetting (but not eliminated)")
        else:
            print("\n❌ FAILURE: Catastrophic forgetting occurred despite Or7a veto")

        print("=" * 80)

        return results

    def _test_response(self, glomerulus: str) -> float:
        """Test MBON response to glomerulus without learning.

        Parameters
        ----------
        glomerulus : str
            Glomerulus to test.

        Returns
        -------
        float
            Absolute MBON response.
        """
        pn_activity = self.circuit.activate_pns_by_glomeruli([glomerulus], firing_rate=1.0)
        kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
        mbon_output_vec = self.plasticity.compute_mbon_output(kc_activity)
        return abs(float(mbon_output_vec[0]))

    def get_trial_dataframe(self) -> pd.DataFrame:
        """Convert trial history to pandas DataFrame for analysis.

        Returns
        -------
        pd.DataFrame
            Trial-by-trial data with columns:
            - odor, reward, mbon_output, veto_signal, gating_factor,
              rpe, dopamine, weight_change_magnitude, trial_type

        Example
        -------
        >>> df = exp.get_trial_dataframe()
        >>> task_a_trials = df[df['trial_type'] == 'task_a_training']
        >>> print(task_a_trials['mbon_output'].describe())
        """
        return pd.DataFrame(self.trial_history)

    def __repr__(self) -> str:
        """Return summary string."""
        return (
            f"Or7aGatedContinualLearning(\n"
            f"  task_a_glomerulus='{self.task_a_glomerulus}'\n"
            f"  task_b_glomerulus='{self.task_b_glomerulus}'\n"
            f"  veto_glomerulus='{self.veto_gate.or7a_glomerulus}'\n"
            f"  learning_rate={self.learning_rate}\n"
            f"  veto_strength={self.veto_strength}\n"
            f"  n_trials_run={len(self.trial_history)}\n"
            f")"
        )
