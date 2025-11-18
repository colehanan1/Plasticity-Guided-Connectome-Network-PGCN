"""Experiment 8: Or7a Ablation Study - Causal Proof of Veto Necessity.

This experiment tests whether Or7a veto is causally necessary for preventing
catastrophic forgetting by comparing continual learning WITH vs WITHOUT Or7a.

Biological Context
------------------
**Or7a Silencing Experiment**:
In Drosophila, genetically silencing Or7a neurons rescues benzaldehyde learning
that was previously blocked. This proves Or7a is causally necessary for blocking.

ML Translation:
- WITH Or7a: Veto gates plasticity → minimal forgetting (~5%)
- WITHOUT Or7a: No veto gating → catastrophic forgetting (~40%)
- Causal proof: Removing veto restores forgetting

Experimental Protocol
---------------------
**Condition 1: WITH Or7a Veto**
- Run Experiment 7 protocol with veto active
- Expected Task A forgetting: ~5%

**Condition 2: WITHOUT Or7a Veto (Ablation)**
- Run Experiment 7 protocol with veto disabled
- Expected Task A forgetting: ~40% (catastrophic)

**Comparison**:
- Or7a contribution = Forgetting_without - Forgetting_with
- Expected contribution: ~35% (massive effect)

Example
-------
>>> from pgcn.experiments.experiment_8_or7a_ablation_study import Or7aAblationStudy
>>>
>>> # Initialize ablation experiment
>>> exp = Or7aAblationStudy(
...     circuit=circuit,
...     plasticity_with=plasticity_copy1,
...     plasticity_without=plasticity_copy2,
...     or7a_glomerulus="DL5",
...     task_a_glomerulus="DA1",
...     task_b_glomerulus="DL3",
... )
>>>
>>> # Run both conditions
>>> results = exp.run_full_experiment(
...     n_task_a_trials=25,
...     n_task_b_trials=25
... )
>>>
>>> # Analyze causal contribution
>>> contribution = results['or7a_contribution']
>>> print(f"Or7a causal contribution: {contribution:.2%}")
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from pgcn.models.or7a_veto_gate import Or7aVetoGate
from pgcn.experiments.experiment_7_or7a_gated_continual_learning import Or7aGatedContinualLearning


class Or7aAblationStudy:
    """Or7a ablation study comparing WITH vs WITHOUT veto.

    This experiment proves that Or7a veto is causally necessary for preventing
    catastrophic forgetting by directly comparing continual learning performance
    in two conditions:
    1. WITH Or7a veto (normal condition)
    2. WITHOUT Or7a veto (ablation condition)

    Biological Rationale
    --------------------
    **Why ablation?**

    Ablation experiments are the gold standard for causal inference in neuroscience.
    If removing component X reverses effect Y, then X is causally necessary for Y.

    In Or7a system:
    - Normal: Or7a active → benzaldehyde learning blocked
    - Ablation: Or7a silenced → benzaldehyde learning restored
    - Conclusion: Or7a causally implements blocking

    ML translation:
    - Normal: Veto active → forgetting prevented (~5%)
    - Ablation: Veto disabled → catastrophic forgetting (~40%)
    - Conclusion: Veto causally implements forgetting prevention

    Parameters
    ----------
    circuit : OlfactoryCircuit
        Feedforward circuit (shared between conditions).
    plasticity_with : DopamineModulatedPlasticity
        Plasticity manager for WITH Or7a condition.
    plasticity_without : DopamineModulatedPlasticity
        Plasticity manager for WITHOUT Or7a condition (independent copy).
    or7a_glomerulus : str
        Or7a glomerulus for veto (e.g., "DL5").
    task_a_glomerulus : str
        Task A glomerulus (e.g., "DA1").
    task_b_glomerulus : str
        Task B glomerulus (e.g., "DL3").
    veto_strength : float, optional
        Veto gating strength for WITH condition. Default: 0.8

    Attributes
    ----------
    circuit : OlfactoryCircuit
        Shared circuit.
    exp_with_or7a : Or7aGatedContinualLearning
        Experiment instance WITH Or7a veto.
    exp_without_or7a : Or7aGatedContinualLearning
        Experiment instance WITHOUT Or7a veto (ablation).
    results : Dict[str, Any]
        Comparison results.

    Notes
    -----
    **Critical design**: We use separate plasticity instances (deep copies) to
    ensure the two conditions are independent. Otherwise, Task A training in
    Condition 1 would affect Condition 2.

    Example
    -------
    >>> # Create independent plasticity copies
    >>> weights = circuit.connectivity.kc_to_mbon.toarray()
    >>> plasticity_1 = DopamineModulatedPlasticity(weights.copy(), learning_rate=0.001)
    >>> plasticity_2 = DopamineModulatedPlasticity(weights.copy(), learning_rate=0.001)
    >>>
    >>> # Run ablation study
    >>> exp = Or7aAblationStudy(
    ...     circuit=circuit,
    ...     plasticity_with=plasticity_1,
    ...     plasticity_without=plasticity_2,
    ...     or7a_glomerulus="DL5",
    ...     task_a_glomerulus="DA1",
    ...     task_b_glomerulus="DL3",
    ... )
    >>> results = exp.run_full_experiment()
    >>>
    >>> # Causal inference
    >>> if results['or7a_contribution'] > 0.30:
    ...     print("✅ Or7a is causally necessary for forgetting prevention!")
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        plasticity_with: DopamineModulatedPlasticity,
        plasticity_without: DopamineModulatedPlasticity,
        or7a_glomerulus: str,
        task_a_glomerulus: str,
        task_b_glomerulus: str,
        veto_strength: float = 0.8,
    ) -> None:
        """Initialize Or7a ablation study."""
        self.circuit = circuit
        self.or7a_glomerulus = or7a_glomerulus
        self.task_a_glomerulus = task_a_glomerulus
        self.task_b_glomerulus = task_b_glomerulus
        self.veto_strength = veto_strength

        # Condition 1: WITH Or7a veto
        veto_gate_with = Or7aVetoGate(
            circuit=circuit,
            or7a_glomerulus=or7a_glomerulus,
            activation_threshold=0.3,
            veto_strength=1.0,  # Full veto strength
            graded=True,
        )

        self.exp_with_or7a = Or7aGatedContinualLearning(
            circuit=circuit,
            plasticity=plasticity_with,
            veto_gate=veto_gate_with,
            task_a_glomerulus=task_a_glomerulus,
            task_b_glomerulus=task_b_glomerulus,
            veto_strength=veto_strength,
        )

        # Condition 2: WITHOUT Or7a veto (ablation)
        # Create veto gate but with zero strength (effectively disabled)
        veto_gate_without = Or7aVetoGate(
            circuit=circuit,
            or7a_glomerulus=or7a_glomerulus,
            activation_threshold=0.3,
            veto_strength=0.0,  # No veto strength (ablation)
            graded=True,
        )

        self.exp_without_or7a = Or7aGatedContinualLearning(
            circuit=circuit,
            plasticity=plasticity_without,
            veto_gate=veto_gate_without,
            task_a_glomerulus=task_a_glomerulus,
            task_b_glomerulus=task_b_glomerulus,
            veto_strength=0.0,  # Ablation: no veto gating
        )

        self.results: Dict[str, Any] = {}

    def run_full_experiment(
        self,
        n_task_a_trials: int = 25,
        n_task_b_trials: int = 25,
    ) -> Dict[str, Any]:
        """Run ablation experiment with both conditions.

        Protocol
        --------
        1. Run Condition 1: WITH Or7a veto
        2. Run Condition 2: WITHOUT Or7a veto (ablation)
        3. Compare forgetting between conditions
        4. Compute Or7a causal contribution

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
            - with_or7a: Dict (Condition 1 results)
            - without_or7a: Dict (Condition 2 results)
            - or7a_contribution: float (forgetting reduction due to Or7a)
            - causal_proof: str (interpretation)

        Example
        -------
        >>> results = exp.run_full_experiment(
        ...     n_task_a_trials=25,
        ...     n_task_b_trials=25
        ... )
        >>>
        >>> print(f"WITH Or7a forgetting: {results['with_or7a']['forgetting_index']:.2%}")
        >>> print(f"WITHOUT Or7a forgetting: {results['without_or7a']['forgetting_index']:.2%}")
        >>> print(f"Or7a contribution: {results['or7a_contribution']:.2%}")
        """
        print("=" * 80)
        print("EXPERIMENT 8: Or7a Ablation Study")
        print("=" * 80)
        print("\nThis experiment tests whether Or7a veto is CAUSALLY NECESSARY")
        print("for preventing catastrophic forgetting.")
        print("\nProtocol:")
        print("  Condition 1: WITH Or7a veto (normal)")
        print("  Condition 2: WITHOUT Or7a veto (ablation)")
        print("  Compare: Catastrophic forgetting in both conditions")
        print("=" * 80)

        # Condition 1: WITH Or7a veto
        print("\n" + "=" * 80)
        print("[Condition 1] WITH Or7a Veto (Normal)")
        print("=" * 80)

        results_with = self.exp_with_or7a.run_full_experiment(
            n_task_a_trials=n_task_a_trials,
            n_task_b_trials=n_task_b_trials,
        )

        # Condition 2: WITHOUT Or7a veto (ablation)
        print("\n" + "=" * 80)
        print("[Condition 2] WITHOUT Or7a Veto (Ablation)")
        print("=" * 80)

        results_without = self.exp_without_or7a.run_full_experiment(
            n_task_a_trials=n_task_a_trials,
            n_task_b_trials=n_task_b_trials,
        )

        # Compute Or7a causal contribution
        forgetting_with = results_with["forgetting_index"]
        forgetting_without = results_without["forgetting_index"]
        or7a_contribution = forgetting_without - forgetting_with

        # Causal interpretation
        if or7a_contribution > 0.30:
            causal_proof = "✅ STRONG CAUSAL EVIDENCE: Or7a is necessary for forgetting prevention"
        elif or7a_contribution > 0.15:
            causal_proof = "⚠️  MODERATE CAUSAL EVIDENCE: Or7a contributes to forgetting prevention"
        elif or7a_contribution > 0:
            causal_proof = "⚠️  WEAK CAUSAL EVIDENCE: Or7a has minimal effect"
        else:
            causal_proof = "❌ NO CAUSAL EVIDENCE: Or7a ablation did not affect forgetting"

        # Store results
        self.results = {
            "with_or7a": results_with,
            "without_or7a": results_without,
            "or7a_contribution": or7a_contribution,
            "causal_proof": causal_proof,
        }

        # Print comparison
        print("\n" + "=" * 80)
        print("ABLATION COMPARISON")
        print("=" * 80)
        print("\nCondition 1: WITH Or7a Veto")
        print(f"  Task A Initial:  {results_with['task_a_initial_response']:.3f}")
        print(f"  Task A Final:    {results_with['task_a_final_response']:.3f}")
        print(f"  Task B Final:    {results_with['task_b_final_response']:.3f}")
        print(f"  Forgetting:      {forgetting_with:.2%}")

        print("\nCondition 2: WITHOUT Or7a Veto (Ablation)")
        print(f"  Task A Initial:  {results_without['task_a_initial_response']:.3f}")
        print(f"  Task A Final:    {results_without['task_a_final_response']:.3f}")
        print(f"  Task B Final:    {results_without['task_b_final_response']:.3f}")
        print(f"  Forgetting:      {forgetting_without:.2%}")

        print("\n" + "-" * 80)
        print("CAUSAL INFERENCE")
        print("-" * 80)
        print(f"Or7a Causal Contribution: {or7a_contribution:+.2%}")
        print(f"  (Forgetting reduction due to Or7a veto)")
        print(f"\nInterpretation: {causal_proof}")
        print("=" * 80)

        return self.results

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Create comparison DataFrame for analysis.

        Returns
        -------
        pd.DataFrame
            Comparison table with columns:
            - condition, task_a_initial, task_a_final, task_b_final, forgetting

        Example
        -------
        >>> df = exp.get_comparison_dataframe()
        >>> print(df)
                condition  task_a_initial  task_a_final  task_b_final  forgetting
        0    WITH Or7a            12.5          11.8          14.2        0.056
        1  WITHOUT Or7a           12.3           7.4          14.5        0.398
        """
        if not self.results:
            raise ValueError("Run experiment first with run_full_experiment()")

        with_data = self.results["with_or7a"]
        without_data = self.results["without_or7a"]

        comparison = pd.DataFrame([
            {
                "condition": "WITH Or7a",
                "task_a_initial": with_data["task_a_initial_response"],
                "task_a_final": with_data["task_a_final_response"],
                "task_b_final": with_data["task_b_final_response"],
                "forgetting": with_data["forgetting_index"],
            },
            {
                "condition": "WITHOUT Or7a",
                "task_a_initial": without_data["task_a_initial_response"],
                "task_a_final": without_data["task_a_final_response"],
                "task_b_final": without_data["task_b_final_response"],
                "forgetting": without_data["forgetting_index"],
            },
        ])

        return comparison

    def __repr__(self) -> str:
        """Return summary string."""
        return (
            f"Or7aAblationStudy(\n"
            f"  or7a_glomerulus='{self.or7a_glomerulus}'\n"
            f"  task_a_glomerulus='{self.task_a_glomerulus}'\n"
            f"  task_b_glomerulus='{self.task_b_glomerulus}'\n"
            f"  veto_strength={self.veto_strength}\n"
            f"  experiment_run={'Yes' if self.results else 'No'}\n"
            f")"
        )
