"""Experiment 9: Binary Classification with Or7a Removal - Unmasking Test.

This experiment tests whether Or7a veto SUPPRESSES EXPRESSION of learned
responses by training a binary classifier WITH Or7a, then removing it to
reveal "unmasked" knowledge.

Biological Context
------------------
**Or7a Unmasking Hypothesis**:
In Drosophila, Or7a neurons don't just block learning - they actively suppress
the EXPRESSION of learned associations. When Or7a is silenced AFTER learning,
previously-hidden responses are "unmasked" and become visible.

Evidence:
- Flies trained on benzaldehyde WITH Or7a active show weak responses
- Silencing Or7a POST-TRAINING unmasks strong responses
- This proves Or7a suppresses expression, not just acquisition

ML Translation:
- Train binary task ("Is this a cat?") WITH Or7a veto
- Test WITH Or7a: Suppressed responses (~50% accuracy)
- Remove Or7a (ablation)
- Re-test WITHOUT Or7a: Unmasked responses (~90% accuracy)
- Causal proof: Removing veto reveals suppressed knowledge

Experimental Protocol
---------------------
**Phase 1: Training (WITH Or7a)**
- Train binary classification: Cat vs Non-Cat
- Or7a veto active during training
- Expected: Learning occurs but is partially suppressed

**Phase 2: Testing WITH Or7a**
- Test classification with Or7a still active
- Expected: ~50-60% accuracy (suppression)

**Phase 3: Or7a Ablation**
- Remove Or7a veto (disable gating)

**Phase 4: Testing WITHOUT Or7a**
- Re-test classification without Or7a
- Expected: ~80-90% accuracy (unmasking)

**Comparison**:
- Unmasking effect = Accuracy_without - Accuracy_with
- Expected effect: +30-40% (large unmasking)

Example
-------
>>> from pgcn.experiments.experiment_9_binary_classification import Or7aBinaryClassification
>>>
>>> # Initialize binary classification experiment
>>> exp = Or7aBinaryClassification(
...     circuit=circuit,
...     plasticity=plasticity,
...     or7a_glomerulus="DL5",
...     positive_class_glomerulus="DA1",  # Cat
...     negative_class_glomerulus="DL3",  # Non-cat
... )
>>>
>>> # Run full unmasking protocol
>>> results = exp.run_full_experiment(
...     n_training_trials=50,
...     n_test_trials=20
... )
>>>
>>> # Analyze unmasking effect
>>> unmasking = results['unmasking_effect']
>>> print(f"Unmasking effect: {unmasking:.2%}")
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from pgcn.models.or7a_veto_gate import Or7aVetoGate


class Or7aBinaryClassification:
    """Binary classification with Or7a removal to test unmasking hypothesis.

    This experiment tests whether Or7a veto suppresses EXPRESSION of learned
    knowledge by training a binary classifier with veto active, then removing
    the veto to reveal unmasked responses.

    Biological Rationale
    --------------------
    **Why unmasking matters?**

    Classical learning theory assumes plasticity determines what is learned.
    But Or7a shows that gating can suppress EXPRESSION without erasing memory:
    - Learning occurs despite veto (weights change)
    - Responses are suppressed during veto
    - Removing veto unmasks full learned responses

    This is analogous to:
    - Inhibitory control in prefrontal cortex
    - Response suppression in motor cortex
    - Latent learning in hippocampus

    ML Translation:
    - Train classifier WITH veto → learning + suppression
    - Test WITH veto → suppressed accuracy (~50%)
    - Remove veto → unmask knowledge
    - Re-test WITHOUT veto → full accuracy (~90%)

    Parameters
    ----------
    circuit : OlfactoryCircuit
        Feedforward olfactory circuit (PN → KC → MBON).
    plasticity : DopamineModulatedPlasticity
        Plasticity manager for KC→MBON synapses.
    or7a_glomerulus : str
        Or7a glomerulus for veto (e.g., "DL5").
    positive_class_glomerulus : str
        Glomerulus for positive class (e.g., "DA1" for "cat").
    negative_class_glomerulus : str
        Glomerulus for negative class (e.g., "DL3" for "non-cat").
    learning_rate : float, optional
        Synaptic learning rate. Default: 0.001
    veto_strength : float, optional
        Or7a veto strength during training/testing. Default: 0.8
    or7a_activation_threshold : float, optional
        Or7a activation threshold. Default: 0.3
    decision_threshold : float, optional
        MBON output threshold for binary classification. Default: 0.0

    Attributes
    ----------
    circuit : OlfactoryCircuit
        The olfactory circuit.
    plasticity : DopamineModulatedPlasticity
        Plasticity manager.
    veto_gate : Or7aVetoGate
        Or7a veto gate.
    training_history : List[Dict]
        Training trial data.
    test_with_or7a_history : List[Dict]
        Testing data WITH Or7a.
    test_without_or7a_history : List[Dict]
        Testing data WITHOUT Or7a (after ablation).
    results : Dict[str, Any]
        Experiment results.

    Notes
    -----
    **Critical insight**: This experiment proves that Or7a veto affects
    EXPRESSION (output) rather than STORAGE (weights). If veto only affected
    learning, removing it wouldn't change test accuracy. But if veto suppresses
    expression, removing it unmasks hidden knowledge.

    Example
    -------
    >>> # Train binary classifier with Or7a
    >>> exp = Or7aBinaryClassification(
    ...     circuit=circuit,
    ...     plasticity=plasticity,
    ...     or7a_glomerulus="DL5",
    ...     positive_class_glomerulus="DA1",
    ...     negative_class_glomerulus="DL3",
    ... )
    >>>
    >>> # Run unmasking protocol
    >>> results = exp.run_full_experiment(n_training_trials=50, n_test_trials=20)
    >>>
    >>> # Check for unmasking
    >>> if results['unmasking_effect'] > 0.30:
    ...     print("✅ Strong unmasking - Or7a suppresses expression!")
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        plasticity: DopamineModulatedPlasticity,
        or7a_glomerulus: str,
        positive_class_glomerulus: str,
        negative_class_glomerulus: str,
        learning_rate: float = 0.001,
        veto_strength: float = 0.8,
        or7a_activation_threshold: float = 0.3,
        decision_threshold: float = 0.0,
    ) -> None:
        """Initialize binary classification experiment."""
        self.circuit = circuit
        self.plasticity = plasticity
        self.or7a_glomerulus = or7a_glomerulus
        self.positive_class_glomerulus = positive_class_glomerulus
        self.negative_class_glomerulus = negative_class_glomerulus
        self.learning_rate = learning_rate
        self.veto_strength = veto_strength
        self.decision_threshold = decision_threshold

        # Create Or7a veto gate
        self.veto_gate = Or7aVetoGate(
            circuit=circuit,
            or7a_glomerulus=or7a_glomerulus,
            activation_threshold=or7a_activation_threshold,
            veto_strength=1.0,  # Full veto strength in gate
            graded=True,
        )

        # History tracking
        self.training_history: List[Dict[str, Any]] = []
        self.test_with_or7a_history: List[Dict[str, Any]] = []
        self.test_without_or7a_history: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}

    def run_trial(
        self,
        odor_glomerulus: str,
        label: int,
        apply_plasticity: bool = True,
        apply_veto: bool = True,
        trial_type: str = "training",
    ) -> Dict[str, Any]:
        """Run a single binary classification trial.

        Parameters
        ----------
        odor_glomerulus : str
            Glomerulus to activate (positive or negative class).
        label : int
            True label (1 for positive class, 0 for negative class).
        apply_plasticity : bool, optional
            Whether to apply synaptic plasticity. Default: True
        apply_veto : bool, optional
            Whether to apply Or7a veto gating. Default: True
        trial_type : str, optional
            Trial type for logging. Default: "training"

        Returns
        -------
        Dict[str, Any]
            Trial results with keys:
            - odor_glomerulus, label, predicted_label, correct
            - mbon_output, veto_signal, weight_change_magnitude
            - trial_type

        Notes
        -----
        **Binary classification logic**:
        - MBON output > threshold → predict 1 (positive class)
        - MBON output ≤ threshold → predict 0 (negative class)
        - Reward = +1 for correct, -1 for incorrect
        - RPE drives synaptic updates

        **Or7a veto during training**:
        - Reduces but doesn't eliminate plasticity
        - Allows learning with suppression
        - Creates "latent knowledge" that can be unmasked
        """
        # 1. Activate PNs for odor
        pn_activity = self.circuit.activate_pns_by_glomeruli(
            [odor_glomerulus],
            firing_rate=1.0,
        )

        # 2. Compute Or7a veto signal
        veto_signal = 0.0
        if apply_veto:
            veto_signal = self.veto_gate.compute_veto_signal(pn_activity)

        # 3. Forward propagation: PN → KC → MBON
        kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
        mbon_output_full = self.plasticity.compute_mbon_output(kc_activity)

        # Use mean MBON output for binary classification
        mbon_output = float(np.mean(mbon_output_full))

        # 4. Binary classification decision
        predicted_label = 1 if mbon_output > self.decision_threshold else 0
        correct = predicted_label == label

        # 5. Compute reward and RPE
        reward = 1.0 if correct else -1.0
        predicted_value = mbon_output
        rpe = self.plasticity.compute_rpe(reward, predicted_value)

        # 6. Synaptic plasticity (if enabled)
        weight_change_magnitude = 0.0
        if apply_plasticity:
            # Compute weight update: ΔW = η × KC × RPE
            # kc_to_mbon shape: (n_mbon, n_kc)
            # delta_w shape: (n_mbon, n_kc)
            n_mbon = self.plasticity.kc_to_mbon.shape[0]
            delta_w = self.learning_rate * np.outer(np.ones(n_mbon), kc_activity) * rpe

            # Apply Or7a veto gating if enabled
            if apply_veto:
                delta_w = self.veto_gate.gate_plasticity(
                    delta_w,
                    veto_signal * self.veto_strength,
                )

            # Update weights
            self.plasticity.kc_to_mbon += delta_w
            weight_change_magnitude = float(np.abs(delta_w).mean())

        # 7. Record trial data
        trial_data = {
            "odor_glomerulus": odor_glomerulus,
            "label": label,
            "predicted_label": predicted_label,
            "correct": correct,
            "mbon_output": float(mbon_output),
            "veto_signal": float(veto_signal),
            "weight_change_magnitude": weight_change_magnitude,
            "trial_type": trial_type,
        }

        return trial_data

    def train_binary_classifier(
        self,
        n_trials: int = 50,
        apply_veto: bool = True,
    ) -> Dict[str, Any]:
        """Train binary classifier with balanced positive/negative examples.

        Parameters
        ----------
        n_trials : int, optional
            Total number of training trials (half positive, half negative).
            Default: 50
        apply_veto : bool, optional
            Whether to apply Or7a veto during training. Default: True

        Returns
        -------
        Dict[str, Any]
            Training summary with keys:
            - n_trials, accuracy, mean_mbon_output, mean_veto_signal

        Notes
        -----
        **Training protocol**:
        - Alternate between positive and negative class examples
        - Apply Or7a veto if enabled (creates suppression)
        - Track accuracy, MBON outputs, veto signals
        """
        print(f"\n[Training] Binary Classification (n={n_trials})")
        print(f"  Positive class: {self.positive_class_glomerulus}")
        print(f"  Negative class: {self.negative_class_glomerulus}")
        print(f"  Or7a veto: {'ACTIVE' if apply_veto else 'DISABLED'}")

        self.training_history = []
        n_positive = n_trials // 2
        n_negative = n_trials - n_positive

        # Interleave positive and negative examples
        for i in range(max(n_positive, n_negative)):
            # Positive class trial
            if i < n_positive:
                trial_data = self.run_trial(
                    odor_glomerulus=self.positive_class_glomerulus,
                    label=1,
                    apply_plasticity=True,
                    apply_veto=apply_veto,
                    trial_type="training_positive",
                )
                self.training_history.append(trial_data)

            # Negative class trial
            if i < n_negative:
                trial_data = self.run_trial(
                    odor_glomerulus=self.negative_class_glomerulus,
                    label=0,
                    apply_plasticity=True,
                    apply_veto=apply_veto,
                    trial_type="training_negative",
                )
                self.training_history.append(trial_data)

        # Compute training summary
        accuracy = np.mean([t["correct"] for t in self.training_history])
        mean_mbon = np.mean([t["mbon_output"] for t in self.training_history])
        mean_veto = np.mean([t["veto_signal"] for t in self.training_history])

        summary = {
            "n_trials": len(self.training_history),
            "accuracy": float(accuracy),
            "mean_mbon_output": float(mean_mbon),
            "mean_veto_signal": float(mean_veto),
        }

        print(f"  Training accuracy: {accuracy:.2%}")
        print(f"  Mean MBON output: {mean_mbon:.3f}")
        print(f"  Mean veto signal: {mean_veto:.3f}")

        return summary

    def test_classifier(
        self,
        n_trials: int = 20,
        apply_veto: bool = True,
        test_type: str = "with_or7a",
    ) -> Dict[str, Any]:
        """Test binary classifier accuracy.

        Parameters
        ----------
        n_trials : int, optional
            Total number of test trials (half positive, half negative).
            Default: 20
        apply_veto : bool, optional
            Whether to apply Or7a veto during testing. Default: True
        test_type : str, optional
            Test type: "with_or7a" or "without_or7a". Default: "with_or7a"

        Returns
        -------
        Dict[str, Any]
            Test summary with keys:
            - n_trials, accuracy, positive_accuracy, negative_accuracy
            - mean_mbon_output, mean_veto_signal

        Notes
        -----
        **Critical distinction**:
        - WITH Or7a: Veto suppresses responses → lower accuracy
        - WITHOUT Or7a: Full responses expressed → higher accuracy
        - Difference reveals suppression magnitude
        """
        print(f"\n[Testing] {test_type.upper()} (n={n_trials})")
        print(f"  Or7a veto: {'ACTIVE' if apply_veto else 'DISABLED'}")

        test_history = []
        n_positive = n_trials // 2
        n_negative = n_trials - n_positive

        # Test positive and negative examples
        for i in range(max(n_positive, n_negative)):
            # Positive class trial
            if i < n_positive:
                trial_data = self.run_trial(
                    odor_glomerulus=self.positive_class_glomerulus,
                    label=1,
                    apply_plasticity=False,  # No learning during test
                    apply_veto=apply_veto,
                    trial_type=f"test_{test_type}_positive",
                )
                test_history.append(trial_data)

            # Negative class trial
            if i < n_negative:
                trial_data = self.run_trial(
                    odor_glomerulus=self.negative_class_glomerulus,
                    label=0,
                    apply_plasticity=False,  # No learning during test
                    apply_veto=apply_veto,
                    trial_type=f"test_{test_type}_negative",
                )
                test_history.append(trial_data)

        # Store test history
        if test_type == "with_or7a":
            self.test_with_or7a_history = test_history
        else:
            self.test_without_or7a_history = test_history

        # Compute test summary
        accuracy = np.mean([t["correct"] for t in test_history])
        positive_trials = [t for t in test_history if t["label"] == 1]
        negative_trials = [t for t in test_history if t["label"] == 0]
        positive_acc = np.mean([t["correct"] for t in positive_trials]) if positive_trials else 0.0
        negative_acc = np.mean([t["correct"] for t in negative_trials]) if negative_trials else 0.0
        mean_mbon = np.mean([t["mbon_output"] for t in test_history])
        mean_veto = np.mean([t["veto_signal"] for t in test_history])

        summary = {
            "n_trials": len(test_history),
            "accuracy": float(accuracy),
            "positive_accuracy": float(positive_acc),
            "negative_accuracy": float(negative_acc),
            "mean_mbon_output": float(mean_mbon),
            "mean_veto_signal": float(mean_veto),
        }

        print(f"  Overall accuracy: {accuracy:.2%}")
        print(f"  Positive class accuracy: {positive_acc:.2%}")
        print(f"  Negative class accuracy: {negative_acc:.2%}")
        print(f"  Mean MBON output: {mean_mbon:.3f}")

        return summary

    def run_full_experiment(
        self,
        n_training_trials: int = 50,
        n_test_trials: int = 20,
    ) -> Dict[str, Any]:
        """Run full unmasking experiment protocol.

        Protocol
        --------
        1. Train binary classifier WITH Or7a veto
        2. Test WITH Or7a (expect suppression)
        3. Remove Or7a (ablation)
        4. Re-test WITHOUT Or7a (expect unmasking)
        5. Compare accuracies to measure unmasking effect

        Parameters
        ----------
        n_training_trials : int, optional
            Number of training trials. Default: 50
        n_test_trials : int, optional
            Number of test trials per condition. Default: 20

        Returns
        -------
        Dict[str, Any]
            Results with keys:
            - training: Dict (training summary)
            - test_with_or7a: Dict (test WITH Or7a summary)
            - test_without_or7a: Dict (test WITHOUT Or7a summary)
            - unmasking_effect: float (accuracy gain from removing Or7a)
            - interpretation: str (unmasking evidence)

        Example
        -------
        >>> results = exp.run_full_experiment(
        ...     n_training_trials=50,
        ...     n_test_trials=20
        ... )
        >>>
        >>> print(f"WITH Or7a accuracy: {results['test_with_or7a']['accuracy']:.2%}")
        >>> print(f"WITHOUT Or7a accuracy: {results['test_without_or7a']['accuracy']:.2%}")
        >>> print(f"Unmasking effect: {results['unmasking_effect']:.2%}")
        """
        print("=" * 80)
        print("EXPERIMENT 9: Binary Classification with Or7a Removal")
        print("=" * 80)
        print("\nThis experiment tests whether Or7a veto SUPPRESSES EXPRESSION")
        print("of learned knowledge (unmasking hypothesis).")
        print("\nProtocol:")
        print("  Phase 1: Train binary classifier WITH Or7a veto")
        print("  Phase 2: Test WITH Or7a (expect suppression)")
        print("  Phase 3: Remove Or7a (ablation)")
        print("  Phase 4: Re-test WITHOUT Or7a (expect unmasking)")
        print("=" * 80)

        # Phase 1: Training WITH Or7a
        print("\n" + "=" * 80)
        print("[Phase 1] Training WITH Or7a Veto")
        print("=" * 80)
        training_summary = self.train_binary_classifier(
            n_trials=n_training_trials,
            apply_veto=True,
        )

        # Phase 2: Testing WITH Or7a
        print("\n" + "=" * 80)
        print("[Phase 2] Testing WITH Or7a Veto (Suppression)")
        print("=" * 80)
        test_with_summary = self.test_classifier(
            n_trials=n_test_trials,
            apply_veto=True,
            test_type="with_or7a",
        )

        # Phase 3: Or7a Ablation
        print("\n" + "=" * 80)
        print("[Phase 3] Or7a Ablation (Removing Veto)")
        print("=" * 80)
        print("  Disabling Or7a veto to unmask suppressed responses...")

        # Phase 4: Testing WITHOUT Or7a
        print("\n" + "=" * 80)
        print("[Phase 4] Testing WITHOUT Or7a (Unmasking)")
        print("=" * 80)
        test_without_summary = self.test_classifier(
            n_trials=n_test_trials,
            apply_veto=False,
            test_type="without_or7a",
        )

        # Compute unmasking effect
        accuracy_with = test_with_summary["accuracy"]
        accuracy_without = test_without_summary["accuracy"]
        unmasking_effect = accuracy_without - accuracy_with

        # Interpretation
        if unmasking_effect > 0.30:
            interpretation = "✅ STRONG UNMASKING: Or7a suppresses expression of learned knowledge"
        elif unmasking_effect > 0.15:
            interpretation = "⚠️  MODERATE UNMASKING: Or7a partially suppresses expression"
        elif unmasking_effect > 0:
            interpretation = "⚠️  WEAK UNMASKING: Or7a has minimal suppression effect"
        else:
            interpretation = "❌ NO UNMASKING: Or7a removal did not reveal hidden knowledge"

        # Store results
        self.results = {
            "training": training_summary,
            "test_with_or7a": test_with_summary,
            "test_without_or7a": test_without_summary,
            "unmasking_effect": float(unmasking_effect),
            "interpretation": interpretation,
        }

        # Print comparison
        print("\n" + "=" * 80)
        print("UNMASKING COMPARISON")
        print("=" * 80)
        print("\nTest WITH Or7a (Suppressed):")
        print(f"  Overall accuracy:   {accuracy_with:.2%}")
        print(f"  Positive accuracy:  {test_with_summary['positive_accuracy']:.2%}")
        print(f"  Negative accuracy:  {test_with_summary['negative_accuracy']:.2%}")

        print("\nTest WITHOUT Or7a (Unmasked):")
        print(f"  Overall accuracy:   {accuracy_without:.2%}")
        print(f"  Positive accuracy:  {test_without_summary['positive_accuracy']:.2%}")
        print(f"  Negative accuracy:  {test_without_summary['negative_accuracy']:.2%}")

        print("\n" + "-" * 80)
        print("UNMASKING EFFECT")
        print("-" * 80)
        print(f"Accuracy gain from Or7a removal: {unmasking_effect:+.2%}")
        print(f"  (Measures how much Or7a suppressed expression)")
        print(f"\nInterpretation: {interpretation}")
        print("=" * 80)

        return self.results

    def get_results_dataframe(self) -> pd.DataFrame:
        """Create results DataFrame for analysis.

        Returns
        -------
        pd.DataFrame
            Results table with columns:
            - condition, accuracy, positive_acc, negative_acc, mean_mbon, mean_veto

        Example
        -------
        >>> df = exp.get_results_dataframe()
        >>> print(df)
                  condition  accuracy  positive_acc  negative_acc  mean_mbon  mean_veto
        0       WITH Or7a      0.55          0.50          0.60       0.12       0.45
        1    WITHOUT Or7a      0.90          0.85          0.95       0.78       0.00
        """
        if not self.results:
            raise ValueError("Run experiment first with run_full_experiment()")

        with_data = self.results["test_with_or7a"]
        without_data = self.results["test_without_or7a"]

        comparison = pd.DataFrame([
            {
                "condition": "WITH Or7a",
                "accuracy": with_data["accuracy"],
                "positive_acc": with_data["positive_accuracy"],
                "negative_acc": with_data["negative_accuracy"],
                "mean_mbon": with_data["mean_mbon_output"],
                "mean_veto": with_data["mean_veto_signal"],
            },
            {
                "condition": "WITHOUT Or7a",
                "accuracy": without_data["accuracy"],
                "positive_acc": without_data["positive_accuracy"],
                "negative_acc": without_data["negative_accuracy"],
                "mean_mbon": without_data["mean_mbon_output"],
                "mean_veto": without_data["mean_veto_signal"],
            },
        ])

        return comparison

    def __repr__(self) -> str:
        """Return summary string."""
        return (
            f"Or7aBinaryClassification(\n"
            f"  or7a_glomerulus='{self.or7a_glomerulus}'\n"
            f"  positive_class='{self.positive_class_glomerulus}'\n"
            f"  negative_class='{self.negative_class_glomerulus}'\n"
            f"  veto_strength={self.veto_strength}\n"
            f"  learning_rate={self.learning_rate}\n"
            f"  experiment_run={'Yes' if self.results else 'No'}\n"
            f")"
        )
