"""Experiment 1: Single-Unit Veto Gate — Pathway-Specific Blocking.

This module implements a computational test of the hypothesis that blocking (failure
to learn CS2 when CS1 already predicts reward) can be mediated by a single learned
"veto channel" that gates plasticity in the CS2 pathway.

Biological Context
------------------
**Blocking** (Kamin, 1968; Tanimoto et al., 2004 in Drosophila):

Phase 1: CS1 → US (reward)
    → Animal learns CS1 predicts reward
Phase 2: CS1 + CS2 → US (compound conditioning)
    → CS2 fails to acquire association despite pairing with reward
Test: CS2 alone
    → Weak/absent response (CS2 "blocked" by CS1)

**Computational hypothesis**:
Blocking may arise from a learned inhibitory pathway where CS1-activated neurons
suppress dopamine-gated plasticity in CS2-responsive synapses. This veto signal
implements predictive coding: "CS1 already predicts this outcome, so don't learn
about CS2."

**Veto gate mechanism**:
- Veto channel v = W_veto^T @ PN_activity
- Plasticity gating: dW = α × KC × MBON × DA × (1 - v)
- When CS1 active → v ≈ 1 → CS2 plasticity suppressed

This is analogous to optogenetic silencing experiments where blocking a specific
glomerulus during conditioning prevents learning.

Experimental Predictions
------------------------
1. **Control (no veto)**: CS2 acquires positive valence when paired with reward
2. **Veto active**: CS2 fails to learn despite CS2→reward pairing
3. **Veto specificity**: CS3 (unrelated odor) still learns normally

Example
-------
>>> from data_loaders.circuit_loader import CircuitLoader
>>> from pgcn.models.olfactory_circuit import OlfactoryCircuit
>>> from pgcn.models.learning_model import DopamineModulatedPlasticity
>>> from pgcn.experiments.experiment_1_veto_gate import VetoGateExperiment
>>>
>>> # Load circuit
>>> loader = CircuitLoader(cache_dir="data/cache")
>>> conn_matrix = loader.load_connectivity_matrix(normalize_weights="row")
>>> circuit = OlfactoryCircuit(conn_matrix, kc_sparsity_target=0.05)
>>>
>>> # Initialize plasticity
>>> plasticity = DopamineModulatedPlasticity(
...     kc_to_mbon_weights=conn_matrix.kc_to_mbon.toarray(),
...     learning_rate=0.01
... )
>>>
>>> # Run veto gate experiment
>>> veto_exp = VetoGateExperiment(
...     circuit=circuit,
...     plasticity=plasticity,
...     veto_glomerulus="DA1"  # DA1 pathway acts as blocker
... )
>>> results = veto_exp.run_full_experiment()
>>>
>>> # Analyze blocking
>>> print(f"OdorA blocked: {results['test_responses']['DA1']:.3f}")
>>> print(f"OdorB control: {results['test_responses']['DL3']:.3f}")
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit


class VetoGateExperiment:
    """Single-unit veto gate implementing pathway-specific blocking.

    This experiment tests whether a learned veto channel (single learned weight
    vector) can selectively block plasticity in one PN→KC→MBON pathway while
    preserving learning in other pathways.

    Biological Rationale
    --------------------
    **Why a veto gate model?**

    1. **Computational efficiency**: A single learned channel (n_pn weights) can
       gate plasticity in thousands of KC→MBON synapses, providing a compact
       mechanism for blocking.

    2. **Biological precedent**: GABAergic inhibition in the MB (APL neuron)
       provides global gating of KC activity. Veto gates extend this concept to
       pathway-specific plasticity gating.

    3. **Predictive coding**: Veto signals implement "don't learn what's already
       predicted" — a core principle of efficient credit assignment.

    **Mathematical form**:
    v(t) = sigmoid(W_veto^T @ PN(t))
    dW_km = α × KC_k × MBON_m × DA × (1 - β × v)

    where β controls veto strength (0 = no veto, 1 = complete block).

    Parameters
    ----------
    circuit : OlfactoryCircuit
        Feedforward circuit for PN activation and propagation.
    plasticity : DopamineModulatedPlasticity
        Plasticity manager for weight updates.
    veto_glomerulus : str
        Glomerulus whose learning will be BLOCKED by the veto signal (e.g., "DL3").
        This is the distractor odor whose plasticity we want to suppress.
    block_learning_rate : float, optional
        Learning rate for training the veto weight vector. If 0, veto is
        hand-crafted (initialized to PNs of veto glomerulus). Default: 0.0
    veto_strength : float, optional
        Gating strength β in (1 - β × v). Range: [0, 1].
        0 = no gating, 1 = complete block. Default: 1.0

    Attributes
    ----------
    circuit : OlfactoryCircuit
        Circuit for PN→KC→MBON propagation.
    plasticity : DopamineModulatedPlasticity
        Plasticity rules for KC→MBON updates.
    veto_glomerulus : str
        Glomerulus implementing veto.
    veto_weight : np.ndarray
        Weight vector for veto computation. Shape: (n_pn,).
    veto_strength : float
        Gating strength parameter.

    Notes
    -----
    **Design choice: Hand-crafted vs learned veto**

    In this implementation, we initialize the veto weight vector to uniformly
    weight the PNs of the veto glomerulus. This is a simplification; in a more
    realistic model, the veto weights would be learned through a separate
    plasticity rule (e.g., Hebbian learning of PN→veto connections).

    Future extensions could implement learned veto weights via:
    dW_veto = α_veto × PN × RPE_veto

    where RPE_veto = (predicted_value - reward) measures prediction accuracy.

    Example
    -------
    >>> # Initialize veto experiment
    >>> veto_exp = VetoGateExperiment(
    ...     circuit=circuit,
    ...     plasticity=plasticity,
    ...     veto_glomerulus="DA1",
    ...     veto_strength=1.0  # Complete blocking
    ... )
    >>>
    >>> # Run blocking protocol
    >>> results = veto_exp.run_full_experiment()
    >>>
    >>> # Extract metrics
    >>> phase1_trials = pd.DataFrame(results['phase1_trials'])
    >>> phase2_trials = pd.DataFrame(results['phase2_trials'])
    >>> print(f"Phase 1 mean RPE: {phase1_trials['rpe'].mean():.3f}")
    >>> print(f"Phase 2 OdorA RPE: {phase2_trials[phase2_trials['odor']=='DA1']['rpe'].mean():.3f}")
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        plasticity: DopamineModulatedPlasticity,
        veto_glomerulus: str,
        block_learning_rate: float = 0.0,
        veto_strength: float = 1.0,
    ) -> None:
        """Initialize veto gate experiment with circuit and veto parameters."""
        self.circuit = circuit
        self.plasticity = plasticity
        self.veto_glomerulus = veto_glomerulus
        self.block_learning_rate = block_learning_rate
        self.veto_strength = veto_strength

        # Initialize veto weight vector
        n_pn = len(circuit.connectivity.pn_ids)
        self.veto_weight = np.zeros(n_pn, dtype=np.float64)

        # Set veto weights for the specified glomerulus
        # Get indices of PNs belonging to veto glomerulus
        glom_indices = circuit.connectivity.get_pn_indices([veto_glomerulus])

        if len(glom_indices) == 0:
            raise ValueError(
                f"No PNs found for glomerulus '{veto_glomerulus}'. "
                f"Available glomeruli: {set(circuit.connectivity.pn_glomeruli.values())}"
            )

        # Initialize uniformly across veto glomerulus PNs
        self.veto_weight[glom_indices] = 1.0 / len(glom_indices)

    def apply_veto(self, pn_activity: np.ndarray) -> float:
        """Compute veto gate signal from PN activity.

        Biological Rationale
        --------------------
        The veto signal v represents the "prediction confidence" of the blocking
        pathway. When PNs of the veto glomerulus are active, v is high, indicating
        that this odor component already predicts the outcome. This suppresses
        learning in other pathways (blocking).

        Mathematical Form
        -----------------
        v = W_veto^T @ PN_activity
        v_clipped = clip(v, 0, 1)  # Ensure gate ∈ [0, 1]

        Alternative (with sigmoid):
        v = sigmoid(W_veto^T @ PN_activity - threshold)

        Parameters
        ----------
        pn_activity : np.ndarray
            PN firing rates. Shape: (n_pn,).

        Returns
        -------
        float
            Veto gate value in [0, 1]. 0 = no veto (learning allowed),
            1 = complete veto (learning blocked).

        Example
        -------
        >>> # Activate veto glomerulus
        >>> pn_act = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)
        >>> v = veto_exp.apply_veto(pn_act)
        >>> print(f"Veto signal: {v:.3f}")  # Should be ≈1.0 for DA1
        """
        veto_signal = np.dot(self.veto_weight, pn_activity)
        return np.clip(veto_signal, 0.0, 1.0)

    def run_trial_with_veto(
        self,
        odor: str,
        reward: float,
        veto_active: bool = True,
    ) -> Dict[str, Any]:
        """Run conditioning trial with optional veto gating.

        This implements the core blocking mechanism: plasticity is gated by
        the veto signal, preventing learning when the veto pathway predicts
        the outcome.

        Biological Rationale
        --------------------
        **Gated plasticity equation**:
        dW_km = α × KC_k × MBON_m × DA × (1 - β × v)

        When v ≈ 1 (veto active), the gating factor (1 - β × v) ≈ 0, suppressing
        weight updates. This implements "predictive suppression of plasticity":
        when CS1 already predicts reward, CS2 synapses don't strengthen despite
        CS2-reward pairing.

        Parameters
        ----------
        odor : str
            Glomerulus name or odor identity.
        reward : float
            Outcome value (0 or 1 for binary conditioning).
        veto_active : bool, optional
            If True, apply veto gating. If False, standard three-factor learning.
            Default: True

        Returns
        -------
        Dict[str, Any]
            Trial outcome with keys:
            - odor: str
            - reward: float
            - mbon_output: float (valence prediction)
            - veto_value: float (veto gate signal)
            - gating_factor: float (1 - β × v, controls plasticity)
            - rpe: float (reward prediction error)
            - weight_change_magnitude: float

        Example
        -------
        >>> # Trial with veto active (blocking)
        >>> trial = veto_exp.run_trial_with_veto(
        ...     odor="DA1",
        ...     reward=1.0,
        ...     veto_active=True
        ... )
        >>> print(f"Gating factor: {trial['gating_factor']:.3f}")  # Should be ~0
        >>>
        >>> # Trial with veto off (normal learning)
        >>> trial = veto_exp.run_trial_with_veto(
        ...     odor="DL3",
        ...     reward=0.0,
        ...     veto_active=False
        ... )
        >>> print(f"Gating factor: {trial['gating_factor']:.3f}")  # Should be ~1
        """
        # 1. Activate PNs for this odor
        pn_activity = self.circuit.activate_pns_by_glomeruli([odor], firing_rate=1.0)

        # 2. Compute veto gate
        veto_value = self.apply_veto(pn_activity) if veto_active else 0.0

        # 3. Compute gating factor with floor to avoid complete shutdown
        if veto_active:
            gating_factor = 1.0 - self.veto_strength * veto_value
        else:
            gating_factor = 1.0
        gating_factor = max(0.1, gating_factor)

        # 4. Forward propagation (PN → KC → MBON)
        kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)

        # Compute raw and normalized MBON activity
        raw_mbon_output = self.plasticity.kc_to_mbon @ kc_activity
        mbon_output_vec = self.plasticity.apply_mbon_activation(raw_mbon_output)
        mbon_output = float(mbon_output_vec[0])

        # 5. Compute RPE
        predicted_value = mbon_output / self.plasticity.mbon_output_max
        predicted_value = np.clip(predicted_value, -1.0, 1.0)
        rpe = self.plasticity.compute_rpe(reward, predicted_value, learning_rate_rpe=0.1)

        # 6. Map RPE to dopamine
        dopamine = rpe

        # 7. Apply gated weight update
        # Modify the standard three-factor update with gating
        prepost = np.outer(raw_mbon_output, kc_activity)
        delta_w = self.plasticity.learning_rate * prepost * dopamine * gating_factor * 1.0  # dt=1

        # Apply frozen/sign-flip masks if present (for microsurgery experiments)
        if self.plasticity._frozen_synapses:
            for (kc_idx, mbon_idx) in self.plasticity._frozen_synapses:
                if 0 <= kc_idx < len(kc_activity) and 0 <= mbon_idx < len(mbon_output_vec):
                    delta_w[mbon_idx, kc_idx] = 0.0

        # Update weights
        self.plasticity.kc_to_mbon += delta_w
        self.plasticity.enforce_connectivity_mask()

        weight_change_magnitude = float(np.linalg.norm(delta_w))

        return {
            "odor": odor,
            "reward": reward,
            "mbon_output": mbon_output,
            "veto_value": veto_value,
            "gating_factor": gating_factor,
            "rpe": rpe,
            "dopamine": dopamine,
            "weight_change_magnitude": weight_change_magnitude,
        }

    def run_full_experiment(
        self,
        n_phase1_trials: int = 5,
        n_phase2_trials: int = 20,
        odor_a: str = "DA1",
        odor_b: str = "DL3",
    ) -> Dict[str, Any]:
        """Execute complete blocking protocol with three phases.

        Biological Protocol
        -------------------
        **Phase 1: Baseline learning (veto off)**
        - Train both OdorA→reward and OdorB→reward
        - Establishes equal baseline (both odors acquire similar associations)
        - Veto weight vector is active but gating is OFF for both

        **Phase 2: Veto blocking test (veto on for OdorB only)**
        - Train OdorA→reward without veto (normal plasticity continues)
        - Train OdorB→reward with veto gate active (plasticity blocked)
        - Prediction: OdorA continues learning, OdorB cannot learn more (blocked)
        - Biological rationale: Block distractor learning, preserve reward learning

        **Phase 3: Test (no learning)**
        - Present OdorA and OdorB without reward
        - Measure MBON responses to assess learned valence
        - Expected: OdorA valence > OdorB valence (blocking effect demonstrated)

        Parameters
        ----------
        n_phase1_trials : int, optional
            Number of trials in Phase 1 (baseline learning). Default: 5
        n_phase2_trials : int, optional
            Number of trials in Phase 2 (blocking test). Default: 20
        odor_a : str, optional
            Target odor (learns normally, NOT blocked). Default: "DA1"
        odor_b : str, optional
            Distractor odor (blocked by veto). Default: "DL3"

        Returns
        -------
        Dict[str, Any]
            Experiment results with keys:
            - phase1_trials: List[Dict] (baseline trials)
            - phase2_trials: List[Dict] (blocking trials)
            - test_responses: Dict[str, float] (final MBON responses)
            - blocking_index: float ((OdorB - OdorA) / (OdorB + OdorA))

        Notes
        -----
        **Interpretation of results**:
        - blocking_index > 0: OdorA learned more than OdorB (blocking successful)
        - blocking_index ≈ 0: Both odors learned equally (no blocking)
        - blocking_index < 0: OdorB learned more (blocking failed)

        **Biological mapping**:
        This protocol mimics Drosophila compound conditioning experiments where
        CS1 training precedes CS1+CS2 compound training. In flies, CS2 shows
        reduced behavioral response despite reward pairing (Tanimoto et al., 2004).

        Example
        -------
        >>> # Run full blocking experiment
        >>> results = veto_exp.run_full_experiment(
        ...     n_phase1_trials=10,
        ...     n_phase2_trials=30,
        ...     odor_a="DA1",
        ...     odor_b="DL3"
        ... )
        >>>
        >>> # Analyze blocking
        >>> blocking_idx = results['blocking_index']
        >>> print(f"Blocking index: {blocking_idx:.3f}")
        >>> if blocking_idx > 0.2:
        ...     print("Blocking effect detected!")
        """
        results = {
            "phase1_trials": [],
            "phase2_trials": [],
            "test_responses": {},
        }

        # Phase 1: Baseline learning (veto off for both)
        # Both odors get reward to establish equal baseline
        for trial_idx in range(n_phase1_trials):
            if trial_idx % 2 == 0:
                # OdorA → reward (veto off)
                trial_data = self.run_trial_with_veto(
                    odor=odor_a,
                    reward=1.0,
                    veto_active=False,  # No gating in Phase 1
                )
                results["phase1_trials"].append(trial_data)
            else:
                # OdorB → reward (veto off, same as OdorA)
                trial_data = self.run_trial_with_veto(
                    odor=odor_b,
                    reward=1.0,  # Same reward as OdorA to establish equal baseline
                    veto_active=False,
                )
                results["phase1_trials"].append(trial_data)

        # Phase 2: Blocking test (veto on for OdorB - the distractor)
        # Interleave OdorA→reward (veto OFF) and OdorB→reward (veto ON)
        # Both odors get reward, but only OdorB's plasticity is blocked by veto
        # Biological rationale: Block distractor learning, allow target learning
        for trial_idx in range(n_phase2_trials):
            current_odor = odor_a if trial_idx % 2 == 0 else odor_b
            veto_active = current_odor == self.veto_glomerulus
            trial_data = self.run_trial_with_veto(
                odor=current_odor,
                reward=1.0,
                veto_active=veto_active,
            )
            results["phase2_trials"].append(trial_data)

        # Phase 3: Test responses (no reward, no learning)
        # Measure final MBON outputs for both odors
        for odor in [odor_a, odor_b]:
            pn_activity = self.circuit.activate_pns_by_glomeruli([odor], firing_rate=1.0)
            kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)
            mbon_output_vec = self.plasticity.compute_mbon_output(kc_activity)
            mbon_output = float(mbon_output_vec[0])
            results["test_responses"][odor] = mbon_output
            results.setdefault("test_responses_abs", {})[odor] = abs(mbon_output)

        # Compute blocking index: (OdorA - OdorB) / (OdorA + OdorB + epsilon)
        # Positive = OdorA (target) > OdorB (distractor) → blocking successful
        odor_a_response = abs(results["test_responses"][odor_a])
        odor_b_response = abs(results["test_responses"][odor_b])
        denom = odor_a_response + odor_b_response + 1e-6
        blocking_index = (odor_a_response - odor_b_response) / denom

        results["blocking_index"] = blocking_index

        # Compute Phase 2 learning effectiveness
        # This measures how much each odor learned during Phase 2 (when veto was active on OdorB)
        phase2_df = pd.DataFrame(results["phase2_trials"])
        odor_a_trials = phase2_df[phase2_df["odor"] == odor_a]
        odor_b_trials = phase2_df[phase2_df["odor"] == odor_b]

        if len(odor_a_trials) > 0 and len(odor_b_trials) > 0:
            odor_a_change = abs(odor_a_trials.iloc[-1]["mbon_output"] - odor_a_trials.iloc[0]["mbon_output"])
            odor_b_change = abs(odor_b_trials.iloc[-1]["mbon_output"] - odor_b_trials.iloc[0]["mbon_output"])

            # Blocking effectiveness: positive means OdorA learned more than OdorB (veto worked)
            # OdorA (target, not blocked) should learn more than OdorB (distractor, blocked)
            blocking_effectiveness = (odor_a_change - odor_b_change) / (odor_a_change + odor_b_change + 1e-9)
            results["blocking_effectiveness"] = blocking_effectiveness
            results["odor_a_phase2_change"] = float(odor_a_change)
            results["odor_b_phase2_change"] = float(odor_b_change)
        else:
            results["blocking_effectiveness"] = 0.0
            results["odor_a_phase2_change"] = 0.0
            results["odor_b_phase2_change"] = 0.0

        return results

    def analyze_blocking_effect(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute blocking metrics from experiment results.

        Metrics
        -------
        - blocking_index: (OdorA - OdorB) / (OdorA + OdorB)
        - phase2_odor_a_learning: Mean |RPE| in Phase 2 for OdorA (target, not blocked)
        - phase2_odor_b_learning: Mean |RPE| in Phase 2 for OdorB (distractor, blocked)
        - veto_efficacy: Mean veto signal for OdorB in Phase 2
        - mean_gating_suppression: Mean plasticity suppression for OdorB in Phase 2

        Parameters
        ----------
        results : Dict[str, Any]
            Output from run_full_experiment().

        Returns
        -------
        Dict[str, float]
            Blocking analysis metrics.

        Example
        -------
        >>> results = veto_exp.run_full_experiment()
        >>> metrics = veto_exp.analyze_blocking_effect(results)
        >>> print(f"Blocking index: {metrics['blocking_index']:.3f}")
        >>> print(f"Veto efficacy: {metrics['veto_efficacy']:.3f}")
        """
        # Extract RPE sequences
        phase2_trials = pd.DataFrame(results["phase2_trials"])

        phase2_odor_a = phase2_trials[
            phase2_trials["odor"] == list(results["test_responses"].keys())[0]
        ]
        phase2_odor_b = phase2_trials[
            phase2_trials["odor"] == list(results["test_responses"].keys())[1]
        ]

        # Veto metrics come from OdorB (the distractor being blocked)
        metrics = {
            "blocking_index": results["blocking_index"],
            "phase2_odor_a_mean_rpe": float(phase2_odor_a["rpe"].abs().mean()) if len(phase2_odor_a) > 0 else 0.0,
            "phase2_odor_b_mean_rpe": float(phase2_odor_b["rpe"].abs().mean()) if len(phase2_odor_b) > 0 else 0.0,
            "veto_efficacy": float(phase2_odor_b["veto_value"].mean()) if len(phase2_odor_b) > 0 else 0.0,
            "mean_gating_suppression": float(1.0 - phase2_odor_b["gating_factor"].mean()) if len(phase2_odor_b) > 0 else 0.0,
        }

        return metrics
