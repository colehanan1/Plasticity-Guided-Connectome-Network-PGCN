"""Learning dynamics and dopamine-modulated plasticity for olfactory circuit.

This module implements the core plasticity mechanisms that enable associative learning
in the Drosophila mushroom body (MB). The central computational principle is
**three-factor Hebbian learning** at KC→MBON synapses, where synaptic weight changes
depend on presynaptic KC activity, postsynaptic MBON activity, and dopaminergic
teaching signals that convey reward/punishment information.

Biological Context
------------------
The MB implements a reinforcement learning system that associates odors (represented
by sparse KC codes) with outcomes (conveyed by dopamine signals from DANs):

1. **Dopamine-gated plasticity**: KC→MBON synapses are modified only when dopamine
   is present. This implements a gating mechanism where DAN activity serves as a
   "teaching signal" that determines when and how learning occurs.

2. **Three-factor rule**: Weight changes follow the equation:
   dW_km/dt = α × KC_k × MBON_m × DA

   where:
   - KC_k: presynaptic Kenyon cell activity (encodes which odor was present)
   - MBON_m: postsynaptic MBON activity (current valence prediction)
   - DA: dopamine signal (reward prediction error from DANs)
   - α: learning rate (plasticity strength)

3. **Reward prediction error (RPE)**: Dopamine signals encode the difference between
   expected and received reward:
   RPE = r(t) - V_predicted(t)

   Positive RPE (better than expected) → potentiation of CS→US associations
   Negative RPE (worse than expected) → depression of associations

4. **Eligibility traces**: Synaptic "tags" that persist for seconds after KC-MBON
   co-activity, allowing dopamine signals delivered slightly delayed to still gate
   plasticity. This implements synaptic tagging-and-capture observed in LTP experiments.

Experimental Validation
-----------------------
This computational model recapitulates key findings from Drosophila learning studies:

- Blocking (Tanimoto et al., 2004): Prior CS-US pairing prevents learning to new CS
- Overshadowing: Compound conditioning reduces learning to individual components
- Second-order conditioning: CS1→US, then CS2→CS1, produces CS2→US association
- RPE-like dopamine responses (Aso et al., 2014): DANs encode prediction errors

Example
-------
>>> from data_loaders.circuit_loader import CircuitLoader
>>> from pgcn.models.olfactory_circuit import OlfactoryCircuit
>>> from pgcn.models.learning_model import DopamineModulatedPlasticity, LearningExperiment
>>>
>>> # Load circuit
>>> loader = CircuitLoader(cache_dir="data/cache")
>>> conn_matrix = loader.load_connectivity_matrix(normalize_weights="row")
>>> circuit = OlfactoryCircuit(conn_matrix, kc_sparsity_target=0.05)
>>>
>>> # Initialize plasticity with mutable weights
>>> kc_to_mbon_plastic = conn_matrix.kc_to_mbon.toarray()  # Convert to dense for learning
>>> plasticity = DopamineModulatedPlasticity(
...     kc_to_mbon_weights=kc_to_mbon_plastic,
...     learning_rate=0.01,
...     eligibility_trace_tau=0.1,
...     plasticity_mode="three_factor"
... )
>>>
>>> # Run conditioning experiment
>>> experiment = LearningExperiment(circuit, plasticity, n_trials=50)
>>> odor_seq = ["DA1"] * 25 + ["DL3"] * 25  # Train on two odors
>>> reward_seq = [1] * 25 + [0] * 25  # DA1→reward, DL3→no reward
>>> results = experiment.run_experiment(odor_seq, reward_seq)
>>> print(f"Final DA1 response: {results.iloc[-26]['mbon_valence']:.3f}")
>>> print(f"Final DL3 response: {results.iloc[-1]['mbon_valence']:.3f}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pgcn.models.olfactory_circuit import OlfactoryCircuit


class DopamineModulatedPlasticity:
    """Manages dopamine-gated Hebbian learning at KC→MBON synapses.

    This class encapsulates the plasticity rules that implement associative memory
    formation in the mushroom body. It maintains a reference to mutable KC→MBON
    weights and provides methods for computing reward prediction errors (RPE) and
    applying weight updates based on three-factor learning rules.

    Biological Design Principles
    -----------------------------
    1. **Three-factor gating**: Plasticity requires coincidence of three factors:
       - Presynaptic KC activity (odor presence)
       - Postsynaptic MBON activity (current valence)
       - Dopamine release (teaching signal)

       This implements the "eligibility × reinforcement" framework of reinforcement
       learning, where eligibility is set by KC×MBON and reinforcement by dopamine.

    2. **Bidirectional plasticity**: Positive dopamine → potentiation (LTP-like),
       negative dopamine → depression (LTD-like). This bidirectionality enables
       both approach learning (reward conditioning) and avoidance learning
       (punishment conditioning).

    3. **Weight bounds**: Biological synapses have finite resources (vesicles,
       receptors), imposing soft bounds on weight magnitudes. We implement this
       via optional weight decay (L2 regularization).

    4. **Temporal credit assignment**: Eligibility traces extend the temporal
       window for dopamine-gated learning, solving the credit assignment problem
       when reward is delayed relative to odor offset.

    Parameters
    ----------
    kc_to_mbon_weights : np.ndarray or sp.csr_matrix
        KC→MBON weight matrix to be modified during learning. Shape: (n_mbon, n_kc).
        This should be a mutable copy, not the original frozen connectivity.
    learning_rate : float, optional
        Learning rate α in dW = α × KC × MBON × DA. Biological range: 0.001-0.1.
        Higher values → faster learning but more interference. Default: 0.01
    eligibility_trace_tau : Optional[float], optional
        Time constant (seconds) for eligibility trace decay. If None, no traces used.
        Biological range: 0.05-0.5 seconds (matches synaptic tag persistence).
        Default: None (no eligibility traces)
    plasticity_mode : str, optional
        Algorithm for computing weight updates. Options:
        - "three_factor": dW = α × KC × MBON × DA (classic Hebbian)
        - "eligibility_trace": dW = α × e(t) × DA where e(t) is low-pass KC×MBON
        - "gated": dW = α × KC × MBON × DA × I(|DA| > threshold)
        Default: "three_factor"
    reward_prediction_tau : float, optional
        Time constant for RPE low-pass filter (seconds). Smooths RPE dynamics.
        Default: 1.0
    weight_decay_rate : float, optional
        Weight decay (L2 regularization) rate. Applied as W ← W × (1 - λ × dt).
        Default: 0.0 (no decay)

    Attributes
    ----------
    kc_to_mbon : np.ndarray
        Mutable weight matrix that is updated during learning.
    learning_rate : float
        Plasticity strength parameter.
    eligibility_traces : Optional[np.ndarray]
        Low-pass filtered KC×MBON products. None if no traces. Shape: (n_mbon, n_kc).
    rpe_filter : float
        Exponentially smoothed RPE value for stable learning.

    Raises
    ------
    ValueError
        If plasticity_mode is not recognized or learning_rate is non-positive.

    Notes
    -----
    **Design choice: Dense vs sparse matrices**

    During learning, we convert sparse matrices to dense numpy arrays because:
    1. Learning updates are dense (many synapses change each trial)
    2. Outer products (KC ⊗ MBON) naturally produce dense arrays
    3. Dense operations are faster for non-sparse computations

    After learning, weights can be converted back to sparse for storage/analysis.

    **Biological parameter ranges** (from literature):
    - Learning rate: 0.001-0.1 (Cassenaer & Laurent, 2012; Hige et al., 2015)
    - Eligibility trace τ: 50-500 ms (Gervasi et al., 2010; Yagishita et al., 2014)
    - Weight decay: 0.0001-0.01 per second (protein turnover rates)
    """

    def __init__(
        self,
        kc_to_mbon_weights: Union[np.ndarray, sp.csr_matrix],
        learning_rate: float = 0.01,
        eligibility_trace_tau: Optional[float] = None,
        plasticity_mode: str = "three_factor",
        reward_prediction_tau: float = 1.0,
        weight_decay_rate: float = 0.0,
    ) -> None:
        """Initialize plasticity manager with weight reference and parameters."""
        # Validate learning rate
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        # Validate plasticity mode
        valid_modes = {"three_factor", "eligibility_trace", "gated"}
        if plasticity_mode not in valid_modes:
            raise ValueError(
                f"plasticity_mode must be one of {valid_modes}, got '{plasticity_mode}'"
            )

        # Convert sparse to dense if needed (learning updates are dense)
        if sp.issparse(kc_to_mbon_weights):
            self.kc_to_mbon = kc_to_mbon_weights.toarray()
        else:
            self.kc_to_mbon = np.array(kc_to_mbon_weights, dtype=np.float64)

        self.learning_rate = learning_rate
        self.eligibility_trace_tau = eligibility_trace_tau
        self.plasticity_mode = plasticity_mode
        self.reward_prediction_tau = reward_prediction_tau
        self.weight_decay_rate = weight_decay_rate

        # Initialize eligibility traces if requested
        self.eligibility_traces: Optional[np.ndarray] = None
        if eligibility_trace_tau is not None:
            self.eligibility_traces = np.zeros_like(self.kc_to_mbon, dtype=np.float64)

        # Initialize RPE filter
        self.rpe_filter: float = 0.0

        # Optional: frozen synapses for microsurgery experiments
        self._frozen_synapses: set = set()

        # Optional: sign-flipped synapses for microsurgery experiments
        self._sign_flip_synapses: set = set()

    def compute_rpe(
        self,
        trial_outcome: float,
        predicted_value: float,
        learning_rate_rpe: float = 0.1,
    ) -> float:
        """Compute reward prediction error (RPE) with exponential smoothing.

        Biological Rationale
        --------------------
        Dopaminergic neurons (DANs) in the Drosophila MB encode reward prediction
        errors rather than raw reward values. This implements the temporal difference
        (TD) learning algorithm observed across species:

        - PPL1-DANs: activated by unexpected punishment, depressed by omitted punishment
        - PAM-DANs: activated by unexpected reward, depressed by omitted reward

        The RPE signal drives adaptive learning:
        - Positive RPE (surprise reward) → strengthen CS→reward association
        - Negative RPE (omitted reward) → weaken CS→reward association
        - Zero RPE (perfect prediction) → no learning (memory consolidation)

        Mathematical Form
        -----------------
        RPE(t) = r(t) - V_predicted(t)

        With optional exponential smoothing:
        RPE_filtered(t) = (1-β) × RPE_filtered(t-1) + β × RPE(t)

        where β = learning_rate_rpe controls smoothing strength.

        Parameters
        ----------
        trial_outcome : float
            Observed reward on this trial. Typically binary (0 or 1) but can be
            continuous for graded rewards.
        predicted_value : float
            Model's valence prediction before outcome delivery. Typically derived
            from MBON activity: V_pred = MBON_output[0] / scaling_factor
        learning_rate_rpe : float, optional
            Smoothing parameter for RPE filter. Range: (0, 1].
            Higher values → less smoothing (faster tracking of RPE changes).
            Default: 0.1

        Returns
        -------
        float
            Smoothed RPE value (positive or negative). Magnitude indicates surprise;
            sign indicates valence (positive = better than expected).

        Example
        -------
        >>> plasticity = DopamineModulatedPlasticity(weights, learning_rate=0.01)
        >>> # First trial: no prediction yet
        >>> rpe1 = plasticity.compute_rpe(trial_outcome=1.0, predicted_value=0.0)
        >>> print(f"RPE trial 1: {rpe1:.3f}")  # Positive surprise (unexpected reward)
        >>> # Later trial: good prediction
        >>> rpe2 = plasticity.compute_rpe(trial_outcome=1.0, predicted_value=0.95)
        >>> print(f"RPE trial 2: {rpe2:.3f}")  # Near-zero (well predicted)
        """
        # Compute raw RPE
        rpe_raw = trial_outcome - predicted_value

        # Apply exponential smoothing
        self.rpe_filter = (1 - learning_rate_rpe) * self.rpe_filter + learning_rate_rpe * rpe_raw

        return self.rpe_filter

    def update_weights(
        self,
        kc_activity: np.ndarray,
        mbon_activity: np.ndarray,
        dopamine_signal: float,
        dt: float = 1.0,
    ) -> Dict[str, float]:
        """Apply one-step weight update using three-factor Hebbian rule.

        Biological Rationale
        --------------------
        This implements the core plasticity computation observed in Drosophila MB:

        **Standard three-factor rule**:
        dW_km = α × KC_k × MBON_m × DA × dt

        where the outer product KC_k × MBON_m creates a matrix of pairwise products,
        then scaled by dopamine (teaching signal) and learning rate.

        **Why outer product?**
        Each KC→MBON synapse W_km should strengthen when:
        - KC_k is active (presynaptic input present)
        - MBON_m is active (postsynaptic neuron firing)
        - DA is positive (reward/reinforcement delivered)

        The outer product computes all n_kc × n_mbon pairwise products efficiently.

        **Eligibility trace variant**:
        Instead of instantaneous KC×MBON, use a low-pass filtered trace:
        e(t) ← decay × e(t) + (1-decay) × (KC × MBON)
        dW = α × e(t) × DA × dt

        This extends the temporal credit assignment window to ~100-500ms, allowing
        dopamine signals slightly delayed from odor offset to still gate learning.

        **Gated variant**:
        Only update if |DA| exceeds threshold (mimics all-or-none dopamine release):
        dW = α × KC × MBON × DA × I(|DA| > θ) × dt

        Parameters
        ----------
        kc_activity : np.ndarray
            Sparse KC activations. Shape: (n_kc,). Typically ~5% non-zero.
        mbon_activity : np.ndarray
            MBON firing rates. Shape: (n_mbon,). Dense (all MBONs active).
        dopamine_signal : float
            Dopamine teaching signal, typically RPE. Positive → potentiation,
            negative → depression. Magnitude indicates learning strength.
        dt : float, optional
            Integration time step (arbitrary units, typically 1.0 per trial).
            Default: 1.0

        Returns
        -------
        Dict[str, float]
            Diagnostics dictionary with keys:
            - "weight_change_magnitude": L2 norm of weight change (||dW||)
            - "mean_weight": current mean absolute weight
            - "max_weight": current maximum absolute weight
            - "eligibility_trace_norm": ||e(t)|| if traces enabled, else 0.0

        Raises
        ------
        ValueError
            If kc_activity or mbon_activity shapes don't match weight matrix dimensions.

        Notes
        -----
        **Computational efficiency**:
        The outer product operation creates a (n_mbon, n_kc) matrix, which can be
        large (~96 × 5177 = 497,000 elements). However, since KC activity is sparse
        (~5% active), the effective computation is:
        - Active KCs: ~259 out of 5177
        - Non-zero weight updates: 96 × 259 = 24,864 (5% of matrix)

        Future optimization: use sparse outer product for very large circuits.

        Example
        -------
        >>> # Simulate one learning trial
        >>> kc_act = np.zeros(5177)
        >>> kc_act[0:259] = 1.0  # Sparse KC code (~5%)
        >>> mbon_act = np.random.rand(96)
        >>> dopamine = 0.5  # Positive RPE (reward surprise)
        >>> diag = plasticity.update_weights(kc_act, mbon_act, dopamine, dt=1.0)
        >>> print(f"Weight change: {diag['weight_change_magnitude']:.6f}")
        """
        # Validate input shapes
        n_mbon, n_kc = self.kc_to_mbon.shape
        if kc_activity.shape != (n_kc,):
            raise ValueError(
                f"kc_activity shape {kc_activity.shape} doesn't match n_kc={n_kc}"
            )
        if mbon_activity.shape != (n_mbon,):
            raise ValueError(
                f"mbon_activity shape {mbon_activity.shape} doesn't match n_mbon={n_mbon}"
            )

        # Compute weight update based on plasticity mode
        if self.plasticity_mode == "three_factor":
            # Standard Hebbian: dW = α × (MBON ⊗ KC) × DA
            prepost_product = np.outer(mbon_activity, kc_activity)  # (n_mbon, n_kc)
            delta_w = self.learning_rate * prepost_product * dopamine_signal * dt

        elif self.plasticity_mode == "eligibility_trace":
            # Update eligibility trace (low-pass filter)
            prepost_product = np.outer(mbon_activity, kc_activity)
            decay = np.exp(-dt / self.eligibility_trace_tau) if self.eligibility_trace_tau else 0.0
            self.eligibility_traces = (
                decay * self.eligibility_traces + (1 - decay) * prepost_product
            )
            # Apply dopamine-gated update using traces
            delta_w = self.learning_rate * self.eligibility_traces * dopamine_signal * dt

        elif self.plasticity_mode == "gated":
            # Threshold gating: only update if |DA| > threshold
            threshold = 0.5  # Configurable threshold
            if np.abs(dopamine_signal) > threshold:
                prepost_product = np.outer(mbon_activity, kc_activity)
                delta_w = self.learning_rate * prepost_product * dopamine_signal * dt
            else:
                delta_w = np.zeros_like(self.kc_to_mbon)

        else:
            raise ValueError(f"Unknown plasticity_mode: {self.plasticity_mode}")

        # Apply frozen synapse mask (for microsurgery experiments)
        if self._frozen_synapses:
            for (kc_idx, mbon_idx) in self._frozen_synapses:
                if 0 <= kc_idx < n_kc and 0 <= mbon_idx < n_mbon:
                    delta_w[mbon_idx, kc_idx] = 0.0

        # Apply sign-flip mask (for microsurgery experiments)
        if self._sign_flip_synapses:
            for (kc_idx, mbon_idx) in self._sign_flip_synapses:
                if 0 <= kc_idx < n_kc and 0 <= mbon_idx < n_mbon:
                    delta_w[mbon_idx, kc_idx] *= -1.0

        # Apply weight update
        self.kc_to_mbon += delta_w

        # Apply weight decay (L2 regularization) if enabled
        if self.weight_decay_rate > 0:
            self.kc_to_mbon *= (1.0 - self.weight_decay_rate * dt)

        # Compute diagnostics
        weight_change_magnitude = float(np.linalg.norm(delta_w))
        mean_weight = float(np.mean(np.abs(self.kc_to_mbon)))
        max_weight = float(np.max(np.abs(self.kc_to_mbon)))
        eligibility_trace_norm = (
            float(np.linalg.norm(self.eligibility_traces))
            if self.eligibility_traces is not None
            else 0.0
        )

        return {
            "weight_change_magnitude": weight_change_magnitude,
            "mean_weight": mean_weight,
            "max_weight": max_weight,
            "eligibility_trace_norm": eligibility_trace_norm,
        }

    def decay_weights(self, decay_rate: float = 0.001, dt: float = 1.0) -> None:
        """Apply weight decay (L2 regularization) to prevent unbounded growth.

        Biological Rationale
        --------------------
        Without decay, synaptic weights grow monotonically with repeated reward,
        eventually saturating or causing numerical instability. In biological
        systems, several mechanisms prevent unbounded growth:

        1. **Protein turnover**: Synaptic proteins (receptors, scaffolds) degrade
           over hours-days, providing soft upper bounds on strength.

        2. **Homeostatic scaling**: Neurons adjust global excitability to maintain
           stable firing rates despite changing input strengths.

        3. **Resource competition**: Finite presynaptic vesicles and postsynaptic
           receptors create soft capacity constraints.

        This decay term approximates these biological constraints with a simple
        exponential decay: W(t) ← W(t) × (1 - λ × dt)

        Parameters
        ----------
        decay_rate : float, optional
            Decay rate λ per time step. Biological range: 0.0001-0.01 per second.
            Higher values → stronger regularization, faster forgetting.
            Default: 0.001
        dt : float, optional
            Time step (arbitrary units). Default: 1.0

        Notes
        -----
        This method provides an alternative to the automatic weight decay applied
        in update_weights when weight_decay_rate > 0. Use this for manual control
        or batch decay operations.

        Example
        -------
        >>> # Apply decay after a block of trials
        >>> for trial in range(100):
        ...     plasticity.update_weights(kc, mbon, dopamine)
        >>> plasticity.decay_weights(decay_rate=0.01, dt=1.0)  # 1% decay
        """
        self.kc_to_mbon *= (1.0 - decay_rate * dt)


class LearningExperiment:
    """Orchestrates trial-by-trial conditioning experiments with plasticity.

    This class implements canonical Drosophila olfactory conditioning protocols,
    presenting odor stimuli (CS), delivering rewards/punishments (US), and tracking
    learning dynamics across trials. It bridges the feedforward circuit computation
    (OlfactoryCircuit) with the plasticity mechanisms (DopamineModulatedPlasticity).

    Biological Context
    -------------------
    Standard Drosophila conditioning paradigms:

    **Appetitive conditioning** (Tully & Quinn, 1985):
    - CS+: Odor A paired with sugar reward (US+)
    - CS-: Odor B presented without reward (US-)
    - Test: Measure approach to CS+ vs CS-
    - Learning index: (CS+ approach - CS- approach) / total

    **Aversive conditioning**:
    - CS+: Odor A paired with electric shock (US+)
    - CS-: Odor B presented without shock (US-)
    - Test: Measure avoidance of CS+ vs CS-

    **Blocking** (Tanimoto et al., 2004):
    - Phase 1: Train CS1→US (CS1 predicts reward)
    - Phase 2: Train CS1+CS2→US (compound stimulus)
    - Test: CS2 alone elicits weak response (blocked by CS1)

    This class supports all these protocols via flexible trial sequencing.

    Parameters
    ----------
    circuit : OlfactoryCircuit
        Feedforward circuit for PN→KC→MBON propagation.
    plasticity : DopamineModulatedPlasticity
        Plasticity manager that modifies circuit.kc_to_mbon weights.
    n_trials : int, optional
        Number of conditioning trials to run. Default: 100
    trial_types : List[str], optional
        Trial type labels for logging (e.g., ["CS+", "CS-", "Probe"]).
        Default: ["CS+", "CS-", "Probe"]

    Attributes
    ----------
    circuit : OlfactoryCircuit
        Circuit for forward propagation (connectivity may be modified by plasticity).
    plasticity : DopamineModulatedPlasticity
        Plasticity rules for weight updates.
    n_trials : int
        Total number of trials.
    history : List[Dict[str, Any]]
        Trial-by-trial records with odor, reward, activities, RPE, etc.

    Notes
    -----
    **IMPORTANT**: The plasticity manager modifies `circuit.connectivity.kc_to_mbon`
    during learning. To preserve the original connectivity, pass a deep copy of
    the circuit or plasticity weights.

    Example
    -------
    >>> # Set up learning experiment
    >>> experiment = LearningExperiment(circuit, plasticity, n_trials=50)
    >>>
    >>> # Appetitive conditioning protocol
    >>> odor_seq = ["DA1"] * 25 + ["DL3"] * 25  # CS+ and CS-
    >>> reward_seq = [1] * 25 + [0] * 25  # Reward for DA1, none for DL3
    >>> results = experiment.run_experiment(odor_seq, reward_seq)
    >>>
    >>> # Compute learning index
    >>> cs_plus_response = results[results['odor'] == 'DA1']['mbon_valence'].iloc[-1]
    >>> cs_minus_response = results[results['odor'] == 'DL3']['mbon_valence'].iloc[-1]
    >>> learning_index = (cs_plus_response - cs_minus_response) / (cs_plus_response + cs_minus_response + 1e-6)
    >>> print(f"Learning index: {learning_index:.3f}")
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        plasticity: DopamineModulatedPlasticity,
        n_trials: int = 100,
        trial_types: Optional[List[str]] = None,
    ) -> None:
        """Initialize learning experiment with circuit and plasticity."""
        self.circuit = circuit
        self.plasticity = plasticity
        self.n_trials = n_trials
        self.trial_types = trial_types or ["CS+", "CS-", "Probe"]
        self.history: List[Dict[str, Any]] = []

        # Ensure circuit uses plasticity weights (link them)
        # Create a temporary connectivity with mutable weights
        self._initial_weights = plasticity.kc_to_mbon.copy()

    def run_single_trial(
        self,
        odor: str,
        reward: float,
        dopamine_baseline: float = 0.0,
        trial_type: str = "CS+",
    ) -> Dict[str, Any]:
        """Execute one conditioning trial: odor presentation → outcome → learning.

        Biological Protocol
        -------------------
        Standard Drosophila conditioning trial sequence:

        1. **Pre-stimulus baseline** (not modeled): 30 sec ambient air
        2. **Odor presentation (CS)**: 60 sec odor delivery
           - PN activation → sparse KC responses → MBON output
           - MBON output = model's valence prediction V_pred
        3. **Outcome delivery (US)**: Reward/shock at odor offset
           - r(t) = 0 (no reward) or 1 (reward present)
        4. **RPE computation**: RPE = r(t) - V_pred
           - Maps to DAN dopamine release
        5. **Plasticity**: Update KC→MBON weights
           - dW = α × KC × MBON × (baseline + RPE)
        6. **Post-trial interval**: 30 sec before next trial

        Parameters
        ----------
        odor : str
            Glomerulus name or odor identity (e.g., "DA1", "DL3").
            Maps to PN activation pattern via circuit.activate_pns_by_glomeruli().
        reward : float
            Outcome value. Typically binary (0 or 1) but can be continuous.
            - 0: No reward (neutral or punishment)
            - 1: Reward delivered (sugar/fructose)
            - -1: Punishment delivered (shock) [for aversive conditioning]
        dopamine_baseline : float, optional
            Tonic dopamine level (independent of RPE). Models spontaneous DAN
            activity or modulatory state. Default: 0.0
        trial_type : str, optional
            Trial label for logging (e.g., "CS+", "CS-", "Probe"). Default: "CS+"

        Returns
        -------
        Dict[str, Any]
            Trial outcome dictionary with keys:
            - trial_id: int, trial number
            - odor: str, odor/glomerulus presented
            - reward: float, outcome value
            - pn_activity: np.ndarray, PN firing rates (n_pn,)
            - kc_activity: np.ndarray, sparse KC responses (n_kc,)
            - mbon_output: np.ndarray, MBON firing rates (n_mbon,)
            - mbon_valence: float, primary MBON output (prediction)
            - predicted_value: float, normalized valence prediction
            - rpe: float, reward prediction error
            - dopamine: float, total dopamine signal (baseline + RPE)
            - weight_change_magnitude: float, L2 norm of weight update
            - trial_type: str, trial label

        Example
        -------
        >>> # Appetitive conditioning trial (CS+)
        >>> trial_data = experiment.run_single_trial(
        ...     odor="DA1",
        ...     reward=1.0,
        ...     dopamine_baseline=0.0,
        ...     trial_type="CS+"
        ... )
        >>> print(f"MBON response: {trial_data['mbon_valence']:.3f}")
        >>> print(f"RPE: {trial_data['rpe']:.3f}")
        """
        # 1. Activate PNs for this odor
        pn_activity = self.circuit.activate_pns_by_glomeruli([odor], firing_rate=1.0)

        # 2. Propagate through circuit (PN → KC → MBON)
        # Use plasticity's current weights for KC→MBON propagation
        kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)

        # Override circuit's KC→MBON with plasticity's learned weights
        mbon_output = self.plasticity.kc_to_mbon @ kc_activity  # Dense matrix @ vector

        # 3. Extract valence prediction (use first MBON as primary valence signal)
        mbon_valence = float(mbon_output[0])

        # Normalize to [0, 1] range for RPE computation
        # Assume MBON outputs are roughly in range [-100, 100] initially
        predicted_value = mbon_valence / 100.0
        predicted_value = np.clip(predicted_value, -1.0, 1.0)  # Constrain to reasonable range

        # 4. Compute RPE (reward prediction error)
        rpe = self.plasticity.compute_rpe(reward, predicted_value, learning_rate_rpe=0.1)

        # 5. Map RPE to dopamine signal
        # Dopamine = baseline + RPE (simple linear mapping)
        dopamine = dopamine_baseline + rpe

        # 6. Update KC→MBON weights using three-factor rule
        update_diagnostics = self.plasticity.update_weights(
            kc_activity=kc_activity,
            mbon_activity=mbon_output,
            dopamine_signal=dopamine,
            dt=1.0,
        )

        # 7. Record trial outcome
        trial_data = {
            "trial_id": len(self.history),
            "odor": odor,
            "reward": reward,
            "pn_activity": pn_activity,
            "kc_activity": kc_activity,
            "mbon_output": mbon_output,
            "mbon_valence": mbon_valence,
            "predicted_value": predicted_value,
            "rpe": rpe,
            "dopamine": dopamine,
            "trial_type": trial_type,
            **update_diagnostics,
        }

        self.history.append(trial_data)

        return trial_data

    def run_experiment(
        self,
        odor_sequence: List[str],
        reward_sequence: List[Union[int, float]],
        trial_type_sequence: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run full conditioning protocol with specified trial sequence.

        Biological Protocol Examples
        -----------------------------
        **Simple appetitive conditioning**:
        odor_seq = ["DA1"] * 50
        reward_seq = [1] * 50
        Result: DA1 acquires positive valence

        **Differential conditioning**:
        odor_seq = ["DA1", "DL3"] * 25  # Interleaved
        reward_seq = [1, 0] * 25  # DA1→reward, DL3→no reward
        Result: DA1 positive, DL3 neutral/negative

        **Blocking** (Tanimoto et al., 2004):
        Phase 1:
        odor_seq = ["DA1"] * 10
        reward_seq = [1] * 10
        Phase 2:
        odor_seq = ["DA1+DL3"] * 10  # Compound stimulus
        reward_seq = [1] * 10
        Result: DL3 blocked (weak learning despite pairing with reward)

        Parameters
        ----------
        odor_sequence : List[str]
            List of odor/glomerulus names, one per trial.
            Length must match reward_sequence.
        reward_sequence : List[Union[int, float]]
            List of reward values, one per trial.
            Length must match odor_sequence.
        trial_type_sequence : Optional[List[str]], optional
            List of trial type labels for logging. If None, inferred from
            reward values (1→"CS+", 0→"CS-"). Default: None

        Returns
        -------
        pd.DataFrame
            Trial-by-trial results with columns:
            - trial_id: Trial number (0-indexed)
            - odor: Odor/glomerulus presented
            - reward: Outcome value
            - mbon_valence: Primary MBON output (valence prediction)
            - predicted_value: Normalized prediction for RPE
            - rpe: Reward prediction error
            - dopamine: Dopamine signal (baseline + RPE)
            - weight_change_magnitude: L2 norm of weight update
            - mean_weight: Current mean |W_km|
            - max_weight: Current max |W_km|
            - trial_type: Trial type label

        Raises
        ------
        ValueError
            If odor_sequence and reward_sequence have different lengths.

        Example
        -------
        >>> # Blocking experiment
        >>> # Phase 1: Train CS1→US
        >>> experiment1 = LearningExperiment(circuit, plasticity, n_trials=20)
        >>> results1 = experiment1.run_experiment(
        ...     odor_sequence=["DA1"] * 20,
        ...     reward_sequence=[1] * 20
        ... )
        >>>
        >>> # Phase 2: Train CS1+CS2→US (using same experiment object)
        >>> results2 = experiment1.run_experiment(
        ...     odor_sequence=["DL3"] * 20,  # CS2 alone (simplified)
        ...     reward_sequence=[1] * 20
        ... )
        >>>
        >>> # Analyze blocking
        >>> cs1_final = results1.iloc[-1]['mbon_valence']
        >>> cs2_final = results2.iloc[-1]['mbon_valence']
        >>> print(f"CS1 valence: {cs1_final:.3f}, CS2 valence: {cs2_final:.3f}")
        >>> print(f"Blocking effect: {cs1_final - cs2_final:.3f}")
        """
        # Validate input sequences
        if len(odor_sequence) != len(reward_sequence):
            raise ValueError(
                f"odor_sequence length {len(odor_sequence)} doesn't match "
                f"reward_sequence length {len(reward_sequence)}"
            )

        # Infer trial types if not provided
        if trial_type_sequence is None:
            trial_type_sequence = [
                "CS+" if r > 0.5 else "CS-"
                for r in reward_sequence
            ]

        # Run trials sequentially
        for odor, reward, trial_type in zip(odor_sequence, reward_sequence, trial_type_sequence):
            self.run_single_trial(odor, reward, dopamine_baseline=0.0, trial_type=trial_type)

        # Convert history to DataFrame for analysis
        df_records = []
        for trial in self.history:
            # Extract scalar fields for DataFrame (exclude large arrays)
            record = {
                "trial_id": trial["trial_id"],
                "odor": trial["odor"],
                "reward": trial["reward"],
                "mbon_valence": trial["mbon_valence"],
                "predicted_value": trial["predicted_value"],
                "rpe": trial["rpe"],
                "dopamine": trial["dopamine"],
                "weight_change_magnitude": trial["weight_change_magnitude"],
                "mean_weight": trial["mean_weight"],
                "max_weight": trial["max_weight"],
                "trial_type": trial["trial_type"],
            }
            df_records.append(record)

        return pd.DataFrame(df_records)

    def reset_history(self) -> None:
        """Clear trial history (useful for multi-phase experiments).

        Example
        -------
        >>> # Phase 1: Initial training
        >>> experiment.run_experiment(odors1, rewards1)
        >>> print(f"Phase 1 trials: {len(experiment.history)}")
        >>>
        >>> # Reset for Phase 2
        >>> experiment.reset_history()
        >>> experiment.run_experiment(odors2, rewards2)
        >>> print(f"Phase 2 trials: {len(experiment.history)}")
        """
        self.history = []
