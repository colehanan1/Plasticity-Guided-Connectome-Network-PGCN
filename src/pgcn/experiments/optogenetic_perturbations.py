"""Optogenetic perturbations for causal circuit manipulation experiments.

This module simulates optogenetic manipulations (silencing, activation, temporal control)
of specific neuron populations during learning trials. These virtual experiments test
causal hypotheses about which circuit elements are necessary for learning by perturbing
activity and measuring the resulting behavioral deficits.

Biological Context
------------------
Optogenetics enables millisecond-precision control of genetically-specified neurons:

1. **CsChrimson activation**: Red light drives depolarization of neurons expressing
   UAS-CsChrimson, forcing spiking activity. Used to test sufficiency (can forced
   activation drive learning even without natural odor input?).

2. **Kir2.1 silencing**: Expression of inward-rectifying K+ channels hyperpolarizes
   neurons, preventing spiking. Used to test necessity (does silencing prevent learning?).

3. **Temporal precision**: Light can be delivered at specific phases of a trial
   (e.g., during odor presentation, during reward delivery, or during inter-trial
   intervals) to dissect when a neuron's activity is critical.

Experimental Paradigms Enabled
------------------------------
**Exp 1: Glomerulus-specific silencing**
- Protocol: Silence DA1 PNs during CS+US pairing
- Prediction: Learning deficit for DA1 odor, but not DL3 odor
- Tests: Is DA1 input causally required for learning DA1 associations?

**Exp 2: KC population silencing**
- Protocol: Silence α/β KCs during conditioning
- Prediction: Impaired long-term memory formation (α/β → LTM)
- Tests: KC subtype specialization for memory timescales

**Exp 3: MBON readout blocking**
- Protocol: Silence MBON-γ1pedc>α/β during test phase
- Prediction: Impaired behavioral expression of learned memory
- Tests: Is this MBON necessary for memory retrieval?

**Exp 4: DAN teaching signal manipulation**
- Protocol: Activate PAM-β'2a DANs artificially during odor presentation
- Prediction: Induce appetitive learning even without natural reward
- Tests: Sufficiency of dopamine for learning

Example
-------
>>> from data_loaders.circuit_loader import CircuitLoader
>>> from pgcn.models.olfactory_circuit import OlfactoryCircuit
>>> from pgcn.models.learning_model import DopamineModulatedPlasticity, LearningExperiment
>>> from pgcn.experiments.optogenetic_perturbations import OptogeneticPerturbation
>>>
>>> # Load circuit
>>> loader = CircuitLoader(cache_dir="data/cache")
>>> conn = loader.load_connectivity_matrix(normalize_weights="row")
>>> circuit = OlfactoryCircuit(conn, kc_sparsity_target=0.05)
>>>
>>> # Create perturbation: silence DA1 PNs during odor presentation
>>> opto = OptogeneticPerturbation(
...     circuit=circuit,
...     perturbation_type="silence",
...     target_neurons="pn",
...     target_specificity=["DA1"],
...     temporal_window=(0.0, 1.0),  # Full trial duration
...     efficacy=1.0  # Complete silencing
... )
>>>
>>> # Run learning trial with perturbation
>>> plasticity = DopamineModulatedPlasticity(conn.kc_to_mbon.toarray(), learning_rate=0.01)
>>> experiment = LearningExperiment(circuit, plasticity, n_trials=10)
>>> trial_result = opto.run_learning_trial_with_opto(experiment, "DA1", reward=1)
>>> print(f"PN activity with silencing: {trial_result['pn_activity_perturbed']:.1f}")
>>> print(f"KC activity with silencing: {trial_result['kc_activity_perturbed']:.1f}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pgcn.models.olfactory_circuit import OlfactoryCircuit


class OptogeneticPerturbation:
    """Simulate optogenetic manipulation of specific neurons during trials.

    This class models the effects of optogenetic activation or silencing on neural
    activity during learning experiments. Perturbations can be targeted to specific
    neuron populations (PNs, KCs, MBONs, DANs) and restricted to temporal windows
    within a trial.

    Biological Design Principles
    -----------------------------
    1. **Specificity**: Perturbations can target specific glomeruli (e.g., DA1) or
       KC subtypes (e.g., α/β). This models genetic targeting via GAL4 driver lines.

    2. **Efficacy**: Real optogenetic manipulations rarely achieve 100% silencing or
       activation. The efficacy parameter (0-1) models incomplete penetrance.

    3. **Temporal control**: Light can be restricted to specific trial phases
       (e.g., odor presentation [0, 1000ms] or reward delivery [1000, 1500ms]).

    4. **Reversibility**: Optogenetic effects are immediate and reversible, unlike
       genetic ablations. Activity returns to baseline when light is off.

    Parameters
    ----------
    circuit : OlfactoryCircuit
        The olfactory circuit to perturb. Used to identify neuron indices and
        propagate perturbed activity.
    perturbation_type : str
        Type of optogenetic manipulation:
        - "silence": Hyperpolarize neurons (scale activity toward zero)
        - "activate": Depolarize neurons (force high activity)
        - "holdover": No change (control condition)
    target_neurons : str
        Which population to perturb: "pn", "kc", "mbon", or "dan"
    target_specificity : Optional[List[str]]
        Restrict perturbation to subset of target population:
        - For PNs: glomerulus names (e.g., ["DA1", "DL3"])
        - For KCs: subtype names (e.g., ["ab", "g_main"])
        - For MBONs/DANs: neuron type labels
        If None, perturb entire population.
    temporal_window : Tuple[float, float]
        Time window during trial when perturbation is active, as fraction [0, 1].
        E.g., (0.0, 0.5) = first half of trial, (0.5, 1.0) = second half.
    efficacy : float
        Strength of perturbation (0 = no effect, 1 = complete silencing/activation).
        Models incomplete optogenetic penetrance.

    Attributes
    ----------
    _target_indices : np.ndarray
        Indices of neurons to perturb, computed during initialization.

    Examples
    --------
    Silence DA1 PNs completely:
    >>> opto = OptogeneticPerturbation(
    ...     circuit, perturbation_type="silence", target_neurons="pn",
    ...     target_specificity=["DA1"], efficacy=1.0
    ... )

    Partially activate γ KCs (70% efficacy):
    >>> opto = OptogeneticPerturbation(
    ...     circuit, perturbation_type="activate", target_neurons="kc",
    ...     target_specificity=["g_main", "g_dorsal"], efficacy=0.7
    ... )

    Silence MBONs during reward window only:
    >>> opto = OptogeneticPerturbation(
    ...     circuit, perturbation_type="silence", target_neurons="mbon",
    ...     temporal_window=(0.7, 1.0), efficacy=1.0
    ... )
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        perturbation_type: str = "silence",
        target_neurons: str = "pn",
        target_specificity: Optional[List[str]] = None,
        temporal_window: Tuple[float, float] = (0.0, 1.0),
        efficacy: float = 1.0,
    ) -> None:
        """Initialize optogenetic perturbation with target specification."""
        if perturbation_type not in ["silence", "activate", "holdover"]:
            raise ValueError(
                f"perturbation_type must be 'silence', 'activate', or 'holdover', "
                f"got '{perturbation_type}'"
            )

        if target_neurons not in ["pn", "kc", "mbon", "dan"]:
            raise ValueError(
                f"target_neurons must be 'pn', 'kc', 'mbon', or 'dan', "
                f"got '{target_neurons}'"
            )

        if not (0.0 <= temporal_window[0] < temporal_window[1] <= 1.0):
            raise ValueError(
                f"temporal_window must satisfy 0 <= start < end <= 1, "
                f"got {temporal_window}"
            )

        if not (0.0 <= efficacy <= 1.0):
            raise ValueError(f"efficacy must be in [0, 1], got {efficacy}")

        self.circuit = circuit
        self.perturbation_type = perturbation_type
        self.target_neurons = target_neurons
        self.target_specificity = target_specificity
        self.temporal_window = temporal_window
        self.efficacy = efficacy

        # Identify target neuron indices
        self._target_indices = self._identify_targets()

    def _identify_targets(self) -> np.ndarray:
        """Map target_specificity to neuron indices in the circuit.

        Returns
        -------
        np.ndarray
            1D array of integer indices specifying which neurons to perturb.

        Notes
        -----
        - For PNs: Uses glomerulus labels from connectivity.glomerulus_labels
        - For KCs: Uses subtype assignments from connectivity.kc_subtypes
        - For MBONs/DANs: Uses all indices (specificity filtering not yet implemented)
        """
        if self.target_neurons == "pn":
            if self.target_specificity:
                # Find PNs with matching glomerulus labels
                glom_labels = self.circuit.connectivity.pn_glomeruli
                indices = []
                for i, pn_id in enumerate(self.circuit.connectivity.pn_ids):
                    glom = glom_labels.get(pn_id, None)
                    if glom in self.target_specificity:
                        indices.append(i)
                return np.array(indices, dtype=int)
            else:
                # Target all PNs
                return np.arange(len(self.circuit.connectivity.pn_ids), dtype=int)

        elif self.target_neurons == "kc":
            if self.target_specificity:
                # Filter KCs by subtype
                kc_subtypes = self.circuit.connectivity.kc_subtypes
                indices = []
                for i, kc_id in enumerate(self.circuit.connectivity.kc_ids):
                    subtype = kc_subtypes.get(kc_id, None)
                    if subtype in self.target_specificity:
                        indices.append(i)
                return np.array(indices, dtype=int)
            else:
                # Target all KCs
                return np.arange(len(self.circuit.connectivity.kc_ids), dtype=int)

        elif self.target_neurons == "mbon":
            # Target all MBONs (specificity not yet implemented)
            return np.arange(len(self.circuit.connectivity.mbon_ids), dtype=int)

        else:  # "dan"
            # Target all DANs (specificity not yet implemented)
            return np.arange(len(self.circuit.connectivity.dan_ids), dtype=int)

    def apply_perturbation(
        self,
        activity: np.ndarray,
        trial_phase: float = 0.5,
    ) -> np.ndarray:
        """Apply optogenetic perturbation to neural activity vector.

        This method modifies the activity of target neurons based on the perturbation
        type and temporal window. It models the immediate effect of optogenetic
        stimulation on firing rates.

        Parameters
        ----------
        activity : np.ndarray
            Neural activity array (n_neurons,). Entries are firing rates or
            normalized activity levels.
        trial_phase : float
            Fraction through trial [0, 1]. Determines whether perturbation is active
            based on temporal_window.

        Returns
        -------
        np.ndarray
            Modified activity with perturbation applied. Original array unchanged.

        Notes
        -----
        - "silence": Scales activity toward zero by (1 - efficacy)
        - "activate": Sets activity to efficacy (forced high activity)
        - "holdover": No change (returns copy of input)

        If trial_phase is outside temporal_window, returns unmodified activity.

        Examples
        --------
        >>> pn_activity = np.ones(100)
        >>> opto = OptogeneticPerturbation(
        ...     circuit, perturbation_type="silence", target_neurons="pn",
        ...     target_specificity=["DA1"], efficacy=0.8
        ... )
        >>> perturbed = opto.apply_perturbation(pn_activity, trial_phase=0.5)
        >>> # DA1 PNs reduced to 20% of original activity
        """
        modified = activity.copy()

        # Check if perturbation is active during this trial phase
        if not (self.temporal_window[0] <= trial_phase <= self.temporal_window[1]):
            return modified

        if self.perturbation_type == "silence":
            # Hyperpolarize: scale activity toward zero
            modified[self._target_indices] *= (1.0 - self.efficacy)

        elif self.perturbation_type == "activate":
            # Depolarize: force high activity
            modified[self._target_indices] = self.efficacy

        elif self.perturbation_type == "holdover":
            # No change (control condition)
            pass

        return modified

    def run_learning_trial_with_opto(
        self,
        experiment: "LearningExperiment",
        odor: str,
        reward: int,
        trial_phase_opto: float = 0.5,
    ) -> Dict[str, Any]:
        """Run single learning trial with optogenetic perturbation.

        This method executes a full learning trial (odor presentation → KC activation
        → MBON output → plasticity update) while applying optogenetic perturbation
        at the specified trial phase.

        Parameters
        ----------
        experiment : LearningExperiment
            Learning experiment instance managing plasticity and trial execution.
        odor : str
            Glomerulus name to activate (e.g., "DA1"). Must be in connectivity.glomerulus_labels.
        reward : int
            Reward delivered on this trial (0 = no reward, 1 = reward). Used to compute
            reward prediction error (RPE) for dopamine signal.
        trial_phase_opto : float
            When during trial to apply perturbation [0, 1]. Used to determine if
            temporal_window is active.

        Returns
        -------
        Dict[str, Any]
            Trial diagnostics containing:
            - 'odor': Glomerulus presented
            - 'reward': Reward delivered (0 or 1)
            - 'pn_activity_perturbed': Sum of PN activity after perturbation
            - 'kc_activity_perturbed': Sum of KC activity after perturbation
            - 'mbon_output': MBON valence output (first MBON)
            - 'rpe': Reward prediction error
            - 'dopamine': Dopamine signal (RPE-driven)
            - 'perturbation_active': Whether perturbation was applied this trial

        Notes
        -----
        Perturbation is applied at different circuit stages depending on target_neurons:
        - "pn": Perturb PN input before KC propagation
        - "kc": Perturb KC activity after PN→KC propagation
        - "mbon": Perturb MBON output before plasticity update
        - "dan": Not yet implemented (would perturb dopamine signal)

        The learning update uses standard three-factor rule even with perturbation,
        allowing us to test how manipulations affect weight changes.

        Examples
        --------
        Test whether silencing DA1 PNs prevents learning:
        >>> opto = OptogeneticPerturbation(
        ...     circuit, perturbation_type="silence", target_neurons="pn",
        ...     target_specificity=["DA1"], efficacy=1.0
        ... )
        >>> results = []
        >>> for trial in range(20):
        ...     result = opto.run_learning_trial_with_opto(experiment, "DA1", reward=1)
        ...     results.append(result['mbon_output'])
        >>> # Expect flat learning curve (no plasticity due to zero PN→KC activity)
        """
        # Activate PNs for the odor
        pn_activity = self.circuit.activate_pns_by_glomeruli([odor], firing_rate=1.0)

        # Apply perturbation to PN activity if targeting PNs
        if self.target_neurons == "pn":
            pn_activity = self.apply_perturbation(pn_activity, trial_phase=trial_phase_opto)

        # Propagate to KC with sparse activation
        kc_activity = self.circuit.propagate_pn_to_kc(pn_activity)

        # Apply perturbation to KC activity if targeting KCs
        if self.target_neurons == "kc":
            kc_activity = self.apply_perturbation(kc_activity, trial_phase=trial_phase_opto)

        # Propagate to MBON
        mbon_output = self.circuit.propagate_kc_to_mbon(kc_activity)

        # Apply perturbation to MBON output if targeting MBONs
        if self.target_neurons == "mbon":
            mbon_output = self.apply_perturbation(mbon_output, trial_phase=trial_phase_opto)

        # Compute reward prediction error (RPE)
        # MBON output is raw sum; normalize to [0, 1] range for RPE computation
        predicted_value = mbon_output[0] / 100.0 if len(mbon_output) > 0 else 0.0
        rpe = experiment.plasticity.compute_rpe(reward, predicted_value)
        dopamine = rpe

        # Update weights via three-factor learning rule
        update_diagnostics = experiment.plasticity.update_weights(
            kc_activity, mbon_output, dopamine
        )

        return {
            'odor': odor,
            'reward': reward,
            'pn_activity_perturbed': float(pn_activity.sum()),
            'kc_activity_perturbed': float(kc_activity.sum()),
            'mbon_output': float(mbon_output[0]) if len(mbon_output) > 0 else 0.0,
            'rpe': float(rpe),
            'dopamine': float(dopamine),
            'perturbation_active': self.temporal_window[0] <= trial_phase_opto <= self.temporal_window[1],
            'weight_change_magnitude': float(update_diagnostics.get('weight_change_magnitude', 0.0)),
        }

    def run_full_experiment(
        self,
        experiment: "LearningExperiment",
        odor_sequence: List[str],
        reward_sequence: List[int],
    ) -> pd.DataFrame:
        """Run full learning experiment with optogenetic perturbation on every trial.

        Parameters
        ----------
        experiment : LearningExperiment
            Learning experiment instance managing plasticity.
        odor_sequence : List[str]
            Sequence of glomeruli to present on each trial (e.g., ["DA1", "DL3", "DA1", ...]).
        reward_sequence : List[int]
            Sequence of rewards (0 or 1) delivered on each trial.

        Returns
        -------
        pd.DataFrame
            Trial-by-trial results with columns: trial_id, odor, reward, pn_activity_perturbed,
            kc_activity_perturbed, mbon_output, rpe, dopamine, perturbation_active.

        Notes
        -----
        This method is a convenience wrapper for running multiple trials in sequence
        with the same perturbation applied to every trial. Compare to control condition
        (no perturbation) to measure learning deficit caused by manipulation.

        Examples
        --------
        Compare learning with and without PN silencing:
        >>> # Control: no perturbation
        >>> experiment_control = LearningExperiment(circuit, plasticity, n_trials=20)
        >>> control_results = experiment_control.run_experiment(
        ...     ["DA1"] * 20, [1] * 20
        ... )
        >>>
        >>> # Experimental: silence DA1 PNs
        >>> opto = OptogeneticPerturbation(
        ...     circuit, perturbation_type="silence", target_neurons="pn",
        ...     target_specificity=["DA1"], efficacy=1.0
        ... )
        >>> experiment_opto = LearningExperiment(circuit, plasticity_copy, n_trials=20)
        >>> opto_results = opto.run_full_experiment(
        ...     experiment_opto, ["DA1"] * 20, [1] * 20
        ... )
        >>>
        >>> # Compare final MBON output (learning curve endpoint)
        >>> control_final = control_results.iloc[-1]['mbon_valence']
        >>> opto_final = opto_results.iloc[-1]['mbon_output']
        >>> learning_deficit = (control_final - opto_final) / control_final
        >>> print(f"Learning deficit: {learning_deficit:.1%}")
        """
        if len(odor_sequence) != len(reward_sequence):
            raise ValueError(
                f"odor_sequence and reward_sequence must have same length, "
                f"got {len(odor_sequence)} and {len(reward_sequence)}"
            )

        results = []
        for trial_id, (odor, reward) in enumerate(zip(odor_sequence, reward_sequence)):
            # Apply perturbation at middle of trial (phase=0.5)
            trial_result = self.run_learning_trial_with_opto(
                experiment, odor, reward, trial_phase_opto=0.5
            )
            trial_result['trial_id'] = trial_id
            results.append(trial_result)

        return pd.DataFrame(results)
