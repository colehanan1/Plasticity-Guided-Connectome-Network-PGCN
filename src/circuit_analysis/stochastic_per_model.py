#!/usr/bin/env python3
"""Stochastic Monte Carlo model for PER probability prediction.

This module implements a biologically realistic stochastic model for predicting
proboscis extension reflex (PER) probability in Drosophila, incorporating:

1. Probabilistic synaptic transmission (binomial release)
2. Sparse KC activation (winner-take-all competition)
3. Temporal integration windows
4. Motor threshold nonlinearity

Biological Context
------------------
The model addresses the limitation of deterministic sigmoid models that
overpredict PER probability (99.3% vs. observed ~50%). Stochasticity arises
from:
- Variable neurotransmitter release (Kazama & Wilson, 2009)
- Sparse KC coding via APL inhibition (Honegger et al., 2011)
- Trial-to-trial timing variability
- Motor threshold nonlinearity (Hige et al., 2015)

References
----------
- Kazama & Wilson (2009). J Neurosci. ORN→PN transmission reliability
- Honegger et al. (2011). Nature. Sparse KC coding
- Hige et al. (2015). eLife. KC→MBON integration thresholds

Author: Claude Code (Anthropic)
Date: 2025-12-01
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


class StochasticPERModel:
    """Monte Carlo simulator for PER probability with stochastic synapses.

    This class implements a multi-layer spiking neural network model with
    probabilistic synaptic transmission at each stage. The model uses Monte
    Carlo simulation to estimate PER probability across many trials.

    Parameters
    ----------
    W_orn_pn : np.ndarray
        ORN→PN connectivity matrix (n_pn × n_orn)
        Values represent synapse counts
    W_pn_kc : np.ndarray
        PN→KC connectivity matrix (n_kc × n_pn)
        Values represent synapse counts
    W_kc_mbon : np.ndarray
        KC→MBON connectivity matrix (n_mbon × n_kc)
        Values represent synapse counts
    kc_mapping : np.ndarray, optional
        Mapping from PN→KC KC indices to KC→MBON KC indices
    W_veto_to_pn : np.ndarray, optional
        Or7a→PN lateral inhibition matrix (n_pn × n_or7a)
        Models aversive veto gate (Or7a suppresses appetitive PNs)
    veto_strength : float, default=0.5
        Veto gate inhibition gain (0-1), default 0.5
        0 = no inhibition, 1 = complete suppression
    release_prob_orn_pn : float, default=0.10
        Synaptic release probability for ORN→PN connections
        Literature: 0.15-0.30 (Kazama & Wilson, 2009)
        Calibrated: 0.10 (slightly below literature to match PER data)
    release_prob_pn_kc : float, default=0.25
        Synaptic release probability for PN→KC connections
    kc_sparsity : float, default=0.05
        Fraction of KCs active via k-WTA competition (5%)
        Literature: 5-10% (Honegger et al., 2011)
    mbon_threshold : float, default=10000.0
        Minimum summed MBON activity for PER trigger
        Calibrated to match behavioral data (was 15.0)
    temporal_window_ms : float, default=200.0
        Integration window for PER decision (milliseconds)
    n_trials : int, default=1000
        Number of Monte Carlo trials per prediction
    random_seed : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    rng : np.random.Generator
        Random number generator for stochastic simulation

    Notes
    -----
    The model assumes:
    - Independent binomial release at each synapse
    - Global competition for KC sparsity (APL-mediated)
    - Deterministic KC→MBON transmission (short-term plasticity ignored)
    - Threshold-linear activation functions

    Examples
    --------
    >>> model = StochasticPERModel(W_orn_pn, W_pn_kc, W_kc_mbon)
    >>> orn_activity = np.array([0.82, 0.72, 0.53, ...])  # DoOR responses
    >>> result = model.predict_per_probability(orn_activity)
    >>> print(f"PER probability: {result['per_probability']:.1%}")
    """

    def __init__(
        self,
        W_orn_pn: np.ndarray,
        W_pn_kc: np.ndarray,
        W_kc_mbon: np.ndarray,
        kc_mapping: Optional[np.ndarray] = None,
        W_veto_to_pn: Optional[np.ndarray] = None,  # Or7a→PN inhibition matrix
        veto_strength: float = 0.5,  # Veto gate inhibition gain (0-1)
        release_prob_orn_pn: float = 0.10,  # Calibrated: 0.10 (was 0.20)
        release_prob_pn_kc: float = 0.25,    # Optimal at literature value
        kc_sparsity: float = 0.05,           # Optimal at literature value
        mbon_threshold: float = 10000.0,     # Calibrated: 10,000 (was 15.0)
        temporal_window_ms: float = 200.0,
        n_trials: int = 1000,
        random_seed: Optional[int] = None,
    ):
        """Initialize stochastic PER model."""
        # Store connectivity matrices
        # Matrices are saved in (pre, post) format but we need (post, pre) for matmul
        # Always transpose to convert (pre, post) → (post, pre)
        self.W_orn_pn = W_orn_pn.T  # (ORN, PN) → (PN, ORN)
        self.W_pn_kc = W_pn_kc.T  # (PN, KC) → (KC, PN)
        self.W_kc_mbon = W_kc_mbon.T  # (KC, MBON) → (MBON, KC)
        self.kc_mapping = kc_mapping

        # Veto gate (Or7a lateral inhibition)
        if W_veto_to_pn is not None:
            self.W_veto_to_pn = W_veto_to_pn.T  # (Or7a, PN) → (PN, Or7a)
        else:
            self.W_veto_to_pn = None
        self.veto_strength = veto_strength

        # Model parameters
        self.release_prob_orn_pn = release_prob_orn_pn
        self.release_prob_pn_kc = release_prob_pn_kc
        self.kc_sparsity = kc_sparsity
        self.mbon_threshold = mbon_threshold
        self.temporal_window_ms = temporal_window_ms
        self.n_trials = n_trials

        # Random number generator
        self.rng = np.random.default_rng(random_seed)

        logger.info(f"Initialized StochasticPERModel:")
        logger.info(
            f"  Circuit: {self.W_orn_pn.shape[1]} ORNs → {self.W_orn_pn.shape[0]} PNs → "
            f"{self.W_pn_kc.shape[0]} KCs → {self.W_kc_mbon.shape[0]} MBONs"
        )
        if self.W_veto_to_pn is not None:
            logger.info(
                f"  Veto gate: {self.W_veto_to_pn.shape[1]} Or7a ORNs → "
                f"{self.W_veto_to_pn.shape[0]} PNs (strength={veto_strength:.2f})"
            )
        logger.info(
            f"  Release probs: ORN→PN={release_prob_orn_pn:.2f}, "
            f"PN→KC={release_prob_pn_kc:.2f}"
        )
        logger.info(f"  KC sparsity: {kc_sparsity:.1%}")
        logger.info(f"  MBON threshold: {mbon_threshold:.1f}")
        logger.info(f"  Trials: {n_trials}")

    def apply_kwta_sparsity(self, activity: np.ndarray, k: float) -> np.ndarray:
        """Apply k-winner-take-all (k-WTA) sparsity constraint.

        Implements global competition via lateral inhibition (APL-mediated).
        Only the top k% of neurons remain active.

        Parameters
        ----------
        activity : np.ndarray
            Neural activity vector
        k : float
            Fraction of neurons to keep active (0-1)

        Returns
        -------
        np.ndarray
            Sparse activity with only top-k% neurons active

        Notes
        -----
        This models the effect of APL neurons providing global gain control
        through feedforward and feedback inhibition to KCs.
        """
        if len(activity) == 0:
            return activity

        n_active = max(1, int(k * len(activity)))
        threshold = np.partition(activity, -n_active)[-n_active]
        sparse_activity = np.where(activity >= threshold, activity, 0)
        return sparse_activity

    def simulate_trial(
        self,
        orn_activity: np.ndarray,
        veto_orn_activity: Optional[np.ndarray] = None,
        include_temporal_jitter: bool = True,
    ) -> Dict[str, Any]:
        """Simulate a single PER trial with stochastic transmission.

        Parameters
        ----------
        orn_activity : np.ndarray
            ORN firing rates (normalized 0-1), typically from DoOR responses
        veto_orn_activity : np.ndarray, optional
            Or7a ORN activity for veto gate (lateral inhibition)
            If provided and W_veto_to_pn is set, applies multiplicative inhibition
        include_temporal_jitter : bool, default=True
            Add trial-to-trial timing variability (±20%)

        Returns
        -------
        Dict[str, Any]
            Trial results containing:
            - pn_activity: PN firing rates
            - kc_activity: KC firing rates (sparse)
            - mbon_activity: MBON firing rates
            - total_mbon: Summed MBON output
            - per_triggered: Binary PER outcome (0 or 1)
            - veto_signal: (optional) Or7a inhibition signal to PNs

        Notes
        -----
        Simulation proceeds in four stages:
        1. ORN→PN: Binomial release at each synapse
        2. Veto gate: Or7a lateral inhibition (multiplicative)
        3. PN→KC: Binomial release + k-WTA sparsity
        4. KC→MBON: Deterministic summation + threshold
        """
        # --- Layer 1: ORN → PN (stochastic transmission) ---
        pn_input = np.zeros(self.W_orn_pn.shape[0])

        for pn_idx in range(self.W_orn_pn.shape[0]):
            for orn_idx in range(self.W_orn_pn.shape[1]):
                n_synapses = self.W_orn_pn[pn_idx, orn_idx]
                if n_synapses > 0 and orn_activity[orn_idx] > 0:
                    # Binomial: n_synapses trials, each fires with release_prob
                    n_released = self.rng.binomial(
                        int(n_synapses),
                        self.release_prob_orn_pn * orn_activity[orn_idx],
                    )
                    pn_input[pn_idx] += n_released

        # --- Veto Gate: Or7a lateral inhibition (multiplicative) ---
        veto_signal = None
        if veto_orn_activity is not None and self.W_veto_to_pn is not None:
            # Compute Or7a inhibition signal to each PN
            veto_signal = self.W_veto_to_pn @ veto_orn_activity

            # DO NOT normalize - signal should scale with Or7a activity
            # Normalization destroys dose-response relationship!

            # Multiplicative inhibition: pn_input *= (1 - strength * signal)
            # Signal is already scaled by connectivity weights and Or7a activity
            inhibition_factor = 1.0 - (self.veto_strength * veto_signal)
            inhibition_factor = np.clip(inhibition_factor, 0, 1)
            pn_input = pn_input * inhibition_factor

        # PN activation (threshold-linear)
        pn_activity = np.maximum(0, pn_input)
        if pn_activity.max() > 0:
            pn_activity = pn_activity / pn_activity.max()  # Normalize

        # --- Layer 2: PN → KC (stochastic transmission + sparsity) ---
        kc_input = np.zeros(self.W_pn_kc.shape[0])

        for kc_idx in range(self.W_pn_kc.shape[0]):
            for pn_idx in range(self.W_pn_kc.shape[1]):
                n_synapses = self.W_pn_kc[kc_idx, pn_idx]
                if n_synapses > 0 and pn_activity[pn_idx] > 0:
                    n_released = self.rng.binomial(
                        int(n_synapses),
                        self.release_prob_pn_kc * pn_activity[pn_idx],
                    )
                    kc_input[kc_idx] += n_released

        # Apply k-WTA sparsity (APL-mediated global inhibition)
        kc_activity = self.apply_kwta_sparsity(kc_input, self.kc_sparsity)

        # --- Layer 3: KC → MBON (deterministic summation) ---
        # Map KC activity from PN→KC space to KC→MBON space
        if self.kc_mapping is not None:
            # Create activity vector in KC→MBON space
            # After transpose, W_kc_mbon is (MBON, KC), so shape[1] = n_kcs
            kc_activity_mbon = np.zeros(self.W_kc_mbon.shape[1])
            valid_mask = self.kc_mapping >= 0
            kc_activity_mbon[self.kc_mapping[valid_mask]] = kc_activity[valid_mask]
        else:
            kc_activity_mbon = kc_activity

        # Assume KC→MBON transmission is reliable
        mbon_activity = self.W_kc_mbon @ kc_activity_mbon

        # --- Temporal integration (optional jitter) ---
        if include_temporal_jitter:
            # Random jitter in integration window (±20%)
            jitter_factor = self.rng.uniform(0.8, 1.2)
            mbon_activity *= jitter_factor

        # --- PER decision (threshold) ---
        total_mbon = np.sum(mbon_activity)
        per_triggered = int(total_mbon > self.mbon_threshold)

        result = {
            "pn_activity": pn_activity,
            "kc_activity": kc_activity,
            "mbon_activity": mbon_activity,
            "total_mbon": total_mbon,
            "per_triggered": per_triggered,
        }

        # Include veto signal if present
        if veto_signal is not None:
            result["veto_signal"] = veto_signal

        return result

    def predict_per_probability(
        self,
        orn_activity: np.ndarray,
        veto_orn_activity: Optional[np.ndarray] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Predict PER probability via Monte Carlo simulation.

        Parameters
        ----------
        orn_activity : np.ndarray
            ORN firing rates for target odorant (from DoOR data)
        veto_orn_activity : np.ndarray, optional
            Or7a ORN activity for veto gate (lateral inhibition)
        return_details : bool, default=False
            If True, return trial-by-trial data (warning: large memory usage)

        Returns
        -------
        Dict[str, Any]
            Prediction results containing:
            - per_probability: Fraction of trials with PER (0-1)
            - mean_mbon_activity: Average MBON response across trials
            - std_mbon_activity: Standard deviation of MBON responses
            - per_variance: Binomial variance = p(1-p)
            - n_trials: Number of Monte Carlo trials
            - trials: (optional) List of individual trial results
            - mean_veto_signal: (optional) Mean Or7a inhibition if veto active

        Notes
        -----
        The per_probability is the Monte Carlo estimate of the true PER
        probability. Variance decreases as 1/sqrt(n_trials).

        Examples
        --------
        >>> result = model.predict_per_probability(orn_activity, return_details=False)
        >>> print(f"PER: {result['per_probability']:.1%} ± "
        ...       f"{np.sqrt(result['per_variance']):.1%}")
        """
        logger.info(f"Running {self.n_trials} Monte Carlo trials...")
        if veto_orn_activity is not None:
            logger.info(f"  Or7a veto gate active (strength={self.veto_strength:.2f})")

        trials = []
        per_count = 0
        mbon_activities = []
        veto_signals = []

        for trial_idx in range(self.n_trials):
            result = self.simulate_trial(orn_activity, veto_orn_activity)
            trials.append(result)
            per_count += result["per_triggered"]
            mbon_activities.append(result["total_mbon"])
            if "veto_signal" in result:
                veto_signals.append(result["veto_signal"])

            if (trial_idx + 1) % 100 == 0:
                current_per = per_count / (trial_idx + 1)
                logger.debug(
                    f"  Trial {trial_idx + 1}/{self.n_trials}: "
                    f"PER rate = {current_per:.1%}"
                )

        per_prob = per_count / self.n_trials
        mean_mbon = np.mean(mbon_activities)
        std_mbon = np.std(mbon_activities)

        logger.info(f"Monte Carlo results:")
        logger.info(f"  PER probability: {per_prob:.1%}")
        logger.info(f"  Mean MBON activity: {mean_mbon:.2f} ± {std_mbon:.2f}")
        logger.info(f"  Threshold: {self.mbon_threshold:.2f}")

        result = {
            "per_probability": per_prob,
            "mean_mbon_activity": mean_mbon,
            "std_mbon_activity": std_mbon,
            "per_variance": per_prob * (1 - per_prob),  # Binomial variance
            "n_trials": self.n_trials,
        }

        # Include veto statistics if veto gate was active
        if len(veto_signals) > 0:
            mean_veto = np.mean([vs.mean() for vs in veto_signals])
            result["mean_veto_signal"] = mean_veto
            logger.info(f"  Mean veto signal: {mean_veto:.3f}")

        if return_details:
            result["trials"] = trials

        return result


def load_connectivity_matrices(cache_dir: str) -> Dict[str, np.ndarray]:
    """Load connectivity matrices from cache directory.

    Parameters
    ----------
    cache_dir : str
        Path to cache directory containing connectivity_matrices/ folder

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys 'W_orn_pn', 'W_pn_kc', 'W_kc_mbon'

    Raises
    ------
    FileNotFoundError
        If connectivity matrices not found

    Examples
    --------
    >>> matrices = load_connectivity_matrices('data/cache/ethyl_butyrate_circuit')
    >>> print(matrices['W_orn_pn'].shape)
    """
    from pathlib import Path

    cache_path = Path(cache_dir)
    matrices_dir = cache_path / "connectivity_matrices"

    if not matrices_dir.exists():
        raise FileNotFoundError(
            f"Connectivity matrices directory not found: {matrices_dir}"
        )

    logger.info(f"Loading connectivity matrices from {matrices_dir}")

    W_orn_pn = sparse.load_npz(matrices_dir / "orn_to_pn.npz").toarray()
    W_pn_kc = sparse.load_npz(matrices_dir / "pn_to_kc.npz").toarray()
    W_kc_mbon = sparse.load_npz(matrices_dir / "kc_to_mbon.npz").toarray()

    # Load index mappings to align KC spaces
    import json

    with open(matrices_dir / "pn_to_kc_indices.json") as f:
        pn_kc_indices = json.load(f)
    with open(matrices_dir / "kc_to_mbon_indices.json") as f:
        kc_mbon_indices = json.load(f)

    # Create mapping from PN→KC KC indices to KC→MBON KC indices
    pn_kc_kc_ids = pn_kc_indices["post_ids"]  # KCs in PN→KC matrix (columns)
    kc_mbon_kc_ids = kc_mbon_indices["pre_ids"]  # KCs in KC→MBON matrix (rows)

    # Build index mapping: for each KC in PN→KC, find its index in KC→MBON
    kc_mbon_id_to_idx = {kc_id: idx for idx, kc_id in enumerate(kc_mbon_kc_ids)}
    kc_mapping = np.array(
        [kc_mbon_id_to_idx.get(kc_id, -1) for kc_id in pn_kc_kc_ids]
    )

    n_mapped = (kc_mapping >= 0).sum()

    logger.info(f"  ORN→PN: {W_orn_pn.shape}")
    logger.info(f"  PN→KC: {W_pn_kc.shape}")
    logger.info(f"  KC→MBON: {W_kc_mbon.shape}")
    logger.info(f"  KC mapping: {n_mapped}/{len(kc_mapping)} KCs project to MBONs")

    return {
        "W_orn_pn": W_orn_pn,
        "W_pn_kc": W_pn_kc,
        "W_kc_mbon": W_kc_mbon,
        "kc_mapping": kc_mapping,
    }
