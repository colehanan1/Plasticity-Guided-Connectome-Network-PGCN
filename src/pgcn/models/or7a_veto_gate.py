"""Or7a-Inspired Veto Gate Module for Continual Learning.

This module implements a biologically-inspired veto gate mechanism based on the
Or7a olfactory receptor pathway in Drosophila, which blocks benzaldehyde learning
despite reward pairing. The Or7a system provides a compact, graded pathway-specific
plasticity gate that enables continual learning with minimal catastrophic forgetting.

Biological Context
------------------
**Or7a Blocking Phenomenon** (Shen et al., 2025):

1. **Architecture**: 41 Or7a ORNs → 2 DL5 PNs → 312 KCs
   - Extreme bottleneck (2 PNs) controls plasticity in 312 downstream KCs
   - Efficient parameter protection with minimal overhead

2. **Graded Veto Mechanism**:
   - Benzaldehyde (Or7a 55% active) → strong veto → no learning
   - Hexanol (Or7a 14% active) → weak veto → partial learning
   - Cross-transfer: Training on benzaldehyde enables hexanol response
   - Not binary: veto strength scales with Or7a activation level

3. **Computational Advantages**:
   - Single pathway (2 neurons) gates entire KC population
   - Graded control allows similar stimuli to benefit from partial transfer
   - Biological safety mechanism (blocks toxic aldehyde associations)

ML Translation
--------------
The Or7a veto gate provides:
- **Pathway-specific plasticity gating**: Detect "Task A features" and block updates
- **Graded control**: Similar inputs get partial veto (enables transfer)
- **Efficient protection**: Minimal overhead (veto neuron detects, gates apply)
- **No rehearsal needed**: Unlike EWC/rehearsal methods, veto is forward-only

Example
-------
>>> from pgcn.models.olfactory_circuit import OlfactoryCircuit
>>> from pgcn.models.learning_model import DopamineModulatedPlasticity
>>> from pgcn.models.or7a_veto_gate import Or7aVetoGate
>>>
>>> # Initialize circuit and plasticity
>>> circuit = OlfactoryCircuit(conn_matrix, kc_sparsity_target=0.05)
>>> plasticity = DopamineModulatedPlasticity(weights, learning_rate=0.001)
>>>
>>> # Create Or7a veto gate for DL5 glomerulus
>>> veto_gate = Or7aVetoGate(
...     circuit=circuit,
...     or7a_glomerulus="DL5",
...     activation_threshold=0.3,
...     veto_strength=1.0
... )
>>>
>>> # Compute veto signal from PN activity
>>> pn_activity = circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.55)
>>> veto_signal = veto_gate.compute_veto_signal(pn_activity)
>>> print(f"Veto signal (55% activation): {veto_signal:.3f}")  # ~0.9 (strong)
>>>
>>> # Test graded veto on similar input
>>> pn_activity_weak = circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.14)
>>> veto_signal_weak = veto_gate.compute_veto_signal(pn_activity_weak)
>>> print(f"Veto signal (14% activation): {veto_signal_weak:.3f}")  # ~0.1 (weak)
>>>
>>> # Apply veto gating to weight updates
>>> delta_w = np.random.randn(96, 5177) * 0.001
>>> gated_delta_w = veto_gate.gate_plasticity(delta_w, veto_signal)
>>> print(f"Gating factor: {1.0 - veto_signal:.3f}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .connectivity_matrix import ConnectivityMatrix
from .olfactory_circuit import OlfactoryCircuit


class Or7aVetoGate:
    """Or7a-inspired graded veto gate for pathway-specific plasticity control.

    This class implements a biologically-grounded veto mechanism that detects
    "Task A features" (via Or7a glomerulus activation) and gates plasticity
    to prevent catastrophic forgetting during sequential task learning.

    Biological Rationale
    --------------------
    **Why Or7a as a veto gate?**

    1. **Graded control**: Or7a activation varies from 14% (hexanol) to 55%
       (benzaldehyde), providing a natural graded veto signal rather than
       binary on/off gating.

    2. **Bottleneck efficiency**: 41 ORNs → 2 PNs → 312 KCs. The 2-neuron
       bottleneck allows minimal veto infrastructure to control plasticity
       across hundreds of downstream neurons.

    3. **Safety mechanism**: In biology, Or7a blocks learning to toxic aldehydes.
       In ML, this translates to "protected features" that shouldn't be overwritten.

    4. **Cross-transfer**: Training on benzaldehyde (strong veto) enables response
       to hexanol (weak veto), demonstrating graded transfer rather than full blocking.

    **Mathematical Form**:

    Veto signal computation:
        v = σ((a - θ) × γ)

    where:
        a = Or7a glomerulus activation (weighted sum of Or7a PNs)
        θ = activation threshold (default: 0.3)
        γ = steepness parameter (default: 10)
        σ = sigmoid function

    Plasticity gating:
        dW_gated = dW × (1 - β × v)

    where:
        dW = ungated weight update
        β = veto strength (0-1, default: 1.0)
        v = veto signal [0, 1]

    Parameters
    ----------
    circuit : OlfactoryCircuit
        Feedforward circuit for PN activation.
    or7a_glomerulus : str
        Glomerulus acting as Or7a pathway (e.g., "DL5").
    activation_threshold : float, optional
        Activation threshold for veto engagement. Biological range: 0.14-0.55.
        Default: 0.3 (midpoint between hexanol 14% and benzaldehyde 55%).
    veto_strength : float, optional
        Gating strength β in (1 - β × v). Range: [0, 1].
        0 = no gating (normal learning)
        1 = full veto (complete block when v=1)
        Default: 1.0
    steepness : float, optional
        Sigmoid steepness γ for graded veto. Higher = sharper threshold.
        Default: 10.0 (smooth graded response)
    graded : bool, optional
        If True, use graded sigmoid veto. If False, use binary threshold.
        Default: True (biological)
    min_gating_factor : float, optional
        Minimum gating factor (floor). Prevents complete plasticity shutdown.
        Default: 0.1 (10% residual plasticity)

    Attributes
    ----------
    circuit : OlfactoryCircuit
        Reference to circuit for PN activation.
    or7a_glomerulus : str
        Or7a pathway glomerulus name.
    veto_weight : np.ndarray
        Weight vector for Or7a veto computation. Shape: (n_pn,).
        Uniformly weights PNs in or7a_glomerulus.
    activation_threshold : float
        Veto activation threshold.
    veto_strength : float
        Gating strength parameter.
    steepness : float
        Sigmoid steepness for graded veto.
    graded : bool
        Whether to use graded (True) or binary (False) veto.
    min_gating_factor : float
        Floor for gating factor (prevents full shutdown).
    veto_history : List[Dict[str, float]]
        History of veto activations for analysis.

    Raises
    ------
    ValueError
        If or7a_glomerulus not found in circuit connectivity.
        If activation_threshold not in [0, 1].
        If veto_strength not in [0, 1].

    Notes
    -----
    **Design choices**:

    1. **Graded vs binary veto**: Biological Or7a shows graded responses
       (55% vs 14%), so default is graded=True. Binary mode available for
       comparison experiments.

    2. **Threshold selection**: Default 0.3 is midpoint between hexanol (0.14)
       and benzaldehyde (0.55). Tune based on task similarity.

    3. **Gating floor**: min_gating_factor=0.1 ensures veto never completely
       freezes plasticity (biological neurons always have some plasticity).

    4. **No learnable parameters**: Veto weight is fixed (uniformly weights
       Or7a PNs). Future extensions could learn optimal veto weights.

    Example
    -------
    >>> # Standard usage
    >>> veto_gate = Or7aVetoGate(
    ...     circuit=circuit,
    ...     or7a_glomerulus="DL5",
    ...     activation_threshold=0.3,
    ...     graded=True
    ... )
    >>>
    >>> # Compute veto for strong Or7a activation (benzaldehyde-like)
    >>> pn_strong = circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.55)
    >>> veto_strong = veto_gate.compute_veto_signal(pn_strong)
    >>> print(f"Strong activation veto: {veto_strong:.3f}")  # ~0.9
    >>>
    >>> # Compute veto for weak Or7a activation (hexanol-like)
    >>> pn_weak = circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.14)
    >>> veto_weak = veto_gate.compute_veto_signal(pn_weak)
    >>> print(f"Weak activation veto: {veto_weak:.3f}")  # ~0.1
    >>>
    >>> # Gate plasticity update
    >>> delta_w = np.random.randn(96, 5177) * 0.001
    >>> gated_update = veto_gate.gate_plasticity(delta_w, veto_strong)
    >>> reduction = 1.0 - (np.linalg.norm(gated_update) / np.linalg.norm(delta_w))
    >>> print(f"Update reduced by: {reduction:.1%}")  # ~90% reduction
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        or7a_glomerulus: str,
        activation_threshold: float = 0.3,
        veto_strength: float = 1.0,
        steepness: float = 10.0,
        graded: bool = True,
        min_gating_factor: float = 0.1,
    ) -> None:
        """Initialize Or7a veto gate with circuit and parameters."""
        # Validate parameters
        if not (0.0 <= activation_threshold <= 1.0):
            raise ValueError(
                f"activation_threshold must be in [0, 1], got {activation_threshold}"
            )
        if not (0.0 <= veto_strength <= 1.0):
            raise ValueError(
                f"veto_strength must be in [0, 1], got {veto_strength}"
            )
        if not (0.0 <= min_gating_factor <= 1.0):
            raise ValueError(
                f"min_gating_factor must be in [0, 1], got {min_gating_factor}"
            )

        self.circuit = circuit
        self.or7a_glomerulus = or7a_glomerulus
        self.activation_threshold = activation_threshold
        self.veto_strength = veto_strength
        self.steepness = steepness
        self.graded = graded
        self.min_gating_factor = min_gating_factor

        # Initialize veto weight vector
        n_pn = len(circuit.connectivity.pn_ids)
        self.veto_weight = np.zeros(n_pn, dtype=np.float64)

        # Set veto weights for Or7a glomerulus PNs
        or7a_pn_indices = circuit.connectivity.get_pn_indices([or7a_glomerulus])

        if len(or7a_pn_indices) == 0:
            available_glomeruli = set(circuit.connectivity.pn_glomeruli.values())
            raise ValueError(
                f"No PNs found for Or7a glomerulus '{or7a_glomerulus}'. "
                f"Available glomeruli: {sorted(available_glomeruli)[:10]}..."
            )

        # Uniform weighting across Or7a PNs (sum to 1.0)
        self.veto_weight[or7a_pn_indices] = 1.0 / len(or7a_pn_indices)

        # Store Or7a PN indices for analysis
        self.or7a_pn_indices = or7a_pn_indices
        self.n_or7a_pns = len(or7a_pn_indices)

        # History tracking
        self.veto_history: List[Dict[str, float]] = []

    def compute_veto_signal(
        self,
        pn_activity: np.ndarray,
        return_diagnostics: bool = False,
    ) -> float | Tuple[float, Dict[str, float]]:
        """Compute Or7a veto signal from PN activity.

        Biological Rationale
        --------------------
        The veto signal v represents the "Task A feature strength" detected by
        the Or7a pathway. When Or7a PNs are strongly activated (benzaldehyde-like,
        55%), the veto signal is high (~0.9), suppressing plasticity. When Or7a
        activation is weak (hexanol-like, 14%), the veto signal is low (~0.1),
        allowing partial learning (cross-transfer).

        Mathematical Form
        -----------------
        1. Compute Or7a activation:
           a = W_veto^T @ PN_activity

        2. Apply graded veto (sigmoid):
           v = σ((a - θ) × γ)

        Or binary veto (threshold):
           v = 1.0 if a > θ else 0.0

        Parameters
        ----------
        pn_activity : np.ndarray
            PN firing rates. Shape: (n_pn,). Values typically in [0, 1].
        return_diagnostics : bool, optional
            If True, return (veto_signal, diagnostics_dict).
            Default: False

        Returns
        -------
        float or Tuple[float, Dict[str, float]]
            If return_diagnostics=False:
                veto_signal: Veto gate value in [0, 1]
            If return_diagnostics=True:
                (veto_signal, diagnostics) where diagnostics contains:
                - or7a_activation: Raw Or7a PN activation
                - veto_raw: Pre-clipping veto value
                - above_threshold: Whether activation exceeds threshold
                - gating_factor: Resulting plasticity gate (1 - β × v)

        Example
        -------
        >>> # Strong Or7a activation (benzaldehyde-like)
        >>> pn_act_strong = circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.55)
        >>> veto_strong = veto_gate.compute_veto_signal(pn_act_strong)
        >>> print(f"Benzaldehyde-like veto: {veto_strong:.3f}")  # ~0.9
        >>>
        >>> # Weak Or7a activation (hexanol-like)
        >>> pn_act_weak = circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.14)
        >>> veto_weak, diag = veto_gate.compute_veto_signal(pn_act_weak, return_diagnostics=True)
        >>> print(f"Hexanol-like veto: {veto_weak:.3f}")  # ~0.1
        >>> print(f"Or7a activation: {diag['or7a_activation']:.3f}")
        """
        # Validate input shape
        if pn_activity.shape != (self.circuit.connectivity.n_pn,):
            raise ValueError(
                f"pn_activity shape {pn_activity.shape} does not match "
                f"expected ({self.circuit.connectivity.n_pn},)"
            )

        # 1. Compute Or7a glomerulus activation (weighted sum)
        or7a_activation = np.dot(self.veto_weight, pn_activity)

        # 2. Apply veto function (graded or binary)
        if self.graded:
            # Graded sigmoid veto
            # v = σ((a - θ) × γ)
            veto_raw = 1.0 / (
                1.0 + np.exp(-self.steepness * (or7a_activation - self.activation_threshold))
            )
        else:
            # Binary threshold veto
            veto_raw = 1.0 if or7a_activation > self.activation_threshold else 0.0

        # 3. Clip to [0, 1] (numerical stability)
        veto_signal = float(np.clip(veto_raw, 0.0, 1.0))

        # 4. Log to history
        self.veto_history.append({
            "or7a_activation": float(or7a_activation),
            "veto_signal": veto_signal,
            "above_threshold": bool(or7a_activation > self.activation_threshold),
        })

        if not return_diagnostics:
            return veto_signal

        # Compute diagnostics
        gating_factor = 1.0 - self.veto_strength * veto_signal
        gating_factor = max(self.min_gating_factor, gating_factor)

        diagnostics = {
            "or7a_activation": float(or7a_activation),
            "veto_raw": float(veto_raw),
            "veto_signal": veto_signal,
            "above_threshold": bool(or7a_activation > self.activation_threshold),
            "gating_factor": gating_factor,
        }

        return veto_signal, diagnostics

    def gate_plasticity(
        self,
        delta_w: np.ndarray,
        veto_signal: float,
    ) -> np.ndarray:
        """Apply Or7a veto gating to weight updates.

        Biological Rationale
        --------------------
        When Or7a detects "Task A features" (benzaldehyde), plasticity is
        suppressed to prevent overwriting Task A memories during Task B training.
        The gating factor (1 - β × v) scales the weight update:

        - v = 1.0 (strong veto) → gating = 0.1 (10% residual plasticity)
        - v = 0.5 (moderate veto) → gating = 0.5 (50% plasticity)
        - v = 0.0 (no veto) → gating = 1.0 (100% plasticity)

        This implements "graded protection": Similar inputs get partial veto,
        enabling cross-transfer while still protecting core Task A features.

        Mathematical Form
        -----------------
        dW_gated = dW × max(min_gating, 1 - β × v)

        where:
            dW = ungated weight update
            β = veto_strength (default: 1.0)
            v = veto_signal [0, 1]
            min_gating = floor (default: 0.1)

        Parameters
        ----------
        delta_w : np.ndarray
            Ungated weight update matrix. Shape: (n_mbon, n_kc).
        veto_signal : float
            Veto gate value in [0, 1]. Typically from compute_veto_signal().

        Returns
        -------
        np.ndarray
            Gated weight update. Shape: (n_mbon, n_kc).
            Magnitude reduced by gating factor.

        Raises
        ------
        ValueError
            If veto_signal not in [0, 1].

        Example
        -------
        >>> # Strong veto (90% suppression)
        >>> delta_w = np.random.randn(96, 5177) * 0.001
        >>> veto_strong = 0.9
        >>> gated_update = veto_gate.gate_plasticity(delta_w, veto_strong)
        >>> reduction = 1.0 - (np.linalg.norm(gated_update) / np.linalg.norm(delta_w))
        >>> print(f"Update reduced by: {reduction:.1%}")  # ~90%
        >>>
        >>> # Weak veto (10% suppression)
        >>> veto_weak = 0.1
        >>> gated_update_weak = veto_gate.gate_plasticity(delta_w, veto_weak)
        >>> reduction_weak = 1.0 - (np.linalg.norm(gated_update_weak) / np.linalg.norm(delta_w))
        >>> print(f"Update reduced by: {reduction_weak:.1%}")  # ~10%
        """
        # Validate veto signal
        if not (0.0 <= veto_signal <= 1.0):
            raise ValueError(f"veto_signal must be in [0, 1], got {veto_signal}")

        # Compute gating factor
        gating_factor = 1.0 - self.veto_strength * veto_signal

        # Apply floor (never fully freeze plasticity)
        gating_factor = max(self.min_gating_factor, gating_factor)

        # Scale weight update
        return delta_w * gating_factor

    def get_veto_statistics(self) -> Dict[str, float]:
        """Compute summary statistics from veto history.

        Returns
        -------
        Dict[str, float]
            Statistics with keys:
            - mean_veto: Mean veto signal across history
            - max_veto: Maximum veto signal
            - min_veto: Minimum veto signal
            - mean_or7a_activation: Mean Or7a activation
            - fraction_above_threshold: Fraction of trials exceeding threshold
            - n_samples: Number of samples in history

        Example
        -------
        >>> # After running trials
        >>> stats = veto_gate.get_veto_statistics()
        >>> print(f"Mean veto: {stats['mean_veto']:.3f}")
        >>> print(f"Fraction above threshold: {stats['fraction_above_threshold']:.1%}")
        """
        if len(self.veto_history) == 0:
            return {
                "mean_veto": 0.0,
                "max_veto": 0.0,
                "min_veto": 0.0,
                "mean_or7a_activation": 0.0,
                "fraction_above_threshold": 0.0,
                "n_samples": 0,
            }

        veto_signals = [h["veto_signal"] for h in self.veto_history]
        or7a_activations = [h["or7a_activation"] for h in self.veto_history]
        above_threshold = [h["above_threshold"] for h in self.veto_history]

        return {
            "mean_veto": float(np.mean(veto_signals)),
            "max_veto": float(np.max(veto_signals)),
            "min_veto": float(np.min(veto_signals)),
            "mean_or7a_activation": float(np.mean(or7a_activations)),
            "fraction_above_threshold": float(np.mean(above_threshold)),
            "n_samples": len(self.veto_history),
        }

    def reset_history(self) -> None:
        """Clear veto history.

        Useful for multi-phase experiments where you want to track veto
        activity separately for each phase.

        Example
        -------
        >>> # Phase 1: Task A training
        >>> for trial in range(50):
        ...     veto = veto_gate.compute_veto_signal(pn_activity)
        >>> stats_phase1 = veto_gate.get_veto_statistics()
        >>>
        >>> # Reset for Phase 2
        >>> veto_gate.reset_history()
        >>> for trial in range(50):
        ...     veto = veto_gate.compute_veto_signal(pn_activity)
        >>> stats_phase2 = veto_gate.get_veto_statistics()
        """
        self.veto_history = []

    def __repr__(self) -> str:
        """Return summary string for debugging and logging."""
        return (
            f"Or7aVetoGate(\n"
            f"  or7a_glomerulus='{self.or7a_glomerulus}'\n"
            f"  n_or7a_pns={self.n_or7a_pns}\n"
            f"  activation_threshold={self.activation_threshold:.3f}\n"
            f"  veto_strength={self.veto_strength:.3f}\n"
            f"  graded={self.graded}\n"
            f"  min_gating_factor={self.min_gating_factor:.3f}\n"
            f"  n_veto_samples={len(self.veto_history)}\n"
            f")"
        )
