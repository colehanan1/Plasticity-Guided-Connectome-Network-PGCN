"""Selective Memory Protection via Veto Gate for Continual Learning.

This module implements a generic selective veto gate mechanism that prevents
catastrophic forgetting during sequential task learning by identifying and protecting
critical synaptic pathways. Unlike the Or7a-specific veto gate, this implementation
dynamically identifies which pathways are critical for Task A retention and gates
plasticity during Task B training based on chemical similarity.

Biological Context
------------------
**Inspired by OR7a Blocking Phenomenon** (Shen et al., 2025):

The Or7a olfactory receptor pathway provides graded plasticity control:
- Benzaldehyde (Or7a 55% active) → strong veto → minimal learning
- Hexanol (Or7a 14% active) → weak veto → partial learning
- Cross-transfer: Training on benzaldehyde enables hexanol response

**Generalized ML Translation**:
1. **Dynamic pathway identification**: Instead of fixed Or7a pathway, identify
   critical KC→MBON synapses for Task A based on gradient magnitude, weight
   magnitude, or activity correlation.

2. **Chemical similarity-based gating**: Use DoOR database to compute odor
   similarity and scale veto strength accordingly:
   - High similarity (Task A vs Task B) → strong veto (preserve Task A)
   - Low similarity → weak veto (allow Task B learning)

3. **Graded protection**: Veto strength ∈ [0.1, 1.0] with floor to preserve
   minimal plasticity even under strong protection.

Example
-------
>>> from pgcn.models.veto_gate import SelectiveVetoGate
>>> from pgcn.models.olfactory_circuit import OlfactoryCircuit
>>> from pgcn.models.learning_model import DopamineModulatedPlasticity
>>>
>>> # Initialize circuit and plasticity
>>> circuit = OlfactoryCircuit(conn_matrix, kc_sparsity_target=0.05)
>>> plasticity = DopamineModulatedPlasticity(weights, learning_rate=0.001)
>>>
>>> # Create selective veto gate
>>> veto_gate = SelectiveVetoGate(
...     circuit=circuit,
...     protection_threshold=0.3,  # Top 30% critical synapses
...     gate_strength=0.9,  # 90% suppression when fully active
...     similarity_metric='chemical'  # Use DoOR chemical similarity
... )
>>>
>>> # Train Task A and identify critical pathways
>>> task_a_data = run_task_a_training(...)
>>> protection_mask = veto_gate.identify_critical_pathways(
...     task_data=task_a_data,
...     method='gradient_magnitude'
... )
>>>
>>> # Train Task B with veto protection
>>> new_odor = 'ethyl butyrate'
>>> protected_odor = 'benzaldehyde'
>>> veto_signal = veto_gate.compute_veto_signal(new_odor, protected_odor)
>>> # veto_signal ≈ 0.4 (moderate similarity → moderate veto)
>>>
>>> # Apply protection to weight update
>>> gated_delta_w = veto_gate.apply_protection(delta_w, protection_mask, veto_signal)
>>> # ~60% plasticity retained (1.0 - 0.9*0.4 = 0.64)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np

# Import PGCN modules
from pgcn.models.connectivity_matrix import ConnectivityMatrix
from pgcn.models.olfactory_circuit import OlfactoryCircuit

# DoOR integration with graceful fallback
try:
    # Add door-python-toolkit to path if needed
    toolkit_path = Path.home() / "Documents/cole/VSCode/door-python-toolkit"
    if toolkit_path.exists() and str(toolkit_path) not in sys.path:
        sys.path.insert(0, str(toolkit_path))

    from door_integration.pgcn_door import PGCNDoorIntegration, DOOR_AVAILABLE
    DOOR_ENABLED = DOOR_AVAILABLE
except ImportError:
    DOOR_ENABLED = False
    print("⚠️  DoOR integration unavailable. Chemical similarity will use fallback (KC overlap).")


class SelectiveVetoGate:
    """Selective veto gate for pathway-specific plasticity control.

    This class provides a generalized veto mechanism that:
    1. Identifies critical synaptic pathways for task retention
    2. Computes odor similarity to determine veto strength
    3. Gates plasticity to protect critical pathways from interference

    Unlike the Or7a-specific veto gate, this implementation dynamically
    identifies which pathways to protect based on learning dynamics.

    Biological Rationale
    --------------------
    **Why selective protection?**

    1. **Not all synapses are equally important**: Some KC→MBON connections
       carry most of the task-relevant information (high weights, strong
       gradients). Protecting these "critical" synapses is sufficient.

    2. **Similarity-scaled gating**: Similar tasks should show stronger
       interference (high veto), while dissimilar tasks can learn independently
       (low veto). Chemical similarity predicts interference strength.

    3. **Graded control**: Veto strength ∈ [min_gating_factor, 1.0] ensures
       some residual plasticity even under protection, matching biological
       reality where synapses are never fully frozen.

    Parameters
    ----------
    circuit : OlfactoryCircuit
        Feedforward circuit for PN activation and KC computation.
    kc_mbon_connectivity : Optional[np.ndarray], optional
        KC→MBON connectivity matrix. Shape: (n_mbon, n_kc).
        If None, extracted from circuit.connectivity. Default: None
    protection_threshold : float, optional
        Percentile threshold for "critical" synapse identification.
        Range: [0, 1]. 0.3 = protect top 30% of synapses.
        Default: 0.3
    gate_strength : float, optional
        Maximum veto strength when fully active. Range: [0, 1].
        0 = no gating, 1 = maximum suppression.
        Default: 0.9 (90% suppression)
    similarity_metric : str, optional
        Method for computing odor similarity. Options:
        - 'chemical': DoOR-based chemical similarity (requires DoOR)
        - 'kc_overlap': KC activation overlap (fallback)
        - 'learned_weights': Similarity based on learned weight patterns
        Default: 'chemical'
    min_gating_factor : float, optional
        Minimum gating factor (floor). Prevents complete plasticity shutdown.
        Range: [0, 1]. Default: 0.1 (10% residual plasticity)
    door_threshold : float, optional
        DoOR receptor activation threshold for glomerulus mapping.
        Default: 0.3 (30% activation)

    Attributes
    ----------
    circuit : OlfactoryCircuit
        Reference to circuit for PN/KC computation.
    connectivity : ConnectivityMatrix
        Circuit connectivity (from circuit).
    kc_mbon_shape : Tuple[int, int]
        Shape of KC→MBON matrix (n_mbon, n_kc).
    protection_threshold : float
        Critical synapse percentile threshold.
    gate_strength : float
        Maximum veto gating strength.
    similarity_metric : str
        Active similarity computation method.
    min_gating_factor : float
        Plasticity floor.
    protection_mask : Optional[np.ndarray]
        Binary mask of protected synapses. Shape: (n_mbon, n_kc).
        Set by identify_critical_pathways().
    protected_odor : Optional[str]
        Odor name for protected task (for similarity computation).
    door_integration : Optional[PGCNDoorIntegration]
        DoOR interface (if available).
    veto_history : List[Dict[str, Any]]
        History of veto activations for analysis.

    Raises
    ------
    ValueError
        If protection_threshold, gate_strength, or min_gating_factor not in [0, 1].
        If similarity_metric is invalid.

    Notes
    -----
    **Design choices**:

    1. **NumPy-based (not PyTorch)**: Consistent with existing PGCN architecture.
       Tracks weight changes manually instead of autograd hooks.

    2. **Dynamic pathway identification**: Unlike Or7a fixed pathway, this
       identifies critical synapses based on learning dynamics.

    3. **DoOR chemical similarity**: Uses odorant receptor response profiles
       to compute biologically-grounded similarity scores.

    4. **Odor name normalization**: Automatically handles 'hexanol' → '1-hexanol'
       and 'ethyl_butyrate' → 'ethyl butyrate' transformations.

    Example
    -------
    >>> # Standard usage with chemical similarity
    >>> veto_gate = SelectiveVetoGate(
    ...     circuit=circuit,
    ...     protection_threshold=0.3,
    ...     gate_strength=0.9,
    ...     similarity_metric='chemical'
    ... )
    >>>
    >>> # Identify critical pathways for Task A
    >>> task_data = {'weight_changes': weight_change_history, 'kc_activity': kc_history}
    >>> protection_mask = veto_gate.identify_critical_pathways(task_data, method='gradient_magnitude')
    >>>
    >>> # Compute veto for Task B
    >>> veto_signal = veto_gate.compute_veto_signal('ethyl butyrate', 'benzaldehyde')
    >>> print(f"Veto strength: {veto_signal:.3f}")  # ~0.4 (moderate similarity)
    >>>
    >>> # Gate plasticity
    >>> gated_update = veto_gate.apply_protection(delta_w, protection_mask, veto_signal)
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        kc_mbon_connectivity: Optional[np.ndarray] = None,
        protection_threshold: float = 0.3,
        gate_strength: float = 0.9,
        similarity_metric: Literal['chemical', 'kc_overlap', 'learned_weights'] = 'chemical',
        min_gating_factor: float = 0.1,
        door_threshold: float = 0.3,
    ) -> None:
        """Initialize selective veto gate with circuit and parameters."""
        # Add current veto signal storage
        self.current_veto_signal: float = 0.0  # Current veto strength (updated per trial)
        # Validate parameters
        if not (0.0 <= protection_threshold <= 1.0):
            raise ValueError(
                f"protection_threshold must be in [0, 1], got {protection_threshold}"
            )
        if not (0.0 <= gate_strength <= 1.0):
            raise ValueError(
                f"gate_strength must be in [0, 1], got {gate_strength}"
            )
        if not (0.0 <= min_gating_factor <= 1.0):
            raise ValueError(
                f"min_gating_factor must be in [0, 1], got {min_gating_factor}"
            )

        valid_metrics = {'chemical', 'kc_overlap', 'learned_weights'}
        if similarity_metric not in valid_metrics:
            raise ValueError(
                f"similarity_metric must be one of {valid_metrics}, got '{similarity_metric}'"
            )

        self.circuit = circuit
        self.connectivity = circuit.connectivity
        self.protection_threshold = protection_threshold
        self.gate_strength = gate_strength
        self.similarity_metric = similarity_metric
        self.min_gating_factor = min_gating_factor
        self.door_threshold = door_threshold

        # Extract KC→MBON shape
        if kc_mbon_connectivity is not None:
            self.kc_mbon_shape = kc_mbon_connectivity.shape
        else:
            kc_to_mbon_sparse = self.connectivity.kc_to_mbon
            self.kc_mbon_shape = kc_to_mbon_sparse.shape  # (n_mbon, n_kc)

        # Initialize protection state
        self.protection_mask: Optional[np.ndarray] = None
        self.protected_odor: Optional[str] = None
        self.current_veto_signal: float = 0.0  # Current veto strength (updated per trial)

        # Initialize DoOR integration if available
        self.door_integration: Optional[Any] = None
        if similarity_metric == 'chemical':
            if DOOR_ENABLED:
                self.door_integration = PGCNDoorIntegration()
                print("✅ DoOR chemical similarity enabled")
            else:
                print("⚠️  DoOR unavailable. Falling back to KC overlap similarity.")
                self.similarity_metric = 'kc_overlap'

        # History tracking
        self.veto_history: List[Dict[str, Any]] = []

    def identify_critical_pathways(
        self,
        task_data: Dict[str, Any],
        method: Literal['gradient_magnitude', 'weight_magnitude', 'activity_correlation'] = 'gradient_magnitude',
    ) -> np.ndarray:
        """Identify KC→MBON synapses critical for task retention.

        Biological Rationale
        --------------------
        Not all synapses contribute equally to task performance. This method
        identifies the "critical" synapses that carry task-relevant information:

        1. **Gradient magnitude**: Synapses with largest ∂Loss/∂w during training
           are most sensitive to task performance → protect these.

        2. **Weight magnitude**: Synapses with largest |w_ij| after training
           contribute most to output → protect strong connections.

        3. **Activity correlation**: KCs most correlated with task output are
           essential for prediction → protect their outputs.

        Mathematical Form
        -----------------
        For gradient magnitude:
            importance_ij = mean(|∂Loss/∂w_ij|) across training trials
            mask_ij = I(importance_ij > percentile(importance, threshold))

        Where threshold=0.3 → protect top 30% of synapses.

        Parameters
        ----------
        task_data : Dict[str, Any]
            Task training data. Required keys depend on method:
            - 'gradient_magnitude': requires 'weight_changes' (List[np.ndarray])
            - 'weight_magnitude': requires 'final_weights' (np.ndarray)
            - 'activity_correlation': requires 'kc_activity' (List[np.ndarray])
              and 'mbon_output' (List[np.ndarray])
        method : str, optional
            Importance estimation method. Default: 'gradient_magnitude'

        Returns
        -------
        np.ndarray
            Binary protection mask. Shape: (n_mbon, n_kc).
            1 = protect this synapse, 0 = allow plasticity.

        Raises
        ------
        ValueError
            If required data keys missing from task_data.
            If method is invalid.

        Example
        -------
        >>> # After training Task A, identify critical synapses
        >>> task_data = {
        ...     'weight_changes': weight_change_history,  # List of delta_w matrices
        ...     'final_weights': plasticity.kc_to_mbon,   # Current weights
        ... }
        >>> protection_mask = veto_gate.identify_critical_pathways(
        ...     task_data, method='gradient_magnitude'
        ... )
        >>> print(f"Protected synapses: {protection_mask.sum()} / {protection_mask.size}")
        >>> # Protected synapses: 4821 / 16096 (30%)
        """
        if method == 'gradient_magnitude':
            # Protect synapses with largest weight changes during training
            if 'weight_changes' not in task_data:
                raise ValueError(
                    "method='gradient_magnitude' requires 'weight_changes' in task_data"
                )

            weight_changes = task_data['weight_changes']  # List[np.ndarray]
            if len(weight_changes) == 0:
                # No training data → protect nothing
                return np.zeros(self.kc_mbon_shape, dtype=bool)

            # Compute mean absolute weight change per synapse
            mean_abs_change = np.mean([np.abs(dw) for dw in weight_changes], axis=0)
            importance = mean_abs_change

        elif method == 'weight_magnitude':
            # Protect synapses with largest final weights
            if 'final_weights' not in task_data:
                raise ValueError(
                    "method='weight_magnitude' requires 'final_weights' in task_data"
                )

            final_weights = task_data['final_weights']  # (n_mbon, n_kc)
            importance = np.abs(final_weights)

        elif method == 'activity_correlation':
            # Protect KCs most correlated with task output
            if 'kc_activity' not in task_data or 'mbon_output' not in task_data:
                raise ValueError(
                    "method='activity_correlation' requires 'kc_activity' and 'mbon_output'"
                )

            kc_activity = np.array(task_data['kc_activity'])  # (n_trials, n_kc)
            mbon_output = np.array(task_data['mbon_output'])  # (n_trials, n_mbon)

            # Compute correlation between each KC and each MBON
            importance = np.zeros(self.kc_mbon_shape)
            for m in range(self.kc_mbon_shape[0]):  # For each MBON
                for k in range(self.kc_mbon_shape[1]):  # For each KC
                    # Pearson correlation
                    corr = np.corrcoef(kc_activity[:, k], mbon_output[:, m])[0, 1]
                    importance[m, k] = np.abs(corr) if not np.isnan(corr) else 0.0

        else:
            raise ValueError(f"Unknown method: {method}")

        # Identify anatomically connected synapses (non-zero importance or non-zero weights)
        # This excludes synapses with zero anatomical connectivity
        connected_mask = (importance > 0)
        n_connected = connected_mask.sum()

        # Compute protection mask: top percentile of CONNECTED synapses only
        if n_connected > 0:
            connected_importance = importance[connected_mask]
            cutoff = np.percentile(connected_importance, (1.0 - self.protection_threshold) * 100)
            protection_mask = (importance >= cutoff) & connected_mask
        else:
            # No connected synapses - protect nothing
            protection_mask = np.zeros(self.kc_mbon_shape, dtype=bool)

        n_protected = protection_mask.sum()
        n_total = protection_mask.size
        protection_fraction_total = n_protected / n_total if n_total > 0 else 0.0
        protection_fraction_connected = n_protected / n_connected if n_connected > 0 else 0.0

        print(f"✅ Identified {n_protected}/{n_connected} connected synapses "
              f"({protection_fraction_connected:.1%} of connected, {protection_fraction_total:.1%} of total) "
              f"using {method}")

        # Store protection mask and mark as set
        self.protection_mask = protection_mask

        print(f"✅ Identified {n_protected}/{n_total} critical synapses "
              f"({n_protected/n_total:.1%}) using {method}")

        return protection_mask

    def compute_veto_signal(
        self,
        new_odor: str,
        protected_odor: str,
        return_diagnostics: bool = False,
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """Compute veto signal based on odor similarity.

        Biological Rationale
        --------------------
        Veto strength should scale with chemical similarity:
        - High similarity → Task B interferes with Task A → strong veto
        - Low similarity → Independent tasks → weak veto

        Chemical similarity is computed using DoOR odorant receptor response
        profiles (cosine similarity of receptor activation patterns).

        Mathematical Form
        -----------------
        For chemical similarity:
            r_new = DoOR.encode(new_odor)  # Receptor responses
            r_protected = DoOR.encode(protected_odor)
            similarity = cosine(r_new, r_protected) ∈ [0, 1]
            veto_signal = similarity

        For KC overlap:
            KC_new = circuit.propagate(activate_PNs(new_odor))
            KC_protected = circuit.propagate(activate_PNs(protected_odor))
            overlap = |KC_new ∩ KC_protected| / |KC_new ∪ KC_protected|
            veto_signal = overlap

        Parameters
        ----------
        new_odor : str
            Odorant name for new task (e.g., 'ethyl butyrate', 'hexanol').
            Names are automatically normalized ('hexanol' → '1-hexanol').
        protected_odor : str
            Odorant name for protected task (e.g., 'benzaldehyde').
        return_diagnostics : bool, optional
            If True, return (veto_signal, diagnostics_dict).
            Default: False

        Returns
        -------
        float or Tuple[float, Dict[str, Any]]
            If return_diagnostics=False:
                veto_signal: Veto strength in [0, 1]
            If return_diagnostics=True:
                (veto_signal, diagnostics) where diagnostics contains:
                - similarity_raw: Raw similarity score
                - method_used: Similarity method ('chemical' or 'kc_overlap')
                - new_odor_glomeruli: Activated glomeruli for new odor
                - protected_odor_glomeruli: Activated glomeruli for protected odor

        Example
        -------
        >>> # Chemical similarity (DoOR)
        >>> veto = veto_gate.compute_veto_signal('ethyl butyrate', 'benzaldehyde')
        >>> print(f"Veto signal: {veto:.3f}")  # ~0.35 (moderate similarity)
        >>>
        >>> # With diagnostics
        >>> veto, diag = veto_gate.compute_veto_signal(
        ...     'hexanol', 'benzaldehyde', return_diagnostics=True
        ... )
        >>> print(f"Method: {diag['method_used']}")
        >>> print(f"Similarity: {diag['similarity_raw']:.3f}")
        """
        # Normalize odor names
        new_odor_norm = self._normalize_odor_name(new_odor)
        protected_odor_norm = self._normalize_odor_name(protected_odor)

        # Store protected odor for reference
        self.protected_odor = protected_odor_norm

        diagnostics = {
            'new_odor_normalized': new_odor_norm,
            'protected_odor_normalized': protected_odor_norm,
        }

        # Compute similarity based on metric
        if self.similarity_metric == 'chemical' and self.door_integration is not None:
            similarity, method_diag = self._compute_chemical_similarity(
                new_odor_norm, protected_odor_norm
            )
            diagnostics.update(method_diag)
            diagnostics['method_used'] = 'chemical'

        elif self.similarity_metric == 'kc_overlap':
            similarity, method_diag = self._compute_kc_overlap_similarity(
                new_odor_norm, protected_odor_norm
            )
            diagnostics.update(method_diag)
            diagnostics['method_used'] = 'kc_overlap'

        elif self.similarity_metric == 'learned_weights':
            # Fallback to KC overlap if learned weights not available
            similarity, method_diag = self._compute_kc_overlap_similarity(
                new_odor_norm, protected_odor_norm
            )
            diagnostics.update(method_diag)
            diagnostics['method_used'] = 'kc_overlap (learned_weights fallback)'

        else:
            # Default fallback
            similarity = 0.5
            diagnostics['method_used'] = 'fallback'

        # Veto signal is direct similarity
        veto_signal = float(np.clip(similarity, 0.0, 1.0))
        diagnostics['similarity_raw'] = float(similarity)
        diagnostics['veto_signal'] = veto_signal

        # Log to history
        self.veto_history.append({
            'new_odor': new_odor_norm,
            'protected_odor': protected_odor_norm,
            'veto_signal': veto_signal,
            'method': diagnostics['method_used'],
        })

        if return_diagnostics:
            return veto_signal, diagnostics
        return veto_signal

    def _normalize_odor_name(self, odor: str) -> str:
        """Normalize odor name for DoOR compatibility.

        Transformations:
        - 'hexanol' → '1-hexanol'
        - 'ethyl_butyrate' → 'ethyl butyrate' (underscore to space)
        - Lowercase handling
        """
        odor_lower = odor.lower().strip()

        # Underscore to space
        odor_normalized = odor_lower.replace('_', ' ')

        # Special cases for DoOR
        name_mappings = {
            'hexanol': '1-hexanol',
            '3octanol': '3-octanol',
            '2heptanone': '2-heptanone',
        }

        if odor_normalized in name_mappings:
            return name_mappings[odor_normalized]

        return odor_normalized

    def _compute_chemical_similarity(
        self, odor1: str, odor2: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute chemical similarity using DoOR receptor responses."""
        try:
            # Encode both odors
            responses1 = self.door_integration.encode_odorant(odor1)
            responses2 = self.door_integration.encode_odorant(odor2)

            # Cosine similarity
            dot_product = float((responses1 * responses2).sum())
            norm1 = float(np.sqrt((responses1 ** 2).sum()))
            norm2 = float(np.sqrt((responses2 ** 2).sum()))

            if norm1 * norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
            else:
                similarity = 0.0

            # Get activated glomeruli
            glomeruli1 = self.door_integration.map_odorant_to_glomeruli(
                odor1, threshold=self.door_threshold
            )
            glomeruli2 = self.door_integration.map_odorant_to_glomeruli(
                odor2, threshold=self.door_threshold
            )

            diagnostics = {
                'similarity': float(similarity),
                'new_odor_glomeruli': glomeruli1,
                'protected_odor_glomeruli': glomeruli2,
                'glomeruli_overlap': list(set(glomeruli1) & set(glomeruli2)),
            }

            return similarity, diagnostics

        except Exception as e:
            print(f"⚠️  Chemical similarity failed: {e}. Falling back to KC overlap.")
            return self._compute_kc_overlap_similarity(odor1, odor2)

    def _compute_kc_overlap_similarity(
        self, odor1: str, odor2: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute KC activation overlap similarity (fallback method)."""
        try:
            # Get glomeruli for each odor (if DoOR available)
            if self.door_integration is not None:
                glomeruli1 = self.door_integration.map_odorant_to_glomeruli(
                    odor1, threshold=self.door_threshold
                )
                glomeruli2 = self.door_integration.map_odorant_to_glomeruli(
                    odor2, threshold=self.door_threshold
                )
            else:
                # Fallback: assume single glomerulus (not realistic)
                glomeruli1 = [odor1]
                glomeruli2 = [odor2]

            # Activate PNs
            pn_activity1 = self.circuit.activate_pns_by_glomeruli(glomeruli1, firing_rate=1.0)
            pn_activity2 = self.circuit.activate_pns_by_glomeruli(glomeruli2, firing_rate=1.0)

            # Propagate to KCs
            kc_activity1 = self.circuit.propagate_pn_to_kc(pn_activity1)
            kc_activity2 = self.circuit.propagate_pn_to_kc(pn_activity2)

            # Compute Jaccard similarity (overlap / union)
            active_kcs1 = set(np.where(kc_activity1 > 0)[0])
            active_kcs2 = set(np.where(kc_activity2 > 0)[0])

            intersection = active_kcs1 & active_kcs2
            union = active_kcs1 | active_kcs2

            if len(union) > 0:
                similarity = len(intersection) / len(union)
            else:
                similarity = 0.0

            diagnostics = {
                'similarity': float(similarity),
                'new_odor_glomeruli': glomeruli1,
                'protected_odor_glomeruli': glomeruli2,
                'kc_overlap_count': len(intersection),
                'kc_union_count': len(union),
            }

            return similarity, diagnostics

        except Exception as e:
            print(f"⚠️  KC overlap computation failed: {e}. Returning default similarity=0.5")
            return 0.5, {'similarity': 0.5, 'error': str(e)}

    def set_veto_signal(self, veto_signal: float) -> None:
        """Set current veto signal strength.

        Call this before update_weights() to set the veto strength for the current trial.
        """
        self.current_veto_signal = veto_signal

    def apply_protection(
        self,
        delta_w: np.ndarray,
        protection_mask: Optional[np.ndarray] = None,
        veto_signal: Optional[float] = None,
    ) -> np.ndarray:
        """Apply veto gate to weight updates.

        Biological Rationale
        --------------------
        Plasticity is suppressed for protected synapses based on veto strength:
        - veto_signal = 1.0 (high similarity) → strong suppression (10% plasticity)
        - veto_signal = 0.5 (moderate similarity) → moderate suppression (55% plasticity)
        - veto_signal = 0.0 (no similarity) → no suppression (100% plasticity)

        Mathematical Form
        -----------------
        gating_factor = max(min_gating, 1 - gate_strength × veto_signal)
        delta_w_protected = delta_w × protection_mask × gating_factor
        delta_w_unprotected = delta_w × (1 - protection_mask)
        delta_w_gated = delta_w_protected + delta_w_unprotected

        Parameters
        ----------
        delta_w : np.ndarray
            Ungated weight update matrix. Shape: (n_mbon, n_kc).
        protection_mask : Optional[np.ndarray], optional
            Binary mask of protected synapses. If None, uses self.protection_mask.
            Default: None
        veto_signal : Optional[float], optional
            Veto strength [0, 1]. If None, defaults to 1.0 (full veto for protected).
            Default: None

        Returns
        -------
        np.ndarray
            Gated weight update. Shape: (n_mbon, n_kc).

        Example
        -------
        >>> # Apply strong veto (high similarity)
        >>> gated_update = veto_gate.apply_protection(delta_w, protection_mask, veto_signal=0.9)
        >>> reduction = 1.0 - (np.linalg.norm(gated_update) / np.linalg.norm(delta_w))
        >>> print(f"Update reduced by: {reduction:.1%}")  # ~27% (0.9 * 0.3)
        """
        # Use stored protection mask if not provided
        if protection_mask is None:
            if self.protection_mask is None:
                raise ValueError(
                    "No protection mask provided and none has been set. "
                    "Call identify_critical_pathways() first or provide protection_mask."
                )
            protection_mask = self.protection_mask

        # Validate shapes
        if delta_w.shape != self.kc_mbon_shape:
            raise ValueError(
                f"delta_w shape {delta_w.shape} does not match expected {self.kc_mbon_shape}"
            )
        if protection_mask.shape != self.kc_mbon_shape:
            raise ValueError(
                f"protection_mask shape {protection_mask.shape} does not match "
                f"expected {self.kc_mbon_shape}"
            )

        # Use provided veto signal, or current veto signal, or default to 0.0
        if veto_signal is None:
            veto_signal = self.current_veto_signal  # Use stored veto signal (can be 0.0)

        # Compute gating factor
        gating_factor = 1.0 - self.gate_strength * veto_signal
        gating_factor = max(self.min_gating_factor, gating_factor)

        # Apply protection: gate only protected synapses
        delta_w_gated = delta_w.copy()
        n_protected = protection_mask.sum()
        original_norm = np.linalg.norm(delta_w_gated[protection_mask])
        delta_w_gated[protection_mask] *= gating_factor
        gated_norm = np.linalg.norm(delta_w_gated[protection_mask])

        # Diagnostic print (only print every 10 calls to avoid spam)
        if not hasattr(self, '_apply_count'):
            self._apply_count = 0
        self._apply_count += 1
        if self._apply_count % 10 == 1:
            reduction = 1.0 - (gated_norm / original_norm) if original_norm > 0 else 0.0
            print(f"  [Veto] signal={veto_signal:.3f}, gate={gating_factor:.3f}, "
                  f"protected={n_protected}, reduction={reduction:.1%}")

        return delta_w_gated

    def get_veto_statistics(self) -> Dict[str, float]:
        """Compute summary statistics from veto history.

        Returns
        -------
        Dict[str, float]
            Statistics with keys:
            - mean_veto: Mean veto signal across history
            - max_veto: Maximum veto signal
            - min_veto: Minimum veto signal
            - n_samples: Number of samples in history
        """
        if len(self.veto_history) == 0:
            return {
                'mean_veto': 0.0,
                'max_veto': 0.0,
                'min_veto': 0.0,
                'n_samples': 0,
            }

        veto_signals = [h['veto_signal'] for h in self.veto_history]

        return {
            'mean_veto': float(np.mean(veto_signals)),
            'max_veto': float(np.max(veto_signals)),
            'min_veto': float(np.min(veto_signals)),
            'n_samples': len(self.veto_history),
        }

    def reset_history(self) -> None:
        """Clear veto history (useful for multi-phase experiments)."""
        self.veto_history = []

    def reset_protection(self) -> None:
        """Clear protection mask and protected odor."""
        self.protection_mask = None
        self.protected_odor = None

    def __repr__(self) -> str:
        """Return summary string for debugging and logging."""
        protected_status = "set" if self.protection_mask is not None else "not set"
        n_protected = self.protection_mask.sum() if self.protection_mask is not None else 0

        return (
            f"SelectiveVetoGate(\n"
            f"  kc_mbon_shape={self.kc_mbon_shape}\n"
            f"  protection_threshold={self.protection_threshold:.3f}\n"
            f"  gate_strength={self.gate_strength:.3f}\n"
            f"  similarity_metric='{self.similarity_metric}'\n"
            f"  min_gating_factor={self.min_gating_factor:.3f}\n"
            f"  protection_mask={protected_status} ({n_protected} synapses)\n"
            f"  protected_odor='{self.protected_odor}'\n"
            f"  n_veto_samples={len(self.veto_history)}\n"
            f")"
        )
