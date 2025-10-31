"""Olfactory circuit forward propagation with sparse KC activation.

This module implements feedforward computation through the Drosophila mushroom
body olfactory circuit (PN→KC→MBON pathway), enforcing the critical biological
constraint of sparse KC activation (~5% active per odor). This sparse coding
is a hallmark of MB function, enabling high-dimensional odor representations
that support efficient associative learning and pattern separation.

Biological Context
------------------
The MB operates as an **expansion recoder** that transforms ~150 PN combinatorial
odor codes into sparse ~2000-5000 KC representations. Key biological principles:

1. **Sparse KC activation (~5%)**: In vivo calcium imaging shows only ~5-10% of
   KCs respond to any given odor, despite each KC receiving input from ~6-8 randomly
   selected PNs. This sparsity is enforced by:
   - High KC firing thresholds (requiring coincident multi-PN input)
   - Lateral inhibition via GABAergic APL neurons that suppress weak KC responses
   - Feedforward inhibition that sets a dynamic activity threshold

2. **K-winners-take-all mechanism**: The circuit implements a form of k-WTA where
   approximately the top 5% of KCs (ranked by PN-driven depolarization) are
   allowed to fire, while the remaining 95% are suppressed. This biological
   computation is modeled here as:
   ```
   h_kc = W_pn_kc @ pn_activity  # Linear integration
   kc_active = top_k(h_kc, k=0.05*n_kc)  # Keep only top 5%
   ```

3. **Pattern separation**: Sparse KC codes separate similar odors into distinct
   representations. Two odors with 70% overlapping PN responses may activate
   nearly disjoint KC populations (e.g., 10-20% overlap), reducing interference
   during memory formation.

4. **Energy efficiency**: Maintaining sparse activity minimizes metabolic cost
   while preserving information capacity. The MB "learns more with less" by
   spreading odor information across many rarely-active neurons.

Example
-------
>>> from data_loaders.circuit_loader import CircuitLoader
>>> from pgcn.models.olfactory_circuit import OlfactoryCircuit
>>>
>>> # Load circuit connectivity
>>> loader = CircuitLoader(cache_dir="data/cache")
>>> conn_matrix = loader.load_connectivity_matrix(normalize_weights="row")
>>>
>>> # Instantiate forward-pass circuit
>>> circuit = OlfactoryCircuit(
...     connectivity=conn_matrix,
...     kc_sparsity_target=0.05  # 5% KC activity
... )
>>>
>>> # Simulate odor presentation
>>> pn_activity = circuit.activate_pns_by_glomeruli(["DA1", "DL3"], firing_rate=1.0)
>>> mbon_output, diagnostics = circuit.forward_pass(pn_activity, return_intermediates=True)
>>> print(f"KC sparsity achieved: {diagnostics['sparsity_fraction']:.2%}")
>>> print(f"MBON response range: [{mbon_output.min():.2f}, {mbon_output.max():.2f}]")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pgcn.models.connectivity_matrix import ConnectivityMatrix


class OlfactoryCircuit:
    """Feedforward olfactory circuit with sparse KC activation.

    This class implements the PN→KC→MBON pathway using connectivity from the
    FlyWire connectome, enforcing biological sparse coding constraints via
    k-winners-take-all lateral inhibition.

    Biological Design Principles
    -----------------------------
    1. **No learning/plasticity**: This is a pure feedforward circuit. Plasticity
       at KC→MBON synapses will be implemented in downstream learning modules
       (Phase 2), which will modify weights based on dopamine-gated Hebbian rules.

    2. **Sparse KC activation**: The ~5% sparsity target is biologically conserved
       across Drosophila species and experimental conditions. This constraint is
       essential for pattern separation and energy efficiency.

    3. **Linear integration**: PNs→KCs and KCs→MBONs use linear weighted sums
       (no nonlinearity yet). Real neurons have nonlinear dendrites and spiking
       thresholds, but this simplification captures first-order circuit dynamics.

    4. **Deterministic computation**: Same PN input produces same KC/MBON output.
       Biological circuits exhibit trial-to-trial variability (noise), but
       determinism is appropriate for testing hypotheses about connectivity structure.

    Parameters
    ----------
    connectivity : ConnectivityMatrix
        Frozen connectivity matrix with PN→KC→MBON sparse matrices.
    kc_sparsity_target : float, optional
        Target fraction of KCs active per odor. Biological range: 0.05-0.10.
        Default: 0.05 (5% active, 95% silent).
    kc_sparsity_mode : str, optional
        Algorithm for enforcing sparsity. Options:
        - "k_winners_take_all": Keep top-k KCs by activation strength (default).
          Biologically motivated by lateral inhibition (APL neuron).
        - "threshold": Apply fixed threshold (not implemented yet).
        Default: "k_winners_take_all"

    Attributes
    ----------
    connectivity : ConnectivityMatrix
        Reference to connectivity structure (immutable).
    sparsity_target : float
        Target KC sparsity fraction (e.g., 0.05 = 5%).
    k_winners : int
        Number of KCs to keep active (int(n_kc * sparsity_target)).

    Raises
    ------
    ValueError
        If sparsity_target is not in (0, 1) or sparsity_mode is invalid.

    Notes
    -----
    Future enhancements (Phase 2+):
    - Add MBON nonlinearity (sigmoid/ReLU)
    - Implement threshold-based sparsity (biological realism)
    - Add DAN→MBON modulation for fast valence signaling
    - Implement KC→KC lateral inhibition (APL explicit model)
    """

    def __init__(
        self,
        connectivity: ConnectivityMatrix,
        kc_sparsity_target: float = 0.05,
        kc_sparsity_mode: str = "k_winners_take_all",
    ) -> None:
        """Initialize olfactory circuit with connectivity and sparsity parameters.

        Parameters
        ----------
        connectivity : ConnectivityMatrix
            Connectivity matrix (frozen, immutable).
        kc_sparsity_target : float
            Fraction of KCs active per odor (biological: 0.05-0.10).
        kc_sparsity_mode : str
            Sparsity enforcement algorithm ("k_winners_take_all").
        """
        # Validate sparsity target
        if not (0.0 < kc_sparsity_target < 1.0):
            raise ValueError(
                f"kc_sparsity_target must be in (0, 1), got {kc_sparsity_target}"
            )

        # Validate sparsity mode
        valid_modes = {"k_winners_take_all", "threshold"}
        if kc_sparsity_mode not in valid_modes:
            raise ValueError(
                f"kc_sparsity_mode must be one of {valid_modes}, got '{kc_sparsity_mode}'"
            )

        self.connectivity = connectivity
        self.sparsity_target = kc_sparsity_target
        self.sparsity_mode = kc_sparsity_mode

        # Pre-compute k for k-winners-take-all
        # Biological interpretation: APL inhibition sets a dynamic threshold that
        # allows ~k most-excited KCs to spike, suppressing the rest
        self.k_winners = max(1, int(connectivity.n_kc * kc_sparsity_target))

    def propagate_pn_to_kc(self, pn_activity: np.ndarray) -> np.ndarray:
        """Propagate PN activity through PN→KC expansion with sparsity enforcement.

        Biological Rationale
        --------------------
        This implements the core MB expansion recoding computation:

        1. **Linear integration**: Each KC sums weighted inputs from its ~6-8
           connected PNs. Weights come from the connectome (synapse counts,
           normalized to sum=1 per KC if using row normalization).

        2. **K-winners-take-all**: After linear integration, only the top ~5%
           of KCs (highest activations) are allowed to respond. This mimics
           lateral inhibition by the GABAergic APL neuron, which provides
           global feedback inhibition that suppresses weakly-activated KCs.

        3. **Sparse coding advantage**: Sparsity ensures that similar odors
           (overlapping PN codes) activate different KC ensembles, reducing
           catastrophic interference in associative memory.

        Mathematical Form
        -----------------
        h_raw = W_pn_kc @ pn_activity  # Shape: (n_kc,)
        kc_activity = top_k(h_raw, k)  # Keep top k, zero rest

        Where W_pn_kc is the sparse (n_kc, n_pn) connectivity matrix.

        Parameters
        ----------
        pn_activity : np.ndarray
            PN firing rates. Shape: (n_pn,). Values typically in [0, 1],
            where 1.0 represents maximal PN response to preferred odorant.

        Returns
        -------
        np.ndarray
            Sparse KC activity. Shape: (n_kc,). Approximately k_winners
            entries are non-zero; remaining entries are exactly 0.0.

        Raises
        ------
        ValueError
            If pn_activity shape does not match connectivity.n_pn.

        Example
        -------
        >>> # Activate subset of PNs
        >>> pn_input = np.zeros(circuit.connectivity.n_pn)
        >>> pn_input[0:10] = 1.0  # Strong activation of first 10 PNs
        >>> kc_output = circuit.propagate_pn_to_kc(pn_input)
        >>> n_active = np.count_nonzero(kc_output)
        >>> print(f"Active KCs: {n_active} / {circuit.connectivity.n_kc}")
        """
        # Validate input shape
        if pn_activity.shape != (self.connectivity.n_pn,):
            raise ValueError(
                f"pn_activity shape {pn_activity.shape} does not match "
                f"expected ({self.connectivity.n_pn},)"
            )

        # Linear integration: h_raw = W @ x
        # pn_to_kc is shape (n_kc, n_pn); pn_activity is (n_pn,) → result is (n_kc,)
        h_raw = self.connectivity.pn_to_kc.dot(pn_activity)

        # Apply sparsity enforcement
        if self.sparsity_mode == "k_winners_take_all":
            kc_activity = self._apply_k_winners_take_all(h_raw, self.k_winners)
        else:
            # Threshold mode not yet implemented
            raise NotImplementedError(
                f"Sparsity mode '{self.sparsity_mode}' not implemented yet. "
                f"Use 'k_winners_take_all'."
            )

        return kc_activity

    def propagate_kc_to_mbon(self, kc_activity: np.ndarray) -> np.ndarray:
        """Propagate sparse KC activity through KC→MBON readout layer.

        Biological Rationale
        --------------------
        MBONs integrate activity across hundreds to thousands of KCs to compute
        learned odor valence (attractiveness/aversiveness). Each MBON samples
        KCs from specific MB lobe compartments, creating a distributed population
        code for behavioral output:

        - Approach MBONs (e.g., MBON-γ1pedc>α/β): activated by rewarded odors,
          drive appetitive responses
        - Avoidance MBONs (e.g., MBON-γ2α'1): activated by punished odors,
          drive aversive responses

        The KC→MBON synapses are the primary site of associative plasticity
        (not implemented here; see Phase 2 learning modules). During learning,
        dopamine from DANs potentiates or depresses these connections, shifting
        MBON responses to reflect odor valence.

        Mathematical Form
        -----------------
        mbon_output = W_kc_mbon @ kc_activity  # Shape: (n_mbon,)

        Where W_kc_mbon is the sparse (n_mbon, n_kc) connectivity matrix.

        Parameters
        ----------
        kc_activity : np.ndarray
            Sparse KC firing rates. Shape: (n_kc,). Typically ~5% non-zero.

        Returns
        -------
        np.ndarray
            MBON firing rates. Shape: (n_mbon,). Represents odor-evoked
            MBON responses before learning/plasticity.

        Raises
        ------
        ValueError
            If kc_activity shape does not match connectivity.n_kc.

        Example
        -------
        >>> kc_input = np.zeros(circuit.connectivity.n_kc)
        >>> kc_input[0:250] = 1.0  # Sparse KC activity (~5% of 5000 KCs)
        >>> mbon_output = circuit.propagate_kc_to_mbon(kc_input)
        >>> print(f"MBON response range: [{mbon_output.min():.2f}, {mbon_output.max():.2f}]")
        """
        # Validate input shape
        if kc_activity.shape != (self.connectivity.n_kc,):
            raise ValueError(
                f"kc_activity shape {kc_activity.shape} does not match "
                f"expected ({self.connectivity.n_kc},)"
            )

        # Linear readout: y = W @ h
        # kc_to_mbon is shape (n_mbon, n_kc); kc_activity is (n_kc,) → result is (n_mbon,)
        mbon_output = self.connectivity.kc_to_mbon.dot(kc_activity)

        return mbon_output

    def forward_pass(
        self,
        pn_activity: np.ndarray,
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Compute full feedforward pass: PN → KC → MBON.

        This orchestrates the two-stage propagation through the circuit,
        optionally returning intermediate KC activations and diagnostics
        for analysis and validation.

        Biological Interpretation
        --------------------------
        This simulates the temporal sequence of neural activity during odor
        presentation in a behaving fly:

        1. Odor molecules bind olfactory receptors → ORN spikes (not modeled)
        2. ORNs activate PNs in antennal lobe (input to this function)
        3. PNs drive sparse KC responses in MB (~5% active)
        4. MBONs integrate KC activity to compute odor valence

        In a naive (untrained) fly, MBON responses reflect random connectivity.
        After learning (Phase 2), MBON responses shift to encode reward/punishment
        associations.

        Parameters
        ----------
        pn_activity : np.ndarray
            PN firing rates. Shape: (n_pn,).
        return_intermediates : bool, optional
            If True, return (mbon_output, diagnostics_dict).
            If False, return only mbon_output.
            Default: False

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, Dict[str, Any]]
            If return_intermediates=False:
                mbon_output: MBON activity, shape (n_mbon,)
            If return_intermediates=True:
                (mbon_output, diagnostics) where diagnostics contains:
                - "kc_activity": KC activity after sparsity, shape (n_kc,)
                - "sparsity_fraction": measured KC sparsity (fraction active)
                - "mbon_mean": mean MBON response
                - "mbon_std": standard deviation of MBON responses

        Example
        -------
        >>> pn_input = circuit.activate_pns_by_glomeruli(["DA1", "VA1d"])
        >>> mbon_out, info = circuit.forward_pass(pn_input, return_intermediates=True)
        >>> print(f"KC sparsity: {info['sparsity_fraction']:.2%}")
        >>> print(f"MBON mean±std: {info['mbon_mean']:.3f} ± {info['mbon_std']:.3f}")
        """
        # PN → KC with sparsity enforcement
        kc_activity = self.propagate_pn_to_kc(pn_activity)

        # KC → MBON linear readout
        mbon_output = self.propagate_kc_to_mbon(kc_activity)

        if not return_intermediates:
            return mbon_output

        # Compute diagnostics
        sparsity_fraction = self.compute_kc_sparsity_fraction(kc_activity)
        diagnostics = {
            "kc_activity": kc_activity,
            "sparsity_fraction": sparsity_fraction,
            "mbon_mean": float(np.mean(mbon_output)),
            "mbon_std": float(np.std(mbon_output)),
        }

        return mbon_output, diagnostics

    def activate_pns_by_glomeruli(
        self,
        glomeruli: List[str],
        firing_rate: float = 1.0,
    ) -> np.ndarray:
        """Create PN activity vector from glomerulus names.

        Biological Rationale
        --------------------
        Each odorant activates a combinatorial subset of olfactory receptors,
        which project to specific glomeruli in the antennal lobe. This method
        enables odor-driven simulations where we specify which glomeruli respond
        (e.g., "activate DA1 and DL3 for ethyl butyrate presentation").

        In real olfaction:
        - Odorants bind multiple receptor types with varying affinities
        - ORNs (olfactory receptor neurons) convert binding into spike rates
        - PNs inherit this combinatorial code via glomerulus connectivity

        Here we simplify by setting uniform firing rates for specified glomeruli,
        modeling saturating odor concentrations where receptor binding is maximal.

        Parameters
        ----------
        glomeruli : List[str]
            Glomerulus names to activate (e.g., ["DA1", "DL3", "VA1d"]).
            PNs belonging to these glomeruli will fire at `firing_rate`;
            all other PNs remain silent (0.0).
        firing_rate : float, optional
            Firing rate for activated PNs. Typically 1.0 (maximal response).
            Default: 1.0

        Returns
        -------
        np.ndarray
            PN activity vector. Shape: (n_pn,).
            Entries corresponding to specified glomeruli = firing_rate;
            all other entries = 0.0.

        Example
        -------
        >>> # Simulate presentation of ethyl butyrate (activates DA1, DL3)
        >>> pn_input = circuit.activate_pns_by_glomeruli(["DA1", "DL3"], firing_rate=1.0)
        >>> n_active_pns = np.count_nonzero(pn_input)
        >>> print(f"Activated {n_active_pns} PNs across 2 glomeruli")
        """
        pn_activity = np.zeros(self.connectivity.n_pn, dtype=np.float64)

        # Get PN indices for specified glomeruli
        pn_indices = self.connectivity.get_pn_indices(glomeruli=glomeruli)

        # Set firing rate for these PNs
        pn_activity[pn_indices] = firing_rate

        return pn_activity

    def compute_kc_sparsity_fraction(self, kc_activity: np.ndarray) -> float:
        """Compute fraction of KCs with non-zero activity.

        Biological Interpretation
        --------------------------
        In vivo calcium imaging experiments measure KC sparsity as the fraction
        of KCs responding (ΔF/F > threshold) to an odor. This metric quantifies
        the expansion recoding efficiency:

        - High sparsity (~5%): efficient pattern separation, low interference
        - Low sparsity (~50%+): poor separation, high interference

        Our target of 5% matches published Drosophila MB imaging data.

        Parameters
        ----------
        kc_activity : np.ndarray
            KC activity vector. Shape: (n_kc,).

        Returns
        -------
        float
            Fraction of KCs with activity > 0. Range: [0, 1].

        Example
        -------
        >>> kc_act = circuit.propagate_pn_to_kc(pn_input)
        >>> sparsity = circuit.compute_kc_sparsity_fraction(kc_act)
        >>> print(f"KC sparsity: {sparsity:.2%} (target: {circuit.sparsity_target:.2%})")
        """
        n_active = np.count_nonzero(kc_activity)
        return n_active / len(kc_activity)

    def _apply_k_winners_take_all(
        self,
        activations: np.ndarray,
        k: int,
        apply_to_connected_only: bool = True,
    ) -> np.ndarray:
        """Keep top-k activations, zero the rest (k-WTA lateral inhibition).

        Biological Rationale
        --------------------
        The MB circuit implements lateral inhibition via the GABAergic APL neuron
        (Anterior Paired Lateral). APL provides global feedback inhibition to
        all KCs:

        1. Strong PN input → some KCs depolarize
        2. Depolarized KCs activate APL (feedforward excitation)
        3. APL releases GABA onto all KCs (feedback inhibition)
        4. Weakly-depolarized KCs are suppressed below spike threshold
        5. Only strongly-depolarized KCs (top ~5%) escape inhibition and fire

        This dynamic threshold mechanism is approximated here as a static k-WTA
        operation: rank KCs by activation strength, keep top k, zero the rest.

        **Critical Fix (2025-10-30)**: When apply_to_connected_only=True (default),
        k-WTA is applied only to KCs receiving input (non-zero activation). This
        ensures that sparsity target (e.g., 5%) is maintained even when sparse
        connectivity means only a subset of KCs receive PN input for a given odor.

        Biological interpretation: APL inhibition suppresses weakly-responding KCs
        among those detecting the odor, not globally across all KCs in the brain.

        Implementation Details
        ----------------------
        Uses np.partition for O(n) complexity instead of O(n log n) sorting.
        This is efficient for large KC populations (~5000+ neurons).

        Parameters
        ----------
        activations : np.ndarray
            Raw KC activations before sparsity. Shape: (n_kc,).
        k : int
            Number of winners to keep active. Typically 0.05 * n_kc.
        apply_to_connected_only : bool, optional
            If True, apply k-WTA only to KCs with non-zero activation.
            This maintains target sparsity when connectivity is sparse.
            Default: True (recommended for biological realism).

        Returns
        -------
        np.ndarray
            Sparse KC activity. Shape: (n_kc,).
            Top-k entries retain original values; remaining entries = 0.0.

        Notes
        -----
        Edge cases:
        - If k >= n_kc_connected: all connected KCs retained
        - If k <= 0: all activations zeroed (complete silence)
        - Ties at threshold: implementation-defined which KCs are kept
          (np.partition behavior); biological equivalent is stochastic

        Example
        -------
        >>> h_raw = np.random.rand(5000)  # Random KC activations
        >>> h_sparse = circuit._apply_k_winners_take_all(h_raw, k=250)  # Keep top 5%
        >>> n_active = np.count_nonzero(h_sparse)
        >>> print(f"Active KCs: {n_active} (target: 250)")
        """
        if k <= 0:
            # No winners → complete silence
            return np.zeros_like(activations, dtype=np.float64)

        # NEW: Apply k-WTA only to KCs with non-zero input (biologically realistic)
        if apply_to_connected_only:
            # Find KCs receiving input
            connected_mask = activations > 0
            n_connected = np.sum(connected_mask)

            if n_connected == 0:
                # No KCs activated at all
                return np.zeros_like(activations, dtype=np.float64)

            if k >= n_connected:
                # Keep all connected KCs (fewer than target)
                return activations.copy()

            # Apply k-WTA to connected KCs only
            connected_activations = activations[connected_mask]
            threshold = np.partition(connected_activations, -k)[-k]

            # Keep activations >= threshold, zero the rest
            sparse_activations = np.where(activations >= threshold, activations, 0.0)

            return sparse_activations

        # ORIGINAL: Apply k-WTA globally (can result in sub-target sparsity)
        if k >= len(activations):
            # All winners → no sparsity
            return activations.copy()

        # Find the k-th largest value using partition (O(n) complexity)
        threshold = np.partition(activations, -k)[-k]

        # Keep activations >= threshold, zero the rest
        sparse_activations = np.where(activations >= threshold, activations, 0.0)

        return sparse_activations


