"""Connectivity matrix representation for Drosophila olfactory circuit.

This module provides an immutable, biologically-grounded representation of the
PN→KC→MBON circuit extracted from FlyWire FAFB v783 connectome data. The design
prioritizes biological realism while maintaining computational efficiency through
sparse matrix representations.

Biological Context
------------------
The Drosophila mushroom body (MB) implements a three-layer circuit for olfactory
learning and memory:

1. **Projection Neurons (PNs)**: ~150-200 olfactory PNs per hemisphere, each
   tuned to specific odorant receptors and organized by glomerulus identity in
   the antennal lobe. PNs provide combinatorial odor codes to Kenyon cells.

2. **Kenyon Cells (KCs)**: ~2000-2500 intrinsic MB neurons per hemisphere that
   perform sparse expansion coding. Each KC receives inputs from ~6-8 random PNs
   (biological sparsity ~3-5% of possible connections), enabling high-dimensional
   odor representations that support pattern separation and generalization.

3. **Mushroom Body Output Neurons (MBONs)**: ~34 cell types that read out KC
   activity to drive behavioral responses. MBON responses are modulated by
   dopaminergic neurons (DANs) during associative learning.

The connectivity matrices here preserve the biological sparse connectivity patterns
observed in the FlyWire connectome, enabling downstream simulations to respect the
anatomical constraints that shape olfactory processing and learning dynamics.

Example
-------
>>> from data_loaders.circuit_loader import CircuitLoader
>>> loader = CircuitLoader(cache_dir="data/cache")
>>> conn_matrix = loader.load_connectivity_matrix(normalize_weights="row")
>>> print(f"PN→KC sparsity: {conn_matrix.pn_to_kc.nnz / conn_matrix.pn_to_kc.size:.2%}")
>>> print(f"KC receiving PN input: {len(conn_matrix.pn_ids)}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class ConnectivityMatrix:
    """Immutable connectivity matrix for PN→KC→MBON circuit.

    This class encapsulates the structural connectivity of the Drosophila mushroom
    body olfactory circuit, storing synapse weights as sparse matrices to reflect
    the biological reality that most neuron pairs are not connected.

    Biological Design Principles
    -----------------------------
    1. **Immutability**: Connectivity is frozen after development (frozen=True).
       Adult fly MB connectivity is stable; plasticity occurs at synaptic weights
       (KC→MBON), not connectivity patterns.

    2. **Sparsity**: Biological circuits exhibit sparse connectivity (~1-5% density).
       Using scipy.sparse.csr_matrix reduces memory from O(n²) to O(k) where k is
       the number of actual synapses, matching biological reality.

    3. **Glomerular organization**: PNs are organized by odorant receptor identity
       (glomerulus), which determines their odor tuning. This metadata enables
       biologically-grounded input patterns.

    4. **KC subtype diversity**: KCs are subdivided into α/β, α'/β', and γ subtypes
       that project to different MB lobes and support distinct memory timescales
       (e.g., γ for short-term, α/β for long-term memory).

    Attributes
    ----------
    pn_ids : np.ndarray
        1D array of PN neuron IDs (int64). Shape: (n_pn,)
        Typically ~150-500 olfactory projection neurons depending on hemisphere
        and filtering criteria.
    kc_ids : np.ndarray
        1D array of KC neuron IDs (int64). Shape: (n_kc,)
        Expected ~5000-5500 Kenyon cells across all subtypes.
    mbon_ids : np.ndarray
        1D array of MBON neuron IDs (int64). Shape: (n_mbon,)
        Expected ~40-100 mushroom body output neurons.
    dan_ids : np.ndarray
        1D array of DAN neuron IDs (int64). Shape: (n_dan,)
        Expected ~200-600 dopaminergic neurons with MB innervation.
    pn_to_kc : sp.csr_matrix
        Sparse matrix of PN→KC synaptic weights. Shape: (n_kc, n_pn)
        Each row is a KC; each column is a PN. Non-zero entries represent
        synaptic connections. Biological constraint: each KC receives from
        ~6-8 PNs (fan-in), resulting in ~3-5% row sparsity.
    kc_to_mbon : sp.csr_matrix
        Sparse matrix of KC→MBON synaptic weights. Shape: (n_mbon, n_kc)
        Each row is an MBON; each column is a KC. MBONs integrate across
        hundreds of KCs to compute odor valence and behavioral outputs.
    dan_to_kc : sp.csr_matrix
        Sparse matrix of DAN→KC synaptic weights. Shape: (n_kc, n_dan)
        DANs provide compartmentalized dopamine signals that gate plasticity
        at KC→MBON synapses (not represented here; used in learning models).
    dan_to_mbon : sp.csr_matrix
        Sparse matrix of DAN→MBON synaptic weights. Shape: (n_mbon, n_dan)
        Direct DAN→MBON connections provide fast valence signaling that
        bypasses KC processing.
    pn_glomeruli : Dict[int, str]
        Mapping from PN neuron ID to glomerulus name (e.g., "DA1", "DL3").
        Glomeruli define odorant receptor specificity; PNs in the same
        glomerulus respond to similar chemical features.
    kc_subtypes : Dict[int, str]
        Mapping from KC neuron ID to subtype label (e.g., "ab", "g_main", "apbp_ap1").
        Subtypes determine axon projection targets and memory timescales:
        - α/β (ab): long-term memory formation
        - α'/β' (apbp): intermediate-term memory
        - γ (g_main/g_dorsal/g_sparse): short-term memory
    mbon_neuropils : Dict[int, List[str]]
        Mapping from MBON neuron ID to list of input neuropil regions.
        Neuropils define spatial compartments in MB lobes (e.g., calyx, peduncle,
        vertical/horizontal lobes) that segregate memory traces.
    dan_neuropils : Dict[int, List[str]]
        Mapping from DAN neuron ID to list of output neuropil regions.
        DANs are compartmentalized by target region (calyx, lobes), enabling
        spatially-specific reinforcement signals.

    Notes
    -----
    Matrix shapes follow the convention (output_neurons, input_neurons) to enable
    efficient matrix-vector multiplication: `output_activity = W @ input_activity`.

    All sparse matrices use CSR (Compressed Sparse Row) format for efficient
    row-slicing and matrix-vector products, which are the primary operations
    during forward propagation through the circuit.
    """

    # Neuron IDs (1D arrays)
    pn_ids: np.ndarray
    kc_ids: np.ndarray
    mbon_ids: np.ndarray
    dan_ids: np.ndarray

    # Connectivity matrices (sparse CSR format)
    pn_to_kc: sp.csr_matrix
    kc_to_mbon: sp.csr_matrix
    dan_to_kc: sp.csr_matrix
    dan_to_mbon: sp.csr_matrix

    # Metadata dictionaries
    pn_glomeruli: Dict[int, str] = field(default_factory=dict)
    kc_subtypes: Dict[int, str] = field(default_factory=dict)
    mbon_neuropils: Dict[int, List[str]] = field(default_factory=dict)
    dan_neuropils: Dict[int, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate connectivity matrix shapes and biological constraints."""
        # Validate array types
        if not isinstance(self.pn_ids, np.ndarray):
            raise TypeError("pn_ids must be numpy array")
        if not isinstance(self.kc_ids, np.ndarray):
            raise TypeError("kc_ids must be numpy array")
        if not isinstance(self.mbon_ids, np.ndarray):
            raise TypeError("mbon_ids must be numpy array")
        if not isinstance(self.dan_ids, np.ndarray):
            raise TypeError("dan_ids must be numpy array")

        # Validate sparse matrix types
        if not sp.issparse(self.pn_to_kc) or not isinstance(self.pn_to_kc, sp.csr_matrix):
            raise TypeError("pn_to_kc must be scipy.sparse.csr_matrix")
        if not sp.issparse(self.kc_to_mbon) or not isinstance(self.kc_to_mbon, sp.csr_matrix):
            raise TypeError("kc_to_mbon must be scipy.sparse.csr_matrix")
        if not sp.issparse(self.dan_to_kc) or not isinstance(self.dan_to_kc, sp.csr_matrix):
            raise TypeError("dan_to_kc must be scipy.sparse.csr_matrix")
        if not sp.issparse(self.dan_to_mbon) or not isinstance(self.dan_to_mbon, sp.csr_matrix):
            raise TypeError("dan_to_mbon must be scipy.sparse.csr_matrix")

        # Validate matrix dimensions match neuron ID arrays
        n_pn, n_kc, n_mbon, n_dan = len(self.pn_ids), len(self.kc_ids), len(self.mbon_ids), len(self.dan_ids)

        if self.pn_to_kc.shape != (n_kc, n_pn):
            raise ValueError(
                f"pn_to_kc shape {self.pn_to_kc.shape} does not match "
                f"expected (n_kc={n_kc}, n_pn={n_pn})"
            )
        if self.kc_to_mbon.shape != (n_mbon, n_kc):
            raise ValueError(
                f"kc_to_mbon shape {self.kc_to_mbon.shape} does not match "
                f"expected (n_mbon={n_mbon}, n_kc={n_kc})"
            )
        if self.dan_to_kc.shape != (n_kc, n_dan):
            raise ValueError(
                f"dan_to_kc shape {self.dan_to_kc.shape} does not match "
                f"expected (n_kc={n_kc}, n_dan={n_dan})"
            )
        if self.dan_to_mbon.shape != (n_mbon, n_dan):
            raise ValueError(
                f"dan_to_mbon shape {self.dan_to_mbon.shape} does not match "
                f"expected (n_mbon={n_mbon}, n_dan={n_dan})"
            )

    @property
    def n_pn(self) -> int:
        """Number of projection neurons."""
        return len(self.pn_ids)

    @property
    def n_kc(self) -> int:
        """Number of Kenyon cells."""
        return len(self.kc_ids)

    @property
    def n_mbon(self) -> int:
        """Number of mushroom body output neurons."""
        return len(self.mbon_ids)

    @property
    def n_dan(self) -> int:
        """Number of dopaminergic neurons."""
        return len(self.dan_ids)

    def slice_kc_subtypes(self, subtypes: List[str]) -> "ConnectivityMatrix":
        """Return new ConnectivityMatrix with only specified KC subtypes.

        Biological Rationale
        --------------------
        Different KC subtypes support distinct memory functions in Drosophila:
        - γ neurons: short-term memory (minutes to hours)
        - α/β neurons: long-term memory (days)
        - α'/β' neurons: intermediate timescales

        This method enables experiments that isolate specific memory systems,
        mimicking genetic ablation studies where specific KC subtypes are
        silenced to assess their behavioral contributions.

        Parameters
        ----------
        subtypes : List[str]
            KC subtype labels to retain (e.g., ["ab", "g_main"]).
            Valid subtypes: "ab", "ab_p", "apbp_main", "apbp_ap1", "apbp_ap2",
            "g_main", "g_dorsal", "g_sparse".

        Returns
        -------
        ConnectivityMatrix
            New matrix with only KCs of specified subtypes. PN and MBON
            populations remain unchanged; only KC-related matrices are filtered.

        Raises
        ------
        ValueError
            If no KCs match the specified subtypes.

        Example
        -------
        >>> # Isolate γ Kenyon cells for short-term memory simulation
        >>> gamma_matrix = conn_matrix.slice_kc_subtypes(["g_main", "g_dorsal"])
        >>> print(f"Retained {gamma_matrix.n_kc} / {conn_matrix.n_kc} KCs")
        """
        # Identify KCs belonging to requested subtypes
        kc_mask = np.array([
            self.kc_subtypes.get(kc_id, "") in subtypes
            for kc_id in self.kc_ids
        ], dtype=bool)

        if not np.any(kc_mask):
            raise ValueError(
                f"No KCs found for subtypes {subtypes}. "
                f"Available subtypes: {set(self.kc_subtypes.values())}"
            )

        # Filter KC IDs and subtype metadata
        filtered_kc_ids = self.kc_ids[kc_mask]
        filtered_kc_subtypes = {
            kc_id: subtype
            for kc_id, subtype in self.kc_subtypes.items()
            if subtype in subtypes
        }

        # Filter connectivity matrices (slice rows/columns corresponding to KCs)
        # PN→KC: slice rows (output dimension)
        filtered_pn_to_kc = self.pn_to_kc[kc_mask, :]

        # KC→MBON: slice columns (input dimension)
        filtered_kc_to_mbon = self.kc_to_mbon[:, kc_mask]

        # DAN→KC: slice rows (output dimension)
        filtered_dan_to_kc = self.dan_to_kc[kc_mask, :]

        # Return new immutable ConnectivityMatrix
        return ConnectivityMatrix(
            pn_ids=self.pn_ids.copy(),  # PNs unchanged
            kc_ids=filtered_kc_ids,
            mbon_ids=self.mbon_ids.copy(),  # MBONs unchanged
            dan_ids=self.dan_ids.copy(),  # DANs unchanged
            pn_to_kc=filtered_pn_to_kc,
            kc_to_mbon=filtered_kc_to_mbon,
            dan_to_kc=filtered_dan_to_kc,
            dan_to_mbon=self.dan_to_mbon.copy(),  # DAN→MBON unchanged
            pn_glomeruli=self.pn_glomeruli.copy(),
            kc_subtypes=filtered_kc_subtypes,
            mbon_neuropils=self.mbon_neuropils.copy(),
            dan_neuropils=self.dan_neuropils.copy(),
        )

    def pn_fan_in(self, kc_index: int) -> np.ndarray:
        """Return PN→KC input weights for a single Kenyon cell.

        Biological Rationale
        --------------------
        Each KC receives inputs from ~6-8 randomly selected PNs (the "claw"),
        a connectivity motif that enables sparse, high-dimensional odor codes.
        This random connectivity supports pattern separation: similar odors
        activate different KC ensembles, reducing interference in memory storage.

        Parameters
        ----------
        kc_index : int
            Index into kc_ids array (not neuron ID). Range: [0, n_kc).

        Returns
        -------
        np.ndarray
            1D array of PN→KC weights for this KC. Shape: (n_pn,)
            Most entries are zero; ~6-8 entries are non-zero (synaptic weights).

        Raises
        ------
        IndexError
            If kc_index is out of bounds.

        Example
        -------
        >>> kc_idx = 100
        >>> pn_inputs = conn_matrix.pn_fan_in(kc_idx)
        >>> n_inputs = np.count_nonzero(pn_inputs)
        >>> print(f"KC {kc_idx} receives input from {n_inputs} PNs")
        """
        if kc_index < 0 or kc_index >= self.n_kc:
            raise IndexError(
                f"kc_index {kc_index} out of bounds for {self.n_kc} KCs"
            )

        # Extract row from sparse matrix and convert to dense 1D array
        return self.pn_to_kc[kc_index, :].toarray().ravel()

    def mbon_fan_in(self, mbon_index: int) -> np.ndarray:
        """Return KC→MBON input weights for a single MBON.

        Biological Rationale
        --------------------
        Each MBON integrates activity across hundreds to thousands of KCs,
        computing a weighted sum that represents learned odor valence. The
        KC→MBON synapses are the primary site of associative plasticity:
        dopamine released by DANs during reward/punishment modulates these
        weights, implementing a three-factor Hebbian learning rule
        (presynaptic KC activity × postsynaptic MBON activity × dopamine).

        Parameters
        ----------
        mbon_index : int
            Index into mbon_ids array (not neuron ID). Range: [0, n_mbon).

        Returns
        -------
        np.ndarray
            1D array of KC→MBON weights for this MBON. Shape: (n_kc,)
            Typically ~10-30% of entries are non-zero, reflecting that MBONs
            sample broadly across the KC population.

        Raises
        ------
        IndexError
            If mbon_index is out of bounds.

        Example
        -------
        >>> mbon_idx = 5
        >>> kc_inputs = conn_matrix.mbon_fan_in(mbon_idx)
        >>> n_inputs = np.count_nonzero(kc_inputs)
        >>> print(f"MBON {mbon_idx} receives input from {n_inputs} KCs")
        """
        if mbon_index < 0 or mbon_index >= self.n_mbon:
            raise IndexError(
                f"mbon_index {mbon_index} out of bounds for {self.n_mbon} MBONs"
            )

        # Extract row from sparse matrix and convert to dense 1D array
        return self.kc_to_mbon[mbon_index, :].toarray().ravel()

    def get_pn_indices(self, glomeruli: Optional[List[str]] = None) -> np.ndarray:
        """Return PN indices optionally filtered by glomerulus identity.

        Biological Rationale
        --------------------
        Glomeruli define odorant receptor specificity. PNs within a glomerulus
        inherit the tuning of their upstream olfactory receptor neurons (ORNs),
        responding to specific molecular features. This method enables targeted
        activation of PNs by chemical structure (e.g., "activate all PNs tuned
        to acetate esters" by selecting DA1, VA1d glomeruli).

        Parameters
        ----------
        glomeruli : Optional[List[str]]
            List of glomerulus names to filter (e.g., ["DA1", "DL3"]).
            If None, returns all PN indices.

        Returns
        -------
        np.ndarray
            1D array of PN indices (positions in pn_ids array).

        Example
        -------
        >>> # Get PNs responding to a specific odorant receptor
        >>> da1_pns = conn_matrix.get_pn_indices(glomeruli=["DA1"])
        >>> print(f"DA1 glomerulus has {len(da1_pns)} PNs")
        """
        if glomeruli is None:
            return np.arange(self.n_pn)

        # Filter PNs by glomerulus membership
        glomeruli_set = set(glomeruli)
        pn_mask = np.array([
            self.pn_glomeruli.get(pn_id, "") in glomeruli_set
            for pn_id in self.pn_ids
        ], dtype=bool)

        return np.where(pn_mask)[0]

    def pn_to_kc_sparsity(self) -> float:
        """Compute fraction of zero entries in PN→KC connectivity matrix.

        Biological Rationale
        --------------------
        The PN→KC projection is highly sparse (~95-97% of connections absent),
        reflecting the random "claw" connectivity where each KC samples only
        6-8 of ~150 PNs. This sparsity is a hallmark of expansion coding in
        the mushroom body, enabling high-dimensional odor representations that
        support efficient associative learning.

        Returns
        -------
        float
            Sparsity fraction in [0, 1]. Value of 0.96 means 96% of possible
            PN→KC connections are absent (only 4% present).

        Example
        -------
        >>> sparsity = conn_matrix.pn_to_kc_sparsity()
        >>> density = 1.0 - sparsity
        >>> print(f"PN→KC connectivity density: {density:.2%}")
        """
        total_possible = self.n_kc * self.n_pn
        if total_possible == 0:
            return 1.0
        return 1.0 - (self.pn_to_kc.nnz / total_possible)

    def kc_to_mbon_sparsity(self) -> float:
        """Compute fraction of zero entries in KC→MBON connectivity matrix.

        Biological Rationale
        --------------------
        KC→MBON connectivity is also sparse but less so than PN→KC (~90-95%).
        Each MBON samples broadly across KCs to integrate odor information,
        while each KC projects to a subset of MBONs, creating a distributed
        readout that supports robust odor classification.

        Returns
        -------
        float
            Sparsity fraction in [0, 1].

        Example
        -------
        >>> sparsity = conn_matrix.kc_to_mbon_sparsity()
        >>> print(f"KC→MBON sparsity: {sparsity:.2%}")
        """
        total_possible = self.n_mbon * self.n_kc
        if total_possible == 0:
            return 1.0
        return 1.0 - (self.kc_to_mbon.nnz / total_possible)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize connectivity matrix to JSON-safe dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing neuron counts, sparsity metrics, and shapes.
            Does not include full matrices (too large); use for inspection only.

        Example
        -------
        >>> import json
        >>> summary = conn_matrix.to_dict()
        >>> print(json.dumps(summary, indent=2))
        """
        return {
            "n_pn": self.n_pn,
            "n_kc": self.n_kc,
            "n_mbon": self.n_mbon,
            "n_dan": self.n_dan,
            "pn_to_kc_shape": self.pn_to_kc.shape,
            "kc_to_mbon_shape": self.kc_to_mbon.shape,
            "dan_to_kc_shape": self.dan_to_kc.shape,
            "dan_to_mbon_shape": self.dan_to_mbon.shape,
            "pn_to_kc_nnz": self.pn_to_kc.nnz,
            "kc_to_mbon_nnz": self.kc_to_mbon.nnz,
            "dan_to_kc_nnz": self.dan_to_kc.nnz,
            "dan_to_mbon_nnz": self.dan_to_mbon.nnz,
            "pn_to_kc_sparsity": self.pn_to_kc_sparsity(),
            "kc_to_mbon_sparsity": self.kc_to_mbon_sparsity(),
            "n_glomeruli": len(set(self.pn_glomeruli.values())),
            "kc_subtypes": list(set(self.kc_subtypes.values())),
        }

    def __repr__(self) -> str:
        """Return summary string for debugging and logging."""
        return (
            f"ConnectivityMatrix(\n"
            f"  PNs: {self.n_pn}, KCs: {self.n_kc}, MBONs: {self.n_mbon}, DANs: {self.n_dan}\n"
            f"  PN→KC: {self.pn_to_kc.shape} ({self.pn_to_kc.nnz} synapses, "
            f"{self.pn_to_kc_sparsity():.1%} sparse)\n"
            f"  KC→MBON: {self.kc_to_mbon.shape} ({self.kc_to_mbon.nnz} synapses, "
            f"{self.kc_to_mbon_sparsity():.1%} sparse)\n"
            f"  Glomeruli: {len(set(self.pn_glomeruli.values()))}, "
            f"KC subtypes: {len(set(self.kc_subtypes.values()))}\n"
            f")"
        )
