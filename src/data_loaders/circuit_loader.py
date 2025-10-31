"""Circuit loader for constructing ConnectivityMatrix from cached FlyWire data.

This module provides a factory class that loads connectome data from disk and
instantiates ConnectivityMatrix objects with biologically-motivated validation
and normalization. The loader handles the complexity of merging multiple data
sources (parquet files, CSVs for different neuron subtypes) into a unified
sparse matrix representation.

Biological Context
------------------
The FlyWire FAFB v783 connectome provides synapse-level connectivity for the
adult Drosophila brain. This loader extracts the olfactory learning circuit
(PN→KC→MBON + DAN modulation) and applies biological constraints:

1. **Synapse-based weights**: Raw synapse counts reflect biological connection
   strength. Normalization strategies (row/global/none) enable different
   interpretations of synaptic integration.

2. **Sparse connectivity**: The MB circuit exhibits ~95-97% sparsity at PN→KC
   (random claw connectivity) and ~90-95% at KC→MBON (distributed readout).
   CSR sparse matrices preserve this structure efficiently.

3. **KC subtype organization**: KCs are segregated into 8 anatomical subtypes
   (ab, ab_p, apbp_main, apbp_ap1, apbp_ap2, g_main, g_dorsal, g_sparse) that
   project to different MB lobes and support distinct memory timescales.

4. **Glomerular code**: PN glomerulus labels define odorant receptor specificity,
   enabling odor-driven simulations that respect biological chemotopy.

Example
-------
>>> from data_loaders.circuit_loader import CircuitLoader
>>> loader = CircuitLoader(cache_dir="data/cache")
>>>
>>> # Load with row normalization (each KC sees inputs summing to 1.0)
>>> conn_matrix = loader.load_connectivity_matrix(normalize_weights="row")
>>>
>>> # Validate connectivity structure
>>> report = loader.validate_connectivity(conn_matrix)
>>> print(f"PN→KC fan-in (median): {report['pn_to_kc_fan_in']['p50']:.1f} PNs/KC")
>>> print(f"KC→MBON fan-in (median): {report['kc_to_mbon_fan_in']['p50']:.1f} KCs/MBON")
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pgcn.models.connectivity_matrix import ConnectivityMatrix


class CircuitLoader:
    """Factory for building ConnectivityMatrix from FlyWire cache.

    This class encapsulates the logic for loading cached connectome data,
    merging neuron metadata, constructing sparse adjacency matrices, and
    applying biologically-motivated weight normalization. It validates that
    loaded data conforms to expected connectivity patterns observed in the
    Drosophila mushroom body.

    Parameters
    ----------
    cache_dir : str, optional
        Path to directory containing cached parquet and CSV files.
        Default: "data/cache"

    Attributes
    ----------
    cache_dir : Path
        Resolved path to cache directory.

    Raises
    ------
    FileNotFoundError
        If cache_dir does not exist or required files are missing.

    Notes
    -----
    The cache directory should contain:
    - nodes.parquet: neuron metadata (id, type, glomerulus, etc.)
    - edges.parquet: PN→KC and KC→MBON connectivity
    - dan_edges.parquet: DAN→KC and DAN→MBON connectivity
    - kc_*.csv: KC subtype assignments (8 files)
    - alpn_extracted.csv: PN glomerulus labels
    - mbon_all.csv: MBON metadata with neuropil annotations
    - dan_all.csv: DAN metadata with neuropil annotations
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        """Initialize CircuitLoader with cache directory.

        Parameters
        ----------
        cache_dir : str
            Path to cache directory containing connectome data.
        """
        self.cache_dir = Path(cache_dir)

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_dir}\n"
                f"Run extraction scripts to populate cache, or use sample data generation."
            )

    def load_connectivity_matrix(
        self,
        normalize_weights: str = "row",
        include_dan: bool = True,
        kc_subtypes_filter: Optional[List[str]] = None,
    ) -> ConnectivityMatrix:
        """Load full connectivity matrix from cache files.

        This method orchestrates loading of all connectome data sources,
        constructs sparse matrices, applies weight normalization, and
        validates the resulting connectivity structure.

        Biological Rationale
        --------------------
        **Weight normalization**: Synaptic integration in real neurons involves
        complex nonlinear dynamics, but three common approximations are:

        1. **"row" normalization**: Each postsynaptic neuron's inputs sum to 1.0,
           modeling synaptic normalization/scaling mechanisms that maintain stable
           firing rates despite variable input counts. This is appropriate for
           rate-coded models where we care about relative PN contributions.

        2. **"global" normalization**: Weights scaled by max synapse count,
           preserving relative differences in connection strength across the
           circuit. Useful when comparing strong vs. weak pathways.

        3. **"none"**: Raw synapse counts, reflecting anatomical connection
           strength. Appropriate for biophysical models that explicitly simulate
           synaptic currents.

        **KC subtype filtering**: Enables isolation of specific memory systems
        (e.g., γ for short-term, α/β for long-term), mimicking genetic ablation
        experiments.

        Parameters
        ----------
        normalize_weights : str, optional
            Weight normalization strategy. Options:
            - "row": each row sums to 1.0 (synaptic normalization)
            - "global": divide by max weight (preserve relative strength)
            - "none": raw synapse counts
            Default: "row"
        include_dan : bool, optional
            If True, load DAN→KC and DAN→MBON connectivity. If False,
            return empty sparse matrices for DAN pathways. Default: True
        kc_subtypes_filter : Optional[List[str]], optional
            If provided, retain only KCs of specified subtypes (e.g.,
            ["ab", "g_main"]). None retains all KCs. Default: None

        Returns
        -------
        ConnectivityMatrix
            Fully validated connectivity matrix with sparse CSR matrices
            and neuron metadata.

        Raises
        ------
        FileNotFoundError
            If required cache files are missing.
        ValueError
            If normalization strategy is invalid or no KCs match filter.

        Example
        -------
        >>> loader = CircuitLoader()
        >>> # Load γ KCs only (short-term memory)
        >>> gamma_conn = loader.load_connectivity_matrix(
        ...     normalize_weights="row",
        ...     kc_subtypes_filter=["g_main", "g_dorsal", "g_sparse"]
        ... )
        >>> print(f"Loaded {gamma_conn.n_kc} γ Kenyon cells")
        """
        # Validate normalization strategy
        valid_normalizations = {"row", "global", "none"}
        if normalize_weights not in valid_normalizations:
            raise ValueError(
                f"normalize_weights must be one of {valid_normalizations}, "
                f"got '{normalize_weights}'"
            )

        # Load neuron metadata
        nodes = self._load_nodes()
        edges, dan_edges = self._load_edges()

        # Extract neuron IDs by type
        pn_nodes = nodes[nodes["type"] == "PN"]
        kc_nodes = nodes[nodes["type"] == "KC"]
        mbon_nodes = nodes[nodes["type"] == "MBON"]
        dan_nodes = nodes[nodes["type"] == "DAN"]

        # Load KC subtype assignments and filter if requested
        kc_subtypes_df = self._load_kc_subtypes()

        # Filter KCs by subtype if requested
        if kc_subtypes_filter is not None:
            kc_subtypes_df = kc_subtypes_df[
                kc_subtypes_df["subtype"].isin(kc_subtypes_filter)
            ]
            if len(kc_subtypes_df) == 0:
                raise ValueError(
                    f"No KCs found for subtypes {kc_subtypes_filter}. "
                    f"Check kc_*.csv files in cache."
                )
            # Filter kc_nodes to match
            kc_nodes = kc_nodes[kc_nodes["node_id"].isin(kc_subtypes_df["root_id"])]

        # Extract neuron ID arrays
        pn_ids = pn_nodes["node_id"].values
        kc_ids = kc_nodes["node_id"].values
        mbon_ids = mbon_nodes["node_id"].values
        dan_ids = dan_nodes["node_id"].values

        # Build ID→index mappings for edge construction
        pn_id_to_idx = {nid: idx for idx, nid in enumerate(pn_ids)}
        kc_id_to_idx = {nid: idx for idx, nid in enumerate(kc_ids)}
        mbon_id_to_idx = {nid: idx for idx, nid in enumerate(mbon_ids)}
        dan_id_to_idx = {nid: idx for idx, nid in enumerate(dan_ids)}

        # Build PN→KC sparse matrix
        pn_to_kc = self._build_sparse_matrix(
            edges=edges,
            source_type="PN",
            target_type="KC",
            source_id_to_idx=pn_id_to_idx,
            target_id_to_idx=kc_id_to_idx,
            n_source=len(pn_ids),
            n_target=len(kc_ids),
            normalize=normalize_weights,
        )

        # Build KC→MBON sparse matrix
        kc_to_mbon = self._build_sparse_matrix(
            edges=edges,
            source_type="KC",
            target_type="MBON",
            source_id_to_idx=kc_id_to_idx,
            target_id_to_idx=mbon_id_to_idx,
            n_source=len(kc_ids),
            n_target=len(mbon_ids),
            normalize=normalize_weights,
        )

        # Build DAN matrices if requested
        if include_dan:
            dan_to_kc = self._build_sparse_matrix(
                edges=dan_edges,
                source_type="DAN",
                target_type="KC",
                source_id_to_idx=dan_id_to_idx,
                target_id_to_idx=kc_id_to_idx,
                n_source=len(dan_ids),
                n_target=len(kc_ids),
                normalize=normalize_weights,
            )
            dan_to_mbon = self._build_sparse_matrix(
                edges=dan_edges,
                source_type="DAN",
                target_type="MBON",
                source_id_to_idx=dan_id_to_idx,
                target_id_to_idx=mbon_id_to_idx,
                n_source=len(dan_ids),
                n_target=len(mbon_ids),
                normalize=normalize_weights,
            )
        else:
            # Empty sparse matrices
            dan_to_kc = sp.csr_matrix((len(kc_ids), len(dan_ids)), dtype=np.float64)
            dan_to_mbon = sp.csr_matrix((len(mbon_ids), len(dan_ids)), dtype=np.float64)

        # Load metadata dictionaries
        pn_glomeruli = self._load_glomeruli(pn_ids)
        kc_subtypes = self._load_kc_subtype_dict(kc_subtypes_df)
        mbon_neuropils, dan_neuropils = self._load_neuropils()

        # Filter neuropils to only include neurons in our ID arrays
        mbon_neuropils = {k: v for k, v in mbon_neuropils.items() if k in mbon_ids}
        dan_neuropils = {k: v for k, v in dan_neuropils.items() if k in dan_ids}

        # Construct ConnectivityMatrix
        conn_matrix = ConnectivityMatrix(
            pn_ids=pn_ids,
            kc_ids=kc_ids,
            mbon_ids=mbon_ids,
            dan_ids=dan_ids,
            pn_to_kc=pn_to_kc,
            kc_to_mbon=kc_to_mbon,
            dan_to_kc=dan_to_kc,
            dan_to_mbon=dan_to_mbon,
            pn_glomeruli=pn_glomeruli,
            kc_subtypes=kc_subtypes,
            mbon_neuropils=mbon_neuropils,
            dan_neuropils=dan_neuropils,
        )

        # Validate before returning
        self.validate_shapes(conn_matrix)

        return conn_matrix

    def _load_nodes(self) -> pd.DataFrame:
        """Load nodes.parquet with type validation.

        Returns
        -------
        pd.DataFrame
            Node metadata with columns: node_id, type, glomerulus, etc.

        Raises
        ------
        FileNotFoundError
            If nodes.parquet missing from cache.
        """
        nodes_path = self.cache_dir / "nodes.parquet"
        if not nodes_path.exists():
            raise FileNotFoundError(
                f"nodes.parquet not found at {nodes_path}. "
                f"Run extraction scripts to generate cache."
            )

        nodes = pd.read_parquet(nodes_path)

        # Validate required columns
        required_cols = {"node_id", "type"}
        if not required_cols.issubset(nodes.columns):
            raise ValueError(
                f"nodes.parquet missing required columns: {required_cols - set(nodes.columns)}"
            )

        return nodes

    def _load_edges(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load edges.parquet and dan_edges.parquet.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (edges, dan_edges) dataframes with columns: source_id, target_id, synapse_weight

        Raises
        ------
        FileNotFoundError
            If required edge files missing from cache.
        """
        edges_path = self.cache_dir / "edges.parquet"
        dan_edges_path = self.cache_dir / "dan_edges.parquet"

        if not edges_path.exists():
            raise FileNotFoundError(
                f"edges.parquet not found at {edges_path}. "
                f"Run extraction scripts to generate cache."
            )

        edges = pd.read_parquet(edges_path)

        # DAN edges optional (can run without DANs)
        if dan_edges_path.exists():
            dan_edges = pd.read_parquet(dan_edges_path)
        else:
            # Empty dataframe with correct schema
            dan_edges = pd.DataFrame(columns=["source_id", "target_id", "synapse_weight"])

        return edges, dan_edges

    def _load_kc_subtypes(self) -> pd.DataFrame:
        """Merge all kc_*.csv files into unified DataFrame with subtype labels.

        Biological Rationale
        --------------------
        KCs are anatomically segregated into subtypes based on their axon
        projections within the mushroom body lobes:
        - ab: α/β neurons (vertical/horizontal lobes, long-term memory)
        - ab_p: α/β posterior (specialized α/β variant)
        - apbp_*: α'/β' neurons (MB-V2 compartments, intermediate memory)
        - g_*: γ neurons (horizontal lobe only, short-term memory)

        Returns
        -------
        pd.DataFrame
            Merged KC metadata with columns: root_id, subtype, (other fields)

        Raises
        ------
        FileNotFoundError
            If no kc_*.csv files found in cache.
        """
        kc_file_pattern = str(self.cache_dir / "kc_*.csv")
        kc_files = glob.glob(kc_file_pattern)

        if not kc_files:
            raise FileNotFoundError(
                f"No KC subtype CSV files found matching {kc_file_pattern}. "
                f"Run scripts/extract_circuit.py to generate KC subtype files."
            )

        # Load and label each subtype file
        kc_dfs = []
        for file_path in kc_files:
            # Extract subtype from filename (e.g., "kc_ab.csv" → "ab")
            subtype = Path(file_path).stem.replace("kc_", "")

            df = pd.read_csv(file_path)
            if len(df) > 0:  # Skip empty files
                df["subtype"] = subtype
                kc_dfs.append(df)

        if not kc_dfs:
            raise ValueError(
                f"All KC subtype files are empty. Check extraction script output."
            )

        # Concatenate all subtype dataframes
        kc_subtypes = pd.concat(kc_dfs, ignore_index=True)

        # Validate root_id column
        if "root_id" not in kc_subtypes.columns:
            raise ValueError(
                "KC subtype CSVs missing 'root_id' column. Check extraction script."
            )

        return kc_subtypes

    def _load_glomeruli(self, pn_ids: np.ndarray) -> Dict[int, str]:
        """Extract glomerulus mappings from alpn_extracted.csv and nodes.parquet.

        Biological Rationale
        --------------------
        Glomeruli are anatomical units in the antennal lobe where olfactory
        receptor neurons (ORNs) synapse onto projection neurons. Each glomerulus
        corresponds to one odorant receptor type, defining the chemical tuning
        of its PNs. Glomerulus identity enables odor-driven simulations where
        we activate PNs based on which receptors an odorant stimulates.

        Parameters
        ----------
        pn_ids : np.ndarray
            PN neuron IDs to extract glomeruli for.

        Returns
        -------
        Dict[int, str]
            Mapping from PN neuron ID → glomerulus name (e.g., "DA1", "DL3").
            PNs without glomerulus assignments map to "unknown".
        """
        pn_glomeruli = {}

        # Load from alpn_extracted.csv if available
        alpn_path = self.cache_dir / "alpn_extracted.csv"
        if alpn_path.exists():
            alpn = pd.read_csv(alpn_path)
            if "root_id" in alpn.columns and "primary_glomerulus" in alpn.columns:
                for _, row in alpn.iterrows():
                    root_id = row["root_id"]
                    glomerulus = row.get("primary_glomerulus", "unknown")
                    if pd.notna(glomerulus) and glomerulus != "":
                        pn_glomeruli[root_id] = str(glomerulus)

        # Fallback to nodes.parquet glomerulus column
        nodes_path = self.cache_dir / "nodes.parquet"
        if nodes_path.exists():
            nodes = pd.read_parquet(nodes_path)
            if "glomerulus" in nodes.columns:
                pn_nodes = nodes[nodes["node_id"].isin(pn_ids)]
                for _, row in pn_nodes.iterrows():
                    root_id = row["node_id"]
                    glomerulus = row.get("glomerulus", "unknown")
                    if pd.notna(glomerulus) and glomerulus != "" and root_id not in pn_glomeruli:
                        pn_glomeruli[root_id] = str(glomerulus)

        # Fill in "unknown" for any PNs without glomerulus labels
        for pn_id in pn_ids:
            if pn_id not in pn_glomeruli:
                pn_glomeruli[pn_id] = "unknown"

        return pn_glomeruli

    def _load_kc_subtype_dict(self, kc_subtypes_df: pd.DataFrame) -> Dict[int, str]:
        """Convert KC subtype DataFrame to dictionary.

        Parameters
        ----------
        kc_subtypes_df : pd.DataFrame
            DataFrame with columns: root_id, subtype

        Returns
        -------
        Dict[int, str]
            Mapping from KC neuron ID → subtype label.
        """
        return dict(zip(kc_subtypes_df["root_id"], kc_subtypes_df["subtype"]))

    def _load_neuropils(self) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
        """Load MBON and DAN neuropil annotations.

        Biological Rationale
        --------------------
        Neuropils define anatomical compartments where neurons arborize (send
        dendrites/axons). In the MB:
        - MBON input neuropils indicate which MB lobe compartments they sample
          (e.g., calyx, vertical lobe, horizontal lobe).
        - DAN output neuropils indicate where they release dopamine, defining
          which KC→MBON synapses are eligible for plasticity.

        This spatial organization segregates different memory types and valence
        signals (e.g., PPL1 DANs target punishment pathways, PAM DANs target reward).

        Returns
        -------
        Tuple[Dict[int, List[str]], Dict[int, List[str]]]
            (mbon_neuropils, dan_neuropils)
            Each maps neuron ID → list of neuropil region names.
        """
        mbon_neuropils = {}
        dan_neuropils = {}

        # Load MBON neuropils from mbon_all.csv
        mbon_path = self.cache_dir / "mbon_all.csv"
        if mbon_path.exists():
            mbon = pd.read_csv(mbon_path)
            if "root_id" in mbon.columns and "input_neuropils" in mbon.columns:
                for _, row in mbon.iterrows():
                    root_id = row["root_id"]
                    neuropils_str = row.get("input_neuropils", "")
                    if pd.notna(neuropils_str) and neuropils_str != "":
                        # Parse pipe-delimited list
                        neuropils = [n.strip() for n in str(neuropils_str).split("|")]
                        mbon_neuropils[root_id] = neuropils
                    else:
                        mbon_neuropils[root_id] = []

        # Load DAN neuropils from dan_mb.csv (MB-only filter applied)
        # Falls back to dan_all.csv if dan_mb.csv doesn't exist
        dan_path = self.cache_dir / "dan_mb.csv"
        if not dan_path.exists():
            dan_path = self.cache_dir / "dan_all.csv"
            if dan_path.exists():
                print(
                    "WARNING: Using dan_all.csv (includes non-MB DANs). "
                    "Run circuit extraction with MB filter to generate dan_mb.csv"
                )

        if dan_path.exists():
            dan = pd.read_csv(dan_path)
            if "root_id" in dan.columns and "output_neuropils" in dan.columns:
                for _, row in dan.iterrows():
                    root_id = row["root_id"]
                    neuropils_str = row.get("output_neuropils", "")
                    if pd.notna(neuropils_str) and neuropils_str != "":
                        # Parse pipe-delimited list
                        neuropils = [n.strip() for n in str(neuropils_str).split("|")]
                        dan_neuropils[root_id] = neuropils
                    else:
                        dan_neuropils[root_id] = []

        return mbon_neuropils, dan_neuropils

    def _build_sparse_matrix(
        self,
        edges: pd.DataFrame,
        source_type: str,
        target_type: str,
        source_id_to_idx: Dict[int, int],
        target_id_to_idx: Dict[int, int],
        n_source: int,
        n_target: int,
        normalize: str,
    ) -> sp.csr_matrix:
        """Convert edge list to sparse CSR matrix with normalization.

        Biological Rationale
        --------------------
        Synaptic weights in the connectome are represented as synapse counts
        (anatomical connection strength). Different normalization strategies
        reflect different biophysical assumptions:

        - **Row normalization**: Models synaptic scaling where each neuron
          maintains stable total input strength. This reflects homeostatic
          plasticity mechanisms observed in real neurons.

        - **Global normalization**: Preserves relative differences in connection
          strength across the circuit, useful for comparing strong vs. weak
          pathways.

        - **No normalization**: Raw synapse counts, appropriate when modeling
          synaptic currents directly.

        Parameters
        ----------
        edges : pd.DataFrame
            Edge list with columns: source_id, target_id, synapse_weight
        source_type : str
            Source neuron type (e.g., "PN", "KC", "DAN")
        target_type : str
            Target neuron type (e.g., "KC", "MBON")
        source_id_to_idx : Dict[int, int]
            Mapping from source neuron ID → column index
        target_id_to_idx : Dict[int, int]
            Mapping from target neuron ID → row index
        n_source : int
            Number of source neurons (matrix columns)
        n_target : int
            Number of target neurons (matrix rows)
        normalize : str
            Normalization strategy: "row" | "global" | "none"

        Returns
        -------
        sp.csr_matrix
            Sparse matrix of shape (n_target, n_source) with normalized weights.
        """
        # Annotate edges with neuron types
        nodes = self._load_nodes()
        node_type_map = dict(zip(nodes["node_id"], nodes["type"]))

        edges = edges.copy()
        edges["source_type"] = edges["source_id"].map(node_type_map)
        edges["target_type"] = edges["target_id"].map(node_type_map)

        # Filter to requested edge type
        filtered_edges = edges[
            (edges["source_type"] == source_type) & (edges["target_type"] == target_type)
        ]

        if len(filtered_edges) == 0:
            # No edges of this type → return empty matrix
            return sp.csr_matrix((n_target, n_source), dtype=np.float64)

        # Map neuron IDs to matrix indices
        row_indices = []
        col_indices = []
        weights = []

        for _, edge in filtered_edges.iterrows():
            source_id = edge["source_id"]
            target_id = edge["target_id"]
            weight = edge["synapse_weight"]

            # Skip edges where source/target not in our filtered neuron sets
            if source_id not in source_id_to_idx or target_id not in target_id_to_idx:
                continue

            col_idx = source_id_to_idx[source_id]  # source → column
            row_idx = target_id_to_idx[target_id]  # target → row

            row_indices.append(row_idx)
            col_indices.append(col_idx)
            weights.append(weight)

        if len(weights) == 0:
            # No valid edges → return empty matrix
            return sp.csr_matrix((n_target, n_source), dtype=np.float64)

        # Construct sparse matrix in COO format, then convert to CSR
        weights_array = np.array(weights, dtype=np.float64)
        matrix_coo = sp.coo_matrix(
            (weights_array, (row_indices, col_indices)),
            shape=(n_target, n_source),
            dtype=np.float64,
        )
        matrix_csr = matrix_coo.tocsr()

        # Apply normalization
        if normalize == "row":
            # Each row (post-synaptic neuron) sums to 1.0
            # This models synaptic normalization/homeostatic scaling
            row_sums = np.array(matrix_csr.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            # Multiply each row by 1/row_sum using sparse operations
            row_inv = sp.diags(1.0 / row_sums, format="csr")
            matrix_csr = row_inv @ matrix_csr

        elif normalize == "global":
            # Divide all weights by maximum weight
            # Preserves relative connection strength across circuit
            max_weight = matrix_csr.max()
            if max_weight > 0:
                matrix_csr = matrix_csr / max_weight

        elif normalize == "none":
            # Keep raw synapse counts
            pass

        return matrix_csr

    def validate_shapes(self, matrix: ConnectivityMatrix) -> None:
        """Check dimensions are reasonable (not exact hardcoded values).

        Biological Rationale
        --------------------
        The Drosophila MB exhibits stereotyped neuron counts across individuals:
        - PNs: ~150-500 (varies by hemisphere and filtering criteria)
        - KCs: ~2000-2500 per hemisphere (adult FlyWire: ~5000-5500 both hemispheres)
        - MBONs: ~34 types (bilateral: ~40-100 total neurons)
        - DANs: ~130 types (bilateral: ~200-600 total neurons)

        This validation ensures loaded data falls within biologically plausible
        ranges, catching errors in extraction scripts or corrupted cache files.

        Parameters
        ----------
        matrix : ConnectivityMatrix
            Loaded connectivity matrix to validate.

        Raises
        ------
        ValueError
            If neuron counts fall outside expected ranges.
        """
        # Validate PN count
        if not (50 <= matrix.n_pn <= 30000):
            raise ValueError(
                f"PN count {matrix.n_pn} outside expected range [50, 30000]. "
                f"Check ALPN extraction and filtering criteria."
            )

        # Validate KC count
        if not (1000 <= matrix.n_kc <= 10000):
            raise ValueError(
                f"KC count {matrix.n_kc} outside expected range [1000, 10000]. "
                f"Check KC extraction script and subtype files."
            )

        # Validate MBON count
        if not (20 <= matrix.n_mbon <= 150):
            raise ValueError(
                f"MBON count {matrix.n_mbon} outside expected range [20, 150]. "
                f"Check MBON extraction script."
            )

        # Validate DAN count
        if matrix.n_dan > 0:  # DAN is optional
            if not (50 <= matrix.n_dan <= 1000):
                raise ValueError(
                    f"DAN count {matrix.n_dan} outside expected range [50, 1000]. "
                    f"Check DAN extraction script."
                )

    def validate_connectivity(self, matrix: ConnectivityMatrix) -> Dict[str, Any]:
        """Return validation report with sparsity metrics and fan-in/out distributions.

        Biological Rationale
        --------------------
        The MB circuit exhibits characteristic connectivity statistics:
        - PN→KC fan-in: Each KC receives from ~6-8 PNs (the "claw")
        - KC→MBON fan-in: Each MBON receives from ~hundreds-thousands of KCs
        - Sparsity: PN→KC ~95-97%, KC→MBON ~90-95%

        This report enables validation against published connectome statistics.

        Parameters
        ----------
        matrix : ConnectivityMatrix
            Connectivity matrix to analyze.

        Returns
        -------
        Dict[str, Any]
            Diagnostic report with keys:
            - pn_to_kc_sparsity: fraction of absent connections
            - kc_to_mbon_sparsity: fraction of absent connections
            - pn_to_kc_fan_in: {p10, p50, p90} quantiles of PN inputs per KC
            - kc_to_mbon_fan_in: {p10, p50, p90} quantiles of KC inputs per MBON
            - orphan_kcs: number of KCs with no PN inputs or MBON outputs
            - orphan_mbons: number of MBONs with no KC inputs

        Example
        -------
        >>> report = loader.validate_connectivity(conn_matrix)
        >>> print(f"PN→KC sparsity: {report['pn_to_kc_sparsity']:.2%}")
        >>> print(f"Median PN fan-in per KC: {report['pn_to_kc_fan_in']['p50']:.1f}")
        """
        # Compute sparsity
        pn_to_kc_sparsity = matrix.pn_to_kc_sparsity()
        kc_to_mbon_sparsity = matrix.kc_to_mbon_sparsity()

        # Compute fan-in distributions (inputs per neuron)
        # PN→KC: count non-zero entries per row
        pn_to_kc_fan_in = np.diff(matrix.pn_to_kc.indptr)  # row counts for CSR
        pn_to_kc_quantiles = {
            "p10": float(np.percentile(pn_to_kc_fan_in, 10)),
            "p50": float(np.percentile(pn_to_kc_fan_in, 50)),
            "p90": float(np.percentile(pn_to_kc_fan_in, 90)),
        }

        # KC→MBON: count non-zero entries per row
        kc_to_mbon_fan_in = np.diff(matrix.kc_to_mbon.indptr)
        kc_to_mbon_quantiles = {
            "p10": float(np.percentile(kc_to_mbon_fan_in, 10)),
            "p50": float(np.percentile(kc_to_mbon_fan_in, 50)),
            "p90": float(np.percentile(kc_to_mbon_fan_in, 90)),
        }

        # Detect orphan neurons (no connections)
        orphan_kcs = int(np.sum((pn_to_kc_fan_in == 0) | (np.diff(matrix.kc_to_mbon.tocsc().indptr) == 0)))
        orphan_mbons = int(np.sum(kc_to_mbon_fan_in == 0))

        return {
            "pn_to_kc_sparsity": pn_to_kc_sparsity,
            "kc_to_mbon_sparsity": kc_to_mbon_sparsity,
            "pn_to_kc_fan_in": pn_to_kc_quantiles,
            "kc_to_mbon_fan_in": kc_to_mbon_quantiles,
            "orphan_kcs": orphan_kcs,
            "orphan_mbons": orphan_mbons,
        }
