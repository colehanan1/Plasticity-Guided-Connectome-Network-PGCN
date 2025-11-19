"""
DoOR (Database of Odorant Responses) integration for CCBPN.

This module converts chemical odorant names to biologically-realistic PN activity
patterns using:
1. DoOR database (odorant × ORN response matrix)
2. FlyWire connectome (ORN glomerulus → PN connectivity)
3. Biological constraints (sparse, graded responses)

The DoOR database provides experimentally-measured responses of Drosophila olfactory
receptor neurons (ORNs) to ~250 odorants. Each ORN type expresses a specific
odorant receptor (Or) gene and projects to a stereotyped glomerulus in the
antennal lobe, where it synapses onto PNs.

References
----------
Münch D, Galizia CG (2016) "DoOR 2.0 - Comprehensive Mapping of Drosophila
melanogaster Odorant Responses" Scientific Reports 6:21841

Example
-------
>>> from pgcn.data.door_integration import DoORIntegration
>>> door = DoORIntegration(cache_dir="data/cache")
>>>
>>> # Convert benzaldehyde to PN activity pattern
>>> pn_activity = door.odor_to_pn_activity("benzaldehyde", n_pn=150)
>>> print(f"Active PNs: {np.sum(pn_activity > 0.1)}")  # ~10-20 PNs
>>>
>>> # Create temporal odor sequence (40ms pulse)
>>> odor_seq = door.create_odor_sequence("hexanol", n_pn=150, odor_duration=40)
>>> print(odor_seq.shape)  # (100, 150) - 100ms simulation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["DoORIntegration", "GLOMERULUS_TO_ORN_MAPPING"]

logger = logging.getLogger(__name__)


# Known glomerulus → ORN receptor mappings from literature
# Source: Couto et al. (2005), Fishilevich & Vosshall (2005), Hallem & Carlson (2006)
GLOMERULUS_TO_ORN_MAPPING = {
    'DA1': 'Or67d',
    'DA2': 'Or56a',
    'DA3': 'Or23a',
    'DA4': 'Or43a',
    'DC1': 'Or19a',
    'DC2': 'Or13a',
    'DC3': 'Or83c',
    'DC4': 'Or85e',
    'DL1': 'Or10a',
    'DL3': 'Or59b',
    'DL4': 'Or49a',
    'DL5': 'Or7a',
    'DM1': 'Or42b',
    'DM2': 'Or22a',
    'DM3': 'Or47a',
    'DM4': 'Or59c',
    'DM5': 'Or85a',
    'DM6': 'Or67a',
    'DP1l': 'Or69a',
    'DP1m': 'Ir75a',
    'V': 'Ir84a',
    'VA1d': 'Or88a',
    'VA1v': 'Or47b',
    'VA2': 'Or92a',
    'VA3': 'Or65a',
    'VA4': 'Or49b',
    'VA5': 'Or85b',
    'VA6': 'Or82a',
    'VA7': 'Or46a',
    'VC1': 'Or33b',
    'VC2': 'Or71a',
    'VC3': 'Or35a',
    'VC4': 'Or67c',
    'VL1': 'Or42a',
    'VL2': 'Ir75b',
    'VM1': 'Ir92a',
    'VM2': 'Or43b',
    'VM3': 'Or98a',
    'VM4': 'Or43a',
    'VM5': 'Or85d',
}


class DoORIntegration:
    """Converts odorant names to biologically-realistic PN activity patterns.

    This class integrates the DoOR database with FlyWire connectome to map:
    Odorant → ORN responses (DoOR) → Glomerulus → PN activity (FlyWire)

    CRITICAL: Explicit mapping from experimental odor names to DoOR database names.
    The DoOR database uses specific naming conventions that don't always match
    experimental labels (e.g., 'hexanol' → '1-hexanol', 'ethyl_butyrate' → 'ethyl butyrate').
    """

    # Map experimental odor names to exact DoOR database names
    # Verified against DoOR database (690 odorants)
    ODOR_NAME_MAP = {
        'hexanol':              '1-hexanol',        # DoOR uses full IUPAC name
        'ethyl_butyrate':       'ethyl butyrate',   # DoOR uses space, not underscore
        'benzaldehyde':         'benzaldehyde',     # Exact match
        '3-octanol':            '3-octanol',        # Exact match
        'citral':               'citral',           # Exact match
        'linalool':             'linalool',         # Exact match
        'apple_cider_vinegar':  'acetic acid',      # Approximate as main component
    }

    """
    Parameters
    ----------
    cache_dir : Path or str
        Path to FlyWire cache directory (contains nodes.parquet)
    door_data_path : Path or str, optional
        Path to DoOR response matrix CSV. If None, attempts to download.

    Attributes
    ----------
    door_data : pd.DataFrame
        DoOR response matrix (odorants × ORN types)
    pn_glomeruli : Dict[int, str]
        Mapping from PN index → glomerulus name
    glomerulus_to_pn : Dict[str, List[int]]
        Reverse mapping: glomerulus → list of PN indices

    Raises
    ------
    FileNotFoundError
        If cache_dir doesn't exist or nodes.parquet missing
    ValueError
        If DoOR data cannot be loaded

    Notes
    -----
    **Biological constraints enforced**:
    - Sparse activation: Only 10-30 glomeruli respond per odor
    - Graded responses: Response magnitudes from DoOR (0-1 normalized)
    - Stereotyped connectivity: Each glomerulus has fixed ORN→PN mapping
    """

    def __init__(
        self,
        cache_dir: Path | str,
        door_data_path: Optional[Path | str] = None,
    ) -> None:
        """Initialize DoOR integration with FlyWire connectivity."""
        self.cache_dir = Path(cache_dir)

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_dir}. "
                f"Run FlyWire extraction scripts first."
            )

        # Load DoOR database
        self.door_data = self._load_door_database(door_data_path)

        # Load PN glomerulus assignments from FlyWire
        self.pn_glomeruli = self._load_pn_glomeruli()

        # Create reverse mapping (glomerulus → list of PNs)
        self.glomerulus_to_pn = self._create_glomerulus_to_pn_mapping()

        # Validate odor coverage
        self._validate_odor_coverage()

        logger.info(f"DoOR integration initialized: {len(self.door_data)} odorants, "
                   f"{len(self.pn_glomeruli)} PNs mapped to glomeruli")

    def _load_door_database(self, door_path: Optional[Path | str] = None) -> pd.DataFrame:
        """Load DoOR response matrix (odorant × ORN type).

        Returns
        -------
        pd.DataFrame
            DoOR response matrix with:
            - Index: odorant names (standardized, lowercase)
            - Columns: ORN types (Or42b, Or59b, Or7a, etc.)
            - Values: response magnitudes (0-1 normalized)
        """
        # Try user-provided path first
        if door_path is not None:
            door_path = Path(door_path)
            if door_path.exists():
                logger.info(f"Loading DoOR data from {door_path}")
                door_data = pd.read_csv(door_path, index_col=0)
                return self._normalize_door_data(door_data)

        # Try cache directory
        cached_path = self.cache_dir / "door_response_matrix.csv"
        if cached_path.exists():
            logger.info(f"Loading cached DoOR data from {cached_path}")
            door_data = pd.read_csv(cached_path, index_col=0)
            return self._normalize_door_data(door_data)

        # Try door-toolkit formatted data (more reliable than GitHub CSV)
        door_toolkit_paths = [
            self.cache_dir / "response_matrix_norm.csv",
            self.cache_dir / "response_matrix_norm.parquet",
            self.cache_dir.parent / "door_cache" / "response_matrix_norm.csv",
            self.cache_dir.parent / "door_cache" / "response_matrix_norm.parquet",
        ]

        for toolkit_path in door_toolkit_paths:
            if toolkit_path.exists():
                logger.info(f"Loading door-toolkit data from {toolkit_path}")
                if toolkit_path.suffix == '.parquet':
                    door_data = pd.read_parquet(toolkit_path)
                else:
                    door_data = pd.read_csv(toolkit_path, index_col=0)

                # Cache in standard location for future use
                door_data.to_csv(cached_path)
                logger.info(f"Cached door-toolkit data to {cached_path}")

                return self._normalize_door_data(door_data)

        # Try to download from DoOR repository
        logger.info("Downloading DoOR database (this may take a minute)...")
        try:
            door_url = "https://raw.githubusercontent.com/ropensci/DoOR.data/master/data/door_response_matrix.csv"
            door_data = pd.read_csv(door_url, index_col=0)

            # Cache for future use
            door_data.to_csv(cached_path)
            logger.info(f"DoOR data cached to {cached_path}")

            return self._normalize_door_data(door_data)
        except Exception as e:
            raise ValueError(
                f"Could not load DoOR database. Please download manually:\n"
                f"  wget https://github.com/ropensci/DoOR.data/raw/master/data/door_response_matrix.csv\n"
                f"  mv door_response_matrix.csv {self.cache_dir}/\n"
                f"Error: {e}"
            )

    def _normalize_door_data(self, door_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize DoOR data to standard format.

        - Standardize odorant names (lowercase, strip whitespace)
        - Normalize response values to [0, 1]
        - Handle missing values (NaN → 0)

        CRITICAL: Keep spaces in odor names! DoOR uses 'ethyl butyrate' not 'ethyl_butyrate'.
        """
        # Validate DoOR data structure
        if len(door_data.columns) == 0:
            raise ValueError(
                "DoOR data is malformed (0 columns - no ORN types found).\n"
                "This usually means the downloaded CSV has InChIKeys with concatenated data.\n\n"
                "Fix: Use door-toolkit formatted data instead:\n"
                "  1. Install door-toolkit: pip install door-toolkit\n"
                "  2. Extract data: door extract --output data/door_cache/\n"
                "  3. Or copy existing: cp data/door_cache/response_matrix_norm.csv data/cache/\n\n"
                "The code will automatically detect door-toolkit formatted files."
            )

        if len(door_data) == 0:
            raise ValueError("DoOR data is empty (0 odorants found)")

        # Standardize odorant names in index (lowercase and strip, but keep spaces!)
        door_data.index = door_data.index.str.lower().str.strip()
        # NOTE: Do NOT replace spaces - DoOR uses spaces in names like 'ethyl butyrate'

        # Fill NaN with 0 (no response)
        door_data = door_data.fillna(0)

        # Normalize response values to [0, 1] per ORN
        for col in door_data.columns:
            col_max = door_data[col].max()
            if col_max > 0:
                door_data[col] = door_data[col] / col_max

        return door_data

    def _load_pn_glomeruli(self) -> Dict[int, str]:
        """Load PN → glomerulus assignments from FlyWire cache.

        Returns
        -------
        Dict[int, str]
            Mapping from PN index → glomerulus name (e.g., {0: 'DA1', 1: 'DL3'})
        """
        nodes_path = self.cache_dir / "nodes.parquet"
        if not nodes_path.exists():
            raise FileNotFoundError(
                f"nodes.parquet not found at {nodes_path}. "
                f"Run FlyWire extraction scripts."
            )

        nodes = pd.read_parquet(nodes_path)
        pn_nodes = nodes[nodes['type'] == 'PN'].copy()

        glomerulus_map = {}

        # Try to extract glomerulus from multiple possible columns
        for idx, row in pn_nodes.iterrows():
            pn_idx = int(row.name) if isinstance(row.name, (int, np.integer)) else idx

            # Try 'glomerulus' column first
            if 'glomerulus' in row and pd.notna(row['glomerulus']) and row['glomerulus'] != '':
                glomerulus_map[pn_idx] = str(row['glomerulus'])
                continue

            # Try 'cell_type' column (sometimes contains glomerulus)
            if 'cell_type' in row and pd.notna(row['cell_type']):
                cell_type = str(row['cell_type'])
                # Extract glomerulus from patterns like "DA1_PN", "PN_DL3", etc.
                for glom in GLOMERULUS_TO_ORN_MAPPING.keys():
                    if glom in cell_type:
                        glomerulus_map[pn_idx] = glom
                        break

        if len(glomerulus_map) == 0:
            logger.warning("No PN glomerulus assignments found! DoOR integration will fail.")
            logger.warning("Check that FlyWire cache includes glomerulus metadata.")
        else:
            logger.info(f"Mapped {len(glomerulus_map)} PNs to glomeruli")

        return glomerulus_map

    def _create_glomerulus_to_pn_mapping(self) -> Dict[str, List[int]]:
        """Create reverse mapping from glomerulus → list of PN indices."""
        glom_to_pn = {}
        for pn_idx, glomerulus in self.pn_glomeruli.items():
            if glomerulus not in glom_to_pn:
                glom_to_pn[glomerulus] = []
            glom_to_pn[glomerulus].append(pn_idx)

        logger.debug(f"Created reverse mapping for {len(glom_to_pn)} glomeruli")
        return glom_to_pn

    def _validate_odor_coverage(self):
        """Check that all experimental odors can be mapped to DoOR and produce non-zero activity.

        This is CRITICAL for training - if odors return zero patterns, the model cannot learn!
        """
        # Standard experimental odors from user specification
        required_odors = list(self.ODOR_NAME_MAP.keys())

        print("\n" + "="*60)
        print("DoOR Integration Validation")
        print("="*60)

        all_success = True
        for odor in required_odors:
            # Test name resolution
            door_name = self._resolve_odor_name(odor)

            if door_name:
                # Test that we get non-zero PN activity
                test_pattern = self.odor_to_pn_activity(odor, n_pn=100)
                n_active = np.sum(test_pattern > 0.1)

                if n_active > 0:
                    marker = '✓' if odor != 'apple_cider_vinegar' else '⚠️ '
                    print(f"  {marker} {odor:25s} → {n_active:3d} active PNs (DoOR: '{door_name}')")
                else:
                    print(f"  ⚠️  {odor:25s} → {n_active:3d} active PNs (DoOR: '{door_name}') - CHECK GLOMERULUS MAPPING!")
                    all_success = False
            else:
                print(f"  ✗ {odor:25s} → NOT FOUND IN DoOR")
                all_success = False

        print("="*60)
        if all_success:
            print("✅ All experimental odors successfully mapped to DoOR!")
            print("   Model can learn odor-specific patterns.")
        else:
            print("❌ Some odors failed validation - check mappings above")
            print("   Model will have difficulty learning!")
        print("="*60 + "\n")

        return all_success

    def odor_to_pn_activity(
        self,
        odor_name: str,
        n_pn: int,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Convert odor name to PN activity pattern.

        Maps: Odorant → ORN responses (DoOR) → Glomerulus → PN activity

        Parameters
        ----------
        odor_name : str
            Standardized odor name (e.g., 'hexanol', 'benzaldehyde').
            Automatically tries common variants (spaces, hyphens, underscores).
        n_pn : int
            Total number of PNs in the circuit
        intensity : float, optional
            Odor concentration scaling factor (0-1, default=1.0)

        Returns
        -------
        np.ndarray
            PN activity vector of shape (n_pn,) with values in [0, 1]

        Biological Constraints
        ----------------------
        - **Sparse activation**: Only PNs in responsive glomeruli are active (~10-30)
        - **Graded responses**: Activity magnitudes from DoOR (not binary)
        - **Stereotyped mapping**: Same odor always activates same glomeruli

        Example
        -------
        >>> door = DoORIntegration("data/cache")
        >>> pn_activity = door.odor_to_pn_activity("benzaldehyde", n_pn=150)
        >>> print(f"Active PNs: {np.sum(pn_activity > 0.1)}")  # ~15-25
        >>> print(f"Max response: {pn_activity.max():.2f}")    # ~0.8-1.0
        """
        pn_activity = np.zeros(n_pn, dtype=np.float32)

        # Resolve experimental odor name to DoOR database name
        door_name = self._resolve_odor_name(odor_name)

        if door_name is None:
            logger.error(f"❌ Odor '{odor_name}' not in DoOR, returning ZERO pattern")
            logger.error(f"   → Model cannot learn from this odor!")
            return pn_activity  # All zeros - model cannot learn!

        # Get ORN responses from DoOR for this odor
        try:
            orn_responses = self.door_data.loc[door_name]  # Series of ORN type → response
        except KeyError:
            logger.error(f"❌ KeyError for resolved odor '{door_name}' (this should never happen!)")
            return pn_activity

        # Map ORN responses → PN activity via glomeruli
        for pn_idx, glomerulus in self.pn_glomeruli.items():
            if pn_idx >= n_pn:
                continue

            # Find ORN type for this glomerulus
            orn_type = GLOMERULUS_TO_ORN_MAPPING.get(glomerulus, None)

            if orn_type is None:
                # Unknown glomerulus → no response
                continue

            # Check if this ORN type is in DoOR columns
            if orn_type in orn_responses.index:
                response = orn_responses[orn_type]
                pn_activity[pn_idx] = float(response) * intensity

        # Normalize to [0, 1] range
        if pn_activity.max() > 0:
            pn_activity = pn_activity / pn_activity.max()

        n_active = np.sum(pn_activity > 0.1)
        logger.debug(f"Odor '{odor_name}': {n_active} PNs active (intensity={intensity:.2f})")

        return pn_activity

    def _resolve_odor_name(self, odor_name: str) -> Optional[str]:
        """Map experimental odor name to DoOR database name.

        This method implements a robust name resolution strategy:
        1. Check explicit ODOR_NAME_MAP first (handles known mismatches)
        2. Try exact match (case-sensitive)
        3. Try common variants (spaces, hyphens, case)
        4. Try case-insensitive search
        5. Return None if not found

        Parameters
        ----------
        odor_name : str
            User's odor name (e.g., 'ethyl_butyrate', 'hexanol')

        Returns
        -------
        str or None
            DoOR database name (e.g., 'ethyl butyrate', '1-hexanol') or None if not found
        """
        # Step 1: Check explicit mapping FIRST (handles 'hexanol' → '1-hexanol', etc.)
        if odor_name in self.ODOR_NAME_MAP:
            door_name = self.ODOR_NAME_MAP[odor_name]
            if door_name in self.door_data.index:
                logger.debug(f"Mapped '{odor_name}' → '{door_name}' via explicit map")
                return door_name
            else:
                logger.error(f"Mapped '{odor_name}' → '{door_name}', but '{door_name}' NOT IN DoOR!")
                logger.error(f"Available DoOR odors sample: {list(self.door_data.index[:10])}...")
                return None

        # Step 2: Try exact match (case-sensitive)
        if odor_name in self.door_data.index:
            logger.debug(f"Exact match: '{odor_name}'")
            return odor_name

        # Step 3: Try common variants (spaces, hyphens, case)
        variants = [
            odor_name.replace('_', ' '),           # ethyl_butyrate → ethyl butyrate
            odor_name.replace('_', '-'),           # 3_octanol → 3-octanol
            odor_name.replace('-', ' '),           # 3-octanol → 3 octanol
            f"1-{odor_name}",                      # hexanol → 1-hexanol
            odor_name.lower(),                     # HEXANOL → hexanol
            odor_name.replace('_', ' ').lower(),   # Ethyl_Butyrate → ethyl butyrate
        ]

        for variant in variants:
            if variant in self.door_data.index:
                logger.info(f"Matched '{odor_name}' → '{variant}' via variant search")
                return variant

        # Step 4: Case-insensitive exact match
        for door_odor in self.door_data.index:
            if door_odor.lower() == odor_name.lower():
                logger.info(f"Matched '{odor_name}' → '{door_odor}' via case-insensitive search")
                return door_odor

        # Step 5: Failed to find
        logger.warning(f"⚠️  Odor '{odor_name}' NOT FOUND in DoOR database!")
        # Show similar odors for debugging
        similar = [o for o in self.door_data.index if len(odor_name) >= 4 and odor_name[:4].lower() in o.lower()]
        if similar:
            logger.warning(f"    Available similar odors: {similar[:5]}")
        else:
            logger.warning(f"    Total DoOR odors: {len(self.door_data.index)}")
        return None

    def create_odor_sequence(
        self,
        odor_name: str,
        n_pn: int,
        sequence_length: int = 100,
        odor_onset: int = 0,
        odor_duration: int = 40,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Create temporal odor presentation sequence.

        Generates a time-series of PN activity simulating odor delivery:
        - Baseline (no odor): t=0 to odor_onset
        - Odor ON: t=odor_onset to odor_onset+odor_duration
        - Washout (no odor): t=odor_onset+odor_duration to sequence_length

        Parameters
        ----------
        odor_name : str
            Odor to present (e.g., 'hexanol')
        n_pn : int
            Number of PNs in circuit
        sequence_length : int, optional
            Total sequence length in timesteps (default=100, representing 100ms)
        odor_onset : int, optional
            When odor turns on, in timesteps (default=0)
        odor_duration : int, optional
            How long odor is on, in timesteps (default=40 = 40ms)
        intensity : float, optional
            Odor concentration (default=1.0 = saturating)

        Returns
        -------
        np.ndarray
            Temporal sequence of shape (sequence_length, n_pn)

        Example
        -------
        >>> door = DoORIntegration("data/cache")
        >>> # 40ms pulse of benzaldehyde starting at t=0
        >>> seq = door.create_odor_sequence("benzaldehyde", n_pn=150, odor_duration=40)
        >>> print(seq.shape)  # (100, 150)
        >>> print(np.sum(seq[0] > 0.1))   # ~15 active PNs during pulse
        >>> print(np.sum(seq[50] > 0.1))  # 0 active PNs after washout
        """
        sequence = np.zeros((sequence_length, n_pn), dtype=np.float32)

        # Get PN activity pattern for this odor
        pn_pattern = self.odor_to_pn_activity(odor_name, n_pn, intensity=intensity)

        # Apply temporal profile (odor ON during specified window)
        odor_offset = min(odor_onset + odor_duration, sequence_length)
        sequence[odor_onset:odor_offset, :] = pn_pattern

        return sequence

    def get_odor_similarity(
        self,
        odor1: str,
        odor2: str,
        n_pn: int = 100,
    ) -> float:
        """Compute similarity between two odors based on PN activation overlap.

        Useful for predicting cross-generalization: flies trained on odor1 should
        generalize to odor2 proportional to their PN pattern similarity.

        Parameters
        ----------
        odor1 : str
            First odor name
        odor2 : str
            Second odor name
        n_pn : int, optional
            Number of PNs for pattern generation (default=100)

        Returns
        -------
        float
            Pearson correlation between PN patterns (range: -1 to 1)
            High correlation (>0.7) = similar odors, likely generalization
            Low correlation (<0.3) = dissimilar odors, unlikely generalization

        Example
        -------
        >>> door = DoORIntegration("data/cache")
        >>> similarity = door.get_odor_similarity("hexanol", "3-octanol", n_pn=150)
        >>> print(f"Hexanol-3-octanol similarity: {similarity:.2f}")
        >>> # Both are alcohols → likely high similarity (>0.6)
        """
        pattern1 = self.odor_to_pn_activity(odor1, n_pn)
        pattern2 = self.odor_to_pn_activity(odor2, n_pn)

        # Handle zero patterns (no DoOR data)
        if np.sum(pattern1) == 0 or np.sum(pattern2) == 0:
            logger.warning(f"Zero pattern for {odor1} or {odor2}, similarity=0")
            return 0.0

        # Pearson correlation
        correlation = np.corrcoef(pattern1, pattern2)[0, 1]

        return float(correlation)
