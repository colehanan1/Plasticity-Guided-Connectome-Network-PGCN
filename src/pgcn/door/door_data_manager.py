"""DoOR Database Integration - Complete Data Access and Management.

This module provides comprehensive access to the Database of Odorant Responses (DoOR) v2.0,
containing olfactory receptor response profiles for 693 odorants across 78 responding units.

The DoOR database is essential for building biologically accurate ORN input layers for the
PGCN model, enabling mechanistic testing of ORN/PN pathway blocking hypotheses.

Data Sources
------------
- DoOR GitHub: https://github.com/ropensci/DoOR.data
- DoOR Functions: https://github.com/ropensci/DoOR.functions
- Zenodo Archive: https://zenodo.org/record/3355719 (v2.0.1)
- Publication: Münch & Galizia (2016), Scientific Reports

Example
-------
>>> from pgcn.door import DoORDataManager
>>>
>>> # Initialize with rpy2 (preferred method)
>>> manager = DoORDataManager(method='rpy2')
>>>
>>> # Install DoOR R packages (first time only)
>>> manager.install_door_packages()
>>>
>>> # Load complete DoOR database
>>> door_data = manager.load_door_data()
>>> print(f"Response matrix shape: {door_data['response_matrix'].shape}")
>>> print(f"Number of odorants: {len(door_data['odor_info'])}")
>>>
>>> # Get complete receptor list
>>> receptors = manager.get_complete_receptor_list()
>>> print(f"Total receptors: {len(receptors['all_receptors'])}")
>>> print(f"OR receptors: {len(receptors['or_receptors'])}")
"""

from __future__ import annotations

import gzip
import logging
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# Optional imports with graceful fallbacks
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr, data
    from rpy2.robjects import pandas2ri
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False
    logging.warning("rpy2 not available. Install with: pip install rpy2")


class DoORDataManager:
    """Complete DoOR database integration with multiple access methods.

    This class provides robust access to the DoOR database through multiple methods:
    1. rpy2 (primary): Direct access to R packages via Python
    2. CSV (backup): Load from preprocessed CSV files
    3. Zenodo (download): Fetch from Zenodo archive

    The DoOR database contains normalized odorant response profiles for all
    characterized Drosophila olfactory receptors, enabling biologically accurate
    ORN modeling for the PGCN framework.

    Biological Context
    ------------------
    DoOR v2.0 consolidates 7,381 data points from 693 odorants across:
    - 60 Odorant Receptors (ORx)
    - 16 Ionotropic Receptors (IRx)
    - 2 Gustatory Receptors (GRx) with olfactory function
    - Multiple sensillum types (ab, ac, pb)

    Response values are normalized to [0, 1] using the global normalization
    approach described in Münch & Galizia (2016).

    Parameters
    ----------
    method : str, optional
        Data access method. Options: 'rpy2' (default), 'csv', 'zenodo'
    backup_csv_path : str or Path, optional
        Path to backup CSV file if rpy2 fails
    cache_dir : str or Path, optional
        Directory for caching downloaded data. Default: 'data/door_cache'

    Attributes
    ----------
    method : str
        Active data access method
    cache_dir : Path
        Cache directory for downloaded data
    logger : logging.Logger
        Logger for status messages and errors
    _door_data_cache : Optional[Dict]
        Cached DoOR data to avoid repeated R calls

    Raises
    ------
    ImportError
        If rpy2 is not available and method='rpy2'
    FileNotFoundError
        If backup_csv_path specified but file doesn't exist

    Notes
    -----
    **Performance considerations**:
    - Initial rpy2 load: ~2-5 seconds
    - Cached access: <0.1 seconds
    - CSV fallback: ~1 second

    **Biological accuracy**:
    DoOR responses are in vitro measurements (heterologous expression systems)
    and may differ from in vivo responses. Use with appropriate biological
    context and validation against behavioral data.

    Examples
    --------
    >>> # Standard usage with rpy2
    >>> manager = DoORDataManager()
    >>> door_data = manager.load_door_data()
    >>>
    >>> # Get Or42b response profile
    >>> or42b_responses = door_data['response_matrix']['Or42b']
    >>> top_ligands = or42b_responses.nlargest(5)
    >>> print("Top Or42b ligands:")
    >>> for ligand, response in top_ligands.items():
    ...     print(f"  {ligand}: {response:.3f}")
    """

    def __init__(
        self,
        method: str = 'rpy2',
        backup_csv_path: Optional[str] = None,
        cache_dir: str = 'data/door_cache'
    ):
        """Initialize DoOR data manager with specified access method."""
        # Validate method
        valid_methods = {'rpy2', 'csv', 'zenodo'}
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{method}'"
            )

        # Check rpy2 availability
        if method == 'rpy2' and not RPY2_AVAILABLE:
            raise ImportError(
                "rpy2 is required for method='rpy2'. "
                "Install with: pip install rpy2"
            )

        self.method = method
        self.backup_csv = Path(backup_csv_path) if backup_csv_path else None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = self._setup_logging()

        # Cache for loaded data
        self._door_data_cache: Optional[Dict[str, pd.DataFrame]] = None

        # Verify backup file exists if specified
        if self.backup_csv and not self.backup_csv.exists():
            raise FileNotFoundError(
                f"Backup CSV file not found: {self.backup_csv}"
            )

    def _setup_logging(self) -> logging.Logger:
        """Configure logger for DoOR data manager."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def install_door_packages(self) -> bool:
        """Install DoOR R packages via rpy2.

        This method installs the official DoOR R packages from GitHub:
        - DoOR.data (v2.0.1): Core database
        - DoOR.functions (v2.0.1): Analysis utilities

        Returns
        -------
        bool
            True if installation succeeded, False otherwise

        Raises
        ------
        ImportError
            If rpy2 is not available

        Notes
        -----
        This only needs to be run once. Subsequent calls will check for
        existing installations and skip if already present.

        Examples
        --------
        >>> manager = DoORDataManager()
        >>> success = manager.install_door_packages()
        >>> if success:
        ...     print("DoOR packages ready")
        """
        if not RPY2_AVAILABLE:
            raise ImportError("rpy2 required for package installation")

        try:
            self.logger.info("Installing DoOR R packages...")
            pandas2ri.activate()
            utils = importr('utils')

            # Install devtools if not present
            try:
                importr('devtools')
                self.logger.info("devtools already installed")
            except Exception:
                self.logger.info("Installing devtools...")
                utils.install_packages('devtools')

            devtools = importr('devtools')

            # Install DoOR packages from GitHub
            self.logger.info("Installing DoOR.data v2.0.1...")
            devtools.install_github("ropensci/DoOR.data", ref="v2.0.1")

            self.logger.info("Installing DoOR.functions v2.0.1...")
            devtools.install_github("ropensci/DoOR.functions", ref="v2.0.1")

            self.logger.info("DoOR packages installed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to install DoOR packages: {e}")
            return False

    def load_door_data(self) -> Dict[str, pd.DataFrame]:
        """Load complete DoOR database with all response matrices.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing:
            - 'response_matrix': Normalized responses (odorants × receptors)
            - 'response_matrix_non_normalized': Raw responses
            - 'odor_info': Odorant metadata (CAS numbers, names, etc.)
            - 'receptor_info': Receptor metadata (expression patterns, etc.)

        Raises
        ------
        Exception
            If all access methods fail

        Examples
        --------
        >>> manager = DoORDataManager()
        >>> door_data = manager.load_door_data()
        >>>
        >>> # Access normalized response matrix
        >>> responses = door_data['response_matrix']
        >>> print(f"Shape: {responses.shape}")
        >>> print(f"Odorants: {responses.index.tolist()[:5]}")
        >>> print(f"Receptors: {responses.columns.tolist()[:5]}")
        """
        # Return cached data if available
        if self._door_data_cache is not None:
            self.logger.info("Returning cached DoOR data")
            return self._door_data_cache

        try:
            if self.method == 'rpy2':
                data = self._load_via_rpy2()
            elif self.method == 'csv' and self.backup_csv:
                data = self._load_via_csv()
            else:
                data = self._download_from_zenodo()

            # Cache the loaded data
            self._door_data_cache = data
            return data

        except Exception as e:
            self.logger.error(f"Failed to load DoOR data via {self.method}: {e}")

            # Try backup methods
            if self.backup_csv and self.method != 'csv':
                self.logger.info("Attempting CSV fallback...")
                return self._load_via_csv()

            raise RuntimeError(f"All DoOR data access methods failed: {e}")

    def _load_via_rpy2(self) -> Dict[str, pd.DataFrame]:
        """Load DoOR data via R packages (primary method).

        Returns
        -------
        Dict[str, pd.DataFrame]
            Complete DoOR database

        Notes
        -----
        This method activates pandas2ri conversion to automatically translate
        R dataframes to pandas DataFrames.
        """
        if not RPY2_AVAILABLE:
            raise ImportError("rpy2 not available")

        self.logger.info("Loading DoOR data via rpy2...")

        # Activate pandas conversion
        pandas2ri.activate()

        try:
            # Import DoOR packages
            door_data_pkg = importr('DoOR.data')
            door_functions_pkg = importr('DoOR.functions')
        except Exception as e:
            raise ImportError(
                f"DoOR packages not installed. Run install_door_packages() first. Error: {e}"
            )

        # Load data using DoOR functions
        ro.r('library(DoOR.data)')
        ro.r('library(DoOR.functions)')
        ro.r('loadData()')

        # Access key data structures
        self.logger.info("Converting R data structures to pandas...")

        response_matrix = pandas2ri.rpy2py(ro.r['response.matrix'])
        response_matrix_non_norm = pandas2ri.rpy2py(ro.r['response.matrix_non.normalized'])
        odor_info = pandas2ri.rpy2py(ro.r['odor'])
        receptor_info = pandas2ri.rpy2py(ro.r['receptor'])

        self.logger.info(f"Loaded DoOR data: {response_matrix.shape} response matrix")

        return {
            'response_matrix': pd.DataFrame(response_matrix),
            'response_matrix_non_normalized': pd.DataFrame(response_matrix_non_norm),
            'odor_info': pd.DataFrame(odor_info),
            'receptor_info': pd.DataFrame(receptor_info)
        }

    def _load_via_csv(self) -> Dict[str, pd.DataFrame]:
        """Load DoOR data from preprocessed CSV files (backup method).

        Returns
        -------
        Dict[str, pd.DataFrame]
            DoOR database from CSV
        """
        if not self.backup_csv:
            raise ValueError("No backup CSV path specified")

        self.logger.info(f"Loading DoOR data from CSV: {self.backup_csv}")

        # Load main response matrix
        if str(self.backup_csv).endswith('.gz'):
            response_matrix = pd.read_csv(self.backup_csv, compression='gzip', index_col=0)
        else:
            response_matrix = pd.read_csv(self.backup_csv, index_col=0)

        self.logger.info(f"Loaded CSV data: {response_matrix.shape}")

        # For CSV mode, return minimal structure
        return {
            'response_matrix': response_matrix,
            'response_matrix_non_normalized': response_matrix,  # Assume normalized
            'odor_info': pd.DataFrame(index=response_matrix.index),
            'receptor_info': pd.DataFrame(index=response_matrix.columns)
        }

    def _download_from_zenodo(self) -> Dict[str, pd.DataFrame]:
        """Download DoOR data from Zenodo archive (fallback method).

        Returns
        -------
        Dict[str, pd.DataFrame]
            DoOR database from Zenodo

        Notes
        -----
        This method downloads the complete DoOR v2.0.1 dataset from Zenodo
        and caches it locally for future use.
        """
        self.logger.info("Downloading DoOR data from Zenodo...")

        # Zenodo DOI for DoOR v2.0.1
        zenodo_url = "https://zenodo.org/record/3355719/files/DoOR_2.0.1.zip"

        import requests
        import zipfile
        from io import BytesIO

        # Download data
        response = requests.get(zenodo_url)
        response.raise_for_status()

        # Extract ZIP
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(self.cache_dir)

        # Load extracted CSV
        response_matrix_path = self.cache_dir / "response_matrix.csv"
        response_matrix = pd.read_csv(response_matrix_path, index_col=0)

        self.logger.info(f"Downloaded and loaded DoOR data: {response_matrix.shape}")

        return {
            'response_matrix': response_matrix,
            'response_matrix_non_normalized': response_matrix,
            'odor_info': pd.DataFrame(index=response_matrix.index),
            'receptor_info': pd.DataFrame(index=response_matrix.columns)
        }

    def get_complete_receptor_list(self) -> Dict[str, List[str]]:
        """Extract complete list of olfactory receptors from DoOR.

        Returns
        -------
        Dict[str, List[str]]
            Categorized receptor lists:
            - 'all_receptors': Complete list
            - 'or_receptors': Odorant Receptors (Orx)
            - 'ir_receptors': Ionotropic Receptors (IRx)
            - 'gr_receptors': Gustatory Receptors with olfactory function
            - 'sensillum_types': Sensillum-based recordings (ab, ac, pb)

        Examples
        --------
        >>> manager = DoORDataManager()
        >>> receptors = manager.get_complete_receptor_list()
        >>> print(f"Total: {len(receptors['all_receptors'])}")
        >>> print(f"ORs: {receptors['or_receptors'][:5]}")
        """
        door_data = self.load_door_data()
        receptor_names = door_data['response_matrix'].columns.tolist()

        # Categorize receptors by type
        or_receptors = [r for r in receptor_names if r.startswith('Or')]
        ir_receptors = [r for r in receptor_names if r.startswith('Ir')]
        gr_receptors = [r for r in receptor_names if r.startswith('Gr')]
        sensillum_types = [
            r for r in receptor_names
            if any(r.startswith(s) for s in ['ab', 'ac', 'pb'])
        ]

        self.logger.info(
            f"Extracted {len(receptor_names)} receptors: "
            f"{len(or_receptors)} ORs, {len(ir_receptors)} IRs, "
            f"{len(gr_receptors)} GRs, {len(sensillum_types)} sensilla"
        )

        return {
            'all_receptors': receptor_names,
            'or_receptors': or_receptors,
            'ir_receptors': ir_receptors,
            'gr_receptors': gr_receptors,
            'sensillum_types': sensillum_types
        }

    def get_receptor_response_profile(
        self,
        receptor_name: str,
        top_n: int = 10
    ) -> pd.Series:
        """Get response profile for a specific receptor.

        Parameters
        ----------
        receptor_name : str
            Receptor name (e.g., 'Or42b', 'Ir8a')
        top_n : int, optional
            Number of top responding odorants to return. Default: 10

        Returns
        -------
        pd.Series
            Top responding odorants with response values

        Examples
        --------
        >>> manager = DoORDataManager()
        >>> or42b_profile = manager.get_receptor_response_profile('Or42b', top_n=5)
        >>> print(or42b_profile)
        """
        door_data = self.load_door_data()
        response_matrix = door_data['response_matrix']

        if receptor_name not in response_matrix.columns:
            raise ValueError(
                f"Receptor '{receptor_name}' not found. "
                f"Available: {response_matrix.columns.tolist()[:10]}..."
            )

        receptor_responses = response_matrix[receptor_name]
        top_responses = receptor_responses.nlargest(top_n)

        return top_responses
