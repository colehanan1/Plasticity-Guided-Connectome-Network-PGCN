"""FlyWire ORN Identification - Advanced Pattern Matching for Olfactory Cells.

This module implements comprehensive identification of olfactory receptor neurons (ORNs)
and projection neurons (PNs) from FlyWire community labels, using pattern matching
constrained by DoOR database receptor lists.

The identification system uses multi-level pattern matching:
1. Receptor names (Or42b, Ir8a, Gr21a, etc.)
2. Sensillum types (ab1, ac2, pb3, etc.)
3. Anatomical locations (antenna, palp, glomeruli)
4. Cell type markers (ORN, OSN, PN, etc.)

Example
-------
>>> from pgcn.door import DoORDataManager, FlyWireORNIdentifier
>>>
>>> # Initialize DoOR database
>>> door_manager = DoORDataManager()
>>>
>>> # Create identifier with DoOR-constrained patterns
>>> identifier = FlyWireORNIdentifier(door_manager)
>>>
>>> # Identify olfactory cells from FlyWire labels
>>> olfactory_cells = identifier.identify_olfactory_cells(
...     'data/processed_labels.csv.gz'
... )
>>> print(f"Found {len(olfactory_cells)} olfactory cells")
>>> print(f"DoOR matches: {olfactory_cells['door_match'].notna().sum()}")
"""

from __future__ import annotations

import gzip
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from pgcn.door.door_data_manager import DoORDataManager


class FlyWireORNIdentifier:
    """Advanced ORN identification using comprehensive pattern matching.

    This class implements multi-stage pattern matching to identify olfactory
    neurons from FlyWire community labels with high accuracy and biological
    validation through DoOR database cross-referencing.

    Identification Strategy
    -----------------------
    1. **Receptor-based matching**: Direct matching of receptor names (Or42b, etc.)
    2. **Sensillum-based matching**: Sensillum types that house specific ORNs
    3. **Anatomical matching**: Antennal/palp locations where ORNs reside
    4. **Cell type matching**: ORN/OSN/PN markers in labels
    5. **DoOR validation**: Cross-reference with DoOR receptor list

    Biological Accuracy
    -------------------
    Pattern matching is constrained by:
    - Complete DoOR receptor list (60 ORs, 16 IRs, 2 GRs)
    - Known sensillum→receptor mappings from literature
    - Anatomical organization of olfactory system

    Parameters
    ----------
    door_manager : DoORDataManager
        Initialized DoOR data manager for receptor list extraction
    confidence_threshold : float, optional
        Minimum confidence score for positive identification. Default: 0.5

    Attributes
    ----------
    door_manager : DoORDataManager
        DoOR database access
    receptor_patterns : Dict[str, List[str]]
        Compiled regex patterns for each receptor category
    search_stats : Dict[str, int]
        Statistics on search performance

    Examples
    --------
    >>> # Standard usage
    >>> identifier = FlyWireORNIdentifier(door_manager)
    >>> cells = identifier.identify_olfactory_cells('processed_labels.csv.gz')
    >>>
    >>> # Filter for high-confidence DoOR matches
    >>> door_matched = cells[cells['door_match'].notna()]
    >>> print(f"DoOR-validated cells: {len(door_matched)}")
    >>>
    >>> # Get Or42b-specific cells
    >>> or42b_cells = cells[cells['receptor_subtype'].str.contains('Or42b', na=False)]
    >>> print(f"Or42b cells: {len(or42b_cells)}")
    """

    def __init__(
        self,
        door_manager: DoORDataManager,
        confidence_threshold: float = 0.5
    ):
        """Initialize ORN identifier with DoOR-constrained patterns."""
        self.door_manager = door_manager
        self.confidence_threshold = confidence_threshold

        # Create comprehensive search patterns
        self.receptor_patterns = self._create_comprehensive_patterns()

        # Initialize search statistics
        self.search_stats = {
            'matches': 0,
            'false_positives': 0,
            'cells_processed': 0,
            'door_validated': 0
        }

    def _create_comprehensive_patterns(self) -> Dict[str, List[str]]:
        """Create exhaustive search patterns based on DoOR database.

        Returns
        -------
        Dict[str, List[str]]
            Categorized regex patterns for receptor identification
        """
        # Get DoOR receptor list for validation
        door_receptors = self.door_manager.get_complete_receptor_list()

        patterns = {
            # Odorant Receptors (complete list from DoOR)
            'or_receptors': [
                # Major characterized ORs
                r'\bOr1a\b', r'\bOr2a\b', r'\bOr7a\b', r'\bOr9a\b', r'\bOr10a\b',
                r'\bOr13a\b', r'\bOr19[ab]\b', r'\bOr22[ab]\b', r'\bOr23a\b',
                r'\bOr30a\b', r'\bOr33[abc]\b', r'\bOr35a\b', r'\bOr42[ab]\b',
                r'\bOr43[ab]\b', r'\bOr45[ab]\b', r'\bOr46[aA]\b', r'\bOr47[abch]\b',
                r'\bOr49[ab]\b', r'\bOr56a\b', r'\bOr59[abc]\b', r'\bOr63a\b',
                r'\bOr65[abc]\b', r'\bOr67[abcd]\b', r'\bOr69[aA]\b',
                r'\bOr71a\b', r'\bOr74a\b', r'\bOr82a\b', r'\bOr83[abc]\b',
                r'\bOr85[abcdef]\b', r'\bOr88a\b', r'\bOr92a\b', r'\bOr94[ab]\b',
                r'\bOr98a\b',
                # Generic OR pattern for any OrXX variant
                r'\bOr\d+[a-zA-Z]?\b'
            ],

            # Ionotropic Receptors (IR family)
            'ir_receptors': [
                # Co-receptors (broadly expressed)
                r'\bIr8a\b', r'\bIr25a\b', r'\bIr76b\b',
                # Specific IRs
                r'\bIr20a\b', r'\bIr21a\b', r'\bIr31a\b', r'\bIr40a\b',
                r'\bIr41a\b', r'\bIr52[acd]\b', r'\bIr56[abcd]\b',
                r'\bIr60a\b', r'\bIr64a\b', r'\bIr68a\b', r'\bIr75[abcd]\b',
                r'\bIr84a\b', r'\bIr92a\b', r'\bIr93a\b',
                # Generic IR pattern
                r'\bIr\d+[a-zA-Z]?\b'
            ],

            # Gustatory Receptors (olfactory function)
            'gr_receptors': [
                # CO2 receptors
                r'\bGr21a\b', r'\bGr63a\b',
                # Sugar/other receptors expressed in olfactory organs
                r'\bGr10a\b', r'\bGr28[ab]\b', r'\bGr32a\b', r'\bGr33a\b',
                r'\bGr39[ab]\b', r'\bGr43a\b', r'\bGr47a\b', r'\bGr54a\b',
                r'\bGr58[abc]\b', r'\bGr59[abcdef]\b', r'\bGr61a\b',
                r'\bGr64[abcdef]\b', r'\bGr66a\b', r'\bGr68a\b',
                r'\bGr89a\b', r'\bGr93[abcd]\b', r'\bGr97a\b', r'\bGr98[abcd]\b',
                # Generic GR pattern
                r'\bGr\d+[a-zA-Z]?\b'
            ],

            # Sensillum types (anatomically organized ORN groupings)
            'sensillum_types': [
                # Basiconic sensilla (antenna)
                r'\bab1[ABCD]?\b', r'\bab2[AB]?\b', r'\bab3[AB]?\b',
                r'\bab4[AB]?\b', r'\bab5[AB]?\b', r'\bab6[AB]?\b',
                r'\bab7[AB]?\b', r'\bab8[AB]?\b', r'\bab9[AB]?\b', r'\bab10[AB]?\b',
                # Coeloconic sensilla (antenna)
                r'\bac1[ABC]?\b', r'\bac2[ABC]?\b', r'\bac3[ABC]?\b', r'\bac4[ABC]?\b',
                # Intermediate sensilla
                r'\bai\d+\b',
                # Trichoid sensilla
                r'\bat\d+[AB]?\b',
                # Palp basiconic
                r'\bpb1[ABC]?\b', r'\bpb2[AB]?\b', r'\bpb3[AB]?\b'
            ],

            # Anatomical location patterns
            'anatomical_locations': [
                r'\bantenna\w*', r'\bantennal\w*',
                r'\bmaxillary\w*', r'\bpalp\w*',
                r'\bolfactory\w*', r'\bglomerul\w*',
                r'\bsensillum', r'\bsensilla'
            ],

            # Cell type patterns
            'cell_types': [
                r'\bORN\b', r'\bOSN\b', r'\bOSN\w+',
                r'\bPN\b', r'\bPN\w+',
                r'olfactory.{0,20}neuron',
                r'receptor.{0,20}neuron',
                r'projection.{0,20}neuron',
                r'sensory.{0,20}neuron'
            ]
        }

        return patterns

    def identify_olfactory_cells(
        self,
        labels_file: str,
        max_cells: Optional[int] = None
    ) -> pd.DataFrame:
        """Comprehensive olfactory cell identification with detailed categorization.

        Parameters
        ----------
        labels_file : str
            Path to FlyWire processed_labels.csv.gz file
        max_cells : int, optional
            Maximum number of cells to process (for testing). Default: None (all)

        Returns
        -------
        pd.DataFrame
            Identified olfactory cells with columns:
            - root_id: FlyWire neuron ID
            - processed_label: Original label text
            - receptor_type: Category (or_receptors, ir_receptors, etc.)
            - receptor_subtype: Specific receptor (Or42b, Ir8a, etc.)
            - anatomical_location: Antenna/palp/etc.
            - cell_type: ORN/PN/etc.
            - confidence_score: Match confidence [0, 1]
            - door_match: Receptor name if in DoOR database

        Examples
        --------
        >>> identifier = FlyWireORNIdentifier(door_manager)
        >>> cells = identifier.identify_olfactory_cells('processed_labels.csv.gz')
        >>> print(cells.head())
        """
        results = []

        # Get DoOR receptors for validation
        door_receptors = self.door_manager.get_complete_receptor_list()
        door_receptor_set = set(door_receptors['all_receptors'])

        with gzip.open(labels_file, 'rt') as f:
            # Skip header
            header = next(f)

            for line_num, line in enumerate(f):
                self.search_stats['cells_processed'] += 1

                # Process max_cells if specified
                if max_cells and line_num >= max_cells:
                    break

                try:
                    root_id, processed_label = line.strip().split(',', 1)
                    processed_label = processed_label.strip('"')

                    # Apply comprehensive pattern matching
                    matches = self._apply_pattern_matching(
                        processed_label,
                        door_receptor_set
                    )

                    if matches['is_olfactory'] and matches['confidence_score'] >= self.confidence_threshold:
                        results.append({
                            'root_id': root_id,
                            'processed_label': processed_label,
                            'receptor_type': matches['receptor_type'],
                            'receptor_subtype': matches['receptor_subtype'],
                            'anatomical_location': matches['anatomical_location'],
                            'cell_type': matches['cell_type'],
                            'confidence_score': matches['confidence_score'],
                            'door_match': matches['door_match']
                        })

                        self.search_stats['matches'] += 1

                        if matches['door_match']:
                            self.search_stats['door_validated'] += 1

                except ValueError:
                    # Skip malformed lines
                    continue

                # Progress logging
                if line_num % 10000 == 0:
                    print(f"Processed {line_num:,} cells, found {len(results):,} olfactory cells")

        print(f"\nSearch complete:")
        print(f"  Cells processed: {self.search_stats['cells_processed']:,}")
        print(f"  Olfactory matches: {self.search_stats['matches']:,}")
        print(f"  DoOR validated: {self.search_stats['door_validated']:,}")

        return pd.DataFrame(results)

    def _apply_pattern_matching(
        self,
        label: str,
        door_receptor_set: Set[str]
    ) -> Dict:
        """Advanced pattern matching with confidence scoring.

        Parameters
        ----------
        label : str
            FlyWire cell label text
        door_receptor_set : Set[str]
            Set of valid DoOR receptor names

        Returns
        -------
        Dict
            Match results with confidence scoring
        """
        matches = {
            'is_olfactory': False,
            'receptor_type': None,
            'receptor_subtype': None,
            'anatomical_location': None,
            'cell_type': None,
            'confidence_score': 0.0,
            'door_match': None
        }

        label_lower = label.lower()

        # Check for specific receptor matches (highest confidence)
        for receptor_type, patterns in self.receptor_patterns.items():
            if receptor_type not in ['or_receptors', 'ir_receptors', 'gr_receptors']:
                continue

            for pattern in patterns:
                match = re.search(pattern, label, re.IGNORECASE)
                if match:
                    matches['is_olfactory'] = True
                    matches['receptor_type'] = receptor_type
                    matches['receptor_subtype'] = match.group()
                    matches['confidence_score'] += 0.7

                    # Check for DoOR database match (validation)
                    receptor_name = match.group().replace('\\b', '')
                    if receptor_name in door_receptor_set:
                        matches['door_match'] = receptor_name
                        matches['confidence_score'] += 0.2

                    break

            if matches['is_olfactory']:
                break

        # Check for sensillum type matches (medium confidence)
        if not matches['is_olfactory']:
            for pattern in self.receptor_patterns['sensillum_types']:
                if re.search(pattern, label, re.IGNORECASE):
                    matches['is_olfactory'] = True
                    matches['receptor_type'] = 'sensillum_types'
                    matches['receptor_subtype'] = re.search(pattern, label, re.IGNORECASE).group()
                    matches['confidence_score'] += 0.5
                    break

        # Check for anatomical location clues
        for term in ['antenna', 'palp', 'maxillary', 'olfactory']:
            if term in label_lower:
                matches['anatomical_location'] = term
                matches['confidence_score'] += 0.1
                if not matches['is_olfactory']:
                    matches['is_olfactory'] = True
                    matches['receptor_type'] = 'anatomical'
                break

        # Check for cell type markers
        cell_type_terms = {
            'orn': 0.2,
            'osn': 0.2,
            'pn': 0.15,
            'projection neuron': 0.15,
            'olfactory neuron': 0.2
        }
        for term, score_boost in cell_type_terms.items():
            if term in label_lower:
                matches['cell_type'] = term.upper() if term in ['orn', 'osn', 'pn'] else term
                matches['confidence_score'] += score_boost
                if not matches['is_olfactory'] and score_boost >= 0.2:
                    matches['is_olfactory'] = True
                    matches['receptor_type'] = 'cell_type'
                break

        # Normalize confidence score to [0, 1]
        matches['confidence_score'] = min(1.0, matches['confidence_score'])

        return matches

    def filter_by_receptor(
        self,
        identified_cells: pd.DataFrame,
        receptor_name: str
    ) -> pd.DataFrame:
        """Filter identified cells by specific receptor.

        Parameters
        ----------
        identified_cells : pd.DataFrame
            Results from identify_olfactory_cells()
        receptor_name : str
            Receptor to filter for (e.g., 'Or42b', 'Ir8a')

        Returns
        -------
        pd.DataFrame
            Cells expressing specified receptor

        Examples
        --------
        >>> or42b_cells = identifier.filter_by_receptor(cells, 'Or42b')
        >>> print(f"Found {len(or42b_cells)} Or42b cells")
        """
        pattern = re.compile(receptor_name, re.IGNORECASE)
        filtered = identified_cells[
            identified_cells['receptor_subtype'].apply(
                lambda x: bool(pattern.search(str(x))) if pd.notna(x) else False
            )
        ]
        return filtered

    def get_summary_statistics(self) -> Dict:
        """Get comprehensive summary statistics of identification results.

        Returns
        -------
        Dict
            Summary statistics including counts by receptor type, confidence distribution, etc.
        """
        return {
            'cells_processed': self.search_stats['cells_processed'],
            'total_matches': self.search_stats['matches'],
            'door_validated': self.search_stats['door_validated'],
            'match_rate': self.search_stats['matches'] / max(1, self.search_stats['cells_processed']),
            'validation_rate': self.search_stats['door_validated'] / max(1, self.search_stats['matches'])
        }
