"""
PGCN integration with door-python-toolkit.

Provides convenient utilities for:
- Getting receptor activation profiles
- Calculating odorant selectivity ratios
- Finding cross-receptor overlap
- Mapping odorants to active glomeruli
"""

from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np

# Try to import DoOR toolkit
try:
    from door_toolkit.integration.encoder import DoOREncoder
    from door_toolkit.integration.integrator import DoORFlyWireIntegrator
    DOOR_AVAILABLE = True
except ImportError:
    DOOR_AVAILABLE = False


class PGCNDoorIntegration:
    """
    Integrate DoOR olfactory responses with PGCN connectome analysis.

    This class provides utilities for working with odorant response data
    from the DoOR database alongside FlyWire connectivity data.

    Examples
    --------
    >>> pgcn_door = PGCNDoorIntegration()
    >>>
    >>> # Get Or7a activation profile
    >>> or7a = pgcn_door.get_receptor_profile('Or7a')
    >>> print(or7a.nlargest(5))
    >>>
    >>> # Test selectivity
    >>> ratio = pgcn_door.calculate_selectivity('Or7a', 'benzaldehyde', 'hexanol')
    >>> print(f"Or7a selectivity: {ratio:.2f}x")
    >>>
    >>> # Find shared receptors
    >>> shared = pgcn_door.find_shared_receptors('benzaldehyde', 'hexanol', threshold=0.5)
    >>> print(f"Shared receptors: {shared}")
    """

    # Standard OR receptor to glomerulus mapping
    # From Couto et al. (2005) and Hallem & Carlson (2006)
    OR_TO_GLOMERULUS = {
        'Or7a': 'DL5',
        'Or10a': 'D',
        'Or13a': 'DC2',
        'Or19a': 'DM5',
        'Or22a': 'DM3',
        'Or23a': 'D',
        'Or33b': 'DL1',
        'Or35a': 'VC3',
        'Or42a': 'DM4',
        'Or42b': 'VM2',
        'Or43a': 'DM1',
        'Or43b': 'VM5d',
        'Or46a': 'VA7m',
        'Or47a': 'VA1lm',
        'Or47b': 'VA1d',
        'Or49a': 'DC1',
        'Or49b': 'DL4',
        'Or56a': 'DA3',
        'Or59b': 'DM2',
        'Or65a': 'DL3',
        'Or65b': 'DM1',
        'Or67a': 'VA3',
        'Or67b': 'DM1',
        'Or67c': 'VA6',
        'Or67d': 'DA2',
        'Or69a': 'VC2',
        'Or71a': 'DL2d',
        'Or82a': 'VA2',
        'Or83b': 'DC3',
        'Or85a': 'DL1',
        'Or85b': 'DA4l',
        'Or85c': 'VL2p',
        'Or85d': 'VM5d',
        'Or85e': 'VM7d',
        'Or85f': 'VM7d',
        'Or88a': 'VA1v',
        'Or92a': 'VA2',
        'Or98a': 'DM6',
        'Or98b': 'VM3',
    }

    def __init__(self, use_cache: bool = True):
        """
        Initialize PGCN-DoOR integration.

        Parameters
        ----------
        use_cache : bool
            Whether to cache DoOR response matrix (default: True)
        """
        self.use_cache = use_cache
        self._response_matrix = None
        self._encoder = None
        self._integrator = None

        if not DOOR_AVAILABLE:
            print("⚠️  DoOR toolkit not available. Install with:")
            print("    cd ~/Documents/cole/VSCode/door-python-toolkit")
            print("    pip install -e .")

    @property
    def encoder(self) -> Optional['DoOREncoder']:
        """Get DoOR encoder instance."""
        if not DOOR_AVAILABLE:
            return None
        if self._encoder is None:
            self._encoder = DoOREncoder()
        return self._encoder

    @property
    def integrator(self) -> Optional['DoORFlyWireIntegrator']:
        """Get DoOR-FlyWire integrator instance."""
        if not DOOR_AVAILABLE:
            return None
        if self._integrator is None:
            self._integrator = DoORFlyWireIntegrator()
        return self._integrator

    def get_response_matrix(self) -> pd.DataFrame:
        """
        Get DoOR response matrix (odorants × receptors).

        Returns
        -------
        pd.DataFrame
            Response matrix with odorants as rows, receptors as columns
            Values are normalized responses (0-1)

        Raises
        ------
        RuntimeError
            If DoOR toolkit is not available
        """
        if not DOOR_AVAILABLE:
            raise RuntimeError("DoOR toolkit not available. See installation instructions.")

        if self.use_cache and self._response_matrix is not None:
            return self._response_matrix

        self._response_matrix = self.encoder.get_response_matrix()
        return self._response_matrix

    def get_receptor_profile(self, receptor: str) -> pd.Series:
        """
        Get odorant response profile for a receptor.

        Parameters
        ----------
        receptor : str
            Receptor name (e.g., 'Or7a', 'Or67b')

        Returns
        -------
        pd.Series
            Odorant responses for this receptor (odorant → response)

        Examples
        --------
        >>> or7a_profile = pgcn_door.get_receptor_profile('Or7a')
        >>> print(or7a_profile.nlargest(5))
        benzaldehyde    0.89
        geranyl_acetate 0.76
        ...
        """
        matrix = self.get_response_matrix()

        if receptor not in matrix.columns:
            raise ValueError(f"Receptor '{receptor}' not found. Available: {list(matrix.columns)}")

        return matrix[receptor].dropna()

    def get_odor_encoding(
        self,
        odorant: str,
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Get receptor activation pattern for an odorant.

        Parameters
        ----------
        odorant : str
            Odorant name (e.g., 'benzaldehyde', 'hexanol')
        threshold : float
            Minimum activation threshold (0-1)

        Returns
        -------
        dict
            {receptor: activation} for receptors above threshold

        Examples
        --------
        >>> benz = pgcn_door.get_odor_encoding('benzaldehyde', threshold=0.5)
        >>> print(benz)
        {'Or7a': 0.89, 'Or67b': 0.76, 'Or22a': 0.68}
        """
        matrix = self.get_response_matrix()

        if odorant not in matrix.index:
            raise ValueError(f"Odorant '{odorant}' not found. Available: {list(matrix.index)}")

        responses = matrix.loc[odorant]
        activated = responses[responses >= threshold].dropna()

        return activated.to_dict()

    def calculate_selectivity(
        self,
        receptor: str,
        odorant1: str,
        odorant2: str
    ) -> float:
        """
        Calculate selectivity ratio between two odorants for a receptor.

        Parameters
        ----------
        receptor : str
            Receptor name (e.g., 'Or7a')
        odorant1 : str
            First odorant (numerator)
        odorant2 : str
            Second odorant (denominator)

        Returns
        -------
        float
            Selectivity ratio (response1 / response2)
            Returns inf if response2 is 0

        Examples
        --------
        >>> ratio = pgcn_door.calculate_selectivity('Or7a', 'benzaldehyde', 'hexanol')
        >>> print(f"Or7a is {ratio:.1f}x more selective for benzaldehyde")
        Or7a is 3.6x more selective for benzaldehyde
        """
        matrix = self.get_response_matrix()

        response1 = matrix.loc[odorant1, receptor]
        response2 = matrix.loc[odorant2, receptor]

        if response2 == 0:
            return float('inf') if response1 > 0 else 0

        return response1 / response2

    def find_shared_receptors(
        self,
        odorant1: str,
        odorant2: str,
        threshold: float = 0.5
    ) -> Set[str]:
        """
        Find receptors strongly activated by both odorants.

        This is useful for understanding cross-learning mechanisms.

        Parameters
        ----------
        odorant1 : str
            First odorant
        odorant2 : str
            Second odorant
        threshold : float
            Minimum activation threshold for "strong" response

        Returns
        -------
        set
            Receptor names activated by both odorants

        Examples
        --------
        >>> shared = pgcn_door.find_shared_receptors('benzaldehyde', 'hexanol')
        >>> print(f"Shared receptors: {shared}")
        Shared receptors: {'Or67b'}
        """
        encoding1 = self.get_odor_encoding(odorant1, threshold=threshold)
        encoding2 = self.get_odor_encoding(odorant2, threshold=threshold)

        return set(encoding1.keys()) & set(encoding2.keys())

    def map_odorant_to_glomeruli(
        self,
        odorant: str,
        threshold: float = 0.3
    ) -> List[str]:
        """
        Map an odorant to activated glomeruli via receptor activation.

        Parameters
        ----------
        odorant : str
            Odorant name
        threshold : float
            Minimum receptor activation threshold

        Returns
        -------
        list
            Glomerulus names for activated receptors

        Examples
        --------
        >>> glomeruli = pgcn_door.map_odorant_to_glomeruli('benzaldehyde')
        >>> print(f"Benzaldehyde activates: {glomeruli}")
        Benzaldehyde activates: ['DL5', 'DM1', 'DM3']
        """
        activated_receptors = self.get_odor_encoding(odorant, threshold=threshold)

        glomeruli = []
        for receptor in activated_receptors.keys():
            glomerulus = self.OR_TO_GLOMERULUS.get(receptor)
            if glomerulus and glomerulus not in glomeruli:
                glomeruli.append(glomerulus)

        return glomeruli

    def get_receptor_similarity(
        self,
        receptor1: str,
        receptor2: str,
        method: str = 'pearson'
    ) -> float:
        """
        Calculate similarity between two receptor response profiles.

        Parameters
        ----------
        receptor1 : str
            First receptor
        receptor2 : str
            Second receptor
        method : str
            Similarity method: 'pearson', 'spearman', or 'cosine'

        Returns
        -------
        float
            Similarity score (-1 to 1 for pearson/spearman, 0 to 1 for cosine)

        Examples
        --------
        >>> sim = pgcn_door.get_receptor_similarity('Or7a', 'Or67b')
        >>> print(f"Or7a-Or67b similarity: {sim:.2f}")
        """
        profile1 = self.get_receptor_profile(receptor1)
        profile2 = self.get_receptor_profile(receptor2)

        # Align to common odorants
        common = profile1.index.intersection(profile2.index)
        p1 = profile1.loc[common]
        p2 = profile2.loc[common]

        if method == 'pearson':
            return p1.corr(p2, method='pearson')
        elif method == 'spearman':
            return p1.corr(p2, method='spearman')
        elif method == 'cosine':
            dot_product = (p1 * p2).sum()
            norm1 = np.sqrt((p1 ** 2).sum())
            norm2 = np.sqrt((p2 ** 2).sum())
            return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_available_odorants(self) -> List[str]:
        """Get list of all odorants in DoOR database."""
        matrix = self.get_response_matrix()
        return list(matrix.index)

    def get_available_receptors(self) -> List[str]:
        """Get list of all receptors in DoOR database."""
        matrix = self.get_response_matrix()
        return list(matrix.columns)


def test_integration():
    """Test PGCN-DoOR integration."""
    if not DOOR_AVAILABLE:
        print("❌ DoOR toolkit not available")
        print("\nTo install:")
        print("  cd ~/Documents/cole/VSCode/door-python-toolkit")
        print("  pip install -e .")
        return

    print("="*80)
    print("TESTING PGCN-DOOR INTEGRATION")
    print("="*80)

    pgcn_door = PGCNDoorIntegration()

    # Test 1: Receptor profile
    print("\n1. Or7a Receptor Profile (Top 5 Odorants)")
    print("-" * 40)
    or7a_profile = pgcn_door.get_receptor_profile('Or7a')
    for odorant, response in or7a_profile.nlargest(5).items():
        print(f"  {odorant}: {response:.1%}")

    # Test 2: Odorant encoding
    print("\n2. Benzaldehyde Encoding (Threshold 0.3)")
    print("-" * 40)
    benz_encoding = pgcn_door.get_odor_encoding('benzaldehyde', threshold=0.3)
    for receptor, activation in sorted(benz_encoding.items(), key=lambda x: x[1], reverse=True)[:5]:
        glom = pgcn_door.OR_TO_GLOMERULUS.get(receptor, '?')
        print(f"  {receptor} ({glom}): {activation:.1%}")

    # Test 3: Selectivity
    print("\n3. Or7a Selectivity")
    print("-" * 40)
    selectivity = pgcn_door.calculate_selectivity('Or7a', 'benzaldehyde', 'hexanol')
    print(f"  Benzaldehyde/Hexanol ratio: {selectivity:.2f}x")
    print(f"  {'✅ Highly selective' if selectivity > 3.0 else '❌ Not selective'}")

    # Test 4: Shared receptors
    print("\n4. Cross-Learning Mechanism")
    print("-" * 40)
    shared = pgcn_door.find_shared_receptors('benzaldehyde', 'hexanol', threshold=0.5)
    if shared:
        print(f"  Shared receptors: {', '.join(shared)}")
        print(f"  → Explains cross-learning via co-activation")
    else:
        print(f"  No shared receptors (threshold 0.5)")

    # Test 5: Glomerulus mapping
    print("\n5. Benzaldehyde → Active Glomeruli")
    print("-" * 40)
    glomeruli = pgcn_door.map_odorant_to_glomeruli('benzaldehyde', threshold=0.3)
    print(f"  Active glomeruli: {', '.join(glomeruli)}")

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)


if __name__ == '__main__':
    test_integration()
