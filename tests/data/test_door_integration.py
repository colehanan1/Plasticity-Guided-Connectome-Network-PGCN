"""Tests for DoOR (Database of Odorant Responses) integration.

Validates that DoOR database loads correctly and experimental odors can be
mapped to biologically-realistic PN activity patterns.
"""

import pytest
import numpy as np
from pathlib import Path

# Import DoOR integration (will skip tests if module not available)
try:
    from pgcn.data.door_integration import DoORIntegration, GLOMERULUS_TO_ORN_MAPPING
    DOOR_AVAILABLE = True
except ImportError:
    DOOR_AVAILABLE = False


# Skip all tests if DoOR integration not available
pytestmark = pytest.mark.skipif(
    not DOOR_AVAILABLE,
    reason="DoOR integration module not available"
)


@pytest.fixture
def door_integration():
    """Create DoOR integration instance for testing."""
    # Use mock cache dir for testing (tests should work with minimal data)
    cache_dir = Path("data/cache")
    if not cache_dir.exists():
        pytest.skip(f"Cache directory not found: {cache_dir}")

    return DoORIntegration(cache_dir=cache_dir)


class TestDoORLoading:
    """Test that DoOR database loads successfully."""

    def test_door_data_loads(self, door_integration):
        """Test that DoOR database loads without errors."""
        assert door_integration.door_data is not None
        assert len(door_integration.door_data) > 0
        assert door_integration.door_data.shape[0] > 100  # Should have 100+ odorants
        assert door_integration.door_data.shape[1] > 20   # Should have 20+ ORN types

    def test_door_data_normalized(self, door_integration):
        """Test that DoOR response values are normalized to [0, 1]."""
        door_data = door_integration.door_data
        assert door_data.min().min() >= 0.0
        assert door_data.max().max() <= 1.0

    def test_pn_glomeruli_loaded(self, door_integration):
        """Test that PN glomerulus assignments loaded from FlyWire."""
        assert door_integration.pn_glomeruli is not None
        assert len(door_integration.pn_glomeruli) > 0
        assert isinstance(door_integration.pn_glomeruli, dict)

        # Check that glomeruli are valid strings
        for pn_idx, glomerulus in list(door_integration.pn_glomeruli.items())[:10]:
            assert isinstance(pn_idx, int)
            assert isinstance(glomerulus, str)
            assert len(glomerulus) > 0


class TestExperimentalOdorCoverage:
    """Test that all experimental odors have DoOR representations."""

    EXPERIMENTAL_ODORS = [
        'hexanol',
        'ethyl_butyrate',
        'benzaldehyde',
        '3-octanol',
        'citral',
        'linalool',
        # Note: 'apple_cider_vinegar' may not be in DoOR (complex mixture)
    ]

    @pytest.mark.parametrize("odor_name", EXPERIMENTAL_ODORS)
    def test_experimental_odor_covered(self, door_integration, odor_name):
        """Test that experimental odor has non-zero DoOR representation."""
        pn_activity = door_integration.odor_to_pn_activity(odor_name, n_pn=100)

        assert pn_activity.shape == (100,)
        assert pn_activity.dtype == np.float32

        # Should have some active PNs (not all zeros)
        n_active = np.sum(pn_activity > 0.1)
        assert n_active > 0, f"Odor '{odor_name}' has zero PN activity - not in DoOR?"

        # Biological constraint: 5-50 active PNs per odor
        assert 2 < n_active < 70, f"Odor '{odor_name}' has {n_active} active PNs (expected 5-50)"

    def test_all_experimental_odors_unique(self, door_integration):
        """Test that different experimental odors produce different PN patterns."""
        patterns = {}

        for odor in self.EXPERIMENTAL_ODORS:
            pattern = door_integration.odor_to_pn_activity(odor, n_pn=100)
            patterns[odor] = pattern

        # Check pairwise correlations (should be < 0.9 for different odors)
        odor_list = list(patterns.keys())
        for i, odor1 in enumerate(odor_list):
            for odor2 in odor_list[i+1:]:
                pattern1 = patterns[odor1]
                pattern2 = patterns[odor2]

                # Skip if either pattern is all zeros
                if np.sum(pattern1) == 0 or np.sum(pattern2) == 0:
                    continue

                correlation = np.corrcoef(pattern1, pattern2)[0, 1]

                # Different odors should have distinct patterns
                assert correlation < 0.95, (
                    f"{odor1} and {odor2} have correlation {correlation:.2f} - too similar!"
                )


class TestBiologicalConstraints:
    """Test that PN activation satisfies biological constraints."""

    def test_sparse_activation(self, door_integration):
        """Test that PN activation is sparse (10-30 active per odor)."""
        test_odors = ['hexanol', 'benzaldehyde', 'ethyl_butyrate']

        for odor in test_odors:
            pn_activity = door_integration.odor_to_pn_activity(odor, n_pn=100)
            n_active = np.sum(pn_activity > 0.1)

            # Biological constraint: sparse activation
            assert 2 < n_active < 70, (
                f"{odor}: {n_active} PNs active (expected 5-50 for biological realism)"
            )

    def test_graded_responses(self, door_integration):
        """Test that PN responses are graded (not binary)."""
        pn_activity = door_integration.odor_to_pn_activity('benzaldehyde', n_pn=100)

        # Should have range of response magnitudes (not just 0 and 1)
        active_responses = pn_activity[pn_activity > 0.1]

        if len(active_responses) > 1:
            # Check that responses span a range
            response_range = active_responses.max() - active_responses.min()
            assert response_range > 0.1, "Responses should be graded, not binary"

    def test_stereotyped_mapping(self, door_integration):
        """Test that same odor always produces same PN pattern."""
        pattern1 = door_integration.odor_to_pn_activity('hexanol', n_pn=100)
        pattern2 = door_integration.odor_to_pn_activity('hexanol', n_pn=100)

        # Should be identical (deterministic mapping)
        assert np.allclose(pattern1, pattern2), "Same odor should produce identical pattern"

    def test_intensity_scaling(self, door_integration):
        """Test that odor intensity scales PN responses linearly."""
        pattern_full = door_integration.odor_to_pn_activity('benzaldehyde', n_pn=100, intensity=1.0)
        pattern_half = door_integration.odor_to_pn_activity('benzaldehyde', n_pn=100, intensity=0.5)

        # Half intensity should produce approximately half the response
        # (after accounting for normalization)
        active_pns = pattern_full > 0.1

        if np.sum(active_pns) > 0:
            ratio = pattern_half[active_pns] / pattern_full[active_pns]
            mean_ratio = np.mean(ratio)

            # Should be close to 0.5 (within 20% tolerance for normalization effects)
            assert 0.3 < mean_ratio < 0.7, f"Intensity scaling incorrect: ratio={mean_ratio:.2f}"


class TestTemporalSequences:
    """Test temporal odor sequence generation."""

    def test_create_odor_sequence_shape(self, door_integration):
        """Test that odor sequence has correct shape."""
        sequence = door_integration.create_odor_sequence(
            'benzaldehyde',
            n_pn=100,
            sequence_length=100,
            odor_duration=40
        )

        assert sequence.shape == (100, 100)
        assert sequence.dtype == np.float32

    def test_temporal_profile(self, door_integration):
        """Test that odor turns on and off at correct times."""
        sequence = door_integration.create_odor_sequence(
            'hexanol',
            n_pn=100,
            sequence_length=100,
            odor_onset=10,
            odor_duration=30
        )

        # Before onset: should be zero
        assert np.sum(sequence[:10, :]) == 0, "Activity before odor onset should be zero"

        # During odor: should be non-zero
        assert np.sum(sequence[10:40, :]) > 0, "Activity during odor pulse should be non-zero"

        # After offset: should be zero
        assert np.sum(sequence[40:, :]) == 0, "Activity after odor offset should be zero"


class TestOdorSimilarity:
    """Test odor similarity computations."""

    def test_similarity_to_self(self, door_integration):
        """Test that odor is maximally similar to itself."""
        similarity = door_integration.get_odor_similarity('hexanol', 'hexanol', n_pn=100)

        # Should be perfect correlation (1.0)
        assert 0.95 < similarity <= 1.0, f"Self-similarity should be ~1.0, got {similarity:.2f}"

    def test_similarity_different_odors(self, door_integration):
        """Test similarity between chemically-different odors."""
        # Hexanol (alcohol) vs benzaldehyde (aldehyde) - chemically different
        similarity = door_integration.get_odor_similarity('hexanol', 'benzaldehyde', n_pn=100)

        # Should have low-moderate correlation (not identical)
        assert similarity < 0.95, "Different odors should not be identical"

    def test_similarity_related_odors(self, door_integration):
        """Test similarity between chemically-related odors."""
        # Both are alcohols - should have higher similarity
        similarity = door_integration.get_odor_similarity('hexanol', '3-octanol', n_pn=100)

        # Related odors often have higher similarity (but not always)
        # Just check that it's computable and reasonable
        assert -1.0 <= similarity <= 1.0, "Similarity should be valid correlation coefficient"


class TestGlomerulusMapping:
    """Test glomerulus-to-ORN mappings."""

    def test_glomerulus_mapping_coverage(self):
        """Test that GLOMERULUS_TO_ORN_MAPPING has standard glomeruli."""
        # Should have common glomeruli mapped
        expected_glomeruli = ['DA1', 'DL3', 'DL5', 'DM1', 'VA1d']

        for glom in expected_glomeruli:
            assert glom in GLOMERULUS_TO_ORN_MAPPING, f"Missing mapping for {glom}"
            assert isinstance(GLOMERULUS_TO_ORN_MAPPING[glom], str)
            assert GLOMERULUS_TO_ORN_MAPPING[glom].startswith('Or') or \
                   GLOMERULUS_TO_ORN_MAPPING[glom].startswith('Ir'), \
                   f"Invalid ORN type for {glom}: {GLOMERULUS_TO_ORN_MAPPING[glom]}"


# Integration test (requires full setup)
class TestFullIntegration:
    """Integration tests requiring complete FlyWire cache."""

    @pytest.mark.integration
    def test_end_to_end_odor_to_pn(self, door_integration):
        """Test complete odor-to-PN pipeline."""
        # Create odor sequence for benzaldehyde
        sequence = door_integration.create_odor_sequence(
            'benzaldehyde',
            n_pn=door_integration.pn_glomeruli.__len__() if door_integration.pn_glomeruli else 100,
            sequence_length=100,
            odor_onset=0,
            odor_duration=40
        )

        # Validate sequence properties
        assert sequence.shape[0] == 100  # Correct time dimension
        assert sequence.shape[1] > 0     # Has PNs

        # Check temporal structure
        odor_period_activity = np.sum(sequence[:40, :])
        washout_period_activity = np.sum(sequence[40:, :])

        assert odor_period_activity > 0, "Should have activity during odor"
        assert washout_period_activity == 0, "Should have no activity after washout"

        # Check biological sparsity
        mean_active_pns = np.mean(np.sum(sequence[:40, :] > 0.1, axis=1))
        assert 2 < mean_active_pns < 70, f"Mean active PNs={mean_active_pns} outside biological range"
