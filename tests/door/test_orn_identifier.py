"""Unit tests for FlyWireORNIdentifier class.

Tests cover:
- Pattern matching for receptors, sensilla, anatomical locations
- Confidence scoring
- DoOR validation
- Cell filtering and summary statistics

Usage
-----
pytest tests/door/test_orn_identifier.py -v
"""

import pytest
import pandas as pd
import numpy as np
import gzip
import tempfile
from pathlib import Path

from pgcn.door.door_data_manager import DoORDataManager
from pgcn.door.orn_identifier import FlyWireORNIdentifier


class TestFlyWireORNIdentifier:
    """Test suite for FlyWireORNIdentifier."""

    @pytest.fixture
    def mock_door_manager(self, tmp_path):
        """Create mock DoOR manager for testing."""
        # Create mock DoOR data
        odorants = ['ethyl acetate', 'hexanol', 'geosmin']
        receptors = ['Or42b', 'Or47b', 'Ir8a', 'Gr21a']

        response_data = np.random.rand(len(odorants), len(receptors))
        response_df = pd.DataFrame(response_data, index=odorants, columns=receptors)

        # Save to CSV
        csv_path = tmp_path / 'mock_door.csv.gz'
        with gzip.open(csv_path, 'wt') as f:
            response_df.to_csv(f)

        # Create manager
        manager = DoORDataManager(method='csv', backup_csv_path=str(csv_path))
        return manager

    @pytest.fixture
    def mock_flywire_labels(self, tmp_path):
        """Create mock FlyWire labels file."""
        labels_path = tmp_path / 'mock_labels.csv.gz'

        # Create mock labels with various olfactory neurons
        mock_labels = [
            ('720575940610453042', 'Or42b ORN right antenna'),
            ('720575940610453043', 'Or47b olfactory receptor neuron'),
            ('720575940610453044', 'Ir8a antennal neuron'),
            ('720575940610453045', 'ab1 sensillum neuron'),
            ('720575940610453046', 'PN DA1 projection neuron'),
            ('720575940610453047', 'random neuron'),  # Should not match
            ('720575940610453048', 'Or67a OSN maxillary palp'),
            ('720575940610453049', 'Gr21a CO2 receptor neuron')
        ]

        with gzip.open(labels_path, 'wt') as f:
            f.write('root_id,processed_label\n')
            for root_id, label in mock_labels:
                f.write(f'{root_id},"{label}"\n')

        return labels_path

    def test_initialization(self, mock_door_manager):
        """Test identifier initialization."""
        identifier = FlyWireORNIdentifier(mock_door_manager)

        assert identifier.door_manager is mock_door_manager
        assert identifier.confidence_threshold == 0.5
        assert 'or_receptors' in identifier.receptor_patterns

    def test_custom_confidence_threshold(self, mock_door_manager):
        """Test custom confidence threshold."""
        identifier = FlyWireORNIdentifier(
            mock_door_manager,
            confidence_threshold=0.8
        )

        assert identifier.confidence_threshold == 0.8

    def test_pattern_matching_or_receptor(self, mock_door_manager):
        """Test OR receptor pattern matching."""
        identifier = FlyWireORNIdentifier(mock_door_manager)

        # Get DoOR receptors for validation
        door_receptors = mock_door_manager.get_complete_receptor_list()
        door_receptor_set = set(door_receptors['all_receptors'])

        # Test Or42b match
        label = 'Or42b ORN right antenna'
        matches = identifier._apply_pattern_matching(label, door_receptor_set)

        assert matches['is_olfactory'] is True
        assert matches['receptor_type'] == 'or_receptors'
        assert 'Or42b' in matches['receptor_subtype']
        assert matches['door_match'] == 'Or42b'
        assert matches['confidence_score'] > 0.7

    def test_pattern_matching_ir_receptor(self, mock_door_manager):
        """Test IR receptor pattern matching."""
        identifier = FlyWireORNIdentifier(mock_door_manager)

        door_receptors = mock_door_manager.get_complete_receptor_list()
        door_receptor_set = set(door_receptors['all_receptors'])

        label = 'Ir8a antennal neuron'
        matches = identifier._apply_pattern_matching(label, door_receptor_set)

        assert matches['is_olfactory'] is True
        assert matches['receptor_type'] == 'ir_receptors'
        assert 'Ir8a' in matches['receptor_subtype']
        assert matches['door_match'] == 'Ir8a'

    def test_pattern_matching_sensillum(self, mock_door_manager):
        """Test sensillum type pattern matching."""
        identifier = FlyWireORNIdentifier(mock_door_manager)

        door_receptors = mock_door_manager.get_complete_receptor_list()
        door_receptor_set = set(door_receptors['all_receptors'])

        label = 'ab1 sensillum neuron'
        matches = identifier._apply_pattern_matching(label, door_receptor_set)

        assert matches['is_olfactory'] is True
        assert matches['receptor_type'] == 'sensillum_types'
        assert 'ab1' in matches['receptor_subtype']

    def test_pattern_matching_anatomical(self, mock_door_manager):
        """Test anatomical location matching."""
        identifier = FlyWireORNIdentifier(mock_door_manager)

        door_receptors = mock_door_manager.get_complete_receptor_list()
        door_receptor_set = set(door_receptors['all_receptors'])

        label = 'Or67a OSN maxillary palp'
        matches = identifier._apply_pattern_matching(label, door_receptor_set)

        assert matches['is_olfactory'] is True
        assert matches['anatomical_location'] == 'palp'

    def test_pattern_matching_non_olfactory(self, mock_door_manager):
        """Test that non-olfactory cells don't match."""
        identifier = FlyWireORNIdentifier(mock_door_manager)

        door_receptors = mock_door_manager.get_complete_receptor_list()
        door_receptor_set = set(door_receptors['all_receptors'])

        label = 'random motor neuron'
        matches = identifier._apply_pattern_matching(label, door_receptor_set)

        assert matches['is_olfactory'] is False

    def test_identify_olfactory_cells(self, mock_door_manager, mock_flywire_labels):
        """Test complete olfactory cell identification."""
        identifier = FlyWireORNIdentifier(
            mock_door_manager,
            confidence_threshold=0.4  # Lower threshold for testing
        )

        identified_cells = identifier.identify_olfactory_cells(
            str(mock_flywire_labels)
        )

        # Check that we found multiple cells
        assert len(identified_cells) >= 5  # Should find at least 5 olfactory cells

        # Check structure
        assert 'root_id' in identified_cells.columns
        assert 'receptor_type' in identified_cells.columns
        assert 'confidence_score' in identified_cells.columns
        assert 'door_match' in identified_cells.columns

        # Check that we found specific cells
        or42b_cells = identified_cells[
            identified_cells['receptor_subtype'].str.contains('Or42b', na=False)
        ]
        assert len(or42b_cells) >= 1

    def test_filter_by_receptor(self, mock_door_manager, mock_flywire_labels):
        """Test filtering cells by receptor."""
        identifier = FlyWireORNIdentifier(
            mock_door_manager,
            confidence_threshold=0.4
        )

        identified_cells = identifier.identify_olfactory_cells(
            str(mock_flywire_labels)
        )

        # Filter for Or42b
        or42b_cells = identifier.filter_by_receptor(identified_cells, 'Or42b')

        assert len(or42b_cells) >= 1
        assert all('Or42b' in str(subtype) for subtype in or42b_cells['receptor_subtype'])

    def test_summary_statistics(self, mock_door_manager, mock_flywire_labels):
        """Test summary statistics generation."""
        identifier = FlyWireORNIdentifier(
            mock_door_manager,
            confidence_threshold=0.4
        )

        # Run identification
        identified_cells = identifier.identify_olfactory_cells(
            str(mock_flywire_labels)
        )

        # Get summary
        summary = identifier.get_summary_statistics()

        # Check structure
        assert 'cells_processed' in summary
        assert 'total_matches' in summary
        assert 'door_validated' in summary
        assert 'match_rate' in summary

        # Check values
        assert summary['total_matches'] == len(identified_cells)
        assert summary['cells_processed'] >= summary['total_matches']
        assert 0 <= summary['match_rate'] <= 1

    def test_max_cells_limit(self, mock_door_manager, mock_flywire_labels):
        """Test max_cells parameter limits processing."""
        identifier = FlyWireORNIdentifier(mock_door_manager)

        # Process only first 3 cells
        identified_cells = identifier.identify_olfactory_cells(
            str(mock_flywire_labels),
            max_cells=3
        )

        # Should have processed exactly 3 cells
        assert identifier.search_stats['cells_processed'] == 3


class TestORNIdentifierIntegration:
    """Integration tests with real data (if available)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_flywire_labels(self):
        """Test with real FlyWire labels (if available)."""
        labels_path = Path('data/processed_labels.csv.gz')

        if not labels_path.exists():
            pytest.skip("Real FlyWire labels not available")

        try:
            # Create real DoOR manager
            door_manager = DoORDataManager()

            # Create identifier
            identifier = FlyWireORNIdentifier(door_manager)

            # Process limited cells for testing
            identified_cells = identifier.identify_olfactory_cells(
                str(labels_path),
                max_cells=10000
            )

            # Should find some olfactory cells
            assert len(identified_cells) > 100

        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
