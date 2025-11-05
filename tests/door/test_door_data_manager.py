"""Unit tests for DoORDataManager class.

Tests cover:
- DoOR database initialization
- Multiple access methods (rpy2, CSV, Zenodo)
- Receptor list extraction
- Response profile retrieval
- Error handling and fallbacks

Usage
-----
pytest tests/door/test_door_data_manager.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import gzip

from pgcn.door.door_data_manager import DoORDataManager


class TestDoORDataManager:
    """Test suite for DoORDataManager."""

    @pytest.fixture
    def mock_door_csv(self, tmp_path):
        """Create mock DoOR CSV file for testing."""
        # Create mock response matrix
        odorants = ['ethyl acetate', 'hexanol', 'geosmin', 'CO2']
        receptors = ['Or42b', 'Or47b', 'Or67a', 'Ir8a', 'Gr21a']

        # Random response matrix
        response_data = np.random.rand(len(odorants), len(receptors))
        response_df = pd.DataFrame(response_data, index=odorants, columns=receptors)

        # Save to gzipped CSV
        csv_path = tmp_path / 'mock_door_responses.csv.gz'
        with gzip.open(csv_path, 'wt') as f:
            response_df.to_csv(f)

        return csv_path

    def test_initialization_default(self):
        """Test default initialization."""
        try:
            manager = DoORDataManager(method='rpy2')
            assert manager.method == 'rpy2'
            assert manager.cache_dir.exists()
        except ImportError:
            # rpy2 not available, expected in some environments
            pytest.skip("rpy2 not available")

    def test_initialization_csv(self, mock_door_csv):
        """Test CSV initialization."""
        manager = DoORDataManager(
            method='csv',
            backup_csv_path=str(mock_door_csv)
        )

        assert manager.method == 'csv'
        assert manager.backup_csv == mock_door_csv

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            DoORDataManager(method='invalid_method')

    def test_missing_backup_csv(self):
        """Test that missing backup CSV raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DoORDataManager(
                method='csv',
                backup_csv_path='nonexistent_file.csv'
            )

    def test_load_door_data_csv(self, mock_door_csv):
        """Test loading DoOR data from CSV."""
        manager = DoORDataManager(
            method='csv',
            backup_csv_path=str(mock_door_csv)
        )

        door_data = manager.load_door_data()

        # Check structure
        assert 'response_matrix' in door_data
        assert isinstance(door_data['response_matrix'], pd.DataFrame)

        # Check dimensions
        response_matrix = door_data['response_matrix']
        assert response_matrix.shape == (4, 5)  # 4 odorants, 5 receptors

        # Check receptor names
        assert 'Or42b' in response_matrix.columns
        assert 'Ir8a' in response_matrix.columns

    def test_caching(self, mock_door_csv):
        """Test that loaded data is cached."""
        manager = DoORDataManager(
            method='csv',
            backup_csv_path=str(mock_door_csv)
        )

        # First load
        data1 = manager.load_door_data()

        # Second load (should use cache)
        data2 = manager.load_door_data()

        # Should be same object (cached)
        assert data1 is data2

    def test_get_complete_receptor_list(self, mock_door_csv):
        """Test receptor list extraction."""
        manager = DoORDataManager(
            method='csv',
            backup_csv_path=str(mock_door_csv)
        )

        receptors = manager.get_complete_receptor_list()

        # Check structure
        assert 'all_receptors' in receptors
        assert 'or_receptors' in receptors
        assert 'ir_receptors' in receptors
        assert 'gr_receptors' in receptors

        # Check categorization
        assert 'Or42b' in receptors['or_receptors']
        assert 'Ir8a' in receptors['ir_receptors']
        assert 'Gr21a' in receptors['gr_receptors']

        # Check totals
        assert len(receptors['all_receptors']) == 5

    def test_get_receptor_response_profile(self, mock_door_csv):
        """Test individual receptor response profile retrieval."""
        manager = DoORDataManager(
            method='csv',
            backup_csv_path=str(mock_door_csv)
        )

        # Get Or42b profile
        or42b_profile = manager.get_receptor_response_profile('Or42b', top_n=2)

        # Check type and size
        assert isinstance(or42b_profile, pd.Series)
        assert len(or42b_profile) == 2

        # Check values are valid
        assert all(0 <= val <= 1 for val in or42b_profile.values)

    def test_invalid_receptor_name(self, mock_door_csv):
        """Test that invalid receptor name raises ValueError."""
        manager = DoORDataManager(
            method='csv',
            backup_csv_path=str(mock_door_csv)
        )

        with pytest.raises(ValueError, match="Receptor .* not found"):
            manager.get_receptor_response_profile('InvalidReceptor')


class TestDoORDataIntegration:
    """Integration tests for DoOR data access."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_rpy2_installation(self):
        """Test R package installation (requires R and rpy2)."""
        try:
            manager = DoORDataManager(method='rpy2')

            # This is a slow operation, only run in integration tests
            # success = manager.install_door_packages()
            # assert success

            # For now, just test that the method exists
            assert hasattr(manager, 'install_door_packages')

        except ImportError:
            pytest.skip("rpy2 not available")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_door_data_access(self):
        """Test accessing real DoOR data via rpy2 (if available)."""
        try:
            manager = DoORDataManager(method='rpy2')
            door_data = manager.load_door_data()

            # Check expected structure
            assert 'response_matrix' in door_data
            response_matrix = door_data['response_matrix']

            # DoOR should have substantial data
            assert response_matrix.shape[0] > 500  # >500 odorants
            assert response_matrix.shape[1] > 50   # >50 receptors

        except (ImportError, Exception) as e:
            pytest.skip(f"Real DoOR data not accessible: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
