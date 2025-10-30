"""Unit tests for CircuitLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.connectivity_matrix import ConnectivityMatrix


@pytest.fixture
def cache_dir():
    """Real cache directory (should be populated)."""
    return Path("data/cache")


@pytest.fixture
def circuit_loader(cache_dir):
    """CircuitLoader instance."""
    return CircuitLoader(cache_dir=cache_dir)


def test_circuit_loader_init(cache_dir):
    """Test CircuitLoader initialization."""
    loader = CircuitLoader(cache_dir=cache_dir)
    assert loader.cache_dir == cache_dir


def test_circuit_loader_missing_cache_raises():
    """Test that missing cache directory raises error."""
    with pytest.raises(FileNotFoundError, match="Cache directory not found"):
        CircuitLoader(cache_dir="nonexistent_dir")


def test_load_connectivity_matrix_default(circuit_loader):
    """Test loading connectivity matrix with default parameters."""
    # Biological validation: Loader should construct valid connectivity from cache
    conn_matrix = circuit_loader.load_connectivity_matrix()

    assert isinstance(conn_matrix, ConnectivityMatrix)

    # Check neuron counts are reasonable (biological ranges)
    assert 50 <= conn_matrix.n_pn <= 30000
    assert 1000 <= conn_matrix.n_kc <= 10000
    assert 20 <= conn_matrix.n_mbon <= 150
    assert conn_matrix.n_dan >= 0  # DANs optional


def test_load_connectivity_matrix_row_normalization(circuit_loader):
    """Test row normalization strategy."""
    # Biological rationale: Row normalization models synaptic scaling
    # where each neuron maintains stable total input strength
    conn_matrix = circuit_loader.load_connectivity_matrix(normalize_weights="row")

    # Check that non-zero rows sum to ~1.0 (within floating point tolerance)
    import numpy as np

    row_sums = np.array(conn_matrix.pn_to_kc.sum(axis=1)).ravel()
    nonzero_rows = row_sums > 0
    if np.any(nonzero_rows):
        np.testing.assert_allclose(
            row_sums[nonzero_rows],
            1.0,
            rtol=1e-5,
            err_msg="Row normalization should make rows sum to 1.0",
        )


def test_load_connectivity_matrix_global_normalization(circuit_loader):
    """Test global normalization strategy."""
    # Biological rationale: Global normalization preserves relative connection strength
    conn_matrix = circuit_loader.load_connectivity_matrix(normalize_weights="global")

    # Max weight should be 1.0 (or close to it)
    max_weight = conn_matrix.pn_to_kc.max()
    assert max_weight <= 1.0 + 1e-6  # Account for floating point


def test_load_connectivity_matrix_no_normalization(circuit_loader):
    """Test raw synapse counts (no normalization)."""
    # Biological rationale: Raw counts reflect anatomical connection strength
    conn_matrix = circuit_loader.load_connectivity_matrix(normalize_weights="none")

    # Weights should be positive integers (synapse counts)
    import numpy as np

    data = conn_matrix.pn_to_kc.data
    assert np.all(data > 0)  # All synapses have positive weight


def test_load_connectivity_matrix_invalid_normalization(circuit_loader):
    """Test that invalid normalization raises error."""
    with pytest.raises(ValueError, match="normalize_weights must be one of"):
        circuit_loader.load_connectivity_matrix(normalize_weights="invalid")


def test_load_connectivity_matrix_without_dan(circuit_loader):
    """Test loading without DAN connectivity."""
    conn_matrix = circuit_loader.load_connectivity_matrix(include_dan=False)

    # DAN matrices should be empty
    assert conn_matrix.dan_to_kc.nnz == 0
    assert conn_matrix.dan_to_mbon.nnz == 0


def test_load_connectivity_matrix_kc_subtype_filter(circuit_loader):
    """Test filtering to specific KC subtypes."""
    # Biological rationale: Isolate specific memory systems (e.g., γ for short-term)
    # Try filtering to γ subtypes
    gamma_subtypes = ["g_main", "g_dorsal", "g_sparse"]

    conn_matrix = circuit_loader.load_connectivity_matrix(
        kc_subtypes_filter=gamma_subtypes
    )

    # KCs should be reduced
    # All retained KCs should be γ subtypes
    for kc_id, subtype in conn_matrix.kc_subtypes.items():
        assert subtype in gamma_subtypes


def test_load_connectivity_matrix_invalid_kc_filter(circuit_loader):
    """Test that invalid KC subtype filter raises error."""
    with pytest.raises(ValueError, match="No KCs found"):
        circuit_loader.load_connectivity_matrix(kc_subtypes_filter=["invalid_subtype"])


def test_validate_shapes(circuit_loader):
    """Test shape validation."""
    conn_matrix = circuit_loader.load_connectivity_matrix()

    # Should not raise (dimensions are valid)
    circuit_loader.validate_shapes(conn_matrix)


def test_validate_connectivity(circuit_loader):
    """Test connectivity validation report."""
    conn_matrix = circuit_loader.load_connectivity_matrix()

    # Biological validation: Report should contain expected metrics
    report = circuit_loader.validate_connectivity(conn_matrix)

    # Check report keys
    assert "pn_to_kc_sparsity" in report
    assert "kc_to_mbon_sparsity" in report
    assert "pn_to_kc_fan_in" in report
    assert "kc_to_mbon_fan_in" in report
    assert "orphan_kcs" in report
    assert "orphan_mbons" in report

    # Check fan-in distributions have quantiles
    assert "p10" in report["pn_to_kc_fan_in"]
    assert "p50" in report["pn_to_kc_fan_in"]
    assert "p90" in report["pn_to_kc_fan_in"]

    # Biological expectation: PN→KC should be highly sparse (~95-97%)
    assert 0.8 < report["pn_to_kc_sparsity"] < 1.0

    # Biological expectation: Median KC receives from ~6-8 PNs
    # (may vary depending on filtering criteria)
    median_pn_fan_in = report["pn_to_kc_fan_in"]["p50"]
    assert 0 < median_pn_fan_in < 50  # Reasonable range
