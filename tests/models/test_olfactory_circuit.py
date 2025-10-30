"""Unit tests for OlfactoryCircuit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.olfactory_circuit import OlfactoryCircuit


@pytest.fixture
def connectivity_matrix():
    """Load real connectivity matrix from cache."""
    loader = CircuitLoader(cache_dir=Path("data/cache"))
    return loader.load_connectivity_matrix(normalize_weights="row")


@pytest.fixture
def olfactory_circuit(connectivity_matrix):
    """OlfactoryCircuit instance with default parameters."""
    return OlfactoryCircuit(connectivity_matrix, kc_sparsity_target=0.05)


def test_olfactory_circuit_init(connectivity_matrix):
    """Test OlfactoryCircuit initialization."""
    # Biological rationale: 5% KC sparsity is canonical in Drosophila MB
    circuit = OlfactoryCircuit(connectivity_matrix, kc_sparsity_target=0.05)

    assert circuit.sparsity_target == 0.05
    assert circuit.k_winners == max(1, int(connectivity_matrix.n_kc * 0.05))
    assert circuit.sparsity_mode == "k_winners_take_all"


def test_olfactory_circuit_invalid_sparsity():
    """Test that invalid sparsity target raises error."""
    from data_loaders.circuit_loader import CircuitLoader

    loader = CircuitLoader(cache_dir=Path("data/cache"))
    conn_matrix = loader.load_connectivity_matrix()

    # Sparsity must be in (0, 1)
    with pytest.raises(ValueError, match="kc_sparsity_target must be in"):
        OlfactoryCircuit(conn_matrix, kc_sparsity_target=0.0)

    with pytest.raises(ValueError, match="kc_sparsity_target must be in"):
        OlfactoryCircuit(conn_matrix, kc_sparsity_target=1.0)

    with pytest.raises(ValueError, match="kc_sparsity_target must be in"):
        OlfactoryCircuit(conn_matrix, kc_sparsity_target=1.5)


def test_olfactory_circuit_invalid_sparsity_mode(connectivity_matrix):
    """Test that invalid sparsity mode raises error."""
    with pytest.raises(ValueError, match="kc_sparsity_mode must be one of"):
        OlfactoryCircuit(connectivity_matrix, kc_sparsity_mode="invalid")


def test_propagate_pn_to_kc_shape(olfactory_circuit):
    """Test PN→KC propagation returns correct shape."""
    # Create PN input
    pn_activity = np.random.rand(olfactory_circuit.connectivity.n_pn)

    # Propagate
    kc_activity = olfactory_circuit.propagate_pn_to_kc(pn_activity)

    # Check output shape
    assert kc_activity.shape == (olfactory_circuit.connectivity.n_kc,)
    assert isinstance(kc_activity, np.ndarray)


def test_propagate_pn_to_kc_wrong_shape(olfactory_circuit):
    """Test that wrong PN input shape raises error."""
    # Wrong shape input
    wrong_pn_activity = np.random.rand(100)  # Arbitrary wrong size

    with pytest.raises(ValueError, match="pn_activity shape"):
        olfactory_circuit.propagate_pn_to_kc(wrong_pn_activity)


def test_k_winners_take_all_enforces_sparsity(olfactory_circuit):
    """Test that k-WTA enforces KC sparsity."""
    # Biological rationale: Lateral inhibition (APL) suppresses weakly-activated KCs
    # Only top ~5% should fire
    pn_activity = np.random.rand(olfactory_circuit.connectivity.n_pn)

    kc_activity = olfactory_circuit.propagate_pn_to_kc(pn_activity)

    # Count active KCs
    n_active = np.count_nonzero(kc_activity)

    # Should be approximately k_winners (±1 for rounding/ties)
    expected_k = olfactory_circuit.k_winners
    assert abs(n_active - expected_k) <= expected_k * 0.1  # Within 10% tolerance


def test_sparsity_fraction_computed_correctly(olfactory_circuit):
    """Test sparsity fraction measurement."""
    # Biological validation: Measured sparsity should match target
    pn_activity = np.random.rand(olfactory_circuit.connectivity.n_pn)
    kc_activity = olfactory_circuit.propagate_pn_to_kc(pn_activity)

    sparsity_fraction = olfactory_circuit.compute_kc_sparsity_fraction(kc_activity)

    # Should be close to target (within ±2% for rounding)
    assert abs(sparsity_fraction - olfactory_circuit.sparsity_target) < 0.02


def test_propagate_kc_to_mbon_shape(olfactory_circuit):
    """Test KC→MBON propagation returns correct shape."""
    # Create sparse KC input
    kc_activity = np.zeros(olfactory_circuit.connectivity.n_kc)
    k = olfactory_circuit.k_winners
    kc_activity[:k] = 1.0  # Activate top k KCs

    # Propagate
    mbon_output = olfactory_circuit.propagate_kc_to_mbon(kc_activity)

    # Check output shape
    assert mbon_output.shape == (olfactory_circuit.connectivity.n_mbon,)
    assert isinstance(mbon_output, np.ndarray)


def test_propagate_kc_to_mbon_wrong_shape(olfactory_circuit):
    """Test that wrong KC input shape raises error."""
    wrong_kc_activity = np.random.rand(100)

    with pytest.raises(ValueError, match="kc_activity shape"):
        olfactory_circuit.propagate_kc_to_mbon(wrong_kc_activity)


def test_forward_pass_zero_input_produces_low_output(olfactory_circuit):
    """Test that zero PN input produces near-zero MBON output."""
    # Biological rationale: No odor → no PN activity → no KC activity → minimal MBON response
    pn_activity = np.zeros(olfactory_circuit.connectivity.n_pn)

    mbon_output = olfactory_circuit.forward_pass(pn_activity)

    # Should be all zeros (or near-zero for numerical reasons)
    assert np.allclose(mbon_output, 0.0, atol=1e-10)


def test_forward_pass_with_intermediates(olfactory_circuit):
    """Test forward pass with intermediate diagnostics."""
    pn_activity = np.random.rand(olfactory_circuit.connectivity.n_pn)

    mbon_output, diagnostics = olfactory_circuit.forward_pass(
        pn_activity, return_intermediates=True
    )

    # Check diagnostics keys
    assert "kc_activity" in diagnostics
    assert "sparsity_fraction" in diagnostics
    assert "mbon_mean" in diagnostics
    assert "mbon_std" in diagnostics

    # Check KC activity shape
    assert diagnostics["kc_activity"].shape == (olfactory_circuit.connectivity.n_kc,)

    # Sparsity should match target
    assert abs(diagnostics["sparsity_fraction"] - olfactory_circuit.sparsity_target) < 0.02


def test_forward_pass_without_intermediates(olfactory_circuit):
    """Test forward pass returns only MBON output."""
    pn_activity = np.random.rand(olfactory_circuit.connectivity.n_pn)

    mbon_output = olfactory_circuit.forward_pass(
        pn_activity, return_intermediates=False
    )

    # Should return only array (not tuple)
    assert isinstance(mbon_output, np.ndarray)
    assert mbon_output.shape == (olfactory_circuit.connectivity.n_mbon,)


def test_activate_pns_by_glomeruli(olfactory_circuit):
    """Test PN activation by glomerulus names."""
    # Biological rationale: Glomeruli define odorant receptor specificity
    # Get available glomeruli from connectivity
    available_glomeruli = list(set(olfactory_circuit.connectivity.pn_glomeruli.values()))
    if "unknown" in available_glomeruli:
        available_glomeruli.remove("unknown")

    if len(available_glomeruli) < 2:
        pytest.skip("Not enough glomeruli in connectivity matrix")

    # Activate first two glomeruli
    target_glomeruli = available_glomeruli[:2]
    pn_activity = olfactory_circuit.activate_pns_by_glomeruli(
        target_glomeruli, firing_rate=1.0
    )

    # Check shape
    assert pn_activity.shape == (olfactory_circuit.connectivity.n_pn,)

    # Count active PNs
    n_active = np.count_nonzero(pn_activity)
    assert n_active > 0  # Should activate some PNs

    # All active PNs should belong to target glomeruli
    active_indices = np.where(pn_activity > 0)[0]
    for idx in active_indices:
        pn_id = olfactory_circuit.connectivity.pn_ids[idx]
        glom = olfactory_circuit.connectivity.pn_glomeruli.get(pn_id, "unknown")
        assert glom in target_glomeruli


def test_edge_case_single_pn_fires(olfactory_circuit):
    """Test circuit handles single PN firing gracefully."""
    # Biological scenario: Very specific odor activates only one glomerulus
    pn_activity = np.zeros(olfactory_circuit.connectivity.n_pn)
    pn_activity[0] = 1.0  # Activate only first PN

    # Should not raise error
    mbon_output = olfactory_circuit.forward_pass(pn_activity)

    # Output should be valid (may be zero if PN has no KC connections)
    assert mbon_output.shape == (olfactory_circuit.connectivity.n_mbon,)
    assert np.all(np.isfinite(mbon_output))


def test_deterministic_forward_pass(olfactory_circuit):
    """Test that same input produces same output (determinism)."""
    # Biological note: Real neurons have trial-to-trial variability,
    # but deterministic model is appropriate for testing connectivity structure
    pn_activity = np.random.rand(olfactory_circuit.connectivity.n_pn)

    # Run forward pass twice
    mbon_output_1 = olfactory_circuit.forward_pass(pn_activity)
    mbon_output_2 = olfactory_circuit.forward_pass(pn_activity)

    # Should be identical
    np.testing.assert_array_equal(mbon_output_1, mbon_output_2)


def test_k_winners_edge_case_k_zero():
    """Test k-WTA with k=0 (all zeros)."""
    from pgcn.models.olfactory_circuit import OlfactoryCircuit

    activations = np.random.rand(100)

    # Access private method for testing
    loader = CircuitLoader(cache_dir=Path("data/cache"))
    conn_matrix = loader.load_connectivity_matrix()
    circuit = OlfactoryCircuit(conn_matrix)

    result = circuit._apply_k_winners_take_all(activations, k=0)

    # Should be all zeros
    assert np.all(result == 0.0)


def test_k_winners_edge_case_k_exceeds_n():
    """Test k-WTA with k > n (all winners)."""
    from pgcn.models.olfactory_circuit import OlfactoryCircuit

    activations = np.random.rand(100)

    loader = CircuitLoader(cache_dir=Path("data/cache"))
    conn_matrix = loader.load_connectivity_matrix()
    circuit = OlfactoryCircuit(conn_matrix)

    result = circuit._apply_k_winners_take_all(activations, k=200)

    # Should retain all activations
    np.testing.assert_array_equal(result, activations)


def test_multiple_trials_maintain_sparsity(olfactory_circuit):
    """Test that KC sparsity is maintained across multiple random trials."""
    # Biological validation: Sparsity should be consistent across odor presentations
    n_trials = 50
    sparsity_values = []

    for _ in range(n_trials):
        pn_activity = np.random.rand(olfactory_circuit.connectivity.n_pn)
        kc_activity = olfactory_circuit.propagate_pn_to_kc(pn_activity)
        sparsity = olfactory_circuit.compute_kc_sparsity_fraction(kc_activity)
        sparsity_values.append(sparsity)

    # Mean sparsity should be close to target
    mean_sparsity = np.mean(sparsity_values)
    assert abs(mean_sparsity - olfactory_circuit.sparsity_target) < 0.01

    # Variance should be low (sparsity enforcement is consistent)
    std_sparsity = np.std(sparsity_values)
    assert std_sparsity < 0.01  # Within 1% variation
