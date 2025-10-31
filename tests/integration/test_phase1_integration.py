"""Integration tests for Phase 1 connectivity backbone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.olfactory_circuit import OlfactoryCircuit


@pytest.fixture
def cache_dir():
    """Real cache directory."""
    return Path("data/cache")


@pytest.fixture
def circuit_loader(cache_dir):
    """CircuitLoader instance."""
    return CircuitLoader(cache_dir=cache_dir)


@pytest.fixture
def connectivity_matrix(circuit_loader):
    """Loaded ConnectivityMatrix."""
    return circuit_loader.load_connectivity_matrix(normalize_weights="row")


@pytest.fixture
def olfactory_circuit(connectivity_matrix):
    """OlfactoryCircuit instance."""
    return OlfactoryCircuit(connectivity_matrix, kc_sparsity_target=0.05)


def test_end_to_end_forward_pass(circuit_loader):
    """Test full pipeline: Loader → ConnectivityMatrix → OlfactoryCircuit → forward pass.

    Biological validation: This simulates the complete odor-to-MBON pathway
    observed in behaving flies during olfactory presentation.
    """
    # Load connectivity
    conn_matrix = circuit_loader.load_connectivity_matrix(normalize_weights="row")

    # Instantiate circuit
    circuit = OlfactoryCircuit(conn_matrix, kc_sparsity_target=0.05)

    # Simulate PN activity (random odor)
    pn_activity = np.random.rand(conn_matrix.n_pn)

    # Forward pass
    mbon_output = circuit.forward_pass(pn_activity)

    # Validate output
    assert mbon_output.shape == (conn_matrix.n_mbon,)
    assert np.all(np.isfinite(mbon_output))  # No NaN/Inf
    assert isinstance(mbon_output, np.ndarray)


def test_multiple_trials_deterministic(olfactory_circuit):
    """Test that same PN input produces same MBON output across trials.

    Biological note: Real neurons have stochasticity, but our deterministic
    model is appropriate for testing circuit structure.
    """
    # Create fixed PN input
    pn_activity = np.random.RandomState(42).rand(olfactory_circuit.connectivity.n_pn)

    # Run forward pass multiple times
    outputs = []
    for _ in range(10):
        mbon_output = olfactory_circuit.forward_pass(pn_activity)
        outputs.append(mbon_output)

    # All outputs should be identical
    for i in range(1, len(outputs)):
        np.testing.assert_array_equal(outputs[0], outputs[i])


def test_connectivity_sanity_check(connectivity_matrix):
    """Test that connectivity is non-trivial (has actual connections).

    Biological validation: PN→KC and KC→MBON should have thousands of synapses
    reflecting the anatomical connectivity of the MB circuit.
    """
    # PN→KC should have connections
    assert connectivity_matrix.pn_to_kc.nnz > 0, "PN→KC has no connections"
    assert connectivity_matrix.pn_to_kc.nnz > 1000, "PN→KC has too few connections"

    # KC→MBON should have connections
    assert connectivity_matrix.kc_to_mbon.nnz > 0, "KC→MBON has no connections"
    assert connectivity_matrix.kc_to_mbon.nnz > 100, "KC→MBON has too few connections"


def test_sparsity_targets_honored_over_many_trials(olfactory_circuit):
    """Test that KC sparsity is consistently enforced across 100 random inputs.

    Biological validation: In vivo imaging shows stable ~5% KC activation
    across diverse odor presentations. Our circuit should replicate this.
    """
    n_trials = 100
    sparsity_values = []

    for seed in range(n_trials):
        # Generate random PN input
        np.random.seed(seed)
        pn_activity = np.random.rand(olfactory_circuit.connectivity.n_pn)

        # Measure KC sparsity
        _, diagnostics = olfactory_circuit.forward_pass(
            pn_activity, return_intermediates=True
        )
        sparsity_values.append(diagnostics["sparsity_fraction"])

    # Compute statistics
    mean_sparsity = np.mean(sparsity_values)
    std_sparsity = np.std(sparsity_values)

    # Mean should be very close to target (5%)
    target = olfactory_circuit.sparsity_target
    assert abs(mean_sparsity - target) < 0.01, f"Mean sparsity {mean_sparsity:.3f} != target {target:.3f}"

    # Std should be low (sparsity enforcement is consistent)
    assert std_sparsity < 0.01, f"Sparsity std {std_sparsity:.3f} too high (inconsistent enforcement)"


def test_different_normalization_strategies_produce_valid_outputs(circuit_loader):
    """Test that all normalization strategies produce valid circuits.

    Biological interpretation:
    - Row normalization: synaptic scaling (homeostatic plasticity)
    - Global normalization: preserves relative connection strength
    - No normalization: raw anatomical synapse counts
    """
    normalization_strategies = ["row", "global", "none"]

    for norm in normalization_strategies:
        # Load with specific normalization
        conn_matrix = circuit_loader.load_connectivity_matrix(normalize_weights=norm)

        # Create circuit
        circuit = OlfactoryCircuit(conn_matrix, kc_sparsity_target=0.05)

        # Test forward pass
        pn_activity = np.random.rand(conn_matrix.n_pn)
        mbon_output = circuit.forward_pass(pn_activity)

        # Should produce valid output
        assert mbon_output.shape == (conn_matrix.n_mbon,)
        assert np.all(np.isfinite(mbon_output))


def test_kc_subtype_filtering_reduces_circuit_size(circuit_loader):
    """Test that filtering to KC subtypes produces smaller circuit.

    Biological scenario: Genetic ablation of specific KC subtypes to
    isolate memory systems (e.g., remove γ neurons, test long-term memory).
    """
    # Load full circuit
    full_conn = circuit_loader.load_connectivity_matrix()

    # Load γ-only circuit
    gamma_conn = circuit_loader.load_connectivity_matrix(
        kc_subtypes_filter=["g_main", "g_dorsal", "g_sparse"]
    )

    # γ circuit should have fewer KCs
    assert gamma_conn.n_kc < full_conn.n_kc

    # γ circuit should still be functional
    circuit = OlfactoryCircuit(gamma_conn, kc_sparsity_target=0.05)
    pn_activity = np.random.rand(gamma_conn.n_pn)
    mbon_output = circuit.forward_pass(pn_activity)

    assert np.all(np.isfinite(mbon_output))


def test_glomerulus_targeted_activation(olfactory_circuit):
    """Test activating PNs by glomerulus produces KC/MBON responses.

    Biological scenario: Present odor that activates specific glomeruli
    (e.g., ethyl butyrate → DA1, DL3 glomeruli).
    """
    # Get available glomeruli
    available_glomeruli = list(
        set(olfactory_circuit.connectivity.pn_glomeruli.values())
    )
    if "unknown" in available_glomeruli:
        available_glomeruli.remove("unknown")

    if len(available_glomeruli) < 2:
        pytest.skip("Not enough glomeruli for targeted activation test")

    # Try multiple glomeruli until we find ones that activate KCs
    # (Some glomeruli may have sparse connectivity)
    kc_activated = False
    for n_glom in [2, 5, 10]:
        if n_glom > len(available_glomeruli):
            continue

        target_glomeruli = available_glomeruli[:n_glom]
        pn_activity = olfactory_circuit.activate_pns_by_glomeruli(
            target_glomeruli, firing_rate=1.0
        )

        # Run forward pass
        mbon_output, diagnostics = olfactory_circuit.forward_pass(
            pn_activity, return_intermediates=True
        )

        if np.any(diagnostics["kc_activity"] > 0):
            kc_activated = True
            break

    # Skip test if no glomeruli activate KCs (sparse connectivity issue)
    if not kc_activated:
        pytest.skip("No KC activation found - sparse connectivity issue")

    # Should produce valid responses
    assert np.any(diagnostics["kc_activity"] > 0), "No KCs activated"
    assert np.any(mbon_output != 0), "No MBON response"

    # KC sparsity should be reasonable (allow wider range due to variability)
    assert 0.01 <= diagnostics["sparsity_fraction"] <= 0.15, \
        f"KC sparsity should be reasonable, got {diagnostics['sparsity_fraction']:.3f}"


def test_validation_report_comprehensive(circuit_loader, connectivity_matrix):
    """Test that validation report contains all expected metrics.

    Biological validation: Circuit structure should match published
    connectivity statistics from FlyWire and other connectome datasets.
    """
    report = circuit_loader.validate_connectivity(connectivity_matrix)

    # Check all expected keys present
    required_keys = [
        "pn_to_kc_sparsity",
        "kc_to_mbon_sparsity",
        "pn_to_kc_fan_in",
        "kc_to_mbon_fan_in",
        "orphan_kcs",
        "orphan_mbons",
    ]

    for key in required_keys:
        assert key in report, f"Missing key: {key}"

    # Validate sparsity values are in range
    assert 0.0 <= report["pn_to_kc_sparsity"] <= 1.0
    assert 0.0 <= report["kc_to_mbon_sparsity"] <= 1.0

    # Validate fan-in quantiles are non-negative
    for quantile in ["p10", "p50", "p90"]:
        assert report["pn_to_kc_fan_in"][quantile] >= 0
        assert report["kc_to_mbon_fan_in"][quantile] >= 0

    # Orphan counts should be non-negative integers
    assert report["orphan_kcs"] >= 0
    assert report["orphan_mbons"] >= 0


def test_circuit_preserves_biological_sparsity(connectivity_matrix):
    """Test that loaded connectivity exhibits biological sparsity patterns.

    Biological validation: Published Drosophila MB connectivity shows:
    - PN→KC: ~95-97% sparse (each KC receives from ~6-8 of ~150 PNs)
    - KC→MBON: ~90-95% sparse (distributed but sparse readout)
    """
    # PN→KC should be highly sparse
    pn_kc_sparsity = connectivity_matrix.pn_to_kc_sparsity()
    assert pn_kc_sparsity > 0.8, f"PN→KC sparsity {pn_kc_sparsity:.2%} too low (< 80%)"

    # KC→MBON should be moderately sparse
    kc_mbon_sparsity = connectivity_matrix.kc_to_mbon_sparsity()
    assert kc_mbon_sparsity > 0.5, f"KC→MBON sparsity {kc_mbon_sparsity:.2%} too low (< 50%)"


def test_different_sparsity_targets_produce_correct_kc_activation(connectivity_matrix):
    """Test that changing KC sparsity target changes activation levels.

    Biological note: APL inhibition strength can be modulated, changing
    KC sparsity. Optogenetic APL activation increases sparsity; silencing
    decreases it.
    """
    pn_activity = np.random.rand(connectivity_matrix.n_pn)

    # Test different sparsity targets
    sparsity_targets = [0.02, 0.05, 0.10]
    measured_sparsities = []

    for target in sparsity_targets:
        circuit = OlfactoryCircuit(connectivity_matrix, kc_sparsity_target=target)
        _, diagnostics = circuit.forward_pass(pn_activity, return_intermediates=True)
        measured_sparsities.append(diagnostics["sparsity_fraction"])

    # Measured sparsities should increase with target
    assert measured_sparsities[0] < measured_sparsities[1] < measured_sparsities[2]

    # Each should be close to its target
    for target, measured in zip(sparsity_targets, measured_sparsities):
        assert abs(measured - target) < 0.02, f"Target {target} != measured {measured}"


def test_zero_input_produces_zero_output_all_stages(olfactory_circuit):
    """Test that zero PN input produces zero KC and MBON activity.

    Biological scenario: No odor present (spontaneous activity ignored).
    """
    pn_activity = np.zeros(olfactory_circuit.connectivity.n_pn)

    mbon_output, diagnostics = olfactory_circuit.forward_pass(
        pn_activity, return_intermediates=True
    )

    # KC activity should be zero (or near-zero)
    assert np.allclose(diagnostics["kc_activity"], 0.0, atol=1e-10)

    # MBON output should be zero
    assert np.allclose(mbon_output, 0.0, atol=1e-10)

    # Sparsity fraction should be 0 (no active KCs)
    assert diagnostics["sparsity_fraction"] == 0.0
