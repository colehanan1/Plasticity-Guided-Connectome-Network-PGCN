"""Unit and integration tests for SelectiveVetoGate.

Tests cover:
1. Veto gate initialization and parameter validation
2. Critical pathway identification (gradient, weight, activity methods)
3. Chemical similarity computation via DoOR
4. KC overlap similarity (fallback)
5. Plasticity gating and protection application
6. Edge cases and error handling
7. Integration with DopamineModulatedPlasticity

Author: PGCN Project
Date: 2025-11-18
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from pgcn.models.veto_gate import SelectiveVetoGate


# ===== FIXTURES =====


@pytest.fixture
def circuit():
    """Load test circuit."""
    loader = CircuitLoader(cache_dir="data/cache")
    connectivity = loader.load_connectivity_matrix(
        normalize_weights='row',
        include_dan=False,
        include_extended=False,
    )
    return OlfactoryCircuit(connectivity=connectivity, kc_sparsity_target=0.05)


@pytest.fixture
def plasticity(circuit):
    """Create plasticity model."""
    kc_to_mbon_dense = circuit.connectivity.kc_to_mbon.toarray()
    return DopamineModulatedPlasticity(
        kc_to_mbon_initial=kc_to_mbon_dense,
        learning_rate=0.001,
        plasticity_mode='three_factor',
    )


@pytest.fixture
def veto_gate(circuit):
    """Create veto gate with default parameters."""
    return SelectiveVetoGate(
        circuit=circuit,
        protection_threshold=0.3,
        gate_strength=0.9,
        similarity_metric='kc_overlap',  # Use fallback for tests
        min_gating_factor=0.1,
    )


# ===== INITIALIZATION TESTS =====


def test_veto_gate_initialization(circuit):
    """Test veto gate initialization with valid parameters."""
    veto = SelectiveVetoGate(
        circuit=circuit,
        protection_threshold=0.5,
        gate_strength=0.8,
        similarity_metric='kc_overlap',
    )

    assert veto.protection_threshold == 0.5
    assert veto.gate_strength == 0.8
    assert veto.similarity_metric == 'kc_overlap'
    assert veto.min_gating_factor == 0.1  # Default
    assert veto.protection_mask is None
    assert veto.protected_odor is None
    assert len(veto.veto_history) == 0


def test_veto_gate_invalid_threshold(circuit):
    """Test that invalid protection threshold raises ValueError."""
    with pytest.raises(ValueError, match="protection_threshold must be in"):
        SelectiveVetoGate(circuit=circuit, protection_threshold=1.5)

    with pytest.raises(ValueError, match="protection_threshold must be in"):
        SelectiveVetoGate(circuit=circuit, protection_threshold=-0.1)


def test_veto_gate_invalid_gate_strength(circuit):
    """Test that invalid gate strength raises ValueError."""
    with pytest.raises(ValueError, match="gate_strength must be in"):
        SelectiveVetoGate(circuit=circuit, gate_strength=1.5)


def test_veto_gate_invalid_similarity_metric(circuit):
    """Test that invalid similarity metric raises ValueError."""
    with pytest.raises(ValueError, match="similarity_metric must be one of"):
        SelectiveVetoGate(circuit=circuit, similarity_metric='invalid_metric')


# ===== PATHWAY IDENTIFICATION TESTS =====


def test_identify_critical_pathways_gradient_magnitude(circuit, veto_gate):
    """Test critical pathway identification using gradient magnitude."""
    # Create mock weight change history
    n_mbon, n_kc = veto_gate.kc_mbon_shape
    weight_changes = [
        np.random.randn(n_mbon, n_kc) * 0.001 for _ in range(10)
    ]

    task_data = {'weight_changes': weight_changes}

    protection_mask = veto_gate.identify_critical_pathways(
        task_data, method='gradient_magnitude'
    )

    # Check shape
    assert protection_mask.shape == veto_gate.kc_mbon_shape

    # Check dtype (should be bool)
    assert protection_mask.dtype == bool

    # Check percentage protected (should be ~30%)
    protected_fraction = protection_mask.sum() / protection_mask.size
    assert 0.25 < protected_fraction < 0.35, f"Expected ~30%, got {protected_fraction:.1%}"


def test_identify_critical_pathways_weight_magnitude(circuit, veto_gate):
    """Test critical pathway identification using weight magnitude."""
    n_mbon, n_kc = veto_gate.kc_mbon_shape

    # Create mock final weights with some large values
    final_weights = np.random.randn(n_mbon, n_kc) * 0.01
    final_weights[:10, :10] = 1.0  # Make some weights large

    task_data = {'final_weights': final_weights}

    protection_mask = veto_gate.identify_critical_pathways(
        task_data, method='weight_magnitude'
    )

    assert protection_mask.shape == veto_gate.kc_mbon_shape
    assert protection_mask[:10, :10].sum() > 0, "Large weights should be protected"


def test_identify_critical_pathways_activity_correlation(circuit, veto_gate):
    """Test critical pathway identification using activity correlation."""
    n_mbon, n_kc = veto_gate.kc_mbon_shape
    n_trials = 20

    # Create mock KC activity and MBON output
    kc_activity = np.random.rand(n_trials, n_kc)
    mbon_output = np.random.rand(n_trials, n_mbon)

    task_data = {
        'kc_activity': kc_activity,
        'mbon_output': mbon_output,
    }

    protection_mask = veto_gate.identify_critical_pathways(
        task_data, method='activity_correlation'
    )

    assert protection_mask.shape == veto_gate.kc_mbon_shape
    protected_fraction = protection_mask.sum() / protection_mask.size
    assert 0.20 < protected_fraction < 0.40  # Should protect some synapses


def test_identify_critical_pathways_missing_data(veto_gate):
    """Test that missing required data raises ValueError."""
    with pytest.raises(ValueError, match="requires 'weight_changes'"):
        veto_gate.identify_critical_pathways({}, method='gradient_magnitude')

    with pytest.raises(ValueError, match="requires 'final_weights'"):
        veto_gate.identify_critical_pathways({}, method='weight_magnitude')


def test_identify_critical_pathways_empty_history(veto_gate):
    """Test pathway identification with empty weight change history."""
    task_data = {'weight_changes': []}

    protection_mask = veto_gate.identify_critical_pathways(
        task_data, method='gradient_magnitude'
    )

    # Should return all-zero mask (protect nothing)
    assert protection_mask.sum() == 0


# ===== SIMILARITY COMPUTATION TESTS =====


def test_compute_veto_signal_kc_overlap(circuit, veto_gate):
    """Test veto signal computation using KC overlap."""
    # Use simple odor names (will be treated as glomeruli in fallback mode)
    veto_signal = veto_gate.compute_veto_signal('DA1', 'DA1')

    # Same odor → high similarity
    assert 0.8 < veto_signal <= 1.0, f"Expected high similarity, got {veto_signal:.3f}"

    # Different odors → low similarity
    veto_signal_diff = veto_gate.compute_veto_signal('DA1', 'DL3')
    assert veto_signal_diff < veto_signal, "Different odors should have lower similarity"


def test_compute_veto_signal_with_diagnostics(veto_gate):
    """Test veto signal computation with diagnostics return."""
    veto_signal, diagnostics = veto_gate.compute_veto_signal(
        'DA1', 'DL3', return_diagnostics=True
    )

    assert isinstance(veto_signal, float)
    assert 0.0 <= veto_signal <= 1.0

    # Check diagnostics keys
    assert 'similarity_raw' in diagnostics
    assert 'method_used' in diagnostics
    assert 'new_odor_normalized' in diagnostics
    assert 'protected_odor_normalized' in diagnostics


def test_odor_name_normalization(veto_gate):
    """Test odor name normalization."""
    # Test underscore to space
    normalized = veto_gate._normalize_odor_name('ethyl_butyrate')
    assert normalized == 'ethyl butyrate'

    # Test hexanol → 1-hexanol
    normalized = veto_gate._normalize_odor_name('hexanol')
    assert normalized == '1-hexanol'

    # Test already normalized
    normalized = veto_gate._normalize_odor_name('benzaldehyde')
    assert normalized == 'benzaldehyde'


# ===== PROTECTION APPLICATION TESTS =====


def test_apply_protection_basic(circuit, veto_gate):
    """Test basic protection application."""
    n_mbon, n_kc = veto_gate.kc_mbon_shape

    # Create mock weight update and protection mask
    delta_w = np.random.randn(n_mbon, n_kc) * 0.01
    protection_mask = np.zeros(veto_gate.kc_mbon_shape, dtype=bool)
    protection_mask[:10, :10] = True  # Protect top-left block

    # Apply protection with strong veto
    gated_delta_w = veto_gate.apply_protection(
        delta_w, protection_mask, veto_signal=0.9
    )

    # Check that protected region is reduced
    protected_region_orig = np.abs(delta_w[:10, :10]).mean()
    protected_region_gated = np.abs(gated_delta_w[:10, :10]).mean()
    assert protected_region_gated < protected_region_orig * 0.2, \
        "Protected region should be strongly suppressed"

    # Check that unprotected region is unchanged
    unprotected_region_orig = np.abs(delta_w[10:, 10:]).mean()
    unprotected_region_gated = np.abs(gated_delta_w[10:, 10:]).mean()
    np.testing.assert_allclose(
        unprotected_region_gated, unprotected_region_orig, rtol=1e-10
    )


def test_apply_protection_veto_strength_scaling(circuit, veto_gate):
    """Test that veto signal scales protection appropriately."""
    n_mbon, n_kc = veto_gate.kc_mbon_shape
    delta_w = np.random.randn(n_mbon, n_kc) * 0.01
    protection_mask = np.ones(veto_gate.kc_mbon_shape, dtype=bool)  # Protect all

    # No veto (veto_signal=0.0) → no suppression
    gated_0 = veto_gate.apply_protection(delta_w, protection_mask, veto_signal=0.0)
    np.testing.assert_allclose(gated_0, delta_w, rtol=1e-10)

    # Moderate veto (veto_signal=0.5) → moderate suppression
    gated_05 = veto_gate.apply_protection(delta_w, protection_mask, veto_signal=0.5)
    reduction_05 = np.linalg.norm(gated_05) / np.linalg.norm(delta_w)
    assert 0.45 < reduction_05 < 0.65, f"Expected ~55% retained, got {reduction_05:.1%}"

    # Strong veto (veto_signal=1.0) → strong suppression (but not complete)
    gated_1 = veto_gate.apply_protection(delta_w, protection_mask, veto_signal=1.0)
    reduction_1 = np.linalg.norm(gated_1) / np.linalg.norm(delta_w)
    assert 0.05 < reduction_1 < 0.15, \
        f"Expected ~10% retained (min_gating_factor), got {reduction_1:.1%}"


def test_apply_protection_min_gating_factor(circuit):
    """Test that min_gating_factor prevents complete plasticity shutdown."""
    veto = SelectiveVetoGate(
        circuit=circuit,
        min_gating_factor=0.2,  # 20% floor
        gate_strength=1.0,
    )

    n_mbon, n_kc = veto.kc_mbon_shape
    delta_w = np.ones((n_mbon, n_kc))  # Uniform updates
    protection_mask = np.ones(veto.kc_mbon_shape, dtype=bool)

    # Full veto (veto_signal=1.0)
    gated = veto.apply_protection(delta_w, protection_mask, veto_signal=1.0)

    # Should retain 20% (min_gating_factor)
    expected = delta_w * 0.2
    np.testing.assert_allclose(gated, expected, rtol=1e-10)


def test_apply_protection_without_mask_set(veto_gate):
    """Test that protection fails gracefully without protection mask."""
    n_mbon, n_kc = veto_gate.kc_mbon_shape
    delta_w = np.random.randn(n_mbon, n_kc) * 0.01

    with pytest.raises(ValueError, match="No protection mask provided"):
        veto_gate.apply_protection(delta_w)  # No mask provided


def test_apply_protection_shape_mismatch(veto_gate):
    """Test that shape mismatch raises ValueError."""
    delta_w_wrong = np.random.randn(10, 10)  # Wrong shape

    with pytest.raises(ValueError, match="shape.*does not match"):
        veto_gate.apply_protection(delta_w_wrong)


# ===== INTEGRATION TESTS =====


def test_integration_with_plasticity(circuit, plasticity):
    """Test end-to-end integration with plasticity model."""
    # Create veto gate
    veto = SelectiveVetoGate(
        circuit=circuit,
        protection_threshold=0.3,
        gate_strength=0.9,
        similarity_metric='kc_overlap',
    )

    # Enable veto protection
    plasticity.enable_veto_protection(veto, track_weight_changes=True)

    # Simulate training trials
    n_kc = circuit.connectivity.n_kc
    n_mbon = circuit.connectivity.n_mbon

    for trial in range(10):
        kc_activity = np.random.rand(n_kc)
        kc_activity[kc_activity < 0.95] = 0  # Sparsify
        mbon_activity = np.random.rand(n_mbon)
        dopamine = 0.5

        plasticity.update_weights(kc_activity, mbon_activity, dopamine, dt=1.0)

    # Check that weight changes were tracked
    assert len(plasticity.weight_change_history) == 10

    # Identify critical pathways
    task_data = plasticity.get_task_data_for_protection()
    protection_mask = veto.identify_critical_pathways(
        task_data, method='gradient_magnitude'
    )

    assert protection_mask.shape == (n_mbon, n_kc)
    assert protection_mask.sum() > 0, "Should identify some critical synapses"


# ===== HISTORY AND STATISTICS TESTS =====


def test_veto_history_tracking(veto_gate):
    """Test that veto history is tracked correctly."""
    # Compute several veto signals
    veto_gate.compute_veto_signal('DA1', 'DL3')
    veto_gate.compute_veto_signal('DA1', 'DA1')
    veto_gate.compute_veto_signal('DL3', 'DA1')

    assert len(veto_gate.veto_history) == 3

    # Check history structure
    for entry in veto_gate.veto_history:
        assert 'new_odor' in entry
        assert 'protected_odor' in entry
        assert 'veto_signal' in entry
        assert 'method' in entry


def test_get_veto_statistics(veto_gate):
    """Test veto statistics computation."""
    # Compute several veto signals with known values
    veto_gate.compute_veto_signal('DA1', 'DA1')  # High similarity
    veto_gate.compute_veto_signal('DA1', 'DL3')  # Low similarity

    stats = veto_gate.get_veto_statistics()

    assert 'mean_veto' in stats
    assert 'max_veto' in stats
    assert 'min_veto' in stats
    assert 'n_samples' in stats

    assert stats['n_samples'] == 2
    assert 0.0 <= stats['mean_veto'] <= 1.0
    assert 0.0 <= stats['max_veto'] <= 1.0
    assert 0.0 <= stats['min_veto'] <= 1.0


def test_reset_history(veto_gate):
    """Test history reset."""
    veto_gate.compute_veto_signal('DA1', 'DL3')
    assert len(veto_gate.veto_history) > 0

    veto_gate.reset_history()
    assert len(veto_gate.veto_history) == 0


def test_reset_protection(veto_gate):
    """Test protection reset."""
    # Set protection mask
    n_mbon, n_kc = veto_gate.kc_mbon_shape
    veto_gate.protection_mask = np.ones((n_mbon, n_kc), dtype=bool)
    veto_gate.protected_odor = 'benzaldehyde'

    assert veto_gate.protection_mask is not None
    assert veto_gate.protected_odor is not None

    veto_gate.reset_protection()

    assert veto_gate.protection_mask is None
    assert veto_gate.protected_odor is None


# ===== STRING REPRESENTATION TEST =====


def test_repr(veto_gate):
    """Test __repr__ method."""
    repr_str = repr(veto_gate)

    assert 'SelectiveVetoGate' in repr_str
    assert 'protection_threshold' in repr_str
    assert 'gate_strength' in repr_str
    assert 'similarity_metric' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
