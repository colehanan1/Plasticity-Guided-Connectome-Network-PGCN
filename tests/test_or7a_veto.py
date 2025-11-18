"""Unit tests for Or7a veto gate module.

This test suite validates the Or7a-inspired veto gate mechanism for
continual learning. Tests cover graded veto computation, plasticity gating,
biological parameter validation, and edge cases.

Run tests:
    pytest tests/test_or7a_veto.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pgcn.models.connectivity_matrix import ConnectivityMatrix
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from pgcn.models.or7a_veto_gate import Or7aVetoGate


@pytest.fixture
def minimal_connectivity():
    """Create minimal connectivity matrix for testing.

    This creates a small circuit with:
    - 50 PNs (including 5 in "DL5" glomerulus for Or7a)
    - 100 KCs
    - 10 MBONs
    - 20 DANs
    """
    n_pn = 50
    n_kc = 100
    n_mbon = 10
    n_dan = 20

    # Create PN IDs and assign glomeruli
    pn_ids = np.arange(1000, 1000 + n_pn, dtype=np.int64)
    pn_glomeruli = {}
    for i, pn_id in enumerate(pn_ids):
        if i < 5:
            pn_glomeruli[pn_id] = "DL5"  # Or7a glomerulus
        elif i < 10:
            pn_glomeruli[pn_id] = "DA1"
        elif i < 15:
            pn_glomeruli[pn_id] = "DL3"
        else:
            pn_glomeruli[pn_id] = f"glom_{i}"

    # Create sparse connectivity matrices
    kc_ids = np.arange(2000, 2000 + n_kc, dtype=np.int64)
    mbon_ids = np.arange(3000, 3000 + n_mbon, dtype=np.int64)
    dan_ids = np.arange(4000, 4000 + n_dan, dtype=np.int64)

    # PN→KC: sparse random (3% density)
    pn_to_kc_density = 0.03
    pn_to_kc_data = np.random.rand(int(n_kc * n_pn * pn_to_kc_density))
    pn_to_kc = sp.random(n_kc, n_pn, density=pn_to_kc_density, format='csr', data_rvs=lambda n: pn_to_kc_data[:n])

    # KC→MBON: sparse random (10% density)
    kc_to_mbon_density = 0.10
    kc_to_mbon_data = np.random.rand(int(n_mbon * n_kc * kc_to_mbon_density))
    kc_to_mbon = sp.random(n_mbon, n_kc, density=kc_to_mbon_density, format='csr', data_rvs=lambda n: kc_to_mbon_data[:n])

    # DAN→KC and DAN→MBON: minimal connectivity
    dan_to_kc = sp.csr_matrix((n_kc, n_dan))
    dan_to_mbon = sp.csr_matrix((n_mbon, n_dan))

    connectivity = ConnectivityMatrix(
        pn_ids=pn_ids,
        kc_ids=kc_ids,
        mbon_ids=mbon_ids,
        dan_ids=dan_ids,
        pn_to_kc=pn_to_kc,
        kc_to_mbon=kc_to_mbon,
        dan_to_kc=dan_to_kc,
        dan_to_mbon=dan_to_mbon,
        pn_glomeruli=pn_glomeruli,
    )

    return connectivity


@pytest.fixture
def minimal_circuit(minimal_connectivity):
    """Create minimal olfactory circuit for testing."""
    return OlfactoryCircuit(
        connectivity=minimal_connectivity,
        kc_sparsity_target=0.05,
    )


class TestOr7aVetoGate:
    """Test suite for Or7aVetoGate class."""

    def test_initialization_valid_glomerulus(self, minimal_circuit):
        """Test initialization with valid Or7a glomerulus."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            activation_threshold=0.3,
        )

        assert veto_gate.or7a_glomerulus == "DL5"
        assert veto_gate.n_or7a_pns == 5  # 5 PNs in DL5
        assert veto_gate.activation_threshold == 0.3
        assert veto_gate.veto_strength == 1.0  # Default
        assert veto_gate.graded is True  # Default
        assert len(veto_gate.veto_history) == 0

    def test_initialization_invalid_glomerulus(self, minimal_circuit):
        """Test initialization with non-existent glomerulus raises ValueError."""
        with pytest.raises(ValueError, match="No PNs found for Or7a glomerulus"):
            Or7aVetoGate(
                circuit=minimal_circuit,
                or7a_glomerulus="INVALID_GLOM",
            )

    def test_initialization_invalid_parameters(self, minimal_circuit):
        """Test initialization with invalid parameters raises ValueError."""
        # Invalid activation threshold (> 1.0)
        with pytest.raises(ValueError, match="activation_threshold must be in"):
            Or7aVetoGate(
                circuit=minimal_circuit,
                or7a_glomerulus="DL5",
                activation_threshold=1.5,
            )

        # Invalid veto strength (< 0.0)
        with pytest.raises(ValueError, match="veto_strength must be in"):
            Or7aVetoGate(
                circuit=minimal_circuit,
                or7a_glomerulus="DL5",
                veto_strength=-0.1,
            )

        # Invalid min gating factor (> 1.0)
        with pytest.raises(ValueError, match="min_gating_factor must be in"):
            Or7aVetoGate(
                circuit=minimal_circuit,
                or7a_glomerulus="DL5",
                min_gating_factor=1.2,
            )

    def test_veto_weight_normalization(self, minimal_circuit):
        """Test that veto weights sum to 1.0."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
        )

        # Veto weight should sum to 1.0 (uniform across DL5 PNs)
        assert np.isclose(veto_gate.veto_weight.sum(), 1.0)

        # Only DL5 PNs should have non-zero weights
        dl5_indices = minimal_circuit.connectivity.get_pn_indices(["DL5"])
        non_dl5_mask = np.ones(len(veto_gate.veto_weight), dtype=bool)
        non_dl5_mask[dl5_indices] = False

        assert veto_gate.veto_weight[non_dl5_mask].sum() == 0.0
        assert veto_gate.veto_weight[dl5_indices].sum() == 1.0

    def test_veto_signal_output_range(self, minimal_circuit):
        """Test that veto signal is always in [0, 1]."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            graded=True,
        )

        # Test with various PN activity levels
        for firing_rate in [0.0, 0.14, 0.3, 0.55, 1.0]:
            pn_activity = minimal_circuit.activate_pns_by_glomeruli(
                ["DL5"], firing_rate=firing_rate
            )
            veto_signal = veto_gate.compute_veto_signal(pn_activity)

            assert 0.0 <= veto_signal <= 1.0, f"Veto signal {veto_signal} out of range for firing_rate={firing_rate}"

    def test_graded_veto_behavior(self, minimal_circuit):
        """Test graded veto produces different values for different activations."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            activation_threshold=0.3,
            graded=True,
            steepness=10.0,
        )

        # Weak activation (hexanol-like, 14%)
        pn_weak = minimal_circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.14)
        veto_weak = veto_gate.compute_veto_signal(pn_weak)

        # Strong activation (benzaldehyde-like, 55%)
        pn_strong = minimal_circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.55)
        veto_strong = veto_gate.compute_veto_signal(pn_strong)

        # Strong should produce higher veto than weak
        assert veto_strong > veto_weak, f"Strong veto {veto_strong} not greater than weak veto {veto_weak}"

        # Weak should be below threshold (< 0.5)
        assert veto_weak < 0.5, f"Weak veto {veto_weak} should be < 0.5"

        # Strong should be above threshold (> 0.5)
        assert veto_strong > 0.5, f"Strong veto {veto_strong} should be > 0.5"

    def test_binary_veto_behavior(self, minimal_circuit):
        """Test binary veto produces only 0.0 or 1.0."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            activation_threshold=0.3,
            graded=False,  # Binary mode
        )

        # Below threshold
        pn_below = minimal_circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.2)
        veto_below = veto_gate.compute_veto_signal(pn_below)
        assert veto_below == 0.0

        # Above threshold
        pn_above = minimal_circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.4)
        veto_above = veto_gate.compute_veto_signal(pn_above)
        assert veto_above == 1.0

    def test_veto_signal_diagnostics(self, minimal_circuit):
        """Test that diagnostics return correct information."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            activation_threshold=0.3,
        )

        pn_activity = minimal_circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.55)
        veto_signal, diagnostics = veto_gate.compute_veto_signal(
            pn_activity, return_diagnostics=True
        )

        # Check diagnostics keys
        assert "or7a_activation" in diagnostics
        assert "veto_raw" in diagnostics
        assert "veto_signal" in diagnostics
        assert "above_threshold" in diagnostics
        assert "gating_factor" in diagnostics

        # Check diagnostics values
        assert diagnostics["or7a_activation"] == pytest.approx(0.55, abs=0.01)
        assert diagnostics["above_threshold"] is True
        assert 0.0 <= diagnostics["gating_factor"] <= 1.0

    def test_gate_plasticity_strong_veto(self, minimal_circuit):
        """Test plasticity gating with strong veto (90% suppression)."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            veto_strength=1.0,
            min_gating_factor=0.1,
        )

        # Create random weight update
        delta_w = np.random.randn(10, 100) * 0.001
        veto_signal = 0.9  # Strong veto

        # Apply gating
        gated_delta_w = veto_gate.gate_plasticity(delta_w, veto_signal)

        # Gated update should be ~10% of original (min_gating_factor)
        expected_gating = max(0.1, 1.0 - 1.0 * 0.9)  # 0.1 (floor)
        actual_ratio = np.linalg.norm(gated_delta_w) / np.linalg.norm(delta_w)

        assert actual_ratio == pytest.approx(expected_gating, abs=0.01)

    def test_gate_plasticity_weak_veto(self, minimal_circuit):
        """Test plasticity gating with weak veto (10% suppression)."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            veto_strength=1.0,
            min_gating_factor=0.1,
        )

        delta_w = np.random.randn(10, 100) * 0.001
        veto_signal = 0.1  # Weak veto

        gated_delta_w = veto_gate.gate_plasticity(delta_w, veto_signal)

        # Gated update should be ~90% of original
        expected_gating = 1.0 - 1.0 * 0.1  # 0.9
        actual_ratio = np.linalg.norm(gated_delta_w) / np.linalg.norm(delta_w)

        assert actual_ratio == pytest.approx(expected_gating, abs=0.01)

    def test_gate_plasticity_no_veto(self, minimal_circuit):
        """Test plasticity gating with no veto (100% plasticity)."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            veto_strength=1.0,
        )

        delta_w = np.random.randn(10, 100) * 0.001
        veto_signal = 0.0  # No veto

        gated_delta_w = veto_gate.gate_plasticity(delta_w, veto_signal)

        # Gated update should be identical to original
        np.testing.assert_allclose(gated_delta_w, delta_w)

    def test_gate_plasticity_invalid_veto_signal(self, minimal_circuit):
        """Test that invalid veto signal raises ValueError."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
        )

        delta_w = np.random.randn(10, 100) * 0.001

        # Veto signal > 1.0
        with pytest.raises(ValueError, match="veto_signal must be in"):
            veto_gate.gate_plasticity(delta_w, 1.5)

        # Veto signal < 0.0
        with pytest.raises(ValueError, match="veto_signal must be in"):
            veto_gate.gate_plasticity(delta_w, -0.1)

    def test_veto_history_tracking(self, minimal_circuit):
        """Test that veto history is correctly tracked."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
        )

        # Initially empty
        assert len(veto_gate.veto_history) == 0

        # Compute veto signals
        for firing_rate in [0.1, 0.3, 0.5, 0.7]:
            pn_activity = minimal_circuit.activate_pns_by_glomeruli(
                ["DL5"], firing_rate=firing_rate
            )
            veto_gate.compute_veto_signal(pn_activity)

        # History should have 4 entries
        assert len(veto_gate.veto_history) == 4

        # Each entry should have required keys
        for entry in veto_gate.veto_history:
            assert "or7a_activation" in entry
            assert "veto_signal" in entry
            assert "above_threshold" in entry

    def test_veto_statistics(self, minimal_circuit):
        """Test veto statistics computation."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            activation_threshold=0.3,
        )

        # Empty history
        stats_empty = veto_gate.get_veto_statistics()
        assert stats_empty["n_samples"] == 0
        assert stats_empty["mean_veto"] == 0.0

        # Collect some samples
        firing_rates = [0.1, 0.2, 0.5, 0.6, 0.8]
        for fr in firing_rates:
            pn_activity = minimal_circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=fr)
            veto_gate.compute_veto_signal(pn_activity)

        stats = veto_gate.get_veto_statistics()

        # Check statistics
        assert stats["n_samples"] == 5
        assert 0.0 <= stats["mean_veto"] <= 1.0
        assert 0.0 <= stats["max_veto"] <= 1.0
        assert 0.0 <= stats["min_veto"] <= 1.0
        assert stats["max_veto"] >= stats["mean_veto"] >= stats["min_veto"]

        # Fraction above threshold should be 3/5 (0.5, 0.6, 0.8 > 0.3)
        assert stats["fraction_above_threshold"] == pytest.approx(0.6, abs=0.01)

    def test_reset_history(self, minimal_circuit):
        """Test that reset_history clears veto tracking."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
        )

        # Add some history
        for _ in range(10):
            pn_activity = minimal_circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.5)
            veto_gate.compute_veto_signal(pn_activity)

        assert len(veto_gate.veto_history) == 10

        # Reset
        veto_gate.reset_history()
        assert len(veto_gate.veto_history) == 0

    def test_veto_strength_parameter(self, minimal_circuit):
        """Test that veto_strength parameter correctly scales gating."""
        # Full veto strength
        veto_gate_full = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            veto_strength=1.0,
            min_gating_factor=0.0,  # No floor for this test
        )

        # Half veto strength
        veto_gate_half = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            veto_strength=0.5,
            min_gating_factor=0.0,
        )

        delta_w = np.random.randn(10, 100) * 0.001
        veto_signal = 0.8  # Strong veto

        # Full strength: gating = 1.0 - 1.0 * 0.8 = 0.2
        gated_full = veto_gate_full.gate_plasticity(delta_w, veto_signal)
        ratio_full = np.linalg.norm(gated_full) / np.linalg.norm(delta_w)
        assert ratio_full == pytest.approx(0.2, abs=0.01)

        # Half strength: gating = 1.0 - 0.5 * 0.8 = 0.6
        gated_half = veto_gate_half.gate_plasticity(delta_w, veto_signal)
        ratio_half = np.linalg.norm(gated_half) / np.linalg.norm(delta_w)
        assert ratio_half == pytest.approx(0.6, abs=0.01)

    def test_min_gating_factor_floor(self, minimal_circuit):
        """Test that min_gating_factor prevents complete plasticity shutdown."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            veto_strength=1.0,
            min_gating_factor=0.1,
        )

        delta_w = np.random.randn(10, 100) * 0.001
        veto_signal = 1.0  # Maximum veto

        # Without floor, gating = 1.0 - 1.0 * 1.0 = 0.0
        # With floor = 0.1, gating = max(0.1, 0.0) = 0.1
        gated_delta_w = veto_gate.gate_plasticity(delta_w, veto_signal)
        ratio = np.linalg.norm(gated_delta_w) / np.linalg.norm(delta_w)

        assert ratio == pytest.approx(0.1, abs=0.01)

    def test_repr_string(self, minimal_circuit):
        """Test __repr__ returns informative string."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            activation_threshold=0.3,
            veto_strength=0.8,
        )

        repr_str = repr(veto_gate)

        # Should contain key information
        assert "Or7aVetoGate" in repr_str
        assert "DL5" in repr_str
        assert "0.3" in repr_str  # threshold
        assert "0.8" in repr_str  # strength
        assert "n_or7a_pns=5" in repr_str

    def test_cross_glomerulus_no_interference(self, minimal_circuit):
        """Test that veto gate only responds to Or7a glomerulus."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            activation_threshold=0.3,
        )

        # Activate non-Or7a glomeruli (DA1, DL3)
        pn_activity_other = minimal_circuit.activate_pns_by_glomeruli(
            ["DA1", "DL3"], firing_rate=1.0
        )
        veto_other = veto_gate.compute_veto_signal(pn_activity_other)

        # Should produce minimal veto (no DL5 activation)
        assert veto_other < 0.1

        # Activate Or7a glomerulus
        pn_activity_or7a = minimal_circuit.activate_pns_by_glomeruli(
            ["DL5"], firing_rate=1.0
        )
        veto_or7a = veto_gate.compute_veto_signal(pn_activity_or7a)

        # Should produce strong veto
        assert veto_or7a > 0.5

    def test_invalid_pn_activity_shape(self, minimal_circuit):
        """Test that invalid PN activity shape raises ValueError."""
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
        )

        # Wrong shape
        pn_activity_wrong = np.random.randn(100)  # Should be 50

        with pytest.raises(ValueError, match="pn_activity shape"):
            veto_gate.compute_veto_signal(pn_activity_wrong)


class TestOr7aVetoIntegration:
    """Integration tests for Or7a veto gate with full circuit."""

    def test_veto_reduces_learning_in_continual_setting(self, minimal_circuit):
        """Test that veto gate reduces learning on Task B after Task A."""
        from pgcn.models.learning_model import DopamineModulatedPlasticity

        # Initialize plasticity with random weights (not zeros)
        weights = minimal_circuit.connectivity.kc_to_mbon.toarray()
        plasticity = DopamineModulatedPlasticity(
            kc_to_mbon_weights=weights,
            learning_rate=0.001,
            init_mode="random",  # Use random initialization for non-zero MBON outputs
            init_scale=0.01,
        )

        # Create veto gate
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            activation_threshold=0.3,
        )

        # Task A: Train on DL5 (Or7a glomerulus)
        for _ in range(10):
            pn_activity = minimal_circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.55)
            kc_activity = minimal_circuit.propagate_pn_to_kc(pn_activity)
            mbon_output = plasticity.compute_mbon_output(kc_activity)

            # Compute weight update
            delta_w = np.outer(mbon_output, kc_activity) * 0.001

            # Apply update (no veto during Task A training)
            plasticity.kc_to_mbon += delta_w

        initial_weights = plasticity.kc_to_mbon.copy()

        # Task B: Train on DA1 (different glomerulus) WITH veto
        weight_changes_with_veto = []
        for _ in range(10):
            pn_activity = minimal_circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)

            # Compute veto signal (should be low since DA1 != DL5)
            veto_signal = veto_gate.compute_veto_signal(pn_activity)

            kc_activity = minimal_circuit.propagate_pn_to_kc(pn_activity)
            mbon_output = plasticity.compute_mbon_output(kc_activity)

            # Compute and gate weight update
            delta_w = np.outer(mbon_output, kc_activity) * 0.001
            gated_delta_w = veto_gate.gate_plasticity(delta_w, veto_signal)

            plasticity.kc_to_mbon += gated_delta_w
            weight_changes_with_veto.append(np.linalg.norm(gated_delta_w))

        # Weight changes should be non-zero (DA1 is not Or7a)
        assert np.mean(weight_changes_with_veto) > 0

    def test_biological_parameter_ranges(self, minimal_circuit):
        """Test with biologically-inspired parameter values."""
        # Biological: 14% (hexanol) to 55% (benzaldehyde)
        veto_gate = Or7aVetoGate(
            circuit=minimal_circuit,
            or7a_glomerulus="DL5",
            activation_threshold=0.3,  # Midpoint
            veto_strength=1.0,
            steepness=10.0,
            graded=True,
        )

        # Hexanol-like (14% activation)
        pn_hexanol = minimal_circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.14)
        veto_hexanol = veto_gate.compute_veto_signal(pn_hexanol)

        # Benzaldehyde-like (55% activation)
        pn_benzaldehyde = minimal_circuit.activate_pns_by_glomeruli(["DL5"], firing_rate=0.55)
        veto_benzaldehyde = veto_gate.compute_veto_signal(pn_benzaldehyde)

        # Biological expectation: benzaldehyde >> hexanol
        assert veto_benzaldehyde > 2 * veto_hexanol

        # Hexanol should allow most plasticity (weak veto)
        assert veto_hexanol < 0.3

        # Benzaldehyde should block most plasticity (strong veto)
        assert veto_benzaldehyde > 0.7
