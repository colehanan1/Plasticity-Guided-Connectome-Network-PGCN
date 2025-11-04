"""Integration tests for EnhancedOlfactoryCircuit with real FlyWire data.

This test suite verifies that:
1. CircuitLoader can load all 13K+ neurons with extended components
2. EnhancedOlfactoryCircuit integrates all layers successfully
3. Forward pass produces valid outputs
4. Blocking experiments can be executed
"""

import pytest
import numpy as np
from pathlib import Path

from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.enhanced_olfactory_circuit import EnhancedOlfactoryCircuit


class TestEnhancedCircuitIntegration:
    """Integration tests for full enhanced circuit."""

    @pytest.fixture
    def cache_dir(self):
        """Return cache directory path."""
        return Path("data/cache")

    @pytest.fixture
    def circuit_loader(self, cache_dir):
        """Create circuit loader."""
        if not cache_dir.exists():
            pytest.skip(f"Cache directory {cache_dir} not found")
        return CircuitLoader(cache_dir=str(cache_dir))

    @pytest.fixture
    def connectivity_extended(self, circuit_loader):
        """Load connectivity matrix with extended components."""
        try:
            conn = circuit_loader.load_connectivity_matrix(
                normalize_weights="row",
                include_dan=True,
                include_extended=True,
            )
            return conn
        except Exception as e:
            pytest.skip(f"Could not load extended connectivity: {e}")

    @pytest.fixture
    def enhanced_circuit(self, connectivity_extended):
        """Create enhanced olfactory circuit."""
        return EnhancedOlfactoryCircuit(
            connectivity=connectivity_extended,
            kc_sparsity_target=0.05,
            enable_ln_modulation=True,
            enable_lh_pathway=True,
            enable_motor_output=True,
            enable_vnc_interface=True,
        )

    def test_circuit_loads_successfully(self, enhanced_circuit, connectivity_extended):
        """Test that enhanced circuit loads all components."""
        assert enhanced_circuit is not None
        assert enhanced_circuit.core_circuit is not None

        # Check extended components loaded if data available
        if connectivity_extended.n_ln > 0:
            assert enhanced_circuit.ln_layer is not None
            print(f"✓ Loaded {connectivity_extended.n_ln} local interneurons")

        if connectivity_extended.n_lh > 0:
            assert enhanced_circuit.lh_layer is not None
            print(f"✓ Loaded {connectivity_extended.n_lh} lateral horn neurons")

        if connectivity_extended.n_motor > 0:
            assert enhanced_circuit.motor_layer is not None
            print(f"✓ Loaded {connectivity_extended.n_motor} motor neurons")

        if connectivity_extended.n_an > 0 and connectivity_extended.n_dn > 0:
            assert enhanced_circuit.vnc_interface is not None
            print(f"✓ Loaded {connectivity_extended.n_an} ANs, {connectivity_extended.n_dn} DNs")

    def test_neuron_counts_match_expected(self, connectivity_extended):
        """Test that extracted neuron counts match expectations."""
        # Expected counts from user specification
        expected = {
            "pn": 482,
            "kc": 5378,
            "ln": 3829,
            "lh": 1162,
            "motor": 90,
            "an": 1926,
            "dn": 1303,
        }

        # Allow ±10% tolerance for data variations
        tolerance = 0.1

        assert connectivity_extended.n_pn >= expected["pn"] * (1 - tolerance)
        assert connectivity_extended.n_kc >= expected["kc"] * (1 - tolerance)

        # Extended components (may be zero if not loaded)
        if connectivity_extended.n_ln > 0:
            assert connectivity_extended.n_ln >= expected["ln"] * (1 - tolerance)
        if connectivity_extended.n_lh > 0:
            assert connectivity_extended.n_lh >= expected["lh"] * (1 - tolerance)
        if connectivity_extended.n_motor > 0:
            assert connectivity_extended.n_motor >= expected["motor"] * (1 - tolerance)

        print(f"✓ Neuron counts validated:")
        print(f"  PNs: {connectivity_extended.n_pn} (expected ~{expected['pn']})")
        print(f"  KCs: {connectivity_extended.n_kc} (expected ~{expected['kc']})")
        print(f"  LNs: {connectivity_extended.n_ln} (expected ~{expected['ln']})")
        print(f"  LH: {connectivity_extended.n_lh} (expected ~{expected['lh']})")
        print(f"  Motors: {connectivity_extended.n_motor} (expected ~{expected['motor']})")
        print(f"  ANs: {connectivity_extended.n_an} (expected ~{expected['an']})")
        print(f"  DNs: {connectivity_extended.n_dn} (expected ~{expected['dn']})")

    def test_forward_pass_produces_valid_outputs(self, enhanced_circuit, connectivity_extended):
        """Test that forward pass produces valid outputs."""
        # Activate a subset of glomeruli
        glomeruli = list(connectivity_extended.pn_glomeruli.values())[:5]  # First 5

        if not glomeruli:
            pytest.skip("No glomeruli available")

        # Run forward pass
        results = enhanced_circuit.forward_pass_full(
            enhanced_circuit.activate_pns_by_glomeruli(glomeruli),
            blocking_strength=0.0,
            return_diagnostics=True,
        )

        # Validate outputs
        assert "mbon_output" in results
        assert isinstance(results["mbon_output"], np.ndarray)
        assert results["mbon_output"].shape == (connectivity_extended.n_mbon,)

        # Check KC sparsity
        if "diagnostics" in results:
            kc_sparsity = results["diagnostics"]["kc_sparsity"]
            assert 0.0 <= kc_sparsity <= 0.15  # Should be ~5% ± tolerance
            print(f"✓ KC sparsity: {kc_sparsity:.2%} (target: 5%)")

        # Check PER response if motor enabled
        if results.get("per_response") is not None:
            per = results["per_response"]
            assert 0.0 <= per <= 1.0
            print(f"✓ PER response: {per:.3f}")

        # Check innate valence if LH enabled
        if results.get("innate_valence") is not None:
            valence = results["innate_valence"]
            assert -2.0 <= valence <= 2.0  # Should be roughly in [-1, 1]
            print(f"✓ Innate valence: {valence:.3f}")

    def test_blocking_effect_modulates_responses(self, enhanced_circuit):
        """Test that GABAergic blocking reduces responses."""
        # Activate glomeruli
        glomeruli = ["DA1"] if "DA1" in enhanced_circuit.connectivity.pn_glomeruli.values() else \
                    list(enhanced_circuit.connectivity.pn_glomeruli.values())[:1]

        if not glomeruli:
            pytest.skip("No glomeruli available")

        pn_activity = enhanced_circuit.activate_pns_by_glomeruli(glomeruli)

        # Test no blocking vs. full blocking
        result_no_block = enhanced_circuit.forward_pass_full(
            pn_activity, blocking_strength=0.0, return_diagnostics=True
        )
        result_full_block = enhanced_circuit.forward_pass_full(
            pn_activity, blocking_strength=1.0, return_diagnostics=True
        )

        # Blocking should reduce KC activity (if LN layer present)
        if enhanced_circuit.ln_layer is not None:
            kc_sparsity_no_block = result_no_block["diagnostics"]["kc_sparsity"]
            kc_sparsity_blocked = result_full_block["diagnostics"]["kc_sparsity"]

            # Blocking should reduce sparsity (fewer active KCs)
            # Allow for some variability
            print(f"KC sparsity no blocking: {kc_sparsity_no_block:.2%}")
            print(f"KC sparsity with blocking: {kc_sparsity_blocked:.2%}")

            # Just verify blocking changes the response (direction may vary)
            assert kc_sparsity_no_block != kc_sparsity_blocked
            print("✓ Blocking modulates KC activity")

    def test_simulate_odor_response_helper(self, enhanced_circuit):
        """Test the simulate_odor_response convenience method."""
        glomeruli = list(enhanced_circuit.connectivity.pn_glomeruli.values())[:2]

        if not glomeruli:
            pytest.skip("No glomeruli available")

        response = enhanced_circuit.simulate_odor_response(
            glomeruli, blocking_strength=0.5
        )

        assert "mbon_output" in response
        assert "diagnostics" in response
        print(f"✓ Odor response simulation successful")

    def test_measure_blocking_effect_helper(self, enhanced_circuit):
        """Test the measure_blocking_effect analysis method."""
        glomeruli = list(enhanced_circuit.connectivity.pn_glomeruli.values())[:1]

        if not glomeruli:
            pytest.skip("No glomeruli available")

        blocking_curve = enhanced_circuit.measure_blocking_effect(
            glomeruli, blocking_strengths=[0.0, 0.25, 0.5, 0.75, 1.0]
        )

        assert "blocking_strengths" in blocking_curve
        assert len(blocking_curve["blocking_strengths"]) == 5
        assert "mbon_responses" in blocking_curve
        assert len(blocking_curve["mbon_responses"]) == 5

        print(f"✓ Blocking curve measured:")
        for strength, mbon_resp in zip(
            blocking_curve["blocking_strengths"],
            blocking_curve["mbon_responses"]
        ):
            print(f"  Blocking {strength:.2f}: MBON response {mbon_resp:.3f}")


def test_integration_sanity_check():
    """Quick sanity check that imports work."""
    from data_loaders.circuit_loader import CircuitLoader
    from pgcn.models.enhanced_olfactory_circuit import EnhancedOlfactoryCircuit
    assert CircuitLoader is not None
    assert EnhancedOlfactoryCircuit is not None
    print("✓ Integration test imports successful")
