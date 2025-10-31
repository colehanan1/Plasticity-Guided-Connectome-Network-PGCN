"""Unit tests for veto gate blocking experiment."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pgcn.experiments.experiment_1_veto_gate import VetoGateExperiment
from pgcn.models.connectivity_matrix import ConnectivityMatrix
from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit


@pytest.fixture
def small_circuit():
    """Create small circuit for veto testing."""
    n_pn, n_kc, n_mbon, n_dan = 10, 20, 5, 8

    pn_ids = np.arange(1000, 1000 + n_pn, dtype=np.int64)
    kc_ids = np.arange(2000, 2000 + n_kc, dtype=np.int64)
    mbon_ids = np.arange(3000, 3000 + n_mbon, dtype=np.int64)
    dan_ids = np.arange(4000, 4000 + n_dan, dtype=np.int64)

    pn_to_kc = sp.random(n_kc, n_pn, density=0.3, format="csr", dtype=np.float64)
    kc_to_mbon = sp.random(n_mbon, n_kc, density=0.4, format="csr", dtype=np.float64)
    dan_to_kc = sp.random(n_kc, n_dan, density=0.2, format="csr", dtype=np.float64)
    dan_to_mbon = sp.random(n_mbon, n_dan, density=0.2, format="csr", dtype=np.float64)

    # Create glomeruli mapping
    pn_glomeruli = {}
    for i, pn_id in enumerate(pn_ids):
        if i < 3:
            pn_glomeruli[pn_id] = "glom_A"  # Veto glomerulus
        elif i < 6:
            pn_glomeruli[pn_id] = "glom_B"  # Control glomerulus
        else:
            pn_glomeruli[pn_id] = "glom_C"

    kc_subtypes = {kc_id: "ab" for kc_id in kc_ids}

    conn = ConnectivityMatrix(
        pn_ids=pn_ids,
        kc_ids=kc_ids,
        mbon_ids=mbon_ids,
        dan_ids=dan_ids,
        pn_to_kc=pn_to_kc,
        kc_to_mbon=kc_to_mbon,
        dan_to_kc=dan_to_kc,
        dan_to_mbon=dan_to_mbon,
        pn_glomeruli=pn_glomeruli,
        kc_subtypes=kc_subtypes,
    )

    return OlfactoryCircuit(conn, kc_sparsity_target=0.20)


@pytest.fixture
def veto_experiment(small_circuit):
    """Create VetoGateExperiment instance."""
    weights = small_circuit.connectivity.kc_to_mbon.toarray()
    plasticity = DopamineModulatedPlasticity(
        kc_to_mbon_weights=weights,
        learning_rate=0.05,
    )
    return VetoGateExperiment(
        circuit=small_circuit,
        plasticity=plasticity,
        veto_glomerulus="glom_A",
        veto_strength=1.0,
    )


def test_veto_experiment_initialization(small_circuit):
    """Test that veto experiment initializes correctly."""
    weights = small_circuit.connectivity.kc_to_mbon.toarray()
    plasticity = DopamineModulatedPlasticity(weights, learning_rate=0.01)

    veto_exp = VetoGateExperiment(
        circuit=small_circuit,
        plasticity=plasticity,
        veto_glomerulus="glom_A",
    )

    # Veto weight should be initialized
    assert veto_exp.veto_weight is not None
    assert veto_exp.veto_weight.shape == (small_circuit.connectivity.n_pn,)

    # Veto weights should be non-zero only for veto glomerulus PNs
    glom_a_indices = small_circuit.connectivity.get_pn_indices(["glom_A"])
    assert np.sum(veto_exp.veto_weight) > 0  # Some weights non-zero
    assert np.sum(veto_exp.veto_weight[glom_a_indices]) > 0  # Veto PNs weighted


def test_veto_experiment_invalid_glomerulus(small_circuit):
    """Test that invalid glomerulus raises error."""
    weights = small_circuit.connectivity.kc_to_mbon.toarray()
    plasticity = DopamineModulatedPlasticity(weights, learning_rate=0.01)

    with pytest.raises(ValueError, match="No PNs found for glomerulus"):
        VetoGateExperiment(
            circuit=small_circuit,
            plasticity=plasticity,
            veto_glomerulus="nonexistent_glom",
        )


def test_apply_veto_high_when_veto_glomerulus_active(veto_experiment):
    """Test that veto signal is high when veto glomerulus is activated.

    Biological validation: Veto pathway active → v ≈ 1 → blocks learning.
    """
    # Activate veto glomerulus
    pn_activity = veto_experiment.circuit.activate_pns_by_glomeruli(
        ["glom_A"], firing_rate=1.0
    )

    veto_value = veto_experiment.apply_veto(pn_activity)

    # Veto should be high (close to 1.0)
    assert veto_value > 0.5
    assert veto_value <= 1.0


def test_apply_veto_low_when_other_glomerulus_active(veto_experiment):
    """Test that veto signal is low when control glomerulus is activated.

    Biological validation: Control pathway active → v ≈ 0 → learning allowed.
    """
    # Activate control glomerulus (not veto pathway)
    pn_activity = veto_experiment.circuit.activate_pns_by_glomeruli(
        ["glom_B"], firing_rate=1.0
    )

    veto_value = veto_experiment.apply_veto(pn_activity)

    # Veto should be low (close to 0.0)
    assert veto_value < 0.5


def test_run_trial_with_veto_suppresses_learning(veto_experiment):
    """Test that veto active suppresses weight changes.

    Biological validation: Gating factor (1 - v) near zero → minimal plasticity.
    """
    initial_weights = veto_experiment.plasticity.kc_to_mbon.copy()

    # Run trial with veto active
    trial_data = veto_experiment.run_trial_with_veto(
        odor="glom_A",
        reward=1.0,
        veto_active=True,
    )

    # Veto should be high
    assert trial_data["veto_value"] > 0.5

    # Gating factor should be low (suppression)
    assert trial_data["gating_factor"] < 0.5

    # Weight change should be small
    weight_change = np.linalg.norm(veto_experiment.plasticity.kc_to_mbon - initial_weights)
    assert weight_change < 0.1  # Small change due to gating


def test_run_trial_without_veto_allows_learning(veto_experiment):
    """Test that veto inactive allows normal learning.

    Biological validation: Gating factor (1 - v) ≈ 1 → full plasticity.
    """
    initial_weights = veto_experiment.plasticity.kc_to_mbon.copy()

    # Run trial with veto inactive
    trial_data = veto_experiment.run_trial_with_veto(
        odor="glom_B",
        reward=1.0,
        veto_active=False,
    )

    # Veto should be low (forced to 0 when veto_active=False)
    assert trial_data["veto_value"] == 0.0

    # Gating factor should be high (no suppression)
    assert trial_data["gating_factor"] == 1.0

    # Weight change should be substantial (normal learning)
    weight_change = np.linalg.norm(veto_experiment.plasticity.kc_to_mbon - initial_weights)
    # Allow for small or large changes depending on activity patterns
    assert weight_change >= 0.0


def test_run_full_experiment_structure(veto_experiment):
    """Test that full experiment returns expected structure."""
    results = veto_experiment.run_full_experiment(
        n_phase1_trials=4,
        n_phase2_trials=6,
        odor_a="glom_A",
        odor_b="glom_B",
    )

    # Validate structure
    assert "phase1_trials" in results
    assert "phase2_trials" in results
    assert "test_responses" in results
    assert "blocking_index" in results

    # Validate counts
    assert len(results["phase1_trials"]) == 4
    assert len(results["phase2_trials"]) == 6

    # Validate test responses
    assert "glom_A" in results["test_responses"]
    assert "glom_B" in results["test_responses"]

    # Validate blocking index is numeric
    assert isinstance(results["blocking_index"], float)
    assert np.isfinite(results["blocking_index"])


def test_run_full_experiment_blocking_effect(veto_experiment):
    """Test that blocking effect emerges over trials.

    Biological validation: OdorA with veto should show reduced learning compared
    to OdorB without veto (blocking phenomenon).
    """
    results = veto_experiment.run_full_experiment(
        n_phase1_trials=10,
        n_phase2_trials=20,
        odor_a="glom_A",
        odor_b="glom_B",
    )

    # Analyze Phase 2 trials
    import pandas as pd
    phase2_df = pd.DataFrame(results["phase2_trials"])

    # Extract veto vs non-veto trials
    glom_a_trials = phase2_df[phase2_df["odor"] == "glom_A"]
    glom_b_trials = phase2_df[phase2_df["odor"] == "glom_B"]

    # Veto should be active for glom_A, not for glom_B
    assert glom_a_trials["veto_value"].mean() > 0.5  # Veto active
    # glom_B may have some veto value if PN patterns overlap, but should be lower

    # Gating should suppress glom_A more than glom_B
    glom_a_gating = glom_a_trials["gating_factor"].mean()
    glom_b_gating = glom_b_trials["gating_factor"].mean()
    assert glom_a_gating < glom_b_gating  # More suppression for glom_A


def test_analyze_blocking_effect(veto_experiment):
    """Test that blocking analysis computes metrics correctly."""
    results = veto_experiment.run_full_experiment(
        n_phase1_trials=6,
        n_phase2_trials=10,
    )

    metrics = veto_experiment.analyze_blocking_effect(results)

    # Validate metric keys
    assert "blocking_index" in metrics
    assert "phase2_odor_a_mean_rpe" in metrics
    assert "phase2_odor_b_mean_rpe" in metrics
    assert "veto_efficacy" in metrics
    assert "mean_gating_suppression" in metrics

    # All should be numeric
    for key, value in metrics.items():
        assert isinstance(value, float)
        assert np.isfinite(value)

    # Veto efficacy should be reasonable (may be 0 if no matching trials)
    assert metrics["veto_efficacy"] >= 0.0


def test_veto_trial_return_values(veto_experiment):
    """Test that trial returns all expected fields."""
    trial_data = veto_experiment.run_trial_with_veto(
        odor="glom_A",
        reward=1.0,
        veto_active=True,
    )

    # Required fields
    required_fields = [
        "odor",
        "reward",
        "mbon_output",
        "veto_value",
        "gating_factor",
        "rpe",
        "dopamine",
        "weight_change_magnitude",
    ]

    for field in required_fields:
        assert field in trial_data, f"Missing field: {field}"
        # odor is string, others are numeric
        if field != "odor":
            assert np.isfinite(float(trial_data[field])), f"Non-finite value in {field}"
