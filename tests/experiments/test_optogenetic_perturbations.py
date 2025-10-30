"""Unit tests for optogenetic perturbation experiments."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from data_loaders.circuit_loader import CircuitLoader
from pgcn.experiments.optogenetic_perturbations import OptogeneticPerturbation
from pgcn.models.learning_model import DopamineModulatedPlasticity, LearningExperiment
from pgcn.models.olfactory_circuit import OlfactoryCircuit


@pytest.fixture
def circuit():
    """Load real circuit from cache for optogenetic experiments."""
    loader = CircuitLoader(cache_dir=Path("data/cache"))
    conn = loader.load_connectivity_matrix(normalize_weights="row")
    return OlfactoryCircuit(conn, kc_sparsity_target=0.05)


@pytest.fixture
def plasticity(circuit):
    """Create plasticity manager with fresh weights."""
    weights = circuit.connectivity.kc_to_mbon.toarray()
    return DopamineModulatedPlasticity(
        kc_to_mbon_weights=weights,
        learning_rate=0.01,
        eligibility_trace_tau=None,
        plasticity_mode="three_factor",
    )


def test_optogenetic_silence_pn_reduces_activity(circuit):
    """Test that silencing PNs reduces downstream KC activity."""
    # Create perturbation to silence DA1 PNs
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="pn",
        target_specificity=["DA1"],
        temporal_window=(0.0, 1.0),
        efficacy=1.0,  # Complete silencing
    )

    # Activate DA1 glomerulus
    pn_activity_control = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)
    pn_activity_perturbed = opto.apply_perturbation(pn_activity_control, trial_phase=0.5)

    # Check PN activity reduced
    assert pn_activity_perturbed.sum() < pn_activity_control.sum(), \
        "Silencing should reduce PN activity"
    assert np.allclose(pn_activity_perturbed.sum(), 0.0, atol=1e-6), \
        "Complete silencing (efficacy=1.0) should zero PN activity"

    # Propagate to KC and verify reduction
    kc_control = circuit.propagate_pn_to_kc(pn_activity_control)
    kc_perturbed = circuit.propagate_pn_to_kc(pn_activity_perturbed)

    assert kc_perturbed.sum() < kc_control.sum(), \
        "Silencing PNs should reduce KC activity"


def test_optogenetic_activate_kc_enhances_output(circuit):
    """Test that activating KCs enhances MBON output."""
    # Create perturbation to activate KCs
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="activate",
        target_neurons="kc",
        target_specificity=None,  # All KCs
        temporal_window=(0.0, 1.0),
        efficacy=0.8,
    )

    # Activate weak PN input
    pn_activity = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=0.3)
    kc_activity_control = circuit.propagate_pn_to_kc(pn_activity)
    kc_activity_perturbed = opto.apply_perturbation(kc_activity_control, trial_phase=0.5)

    # Check KC activity increased
    assert kc_activity_perturbed.sum() > kc_activity_control.sum(), \
        "Activation should increase KC activity"

    # Propagate to MBON and verify enhancement
    mbon_control = circuit.propagate_kc_to_mbon(kc_activity_control)
    mbon_perturbed = circuit.propagate_kc_to_mbon(kc_activity_perturbed)

    assert mbon_perturbed[0] > mbon_control[0], \
        "Activating KCs should enhance MBON output"


def test_temporal_window_specificity(circuit):
    """Test that perturbation only active during specified temporal window."""
    # Create perturbation active only in first half of trial
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="pn",
        target_specificity=["DA1"],
        temporal_window=(0.0, 0.5),  # First half only
        efficacy=1.0,
    )

    pn_activity = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)

    # Apply at trial_phase=0.3 (inside window)
    perturbed_early = opto.apply_perturbation(pn_activity, trial_phase=0.3)
    assert perturbed_early.sum() < pn_activity.sum(), \
        "Perturbation should be active during first half"

    # Apply at trial_phase=0.7 (outside window)
    perturbed_late = opto.apply_perturbation(pn_activity, trial_phase=0.7)
    assert np.allclose(perturbed_late, pn_activity), \
        "Perturbation should be inactive during second half"


def test_perturbation_specificity_to_glomerulus(circuit):
    """Test that perturbation only affects specified glomerulus."""
    # Silence only DA1, not DL3
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="pn",
        target_specificity=["DA1"],
        efficacy=1.0,
    )

    # Activate both DA1 and DL3
    pn_da1 = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)
    pn_dl3 = circuit.activate_pns_by_glomeruli(["DL3"], firing_rate=1.0)
    pn_both = pn_da1 + pn_dl3

    # Apply perturbation
    perturbed = opto.apply_perturbation(pn_both, trial_phase=0.5)

    # Check DA1 PNs silenced but DL3 PNs unaffected
    # DA1 should have zero activity
    da1_idx = [i for i, pid in enumerate(circuit.connectivity.pn_ids)
               if circuit.connectivity.pn_glomeruli.get(pid) == "DA1"]
    if len(da1_idx) > 0:
        assert np.allclose(perturbed[da1_idx].sum(), 0.0, atol=1e-6), \
            "DA1 PNs should be silenced"

    # DL3 should retain activity
    dl3_idx = [i for i, pid in enumerate(circuit.connectivity.pn_ids)
               if circuit.connectivity.pn_glomeruli.get(pid) == "DL3"]
    if len(dl3_idx) > 0:
        assert perturbed[dl3_idx].sum() > 0.0, \
            "DL3 PNs should retain activity"


def test_learning_deficit_from_pn_silencing(circuit, plasticity):
    """Test that silencing PNs causes learning deficit."""
    # Create two experiments: control vs silencing
    plasticity_control = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )
    plasticity_opto = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )

    # Control: normal learning
    experiment_control = LearningExperiment(circuit, plasticity_control, n_trials=10)
    control_results = experiment_control.run_experiment(
        odor_sequence=["DA1"] * 10,
        reward_sequence=[1] * 10,
    )

    # Experimental: silence DA1 PNs during learning
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="pn",
        target_specificity=["DA1"],
        efficacy=1.0,
    )
    experiment_opto = LearningExperiment(circuit, plasticity_opto, n_trials=10)
    opto_results = opto.run_full_experiment(
        experiment_opto,
        odor_sequence=["DA1"] * 10,
        reward_sequence=[1] * 10,
    )

    # Compare final MBON output (learning curve endpoint)
    control_final = control_results.iloc[-1]['mbon_valence']
    opto_final = opto_results.iloc[-1]['mbon_output']

    # Silencing should impair learning
    assert opto_final < control_final, \
        "Silencing PNs should reduce learning compared to control"

    # With complete silencing, learning should be near zero
    assert opto_final < 0.1 * control_final, \
        "Complete PN silencing should prevent nearly all learning"


def test_full_learning_curve_comparison(circuit):
    """Test full learning curve comparison between control and perturbed conditions."""
    # Create fresh plasticity for each condition
    plasticity_control = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )
    plasticity_opto = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )

    n_trials = 20
    odor_seq = ["DA1"] * n_trials
    reward_seq = [1] * n_trials

    # Control condition
    experiment_control = LearningExperiment(circuit, plasticity_control, n_trials=n_trials)
    control_results = experiment_control.run_experiment(odor_seq, reward_seq)

    # Perturbed condition: mild silencing (efficacy=0.2 to allow some learning)
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="pn",
        target_specificity=["DA1"],
        efficacy=0.2,  # 20% silencing - mild enough to allow learning
    )
    experiment_opto = LearningExperiment(circuit, plasticity_opto, n_trials=n_trials)
    opto_results = opto.run_full_experiment(experiment_opto, odor_seq, reward_seq)

    # Check learning curves
    # 1. Control should show clear learning
    control_early = control_results.iloc[0]['mbon_valence']
    control_late = control_results.iloc[-1]['mbon_valence']
    assert control_late > control_early, "Control should show learning"

    opto_early = opto_results.iloc[0]['mbon_output']
    opto_late = opto_results.iloc[-1]['mbon_output']

    # 2. Perturbed should show reduced final response compared to control
    assert opto_late < control_late, \
        "Perturbed condition should show lower final response than control"

    # 3. Calculate learning magnitudes
    control_learning = control_late - control_early
    opto_learning = opto_late - opto_early

    # Perturbed should show less learning (may be zero if severely impaired)
    assert opto_learning <= control_learning, \
        "Perturbed condition should not learn more than control"

    # 4. Test that silencing causes measurable deficit
    # With mild silencing, learning should be detectable but reduced
    if opto_learning > 0:
        learning_deficit_ratio = opto_learning / control_learning
        assert learning_deficit_ratio < 1.0, \
            f"Silencing should reduce learning, got ratio {learning_deficit_ratio:.2f}"


def test_kc_subtype_targeting(circuit):
    """Test that perturbation can target specific KC subtypes."""
    # Create perturbation targeting α/β KCs only
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="kc",
        target_specificity=["ab", "ab_p"],  # α/β subtypes
        efficacy=1.0,
    )

    # Activate PNs and propagate to KC
    pn_activity = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)
    kc_activity_control = circuit.propagate_pn_to_kc(pn_activity)
    kc_activity_perturbed = opto.apply_perturbation(kc_activity_control, trial_phase=0.5)

    # Check that some KCs are silenced but not all
    assert kc_activity_perturbed.sum() < kc_activity_control.sum(), \
        "Silencing α/β KCs should reduce total KC activity"

    n_silenced = np.sum(kc_activity_perturbed < kc_activity_control)
    n_total = len(kc_activity_control)
    assert 0 < n_silenced < n_total, \
        "Should silence some but not all KCs (subtype-specific)"


def test_mbon_perturbation_affects_learning(circuit):
    """Test that perturbing MBON output affects learning update."""
    plasticity = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )

    # Create MBON silencing perturbation
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="mbon",
        efficacy=1.0,
    )

    experiment = LearningExperiment(circuit, plasticity, n_trials=1)

    # Run trial with MBON silencing
    result = opto.run_learning_trial_with_opto(experiment, "DA1", reward=1)

    # Check MBON output silenced
    assert result['mbon_output'] < 1.0, \
        "MBON silencing should reduce output"

    # RPE should be affected (reward expectation is wrong)
    assert result['rpe'] != 0.0, \
        "RPE should be computed based on perturbed MBON output"


def test_efficacy_parameter_controls_strength(circuit):
    """Test that efficacy parameter controls perturbation strength."""
    pn_activity = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)

    # Test different efficacy levels
    efficacies = [0.0, 0.25, 0.5, 0.75, 1.0]
    silenced_activities = []

    for efficacy in efficacies:
        opto = OptogeneticPerturbation(
            circuit=circuit,
            perturbation_type="silence",
            target_neurons="pn",
            target_specificity=["DA1"],
            efficacy=efficacy,
        )
        perturbed = opto.apply_perturbation(pn_activity, trial_phase=0.5)
        silenced_activities.append(perturbed.sum())

    # Check monotonic decrease with increasing efficacy
    for i in range(len(silenced_activities) - 1):
        assert silenced_activities[i] >= silenced_activities[i + 1], \
            f"Higher efficacy should cause more silencing: {efficacies[i]} vs {efficacies[i+1]}"

    # efficacy=0.0 should leave activity unchanged
    assert np.allclose(silenced_activities[0], pn_activity.sum()), \
        "efficacy=0.0 should not change activity"

    # efficacy=1.0 should zero activity
    assert np.allclose(silenced_activities[-1], 0.0, atol=1e-6), \
        "efficacy=1.0 should completely silence"


def test_integration_optogenetic_conditioning_experiment(circuit):
    """Integration test: full optogenetic conditioning experiment protocol.

    Protocol:
    Phase 1: Train control flies on DA1→reward (20 trials)
    Phase 2: Train experimental flies on DA1→reward with DA1 PN silencing (20 trials)
    Phase 3: Test both groups on DA1 alone (5 trials, no silencing)

    Prediction: Control flies show strong DA1 response, experimental flies show weak response.
    """
    n_train_trials = 20  # Increased for more robust learning

    # Control group
    plasticity_control = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )
    experiment_control = LearningExperiment(circuit, plasticity_control, n_trials=n_train_trials)
    control_training = experiment_control.run_experiment(
        odor_sequence=["DA1"] * n_train_trials,
        reward_sequence=[1] * n_train_trials,
    )

    # Experimental group: partial silencing during training (70% efficacy)
    plasticity_opto = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="pn",
        target_specificity=["DA1"],
        efficacy=0.7,  # 70% silencing - severe but allows some learning
    )
    experiment_opto = LearningExperiment(circuit, plasticity_opto, n_trials=n_train_trials)
    opto_training = opto.run_full_experiment(
        experiment_opto,
        odor_sequence=["DA1"] * n_train_trials,
        reward_sequence=[1] * n_train_trials,
    )

    # Test phase: both groups tested without perturbation
    # Use trained weights from plasticity managers
    # Control group test
    control_test_results = []
    for _ in range(5):
        pn_act = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)
        kc_act = circuit.propagate_pn_to_kc(pn_act)
        # Use trained control weights
        mbon_out = np.dot(kc_act, plasticity_control.kc_to_mbon.T)
        control_test_results.append(mbon_out[0])

    # Experimental group test (using weights trained with silencing)
    opto_test_results = []
    for _ in range(5):
        pn_act = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)
        kc_act = circuit.propagate_pn_to_kc(pn_act)
        # Use trained opto weights
        mbon_out = np.dot(kc_act, plasticity_opto.kc_to_mbon.T)
        opto_test_results.append(mbon_out[0])

    # Compare test responses
    control_test_mean = np.mean(control_test_results)
    opto_test_mean = np.mean(opto_test_results)

    assert control_test_mean > opto_test_mean, \
        "Control group should show stronger test response than experimental group"

    # Control group should show substantial learning
    control_initial = control_training.iloc[0]['mbon_valence']
    control_final = control_training.iloc[-1]['mbon_valence']
    assert control_final > control_initial + 0.05, \
        "Control group should show clear learning"

    # Experimental group should show reduced learning
    opto_initial = opto_training.iloc[0]['mbon_output']
    opto_final = opto_training.iloc[-1]['mbon_output']
    learning_magnitude = opto_final - opto_initial
    assert learning_magnitude < 0.6 * (control_final - control_initial), \
        "Experimental group should show impaired learning"
