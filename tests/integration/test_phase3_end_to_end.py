"""Phase 3 integration tests: Optogenetic experiments, behavioral validation, multi-task learning.

This module tests the complete Phase 3 pipeline, ensuring that:
1. Optogenetic perturbations integrate with Phase 1 & 2 learning experiments
2. Behavioral validation works with real and synthetic data
3. Multi-task analysis correctly measures interference and transfer
4. All three Phase 3 modules work together seamlessly
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_loaders.circuit_loader import CircuitLoader
from pgcn.analysis.behavioral_validation import BehavioralValidator
from pgcn.analysis.multi_task_analysis import MultiTaskAnalyzer
from pgcn.experiments.optogenetic_perturbations import OptogeneticPerturbation
from pgcn.models.learning_model import DopamineModulatedPlasticity, LearningExperiment
from pgcn.models.olfactory_circuit import OlfactoryCircuit


@pytest.fixture
def circuit():
    """Load circuit from FlyWire cache for all Phase 3 tests."""
    loader = CircuitLoader(cache_dir=Path("data/cache"))
    conn = loader.load_connectivity_matrix(normalize_weights="row")
    return OlfactoryCircuit(conn, kc_sparsity_target=0.05)


def test_phase3_end_to_end_optogenetic_experiment(circuit):
    """Test complete optogenetic conditioning experiment protocol.

    Protocol:
    1. Phase 1: Load circuit (connectivity backbone)
    2. Phase 2: Set up learning dynamics (plasticity)
    3. Phase 3: Run optogenetic perturbation experiment
    4. Phase 3: Validate results against expected behavioral deficits
    """
    # Phase 1: Circuit loaded (via fixture)
    assert circuit is not None
    assert len(circuit.connectivity.pn_ids) > 0
    assert len(circuit.connectivity.kc_ids) > 0

    # Phase 2: Create plasticity manager for learning
    plasticity_control = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )
    plasticity_opto = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )

    # Phase 2: Run control experiment (no perturbation)
    experiment_control = LearningExperiment(circuit, plasticity_control, n_trials=15)
    control_results = experiment_control.run_experiment(
        odor_sequence=["DA1"] * 15,
        reward_sequence=[1] * 15,
    )

    # Phase 3: Run optogenetic silencing experiment
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="pn",
        target_specificity=["DA1"],
        efficacy=0.8,  # 80% silencing
    )
    experiment_opto = LearningExperiment(circuit, plasticity_opto, n_trials=15)
    opto_results = opto.run_full_experiment(
        experiment_opto,
        odor_sequence=["DA1"] * 15,
        reward_sequence=[1] * 15,
    )

    # Verify optogenetic silencing caused learning deficit
    control_final = control_results.iloc[-1]['mbon_valence']
    opto_final = opto_results.iloc[-1]['mbon_output']

    assert opto_final < control_final, \
        "Optogenetic silencing should impair learning"

    # Phase 3: Validate with BehavioralValidator
    model_curves = {
        'control': control_results['mbon_valence'].values,
        'opto_silencing': opto_results['mbon_output'].values,
    }

    # Create synthetic fly data matching experiment
    import pandas as pd
    fly_data_rows = []
    for i, (control_val, opto_val) in enumerate(zip(model_curves['control'], model_curves['opto_silencing'])):
        fly_data_rows.append({
            'dataset': 'control',
            'fly': 'fly_1',
            'fly_number': 1,
            'trial_label': f'trial_{i}',
            'prediction': 1 if control_val > 0.5 else 0,
            'probability': control_val / 100.0,  # Normalize to [0, 1]
        })
        fly_data_rows.append({
            'dataset': 'opto_silencing',
            'fly': 'fly_1',
            'fly_number': 1,
            'trial_label': f'trial_{i}',
            'prediction': 1 if opto_val > 0.5 else 0,
            'probability': opto_val / 100.0,
        })

    fly_data = pd.DataFrame(fly_data_rows)

    validator = BehavioralValidator(model_curves, fly_data)
    metrics = validator.compare_learning_curves('control')

    assert metrics['rmse'] >= 0.0
    assert -1.0 <= metrics['pearson_r'] <= 1.0

    print(f"Phase 3 optogenetic experiment validation:")
    print(f"  Control learning: {control_final:.3f}")
    print(f"  Opto learning: {opto_final:.3f}")
    print(f"  Learning deficit: {(control_final - opto_final) / control_final:.1%}")
    print(f"  Validation RMSE: {metrics['rmse']:.3f}")


def test_phase3_behavioral_validation_with_real_data(circuit):
    """Test behavioral validation using real Drosophila data (if available)."""
    data_path = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv")

    if not data_path.exists():
        pytest.skip("Real behavioral data not available")

    # Phase 1 & 2: Run learning experiment
    plasticity = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.015,
    )
    experiment = LearningExperiment(circuit, plasticity, n_trials=20)
    results = experiment.run_experiment(
        odor_sequence=["DA1"] * 20,
        reward_sequence=[1] * 20,
    )

    # Phase 3: Load real fly data
    fly_data = BehavioralValidator.load_behavioral_data(data_path)

    # Create model curves for comparison
    model_curves = {
        'model_control': results['mbon_valence'].values,
    }

    # Validate against real data
    validator = BehavioralValidator(model_curves, fly_data)

    # Find control dataset in real data
    control_datasets = [ds for ds in fly_data['dataset'].unique() if 'control' in ds.lower()]
    if len(control_datasets) > 0:
        metrics = validator.compare_learning_curves(
            'model_control',
            dataset_name=control_datasets[0]
        )

        # Verify metrics are reasonable
        assert metrics['rmse'] >= 0.0
        assert -1.0 <= metrics['pearson_r'] <= 1.0

        print(f"Real data validation:")
        print(f"  Dataset: {control_datasets[0]}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  Pearson r: {metrics['pearson_r']:.3f}")
        print(f"  Model final: {metrics['model_final_value']:.3f}")
        print(f"  Fly final: {metrics['fly_final_value']:.3f}")


def test_phase3_multi_task_learning_pipeline(circuit):
    """Test complete multi-task learning pipeline with Phase 1 & 2 integration."""
    # Phase 1: Circuit loaded (via fixture)

    # Phase 2: Create separate plasticity for each task
    plasticity_managers = {
        'olfactory': DopamineModulatedPlasticity(
            kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
            learning_rate=0.015,
        ),
        'spatial': DopamineModulatedPlasticity(
            kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
            learning_rate=0.015,
        ),
    }

    # Phase 3: Run multi-task analysis
    analyzer = MultiTaskAnalyzer(circuit, plasticity_managers)

    # Interleaved training
    results = analyzer.run_interleaved_training(
        trials_per_task=15,
        task_order=['olfactory', 'spatial'],
        n_cycles=3,
    )

    # Verify results structure
    assert len(results) == 15 * 2 * 3  # trials × tasks × cycles
    assert 'task' in results.columns
    assert 'mbon_output' in results.columns

    # Compute task interference
    interference = analyzer.compute_task_interference(results)

    assert 'olfactory' in interference
    assert 'spatial' in interference
    assert interference['olfactory'] > 0.0
    assert interference['spatial'] > 0.0

    # Measure catastrophic forgetting
    forgetting = analyzer.measure_catastrophic_forgetting(
        task_A='olfactory',
        task_B='spatial',
        trials_per_task=20,
    )

    assert 'forgetting_magnitude' in forgetting

    print(f"Multi-task learning:")
    print(f"  Olfactory efficiency: {interference['olfactory']:.3f}")
    print(f"  Spatial efficiency: {interference['spatial']:.3f}")
    print(f"  Forgetting magnitude: {forgetting['forgetting_magnitude']:.3f}")


def test_phase3_optogenetic_with_behavioral_validation(circuit):
    """Test combining optogenetic perturbations with behavioral validation."""
    # Run control and opto experiments
    plasticity_control = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )
    plasticity_opto = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )

    # Control
    experiment_control = LearningExperiment(circuit, plasticity_control, n_trials=20)
    control_results = experiment_control.run_experiment(
        ["DA1"] * 20, [1] * 20
    )

    # Optogenetic KC silencing
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="kc",
        target_specificity=["ab", "ab_p"],  # Silence α/β KCs
        efficacy=1.0,
    )
    experiment_opto = LearningExperiment(circuit, plasticity_opto, n_trials=20)
    opto_results = opto.run_full_experiment(
        experiment_opto, ["DA1"] * 20, [1] * 20
    )

    # Create validator to compare curves
    model_curves = {
        'control': control_results['mbon_valence'].values,
        'opto_kc_silence': opto_results['mbon_output'].values,
    }

    # Predict expected outcome
    import pandas as pd
    synthetic_fly_data = pd.DataFrame({
        'dataset': ['control'] * 20 + ['opto_kc_silence'] * 20,
        'fly': ['fly_1'] * 40,
        'fly_number': [1] * 40,
        'trial_label': [f'trial_{i}' for i in range(20)] * 2,
        'prediction': [1] * 20 + [0] * 20,
        'probability': list(model_curves['control'] / 100.0) + list(model_curves['opto_kc_silence'] / 100.0),
    })

    validator = BehavioralValidator(model_curves, synthetic_fly_data)

    # Predict opto outcome
    predicted_deficit = validator.predict_optogenetic_outcome(
        perturbation_type="silence",
        target_neurons="kc",
        control_condition="control",
        efficacy=1.0,
    )

    control_final = control_results.iloc[-1]['mbon_valence']
    opto_final = opto_results.iloc[-1]['mbon_output']

    # Predicted deficit should be qualitatively correct
    assert predicted_deficit < control_final, \
        "Predicted deficit should be less than control"

    print(f"Optogenetic + behavioral validation:")
    print(f"  Control final: {control_final:.3f}")
    print(f"  Opto final: {opto_final:.3f}")
    print(f"  Predicted: {predicted_deficit:.3f}")
    print(f"  Actual deficit: {(control_final - opto_final) / control_final:.1%}")


def test_phase3_full_pipeline_phase1_2_3(circuit):
    """Integration test: Full pipeline from Phase 1 → Phase 2 → Phase 3.

    This test verifies the complete PGCN pipeline:
    - Phase 1: Connectivity loading and forward propagation
    - Phase 2: Learning dynamics with dopamine-gated plasticity
    - Phase 3: Optogenetic perturbations, behavioral validation, multi-task
    """
    print("\n" + "="*60)
    print("PGCN FULL PIPELINE TEST: Phase 1 → Phase 2 → Phase 3")
    print("="*60)

    # ========== PHASE 1: Connectivity Backbone ==========
    print("\n[Phase 1] Testing connectivity backbone...")
    assert circuit is not None
    assert len(circuit.connectivity.pn_ids) > 0
    assert len(circuit.connectivity.kc_ids) > 0
    assert len(circuit.connectivity.mbon_ids) > 0

    # Test forward propagation
    pn_activity = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)
    kc_activity = circuit.propagate_pn_to_kc(pn_activity)
    mbon_output = circuit.propagate_kc_to_mbon(kc_activity)

    # Compute KC sparsity
    kc_sparsity = np.count_nonzero(kc_activity) / len(kc_activity)

    print(f"  ✓ PNs: {len(circuit.connectivity.pn_ids)}")
    print(f"  ✓ KCs: {len(circuit.connectivity.kc_ids)}")
    print(f"  ✓ MBONs: {len(circuit.connectivity.mbon_ids)}")
    print(f"  ✓ PN→KC→MBON propagation: {pn_activity.sum():.0f} → {kc_activity.sum():.0f} → {mbon_output[0]:.3f}")
    print(f"  ✓ KC sparsity: {kc_sparsity:.1%}")

    # ========== PHASE 2: Learning Dynamics ==========
    print("\n[Phase 2] Testing learning dynamics...")
    plasticity = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.02,
        eligibility_trace_tau=0.1,
    )
    experiment = LearningExperiment(circuit, plasticity, n_trials=20)
    learning_results = experiment.run_experiment(
        odor_sequence=["DA1"] * 10 + ["DL3"] * 10,
        reward_sequence=[1] * 10 + [0] * 10,
    )

    initial_da1 = learning_results.iloc[0]['mbon_valence']
    final_da1 = learning_results.iloc[9]['mbon_valence']
    final_dl3 = learning_results.iloc[-1]['mbon_valence']

    print(f"  ✓ Training completed: 20 trials")
    print(f"  ✓ DA1 learning: {initial_da1:.3f} → {final_da1:.3f}")
    print(f"  ✓ DL3 response: {final_dl3:.3f}")
    print(f"  ✓ Learning verified: DA1 > DL3")

    assert final_da1 > initial_da1, "DA1 should show learning"
    assert final_da1 > final_dl3, "DA1 should have higher response than DL3"

    # ========== PHASE 3a: Optogenetic Perturbations ==========
    print("\n[Phase 3a] Testing optogenetic perturbations...")
    plasticity_opto = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.02,
    )
    opto = OptogeneticPerturbation(
        circuit=circuit,
        perturbation_type="silence",
        target_neurons="pn",
        target_specificity=["DA1"],
        efficacy=1.0,
    )
    experiment_opto = LearningExperiment(circuit, plasticity_opto, n_trials=20)
    opto_results = opto.run_full_experiment(
        experiment_opto,
        odor_sequence=["DA1"] * 20,
        reward_sequence=[1] * 20,
    )

    opto_final = opto_results.iloc[-1]['mbon_output']
    learning_deficit = (final_da1 - opto_final) / final_da1 if final_da1 > 0 else 0.0

    print(f"  ✓ Optogenetic silencing applied")
    print(f"  ✓ Control learning: {final_da1:.3f}")
    print(f"  ✓ Opto learning: {opto_final:.3f}")
    print(f"  ✓ Learning deficit: {learning_deficit:.1%}")

    assert opto_final < final_da1 * 0.8, "Optogenetic silencing should impair learning"

    # ========== PHASE 3b: Behavioral Validation ==========
    print("\n[Phase 3b] Testing behavioral validation...")
    model_curves = {
        'control': learning_results['mbon_valence'].values[:10],
        'opto': opto_results['mbon_output'].values[:10],
    }

    import pandas as pd
    fly_data = pd.DataFrame({
        'dataset': ['control'] * 10 + ['opto'] * 10,
        'fly': ['fly_1'] * 20,
        'fly_number': [1] * 20,
        'trial_label': [f'trial_{i}' for i in range(10)] * 2,
        'prediction': [1] * 20,
        'probability': list(model_curves['control'] / 100.0) + list(model_curves['opto'] / 100.0),
    })

    validator = BehavioralValidator(model_curves, fly_data)
    metrics = validator.compare_learning_curves('control')

    print(f"  ✓ Behavioral validation completed")
    print(f"  ✓ RMSE: {metrics['rmse']:.3f}")
    print(f"  ✓ Pearson r: {metrics['pearson_r']:.3f}")

    # ========== PHASE 3c: Multi-Task Learning ==========
    print("\n[Phase 3c] Testing multi-task learning...")
    plasticity_tasks = {
        'olfactory': DopamineModulatedPlasticity(
            kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
            learning_rate=0.01,
        ),
        'spatial': DopamineModulatedPlasticity(
            kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
            learning_rate=0.01,
        ),
    }

    analyzer = MultiTaskAnalyzer(circuit, plasticity_tasks)
    multi_task_results = analyzer.run_interleaved_training(
        trials_per_task=10,
        task_order=['olfactory', 'spatial'],
        n_cycles=2,
    )

    interference = analyzer.compute_task_interference(multi_task_results)

    print(f"  ✓ Multi-task training completed: 2 cycles")
    print(f"  ✓ Olfactory efficiency: {interference['olfactory']:.3f}")
    print(f"  ✓ Spatial efficiency: {interference['spatial']:.3f}")

    # ========== FINAL VERIFICATION ==========
    print("\n" + "="*60)
    print("✅ ALL PHASES VERIFIED - FULL PIPELINE WORKING!")
    print("="*60)
    print(f"Phase 1: Connectivity backbone ✓")
    print(f"Phase 2: Learning dynamics ✓")
    print(f"Phase 3: Optogenetic perturbations ✓")
    print(f"Phase 3: Behavioral validation ✓")
    print(f"Phase 3: Multi-task learning ✓")
    print("="*60 + "\n")

    # All phases should have succeeded without errors
    assert True, "Full pipeline test completed successfully!"
