"""Unit tests for behavioral validation against real fly data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pgcn.analysis.behavioral_validation import BehavioralValidator


@pytest.fixture
def synthetic_fly_data():
    """Create synthetic fly behavioral data for testing."""
    n_flies = 3
    n_trials = 10

    data = []
    for fly_id in range(n_flies):
        for trial in range(n_trials):
            # Control condition: gradually increasing probability (learning)
            data.append({
                'dataset': 'control',
                'fly': f'fly_{fly_id}',
                'fly_number': fly_id,
                'trial_label': f'trial_{trial}',
                'prediction': 1 if trial > 5 else 0,
                'probability': 0.1 + 0.08 * trial,  # Linear increase
            })

            # Opto condition: reduced learning
            data.append({
                'dataset': 'opto_silencing',
                'fly': f'fly_{fly_id}',
                'fly_number': fly_id,
                'trial_label': f'trial_{trial}',
                'prediction': 1 if trial > 7 else 0,
                'probability': 0.1 + 0.03 * trial,  # Slower increase
            })

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_model_curves():
    """Create synthetic model learning curves."""
    return {
        'control': np.linspace(0.1, 0.9, 10),
        'opto_silencing': np.linspace(0.1, 0.4, 10),
    }


@pytest.fixture
def validator(synthetic_model_curves, synthetic_fly_data):
    """Create BehavioralValidator with synthetic data."""
    return BehavioralValidator(synthetic_model_curves, synthetic_fly_data)


def test_load_real_behavioral_data():
    """Test loading real behavioral data from CSV."""
    data_path = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv")

    if not data_path.exists():
        pytest.skip("Real behavioral data file not found")

    fly_data = BehavioralValidator.load_behavioral_data(data_path)

    # Check structure
    assert len(fly_data) > 0, "Behavioral data should not be empty"
    assert 'dataset' in fly_data.columns
    assert 'fly' in fly_data.columns
    assert 'trial_label' in fly_data.columns
    assert 'prediction' in fly_data.columns
    assert 'probability' in fly_data.columns

    # Check datasets present
    datasets = fly_data['dataset'].unique()
    assert len(datasets) > 0, "Should have at least one dataset"
    print(f"Found datasets: {datasets}")


def test_compute_learning_index():
    """Test learning index computation."""
    # Strong aversive learning: avoid CS+, approach CS-
    mbon_cs_plus = np.array([0.1, 0.1, 0.1])  # Low MBON → avoidance
    mbon_cs_minus = np.array([0.9, 0.9, 0.9])  # High MBON → approach

    li = BehavioralValidator.compute_learning_index(mbon_cs_plus, mbon_cs_minus)

    assert -1.0 <= li <= 1.0, "Learning index should be in [-1, 1]"
    assert li > 0.5, "Strong learning should have LI > 0.5"

    # No learning: equal responses
    mbon_equal = np.array([0.5, 0.5, 0.5])
    li_no_learning = BehavioralValidator.compute_learning_index(mbon_equal, mbon_equal)
    assert np.abs(li_no_learning) < 0.1, "Equal responses should give LI ≈ 0"


def test_compare_learning_curves_with_synthetic_data(validator):
    """Test comparing model curves to synthetic fly data."""
    metrics = validator.compare_learning_curves('control')

    # Check all expected metrics present
    assert 'rmse' in metrics
    assert 'pearson_r' in metrics
    assert 'pearson_p' in metrics
    assert 'saturation_similarity' in metrics
    assert 'learning_rate_similarity' in metrics
    assert 'model_final_value' in metrics
    assert 'fly_final_value' in metrics

    # Check metrics in valid ranges
    assert metrics['rmse'] >= 0.0, "RMSE should be non-negative"
    assert -1.0 <= metrics['pearson_r'] <= 1.0, "Pearson r should be in [-1, 1]"
    assert 0.0 <= metrics['saturation_similarity'] <= 1.0
    assert 0.0 <= metrics['learning_rate_similarity'] <= 1.0

    # Model and fly both show increasing learning, so should correlate
    assert metrics['pearson_r'] > 0.5, "Both curves increase, should correlate positively"


def test_compare_learning_curves_control_vs_opto(validator):
    """Test comparing control vs optogenetic conditions."""
    control_metrics = validator.compare_learning_curves('control')
    opto_metrics = validator.compare_learning_curves('opto_silencing')

    # Control should show stronger final learning than opto
    assert control_metrics['fly_final_value'] > opto_metrics['fly_final_value'], \
        "Control flies should learn more than opto-silenced flies"

    # Both should have reasonable correlations (model matches data pattern)
    assert control_metrics['pearson_r'] > 0.3, "Control model should correlate with data"
    assert opto_metrics['pearson_r'] > 0.0, "Opto model should show some correlation"


def test_predict_optogenetic_outcome(validator):
    """Test predicting optogenetic perturbation effects."""
    # Predict PN silencing effect
    predicted_pn = validator.predict_optogenetic_outcome(
        perturbation_type="silence",
        target_neurons="pn",
        control_condition="control",
        efficacy=1.0,
    )

    control_final = validator.model_curves['control'][-1]

    # PN silencing should reduce learning but not eliminate it (redundant input)
    assert predicted_pn < control_final, "Silencing should reduce learning"
    assert predicted_pn > 0.5 * control_final, "PN silencing shouldn't eliminate all learning"

    # Predict MBON silencing (more severe)
    predicted_mbon = validator.predict_optogenetic_outcome(
        perturbation_type="silence",
        target_neurons="mbon",
        control_condition="control",
        efficacy=1.0,
    )

    assert predicted_mbon < predicted_pn, \
        "MBON silencing should be more severe than PN silencing"

    # Predict DAN silencing (most severe - teaching signal)
    predicted_dan = validator.predict_optogenetic_outcome(
        perturbation_type="silence",
        target_neurons="dan",
        control_condition="control",
        efficacy=1.0,
    )

    assert predicted_dan < predicted_mbon, \
        "DAN silencing should be most severe (no teaching signal)"
    assert predicted_dan < 0.1 * control_final, \
        "DAN silencing should nearly eliminate learning"


def test_predict_optogenetic_activation(validator):
    """Test predicting optogenetic activation effects."""
    control_final = validator.model_curves['control'][-1]

    # Predict DAN activation (should enhance learning)
    predicted_dan_activate = validator.predict_optogenetic_outcome(
        perturbation_type="activate",
        target_neurons="dan",
        control_condition="control",
        efficacy=0.8,
    )

    assert predicted_dan_activate > control_final, \
        "DAN activation should enhance learning beyond control"


def test_efficacy_parameter_in_prediction(validator):
    """Test that efficacy parameter scales prediction appropriately."""
    control_final = validator.model_curves['control'][-1]

    # Test different efficacy levels
    efficacies = [0.0, 0.25, 0.5, 0.75, 1.0]
    predictions = []

    for efficacy in efficacies:
        pred = validator.predict_optogenetic_outcome(
            perturbation_type="silence",
            target_neurons="kc",
            control_condition="control",
            efficacy=efficacy,
        )
        predictions.append(pred)

    # Higher efficacy should cause larger deficit (monotonic decrease)
    for i in range(len(predictions) - 1):
        assert predictions[i] >= predictions[i + 1], \
            f"Higher efficacy should cause larger deficit: {efficacies[i]} vs {efficacies[i+1]}"

    # efficacy=0 should equal control
    assert np.abs(predictions[0] - control_final) < 1e-6, \
        "efficacy=0 should leave learning unchanged"


def test_compute_aggregate_validation_metrics(validator):
    """Test computing aggregate metrics across multiple conditions."""
    aggregate = validator.compute_aggregate_validation_metrics()

    # Should have one row per condition
    assert len(aggregate) >= 1, "Should have at least one condition compared"

    # Check columns present
    assert 'condition' in aggregate.columns
    assert 'rmse' in aggregate.columns
    assert 'pearson_r' in aggregate.columns

    # All RMSE values should be non-negative
    assert (aggregate['rmse'] >= 0).all(), "All RMSE values should be non-negative"

    # All Pearson r values should be in [-1, 1]
    assert (aggregate['pearson_r'] >= -1.0).all()
    assert (aggregate['pearson_r'] <= 1.0).all()


def test_integration_with_real_behavioral_data():
    """Integration test: Load real data and compare to model curves."""
    data_path = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv")

    if not data_path.exists():
        pytest.skip("Real behavioral data file not found")

    # Load real fly data
    fly_data = BehavioralValidator.load_behavioral_data(data_path)

    # Create model curves for available datasets
    datasets = fly_data['dataset'].unique()
    model_curves = {}
    for dataset in datasets:
        # Create synthetic learning curves (in real use, these would come from model)
        # Control datasets: strong learning
        if 'control' in dataset.lower():
            model_curves[dataset] = np.linspace(0.1, 0.85, 20)
        # Opto datasets: reduced learning
        else:
            model_curves[dataset] = np.linspace(0.1, 0.45, 20)

    # Create validator
    validator = BehavioralValidator(model_curves, fly_data)

    # Compare one condition
    test_dataset = datasets[0]
    if test_dataset in model_curves:
        metrics = validator.compare_learning_curves(test_dataset, dataset_name=test_dataset)

        # Verify metrics computed
        assert 'rmse' in metrics
        assert 'pearson_r' in metrics
        assert metrics['rmse'] >= 0.0
        assert -1.0 <= metrics['pearson_r'] <= 1.0

        print(f"Real data validation for {test_dataset}:")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  Pearson r: {metrics['pearson_r']:.3f}")
        print(f"  Saturation similarity: {metrics['saturation_similarity']:.3f}")


def test_curve_interpolation_with_different_lengths(validator):
    """Test that curves of different lengths are properly interpolated."""
    # Create model curve with different length than fly data
    short_model_curve = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # 5 points

    validator.model_curves['short_condition'] = short_model_curve

    # Synthetic fly data has 10 trials per condition
    # This should trigger interpolation
    metrics = validator.compare_learning_curves('short_condition', dataset_name='control')

    # Should succeed without error (interpolation handled internally)
    assert 'rmse' in metrics
    assert 'pearson_r' in metrics
    assert metrics['n_trials_compared'] == 5, \
        "Should compare based on model curve length"


def test_validation_with_learning_curve_from_model_experiment():
    """Test validation using actual learning curves from Phase 2 learning experiments."""
    from pathlib import Path
    from data_loaders.circuit_loader import CircuitLoader
    from pgcn.models.olfactory_circuit import OlfactoryCircuit
    from pgcn.models.learning_model import DopamineModulatedPlasticity, LearningExperiment

    # Load circuit
    loader = CircuitLoader(cache_dir=Path("data/cache"))
    conn = loader.load_connectivity_matrix(normalize_weights="row")
    circuit = OlfactoryCircuit(conn, kc_sparsity_target=0.05)

    # Run learning experiment to generate model curve
    plasticity = DopamineModulatedPlasticity(
        kc_to_mbon_weights=conn.kc_to_mbon.toarray(),
        learning_rate=0.01,
    )
    experiment = LearningExperiment(circuit, plasticity, n_trials=20)
    results = experiment.run_experiment(
        odor_sequence=["DA1"] * 20,
        reward_sequence=[1] * 20,
    )

    # Extract learning curve (MBON output over trials)
    model_curve = results['mbon_valence'].values

    # Load real fly data (if available)
    data_path = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv")
    if not data_path.exists():
        pytest.skip("Real behavioral data not found")

    fly_data = BehavioralValidator.load_behavioral_data(data_path)

    # Create validator with model-generated curve
    model_curves = {'model_control': model_curve}
    validator = BehavioralValidator(model_curves, fly_data)

    # Compare to control dataset in real data
    control_datasets = [ds for ds in fly_data['dataset'].unique() if 'control' in ds.lower()]
    if len(control_datasets) > 0:
        metrics = validator.compare_learning_curves(
            'model_control',
            dataset_name=control_datasets[0]
        )

        # Verify comparison succeeded
        assert 'rmse' in metrics
        assert 'pearson_r' in metrics
        assert metrics['rmse'] >= 0.0

        print(f"Model learning curve validation:")
        print(f"  Model final value: {metrics['model_final_value']:.3f}")
        print(f"  Fly final value: {metrics['fly_final_value']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  Pearson r: {metrics['pearson_r']:.3f}")
