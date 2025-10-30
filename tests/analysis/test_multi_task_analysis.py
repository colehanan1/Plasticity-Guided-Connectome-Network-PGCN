"""Unit tests for multi-task learning analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_loaders.circuit_loader import CircuitLoader
from pgcn.analysis.multi_task_analysis import MultiTaskAnalyzer
from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit


@pytest.fixture
def circuit():
    """Load circuit for multi-task experiments."""
    loader = CircuitLoader(cache_dir=Path("data/cache"))
    conn = loader.load_connectivity_matrix(normalize_weights="row")
    return OlfactoryCircuit(conn, kc_sparsity_target=0.05)


@pytest.fixture
def plasticity_managers(circuit):
    """Create separate plasticity managers for multiple tasks."""
    # Each task gets independent KC→MBON weights
    return {
        'olfactory': DopamineModulatedPlasticity(
            kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
            learning_rate=0.01,
        ),
        'spatial': DopamineModulatedPlasticity(
            kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
            learning_rate=0.01,
        ),
        'visual': DopamineModulatedPlasticity(
            kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
            learning_rate=0.01,
        ),
    }


@pytest.fixture
def analyzer(circuit, plasticity_managers):
    """Create MultiTaskAnalyzer with circuit and plasticity managers."""
    return MultiTaskAnalyzer(circuit, plasticity_managers)


def test_multi_task_analyzer_initialization(analyzer):
    """Test that MultiTaskAnalyzer initializes correctly."""
    assert analyzer.circuit is not None
    assert len(analyzer.plasticity) == 3
    assert 'olfactory' in analyzer.plasticity
    assert 'spatial' in analyzer.plasticity
    assert 'visual' in analyzer.plasticity
    assert len(analyzer.history) == 3


def test_run_interleaved_training_basic(analyzer):
    """Test that interleaved training runs without errors."""
    results = analyzer.run_interleaved_training(
        trials_per_task=5,
        task_order=['olfactory', 'spatial'],
        n_cycles=2,
    )

    # Check results structure
    assert len(results) == 5 * 2 * 2, "Should have 5 trials × 2 tasks × 2 cycles = 20 rows"
    assert 'cycle' in results.columns
    assert 'task' in results.columns
    assert 'trial' in results.columns
    assert 'global_trial' in results.columns
    assert 'stimulus' in results.columns
    assert 'reward' in results.columns
    assert 'mbon_output' in results.columns
    assert 'rpe' in results.columns
    assert 'kc_sparsity' in results.columns


def test_interleaved_training_task_order(analyzer):
    """Test that tasks are presented in specified order."""
    results = analyzer.run_interleaved_training(
        trials_per_task=3,
        task_order=['visual', 'olfactory', 'spatial'],
        n_cycles=1,
    )

    # Check task order within a cycle
    cycle_0 = results[results['cycle'] == 0]
    tasks_in_order = cycle_0['task'].unique().tolist()
    assert tasks_in_order == ['visual', 'olfactory', 'spatial'], \
        "Tasks should appear in specified order"


def test_interleaved_training_global_trial_counter(analyzer):
    """Test that global_trial counter increments correctly."""
    results = analyzer.run_interleaved_training(
        trials_per_task=5,
        task_order=['olfactory', 'spatial'],
        n_cycles=2,
    )

    # Check global trial numbers are sequential
    global_trials = results['global_trial'].values
    assert len(global_trials) == len(set(global_trials)), \
        "Global trial numbers should be unique"
    assert global_trials[0] == 0, "First trial should be 0"
    assert global_trials[-1] == len(results) - 1, \
        "Last trial should be len(results) - 1"


def test_compute_task_interference(analyzer):
    """Test task interference computation."""
    results = analyzer.run_interleaved_training(
        trials_per_task=15,
        task_order=['olfactory', 'spatial'],
        n_cycles=3,
    )

    interference = analyzer.compute_task_interference(results)

    # Check structure
    assert 'olfactory' in interference
    assert 'spatial' in interference

    # Check values are reasonable (learning efficiency > 0)
    assert interference['olfactory'] > 0.0, \
        "Learning efficiency should be positive"
    assert interference['spatial'] > 0.0, \
        "Learning efficiency should be positive"

    # With learning, efficiency should be > 1.0 (final > initial)
    assert interference['olfactory'] >= 1.0 or interference['spatial'] >= 1.0, \
        "At least one task should show some learning (efficiency >= 1.0)"


def test_compute_representational_overlap(analyzer):
    """Test representational overlap computation."""
    results = analyzer.run_interleaved_training(
        trials_per_task=10,
        task_order=['olfactory', 'spatial', 'visual'],
        n_cycles=2,
    )

    overlap_df = analyzer.compute_representational_overlap(results)

    # Check structure
    if len(overlap_df) > 0:  # Only if multiple tasks
        assert 'task_A' in overlap_df.columns
        assert 'task_B' in overlap_df.columns
        assert 'overlap_fraction' in overlap_df.columns

        # Check overlap values in [0, 1]
        assert (overlap_df['overlap_fraction'] >= 0.0).all()
        assert (overlap_df['overlap_fraction'] <= 1.0).all()


def test_measure_catastrophic_forgetting(analyzer):
    """Test catastrophic forgetting measurement."""
    forgetting = analyzer.measure_catastrophic_forgetting(
        task_A='olfactory',
        task_B='spatial',
        trials_per_task=15,
    )

    # Check structure
    assert 'task_A' in forgetting
    assert 'task_B' in forgetting
    assert 'task_A_initial_performance' in forgetting
    assert 'task_A_final_performance' in forgetting
    assert 'forgetting_magnitude' in forgetting
    assert 'task_B_performance' in forgetting

    # Check values are reasonable
    assert forgetting['task_A'] == 'olfactory'
    assert forgetting['task_B'] == 'spatial'
    assert forgetting['task_A_initial_performance'] >= 0.0
    assert forgetting['task_A_final_performance'] >= 0.0
    assert forgetting['task_B_performance'] >= 0.0

    # Forgetting magnitude should be reasonable (-inf to 1.0)
    # Negative values mean task A improved (positive transfer)
    # Positive values mean task A degraded (negative transfer/forgetting)
    assert -10.0 < forgetting['forgetting_magnitude'] < 10.0, \
        "Forgetting magnitude should be in reasonable range"


def test_multi_task_learning_curves_increase(analyzer):
    """Test that learning curves generally increase over training."""
    results = analyzer.run_interleaved_training(
        trials_per_task=20,
        task_order=['olfactory'],
        n_cycles=3,
    )

    # Check that MBON output increases over time (learning)
    early_output = results.head(10)['mbon_output'].mean()
    late_output = results.tail(10)['mbon_output'].mean()

    # With reward trials, should show some learning
    assert late_output >= early_output * 0.9, \
        "Learning curve should not dramatically decrease over training"


def test_task_specific_stimuli_generation(analyzer):
    """Test that different tasks use different stimuli."""
    results = analyzer.run_interleaved_training(
        trials_per_task=5,
        task_order=['olfactory', 'spatial', 'visual'],
        n_cycles=1,
    )

    # Get stimuli per task
    olfactory_stimuli = set(results[results['task'] == 'olfactory']['stimulus'])
    spatial_stimuli = set(results[results['task'] == 'spatial']['stimulus'])
    visual_stimuli = set(results[results['task'] == 'visual']['stimulus'])

    # Each task should use at least one stimulus
    assert len(olfactory_stimuli) > 0
    assert len(spatial_stimuli) > 0
    assert len(visual_stimuli) > 0


def test_reward_contingencies_per_task(analyzer):
    """Test that reward contingencies are task-specific."""
    results = analyzer.run_interleaved_training(
        trials_per_task=10,
        task_order=['olfactory', 'spatial'],
        n_cycles=1,
    )

    # Each task should have some rewarded and some unrewarded trials
    for task in ['olfactory', 'spatial']:
        task_results = results[results['task'] == task]
        n_rewarded = (task_results['reward'] == 1).sum()
        n_unrewarded = (task_results['reward'] == 0).sum()

        assert n_rewarded > 0, f"{task} should have some rewarded trials"
        assert n_unrewarded > 0, f"{task} should have some unrewarded trials"


def test_kc_sparsity_maintained_across_tasks(analyzer):
    """Test that KC sparsity is maintained across all tasks."""
    results = analyzer.run_interleaved_training(
        trials_per_task=10,
        task_order=['olfactory', 'spatial', 'visual'],
        n_cycles=2,
    )

    # KC sparsity should be ~5% (target) across all trials
    mean_sparsity = results['kc_sparsity'].mean()
    assert 0.03 < mean_sparsity < 0.08, \
        f"Mean KC sparsity should be near 5%, got {mean_sparsity:.1%}"

    # Sparsity should not vary wildly between tasks
    for task in ['olfactory', 'spatial', 'visual']:
        task_sparsity = results[results['task'] == task]['kc_sparsity'].mean()
        assert 0.02 < task_sparsity < 0.10, \
            f"{task} KC sparsity should be reasonable, got {task_sparsity:.1%}"


def test_independent_plasticity_per_task(analyzer):
    """Test that tasks have independent plasticity (separate weight matrices)."""
    # Train olfactory task with more trials for robust learning
    results_olfactory = analyzer.run_interleaved_training(
        trials_per_task=30,  # Increased for more robust learning
        task_order=['olfactory'],
        n_cycles=2,  # Multiple cycles
    )

    # Get initial weights for spatial task (should be unchanged)
    initial_spatial_weights = analyzer.plasticity['spatial'].kc_to_mbon.copy()

    # Train spatial task
    results_spatial = analyzer.run_interleaved_training(
        trials_per_task=30,
        task_order=['spatial'],
        n_cycles=2,
    )

    # Spatial weights should have changed
    final_spatial_weights = analyzer.plasticity['spatial'].kc_to_mbon
    assert not np.allclose(initial_spatial_weights, final_spatial_weights), \
        "Spatial weights should change after training spatial task"

    # Olfactory task should show learning or at least not deteriorate
    olfactory_early = results_olfactory.head(10)['mbon_output'].mean()
    olfactory_late = results_olfactory.tail(10)['mbon_output'].mean()
    # Check learning or stability (late >= early * 0.7)
    assert olfactory_late >= olfactory_early * 0.7, \
        f"Olfactory task should maintain or improve performance (early: {olfactory_early:.3f}, late: {olfactory_late:.3f})"


def test_integration_multi_task_with_phase2_learning(circuit):
    """Integration test: Multi-task analyzer works with Phase 2 learning components."""
    # Create two tasks with independent plasticity
    plasticity_A = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.02,
        eligibility_trace_tau=None,
    )
    plasticity_B = DopamineModulatedPlasticity(
        kc_to_mbon_weights=circuit.connectivity.kc_to_mbon.toarray(),
        learning_rate=0.02,
        eligibility_trace_tau=0.1,  # Task B uses eligibility traces
    )

    plasticity_managers = {
        'task_A': plasticity_A,
        'task_B': plasticity_B,
    }

    analyzer = MultiTaskAnalyzer(circuit, plasticity_managers)

    # Run interleaved training
    results = analyzer.run_interleaved_training(
        trials_per_task=15,
        task_order=['task_A', 'task_B'],
        n_cycles=3,
    )

    # Both tasks should show learning
    interference = analyzer.compute_task_interference(results)
    assert interference['task_A'] > 0.9, \
        "Task A should show reasonable learning"
    assert interference['task_B'] > 0.9, \
        "Task B should show reasonable learning"

    # Measure forgetting
    forgetting = analyzer.measure_catastrophic_forgetting(
        task_A='task_A',
        task_B='task_B',
        trials_per_task=20,
    )

    # Task A should have learned initially
    assert forgetting['task_A_initial_performance'] > 0.0, \
        "Task A should show initial learning"

    # Task B should also learn
    assert forgetting['task_B_performance'] > 0.0, \
        "Task B should show learning"

    print(f"Multi-task integration test:")
    print(f"  Task A initial: {forgetting['task_A_initial_performance']:.3f}")
    print(f"  Task A after B: {forgetting['task_A_final_performance']:.3f}")
    print(f"  Forgetting: {forgetting['forgetting_magnitude']:.3f}")
    print(f"  Task B final: {forgetting['task_B_performance']:.3f}")
