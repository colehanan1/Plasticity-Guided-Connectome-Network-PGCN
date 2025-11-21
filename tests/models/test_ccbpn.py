"""Tests for Connectome-Constrained Behavioral Prediction Network (CCBPN).

This test suite validates:
1. CCBPN initialization with FlyWire connectivity
2. Forward pass with correct tensor shapes
3. Connectivity mask enforcement during training
4. Behavioral task loss computation
5. Neuron selectivity predictions
6. Integration with behavioral data loaders

Tests follow PGCN conventions and ensure backward compatibility.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import CCBPN components
from pgcn.models.ccbpn import (
    ConnectomeConstrainedBehavioralPredictor,
    BehavioralTaskLoss,
    CCBPNConfig,
)


@pytest.fixture
def mock_cache_dir(tmp_path):
    """Create mock cache directory with minimal FlyWire data."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create minimal nodes.parquet
    import pandas as pd

    nodes = pd.DataFrame({
        'node_id': list(range(200)),  # 50 PNs + 100 KCs + 40 MBONs + 10 DANs
        'type': ['PN'] * 50 + ['KC'] * 100 + ['MBON'] * 40 + ['DAN'] * 10
    })
    nodes.to_parquet(cache_dir / "nodes.parquet")

    # Create minimal edges.parquet (PN→KC, KC→MBON)
    edges = pd.DataFrame({
        'source_id': [0] * 100 + [50] * 40,  # PN 0 → KCs 50-149, KC 50 → MBONs 150-189
        'target_id': list(range(50, 150)) + list(range(150, 190)),
        'synapse_weight': [1.0] * 140
    })
    edges.to_parquet(cache_dir / "edges.parquet")

    # Create minimal dan_edges.parquet
    dan_edges = pd.DataFrame({
        'source_id': [190] * 10,  # DAN 190 → KCs 50-59
        'target_id': list(range(50, 60)),
        'synapse_weight': [1.0] * 10
    })
    dan_edges.to_parquet(cache_dir / "dan_edges.parquet")

    # Create minimal KC subtype files
    for subtype in ['ab', 'g_main']:
        kc_df = pd.DataFrame({
            'root_id': list(range(50, 100)) if subtype == 'ab' else list(range(100, 150)),
            'subtype': [subtype] * 50
        })
        kc_df.to_csv(cache_dir / f"kc_{subtype}.csv", index=False)

    # Create minimal PN glomerulus file
    alpn = pd.DataFrame({
        'root_id': list(range(50)),
        'primary_glomerulus': [f"glom_{i}" for i in range(50)]
    })
    alpn.to_csv(cache_dir / "alpn_extracted.csv", index=False)

    # Create minimal MBON/DAN metadata
    mbon = pd.DataFrame({
        'root_id': list(range(150, 190)),
        'input_neuropils': ["MB(calyx)"] * 40
    })
    mbon.to_csv(cache_dir / "mbon_all.csv", index=False)

    dan = pd.DataFrame({
        'root_id': list(range(190, 200)),
        'output_neuropils': ["MB(γ)"] * 10
    })
    dan.to_csv(cache_dir / "dan_mb.csv", index=False)

    return cache_dir


class TestCCBPNInitialization:
    """Test CCBPN model initialization."""

    def test_init_loads_connectivity(self, mock_cache_dir):
        """Test that CCBPN loads FlyWire connectivity on init."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir,
            behavioral_task="odor_discrimination"
        )

        assert model.n_pn == 50
        assert model.n_kc == 100
        assert model.n_mbon == 40
        assert model.n_dan == 10

    def test_init_creates_connectivity_masks(self, mock_cache_dir):
        """Test that connectivity masks are registered as buffers."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        # Check masks exist and are buffers (not parameters)
        assert hasattr(model, 'pn_kc_mask')
        assert hasattr(model, 'kc_mbon_mask')
        assert isinstance(model.pn_kc_mask, torch.Tensor)
        assert isinstance(model.kc_mbon_mask, torch.Tensor)

        # Check masks are binary
        assert torch.all((model.pn_kc_mask == 0) | (model.pn_kc_mask == 1))
        assert torch.all((model.kc_mbon_mask == 0) | (model.kc_mbon_mask == 1))

    def test_init_creates_trainable_weights(self, mock_cache_dir):
        """Test that synaptic weights are trainable parameters."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        # Check weights exist and require gradients
        assert hasattr(model, 'pn_kc_weights')
        assert hasattr(model, 'kc_mbon_weights')
        assert model.pn_kc_weights.requires_grad
        assert model.kc_mbon_weights.requires_grad

    def test_init_invalid_cache_dir(self):
        """Test that invalid cache dir raises error."""
        with pytest.raises(FileNotFoundError, match="Cache directory not found"):
            ConnectomeConstrainedBehavioralPredictor(
                cache_dir="/nonexistent/path"
            )

    def test_init_invalid_task(self, mock_cache_dir):
        """Test that invalid task type raises error."""
        with pytest.raises(ValueError, match="behavioral_task must be one of"):
            ConnectomeConstrainedBehavioralPredictor(
                cache_dir=mock_cache_dir,
                behavioral_task="invalid_task"
            )

    def test_init_invalid_sparsity(self, mock_cache_dir):
        """Test that invalid KC sparsity raises error."""
        with pytest.raises(ValueError, match="kc_sparsity_target must be in"):
            ConnectomeConstrainedBehavioralPredictor(
                cache_dir=mock_cache_dir,
                kc_sparsity_target=1.5  # Invalid: > 1.0
            )


class TestCCBPNForwardPass:
    """Test CCBPN forward pass."""

    def test_forward_correct_shapes(self, mock_cache_dir):
        """Test forward pass produces correct output shapes."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        batch_size = 4
        time_steps = 50

        # Create test inputs
        odor_seq = torch.randn(batch_size, time_steps, model.n_pn)
        dopa_sig = torch.zeros(batch_size, time_steps)

        # Forward pass
        outputs = model(odor_seq, dopa_sig, return_intermediates=True)

        # Check output shapes
        assert outputs['mbon_activity'].shape == (batch_size, time_steps, model.n_mbon)
        assert outputs['behavioral_output'].shape == (batch_size, time_steps)
        assert outputs['kc_activity'].shape == (batch_size, time_steps, model.n_kc)
        assert outputs['sparsity_fraction'].shape == (batch_size, time_steps)

    def test_forward_kc_sparsity_enforced(self, mock_cache_dir):
        """Test that KC sparsity is enforced (~5% active)."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir,
            kc_sparsity_target=0.05
        )

        batch_size = 8
        time_steps = 20

        odor_seq = torch.randn(batch_size, time_steps, model.n_pn)
        dopa_sig = torch.zeros(batch_size, time_steps)

        outputs = model(odor_seq, dopa_sig, return_intermediates=True)

        # Check sparsity is approximately 5%
        mean_sparsity = outputs['sparsity_fraction'].mean().item()
        assert 0.04 <= mean_sparsity <= 0.06, f"KC sparsity {mean_sparsity} not near 0.05"

    def test_forward_invalid_input_shapes(self, mock_cache_dir):
        """Test that invalid input shapes raise errors."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        # Wrong number of dimensions
        odor_seq_2d = torch.randn(4, model.n_pn)
        dopa_sig = torch.zeros(4, 50)

        with pytest.raises(ValueError, match="must be 3D"):
            model(odor_seq_2d, dopa_sig)

        # Wrong PN dimension
        odor_seq_wrong_pn = torch.randn(4, 50, 999)
        with pytest.raises(ValueError, match="must match n_pn"):
            model(odor_seq_wrong_pn, dopa_sig)

    def test_forward_behavioral_output_in_range(self, mock_cache_dir):
        """Test that behavioral output is in [0, 1] (sigmoid output)."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        odor_seq = torch.randn(2, 10, model.n_pn)
        dopa_sig = torch.zeros(2, 10)

        outputs = model(odor_seq, dopa_sig)
        behavioral_out = outputs['behavioral_output']

        assert torch.all(behavioral_out >= 0.0)
        assert torch.all(behavioral_out <= 1.0)


class TestConnectivityConstraintEnforcement:
    """Test that connectivity constraints are maintained during training."""

    def test_enforce_connectivity_constraints(self, mock_cache_dir):
        """Test that connectivity masks zero out unconnected weights."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        # Modify weights to violate connectivity (set all to 1.0)
        with torch.no_grad():
            model.pn_kc_weights.fill_(1.0)
            model.kc_mbon_weights.fill_(1.0)

        # Enforce constraints
        model.enforce_connectivity_constraints()

        # Check that only connected synapses have non-zero weights
        assert torch.all((model.pn_kc_weights == 0) | (model.pn_kc_mask == 1))
        assert torch.all((model.kc_mbon_weights == 0) | (model.kc_mbon_mask == 1))

    def test_gradient_descent_preserves_topology(self, mock_cache_dir):
        """Test that gradient descent doesn't create new connections."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        # Store initial connectivity pattern
        initial_pn_kc_connected = (model.pn_kc_mask == 1)
        initial_kc_mbon_connected = (model.kc_mbon_mask == 1)

        # Simulate training step
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        odor_seq = torch.randn(2, 10, model.n_pn)
        dopa_sig = torch.zeros(2, 10)
        labels = torch.ones(2)

        outputs = model(odor_seq, dopa_sig)
        loss = nn.BCELoss()(outputs['behavioral_output'][:, -1], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Enforce connectivity constraints
        model.enforce_connectivity_constraints()

        # Check topology unchanged
        final_pn_kc_connected = (model.pn_kc_weights != 0)
        final_kc_mbon_connected = (model.kc_mbon_weights != 0)

        # Connected synapses should remain subset of initial connectivity
        assert torch.all(final_pn_kc_connected <= initial_pn_kc_connected)
        assert torch.all(final_kc_mbon_connected <= initial_kc_mbon_connected)


class TestBehavioralTaskLoss:
    """Test behavioral task loss function."""

    def test_discrimination_loss_basic(self):
        """Test basic discrimination loss computation."""
        loss_fn = BehavioralTaskLoss(task_type="odor_discrimination")

        # Perfect predictions
        predicted = torch.tensor([0.9, 0.1, 0.8, 0.2])
        observed = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = loss_fn(predicted, observed)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0.0  # Loss should be positive
        assert loss.item() < 1.0  # Good predictions → low loss

    def test_loss_temporal_sequences(self):
        """Test loss with temporal prediction sequences."""
        loss_fn = BehavioralTaskLoss(task_type="odor_discrimination")

        # Temporal predictions (batch=2, time=10)
        predicted = torch.rand(2, 10)
        observed = torch.tensor([1.0, 0.0])

        loss = loss_fn(predicted, observed)  # Should use final timestep

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

    def test_invalid_task_type(self):
        """Test that invalid task type raises error."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            BehavioralTaskLoss(task_type="invalid_task")


class TestNeuronSelectivity:
    """Test neuron selectivity prediction methods."""

    def test_get_neuron_selectivity_kc(self, mock_cache_dir):
        """Test KC odor selectivity prediction."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        # Create test odors
        n_odors = 10
        test_odors = torch.randn(n_odors, model.n_pn)

        # Get KC tuning curves
        kc_tuning = model.get_neuron_selectivity(test_odors, neuron_type='KC')

        # Check output structure
        assert isinstance(kc_tuning, dict)
        assert len(kc_tuning) == model.n_kc

        # Check each tuning curve has correct shape
        for neuron_id, tuning_curve in kc_tuning.items():
            assert tuning_curve.shape == (n_odors,)
            assert isinstance(neuron_id, int)
            assert 0 <= neuron_id < model.n_kc

    def test_get_neuron_selectivity_mbon(self, mock_cache_dir):
        """Test MBON odor selectivity prediction."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        n_odors = 15
        test_odors = torch.randn(n_odors, model.n_pn)

        mbon_tuning = model.get_neuron_selectivity(test_odors, neuron_type='MBON')

        assert isinstance(mbon_tuning, dict)
        assert len(mbon_tuning) == model.n_mbon

    def test_selectivity_invalid_neuron_type(self, mock_cache_dir):
        """Test that invalid neuron type raises error."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        test_odors = torch.randn(5, model.n_pn)

        with pytest.raises(ValueError, match="neuron_type must be"):
            model.get_neuron_selectivity(test_odors, neuron_type='INVALID')


class TestCCBPNIntegration:
    """Integration tests for CCBPN with behavioral data."""

    def test_end_to_end_training_step(self, mock_cache_dir):
        """Test complete training step (forward + backward + enforce)."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BehavioralTaskLoss(task_type="odor_discrimination")

        # Create synthetic batch
        batch_size = 4
        sequence_length = 20

        odor_seq = torch.randn(batch_size, sequence_length, model.n_pn)
        dopa_sig = torch.zeros(batch_size, sequence_length)
        labels = torch.randint(0, 2, (batch_size,)).float()

        # Training step
        model.train()
        optimizer.zero_grad()

        outputs = model(odor_seq, dopa_sig)
        loss = loss_fn(outputs['behavioral_output'], labels)

        loss.backward()
        optimizer.step()

        # Enforce connectivity constraints
        model.enforce_connectivity_constraints()

        # Check that loss is reasonable
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0.0

    def test_model_checkpoint_save_load(self, mock_cache_dir, tmp_path):
        """Test saving and loading model checkpoints."""
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': {'task': 'odor_discrimination', 'kc_sparsity': 0.05},
        }, checkpoint_path)

        # Load into new model
        checkpoint = torch.load(checkpoint_path)
        model_loaded = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=mock_cache_dir
        )
        model_loaded.load_state_dict(checkpoint['model_state_dict'])

        # Check weights match
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), model_loaded.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)


class TestBackwardCompatibility:
    """Test backward compatibility with existing PGCN modules."""

    def test_no_import_conflicts(self):
        """Test that CCBPN imports don't break existing modules."""
        # Import existing modules
        from pgcn.models.olfactory_circuit import OlfactoryCircuit
        from pgcn.models.reservoir import DrosophilaReservoir
        from pgcn.models.multi_task_model import MultiTaskDrosophilaModel

        # Should not raise any errors
        assert OlfactoryCircuit is not None
        assert DrosophilaReservoir is not None
        assert MultiTaskDrosophilaModel is not None

    def test_ccbpn_coexists_with_other_models(self, mock_cache_dir):
        """Test that CCBPN can coexist with other PGCN models."""
        from pgcn.models.ccbpn import ConnectomeConstrainedBehavioralPredictor

        # Create CCBPN
        ccbpn = ConnectomeConstrainedBehavioralPredictor(cache_dir=mock_cache_dir)

        # Both should work
        assert ccbpn is not None
        assert ccbpn.n_pn == 50
