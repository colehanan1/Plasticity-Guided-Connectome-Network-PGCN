"""Tests for Shapley blocking analysis."""

import numpy as np
import pytest
import scipy.sparse as sp

from pgcn.experiments.experiment_6_shapley_analysis import ShapleyBlockingAnalysis
from pgcn.models.connectivity_matrix import ConnectivityMatrix
from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit


@pytest.fixture
def shapley_experiment():
    """Create Shapley analysis experiment."""
    n_pn, n_kc, n_mbon = 10, 20, 5
    pn_ids = np.arange(1000, 1000 + n_pn, dtype=np.int64)
    kc_ids = np.arange(2000, 2000 + n_kc, dtype=np.int64)
    mbon_ids = np.arange(3000, 3000 + n_mbon, dtype=np.int64)
    dan_ids = np.arange(4000, 4008, dtype=np.int64)

    pn_to_kc = sp.random(n_kc, n_pn, density=0.3, format="csr", dtype=np.float64)
    kc_to_mbon = sp.random(n_mbon, n_kc, density=0.4, format="csr", dtype=np.float64)

    pn_glomeruli = {pn_id: f"glom_{i % 3}" for i, pn_id in enumerate(pn_ids)}

    conn = ConnectivityMatrix(
        pn_ids=pn_ids, kc_ids=kc_ids, mbon_ids=mbon_ids, dan_ids=dan_ids,
        pn_to_kc=pn_to_kc, kc_to_mbon=kc_to_mbon,
        dan_to_kc=sp.csr_matrix((n_kc, 8)), dan_to_mbon=sp.csr_matrix((n_mbon, 8)),
        pn_glomeruli=pn_glomeruli, kc_subtypes={kc: "ab" for kc in kc_ids},
    )

    circuit = OlfactoryCircuit(conn, kc_sparsity_target=0.20)
    plasticity = DopamineModulatedPlasticity(conn.kc_to_mbon.toarray(), learning_rate=0.05)
    return ShapleyBlockingAnalysis(circuit, plasticity)


def test_compute_shapley_contribution(shapley_experiment):
    """Test Shapley value computation."""
    dataset = [{"odor": "glom_0", "reward": 1.0}] * 5
    shapley_value = shapley_experiment.compute_shapley_contribution(kc_idx=5, dataset=dataset, n_permutations=3)

    assert isinstance(shapley_value, float)
    assert np.isfinite(shapley_value)


def test_identify_top_blockers(shapley_experiment):
    """Test identifying top blocker KCs."""
    dataset = [{"odor": "glom_0", "reward": 1.0}] * 5
    top_blockers = shapley_experiment.identify_top_blockers(dataset, k=3, n_permutations=3)

    assert len(top_blockers) == 3
    assert all(isinstance(kc_idx, (int, np.integer)) for kc_idx, _ in top_blockers)
    assert all(isinstance(shapley_val, float) for _, shapley_val in top_blockers)


def test_edit_blockers_prune(shapley_experiment):
    """Test prune edit mode."""
    top_blockers = [(5, -0.5), (10, -0.3)]
    edited_conn = shapley_experiment.edit_blockers(top_blockers, edit_mode="prune")

    assert edited_conn is not None


def test_measure_recovery(shapley_experiment):
    """Test recovery metric computation."""
    recovery = shapley_experiment.measure_recovery(original_learning_rate=0.01, edited_learning_rate=0.02)

    assert isinstance(recovery, float)
    assert np.isclose(recovery, 2.0, rtol=1e-3)  # 0.02 / 0.01 with tolerance
