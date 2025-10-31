"""Tests for eligibility trace experiment."""

import numpy as np
import pytest
import scipy.sparse as sp

from pgcn.experiments.experiment_3_eligibility_traces import EligibilityTraceExperiment
from pgcn.models.connectivity_matrix import ConnectivityMatrix
from pgcn.models.olfactory_circuit import OlfactoryCircuit


@pytest.fixture
def eligibility_experiment():
    """Create eligibility trace experiment."""
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
    return EligibilityTraceExperiment(circuit, eligibility_tau=0.1)


def test_phase_1_training(eligibility_experiment):
    """Test Phase 1 training."""
    results = eligibility_experiment.run_phase_1_training(odor="glom_0", n_trials=10)

    assert len(results) == 10
    assert "mbon_valence" in results.columns
    assert "rpe" in results.columns


def test_phase_2_comparison(eligibility_experiment):
    """Test Phase 2 comparison."""
    results = eligibility_experiment.run_phase_2_comparison(odor_b="glom_1", n_trials=10)

    assert "control" in results
    assert "eligibility_trace" in results

    for method_results in results.values():
        assert len(method_results) == 10
