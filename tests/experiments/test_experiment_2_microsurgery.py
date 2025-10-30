"""Tests for counterfactual microsurgery experiment."""

import numpy as np
import pytest
import scipy.sparse as sp

from pgcn.experiments.experiment_1_veto_gate import VetoGateExperiment
from pgcn.experiments.experiment_2_counterfactual_microsurgery import CounterfactualMicrosurgeryExperiment
from pgcn.models.connectivity_matrix import ConnectivityMatrix
from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.models.olfactory_circuit import OlfactoryCircuit


@pytest.fixture
def surgery_experiment():
    """Create microsurgery experiment with baseline veto."""
    # Create small circuit
    n_pn, n_kc, n_mbon = 10, 20, 5
    pn_ids = np.arange(1000, 1000 + n_pn, dtype=np.int64)
    kc_ids = np.arange(2000, 2000 + n_kc, dtype=np.int64)
    mbon_ids = np.arange(3000, 3000 + n_mbon, dtype=np.int64)
    dan_ids = np.arange(4000, 4008, dtype=np.int64)

    pn_to_kc = sp.random(n_kc, n_pn, density=0.3, format="csr", dtype=np.float64)
    kc_to_mbon = sp.random(n_mbon, n_kc, density=0.4, format="csr", dtype=np.float64)

    pn_glomeruli = {pn_id: "glom_A" if i < 3 else "glom_B" for i, pn_id in enumerate(pn_ids)}

    conn = ConnectivityMatrix(
        pn_ids=pn_ids, kc_ids=kc_ids, mbon_ids=mbon_ids, dan_ids=dan_ids,
        pn_to_kc=pn_to_kc, kc_to_mbon=kc_to_mbon,
        dan_to_kc=sp.csr_matrix((n_kc, 8)), dan_to_mbon=sp.csr_matrix((n_mbon, 8)),
        pn_glomeruli=pn_glomeruli, kc_subtypes={kc: "ab" for kc in kc_ids},
    )

    circuit = OlfactoryCircuit(conn, kc_sparsity_target=0.20)
    plasticity = DopamineModulatedPlasticity(conn.kc_to_mbon.toarray(), learning_rate=0.05)
    veto_exp = VetoGateExperiment(circuit, plasticity, "glom_A")

    return CounterfactualMicrosurgeryExperiment(veto_exp, "glom_A")


def test_variant_i_ablation(surgery_experiment):
    """Test PN→KC ablation variant."""
    results = surgery_experiment.variant_i_ablate_pn_inputs(n_trials=5)

    assert results["variant"] == "ablate_pn_inputs"
    assert "results" in results
    assert "blocking_index" in results
    assert np.isfinite(results["blocking_index"])


def test_variant_ii_freezing(surgery_experiment):
    """Test synapse freezing variant."""
    results = surgery_experiment.variant_ii_freeze_veto_synapses(n_trials=5)

    assert results["variant"] == "freeze_veto_synapses"
    assert "frozen_synapses" in results
    assert results["n_frozen"] > 0


def test_variant_iii_sign_flip(surgery_experiment):
    """Test dopamine sign-flip variant."""
    results = surgery_experiment.variant_iii_sign_flip_dopamine(n_trials=5)

    assert results["variant"] == "sign_flip_dopamine"
    assert "sign_flip_synapses" in results
    assert results["n_flipped"] > 0


def test_run_all_variants(surgery_experiment):
    """Test running all three variants."""
    all_results = surgery_experiment.run_all_variants(n_trials_per_variant=5)

    assert "ablate" in all_results
    assert "freeze" in all_results
    assert "sign_flip" in all_results

    for variant_results in all_results.values():
        assert "recovery_metric" in variant_results
        assert np.isfinite(variant_results["recovery_metric"])
