"""Tests for chemical feature utilities and similarity constraints."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pgcn.chemical import (
    CHEMICAL_FEATURE_NAMES,
    COMPLETE_ODOR_MAPPINGS,
    compute_chemical_similarity_constraint,
    get_chemical_features,
)


def test_complete_odor_mappings_include_training_conditions():
    assert set(COMPLETE_ODOR_MAPPINGS) == {"opto_EB", "opto_benz_1", "opto_hex", "hex_control"}
    assert COMPLETE_ODOR_MAPPINGS["opto_EB"]["testing_2"] == "ethyl_butyrate"


def test_get_chemical_features_returns_expected_vector():
    features = get_chemical_features("ethyl_butyrate", as_tensor=False)
    assert isinstance(features, np.ndarray)
    assert features.shape == (len(CHEMICAL_FEATURE_NAMES),)
    # Molecular descriptors occupy the first three entries
    np.testing.assert_allclose(features[:3], np.array([116.16, 6.0, 121.0]))
    # Ester functional group should be active
    ester_index = CHEMICAL_FEATURE_NAMES.index("ester")
    assert math.isclose(features[ester_index], 1.0)
    alcohol_index = CHEMICAL_FEATURE_NAMES.index("alcohol")
    assert math.isclose(features[alcohol_index], 0.0)


def test_similarity_constraint_balances_components():
    constraint = compute_chemical_similarity_constraint("ethyl_butyrate", "hexanol")
    expected_mw_similarity = 1.0 - abs(116.16 - 102.17) / 116.16
    expected_overall = 0.3 * 0.5 + 0.0 * 0.3 + expected_mw_similarity * 0.2
    assert math.isclose(constraint["learning_rate_modifier"], expected_overall, rel_tol=1e-5)
    assert math.isclose(constraint["expected_generalization"], expected_overall, rel_tol=1e-5)
    assert math.isclose(constraint["plasticity_constraint"], 1.0 - expected_overall, rel_tol=1e-5)


def test_similarity_uses_wildcard_for_acid():
    constraint = compute_chemical_similarity_constraint("apple_cider_vinegar", "hexanol")
    assert math.isclose(constraint["learning_rate_modifier"], constraint["expected_generalization"])
    assert constraint["learning_rate_modifier"] < 0.2  # Very weak similarity


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
