"""Similarity utilities combining chemical descriptors and empirical priors."""

from __future__ import annotations

from typing import Dict, Tuple

from .mappings import (
    CHEMICAL_PROPERTIES,
    CHEMICAL_SIMILARITY_MATRIX,
    canonicalise_odor_name,
)

_DEFAULT_DIRECT_SIMILARITY = 0.2


def _lookup_direct_similarity(training_odor: str, test_odor: str) -> float:
    key = (training_odor, test_odor)
    if key in CHEMICAL_SIMILARITY_MATRIX:
        return float(CHEMICAL_SIMILARITY_MATRIX[key])
    reversed_key = (test_odor, training_odor)
    if reversed_key in CHEMICAL_SIMILARITY_MATRIX:
        return float(CHEMICAL_SIMILARITY_MATRIX[reversed_key])
    wildcard_key = (training_odor, "*")
    if wildcard_key in CHEMICAL_SIMILARITY_MATRIX:
        return float(CHEMICAL_SIMILARITY_MATRIX[wildcard_key])
    wildcard_key = (test_odor, "*")
    if wildcard_key in CHEMICAL_SIMILARITY_MATRIX:
        return float(CHEMICAL_SIMILARITY_MATRIX[wildcard_key])
    return _DEFAULT_DIRECT_SIMILARITY


def compute_chemical_similarity_constraint(training_odor: str, test_odor: str) -> Dict[str, float]:
    """Combine structural and functional similarity into learning constraints."""

    train_key = canonicalise_odor_name(training_odor)
    test_key = canonicalise_odor_name(test_odor)

    direct_similarity = _lookup_direct_similarity(train_key, test_key)

    train_groups = set(CHEMICAL_PROPERTIES[train_key]["functional_groups"])
    test_groups = set(CHEMICAL_PROPERTIES[test_key]["functional_groups"])
    union = train_groups.union(test_groups)
    functional_similarity = len(train_groups.intersection(test_groups)) / len(union) if union else 0.0

    mw_train = float(CHEMICAL_PROPERTIES[train_key]["molecular_weight"])
    mw_test = float(CHEMICAL_PROPERTIES[test_key]["molecular_weight"])
    mw_similarity = 1.0 - abs(mw_train - mw_test) / max(mw_train, mw_test)

    overall_similarity = (
        direct_similarity * 0.5 + functional_similarity * 0.3 + mw_similarity * 0.2
    )

    overall_similarity = max(0.0, min(1.0, overall_similarity))

    return {
        "learning_rate_modifier": overall_similarity,
        "expected_generalization": overall_similarity,
        "plasticity_constraint": 1.0 - overall_similarity,
    }


__all__ = ["compute_chemical_similarity_constraint"]
