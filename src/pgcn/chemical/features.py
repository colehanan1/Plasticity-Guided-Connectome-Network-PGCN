"""Feature engineering helpers for chemical descriptors."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - handled gracefully
    torch = None  # type: ignore[assignment]

from .mappings import CHEMICAL_PROPERTIES

# Derive the canonical functional group ordering once at import time.
_FUNCTIONAL_GROUPS: List[str] = sorted(
    {group for props in CHEMICAL_PROPERTIES.values() for group in props["functional_groups"]}
)

CHEMICAL_FEATURE_NAMES: Sequence[str] = (
    "molecular_weight",
    "carbon_length",
    "boiling_point",
    *_FUNCTIONAL_GROUPS,
)


def _functional_group_vector(groups: Iterable[str]) -> np.ndarray:
    group_set = {g.lower() for g in groups}
    return np.array([1.0 if group in group_set else 0.0 for group in _FUNCTIONAL_GROUPS], dtype=float)


def get_chemical_features(odor: str, *, as_tensor: bool | None = None):
    """Return a feature vector encoding key physicochemical attributes."""

    odor_key = odor.lower()
    if odor_key not in CHEMICAL_PROPERTIES:
        raise KeyError(f"Unknown odor '{odor}'. Known odors: {sorted(CHEMICAL_PROPERTIES)}")

    props = CHEMICAL_PROPERTIES[odor_key]
    vector = np.array(
        [
            float(props["molecular_weight"]),
            float(props["carbon_length"]),
            float(props["boiling_point"]),
        ],
        dtype=float,
    )
    vector = np.concatenate([vector, _functional_group_vector(props["functional_groups"])])

    if as_tensor is False or (as_tensor is None and torch is None):
        return vector
    if torch is None:
        raise ImportError("PyTorch is required for tensor features but is not installed.")
    return torch.tensor(vector, dtype=torch.float32)


__all__ = ["CHEMICAL_FEATURE_NAMES", "get_chemical_features"]
