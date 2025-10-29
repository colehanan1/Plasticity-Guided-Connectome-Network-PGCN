"""Connectivity helpers for building PN→KC matrices from local FlyWire data."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd
from scipy import sparse

__all__ = [
    "filter_mushroom_body_connections",
    "select_kc_pn_connections",
    "build_kc_pn_matrix",
]

DEFAULT_MUSHROOM_BODY_REGIONS = {
    "MB",  # generic annotation
    "MB_CA",  # calyx
    "MB_CALYX",
    "MB_PEDUNCLE",
    "MB_MEDIAL_LOBE",
    "MB_LATERAL_LOBE",
    "MB_CLAW",
}


def filter_mushroom_body_connections(
    connections_df: pd.DataFrame,
    *,
    neuropil_column: str = "neuropil",
    regions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return connections occurring within the mushroom body neuropils."""

    frame = connections_df.copy()
    if regions is None:
        mask = frame[neuropil_column].astype(str).str.contains("MB|mushroom", case=False, na=False)
        return frame.loc[mask].reset_index(drop=True)

    target = {region.lower() for region in regions}
    mask = frame[neuropil_column].astype(str).str.lower().isin(target)
    return frame.loc[mask].reset_index(drop=True)


def select_kc_pn_connections(
    connections_df: pd.DataFrame,
    *,
    kc_ids: Iterable[int],
    pn_ids: Iterable[int],
    min_synapses: int = 5,
) -> pd.DataFrame:
    """Filter ``connections_df`` to only KC⇄PN motifs."""

    kc_set = {int(node) for node in kc_ids}
    pn_set = {int(node) for node in pn_ids}

    mask = (
        connections_df["pre_root_id"].isin(pn_set)
        & connections_df["post_root_id"].isin(kc_set)
        & (connections_df["syn_count"] >= int(min_synapses))
    )
    return connections_df.loc[mask].reset_index(drop=True)


def build_kc_pn_matrix(
    connections_df: pd.DataFrame,
    kc_ids: Sequence[int],
    pn_ids: Sequence[int],
    *,
    weight_column: str = "syn_count",
) -> sparse.csr_matrix:
    """Create a sparse PN→KC connectivity matrix indexed by ``kc_ids`` and ``pn_ids``."""

    if weight_column not in connections_df.columns:
        raise ValueError(f"Connections frame missing '{weight_column}' column.")

    kc_index: Mapping[int, int] = {int(node): idx for idx, node in enumerate(kc_ids)}
    pn_index: Mapping[int, int] = {int(node): idx for idx, node in enumerate(pn_ids)}

    sub = select_kc_pn_connections(connections_df, kc_ids=kc_index.keys(), pn_ids=pn_index.keys(), min_synapses=0)
    if sub.empty:
        return sparse.csr_matrix((len(kc_index), len(pn_index)), dtype=float)

    rows = sub["post_root_id"].map(kc_index)
    cols = sub["pre_root_id"].map(pn_index)
    data = sub[weight_column].astype(float)

    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(kc_index), len(pn_index)), dtype=float)
    return matrix.tocsr()
