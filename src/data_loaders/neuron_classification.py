"""Neuron classification utilities for FlyWire FAFB datasets."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from utils.data_validation import validate_dataframe_columns

__all__ = [
    "get_kc_neurons",
    "get_pn_neurons",
    "get_mbon_neurons",
    "get_dan_neurons",
    "extract_neurotransmitter_info",
    "map_brain_regions",
]

_KC_KEYWORDS = ("kenyon", "kc", "mushroom body intrinsic")
_PN_KEYWORDS = ("projection", "pn", "olfactory", "alpn")
_MBON_KEYWORDS = (
    "mbon",
    "mushroom body output",
    "mbona",
    "mbon-",
)
_DAN_KEYWORDS = (
    "pam",
    "ppl",
    "dopamin",
    "dan",
    "dopaminergic",
    "octopamin",
)


def _merge_classification(
    cell_types_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> pd.DataFrame:
    validate_dataframe_columns(cell_types_df, ["root_id"], frame_name="cell_types")
    validate_dataframe_columns(classification_df, ["root_id"], frame_name="classification")
    merged = cell_types_df.merge(classification_df, on="root_id", how="left", suffixes=("", "_classification"))
    return merged


def _keyword_mask(series: pd.Series | None, keywords: Sequence[str]) -> pd.Series:
    if series is None:
        series = pd.Series(dtype="string")
    return series.astype(str).str.contains("|".join(keywords), case=False, na=False)


def get_kc_neurons(cell_types_df: pd.DataFrame, classification_df: pd.DataFrame) -> pd.DataFrame:
    """Return Kenyon cell annotations with merged classification metadata."""

    merged = _merge_classification(cell_types_df, classification_df)
    mask = (
        _keyword_mask(merged.get("cell_type"), _KC_KEYWORDS)
        | _keyword_mask(merged.get("cell_type_aliases"), _KC_KEYWORDS)
        | _keyword_mask(merged.get("super_class"), _KC_KEYWORDS)
        | _keyword_mask(merged.get("class"), _KC_KEYWORDS)
        | _keyword_mask(merged.get("sub_class"), _KC_KEYWORDS)
    )
    return merged.loc[mask].drop_duplicates(subset=["root_id"]).reset_index(drop=True)


def get_pn_neurons(cell_types_df: pd.DataFrame, classification_df: pd.DataFrame) -> pd.DataFrame:
    """Return projection neuron annotations with merged classification metadata."""

    merged = _merge_classification(cell_types_df, classification_df)
    mask = (
        _keyword_mask(merged.get("cell_type"), _PN_KEYWORDS)
        | _keyword_mask(merged.get("cell_type_aliases"), _PN_KEYWORDS)
        | _keyword_mask(merged.get("super_class"), _PN_KEYWORDS)
        | _keyword_mask(merged.get("class"), _PN_KEYWORDS)
        | _keyword_mask(merged.get("sub_class"), _PN_KEYWORDS)
    )
    return merged.loc[mask].drop_duplicates(subset=["root_id"]).reset_index(drop=True)


def get_mbon_neurons(cell_types_df: pd.DataFrame, classification_df: pd.DataFrame) -> pd.DataFrame:
    """Return mushroom body output neuron annotations."""

    merged = _merge_classification(cell_types_df, classification_df)
    mask = (
        _keyword_mask(merged.get("cell_type"), _MBON_KEYWORDS)
        | _keyword_mask(merged.get("cell_type_aliases"), _MBON_KEYWORDS)
        | _keyword_mask(merged.get("super_class"), _MBON_KEYWORDS)
        | _keyword_mask(merged.get("class"), _MBON_KEYWORDS)
        | _keyword_mask(merged.get("sub_class"), _MBON_KEYWORDS)
    )
    return merged.loc[mask].drop_duplicates(subset=["root_id"]).reset_index(drop=True)


def get_dan_neurons(cell_types_df: pd.DataFrame, classification_df: pd.DataFrame) -> pd.DataFrame:
    """Return dopaminergic neuron annotations."""

    merged = _merge_classification(cell_types_df, classification_df)
    mask = (
        _keyword_mask(merged.get("cell_type"), _DAN_KEYWORDS)
        | _keyword_mask(merged.get("cell_type_aliases"), _DAN_KEYWORDS)
        | _keyword_mask(merged.get("super_class"), _DAN_KEYWORDS)
        | _keyword_mask(merged.get("class"), _DAN_KEYWORDS)
        | _keyword_mask(merged.get("sub_class"), _DAN_KEYWORDS)
    )
    return merged.loc[mask].drop_duplicates(subset=["root_id"]).reset_index(drop=True)


def extract_neurotransmitter_info(
    neurons_df: pd.DataFrame,
    neuron_ids: Iterable[int],
    *,
    preferred_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return neurotransmitter predictions for ``neuron_ids``."""

    validate_dataframe_columns(neurons_df, ["root_id"], frame_name="neurons")
    subset = neurons_df[neurons_df["root_id"].isin({int(node) for node in neuron_ids})]
    if preferred_columns is None:
        preferred_columns = [
            column
            for column in ("nt_type", "predicted_nt", "primary_nt", "neurotransmitter")
            if column in neurons_df.columns
        ]
    columns = ["root_id", *[column for column in preferred_columns if column in neurons_df.columns]]
    return subset.loc[:, columns].drop_duplicates(subset=["root_id"]).reset_index(drop=True)


def map_brain_regions(
    names_df: pd.DataFrame,
    neuron_ids: Iterable[int],
    *,
    region_column: str = "group",
) -> pd.DataFrame:
    """Return brain region assignments for ``neuron_ids``."""

    validate_dataframe_columns(names_df, ["root_id"], frame_name="names")
    if region_column not in names_df.columns:
        raise ValueError(f"names dataframe is missing '{region_column}' column.")
    subset = names_df[names_df["root_id"].isin({int(node) for node in neuron_ids})]
    return subset.loc[:, ["root_id", region_column]].drop_duplicates(subset=["root_id"]).reset_index(drop=True)
