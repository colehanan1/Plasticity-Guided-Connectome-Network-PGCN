"""Neuron classification utilities for FlyWire FAFB datasets."""

from __future__ import annotations

import ast
import re
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from utils.data_validation import validate_dataframe_columns

__all__ = [
    "get_kc_neurons",
    "get_pn_neurons",
    "get_mbon_neurons",
    "get_dan_neurons",
    "extract_neurotransmitter_info",
    "map_brain_regions",
    "infer_pn_glomerulus_labels",
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


_CANONICAL_GLOMERULI: Sequence[str] = (
    "DA1",
    "DA2",
    "DA3",
    "DA4l",
    "DA4m",
    "DA4",
    "DA5",
    "DA6",
    "DA7",
    "DL1",
    "DL2d",
    "DL2v",
    "DL2",
    "DL3",
    "DL4",
    "DL5",
    "DM1",
    "DM2",
    "DM3",
    "DM4",
    "DM5",
    "DM6",
    "DM7",
    "DM8",
    "VA1d",
    "VA1v",
    "VA2",
    "VA3",
    "VA4",
    "VA5",
    "VA6",
    "VA7l",
    "VA7m",
    "VA7",
    "VA8",
    "VA9",
    "VM1",
    "VM2",
    "VM3",
    "VM4",
    "VM5",
    "VM6",
    "VM7d",
    "VM7v",
    "VM7",
    "VM8",
    "VM9",
    "VL1",
    "VL2",
    "VL3",
    "VP1",
    "VP2",
    "VP3",
    "VP4",
    "VP5",
    "VC1",
    "VC2",
    "DC1",
    "DC2",
    "DC3",
    "DC4",
    "DC5",
    "DC6",
    "DC7",
    "DP1",
    "DP1l",
    "DP1m",
    "DP2",
    "DP3",
    "DP4",
    "DP5",
    "DP6",
    "DP7",
)

_CANONICAL_LOOKUP: Dict[str, str] = {value.upper(): value for value in _CANONICAL_GLOMERULI}
_ALLOWED_PREFIXES = {re.match(r"([A-Z]+)\d", glomerulus).group(1) for glomerulus in _CANONICAL_GLOMERULI if re.match(r"([A-Z]+)\d", glomerulus)}
_GLOMERULUS_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


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


def _normalise_glomerulus_token(token: str) -> str | None:
    token = token.strip()
    if not token:
        return None
    token = re.sub(r"glomerulus", "", token, flags=re.IGNORECASE)
    token = token.strip("-_ ")
    if not token:
        return None
    upper = token.upper()
    if upper in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[upper]

    match = re.match(r"([A-Z]+)(\d+)([A-Z]*)", upper)
    if not match:
        return None
    prefix, digits, suffix = match.groups()
    if prefix not in _ALLOWED_PREFIXES:
        return None
    canonical_upper = prefix + digits + suffix.lower()
    if canonical_upper.upper() in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[canonical_upper.upper()]
    return prefix + digits + suffix.lower()


def _split_candidate_text(text: str) -> List[str]:
    return [segment for segment in _GLOMERULUS_SPLIT_RE.split(text) if segment]


def _extract_glomerulus_from_candidates(candidates: Iterable[str]) -> str | pd.NA:
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        for segment in _split_candidate_text(candidate):
            normalised = _normalise_glomerulus_token(segment)
            if normalised:
                return normalised
    return pd.NA


def _parse_processed_label_entry(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, bytes))]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if isinstance(item, (str, bytes))]
        return [value]
    return []


def _build_processed_label_lookup(processed_labels_df: pd.DataFrame | None) -> Dict[int, List[str]]:
    if processed_labels_df is None or processed_labels_df.empty:
        return {}

    validate_dataframe_columns(processed_labels_df, ["root_id"], frame_name="processed_labels")

    label_column = "processed_labels" if "processed_labels" in processed_labels_df.columns else None
    if label_column is None:
        for column in processed_labels_df.columns:
            if column == "root_id":
                continue
            if processed_labels_df[column].dtype == object:
                label_column = column
                break
    if label_column is None:
        return {}

    lookup: Dict[int, List[str]] = {}
    for row in processed_labels_df.itertuples(index=False):
        root_id = getattr(row, "root_id", None)
        if pd.isna(root_id):
            continue
        labels = _parse_processed_label_entry(getattr(row, label_column))
        if labels:
            lookup[int(root_id)] = labels
    return lookup


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


def infer_pn_glomerulus_labels(
    pn_df: pd.DataFrame,
    *,
    processed_labels_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Infer glomerulus assignments for projection neurons."""

    if pn_df.empty:
        return pd.Series(dtype="object")

    validate_dataframe_columns(pn_df, ["root_id"], frame_name="pn_neurons")
    label_lookup = _build_processed_label_lookup(processed_labels_df)

    glomeruli: List[str | pd.NA] = []
    for row in pn_df.itertuples(index=False):
        root_id = getattr(row, "root_id")
        candidates: List[str] = []
        for column in ("cell_type", "cell_type_aliases", "super_class", "class", "sub_class"):
            value = getattr(row, column, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        candidates.extend(label_lookup.get(int(root_id), []))
        glomeruli.append(_extract_glomerulus_from_candidates(candidates))

    return pd.Series(glomeruli, index=pn_df.index, dtype="object")


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
