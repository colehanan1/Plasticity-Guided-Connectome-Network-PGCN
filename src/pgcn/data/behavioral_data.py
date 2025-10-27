"""Utilities for loading and structuring behavioral trial data.

The helpers in this module guarantee a deterministic ordering of trials by
first sorting on the ``fly`` identifier and then the ``trial_label`` column.
This ensures downstream consumers (e.g. tensor exports) observe a stable
ordering regardless of the layout in the raw CSV.  When the canonical dataset
is loaded from :data:`BEHAVIORAL_DATA_PATH` additional validation checks assert
expected dataset sizes and grouping invariants so that downstream modelling can
trust the behavioural annotations.  The loader honours the
``PGCN_BEHAVIORAL_DATA`` environment variable and a compatibility alias,
``PGCN_BEHAVIORAL_DATA_PATH``, to support bespoke deployments that relocate the
behavioural CSV outside the repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

import pandas as pd

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None

try:  # pragma: no cover - optional dependency
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None

try:  # pragma: no cover - optional dependency
    from sklearn.model_selection import GroupKFold
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GroupKFold = None

PathLike = str | Path

#: Environment variable override for the behavioural dataset location.
BEHAVIORAL_DATA_ENV = "PGCN_BEHAVIORAL_DATA"

#: Backward-compatible aliases for environment variable overrides.
BEHAVIORAL_DATA_ENV_ALIASES: tuple[str, ...] = ("PGCN_BEHAVIORAL_DATA_PATH",)


def _default_data_path() -> Path:
    env_value = os.environ.get(BEHAVIORAL_DATA_ENV)
    if not env_value:
        for alias in BEHAVIORAL_DATA_ENV_ALIASES:
            env_value = os.environ.get(alias)
            if env_value:
                break
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "data" / "model_predictions.csv"


#: Default location for the behavioral CSV relative to the project root.
BEHAVIORAL_DATA_PATH: Path = _default_data_path()

EXPECTED_ROW_COUNT = 440
EXPECTED_FLY_COUNT = 35

_BASE_COLUMNS = ("dataset", "fly", "trial_label", "prediction")


@dataclass(frozen=True, slots=True)
class BehavioralTrial:
    """Immutable representation of a single behavioural trial."""

    dataset: str
    fly: str
    trial_label: str
    prediction: float
    metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class BehavioralTrialSet:
    """Collection of behavioural trials and their ordering metadata."""

    trials: tuple[BehavioralTrial, ...]
    fly_order: tuple[str, ...]
    trial_order: tuple[str, ...]

__all__ = [
    "BEHAVIORAL_DATA_ENV",
    "BEHAVIORAL_DATA_ENV_ALIASES",
    "BEHAVIORAL_DATA_PATH",
    "BehavioralTrial",
    "BehavioralTrialSet",
    "load_behavioral_dataframe",
    "load_behavioral_trials",
    "load_behavioral_tensor",
    "load_behavioral_model_frames",
    "load_behavioral_model_tensors",
    "load_behavioral_trial_matrix",
    "make_group_kfold",
]


def _to_path(path: Optional[PathLike]) -> Path:
    return Path(path) if path is not None else BEHAVIORAL_DATA_PATH


def _unique_in_order(values: Sequence) -> list:
    seen: set = set()
    ordered: list = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _validate_behavioral_dataframe(df: pd.DataFrame) -> None:
    if len(df) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} behavioural trials, found {len(df)}."
        )

    unique_flies = df["fly"].nunique()
    if unique_flies != EXPECTED_FLY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FLY_COUNT} unique flies, found {unique_flies}."
        )

    dataset_per_fly = df.groupby("fly")["dataset"].nunique()
    inconsistent = dataset_per_fly[dataset_per_fly != 1]
    if not inconsistent.empty:
        raise ValueError(
            "Each fly must belong to a single dataset, violations: "
            f"{inconsistent.to_dict()}"
        )

    trials_per_fly = df.groupby("fly")["trial_label"].nunique()
    if trials_per_fly.nunique() != 1:
        raise ValueError(
            "All flies must observe the same number of unique trial labels."
        )


def load_behavioral_dataframe(
    path: Optional[PathLike] = None,
    *,
    validate: Optional[bool] = None,
) -> pd.DataFrame:
    """Return the behavioral trials sorted by fly and trial label.

    Parameters
    ----------
    path:
        Optional override for the CSV path.  When omitted the loader reads the
        canonical dataset from :data:`BEHAVIORAL_DATA_PATH`.
    validate:
        When ``True`` perform dataset-level validation checks (row count,
        grouping invariants).  ``None`` enables validation automatically when
        reading from :data:`BEHAVIORAL_DATA_PATH` and the file exists.

    Returns
    -------
    pandas.DataFrame
        The behavioral trials sorted by ``["fly", "trial_label"]`` with a
        freshly reset integer index.  Callers can rely on the returned order to
        be deterministic across runs and independent of the on-disk layout.
    """

    csv_path = _to_path(path)
    df = pd.read_csv(csv_path)
    sorted_df = df.sort_values(["fly", "trial_label"], kind="mergesort").reset_index(drop=True)
    should_validate = validate if validate is not None else csv_path == BEHAVIORAL_DATA_PATH and csv_path.exists()
    if should_validate:
        _validate_behavioral_dataframe(sorted_df)
    return sorted_df


def _row_to_trial(row: Mapping[str, object]) -> BehavioralTrial:
    missing_columns = [column for column in _BASE_COLUMNS if column not in row]
    if missing_columns:
        raise KeyError(f"Behavioral dataset missing required columns: {missing_columns}")

    base: MutableMapping[str, object] = {key: row[key] for key in _BASE_COLUMNS}
    metadata = MappingProxyType({key: value for key, value in row.items() if key not in _BASE_COLUMNS})
    return BehavioralTrial(
        dataset=str(base["dataset"]),
        fly=str(base["fly"]),
        trial_label=str(base["trial_label"]),
        prediction=float(base["prediction"]),
        metadata=metadata,
    )


def load_behavioral_trials(
    path: Optional[PathLike] = None,
    *,
    validate: Optional[bool] = None,
) -> BehavioralTrialSet:
    """Load behavioural trials into typed dataclasses.

    Returns
    -------
    BehavioralTrialSet
        Immutable container with ordered trials plus fly and trial label
        catalogues for downstream consumers.
    """

    df = load_behavioral_dataframe(path, validate=validate)
    trials = tuple(_row_to_trial(row) for row in df.to_dict(orient="records"))
    fly_order = tuple(_unique_in_order(df["fly"].tolist()))
    trial_order = tuple(_unique_in_order(df["trial_label"].tolist()))
    return BehavioralTrialSet(trials=trials, fly_order=fly_order, trial_order=trial_order)


def load_behavioral_tensor(
    path: Optional[PathLike] = None,
    *,
    dtype: Optional["torch.dtype"] = None,
    device: Optional["torch.device"] = None,
    columns: Iterable[str] = ("prediction",),
) -> "torch.Tensor":
    """Load behavioral data and return a tensor aligned with the sorted trials.

    The tensor stacks the requested ``columns`` following the deterministic
    ordering produced by :func:`load_behavioral_dataframe`.  Each column becomes
    a feature dimension in the output, making the result suitable for downstream
    modeling without additional sorting.

    Raises
    ------
    ImportError
        If PyTorch is not available in the environment.
    ValueError
        If any requested column is missing from the CSV.
    """

    if torch is None:  # pragma: no cover - exercised when torch missing
        raise ImportError("PyTorch is required to load behavioral tensors.")

    df = load_behavioral_dataframe(path, validate=None)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing!r} are not present in the behavioral dataset.")

    matrix = df.loc[:, list(columns)].to_numpy()
    tensor = torch.as_tensor(matrix, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def load_behavioral_model_frames(
    path: Optional[PathLike] = None,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    label_column: str = "prediction",
    group_column: str = "fly",
    validate: Optional[bool] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return modelling-ready pandas objects following the sorted ordering."""

    df = load_behavioral_dataframe(path, validate=validate)

    for required in (label_column, group_column):
        if required not in df.columns:
            raise ValueError(f"Required column '{required}' missing from behavioural dataset.")

    if feature_columns is None:
        feature_columns = [
            column
            for column in df.columns
            if column not in {label_column, group_column}
        ]

    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(
            f"Feature columns {missing_features!r} are not present in the behavioural dataset."
        )

    features = df.loc[:, list(feature_columns)].copy()
    labels = df[label_column].copy()
    groups = df[group_column].copy()
    return features, labels, groups


def load_behavioral_model_tensors(
    path: Optional[PathLike] = None,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    label_column: str = "prediction",
    group_column: str = "fly",
    dtype: Optional["torch.dtype"] = None,
    label_dtype: Optional["torch.dtype"] = None,
    device: Optional["torch.device"] = None,
    validate: Optional[bool] = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Return ``(features, labels, groups)`` tensors respecting the sorted order."""

    if torch is None:  # pragma: no cover - exercised when torch missing
        raise ImportError("PyTorch is required to load behavioural tensors.")
    if np is None:  # pragma: no cover - exercised when numpy missing
        raise ImportError("NumPy is required to prepare behavioural tensors.")

    features_df, labels, groups = load_behavioral_model_frames(
        path,
        feature_columns=feature_columns,
        label_column=label_column,
        group_column=group_column,
        validate=validate,
    )

    if features_df.empty:
        raise ValueError("Feature matrix is empty; specify feature_columns with numeric values.")

    feature_matrix = features_df.to_numpy()
    label_array = labels.to_numpy()
    group_codes = pd.Categorical(groups).codes.astype("int64", copy=True)

    features_tensor = torch.as_tensor(feature_matrix, dtype=dtype)
    labels_tensor = torch.as_tensor(label_array, dtype=label_dtype or dtype)
    groups_tensor = torch.as_tensor(group_codes, dtype=torch.long)

    if device is not None:
        features_tensor = features_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        groups_tensor = groups_tensor.to(device)

    return features_tensor, labels_tensor, groups_tensor


def load_behavioral_trial_matrix(path: Optional[PathLike] = None) -> pd.DataFrame:
    """Pivot the sorted behavioral data into a fly-by-trial matrix.

    The returned DataFrame preserves the deterministic ordering introduced by
    :func:`load_behavioral_dataframe`, making it straightforward to align model
    predictions with specific flies and trial labels.
    """

    df = load_behavioral_dataframe(path)
    fly_order = _unique_in_order(df["fly"].tolist())
    trial_order = _unique_in_order(df["trial_label"].tolist())

    matrix = (
        df.set_index(["fly", "trial_label"])["prediction"].unstack("trial_label")
    )
    matrix = matrix.reindex(index=fly_order, columns=trial_order)
    return matrix


def make_group_kfold(
    path: Optional[PathLike] = None,
    *,
    n_splits: int = 5,
    groups: Optional[Sequence] = None,
    validate: Optional[bool] = None,
) -> Iterator[tuple[Sequence[int], Sequence[int]]]:
    """Yield reproducible ``(train_idx, test_idx)`` splits grouped by fly."""

    if GroupKFold is None:  # pragma: no cover - exercised when sklearn missing
        raise ImportError("scikit-learn is required to create group K-fold splits.")
    if np is None:  # pragma: no cover - exercised when numpy missing
        raise ImportError("NumPy is required to create group K-fold splits.")

    df = load_behavioral_dataframe(path, validate=validate)
    feature_index = np.arange(len(df))
    fold_groups = groups if groups is not None else df["fly"].to_numpy()

    splitter = GroupKFold(n_splits=n_splits)
    for train, test in splitter.split(feature_index, groups=fold_groups):
        yield train, test
