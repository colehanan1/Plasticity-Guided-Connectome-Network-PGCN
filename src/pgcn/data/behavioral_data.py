"""Utilities for loading and validating behavioral prediction data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

if TYPE_CHECKING:
    import torch

EXPECTED_TRIAL_COUNT: int = 440
EXPECTED_FLY_COUNT: int = 35
EXPECTED_TRIAL_LABELS: Tuple[str, ...] = tuple(f"testing_{i}" for i in range(1, 11))
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "model_predictions.csv"


@dataclass(frozen=True)
class BehavioralTrial:
    """Single behavioral trial entry.

    Attributes
    ----------
    dataset:
        Experimental training condition identifier.
    fly:
        Identifier for the individual fly (group label for CV).
    trial_label:
        Name of the testing odor exposure (`testing_1`‒`testing_10`).
    prediction:
        Binary response prediction (0=no approach, 1=approach).
    """

    dataset: str
    fly: str
    trial_label: str
    prediction: int


@dataclass(frozen=True)
class FlyBehavioralRecord:
    """Collection of behavioral trials recorded for a single fly."""

    fly: str
    dataset: str
    trials: Tuple[BehavioralTrial, ...]


def load_behavioral_dataframe(
    path: Optional[Path] = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the behavioral predictions CSV into a validated DataFrame.

    Parameters
    ----------
    path:
        Optional override for the CSV path. Defaults to the canonical dataset
        location under :mod:`data/model_predictions.csv`.
    validate:
        When ``True`` (default) enforce dataset integrity checks.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed in the original row order with string columns for the
        categorical values and an integer ``prediction`` column.
    """

    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find behavioral predictions CSV at '{csv_path}'."
        )

    df = pd.read_csv(
        csv_path,
        dtype={
            "dataset": "string",
            "fly": "string",
            "trial_label": "string",
            "prediction": "int64",
        },
    )
    df = df.reset_index(drop=True)

    if validate:
        df = _validate_behavioral_dataframe(df)

    return df


def _validate_behavioral_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate basic integrity constraints for the behavioral dataset."""

    expected_columns = {"dataset", "fly", "trial_label", "prediction"}
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "model_predictions.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if len(df) != EXPECTED_TRIAL_COUNT:
        raise ValueError(
            f"Behavioral dataset expected {EXPECTED_TRIAL_COUNT} rows but "
            f"found {len(df)}."
        )

    fly_counts = df["fly"].nunique(dropna=False)
    if fly_counts != EXPECTED_FLY_COUNT:
        raise ValueError(
            f"Behavioral dataset expected {EXPECTED_FLY_COUNT} unique flies "
            f"but found {fly_counts}."
        )

    dataset_per_fly = df.groupby("fly", dropna=False)["dataset"].nunique()
    if not bool((dataset_per_fly == 1).all()):
        raise ValueError("Each fly must belong to a single training dataset.")

    unique_trial_labels = set(df["trial_label"].dropna().astype(str))
    invalid_labels = unique_trial_labels.difference(EXPECTED_TRIAL_LABELS)
    if invalid_labels:
        raise ValueError(
            "Unexpected trial labels present: " + ", ".join(sorted(invalid_labels))
        )

    if not set(df["prediction"].unique()).issubset({0, 1}):
        raise ValueError("Predictions must be binary (0 or 1).")

    # Ensure stable ordering for reproducibility by sorting within fly groups.
    categorical_order = pd.CategoricalDtype(categories=EXPECTED_TRIAL_LABELS, ordered=True)
    df = df.copy()
    df["trial_label"] = df["trial_label"].astype(categorical_order)
    df.sort_values(["fly", "trial_label"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df.astype({"dataset": "string", "fly": "string", "trial_label": "string"})


def load_behavioral_trials(
    path: Optional[Path] = None,
    *,
    validate: bool = True,
) -> Tuple[BehavioralTrial, ...]:
    """Load the behavioral dataset into immutable :class:`BehavioralTrial` records."""

    df = load_behavioral_dataframe(path=path, validate=validate)
    trials = tuple(
        BehavioralTrial(
            dataset=row.dataset,
            fly=row.fly,
            trial_label=row.trial_label,
            prediction=int(row.prediction),
        )
        for row in df.itertuples(index=False)
    )
    return trials


def iter_fly_records(
    trials: Sequence[BehavioralTrial],
) -> Iterator[FlyBehavioralRecord]:
    """Group :class:`BehavioralTrial` entries by fly identifier."""

    fly_to_trials: dict[str, list[BehavioralTrial]] = {}
    for trial in trials:
        fly_to_trials.setdefault(trial.fly, []).append(trial)

    for fly, fly_trials in fly_to_trials.items():
        dataset_names = {trial.dataset for trial in fly_trials}
        if len(dataset_names) != 1:
            raise ValueError(
                f"Fly '{fly}' is associated with multiple datasets: {dataset_names}"
            )
        yield FlyBehavioralRecord(
            fly=fly,
            dataset=next(iter(dataset_names)),
            trials=tuple(sorted(fly_trials, key=lambda t: t.trial_label)),
        )


def get_model_ready_dataframe(
    df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare design matrices for classical ML models using pandas objects."""

    if df is None:
        df = load_behavioral_dataframe()

    categorical_features = pd.get_dummies(
        df[["dataset", "trial_label"]],
        prefix=["dataset", "trial"],
        dtype="float32",
    )
    labels = df["prediction"].astype("int64")
    groups = df["fly"].astype("string")

    return categorical_features, labels, groups


def get_model_ready_tensors(
    df: Optional[pd.DataFrame] = None,
    *,
    dtype: "torch.dtype" | None = None,
    device: Optional["torch.device"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Prepare PyTorch tensors for downstream modelling."""

    import torch

    if dtype is None:
        dtype = torch.float32

    features_df, labels, groups = get_model_ready_dataframe(df=df)
    features_tensor = torch.as_tensor(
        features_df.to_numpy(copy=True),
        dtype=dtype,
        device=device,
    )
    labels_tensor = torch.as_tensor(
        labels.to_numpy(copy=True),
        dtype=torch.float32,
        device=device,
    )
    group_codes, _ = pd.factorize(groups)
    groups_tensor = torch.as_tensor(group_codes, dtype=torch.long, device=device)

    return features_tensor, labels_tensor, groups_tensor


def make_group_kfold(
    n_splits: int = 5,
    *,
    groups: Optional[Iterable[str]] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield reproducible train/test indices using GroupKFold."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if groups is None:
        groups_series = load_behavioral_dataframe()["fly"].astype("string").reset_index(drop=True)
    else:
        groups_series = pd.Series(list(groups), dtype="string").reset_index(drop=True)

    if groups_series.nunique() < n_splits:
        raise ValueError(
            "Number of unique groups must be at least n_splits for GroupKFold."
        )

    indices = np.arange(len(groups_series))
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(np.zeros(len(indices)), groups=groups_series):
        yield train_idx, test_idx
