"""Utilities for loading and validating behavioral prediction data."""
from __future__ import annotations

import os
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
BEHAVIORAL_DATA_ENV_VAR = "PGCN_BEHAVIORAL_DATA_PATH"


@dataclass(frozen=True)
class BehavioralTrial:
    """Single behavioral trial entry.

    Attributes
    ----------
    dataset:
        Experimental training condition identifier.
    fly:
        Identifier for the individual fly (group label for CV).
    fly_number:
        Stable integer identifier for the fly when provided by the dataset.
    trial_label:
        Name of the testing odor exposure (`testing_1`‒`testing_10`).
    prediction:
        Binary response prediction (0=no approach, 1=approach).
    probability:
        Model probability for the predicted response (0‒1) when available.
    """

    dataset: str
    fly: str
    trial_label: str
    prediction: int
    fly_number: Optional[int] = None
    probability: Optional[float] = None


@dataclass(frozen=True)
class FlyBehavioralRecord:
    """Collection of behavioral trials recorded for a single fly."""

    fly: str
    dataset: str
    fly_number: Optional[int]
    trials: Tuple[BehavioralTrial, ...]


def resolve_behavioral_data_path(path: Optional[Path | str] = None) -> Path:
    """Return the resolved CSV path using overrides or environment settings."""

    if path is not None:
        return Path(os.path.expandvars(os.path.expanduser(str(path))))

    env_path = os.environ.get(BEHAVIORAL_DATA_ENV_VAR)
    if env_path:
        return Path(os.path.expandvars(os.path.expanduser(env_path)))

    return DEFAULT_DATA_PATH


def load_behavioral_dataframe(
    path: Optional[Path | str] = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the behavioral predictions CSV into a validated DataFrame.

    Parameters
    ----------
    path:
        Optional override for the CSV path. When ``None`` the loader first checks
        the ``PGCN_BEHAVIORAL_DATA_PATH`` environment variable and falls back to
        :mod:`data/model_predictions.csv`.
    validate:
        When ``True`` (default) enforce dataset integrity checks.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed in the original row order with string columns for the
        categorical values and an integer ``prediction`` column.
    """

    csv_path = resolve_behavioral_data_path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find behavioral predictions CSV at '{csv_path}'."
        )

    df = pd.read_csv(
        csv_path,
        dtype={
            "dataset": "string",
            "fly": "string",
            "fly_number": "Int64",
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

    expected_columns = {
        "dataset",
        "fly",
        "fly_number",
        "trial_label",
        "prediction",
        "probability",
    }
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

    fly_number_per_fly = df.groupby("fly", dropna=False)["fly_number"].nunique()
    if not bool((fly_number_per_fly == 1).all()):
        raise ValueError("Each fly must map to a unique fly_number identifier.")

    if df["fly_number"].isna().any():
        raise ValueError("fly_number values must be present for every row.")

    if df["probability"].isna().any():
        raise ValueError("probability values must be present for every row.")

    if not ((df["probability"] >= 0).all() and (df["probability"] <= 1).all()):
        raise ValueError("probability must fall within the [0, 1] interval.")

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

    df = df.astype(
        {
            "dataset": "string",
            "fly": "string",
            "trial_label": "string",
            "prediction": "int64",
        }
    )
    df["fly_number"] = df["fly_number"].astype("int64")
    df["probability"] = df["probability"].astype("float32")

    return df


def load_behavioral_trials(
    path: Optional[Path | str] = None,
    *,
    validate: bool = True,
) -> Tuple[BehavioralTrial, ...]:
    """Load the behavioral dataset into immutable :class:`BehavioralTrial` records."""

    df = load_behavioral_dataframe(path=path, validate=validate)
    trials = tuple(
        BehavioralTrial(
            dataset=row.dataset,
            fly=row.fly,
            fly_number=int(row.fly_number) if getattr(row, "fly_number", None) is not None else None,
            trial_label=row.trial_label,
            prediction=int(row.prediction),
            probability=float(row.probability)
            if getattr(row, "probability", None) is not None
            else None,
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
        fly_numbers = {trial.fly_number for trial in fly_trials if trial.fly_number is not None}
        if len(fly_numbers) > 1:
            raise ValueError(
                f"Fly '{fly}' has inconsistent fly_number identifiers: {fly_numbers}"
            )
        yield FlyBehavioralRecord(
            fly=fly,
            dataset=next(iter(dataset_names)),
            fly_number=next(iter(fly_numbers)) if fly_numbers else None,
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
    numeric_frames: list[pd.DataFrame] = []
    if "fly_number" in df.columns:
        numeric_frames.append(df[["fly_number"]].astype("float32"))
    if "probability" in df.columns:
        numeric_frames.append(df[["probability"]].astype("float32"))

    if numeric_frames:
        numeric_feature_frame = pd.concat(numeric_frames, axis=1)
        categorical_features = pd.concat(
            [categorical_features, numeric_feature_frame], axis=1
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
