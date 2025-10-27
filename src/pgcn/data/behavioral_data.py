"""Utilities for loading and structuring behavioral trial data.

The helpers in this module guarantee a deterministic ordering of trials by first
sorting on the ``fly`` identifier and then the ``trial_label`` column.  This
ensures downstream consumers (e.g. tensor exports) observe a stable ordering
regardless of the layout in the raw CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None

PathLike = str | Path

#: Default location for the behavioral CSV relative to the project root.
BEHAVIORAL_DATA_PATH: Path = Path(__file__).resolve().parents[3] / "data" / "model_predictions.csv"

__all__ = [
    "BEHAVIORAL_DATA_PATH",
    "load_behavioral_dataframe",
    "load_behavioral_tensor",
    "load_behavioral_trial_matrix",
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


def load_behavioral_dataframe(path: Optional[PathLike] = None) -> pd.DataFrame:
    """Return the behavioral trials sorted by fly and trial label.

    Parameters
    ----------
    path:
        Optional override for the CSV path.  When omitted the loader reads the
        canonical dataset from :data:`BEHAVIORAL_DATA_PATH`.

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
    return sorted_df


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

    df = load_behavioral_dataframe(path)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing!r} are not present in the behavioral dataset.")

    matrix = df.loc[:, list(columns)].to_numpy()
    tensor = torch.as_tensor(matrix, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


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
