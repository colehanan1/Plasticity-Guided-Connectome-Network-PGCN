"""Utilities for validating FlyWire dataset integrity."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

__all__ = [
    "validate_file_exists",
    "validate_dataframe_columns",
    "ensure_no_missing_root_ids",
]


def validate_file_exists(path: Path) -> None:
    """Raise ``FileNotFoundError`` if ``path`` does not exist."""

    if not path.exists():
        raise FileNotFoundError(f"Required dataset missing: {path}")


def validate_dataframe_columns(df: pd.DataFrame, columns: Sequence[str], *, frame_name: str) -> None:
    """Ensure that ``df`` exposes all ``columns``."""

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def ensure_no_missing_root_ids(df: pd.DataFrame, *, columns: Iterable[str], frame_name: str) -> None:
    """Validate that ``df`` has no null root identifiers in ``columns``."""

    for column in columns:
        if df[column].isna().any():
            raise ValueError(f"{frame_name} contains null values in '{column}'.")
        if not pd.api.types.is_integer_dtype(df[column]):
            try:
                df[column] = pd.to_numeric(df[column], errors="raise", downcast="integer")
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"{frame_name} column '{column}' cannot be coerced to integers: {exc}"
                ) from exc
