"""Load FlyWire FAFB v783 datasets from local CSV exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, MutableMapping, Optional

import pandas as pd

from config import paths
from utils.data_validation import (
    ensure_no_missing_root_ids,
    validate_dataframe_columns,
    validate_file_exists,
)

__all__ = [
    "FlyWireLocalDataLoader",
    "load_flywire_connections",
]


_CONNECTION_DTYPES = {
    "pre_root_id": "int64",
    "post_root_id": "int64",
    "syn_count": "int64",
}

_CELL_TYPE_COLUMNS = ["root_id", "cell_type"]
_CLASSIFICATION_COLUMNS = ["root_id", "super_class", "sub_class"]
_CONNECTION_COLUMNS = ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"]


def _normalise_cell_types(df: pd.DataFrame) -> pd.DataFrame:
    """Rename FlyWire cell-type exports to the canonical column contract."""

    rename_map: dict[str, str] = {}
    if "cell_type" not in df.columns and "primary_type" in df.columns:
        rename_map["primary_type"] = "cell_type"
    if "cell_type_aliases" not in df.columns and "additional_type(s)" in df.columns:
        rename_map["additional_type(s)"] = "cell_type_aliases"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "cell_type_aliases" not in df.columns:
        df["cell_type_aliases"] = pd.NA
    return df

@dataclass(slots=True)
class FlyWireLocalDataLoader:
    """Lazy loader for FAFB FlyWire CSV exports."""

    dataset_dir: Path = field(default_factory=lambda: paths.DATA_ROOT)
    cache_results: bool = True
    _cache: MutableMapping[str, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_connections(
        self,
        *,
        neuropil_filter: Optional[Iterable[str] | Callable[[str], bool]] = None,
        min_synapses: int = 5,
        chunk_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load the FlyWire connection table with optional filtering."""

        path = self._resolve_path(paths.CONNECTIONS_FILE)
        return load_flywire_connections(
            path,
            neuropil_filter=neuropil_filter,
            min_synapses=min_synapses,
            chunk_size=chunk_size,
        )

    def load_cell_types(self) -> pd.DataFrame:
        cache_key = "cell_types"
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

        path = self._resolve_path(paths.CELL_TYPES_FILE)
        df = pd.read_csv(path, compression="gzip")
        df = _normalise_cell_types(df)
        validate_dataframe_columns(df, _CELL_TYPE_COLUMNS, frame_name=cache_key)
        ensure_no_missing_root_ids(df, columns=["root_id"], frame_name=cache_key)
        if self.cache_results:
            self._cache[cache_key] = df
        return df

    def load_classification(self) -> pd.DataFrame:
        return self._load_csv(
            "classification",
            paths.CLASSIFICATION_FILE,
            columns=_CLASSIFICATION_COLUMNS,
        )

    def load_neurotransmitters(self) -> pd.DataFrame:
        return self._load_csv("neurons", paths.NEUROTRANSMITTER_FILE)

    def load_names(self) -> pd.DataFrame:
        return self._load_csv("names", paths.NAMES_FILE)

    def load_processed_labels(self) -> pd.DataFrame:
        return self._load_csv("processed_labels", paths.PROCESSED_LABELS_FILE)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, default_path: Path) -> Path:
        candidate = self.dataset_dir / default_path.name
        path = candidate if candidate.exists() else default_path
        validate_file_exists(path)
        return path

    def _load_csv(
        self,
        cache_key: str,
        default_path: Path,
        *,
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

        path = self._resolve_path(default_path)
        df = pd.read_csv(path, compression="gzip")
        if columns is not None:
            validate_dataframe_columns(df, columns, frame_name=cache_key)
        if "root_id" in df.columns:
            ensure_no_missing_root_ids(df, columns=["root_id"], frame_name=cache_key)
        if self.cache_results:
            self._cache[cache_key] = df
        return df


def _normalise_neuropil_filter(
    neuropil_filter: Optional[Iterable[str] | Callable[[str], bool]],
) -> Optional[Callable[[str], bool]]:
    if neuropil_filter is None:
        return None
    if callable(neuropil_filter):
        return neuropil_filter
    values = {value.lower() for value in neuropil_filter}
    return lambda neuropil: neuropil.lower() in values


@lru_cache(maxsize=8)
def _load_connections_frame(path: Path) -> pd.DataFrame:
    validate_file_exists(path)
    df = pd.read_csv(
        path,
        compression="gzip",
        dtype=_CONNECTION_DTYPES,
        usecols=_CONNECTION_COLUMNS,
        low_memory=False,
    )
    validate_dataframe_columns(df, _CONNECTION_COLUMNS, frame_name="connections")
    ensure_no_missing_root_ids(
        df,
        columns=["pre_root_id", "post_root_id"],
        frame_name="connections",
    )
    return df


def _filter_connections(
    df: pd.DataFrame,
    *,
    neuropil_filter: Optional[Callable[[str], bool]],
    min_synapses: int,
) -> pd.DataFrame:
    filtered = df[df["syn_count"] >= int(min_synapses)]
    if neuropil_filter is not None and "neuropil" in filtered.columns:
        mask = filtered["neuropil"].astype(str).apply(neuropil_filter)
        filtered = filtered[mask]
    return filtered.reset_index(drop=True)


def load_flywire_connections(
    data_path: Path | str,
    neuropil_filter: Optional[Iterable[str] | Callable[[str], bool]] = None,
    *,
    min_synapses: int = 5,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load the Princeton-processed connection table."""

    path = Path(data_path)
    validate_file_exists(path)
    predicate = _normalise_neuropil_filter(neuropil_filter)

    if chunk_size:
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            dtype=_CONNECTION_DTYPES,
            usecols=_CONNECTION_COLUMNS,
            low_memory=False,
            chunksize=chunk_size,
        ):
            filtered = _filter_connections(chunk, neuropil_filter=predicate, min_synapses=min_synapses)
            if not filtered.empty:
                chunks.append(filtered)
        if not chunks:
            return pd.DataFrame(columns=_CONNECTION_COLUMNS)
        return pd.concat(chunks, ignore_index=True)

    df = _load_connections_frame(path)
    return _filter_connections(df, neuropil_filter=predicate, min_synapses=min_synapses)
