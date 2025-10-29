"""Factories for constructing task-specific datasets used in multi-task training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .behavioral_data import load_behavioral_dataframe

__all__ = [
    "ReservoirSpec",
    "TaskSpec",
    "MultiTaskConfig",
    "load_multi_task_config",
    "TaskDataLoaderFactory",
]


@dataclass(frozen=True)
class ReservoirSpec:
    """Configuration parameters controlling reservoir construction."""

    cache_dir: Optional[Path] = None
    n_pn: Optional[int] = None
    n_kc: Optional[int] = None
    n_mbon: Optional[int] = None
    sparsity: float = 0.05
    freeze_pn_kc: bool = True


@dataclass(frozen=True)
class TaskSpec:
    """Training metadata for a single task head."""

    name: str
    input_dim: int
    output_dim: int
    loss_function: str
    data_loader: str
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    activation: Optional[str] = None
    dropout: float = 0.0
    use_reservoir_head: bool = False
    feature_table: Optional[Path] = None
    target_column: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiTaskConfig:
    """Parsed configuration file describing the multi-task experiment."""

    reservoir: ReservoirSpec
    tasks: Mapping[str, TaskSpec]


def _coerce_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value)


def load_multi_task_config(path: Path | str) -> MultiTaskConfig:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "reservoir" not in config or "tasks" not in config:
        raise ValueError("Multi-task configuration must define 'reservoir' and 'tasks' sections.")

    reservoir_payload = config["reservoir"]
    reservoir = ReservoirSpec(
        cache_dir=_coerce_path(reservoir_payload.get("cache_dir")),
        n_pn=reservoir_payload.get("n_pn"),
        n_kc=reservoir_payload.get("n_kc"),
        n_mbon=reservoir_payload.get("n_mbon"),
        sparsity=reservoir_payload.get("sparsity", 0.05),
        freeze_pn_kc=reservoir_payload.get("freeze_pn_kc", True),
    )

    tasks_config: MutableMapping[str, TaskSpec] = {}
    for name, payload in config["tasks"].items():
        feature_table = _coerce_path(payload.get("feature_table"))
        task_spec = TaskSpec(
            name=name,
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            loss_function=str(payload["loss_function"]),
            data_loader=str(payload["data_loader"]),
            batch_size=int(payload.get("batch_size", 64)),
            epochs=int(payload.get("epochs", 20)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            activation=payload.get("activation"),
            dropout=float(payload.get("dropout", 0.0)),
            use_reservoir_head=bool(payload.get("use_reservoir_head", False)),
            feature_table=feature_table,
            target_column=payload.get("target_column"),
            metadata=payload.get("metadata", {}),
        )
        tasks_config[name] = task_spec
    return MultiTaskConfig(reservoir=reservoir, tasks=dict(tasks_config))


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature table '{path}' does not exist.")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".tsv"}:
        sep = "," if path.suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported feature table format: '{path.suffix}'.")


def _target_tensor(task: TaskSpec, series: pd.Series) -> torch.Tensor:
    if task.loss_function == "categorical_crossentropy":
        return torch.as_tensor(series.to_numpy(), dtype=torch.long)
    array = series.to_numpy(dtype=np.float32)
    if task.output_dim == 1:
        array = array.reshape(-1, 1)
    return torch.as_tensor(array, dtype=torch.float32)


def _features_tensor(task: TaskSpec, frame: pd.DataFrame) -> torch.Tensor:
    features = frame.to_numpy(dtype=np.float32)
    if features.shape[1] != task.input_dim:
        raise ValueError(
            "Feature dimensionality mismatch for task "
            f"'{task.name}': expected {task.input_dim}, observed {features.shape[1]}."
        )
    return torch.as_tensor(features, dtype=torch.float32)


def _tabular_loader(task: TaskSpec, *, shuffle: bool) -> DataLoader:
    if task.feature_table is None or task.target_column is None:
        raise ValueError(f"Task '{task.name}' requires 'feature_table' and 'target_column'.")
    table = _read_table(task.feature_table)
    if task.target_column not in table.columns:
        raise ValueError(
            f"Target column '{task.target_column}' missing from '{task.feature_table}'."
        )
    targets = _target_tensor(task, table[task.target_column])
    features = _features_tensor(task, table.drop(columns=[task.target_column]))
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=task.batch_size, shuffle=shuffle)


def _behavioral_loader(task: TaskSpec, *, shuffle: bool) -> DataLoader:
    dataloader = _tabular_loader(task, shuffle=shuffle)
    expected_trials = load_behavioral_dataframe().shape[0]
    observed_trials = len(dataloader.dataset)
    if observed_trials != expected_trials:
        raise ValueError(
            "Behavioral task feature table must align with behavioral dataset length: "
            f"expected {expected_trials}, observed {observed_trials}."
        )
    return dataloader


LoaderFn = Callable[[TaskSpec, bool], DataLoader]


class TaskDataLoaderFactory:
    """Registry mapping task loader identifiers to concrete implementations."""

    def __init__(self) -> None:
        self._registry: Dict[str, LoaderFn] = {
            "parquet_tensor": _tabular_loader,
            "behavioral_data": _behavioral_loader,
        }

    def register(self, name: str, loader: LoaderFn) -> None:
        self._registry[name] = loader

    def create(self, task: TaskSpec, *, shuffle: bool = True) -> DataLoader:
        if task.data_loader not in self._registry:
            raise KeyError(f"Unknown data loader '{task.data_loader}' for task '{task.name}'.")
        loader = self._registry[task.data_loader]
        return loader(task, shuffle)

    def available(self) -> Mapping[str, LoaderFn]:
        return dict(self._registry)
