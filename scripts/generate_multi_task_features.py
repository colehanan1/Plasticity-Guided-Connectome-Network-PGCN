"""Generate PN feature tables required by the multi-task trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from pgcn.data import load_behavioral_dataframe, load_multi_task_config


@dataclass(frozen=True)
class GenerationResult:
    task_name: str
    rows: int
    path: Path
    created: bool


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/multi_task_config.yaml",
        help="Path to the multi-task YAML configuration file.",
    )
    parser.add_argument(
        "--behavior-csv",
        default=None,
        help="Optional override for the behavioural CSV (defaults to repository configuration).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2048,
        help="Number of rows to synthesise for non-behavioural tasks (default: 2048).",
    )
    parser.add_argument(
        "--active-fraction",
        type=float,
        default=0.05,
        help="Fraction of PN features to activate when constructing sparse vectors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature tables instead of skipping them.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write a machine-readable summary of generated artefacts.",
    )
    return parser


def _pn_column_names(n_features: int) -> list[str]:
    return [f"pn_{idx}" for idx in range(n_features)]


def _hash_seed(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _sparse_feature_matrix(
    keys: Iterable[str],
    n_features: int,
    *,
    active_fraction: float,
) -> np.ndarray:
    keys = list(keys)
    if not 0.0 < active_fraction <= 1.0:
        raise ValueError("active_fraction must lie within (0, 1].")
    active = max(1, int(round(n_features * active_fraction)))
    matrix = np.zeros((len(keys), n_features), dtype=np.float32)
    for row_index, key in enumerate(keys):
        rng = np.random.default_rng(_hash_seed(key))
        active_indices = rng.choice(n_features, size=active, replace=False)
        matrix[row_index, active_indices] = 1.0
    return matrix


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _behavioural_keys(df: pd.DataFrame) -> list[str]:
    dataset = df.get("dataset", pd.Series(["unknown"] * len(df)))
    fly = df.get("fly", pd.Series(["fly"] * len(df)))
    label = df.get("trial_label", pd.Series(["trial"] * len(df)))
    return [f"{d}|{f}|{l}|{idx}" for idx, (d, f, l) in enumerate(zip(dataset, fly, label, strict=False))]


def _generate_behavioral_task(
    task_name: str,
    *,
    target_column: str,
    feature_count: int,
    df: pd.DataFrame,
    active_fraction: float,
    output_path: Path,
    overwrite: bool,
) -> GenerationResult:
    if target_column not in df.columns:
        raise KeyError(
            f"Behavioral dataset is missing required target column '{target_column}' for task '{task_name}'."
        )
    if output_path.exists() and not overwrite:
        return GenerationResult(task_name, len(df), output_path, created=False)

    keys = _behavioural_keys(df)
    features = _sparse_feature_matrix(keys, feature_count, active_fraction=active_fraction)
    frame = pd.DataFrame(features, columns=_pn_column_names(feature_count))
    frame[target_column] = df[target_column].to_numpy()
    _write_parquet(frame, output_path)
    return GenerationResult(task_name, len(frame), output_path, created=True)


def _generate_synthetic_task(
    task_name: str,
    *,
    feature_count: int,
    output_dim: int,
    loss_function: str,
    target_column: Optional[str],
    rows: int,
    active_fraction: float,
    output_path: Path,
    overwrite: bool,
) -> GenerationResult:
    if target_column is None:
        raise ValueError(
            f"Task '{task_name}' must define 'target_column' to generate synthetic data."
        )
    if output_path.exists() and not overwrite:
        return GenerationResult(task_name, rows, output_path, created=False)

    keys = [f"{task_name}:{index}" for index in range(rows)]
    base = _sparse_feature_matrix(keys, feature_count, active_fraction=active_fraction)
    rng = np.random.default_rng(_hash_seed(task_name))
    noise = rng.normal(loc=0.0, scale=0.05, size=base.shape).astype(np.float32)
    features = np.clip(base + noise, 0.0, None)
    frame = pd.DataFrame(features, columns=_pn_column_names(feature_count))

    if loss_function == "categorical_crossentropy":
        targets = rng.integers(0, max(1, output_dim), size=rows, dtype=np.int64)
        frame[target_column] = targets
    elif loss_function == "binary_crossentropy":
        targets = rng.integers(0, 2, size=rows, dtype=np.int64)
        frame[target_column] = targets.astype(np.float32)
    elif loss_function == "mse":
        if output_dim == 1:
            targets = rng.normal(loc=0.0, scale=1.0, size=rows).astype(np.float32)
            frame[target_column] = targets
        else:
            for output_index in range(output_dim):
                column = f"{target_column}_{output_index}"
                frame[column] = rng.normal(loc=0.0, scale=1.0, size=rows).astype(np.float32)
    else:
        raise ValueError(f"Unsupported loss function '{loss_function}' for task '{task_name}'.")

    _write_parquet(frame, output_path)
    return GenerationResult(task_name, rows, output_path, created=True)


def generate_feature_tables(args: argparse.Namespace) -> list[GenerationResult]:
    config = load_multi_task_config(Path(args.config))
    behavioural_df: Optional[pd.DataFrame] = None
    if any(spec.data_loader == "behavioral_data" for spec in config.tasks.values()):
        behavioural_df = load_behavioral_dataframe(args.behavior_csv)

    results: list[GenerationResult] = []
    for task_name, spec in config.tasks.items():
        output_path = Path(spec.feature_table) if spec.feature_table is not None else None
        if output_path is None:
            raise ValueError(f"Task '{task_name}' must define a feature_table path.")
        if spec.input_dim is None:
            raise ValueError(f"Task '{task_name}' is missing input_dim in configuration.")

        if spec.data_loader == "behavioral_data":
            if behavioural_df is None:
                raise RuntimeError("Behavioral dataframe failed to load.")
            result = _generate_behavioral_task(
                task_name,
                target_column=spec.target_column or "prediction",
                feature_count=spec.input_dim,
                df=behavioural_df,
                active_fraction=args.active_fraction,
                output_path=output_path,
                overwrite=args.overwrite,
            )
        else:
            result = _generate_synthetic_task(
                task_name,
                feature_count=spec.input_dim,
                output_dim=spec.output_dim,
                loss_function=spec.loss_function,
                target_column=spec.target_column,
                rows=args.rows,
                active_fraction=args.active_fraction,
                output_path=output_path,
                overwrite=args.overwrite,
            )
        results.append(result)
    return results


def _print_report(results: Iterable[GenerationResult]) -> None:
    for result in results:
        status = "created" if result.created else "skipped"
        print(f"[{status}] {result.task_name}: rows={result.rows} -> {result.path}")


def _write_report(results: Iterable[GenerationResult], destination: Path) -> None:
    payload = [
        {
            "task": result.task_name,
            "rows": result.rows,
            "path": str(result.path),
            "created": result.created,
        }
        for result in results
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = _argument_parser()
    args = parser.parse_args()
    results = generate_feature_tables(args)
    _print_report(results)
    if args.report_json is not None:
        _write_report(results, Path(args.report_json))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
