"""Inspect locally stored FlyWire FAFB datasets and report schema details."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

from config import paths
from data_loaders.neuron_classification import get_kc_neurons, get_pn_neurons
from utils.data_validation import validate_file_exists

# Expected columns for quick validation and reporting.
_EXPECTED_COLUMNS: Mapping[str, Iterable[str]] = {
    "connections": ("pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"),
    "cell_types": ("root_id", "cell_type", "super_class"),
    "classification": ("root_id", "super_class", "sub_class"),
    "neurons": ("root_id",),
    "names": ("root_id", "group"),
    "processed_labels": ("root_id",),
}


def _resolve_dataset_paths(data_dir: Optional[Path]) -> Dict[str, Path]:
    """Return dataset paths, preferring ``data_dir`` when provided."""

    resolved: Dict[str, Path] = {}
    base_dir = Path(data_dir) if data_dir is not None else None

    for key, default_path in paths.get_dataset_paths().items():
        default_path = Path(default_path)
        candidate = base_dir / default_path.name if base_dir is not None else None
        if candidate is not None and candidate.exists():
            resolved[key] = candidate
        else:
            resolved[key] = default_path
    return resolved


def _preview_frame(path: Path, head: int) -> pd.DataFrame:
    """Return a small preview of the CSV contents."""

    return pd.read_csv(path, compression="gzip", nrows=head)


def _load_full_frame(path: Path) -> pd.DataFrame:
    """Load the full dataframe for downstream analysis."""

    return pd.read_csv(path, compression="gzip")


def _summarise_columns(key: str, df: pd.DataFrame) -> Dict[str, object]:
    """Describe dataframe columns, highlighting missing expectations."""

    summary: Dict[str, object] = {
        "column_count": df.shape[1],
        "columns": list(df.columns),
    }
    expected = set(_EXPECTED_COLUMNS.get(key, ()))
    if expected:
        missing = sorted(expected.difference(df.columns))
        if missing:
            summary["missing_expected_columns"] = missing
    summary["dtypes"] = {column: str(dtype) for column, dtype in df.dtypes.items()}
    return summary


def _value_counts(df: pd.DataFrame, column: str, limit: int) -> Dict[str, int]:
    counts = df[column].astype("string").value_counts(dropna=False).head(limit)
    return counts.to_dict()


def _describe_keyword_hits(df: pd.DataFrame, keywords: Iterable[str]) -> Dict[str, int]:
    description: Dict[str, int] = {}
    for column in df.columns:
        mask = df[column].astype("string").str.contains("|".join(keywords), case=False, na=False)
        hits = int(mask.sum())
        if hits:
            description[column] = hits
    return description


def inspect_datasets(
    data_dir: Optional[Path],
    *,
    head: int,
    value_count_limit: int,
    output_json: bool,
) -> None:
    """Print inspection details for all configured FlyWire CSV exports."""

    paths_map = _resolve_dataset_paths(data_dir)
    report: Dict[str, Dict[str, object]] = {}
    cached_frames: Dict[str, pd.DataFrame] = {}

    for key, dataset_path in paths_map.items():
        entry: Dict[str, object] = {
            "resolved_path": str(dataset_path),
        }
        if not dataset_path.exists():
            entry["status"] = "missing"
            report[key] = entry
            continue

        validate_file_exists(dataset_path)
        entry["file_size_mb"] = round(dataset_path.stat().st_size / 1_048_576, 2)

        preview = _preview_frame(dataset_path, head=head)
        entry["preview"] = preview.to_dict(orient="records")

        if key == "connections":
            row_count = 0
            first_chunk: Optional[pd.DataFrame] = None
            for chunk in pd.read_csv(dataset_path, compression="gzip", chunksize=1_000_000):
                if first_chunk is None:
                    first_chunk = chunk
                row_count += len(chunk)
            if first_chunk is None:
                first_chunk = preview
            entry.update(_summarise_columns(key, first_chunk))
            entry["row_count"] = int(row_count)
        else:
            full_df = _load_full_frame(dataset_path)
            cached_frames[key] = full_df
            entry.update(_summarise_columns(key, full_df))
            entry["row_count"] = int(full_df.shape[0])

            if key in {"cell_types", "classification"}:
                for column in ("cell_type", "super_class", "sub_class"):
                    if column in full_df.columns:
                        entry[f"top_{column}_values"] = _value_counts(full_df, column, value_count_limit)
                if key == "classification":
                    entry["keyword_hits"] = _describe_keyword_hits(full_df, ("kenyon", "kc", "projection", "pn", "olfactory"))

            if key == "neurons":
                for column in ("nt_type", "predicted_nt", "primary_nt", "neurotransmitter"):
                    if column in full_df.columns:
                        entry[f"top_{column}_values"] = _value_counts(full_df, column, value_count_limit)

            if key == "names" and "group" in full_df.columns:
                entry["top_groups"] = _value_counts(full_df, "group", value_count_limit)

        report[key] = entry

    # Derive PN/KC membership if possible.
    cell_types_df = report.get("cell_types")
    classification_df = report.get("classification")
    if (
        cell_types_df
        and cell_types_df.get("status") != "missing"
        and classification_df
        and classification_df.get("status") != "missing"
    ):
        try:
            cell_types_frame = cached_frames.get("cell_types")
            if cell_types_frame is None:
                cell_types_frame = _load_full_frame(Path(cell_types_df["resolved_path"]))
            classification_frame = cached_frames.get("classification")
            if classification_frame is None:
                classification_frame = _load_full_frame(Path(classification_df["resolved_path"]))
            kc_df = get_kc_neurons(cell_types_frame, classification_frame)
            pn_df = get_pn_neurons(cell_types_frame, classification_frame)
            report["kc_summary"] = {
                "count": int(kc_df.shape[0]),
                "sample_root_ids": kc_df["root_id"].astype("string").head(10).tolist(),
                "available_columns": list(kc_df.columns),
            }
            report["pn_summary"] = {
                "count": int(pn_df.shape[0]),
                "sample_root_ids": pn_df["root_id"].astype("string").head(10).tolist(),
                "available_columns": list(pn_df.columns),
            }
        except Exception as exc:  # pragma: no cover - diagnostic output only
            report["classification_error"] = str(exc)

    if output_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    for key, entry in report.items():
        print(f"\n=== {key} ===")
        for field, value in entry.items():
            if field == "preview":
                print(f"{field}:")
                preview_df = pd.DataFrame(value)
                print(preview_df.to_string(index=False))
            elif isinstance(value, dict):
                print(f"{field}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{field}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise FlyWire CSV.gz exports to diagnose schema mismatches.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing FlyWire CSV.gz files (defaults to config.paths.DATA_ROOT)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of preview rows to display per dataset.",
    )
    parser.add_argument(
        "--value-count-limit",
        type=int,
        default=10,
        help="Maximum number of value-count entries to report per categorical column.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of formatted text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_datasets(
        data_dir=args.data_dir,
        head=args.head,
        value_count_limit=args.value_count_limit,
        output_json=args.json,
    )


if __name__ == "__main__":
    main()
