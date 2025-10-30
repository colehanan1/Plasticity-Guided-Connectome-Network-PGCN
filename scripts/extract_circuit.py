"""Extract Kenyon cell subtypes, MBONs, and DANs from FlyWire FAFB exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

DEFAULT_DATA_FILES: Dict[str, str] = {
    "classification": "classification.csv.gz",
    "cell_types": "consolidated_cell_types.csv.gz",
    "neurons": "neurons.csv.gz",
    "connections": "connections_princeton.csv.gz",
}


def _load_csv(dataset_dir: Path, filename: str, label: str) -> pd.DataFrame:
    path = dataset_dir / filename
    if not path.exists():
        raise SystemExit(
            f"Required {label} file not found at {path}. Ensure the FlyWire exports are present."
        )
    df = pd.read_csv(path, compression="gzip")
    print(f"Loaded {label}: {len(df):,} rows from {path}")
    return df


def _merge_metadata(
    classification: pd.DataFrame, cell_types: pd.DataFrame, neurons: pd.DataFrame
) -> pd.DataFrame:
    cells = classification.merge(
        cell_types[["root_id", "primary_type"]],
        on="root_id",
        how="left",
        validate="one_to_one",
    )
    neuron_columns = [
        column
        for column in ("root_id", "nt_type", "group", "output_neuropils")
        if column in neurons.columns
    ]
    cells = cells.merge(neurons[neuron_columns], on="root_id", how="left", validate="one_to_one")
    cells["root_id"] = pd.to_numeric(cells["root_id"], errors="coerce").astype("Int64")
    return cells


def _filter_subtype(cells: pd.DataFrame, subtype_pattern: str) -> pd.DataFrame:
    primary_types = cells["primary_type"].astype("string")
    mask = (cells["class"] == "Kenyon_Cell") & primary_types.str.contains(
        subtype_pattern, case=False, na=False
    )
    return cells[mask].copy()


def _write_subset(df: pd.DataFrame, columns: Iterable[str], output_path: Path, label: str) -> None:
    subset = [column for column in columns if column in df.columns]
    df.loc[:, subset].to_csv(output_path, index=False)
    print(f"{label}: {len(df):,} neurons → saved to {output_path}")


def _summarise_total(total: int, label: str, subset_count: int) -> str:
    percentage = (subset_count / total * 100) if total else 0
    return f"  {label}: {subset_count:,} ({percentage:.1f}%)"


def _load_connections(dataset_dir: Path, filename: str) -> Optional[pd.DataFrame]:
    path = dataset_dir / filename
    if not path.exists():
        print(
            f"Connections file not found at {path}. Continuing without neuropil annotations."
        )
        return None

    print(f"Loading connections (this may take a moment): {path}")
    try:
        dtype = {
            "pre_root_id": "int64",
            "post_root_id": "int64",
            "neuropil": "string",
        }
        connections = pd.read_csv(
            path,
            compression="gzip",
            usecols=["pre_root_id", "post_root_id", "neuropil"],
            dtype=dtype,
            low_memory=False,
        )
    except Exception as exc:  # pragma: no cover - defensive against CSV issues
        print(
            "Failed to load connections file due to error: "
            f"{exc}. Continuing without neuropil annotations."
        )
        return None

    print(f"Loaded connections: {len(connections):,} rows from {path}")
    return connections


def _join_unique(values: pd.Series) -> str:
    strings = values.astype("string").dropna()
    unique_values = sorted({value for value in strings if value and value != "<NA>"})
    return "|".join(unique_values)


def get_output_neuropils(
    root_ids: Iterable[int], connections_df: Optional[pd.DataFrame]
) -> pd.Series:
    """Get neuropils where pre_root_id outputs."""

    if connections_df is None:
        return pd.Series(dtype="string")

    root_index = pd.Index(root_ids).dropna().unique()
    if root_index.empty:
        return pd.Series(dtype="string")

    subset = connections_df[connections_df["pre_root_id"].isin(root_index)][
        ["pre_root_id", "neuropil"]
    ]
    if subset.empty:
        return pd.Series(dtype="string")

    subset = subset.assign(neuropil=subset["neuropil"].astype("string"))
    neuropils = subset.drop_duplicates().groupby("pre_root_id")["neuropil"].apply(_join_unique)
    return neuropils


def get_input_neuropils(
    root_ids: Iterable[int], connections_df: Optional[pd.DataFrame]
) -> pd.Series:
    """Get neuropils where post_root_id receives input."""

    if connections_df is None:
        return pd.Series(dtype="string")

    root_index = pd.Index(root_ids).dropna().unique()
    if root_index.empty:
        return pd.Series(dtype="string")

    subset = connections_df[connections_df["post_root_id"].isin(root_index)][
        ["post_root_id", "neuropil"]
    ]
    if subset.empty:
        return pd.Series(dtype="string")

    subset = subset.assign(neuropil=subset["neuropil"].astype("string"))
    neuropils = subset.drop_duplicates().groupby("post_root_id")["neuropil"].apply(_join_unique)
    return neuropils


def extract_circuit(dataset_dir: Path, output_dir: Path) -> None:
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    classification = _load_csv(
        dataset_dir, DEFAULT_DATA_FILES["classification"], "classification"
    )
    cell_types = _load_csv(dataset_dir, DEFAULT_DATA_FILES["cell_types"], "cell_types")
    neurons = _load_csv(dataset_dir, DEFAULT_DATA_FILES["neurons"], "neurons")

    cells = _merge_metadata(classification, cell_types, neurons)
    connections = _load_connections(dataset_dir, DEFAULT_DATA_FILES["connections"])

    print("\n=== TASK 4: KC SUBTYPES ===")
    subtype_specs: Dict[str, Tuple[str, str]] = {
        "KCab": (r"KCab(?!-)", "kc_ab.csv"),
        "KCab-p": (r"KCab-p", "kc_ab_p.csv"),
        "KCg-m": (r"KCg-m", "kc_g_main.csv"),
        "KCg-d": (r"KCg-d", "kc_g_dorsal.csv"),
        "KCg-s": (r"KCg-s", "kc_g_sparse.csv"),
        "KCapbp-m": (r"KCapbp-m", "kc_apbp_main.csv"),
        "KCapbp-ap1": (r"KCapbp-ap1", "kc_apbp_ap1.csv"),
        "KCapbp-ap2": (r"KCapbp-ap2", "kc_apbp_ap2.csv"),
    }
    subtype_frames: Dict[str, pd.DataFrame] = {}
    for label, (pattern, filename) in subtype_specs.items():
        frame = _filter_subtype(cells, pattern)
        subtype_frames[label] = frame
        _write_subset(
            frame,
            ("root_id", "primary_type", "class", "nt_type", "group"),
            output_dir / filename,
            label,
        )

    all_kc = cells[cells["class"] == "Kenyon_Cell"].copy()
    total_kc = len(all_kc)
    print(f"\nTotal Kenyon Cells: {total_kc:,}")
    for label, frame in subtype_frames.items():
        print(_summarise_total(total_kc, label, len(frame)))

    print("\n=== TASK 5: MBONs ===")
    mbon_all = cells[cells["class"] == "MBON"].copy()
    if connections is not None:
        print("  Deriving input neuropils from connections...")
        mbon_input_neuropils = get_input_neuropils(mbon_all["root_id"], connections)
        mbon_all = mbon_all.merge(
            mbon_input_neuropils.rename("input_neuropils"),
            left_on="root_id",
            right_index=True,
            how="left",
        )
    else:
        mbon_all = mbon_all.assign(input_neuropils=pd.NA)
        print("  Neuropil derivation skipped for MBONs (connections unavailable).")

    _write_subset(
        mbon_all,
        (
            "root_id",
            "primary_type",
            "class",
            "nt_type",
            "group",
            "input_neuropils",
        ),
        output_dir / "mbon_all.csv",
        "All MBONs",
    )

    mbon_inputs = mbon_all["input_neuropils"].astype("string")
    mbon_calyx = mbon_all[mbon_inputs.str.contains("CA", case=False, na=False)].copy()
    _write_subset(
        mbon_calyx,
        (
            "root_id",
            "primary_type",
            "class",
            "nt_type",
            "group",
            "input_neuropils",
        ),
        output_dir / "mbon_calyx.csv",
        "MBONs with calyx input",
    )

    mbon_ml = mbon_all[mbon_inputs.str.contains("MB_ML", case=False, na=False)].copy()
    _write_subset(
        mbon_ml,
        (
            "root_id",
            "primary_type",
            "class",
            "nt_type",
            "group",
            "input_neuropils",
        ),
        output_dir / "mbon_ml.csv",
        "MBONs with medial lobe input",
    )

    mbon_glut = mbon_all[mbon_all["nt_type"] == "GLUT"].copy()
    _write_subset(
        mbon_glut,
        ("root_id", "primary_type", "class", "nt_type", "group"),
        output_dir / "mbon_glut.csv",
        "Glutamatergic MBONs",
    )

    print("\n=== TASK 6: DANs (Dopaminergic Neurons) ===")
    dans = cells[cells["nt_type"] == "DA"].copy()
    if connections is not None:
        print("  Deriving output neuropils from connections...")
        dan_output_neuropils = get_output_neuropils(dans["root_id"], connections)
        dans = dans.merge(
            dan_output_neuropils.rename("output_neuropils"),
            left_on="root_id",
            right_index=True,
            how="left",
        )
    else:
        dans = dans.assign(output_neuropils=pd.NA)
        print("  Neuropil derivation skipped for DANs (connections unavailable).")

    _write_subset(
        dans,
        (
            "root_id",
            "primary_type",
            "class",
            "nt_type",
            "group",
            "output_neuropils",
        ),
        output_dir / "dan_all.csv",
        "All DANs",
    )

    dan_outputs = dans["output_neuropils"].astype("string")
    dan_mb = dans[dan_outputs.str.contains("MB_", case=False, na=False)].copy()
    _write_subset(
        dan_mb,
        ("root_id", "primary_type", "class", "nt_type", "group", "output_neuropils"),
        output_dir / "dan_mb.csv",
        "DANs targeting MB",
    )

    dan_calyx = dans[dan_outputs.str.contains("MB_CA", case=False, na=False)].copy()
    _write_subset(
        dan_calyx,
        (
            "root_id",
            "primary_type",
            "class",
            "nt_type",
            "group",
            "output_neuropils",
        ),
        output_dir / "dan_calyx.csv",
        "DANs targeting calyx",
    )

    dan_medial = dans[dan_outputs.str.contains("MB_ML", case=False, na=False)].copy()
    _write_subset(
        dan_medial,
        (
            "root_id",
            "primary_type",
            "class",
            "nt_type",
            "group",
            "output_neuropils",
        ),
        output_dir / "dan_ml.csv",
        "DANs targeting medial lobes",
    )

    print("\n=== SUMMARY ===")
    print("Complete olfactory learning circuit exports created:")
    print("✅ ALPNs (reuse outputs from extract_alpn_projection_neurons.py)")
    print(f"✅ KCs ({total_kc:,} total) with subtype CSVs")
    print(f"✅ MBONs ({len(mbon_all):,} total)")
    print(f"✅ DANs ({len(dans):,} total)")
    print(f"All CSVs saved to: {output_dir.resolve()}")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "flywire",
        help="Directory containing FlyWire CSV exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "cache",
        help="Directory for generated CSV summaries.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    extract_circuit(args.dataset_dir, args.output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
