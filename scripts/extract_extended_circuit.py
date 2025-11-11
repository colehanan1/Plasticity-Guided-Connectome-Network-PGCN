"""Extract extended neural components (LN, LH, Motor, AN, DN) from FlyWire FAFB exports.

This script extends the base circuit extraction to include additional neural
components required for blocking experiments:
- Local interneurons (LN): GABAergic and Cholinergic AL interneurons
- Lateral horn neurons (LH): Innate valence pathway
- Motor neurons (Motor): Proboscis extension reflex (PER) measurement
- Ascending neurons (AN): VNC→Brain sensory/state signals
- Descending neurons (DN): Brain→VNC motor commands

Usage:
    python scripts/extract_extended_circuit.py --dataset-dir data/flywire --output-dir data/cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loaders.neuron_classification import (
    get_local_interneurons,
    get_lateral_horn_neurons,
    get_motor_neurons,
    get_ascending_neurons,
    get_descending_neurons,
    get_cb0191_neurons,
    get_sez_nsc_capa_neurons,
)

DEFAULT_DATA_FILES: Dict[str, str] = {
    "classification": "classification.csv.gz",
    "cell_types": "consolidated_cell_types.csv.gz",
    "neurons": "neurons.csv.gz",
    "connections": "connections_princeton.csv.gz",
    "names": "names.csv.gz",  # For group annotations
    "processed_labels": "processed_labels.csv.gz",
}


def _load_csv(dataset_dir: Path, filename: str, label: str, required: bool = True) -> Optional[pd.DataFrame]:
    """Load a CSV file with optional requirement checking."""
    path = dataset_dir / filename
    if not path.exists():
        if required:
            raise SystemExit(
                f"Required {label} file not found at {path}. Ensure the FlyWire exports are present."
            )
        else:
            print(f"Optional {label} file not found at {path}. Skipping.")
            return None

    df = pd.read_csv(path, compression="gzip")
    print(f"Loaded {label}: {len(df):,} rows from {path}")
    return df


def _write_subset(df: pd.DataFrame, columns: Iterable[str], output_path: Path, label: str) -> None:
    """Write a subset of columns to CSV."""
    subset = [column for column in columns if column in df.columns]
    df.loc[:, subset].to_csv(output_path, index=False)
    print(f"{label}: {len(df):,} neurons → saved to {output_path}")


def _join_unique(values: pd.Series) -> str:
    """Join unique non-null values with pipe delimiter."""
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


def extract_extended_circuit(dataset_dir: Path, output_dir: Path) -> None:
    """Extract extended circuit components and save to CSV files.

    Parameters
    ----------
    dataset_dir : Path
        Directory containing FlyWire CSV exports (classification.csv.gz, etc.)
    output_dir : Path
        Directory where output CSV files will be written
    """
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load required dataframes
    print("Loading FlyWire datasets...")
    classification = _load_csv(
        dataset_dir, DEFAULT_DATA_FILES["classification"], "classification", required=True
    )
    cell_types = _load_csv(
        dataset_dir, DEFAULT_DATA_FILES["cell_types"], "cell_types", required=True
    )
    neurons = _load_csv(
        dataset_dir, DEFAULT_DATA_FILES["neurons"], "neurons", required=True
    )
    names = _load_csv(
        dataset_dir, DEFAULT_DATA_FILES["names"], "names", required=False
    )
    processed_labels = _load_csv(
        dataset_dir, DEFAULT_DATA_FILES["processed_labels"], "processed_labels", required=False
    )
    connections = _load_csv(
        dataset_dir, DEFAULT_DATA_FILES["connections"], "connections", required=False
    )

    # Extract Local Interneurons (LN)
    print("\n=== EXTRACTING LOCAL INTERNEURONS (LN) ===")
    ln_neurons = get_local_interneurons(
        cell_types, classification, names_df=names, neurons_df=neurons
    )

    if len(ln_neurons) == 0:
        print("WARNING: No local interneurons found. Check FlyWire data filters.")
    else:
        # Separate by neurotransmitter type
        ln_all_columns = (
            "root_id", "cell_type", "class", "super_class", "nt_type", "group"
        )
        _write_subset(ln_neurons, ln_all_columns, output_dir / "ln_all.csv", "All LNs")

        # GABAergic LNs (inhibitory)
        if "nt_type" in ln_neurons.columns:
            ln_gaba = ln_neurons[ln_neurons["nt_type"] == "GABA"].copy()
            _write_subset(ln_gaba, ln_all_columns, output_dir / "ln_gaba.csv", "GABAergic LNs")

            # Cholinergic LNs (excitatory)
            ln_chol = ln_neurons[ln_neurons["nt_type"].isin(["ACH", "CHOL"])].copy()
            _write_subset(ln_chol, ln_all_columns, output_dir / "ln_chol.csv", "Cholinergic LNs")

            print(f"  Total LNs: {len(ln_neurons):,}")
            print(f"  GABAergic: {len(ln_gaba):,} ({len(ln_gaba)/len(ln_neurons)*100:.1f}%)")
            print(f"  Cholinergic: {len(ln_chol):,} ({len(ln_chol)/len(ln_neurons)*100:.1f}%)")

    # Extract Lateral Horn Neurons (LH)
    print("\n=== EXTRACTING LATERAL HORN NEURONS (LH) ===")
    lh_neurons = get_lateral_horn_neurons(cell_types, classification, names_df=names)

    if len(lh_neurons) == 0:
        print("WARNING: No lateral horn neurons found. Check FlyWire data filters.")
    else:
        lh_columns = ("root_id", "cell_type", "class", "super_class", "group")
        _write_subset(lh_neurons, lh_columns, output_dir / "lh_all.csv", "All LH neurons")

        # Separate LHLN (local) from LHCENT (output)
        if "class" in lh_neurons.columns:
            lh_local = lh_neurons[lh_neurons["class"].str.contains("lhln", case=False, na=False)].copy()
            _write_subset(lh_local, lh_columns, output_dir / "lh_local.csv", "LH local neurons (LHLN)")

            lh_output = lh_neurons[lh_neurons["class"].str.contains("lhcent", case=False, na=False)].copy()
            _write_subset(lh_output, lh_columns, output_dir / "lh_output.csv", "LH output neurons (LHCENT)")

            print(f"  Total LH: {len(lh_neurons):,}")
            print(f"  Local (LHLN): {len(lh_local):,} ({len(lh_local)/len(lh_neurons)*100:.1f}%)")
            print(f"  Output (LHCENT): {len(lh_output):,} ({len(lh_output)/len(lh_neurons)*100:.1f}%)")

    # Extract Motor Neurons (Motor)
    print("\n=== EXTRACTING MOTOR NEURONS (Motor) ===")
    motor_all = get_motor_neurons(
        cell_types,
        classification,
        names_df=names,
        proboscis_only=False,
    )
    motor_proboscis = get_motor_neurons(
        cell_types,
        classification,
        names_df=names,
        processed_labels_df=processed_labels,
        proboscis_only=True,
    )

    if len(motor_all) == 0:
        print("WARNING: No motor neurons found. Check FlyWire data filters.")
        print("  Checking classification columns...")
        if "super_class" in classification.columns:
            motor_vals = classification[classification["super_class"].str.contains("motor", case=False, na=False)]
            print(f"  Found {len(motor_vals)} rows with 'motor' in super_class")
    else:
        motor_columns = ("root_id", "cell_type", "super_class", "sub_class")
        _write_subset(motor_all, motor_columns, output_dir / "motor_all.csv", "All motor neurons")
        _write_subset(
            motor_proboscis,
            motor_columns,
            output_dir / "motor_proboscis.csv",
            "Proboscis motor neurons"
        )

        print(f"  Total motor: {len(motor_all):,}")
        if len(motor_all) > 0:
            print(f"  Proboscis: {len(motor_proboscis):,} ({len(motor_proboscis)/len(motor_all)*100:.1f}%)")
        else:
            print(f"  Proboscis: {len(motor_proboscis):,}")

    if len(motor_proboscis) == 0:
        print("  WARNING: No proboscis motor neurons found!")
        print("  Checking for 'proboscis' in cell_types fields...")
        if "cell_type" in cell_types.columns:
            proboscis_check = cell_types[cell_types["cell_type"].str.contains("proboscis", case=False, na=False)]
            print(f"  Found {len(proboscis_check)} entries with 'proboscis' in cell_type")

    # Extract Ascending Neurons (AN)
    print("\n=== EXTRACTING ASCENDING NEURONS (AN) ===")
    an_neurons = get_ascending_neurons(cell_types, classification)

    if len(an_neurons) == 0:
        print("WARNING: No ascending neurons found. Check FlyWire data filters.")
    else:
        an_columns = ("root_id", "cell_type", "class", "super_class")
        _write_subset(an_neurons, an_columns, output_dir / "an_all.csv", "All ascending neurons")

        # Add input/output neuropils if connections available
        if connections is not None:
            print("  Deriving neuropils from connections...")
            an_input_neuropils = get_input_neuropils(an_neurons["root_id"], connections)
            an_output_neuropils = get_output_neuropils(an_neurons["root_id"], connections)

            an_neurons = an_neurons.merge(
                an_input_neuropils.rename("input_neuropils"),
                left_on="root_id",
                right_index=True,
                how="left",
            ).merge(
                an_output_neuropils.rename("output_neuropils"),
                left_on="root_id",
                right_index=True,
                how="left",
            )

            an_full_columns = (*an_columns, "input_neuropils", "output_neuropils")
            _write_subset(
                an_neurons,
                an_full_columns,
                output_dir / "an_all_neuropils.csv",
                "ANs with neuropils"
            )

        print(f"  Total ANs: {len(an_neurons):,}")

    # Extract Descending Neurons (DN)
    print("\n=== EXTRACTING DESCENDING NEURONS (DN) ===")
    dn_neurons = get_descending_neurons(cell_types, classification)

    if len(dn_neurons) == 0:
        print("WARNING: No descending neurons found. Check FlyWire data filters.")
        print("  Checking classification columns...")
        if "super_class" in classification.columns:
            desc_vals = classification[classification["super_class"].str.contains("descending", case=False, na=False)]
            print(f"  Found {len(desc_vals)} rows with 'descending' in super_class")
            if len(desc_vals) > 0:
                print(f"  Sample super_class values: {desc_vals['super_class'].unique()[:5]}")
    else:
        dn_columns = ("root_id", "cell_type", "super_class", "class")
        _write_subset(dn_neurons, dn_columns, output_dir / "dn_all.csv", "All descending neurons")

        # Add input/output neuropils if connections available
        if connections is not None:
            print("  Deriving neuropils from connections...")
            dn_input_neuropils = get_input_neuropils(dn_neurons["root_id"], connections)
            dn_output_neuropils = get_output_neuropils(dn_neurons["root_id"], connections)

            dn_neurons = dn_neurons.merge(
                dn_input_neuropils.rename("input_neuropils"),
                left_on="root_id",
                right_index=True,
                how="left",
            ).merge(
                dn_output_neuropils.rename("output_neuropils"),
                left_on="root_id",
                right_index=True,
                how="left",
            )

            dn_full_columns = (*dn_columns, "input_neuropils", "output_neuropils")
            _write_subset(
                dn_neurons,
                dn_full_columns,
                output_dir / "dn_all_neuropils.csv",
                "DNs with neuropils"
            )

        print(f"  Total DNs: {len(dn_neurons):,}")

    # Extract CB0191 Neurons
    print("\n=== EXTRACTING CB0191 NEURONS ===")
    cb0191_neurons = get_cb0191_neurons(
        cell_types,
        classification,
        names_df=names,
        processed_labels_df=processed_labels,
    )

    if len(cb0191_neurons) == 0:
        print("WARNING: No CB0191 neurons found! Check classification fields.")
        print("  Checking for 'CB0191' in cell_types fields...")
        if "cell_type" in cell_types.columns:
            cb0191_check = cell_types[cell_types["cell_type"].str.contains("cb0191|cb-0191", case=False, na=False)]
            print(f"  Found {len(cb0191_check)} entries with 'CB0191' in cell_type")
        if processed_labels is not None and "processed_labels" in processed_labels.columns:
            cb0191_labels = processed_labels[processed_labels["processed_labels"].astype(str).str.contains("cb0191|cb-0191", case=False, na=False)]
            print(f"  Found {len(cb0191_labels)} entries with 'CB0191' in processed_labels")
    else:
        cb0191_columns = ("root_id", "cell_type", "super_class", "class", "sub_class")
        _write_subset(cb0191_neurons, cb0191_columns, output_dir / "cb0191_neurons.csv", "CB0191 neurons")
        print(f"  Total CB0191: {len(cb0191_neurons):,}")

    # Extract SEZ-NSC^CAPA Neurons
    print("\n=== EXTRACTING SEZ-NSC^CAPA NEURONS ===")
    sez_nsc_capa = get_sez_nsc_capa_neurons(
        cell_types,
        classification,
        names_df=names,
        neurons_df=neurons,
        processed_labels_df=processed_labels,
    )

    if len(sez_nsc_capa) == 0:
        print("WARNING: No SEZ-NSC^CAPA neurons found! Check endocrine classification.")
        print("  Checking for endocrine cells...")
        if "super_class" in classification.columns:
            endocrine_check = classification[classification["super_class"].str.contains("endocrine", case=False, na=False)]
            print(f"  Found {len(endocrine_check)} endocrine cells in classification")
            if len(endocrine_check) > 0 and "cell_type" in endocrine_check.columns:
                capa_in_endocrine = endocrine_check[endocrine_check["cell_type"].astype(str).str.contains("capa", case=False, na=False)]
                print(f"  Found {len(capa_in_endocrine)} with 'CAPA' in cell_type")
    else:
        sez_columns = ("root_id", "cell_type", "super_class", "class", "group")
        _write_subset(sez_nsc_capa, sez_columns, output_dir / "sez_nsc_capa.csv", "SEZ-NSC^CAPA neurons")

        expected_count = 2
        actual_count = len(sez_nsc_capa)
        if actual_count != expected_count:
            print(f"  WARNING: Expected {expected_count} SEZ-NSC^CAPA neurons, found {actual_count}")
        else:
            print(f"  ✓ SEZ-NSC^CAPA: {actual_count} neurons (expected count)")

    # Summary
    print("\n=== SUMMARY ===")
    print("Extended olfactory learning circuit components extracted:")
    print(f"✅ Local Interneurons (LN): {len(ln_neurons):,}")
    print(f"✅ Lateral Horn (LH): {len(lh_neurons):,}")
    print(f"✅ Motor Neurons: {len(motor_all):,} ({len(motor_proboscis):,} proboscis)")
    print(f"✅ Ascending Neurons (AN): {len(an_neurons):,}")
    print(f"✅ Descending Neurons (DN): {len(dn_neurons):,}")
    print(f"✅ CB0191 Neurons: {len(cb0191_neurons):,}")
    print(f"✅ SEZ-NSC^CAPA Neurons: {len(sez_nsc_capa):,}")
    print(f"\nAll CSVs saved to: {output_dir.resolve()}")
    print("\nNext steps:")
    print("1. Run scripts/extract_circuit.py for core PN/KC/MBON/DAN extraction")
    print("2. Use CircuitLoader to load extended connectivity matrix")
    print("3. Run blocking experiments with veto mechanisms")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    """Main entry point for script execution."""
    args = _parse_args(argv)
    extract_extended_circuit(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
