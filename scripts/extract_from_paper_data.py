#!/usr/bin/env python
"""
Extract Taste Circuits from Shen et al. (2025) Paper Data

This script loads validated neuron lists and connectivity matrices from
Shen et al. (2025) Current Biology supplementary data, replacing FlyWire
query-based extraction with ground-truth experimental data.

Key advantages:
- Calcium imaging-validated neurons (functionally confirmed)
- Taste modality assignments (sugar vs bitter vs water)
- Pre-built connectivity matrices (no FlyWire querying)
- Published & peer-reviewed (reproducible ground truth)

Reference:
    Shen, K. et al. (2025). Functional imaging and connectome analyses reveal
    organizing principles of processing taste modality in the Drosophila brain.
    Current Biology, 35(9), 1955-1970.e6.
    DOI: 10.1016/j.cub.2025.04.066

Usage:
    # Appetitive mode (sugar only - for PGCN model)
    python scripts/extract_from_paper_data.py \\
      --paper-data-dir data/10.1016 \\
      --flywire-names data/flywire/names.csv.gz \\
      --output-dir data/cache \\
      --mode appetitive

    # Full mode (all taste - for validation)
    python scripts/extract_from_paper_data.py --mode full
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

__version__ = "1.0.0"


def load_flywire_names(names_file: Path) -> pd.DataFrame:
    """
    Load FlyWire neuron name → root ID mapping.

    Args:
        names_file: Path to names.csv.gz from FlyWire Codex

    Returns:
        DataFrame with columns [root_id, name, group]

    Source:
        https://codex.flywire.ai/api/download?dataset=fafb
        "Proofread Cell Names And Groups (1,182 KB)"

    Note:
        This file has 139,255 neurons. We only need SEZ-related neurons,
        but loading the full file is fast (~0.5 sec) and avoids filtering issues.
    """
    print(f"\n[Loading FlyWire names from {names_file}...]")

    # Check if file exists
    if not names_file.exists():
        raise FileNotFoundError(
            f"FlyWire names file not found: {names_file}\n"
            f"Download from: https://codex.flywire.ai/api/download?dataset=fafb\n"
            f"Save as: data/flywire/names.csv.gz"
        )

    # Load compressed CSV
    names = pd.read_csv(names_file, compression='gzip')

    print(f"  ✓ Loaded {len(names):,} neurons")
    print(f"  ✓ Columns: {names.columns.tolist()}")

    # Validate expected columns
    required_cols = ['root_id', 'name']
    missing = [c for c in required_cols if c not in names.columns]
    if missing:
        raise ValueError(f"Missing columns in names file: {missing}")

    return names


def load_grns(
    neuron_list_file: Path,
    mode: str = 'appetitive'
) -> pd.DataFrame:
    """
    Load GRN list from Shen et al. (2025) supplementary data.

    Args:
        neuron_list_file: Path to "Neurons-list-v783.xlsx"
        mode: 'appetitive' (sweet only) or 'full' (all GRNs)

    Returns:
        DataFrame with columns:
        - v783: Neuron name in FlyWire v783
        - root_id: FlyWire segment ID (renamed from "segment ID")
        - grn_type: Taste modality (sweet, bitter, IR94e, etc.)
        - side: Left/right hemisphere

    Filtering:
        - mode='appetitive': ONLY GRN type == "sweet"
        - mode='full': All GRN types

    Reference:
        Shen et al. (2025) Figure 1: "Sweet GRNs activate approach circuits"
    """
    print(f"\n[Loading GRNs from {neuron_list_file}...]")

    # Load GRN sheet
    grn_data = pd.read_excel(
        neuron_list_file,
        sheet_name='GRN',
        engine='openpyxl'
    )

    # Rename columns for clarity
    grn_data = grn_data.rename(columns={
        'segment ID': 'root_id',
        'GRN type': 'grn_type',
        'Side': 'side'
    })

    # Filter by mode
    if mode == 'appetitive':
        grn_filtered = grn_data[
            grn_data['grn_type'].str.lower() == 'sweet'
        ].copy()
        print(f"  ✓ Mode: Appetitive (sugar only)")
        print(f"  ✓ Filtered to sweet GRNs: {len(grn_filtered)}")

    elif mode == 'full':
        grn_filtered = grn_data.copy()
        print(f"  ✓ Mode: Full gustatory")
        print(f"  ✓ All GRN types: {len(grn_filtered)}")

        # Show breakdown by type
        type_counts = grn_filtered['grn_type'].value_counts()
        print(f"\n  GRN type distribution:")
        for grn_type, count in type_counts.items():
            print(f"    {grn_type}: {count}")

    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'appetitive' or 'full'")

    # Validate root IDs exist
    n_missing = grn_filtered['root_id'].isna().sum()
    if n_missing > 0:
        print(f"  ⚠️  WARNING: {n_missing} GRNs missing root IDs")

    # Keep essential columns
    grn_output = grn_filtered[['v783', 'root_id', 'grn_type', 'side']].copy()

    return grn_output


def extract_sez_pns(
    connectivity_file: Path,
    grn_filter: pd.DataFrame,
    names_lookup: pd.DataFrame,
    min_synapses: int = 1
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract SEZ-PNs receiving input from filtered GRNs.

    Args:
        connectivity_file: "GRN-vs-directly-connected-SEZ-PN-connectivity_final.xlsx"
        grn_filter: Filtered GRN DataFrame from load_grns()
        names_lookup: FlyWire names.csv.gz mapping
        min_synapses: Minimum synapses required (default: 1)

    Returns:
        (sez_pn_df, connectivity_matrix)
        - sez_pn_df: DataFrame with columns [name, root_id, total_input_synapses]
        - connectivity_matrix: NumPy array (n_grns × n_sez_pns)

    Filtering logic:
        1. Load full connectivity matrix
        2. Filter ROWS to GRNs in grn_filter
        3. Filter COLUMNS to SEZ-PNs with ≥min_synapses from any filtered GRN

    Example:
        If grn_filter has 35 sweet GRNs, and 22 SEZ-PNs receive ≥1 synapse
        from sweet GRNs, returns (22 SEZ-PNs, 35×22 connectivity matrix)
    """
    print(f"\n[Extracting SEZ-PNs from {connectivity_file}...]")

    # Load connectivity matrix
    conn_data = pd.read_excel(
        connectivity_file,
        sheet_name='raw connectivity v783',
        engine='openpyxl'
    )

    print(f"  ✓ Loaded connectivity: {conn_data.shape[0]} GRNs × {conn_data.shape[1]-4} SEZ-PNs")

    # Filter rows to matching GRNs
    grn_names_to_keep = set(grn_filter['v783'].values)

    # Match by "Name" column (v783 name)
    conn_filtered = conn_data[
        conn_data['Name'].isin(grn_names_to_keep)
    ].copy()

    print(f"  ✓ Filtered to {len(conn_filtered)} GRNs matching filter")

    # Extract connectivity matrix (columns 5+)
    sez_pn_names = conn_filtered.columns[4:]  # Skip first 4 metadata columns
    connectivity_matrix = conn_filtered[sez_pn_names].fillna(0).values

    print(f"  ✓ Connectivity matrix: {connectivity_matrix.shape}")

    # Identify SEZ-PNs with ≥min_synapses from any GRN
    syn_per_pn = connectivity_matrix.sum(axis=0)  # Sum across GRNs for each PN
    pn_mask = syn_per_pn >= min_synapses

    sez_pn_names_filtered = sez_pn_names[pn_mask]
    connectivity_filtered = connectivity_matrix[:, pn_mask]

    print(f"  ✓ SEZ-PNs with ≥{min_synapses} synapses: {len(sez_pn_names_filtered)}")

    # Map SEZ-PN names to root IDs
    sez_pn_df = pd.DataFrame({
        'name': sez_pn_names_filtered,
        'total_input_synapses': syn_per_pn[pn_mask]
    })

    # Merge with names lookup
    sez_pn_df = sez_pn_df.merge(
        names_lookup[['name', 'root_id']],
        on='name',
        how='left'
    )

    # Check for unmapped neurons
    n_unmapped = sez_pn_df['root_id'].isna().sum()
    if n_unmapped > 0:
        print(f"  ⚠️  WARNING: {n_unmapped} SEZ-PNs not found in names.csv.gz")
        unmapped_names = sez_pn_df[sez_pn_df['root_id'].isna()]['name'].tolist()
        print(f"     Unmapped neurons: {unmapped_names[:5]}" +
              (f" ... and {len(unmapped_names)-5} more" if len(unmapped_names) > 5 else ""))

    return sez_pn_df, connectivity_filtered


def extract_ach_lns(
    connectivity_file: Path,
    grn_filter: pd.DataFrame,
    names_lookup: pd.DataFrame,
    min_synapses: int = 1
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract cholinergic local neurons receiving input from filtered GRNs.

    Same logic as extract_sez_pns but for ACh-LNs.

    Reference:
        Shen et al. (2025) Figure 2: "Cholinergic LNs relay sweet signals"
    """
    print(f"\n[Extracting ACh-LNs from {connectivity_file}...]")

    # Load connectivity matrix
    conn_data = pd.read_excel(
        connectivity_file,
        sheet_name='raw connectivity v783',
        engine='openpyxl'
    )

    print(f"  ✓ Loaded connectivity: {conn_data.shape[0]} GRNs × {conn_data.shape[1]-4} ACh-LNs")

    # Filter rows to matching GRNs
    grn_names_to_keep = set(grn_filter['v783'].values)

    conn_filtered = conn_data[
        conn_data['Name'].isin(grn_names_to_keep)
    ].copy()

    print(f"  ✓ Filtered to {len(conn_filtered)} GRNs matching filter")

    # Extract connectivity matrix (columns 5+)
    ach_ln_names = conn_filtered.columns[4:]
    connectivity_matrix = conn_filtered[ach_ln_names].fillna(0).values

    print(f"  ✓ Connectivity matrix: {connectivity_matrix.shape}")

    # Identify ACh-LNs with ≥min_synapses from any GRN
    syn_per_ln = connectivity_matrix.sum(axis=0)
    ln_mask = syn_per_ln >= min_synapses

    ach_ln_names_filtered = ach_ln_names[ln_mask]
    connectivity_filtered = connectivity_matrix[:, ln_mask]

    print(f"  ✓ ACh-LNs with ≥{min_synapses} synapses: {len(ach_ln_names_filtered)}")

    # Map ACh-LN names to root IDs
    ach_ln_df = pd.DataFrame({
        'name': ach_ln_names_filtered,
        'total_input_synapses': syn_per_ln[ln_mask]
    })

    # Merge with names lookup
    ach_ln_df = ach_ln_df.merge(
        names_lookup[['name', 'root_id']],
        on='name',
        how='left'
    )

    # Check for unmapped neurons
    n_unmapped = ach_ln_df['root_id'].isna().sum()
    if n_unmapped > 0:
        print(f"  ⚠️  WARNING: {n_unmapped} ACh-LNs not found in names.csv.gz")
        unmapped_names = ach_ln_df[ach_ln_df['root_id'].isna()]['name'].tolist()
        print(f"     Unmapped neurons: {unmapped_names[:5]}" +
              (f" ... and {len(unmapped_names)-5} more" if len(unmapped_names) > 5 else ""))

    return ach_ln_df, connectivity_filtered


def generate_validation_report(
    grn_df: pd.DataFrame,
    sez_pn_df: pd.DataFrame,
    ach_ln_df: pd.DataFrame,
    mode: str,
    output_dir: Path
) -> dict:
    """
    Generate validation report comparing extraction to paper expectations.

    Checks:
    1. Neuron counts within expected ranges
    2. Root ID mapping success rate
    3. Connectivity statistics
    4. Comparison to Shen et al. (2025) reported counts

    Outputs:
    - JSON validation report
    - Console summary
    """
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)

    report = {
        'extraction_mode': mode,
        'timestamp': datetime.now().isoformat(),
        'source': 'Shen et al. (2025) Current Biology 35(9):1955-1970',
        'flywire_version': 'v783',
        'neuron_counts': {
            'grns': int(len(grn_df)),
            'sez_pns': int(len(sez_pn_df)),
            'ach_lns': int(len(ach_ln_df))
        },
        'root_id_mapping': {
            'grns_mapped': int((~grn_df['root_id'].isna()).sum()),
            'sez_pns_mapped': int((~sez_pn_df['root_id'].isna()).sum()),
            'ach_lns_mapped': int((~ach_ln_df['root_id'].isna()).sum())
        }
    }

    # Expected counts (from paper)
    if mode == 'appetitive':
        expected_grns = (30, 50)
        expected_sez_pns = (15, 35)
        expected_ach_lns = (25, 50)
    else:  # full
        expected_grns = (120, 150)
        expected_sez_pns = (57, 57)  # Exactly 57 in paper
        expected_ach_lns = (83, 83)  # Exactly 83 in paper

    # Validate counts
    print(f"\n📊 Neuron Counts:")
    print(f"  GRNs: {len(grn_df)}")

    if expected_grns[0] <= len(grn_df) <= expected_grns[1]:
        print(f"    ✅ Within expected range ({expected_grns[0]}-{expected_grns[1]})")
        report['validation_grns'] = 'PASS'
    else:
        print(f"    ⚠️  Outside expected range ({expected_grns[0]}-{expected_grns[1]})")
        report['validation_grns'] = 'CHECK'

    print(f"  SEZ-PNs: {len(sez_pn_df)}")
    if expected_sez_pns[0] <= len(sez_pn_df) <= expected_sez_pns[1]:
        print(f"    ✅ Within expected range ({expected_sez_pns[0]}-{expected_sez_pns[1]})")
        report['validation_sez_pns'] = 'PASS'
    else:
        print(f"    ⚠️  Outside expected range ({expected_sez_pns[0]}-{expected_sez_pns[1]})")
        report['validation_sez_pns'] = 'CHECK'

    print(f"  ACh-LNs: {len(ach_ln_df)}")
    if expected_ach_lns[0] <= len(ach_ln_df) <= expected_ach_lns[1]:
        print(f"    ✅ Within expected range ({expected_ach_lns[0]}-{expected_ach_lns[1]})")
        report['validation_ach_lns'] = 'PASS'
    else:
        print(f"    ⚠️  Outside expected range ({expected_ach_lns[0]}-{expected_ach_lns[1]})")
        report['validation_ach_lns'] = 'CHECK'

    # Root ID mapping rates
    print(f"\n📍 Root ID Mapping:")
    grn_rate = report['root_id_mapping']['grns_mapped'] / len(grn_df) * 100 if len(grn_df) > 0 else 0
    pn_rate = report['root_id_mapping']['sez_pns_mapped'] / len(sez_pn_df) * 100 if len(sez_pn_df) > 0 else 0
    ln_rate = report['root_id_mapping']['ach_lns_mapped'] / len(ach_ln_df) * 100 if len(ach_ln_df) > 0 else 0

    print(f"  GRNs: {report['root_id_mapping']['grns_mapped']}/{len(grn_df)} ({grn_rate:.1f}%)")
    print(f"  SEZ-PNs: {report['root_id_mapping']['sez_pns_mapped']}/{len(sez_pn_df)} ({pn_rate:.1f}%)")
    print(f"  ACh-LNs: {report['root_id_mapping']['ach_lns_mapped']}/{len(ach_ln_df)} ({ln_rate:.1f}%)")

    if grn_rate >= 95 and pn_rate >= 95 and ln_rate >= 95:
        print(f"    ✅ All mapping rates >95%")
        report['validation_mapping'] = 'PASS'
    else:
        print(f"    ⚠️  Some mapping rates <95%")
        report['validation_mapping'] = 'CHECK'

    # Save report
    report_file = output_dir / f"shen2025_{mode}_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Validation report saved: {report_file}")

    return report


def main():
    """
    Main extraction pipeline.

    Usage:
        # Appetitive mode (sugar only - for PGCN model)
        python scripts/extract_from_paper_data.py \\
          --paper-data-dir data/10.1016 \\
          --flywire-names data/flywire/names.csv.gz \\
          --output-dir data/cache \\
          --mode appetitive

        # Full mode (all taste - for validation)
        python scripts/extract_from_paper_data.py --mode full
    """
    parser = argparse.ArgumentParser(
        description='Extract taste circuits from Shen et al. (2025) paper data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract sugar circuits only (for PGCN appetitive learning)
  python scripts/extract_from_paper_data.py --mode appetitive

  # Extract all taste circuits (for validation)
  python scripts/extract_from_paper_data.py --mode full

  # Custom data locations
  python scripts/extract_from_paper_data.py \\
    --paper-data-dir /path/to/paper/data \\
    --flywire-names /path/to/names.csv.gz \\
    --output-dir /path/to/output
        """
    )
    parser.add_argument(
        '--paper-data-dir',
        type=Path,
        default=Path('data/10.1016'),
        help='Directory containing Shen et al. (2025) Excel files'
    )
    parser.add_argument(
        '--flywire-names',
        type=Path,
        default=Path('data/flywire/names.csv.gz'),
        help='FlyWire names.csv.gz mapping file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/cache'),
        help='Output directory for extracted data'
    )
    parser.add_argument(
        '--mode',
        choices=['appetitive', 'full'],
        default='appetitive',
        help='Extraction mode: appetitive (sugar only) or full (all taste)'
    )
    parser.add_argument(
        '--min-synapses',
        type=int,
        default=1,
        help='Minimum synapses required for connectivity (default: 1)'
    )

    args = parser.parse_args()

    # Validate input files exist
    neuron_list_file = args.paper_data_dir / 'Neurons-list-v783.xlsx'
    grn_pn_conn_file = args.paper_data_dir / 'GRN-vs-directly-connected-SEZ-PN-connectivity_final.xlsx'
    grn_ach_conn_file = args.paper_data_dir / 'GRN-vs-ACh-LNs-connectivity_final.xlsx'

    missing_files = []
    for f in [neuron_list_file, grn_pn_conn_file, grn_ach_conn_file, args.flywire_names]:
        if not f.exists():
            missing_files.append(str(f))

    if missing_files:
        print("❌ ERROR: Required files not found:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure Shen et al. (2025) supplementary files are in:")
        print(f"  {args.paper_data_dir}")
        print("\nExpected files:")
        print("  - Neurons-list-v783.xlsx")
        print("  - GRN-vs-directly-connected-SEZ-PN-connectivity_final.xlsx")
        print("  - GRN-vs-ACh-LNs-connectivity_final.xlsx")
        print("\nAnd FlyWire names file:")
        print(f"  - {args.flywire_names}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("EXTRACT TASTE CIRCUITS FROM SHEN ET AL. (2025)")
    print("="*70)
    print(f"\nMode: {args.mode}")
    print(f"Paper data: {args.paper_data_dir}")
    print(f"Output: {args.output_dir}")

    try:
        # [1] Load FlyWire names
        names_lookup = load_flywire_names(args.flywire_names)

        # [2] Load and filter GRNs
        grn_df = load_grns(neuron_list_file, mode=args.mode)

        # [3] Extract SEZ-PNs
        sez_pn_df, conn_grn_pn = extract_sez_pns(
            grn_pn_conn_file,
            grn_df,
            names_lookup,
            min_synapses=args.min_synapses
        )

        # [4] Extract ACh-LNs
        ach_ln_df, conn_grn_ach = extract_ach_lns(
            grn_ach_conn_file,
            grn_df,
            names_lookup,
            min_synapses=args.min_synapses
        )

        # [5] Export data
        prefix = f"shen2025_{args.mode}"

        # Export neuron lists
        grn_file = args.output_dir / f"{prefix}_grn.csv"
        sez_pn_file = args.output_dir / f"{prefix}_sez_pn.csv"
        ach_ln_file = args.output_dir / f"{prefix}_sez_ln_ach.csv"

        grn_df.to_csv(grn_file, index=False)
        sez_pn_df.to_csv(sez_pn_file, index=False)
        ach_ln_df.to_csv(ach_ln_file, index=False)

        print(f"\n[Exporting data...]")
        print(f"  ✓ {grn_file.name}: {len(grn_df)} neurons")
        print(f"  ✓ {sez_pn_file.name}: {len(sez_pn_df)} neurons")
        print(f"  ✓ {ach_ln_file.name}: {len(ach_ln_df)} neurons")

        # Save connectivity matrices
        conn_grn_pn_file = args.output_dir / f"{prefix}_connectivity_grn_pn.npz"
        conn_grn_ach_file = args.output_dir / f"{prefix}_connectivity_grn_ach.npz"

        np.savez_compressed(
            conn_grn_pn_file,
            connectivity=conn_grn_pn,
            grn_ids=grn_df['root_id'].values,
            sez_pn_ids=sez_pn_df['root_id'].values
        )

        np.savez_compressed(
            conn_grn_ach_file,
            connectivity=conn_grn_ach,
            grn_ids=grn_df['root_id'].values,
            ach_ln_ids=ach_ln_df['root_id'].values
        )

        print(f"  ✓ {conn_grn_pn_file.name}: {conn_grn_pn.shape} matrix")
        print(f"  ✓ {conn_grn_ach_file.name}: {conn_grn_ach.shape} matrix")

        # [6] Generate validation report
        report = generate_validation_report(
            grn_df, sez_pn_df, ach_ln_df, args.mode, args.output_dir
        )

        print("\n" + "="*70)
        print("✅ EXTRACTION COMPLETE")
        print("="*70)
        print(f"\nOutput files in {args.output_dir}:")
        print(f"  - {prefix}_grn.csv ({len(grn_df)} neurons)")
        print(f"  - {prefix}_sez_pn.csv ({len(sez_pn_df)} neurons)")
        print(f"  - {prefix}_sez_ln_ach.csv ({len(ach_ln_df)} neurons)")
        print(f"  - {prefix}_connectivity_grn_pn.npz")
        print(f"  - {prefix}_connectivity_grn_ach.npz")
        print(f"  - {prefix}_validation_report.json")

        # Check if all validations passed
        all_pass = all([
            report.get('validation_grns') == 'PASS',
            report.get('validation_sez_pns') == 'PASS',
            report.get('validation_ach_lns') == 'PASS',
            report.get('validation_mapping') == 'PASS'
        ])

        if all_pass:
            print("\n✅ All validation checks PASSED")
            return 0
        else:
            print("\n⚠️  Some validation checks need review (see report)")
            return 0

    except Exception as e:
        print(f"\n❌ ERROR: Extraction failed")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
