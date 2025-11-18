#!/usr/bin/env python
"""
Extract Taste Circuits from Shen et al. (2025) Paper Data

This script loads validated neuron lists and connectivity matrices from
Shen et al. (2025) Current Biology supplementary data, using connectivity
files to directly extract sweet/water circuits with synapse-level resolution.

Key advantages:
- Filters by GRN type directly in connectivity files (more reliable)
- Preserves actual synapse counts (not binary)
- Extracts from all three connectivity files (SEZ-PN, ACh-LN, GABA-LN)
- Calcium imaging-validated neurons (functionally confirmed)

Reference:
    Shen, K. et al. (2025). Functional imaging and connectome analyses reveal
    organizing principles of processing taste modality in the Drosophageal brain.
    Current Biology, 35(9), 1955-1970.e6.
    DOI: 10.1016/j.cub.2025.04.066

Usage:
    # Appetitive mode (sweet only - for PGCN model)
    python scripts/extract_from_paper_data.py \\
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

__version__ = "2.0.0"


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
    """
    print(f"\n[Loading FlyWire names from {names_file}...]")

    if not names_file.exists():
        raise FileNotFoundError(
            f"FlyWire names file not found: {names_file}\n"
            f"Download from: https://codex.flywire.ai/api/download?dataset=fafb\n"
            f"Save as: data/flywire/names.csv.gz"
        )

    names = pd.read_csv(names_file, compression='gzip')

    print(f"  ✓ Loaded {len(names):,} neurons")

    required_cols = ['root_id', 'name']
    missing = [c for c in required_cols if c not in names.columns]
    if missing:
        raise ValueError(f"Missing columns in names file: {missing}")

    return names


def load_grn_catalog(
    neuron_list_file: Path,
    mode: str = 'appetitive'
) -> pd.DataFrame:
    """
    Load GRN catalog to get root IDs for GRN names.

    This function loads the neuron catalog file ONLY to get root ID mappings.
    Actual filtering by GRN type happens in the connectivity files.

    Args:
        neuron_list_file: Path to "Neurons-list-v783.xlsx"
        mode: 'appetitive' (sweet/water) or 'full' (all GRNs)

    Returns:
        DataFrame with columns [v783, root_id, grn_type, side]
        Used to map GRN names to root IDs
    """
    print(f"\n[Loading GRN catalog from {neuron_list_file}...]")

    # Try to load GRN sheet
    grn_data = None
    for sheet_name in ['GRN', 'grn', 'Grn']:
        try:
            grn_data = pd.read_excel(
                neuron_list_file,
                sheet_name=sheet_name,
                engine='openpyxl'
            )
            print(f"  ✓ Loaded sheet: '{sheet_name}'")
            break
        except Exception:
            continue

    if grn_data is None:
        raise ValueError("Could not find GRN sheet in neuron list file")

    print(f"  [DEBUG] Columns: {grn_data.columns.tolist()}")
    print(f"  [DEBUG] Total rows: {len(grn_data)}")

    # Rename columns (flexible mapping)
    rename_map = {}
    for col in grn_data.columns:
        col_lower = col.lower().strip()
        if 'segment' in col_lower and 'id' in col_lower:
            rename_map[col] = 'root_id'
        elif col_lower in ['grn type', 'grn_type', 'type']:
            rename_map[col] = 'grn_type'
        elif col_lower == 'side':
            rename_map[col] = 'side'
        elif col_lower == 'v783':
            rename_map[col] = 'v783'

    grn_data = grn_data.rename(columns=rename_map)

    # Keep essential columns
    keep_cols = ['v783', 'root_id']
    if 'grn_type' in grn_data.columns:
        keep_cols.append('grn_type')
    if 'side' in grn_data.columns:
        keep_cols.append('side')

    grn_catalog = grn_data[keep_cols].copy()

    print(f"  ✓ Loaded {len(grn_catalog)} GRNs from catalog")
    print(f"  ℹ️  Note: Filtering by GRN type happens in connectivity files")

    return grn_catalog


def extract_from_connectivity_file(
    connectivity_file: Path,
    grn_catalog: pd.DataFrame,
    names_lookup: pd.DataFrame,
    mode: str = 'appetitive',
    min_synapses: int = 1
) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Extract downstream neurons from a single connectivity file.

    CRITICAL: This function filters by "GRN type" column DIRECTLY in the
    connectivity file, not by matching external catalog names.

    Args:
        connectivity_file: Path to connectivity Excel file
        grn_catalog: GRN catalog (for root ID lookups)
        names_lookup: FlyWire names.csv.gz mapping
        mode: 'appetitive' (sweet only) or 'full' (all GRNs)
        min_synapses: Minimum synapses required (default: 1)

    Returns:
        (neuron_df, connectivity_matrix, grn_names)
        - neuron_df: DataFrame [name, root_id, total_input_synapses]
        - connectivity_matrix: NumPy array WITH SYNAPSE COUNTS (not binary!)
        - grn_names: List of GRN names (v783) for this file

    File structure:
        - Columns 0-3: GRN metadata (GRN type, GRN #, Side, Name)
        - Columns 4+: Downstream neuron names
        - Values: Synapse counts (integers)
    """
    print(f"\n[Extracting from {connectivity_file.name}...]")

    # DEBUG: Show available sheets
    try:
        xl_file = pd.ExcelFile(connectivity_file, engine='openpyxl')
        available_sheets = xl_file.sheet_names
        print(f"  [DEBUG] Available sheets: {available_sheets}")
    except Exception as e:
        print(f"  [DEBUG] Could not list sheets: {e}")

    # Load connectivity matrix (try different sheet names)
    conn_data = None
    for sheet_name in ['raw connectivity v783', 'Raw connectivity v783', 'connectivity']:
        try:
            conn_data = pd.read_excel(
                connectivity_file,
                sheet_name=sheet_name,
                engine='openpyxl'
            )
            print(f"  ✓ Loaded sheet: '{sheet_name}'")
            break
        except Exception:
            continue

    if conn_data is None:
        raise ValueError(f"Could not find connectivity sheet. Available: {available_sheets}")

    print(f"  ✓ Loaded: {conn_data.shape[0]} GRNs × {conn_data.shape[1]-4} downstream neurons")
    print(f"  [DEBUG] Columns: {conn_data.columns.tolist()[:6]}...")

    # Find GRN type column
    grn_type_col = None
    for col in conn_data.columns[:4]:
        if 'grn' in str(col).lower() and 'type' in str(col).lower():
            grn_type_col = col
            break

    if grn_type_col is None:
        print(f"  [DEBUG] ⚠️  Could not find 'GRN type' column")
        print(f"  [DEBUG] First 4 columns: {conn_data.columns[:4].tolist()}")
        grn_type_col = conn_data.columns[0]  # Default to first column
        print(f"  [DEBUG] Using column: '{grn_type_col}'")

    # Show unique GRN types in this file
    unique_types = conn_data[grn_type_col].dropna().unique()
    print(f"  [DEBUG] GRN types in this file: {sorted(unique_types)}")

    # CRITICAL: Filter by GRN type FROM THIS FILE (not external catalog)
    if mode == 'appetitive':
        # Filter to sweet GRNs (case-insensitive, flexible matching)
        grn_filtered = conn_data[
            conn_data[grn_type_col].astype(str).str.lower().str.contains(
                'sweet|water',
                case=False,
                na=False,
                regex=True
            )
        ].copy()
        print(f"  ✓ Filtered to sweet/water GRNs: {len(grn_filtered)}")
    else:
        grn_filtered = conn_data.copy()
        print(f"  ✓ Using all GRN types: {len(grn_filtered)}")

    if len(grn_filtered) == 0:
        print(f"  [DEBUG] ⚠️  No GRNs matched filter!")
        print(f"  [DEBUG] Available types: {conn_data[grn_type_col].value_counts().to_dict()}")
        return pd.DataFrame(), np.array([]), []

    # Find name column
    name_col = None
    for col in conn_data.columns[:4]:
        if 'name' in str(col).lower():
            name_col = col
            break

    if name_col is None:
        print(f"  [DEBUG] ⚠️  Could not find 'Name' column, using column 3")
        name_col = conn_data.columns[3]

    # Get GRN names
    grn_names = grn_filtered[name_col].tolist()
    print(f"  [DEBUG] Sample GRN names: {grn_names[:3]}")

    # Extract connectivity matrix (columns 4+)
    neuron_names = grn_filtered.columns[4:]

    # CRITICAL: PRESERVE SYNAPSE COUNTS (don't binarize!)
    connectivity_matrix = grn_filtered[neuron_names].fillna(0).values.astype(int)

    print(f"  ✓ Connectivity matrix: {connectivity_matrix.shape}")
    print(f"  ✓ Total synapses: {connectivity_matrix.sum():,}")
    print(f"  [DEBUG] Synapse range: {connectivity_matrix.min()} - {connectivity_matrix.max()}")

    # Identify neurons with ≥min_synapses from filtered GRNs
    synapses_per_neuron = connectivity_matrix.sum(axis=0)
    neuron_mask = synapses_per_neuron >= min_synapses

    # Filter neurons and connectivity
    neurons_kept = neuron_names[neuron_mask]
    connectivity_kept = connectivity_matrix[:, neuron_mask]
    synapses_kept = synapses_per_neuron[neuron_mask]

    print(f"  ✓ Neurons with ≥{min_synapses} synapses: {len(neurons_kept)}")

    if len(neurons_kept) > 0:
        print(f"    Synapse stats: min={synapses_kept.min()}, "
              f"median={int(np.median(synapses_kept))}, "
              f"max={synapses_kept.max()}")

    # Build neuron DataFrame
    neuron_df = pd.DataFrame({
        'name': neurons_kept,
        'total_input_synapses': synapses_kept
    })

    # Map to root IDs
    neuron_df = neuron_df.merge(
        names_lookup[['name', 'root_id']],
        on='name',
        how='left'
    )

    # Check for unmapped neurons
    n_unmapped = neuron_df['root_id'].isna().sum()
    if n_unmapped > 0:
        print(f"  ⚠️  WARNING: {n_unmapped}/{len(neuron_df)} neurons not mapped to root IDs")

    return neuron_df, connectivity_kept, grn_names


def map_grn_names_to_root_ids(
    grn_names: list,
    grn_catalog: pd.DataFrame
) -> np.ndarray:
    """
    Map GRN names (v783) to root IDs using catalog.

    Args:
        grn_names: List of GRN names from connectivity file
        grn_catalog: GRN catalog with v783 and root_id columns

    Returns:
        Array of root IDs (same order as grn_names)
    """
    name_to_id = dict(zip(grn_catalog['v783'], grn_catalog['root_id']))

    root_ids = []
    unmapped = []

    for name in grn_names:
        if name in name_to_id:
            root_ids.append(name_to_id[name])
        else:
            root_ids.append(np.nan)
            unmapped.append(name)

    if unmapped:
        print(f"  ⚠️  {len(unmapped)} GRN names not found in catalog: {unmapped[:5]}")

    return np.array(root_ids)


def generate_validation_report(
    grn_count: int,
    sez_pn_count: int,
    ach_ln_count: int,
    gaba_ln_count: int,
    conn_grn_pn: np.ndarray,
    conn_grn_ach: np.ndarray,
    conn_grn_gaba: np.ndarray,
    mode: str,
    output_dir: Path
) -> dict:
    """
    Generate validation report with synapse statistics.
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
            'grns': int(grn_count),
            'sez_pns': int(sez_pn_count),
            'ach_lns': int(ach_ln_count),
            'gaba_lns': int(gaba_ln_count)
        },
        'synapse_statistics': {
            'grn_to_pn_total': int(conn_grn_pn.sum()),
            'grn_to_pn_mean': float(conn_grn_pn[conn_grn_pn > 0].mean()) if (conn_grn_pn > 0).any() else 0,
            'grn_to_pn_max': int(conn_grn_pn.max()),
            'grn_to_ach_total': int(conn_grn_ach.sum()),
            'grn_to_ach_mean': float(conn_grn_ach[conn_grn_ach > 0].mean()) if (conn_grn_ach > 0).any() else 0,
            'grn_to_ach_max': int(conn_grn_ach.max()),
            'grn_to_gaba_total': int(conn_grn_gaba.sum()),
            'grn_to_gaba_mean': float(conn_grn_gaba[conn_grn_gaba > 0].mean()) if (conn_grn_gaba > 0).any() else 0,
            'grn_to_gaba_max': int(conn_grn_gaba.max())
        }
    }

    # Expected counts
    if mode == 'appetitive':
        expected_grns = (30, 50)
        expected_sez_pns = (15, 35)
        expected_ach_lns = (40, 70)
        expected_gaba_lns = (25, 50)
    else:
        expected_grns = (80, 100)
        expected_sez_pns = (57, 57)
        expected_ach_lns = (82, 83)
        expected_gaba_lns = (50, 50)

    # Validate counts
    print(f"\n📊 Neuron Counts:")
    print(f"  GRNs: {grn_count}")
    if expected_grns[0] <= grn_count <= expected_grns[1]:
        print(f"    ✅ Within expected range ({expected_grns[0]}-{expected_grns[1]})")
        report['validation_grns'] = 'PASS'
    else:
        print(f"    ⚠️  Outside expected range ({expected_grns[0]}-{expected_grns[1]})")
        report['validation_grns'] = 'CHECK'

    print(f"  SEZ-PNs: {sez_pn_count}")
    if expected_sez_pns[0] <= sez_pn_count <= expected_sez_pns[1]:
        print(f"    ✅ Within expected range ({expected_sez_pns[0]}-{expected_sez_pns[1]})")
        report['validation_sez_pns'] = 'PASS'
    else:
        print(f"    ⚠️  Outside expected range ({expected_sez_pns[0]}-{expected_sez_pns[1]})")
        report['validation_sez_pns'] = 'CHECK'

    print(f"  ACh-LNs: {ach_ln_count}")
    if expected_ach_lns[0] <= ach_ln_count <= expected_ach_lns[1]:
        print(f"    ✅ Within expected range ({expected_ach_lns[0]}-{expected_ach_lns[1]})")
        report['validation_ach_lns'] = 'PASS'
    else:
        print(f"    ⚠️  Outside expected range ({expected_ach_lns[0]}-{expected_ach_lns[1]})")
        report['validation_ach_lns'] = 'CHECK'

    print(f"  GABA-LNs: {gaba_ln_count}")
    if expected_gaba_lns[0] <= gaba_ln_count <= expected_gaba_lns[1]:
        print(f"    ✅ Within expected range ({expected_gaba_lns[0]}-{expected_gaba_lns[1]})")
        report['validation_gaba_lns'] = 'PASS'
    else:
        print(f"    ⚠️  Outside expected range ({expected_gaba_lns[0]}-{expected_gaba_lns[1]})")
        report['validation_gaba_lns'] = 'CHECK'

    # Synapse statistics
    print(f"\n📈 Synapse Statistics:")
    print(f"  GRN→SEZ-PN: {report['synapse_statistics']['grn_to_pn_total']:,} total, "
          f"mean={report['synapse_statistics']['grn_to_pn_mean']:.1f}, "
          f"max={report['synapse_statistics']['grn_to_pn_max']}")
    print(f"  GRN→ACh-LN: {report['synapse_statistics']['grn_to_ach_total']:,} total, "
          f"mean={report['synapse_statistics']['grn_to_ach_mean']:.1f}, "
          f"max={report['synapse_statistics']['grn_to_ach_max']}")
    print(f"  GRN→GABA-LN: {report['synapse_statistics']['grn_to_gaba_total']:,} total, "
          f"mean={report['synapse_statistics']['grn_to_gaba_mean']:.1f}, "
          f"max={report['synapse_statistics']['grn_to_gaba_max']}")

    # Save report
    report_file = output_dir / f"shen2025_{mode}_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Validation report saved: {report_file}")

    return report


def main():
    """
    Main extraction pipeline.

    Extracts from ALL THREE connectivity files:
    1. GRN-vs-directly-connected-SEZ-PN-connectivity_final.xlsx
    2. GRN-vs-ACh-LNs-connectivity_final.xlsx
    3. GRN-vs-GABA-LNs_connectivity_final.xlsx
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
    grn_gaba_conn_file = args.paper_data_dir / 'GRN-vs-GABA-LNs_connectivity_final.xlsx'

    missing_files = []
    for f in [neuron_list_file, grn_pn_conn_file, grn_ach_conn_file, grn_gaba_conn_file, args.flywire_names]:
        if not f.exists():
            missing_files.append(str(f))

    if missing_files:
        print("❌ ERROR: Required files not found:")
        for f in missing_files:
            print(f"  - {f}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("EXTRACT TASTE CIRCUITS FROM SHEN ET AL. (2025)")
    print("Using connectivity matrices with synapse-level resolution")
    print("="*70)
    print(f"\nMode: {args.mode}")
    print(f"Paper data: {args.paper_data_dir}")
    print(f"Output: {args.output_dir}")

    try:
        # [1] Load FlyWire names
        names_lookup = load_flywire_names(args.flywire_names)

        # [2] Load GRN catalog (for root ID lookups)
        grn_catalog = load_grn_catalog(neuron_list_file, mode=args.mode)

        # [3] Extract from SEZ-PN connectivity file
        sez_pn_df, conn_grn_pn, grn_names_pn = extract_from_connectivity_file(
            grn_pn_conn_file,
            grn_catalog,
            names_lookup,
            mode=args.mode,
            min_synapses=args.min_synapses
        )

        # [4] Extract from ACh-LN connectivity file
        ach_ln_df, conn_grn_ach, grn_names_ach = extract_from_connectivity_file(
            grn_ach_conn_file,
            grn_catalog,
            names_lookup,
            mode=args.mode,
            min_synapses=args.min_synapses
        )

        # [5] Extract from GABA-LN connectivity file
        gaba_ln_df, conn_grn_gaba, grn_names_gaba = extract_from_connectivity_file(
            grn_gaba_conn_file,
            grn_catalog,
            names_lookup,
            mode=args.mode,
            min_synapses=args.min_synapses
        )

        # [6] Build unified GRN list (combine GRNs from all 3 files)
        all_grn_names = sorted(set(grn_names_pn + grn_names_ach + grn_names_gaba))
        grn_root_ids = map_grn_names_to_root_ids(all_grn_names, grn_catalog)

        grn_df = pd.DataFrame({
            'v783': all_grn_names,
            'root_id': grn_root_ids
        })

        print(f"\n[Building unified GRN list...]")
        print(f"  ✓ Unique GRNs across all files: {len(grn_df)}")
        print(f"  ✓ GRNs in SEZ-PN file: {len(grn_names_pn)}")
        print(f"  ✓ GRNs in ACh-LN file: {len(grn_names_ach)}")
        print(f"  ✓ GRNs in GABA-LN file: {len(grn_names_gaba)}")

        # [7] Export data
        prefix = f"shen2025_{args.mode}"

        print(f"\n[Exporting data...]")

        # Export neuron lists
        grn_file = args.output_dir / f"{prefix}_grn.csv"
        sez_pn_file = args.output_dir / f"{prefix}_sez_pn.csv"
        ach_ln_file = args.output_dir / f"{prefix}_sez_ln_ach.csv"
        gaba_ln_file = args.output_dir / f"{prefix}_sez_ln_gaba.csv"

        grn_df.to_csv(grn_file, index=False)
        sez_pn_df.to_csv(sez_pn_file, index=False)
        ach_ln_df.to_csv(ach_ln_file, index=False)
        gaba_ln_df.to_csv(gaba_ln_file, index=False)

        print(f"  ✓ {grn_file.name}: {len(grn_df)} neurons")
        print(f"  ✓ {sez_pn_file.name}: {len(sez_pn_df)} neurons")
        print(f"  ✓ {ach_ln_file.name}: {len(ach_ln_df)} neurons")
        print(f"  ✓ {gaba_ln_file.name}: {len(gaba_ln_df)} neurons")

        # Save connectivity matrices WITH SYNAPSE COUNTS
        conn_grn_pn_file = args.output_dir / f"{prefix}_connectivity_grn_pn.npz"
        conn_grn_ach_file = args.output_dir / f"{prefix}_connectivity_grn_ach.npz"
        conn_grn_gaba_file = args.output_dir / f"{prefix}_connectivity_grn_gaba.npz"

        # Map GRN names to indices in unified list
        grn_name_to_idx = {name: i for i, name in enumerate(all_grn_names)}

        # Reorder connectivity matrices to match unified GRN list
        pn_grn_indices = [grn_name_to_idx[name] for name in grn_names_pn if name in grn_name_to_idx]
        ach_grn_indices = [grn_name_to_idx[name] for name in grn_names_ach if name in grn_name_to_idx]
        gaba_grn_indices = [grn_name_to_idx[name] for name in grn_names_gaba if name in grn_name_to_idx]

        # Create full connectivity matrices (all GRNs × downstream neurons)
        conn_full_pn = np.zeros((len(all_grn_names), conn_grn_pn.shape[1]), dtype=int)
        conn_full_ach = np.zeros((len(all_grn_names), conn_grn_ach.shape[1]), dtype=int)
        conn_full_gaba = np.zeros((len(all_grn_names), conn_grn_gaba.shape[1]), dtype=int)

        conn_full_pn[pn_grn_indices, :] = conn_grn_pn
        conn_full_ach[ach_grn_indices, :] = conn_grn_ach
        conn_full_gaba[gaba_grn_indices, :] = conn_grn_gaba

        np.savez_compressed(
            conn_grn_pn_file,
            connectivity=conn_full_pn,
            grn_ids=grn_df['root_id'].values,
            sez_pn_ids=sez_pn_df['root_id'].values
        )

        np.savez_compressed(
            conn_grn_ach_file,
            connectivity=conn_full_ach,
            grn_ids=grn_df['root_id'].values,
            ach_ln_ids=ach_ln_df['root_id'].values
        )

        np.savez_compressed(
            conn_grn_gaba_file,
            connectivity=conn_full_gaba,
            grn_ids=grn_df['root_id'].values,
            gaba_ln_ids=gaba_ln_df['root_id'].values
        )

        print(f"  ✓ {conn_grn_pn_file.name}: {conn_full_pn.shape} matrix")
        print(f"  ✓ {conn_grn_ach_file.name}: {conn_full_ach.shape} matrix")
        print(f"  ✓ {conn_grn_gaba_file.name}: {conn_full_gaba.shape} matrix")

        # [8] Generate validation report
        report = generate_validation_report(
            len(grn_df),
            len(sez_pn_df),
            len(ach_ln_df),
            len(gaba_ln_df),
            conn_full_pn,
            conn_full_ach,
            conn_full_gaba,
            args.mode,
            args.output_dir
        )

        print("\n" + "="*70)
        print("✅ EXTRACTION COMPLETE")
        print("="*70)
        print(f"\nOutput files in {args.output_dir}:")
        print(f"  - {prefix}_grn.csv ({len(grn_df)} neurons)")
        print(f"  - {prefix}_sez_pn.csv ({len(sez_pn_df)} neurons)")
        print(f"  - {prefix}_sez_ln_ach.csv ({len(ach_ln_df)} neurons)")
        print(f"  - {prefix}_sez_ln_gaba.csv ({len(gaba_ln_df)} neurons)")
        print(f"  - {prefix}_connectivity_grn_pn.npz")
        print(f"  - {prefix}_connectivity_grn_ach.npz")
        print(f"  - {prefix}_connectivity_grn_gaba.npz")
        print(f"  - {prefix}_validation_report.json")

        # Check validation
        all_pass = all([
            report.get('validation_grns') == 'PASS',
            report.get('validation_sez_pns') == 'PASS',
            report.get('validation_ach_lns') == 'PASS',
            report.get('validation_gaba_lns') == 'PASS'
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
