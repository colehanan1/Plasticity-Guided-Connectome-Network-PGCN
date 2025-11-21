#!/usr/bin/env python3
"""
Convert DoOR InChIKey indices to common chemical names.

This script converts door-toolkit's InChIKey-indexed response matrix to
a common-name-indexed matrix that can be used for odor lookups.

Usage:
    python convert_inchikey_to_names.py

The script will:
1. Load response_matrix_norm.csv (InChIKey indices)
2. Load odor_metadata.parquet (InChIKey → Name mappings)
3. Replace InChIKey indices with common names
4. Save to both data/cache/ and data/door_cache/
"""

import pandas as pd
from pathlib import Path
import sys


def convert_inchikey_to_names(
    response_matrix_path: Path,
    metadata_path: Path,
    output_paths: list[Path]
):
    """Convert InChIKey indices to common names.

    Parameters
    ----------
    response_matrix_path : Path
        Path to response_matrix_norm.csv with InChIKey indices
    metadata_path : Path
        Path to odor_metadata.parquet with Name → InChIKey mappings
    output_paths : list[Path]
        Paths to save the converted matrix
    """
    print("=" * 70)
    print("DoOR InChIKey → Name Conversion")
    print("=" * 70)
    print()

    # Step 1: Load response matrix
    print(f"[1/4] Loading response matrix from {response_matrix_path}")
    if not response_matrix_path.exists():
        print(f"❌ ERROR: {response_matrix_path} not found")
        print()
        print("Please ensure you have door-toolkit data extracted:")
        print("  pip install door-toolkit")
        print("  door extract --output data/door_cache/")
        sys.exit(1)

    response_matrix = pd.read_csv(response_matrix_path, index_col=0)
    print(f"   ✓ Loaded {len(response_matrix)} odorants × {len(response_matrix.columns)} ORN types")
    print(f"   Sample indices: {list(response_matrix.index[:3])}")
    print()

    # Step 2: Load metadata
    print(f"[2/4] Loading metadata from {metadata_path}")
    if not metadata_path.exists():
        print(f"❌ ERROR: {metadata_path} not found")
        print()
        print("The metadata file should be in the same directory as response_matrix_norm.csv")
        sys.exit(1)

    metadata = pd.read_parquet(metadata_path)
    print(f"   ✓ Loaded {len(metadata)} metadata entries")
    print()

    # Step 3: Create InChIKey → Name mapping
    print("[3/4] Creating InChIKey → Name mapping")
    inchikey_to_name = {}
    for idx, row in metadata.iterrows():
        if pd.notna(row.get('Name')) and pd.notna(row.get('InChIKey')):
            name = str(row['Name']).lower().strip()
            inchikey = str(row['InChIKey']).lower().strip()
            inchikey_to_name[inchikey] = name

    print(f"   ✓ Created {len(inchikey_to_name)} InChIKey → Name mappings")
    print()

    # Step 4: Replace indices
    print("[4/4] Converting InChIKey indices to common names")
    new_index = []
    converted_count = 0
    unmapped_samples = []

    for inchikey in response_matrix.index:
        inchikey_lower = str(inchikey).lower().strip()
        if inchikey_lower in inchikey_to_name:
            new_index.append(inchikey_to_name[inchikey_lower])
            converted_count += 1
        else:
            new_index.append(inchikey)
            if len(unmapped_samples) < 5:
                unmapped_samples.append(inchikey)

    response_matrix.index = new_index
    print(f"   ✓ Converted {converted_count}/{len(response_matrix)} indices to common names")

    if unmapped_samples:
        print(f"   ⚠️  {len(response_matrix) - converted_count} InChIKeys had no common name mapping")
        print(f"   Sample unmapped: {unmapped_samples[:3]}")
    print()

    # Verify critical odors
    print("Verifying critical odor names:")
    critical_odors = [
        '1-hexanol',
        'benzaldehyde',
        'acetic acid',
        'ethyl butyrate',
        '3-octanol',
        'citral',
        'linalool'
    ]

    found_count = 0
    for odor in critical_odors:
        if odor in response_matrix.index:
            print(f"   ✓ {odor:20s} found")
            found_count += 1
        else:
            print(f"   ✗ {odor:20s} NOT FOUND")

    print()
    print(f"Critical odors found: {found_count}/{len(critical_odors)}")
    print()

    # Save converted matrix
    print("Saving converted matrix:")
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        response_matrix.to_csv(output_path)
        print(f"   ✓ Saved to {output_path}")

    print()
    print("=" * 70)
    print("✅ Conversion complete!")
    print("=" * 70)
    print()
    print("The converted matrix now has common names as indices:")
    print(f"   Before: {list(response_matrix.index[:3])}")
    print()
    print("You can now run training and odor lookups should work:")
    print("   python src/scripts/train_ccbpn.py --task odor_discrimination ...")
    print()


if __name__ == "__main__":
    # Define paths
    response_matrix_path = Path("data/door_cache/response_matrix_norm.csv")
    metadata_path = Path("data/door_cache/odor_metadata.parquet")
    output_paths = [
        Path("data/cache/door_response_matrix.csv"),
        Path("data/door_cache/door_response_matrix.csv"),
    ]

    # Run conversion
    convert_inchikey_to_names(response_matrix_path, metadata_path, output_paths)
