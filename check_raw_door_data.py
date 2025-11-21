#!/usr/bin/env python3
"""Check the raw door-toolkit data BEFORE conversion."""

import pandas as pd
from pathlib import Path


def check_raw_door():
    """Check raw door-toolkit data."""
    print("=" * 70)
    print("Raw door-toolkit Data Check (BEFORE InChIKey Conversion)")
    print("=" * 70)
    print()

    # Check response_matrix_norm.csv
    raw_path = Path("data/door_cache/response_matrix_norm.csv")
    if not raw_path.exists():
        print(f"❌ File not found: {raw_path}")
        return

    print(f"Loading: {raw_path}")
    raw_data = pd.read_csv(raw_path, index_col=0)

    print()
    print("RAW DATA STRUCTURE:")
    print(f"  Shape: {raw_data.shape}")
    print(f"  Rows: {len(raw_data)}")
    print(f"  Columns: {len(raw_data.columns)}")
    print()

    print("RAW INDEX (first 10 - should be InChIKeys):")
    for idx in raw_data.index[:10]:
        print(f"  {idx}")
    print()

    print("RAW COLUMNS (first 20 - should be ORN types):")
    for col in raw_data.columns[:20]:
        print(f"  {col}")
    print()

    print("RAW COLUMNS (ALL):")
    print(f"  Total columns: {len(raw_data.columns)}")
    for i, col in enumerate(raw_data.columns, 1):
        print(f"  {i:3d}. {col}")
    print()

    # Check metadata
    meta_path = Path("data/door_cache/odor_metadata.parquet")
    if meta_path.exists():
        print(f"Loading: {meta_path}")
        metadata = pd.read_parquet(meta_path)

        print()
        print("METADATA STRUCTURE:")
        print(f"  Shape: {metadata.shape}")
        print(f"  Columns: {list(metadata.columns)}")
        print()

        # Find citral
        citral_meta = metadata[metadata['Name'].str.lower() == 'citral']
        if len(citral_meta) > 0:
            print("CITRAL METADATA:")
            print(citral_meta[['Name', 'InChIKey']].to_string())
            print()

            # Find citral in raw data by InChIKey
            citral_inchikey = citral_meta.iloc[0]['InChIKey'].lower().strip()
            print(f"Looking for InChIKey: {citral_inchikey}")

            # Try to find it in the index
            found = False
            for idx in raw_data.index:
                if str(idx).lower().strip() == citral_inchikey:
                    print(f"✓ Found in raw data at index: {idx}")
                    citral_row = raw_data.loc[idx]
                    print(f"  Type: {type(citral_row)}")
                    print(f"  Shape: {citral_row.shape if hasattr(citral_row, 'shape') else 'N/A'}")
                    print(f"  Length: {len(citral_row)}")

                    if isinstance(citral_row, pd.Series):
                        non_zero = citral_row[citral_row > 0.05]
                        print(f"  Non-zero responses (>0.05): {len(non_zero)}")
                        if len(non_zero) > 0:
                            print("  Top responses:")
                            for orn, resp in non_zero.sort_values(ascending=False).head(10).items():
                                print(f"    {orn:20s}: {resp:.4f}")
                    found = True
                    break

            if not found:
                print(f"✗ InChIKey not found in raw data index")
        else:
            print("✗ citral not found in metadata")
    else:
        print(f"❌ Metadata not found: {meta_path}")

    print()
    print("=" * 70)


if __name__ == '__main__':
    check_raw_door()
