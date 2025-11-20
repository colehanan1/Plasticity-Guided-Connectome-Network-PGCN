#!/usr/bin/env python3
"""Check citral with its specific InChIKey."""

import pandas as pd
from pathlib import Path


def check_citral_inchikey():
    """Check citral using its InChIKey."""
    print("=" * 70)
    print("Citral InChIKey Check")
    print("=" * 70)
    print()

    citral_inchikey = "WTEVQBCEXWBHNA-JXMROGBWSA-N"
    print(f"Citral InChIKey: {citral_inchikey}")
    print()

    # Check raw data
    print("[1] Checking RAW door-toolkit data...")
    raw_path = Path("data/door_cache/response_matrix_norm.csv")

    if raw_path.exists():
        print(f"    Loading: {raw_path}")
        raw_data = pd.read_csv(raw_path, index_col=0)
        print(f"    Shape: {raw_data.shape}")
        print(f"    Columns: {len(raw_data.columns)}")
        print()

        # Look for citral InChIKey (case-insensitive)
        citral_found = False
        for idx in raw_data.index:
            if str(idx).upper() == citral_inchikey.upper():
                print(f"    ✓ Found citral at index: {idx}")
                citral_row = raw_data.loc[idx]
                print(f"      Type: {type(citral_row)}")
                print(f"      Shape: {citral_row.shape}")
                print(f"      Columns: {len(citral_row) if isinstance(citral_row, pd.Series) else len(citral_row.columns)}")

                if isinstance(citral_row, pd.Series):
                    non_zero = citral_row[citral_row > 0.05]
                    print(f"      Non-zero responses: {len(non_zero)}")
                    if len(non_zero) > 0:
                        print(f"      Top 10 responses:")
                        for orn, resp in non_zero.sort_values(ascending=False).head(10).items():
                            print(f"        {orn:20s}: {resp:.4f}")
                    else:
                        print(f"      ⚠️  No responses above 0.05")
                        # Show all columns
                        print(f"      All columns and their values:")
                        for col in raw_data.columns:
                            print(f"        {col:20s}: {citral_row[col]:.4f}")
                else:
                    print(f"      ❌ ERROR: Expected Series, got DataFrame!")
                    print(f"      DataFrame preview:")
                    print(citral_row)

                citral_found = True
                break

        if not citral_found:
            print(f"    ✗ Citral InChIKey NOT found in raw data")
            print(f"    Sample indices (first 5):")
            for idx in raw_data.index[:5]:
                print(f"      {idx}")
    else:
        print(f"    ❌ File not found: {raw_path}")

    print()

    # Check converted/cached data
    print("[2] Checking CONVERTED/CACHED data...")
    cache_path = Path("data/cache/door_response_matrix.csv")

    if cache_path.exists():
        print(f"    Loading: {cache_path}")
        cache_data = pd.read_csv(cache_path, index_col=0)
        print(f"    Shape: {cache_data.shape}")
        print(f"    Rows: {len(cache_data)}")
        print(f"    Columns: {len(cache_data.columns)}")
        print()

        print(f"    First 10 rows (odorants):")
        for odor in cache_data.index[:10]:
            print(f"      {odor}")
        print()

        print(f"    All columns (ORN types):")
        for i, col in enumerate(cache_data.columns, 1):
            print(f"      {i:3d}. {col}")
        print()

        # Look for citral by name
        if 'citral' in cache_data.index:
            print(f"    ✓ Found 'citral' in converted data")
            citral_row = cache_data.loc['citral']
            print(f"      Type: {type(citral_row)}")
            print(f"      Shape: {citral_row.shape}")
            print(f"      Columns: {len(citral_row) if isinstance(citral_row, pd.Series) else len(citral_row.columns)}")

            if isinstance(citral_row, pd.Series):
                non_zero = citral_row[citral_row > 0.05]
                print(f"      Non-zero responses: {len(non_zero)}")
                if len(non_zero) > 0:
                    print(f"      Top 10 responses:")
                    for orn, resp in non_zero.sort_values(ascending=False).head(10).items():
                        print(f"        {orn:20s}: {resp:.4f}")
                else:
                    print(f"      ⚠️  No responses above 0.05")
            else:
                print(f"      ❌ ERROR: Expected Series, got DataFrame!")
                print(f"      This explains why citral returns 0 PNs!")
                print(f"      DataFrame shape: {citral_row.shape}")
                print(f"      DataFrame columns: {list(citral_row.columns)}")
                print(f"      DataFrame preview:")
                print(citral_row)
        else:
            print(f"    ✗ 'citral' NOT found in converted data")
            print(f"    Available odors (first 20):")
            for odor in cache_data.index[:20]:
                print(f"      {odor}")
    else:
        print(f"    ❌ File not found: {cache_path}")

    print()
    print("=" * 70)
    print()

    # DIAGNOSIS
    print("DIAGNOSIS:")
    print("-" * 70)
    if cache_path.exists():
        cache_data = pd.read_csv(cache_path, index_col=0)
        if cache_data.shape[1] < 10:
            print("❌ CRITICAL: DoOR data has only", cache_data.shape[1], "columns!")
            print("   Expected: 78+ ORN types (Or22a, Or65a, etc.)")
            print("   Actual:", list(cache_data.columns))
            print()
            print("   LIKELY CAUSE: Data was transposed or conversion failed")
            print()
            print("   FIX: Check the InChIKey conversion in door_integration.py")
            print("        The _convert_inchikey_to_names() method may be corrupting the data")
        elif 'citral' not in cache_data.index:
            print("❌ Citral not converted from InChIKey to name")
            print("   Check odor_metadata.parquet has citral entry")
        else:
            citral_row = cache_data.loc['citral']
            if isinstance(citral_row, pd.DataFrame):
                print("❌ Citral row is DataFrame (should be Series)")
                print("   This means multiple rows have the name 'citral'")
                print("   Or the data structure is fundamentally wrong")


if __name__ == '__main__':
    check_citral_inchikey()
