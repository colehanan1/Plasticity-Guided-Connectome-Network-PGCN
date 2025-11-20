#!/usr/bin/env python3
"""Check the actual structure of the DoOR data."""

import pandas as pd
from pathlib import Path


def check_door_structure():
    """Check DoOR data structure."""
    print("=" * 70)
    print("DoOR Data Structure Check")
    print("=" * 70)
    print()

    cache_path = Path("data/cache/door_response_matrix.csv")

    if not cache_path.exists():
        print(f"❌ File not found: {cache_path}")
        print("   Run training once to generate the cached file.")
        return

    print(f"Loading: {cache_path}")
    door_data = pd.read_csv(cache_path, index_col=0)

    print()
    print("STRUCTURE:")
    print(f"  Shape: {door_data.shape}")
    print(f"  Rows (odorants): {len(door_data)}")
    print(f"  Columns (ORN types): {len(door_data.columns)}")
    print()

    print("INDEX (first 10 odorants):")
    for odor in door_data.index[:10]:
        print(f"  {odor}")
    print()

    print("COLUMNS (first 20 ORN types):")
    for orn in door_data.columns[:20]:
        print(f"  {orn}")
    print()

    # Check citral specifically
    if 'citral' in door_data.index:
        print("CITRAL ROW:")
        citral = door_data.loc['citral']
        print(f"  Type: {type(citral)}")
        print(f"  Shape: {citral.shape if hasattr(citral, 'shape') else 'N/A'}")
        print(f"  Length: {len(citral)}")
        print()

        # Show non-zero values
        if isinstance(citral, pd.Series):
            non_zero = citral[citral > 0.05]
            print(f"  Non-zero ORN responses (>0.05): {len(non_zero)}")
            if len(non_zero) > 0:
                print("  Top responses:")
                for orn, resp in non_zero.sort_values(ascending=False).head(10).items():
                    print(f"    {orn:20s}: {resp:.4f}")
            else:
                print("  ⚠️  No ORN responses above 0.05!")
        else:
            print(f"  ❌ ERROR: Expected Series, got {type(citral)}")
            print(f"  Data preview:")
            print(citral)
    else:
        print("❌ citral not found in DoOR index!")

    print()
    print("=" * 70)


if __name__ == '__main__':
    check_door_structure()
