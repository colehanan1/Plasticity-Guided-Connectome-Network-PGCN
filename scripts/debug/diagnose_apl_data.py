#!/usr/bin/env python3
"""
Diagnose APL extraction missing data errors.

This script validates that PN and classification data have the required
columns for APL neuron extraction.
"""

import sys
from pathlib import Path
import pandas as pd

def diagnose_apl_data():
    """Run comprehensive APL data diagnostics."""
    cache_dir = Path('data/cache/ethyl_butyrate_circuit')
    flywire_dir = Path('data/flywire')

    print("=" * 80)
    print("APL EXTRACTION DATA DIAGNOSTICS")
    print("=" * 80)

    # 1. Check PN data
    print("\n1. PN DATA STRUCTURE:")
    pns = pd.read_csv(cache_dir / 'dm3_vm2_dm1_pns.csv')
    print(f"   Rows: {len(pns)}")
    print(f"   Columns: {pns.columns.tolist()}")

    print("\n2. REQUIRED COLUMNS CHECK:")
    required_cols = ['root_id', 'pn_root_id', 'glomerulus']
    for col in required_cols:
        if col in pns.columns:
            print(f"   ✅ '{col}' column EXISTS")
            if col == 'glomerulus':
                print(f"      Unique values: {pns[col].unique().tolist()}")
                print(f"      Value counts:")
                print(f"      {pns[col].value_counts().to_string().replace(chr(10), chr(10) + '      ')}")
        else:
            print(f"   ❌ '{col}' column MISSING")

    # Check for root_id issues
    if 'pn_root_id' in pns.columns and 'root_id' not in pns.columns:
        print(f"\n   ⚠️  WARNING: Column is 'pn_root_id', not 'root_id'")
        print(f"      APL extraction expects 'root_id'")
        print(f"      FIX: Rename column before passing to extract_apl_neurons()")

    print("\n3. PN SAMPLE DATA:")
    print(pns.head(3).to_string())

    # 2. Check classification data
    print("\n" + "=" * 80)
    print("4. CLASSIFICATION DATA STRUCTURE:")
    classification = pd.read_csv(flywire_dir / 'processed_labels.csv.gz',
                                  compression='gzip', nrows=10)
    print(f"   Columns: {classification.columns.tolist()}")

    print("\n5. CLASSIFICATION COLUMN CHECK:")
    expected_cols = ['root_id', 'cell_type', 'cell_class', 'nt_type', 'processed_labels']
    for col in expected_cols:
        if col in classification.columns:
            print(f"   ✅ '{col}' EXISTS")
        else:
            print(f"   ❌ '{col}' MISSING")

    if 'processed_labels' in classification.columns:
        print(f"\n   NOTE: Classification uses 'processed_labels' instead of 'cell_type'")
        print(f"   Sample values:")
        for i, val in enumerate(classification['processed_labels'].head(3)):
            print(f"      [{i}]: {val}")

    # 3. Check connections data
    print("\n" + "=" * 80)
    print("6. CONNECTIONS DATA STRUCTURE:")
    connections = pd.read_csv(flywire_dir / 'connections_princeton.csv.gz',
                              compression='gzip', nrows=10)
    print(f"   Columns: {connections.columns.tolist()}")

    print("\n7. NT_TYPE LOCATION:")
    if 'nt_type' in connections.columns:
        print(f"   ✅ 'nt_type' found in CONNECTIONS table (correct!)")
        print(f"   Unique values: {connections['nt_type'].unique().tolist()}")
    else:
        print(f"   ❌ 'nt_type' NOT in connections")

    if 'nt_type' in classification.columns:
        print(f"   ⚠️  'nt_type' also in classification (unexpected)")
    else:
        print(f"   ✅ 'nt_type' NOT in classification (expected)")

    # 4. Root cause summary
    print("\n" + "=" * 80)
    print("8. ROOT CAUSE ANALYSIS:")

    issues = []

    if 'root_id' not in pns.columns and 'pn_root_id' in pns.columns:
        issues.append("PN column name: 'pn_root_id' instead of 'root_id'")

    if 'glomerulus' not in pns.columns:
        issues.append("PNs missing 'glomerulus' column")

    if 'cell_type' not in classification.columns:
        issues.append("Classification missing 'cell_type' (uses 'processed_labels' instead)")

    if issues:
        print("   Issues identified:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("   ✅ No data structure issues found!")

    print("\n9. RECOMMENDED FIXES:")
    if 'root_id' not in pns.columns and 'pn_root_id' in pns.columns:
        print("   Fix 1: Rename column before calling extract_apl_neurons()")
        print("   ```python")
        print("   if 'pn_root_id' in pns.columns and 'root_id' not in pns.columns:")
        print("       pns = pns.rename(columns={'pn_root_id': 'root_id'})")
        print("   ```")

    if 'cell_type' not in classification.columns:
        print("\n   Fix 2: Update extract_apl_neurons() to use 'processed_labels'")
        print("   ```python")
        print("   if 'processed_labels' in classification.columns:")
        print("       matches = classification[")
        print("           classification['processed_labels'].astype(str).str.contains('APL', ...)")
        print("       ]")
        print("   ```")

    if 'nt_type' not in connections.columns:
        print("\n   Fix 3: Use connections table for nt_type, not classification")
        print("   (This is already implemented correctly)")

    print("=" * 80)
    return len(issues)

if __name__ == '__main__':
    num_issues = diagnose_apl_data()
    sys.exit(0 if num_issues == 0 else 1)
