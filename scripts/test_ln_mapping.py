"""
Quick validation script for LN-glomerulus mapping.

This script runs basic checks to ensure the LN mapping is working correctly.

Usage:
    python scripts/test_ln_mapping.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

def test_ln_mapping_outputs():
    """Test that all expected output files exist and have correct structure."""

    results_dir = Path("results/ln_mapping_test")

    print("=" * 80)
    print("LN-GLOMERULUS MAPPING OUTPUT VALIDATION")
    print("=" * 80)

    # Expected files
    expected_files = [
        'ln_glomerulus_associations.csv',
        'ln_primary_glomerulus.csv',
        'glomerulus_ln_summary.csv',
        'ln_categories.csv'
    ]

    all_good = True

    for filename in expected_files:
        filepath = results_dir / filename

        if not filepath.exists():
            print(f"\n❌ MISSING: {filename}")
            all_good = False
            continue

        # Load and check
        df = pd.read_csv(filepath)
        print(f"\n✅ {filename}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")

        # File-specific checks
        if filename == 'ln_glomerulus_associations.csv':
            required_cols = ['ln_id', 'glomerulus', 'total_synapses', 'ln_category']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"   ⚠️  Missing columns: {missing}")
                all_good = False
            else:
                print(f"   LNs mapped: {df['ln_id'].nunique():,}")
                print(f"   Glomeruli: {df['glomerulus'].nunique()}")
                print(f"   Total associations: {len(df):,}")

        elif filename == 'ln_primary_glomerulus.csv':
            if 'total_synapses' in df.columns:
                print(f"   LNs with primary glomerulus: {len(df):,}")

        elif filename == 'glomerulus_ln_summary.csv':
            if 'num_lns' in df.columns:
                print(f"   Total glomeruli: {len(df)}")
                print(f"   Top glomeruli: {df.nlargest(5, 'num_lns')['glomerulus'].tolist()}")

        elif filename == 'ln_categories.csv':
            if 'ln_category' in df.columns:
                print(f"   Category distribution:")
                for cat, count in df['ln_category'].value_counts().items():
                    pct = 100 * count / len(df)
                    print(f"     {cat}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 80)
    if all_good:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ SOME CHECKS FAILED")
    print("=" * 80)

    return all_good


def test_specific_glomerulus(glomerulus='DL5'):
    """Test LN associations for a specific glomerulus."""

    results_dir = Path("results/ln_mapping_test")
    assoc_file = results_dir / 'ln_glomerulus_associations.csv'

    if not assoc_file.exists():
        print(f"\n⚠️  Association file not found: {assoc_file}")
        return

    df = pd.read_csv(assoc_file)

    print("\n" + "=" * 80)
    print(f"LN ASSOCIATIONS FOR GLOMERULUS: {glomerulus}")
    print("=" * 80)

    glom_lns = df[df['glomerulus'] == glomerulus].copy()

    if len(glom_lns) == 0:
        print(f"\n⚠️  No LNs found for glomerulus {glomerulus}")
        return

    print(f"\nTotal LNs associated with {glomerulus}: {glom_lns['ln_id'].nunique()}")
    print(f"Total associations: {len(glom_lns)}")

    # Connection direction breakdown
    if 'connection_direction' in glom_lns.columns:
        print(f"\nConnection directions:")
        for direction, count in glom_lns['connection_direction'].value_counts().items():
            pct = 100 * count / len(glom_lns)
            print(f"  {direction}: {count} ({pct:.1f}%)")

    # Top LNs by synapse count
    if 'total_synapses' in glom_lns.columns:
        print(f"\nTop 10 LNs by synapse strength:")
        top_lns = glom_lns.nlargest(10, 'total_synapses')
        for idx, row in top_lns.iterrows():
            print(f"  LN {row['ln_id']}: {row['total_synapses']:.0f} synapses "
                  f"({row['connection_direction'] if 'connection_direction' in row else 'unknown'})")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Run tests
    success = test_ln_mapping_outputs()

    # Test specific glomerulus
    test_specific_glomerulus('DL5')

    # Exit with appropriate code
    sys.exit(0 if success else 1)
