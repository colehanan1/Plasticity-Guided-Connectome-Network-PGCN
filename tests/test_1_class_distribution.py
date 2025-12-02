#!/usr/bin/env python
"""Test 1: Check class distribution in behavioral data.

This test checks if all trials have the same label (trivial task).
If variance < 0.1, model can achieve 100% by predicting majority class.
"""

import pandas as pd
import numpy as np
import sys

def test_class_distribution(csv_path):
    """Check if labels have sufficient variance."""

    print("="*70)
    print("TEST 1: CLASS DISTRIBUTION ANALYSIS")
    print("="*70)

    # Load behavioral data
    df = pd.read_csv(csv_path)

    # Overall distribution
    n_total = len(df)
    n_approach = df['prediction'].sum()
    n_avoid = n_total - n_approach
    pct_approach = (n_approach / n_total) * 100
    variance = df['prediction'].std()

    print(f"\nOVERALL:")
    print(f"  Total trials: {n_total}")
    print(f"  Approach: {int(n_approach)} ({pct_approach:.1f}%)")
    print(f"  Avoid: {int(n_avoid)} ({100-pct_approach:.1f}%)")
    print(f"  Variance: {variance:.3f}")

    # Check if trivially easy
    test_passed = True
    if variance < 0.1:
        print("\n❌ CRITICAL: Labels have almost NO variance!")
        print("   All trials are nearly the same class.")
        print("   Model can achieve 100% by just predicting majority class!")
        test_passed = False
    elif pct_approach > 95 or pct_approach < 5:
        print(f"\n❌ CRITICAL: Labels are {pct_approach:.1f}% one class!")
        print("   Task is trivially easy - model just predicts majority.")
        test_passed = False
    else:
        print(f"\n✅ PASS: Labels have good variance ({variance:.3f})")
        print(f"   Class distribution: {pct_approach:.1f}% / {100-pct_approach:.1f}%")

    # Per-dataset distribution
    print("\nPER-DATASET:")
    datasets_ok = True
    for dataset in sorted(df['dataset'].unique()):
        subset = df[df['dataset'] == dataset]
        n_trials = len(subset)
        n_app = subset['prediction'].sum()
        pct_app = (n_app / n_trials) * 100

        print(f"  {dataset:20s}: {n_trials:4d} trials, "
              f"{int(n_app):4d} approach ({pct_app:5.1f}%), "
              f"{n_trials-int(n_app):4d} avoid ({100-pct_app:5.1f}%)")

        if pct_app > 98 or pct_app < 2:
            print(f"    ⚠️  WARNING: Nearly all one class! ({pct_app:.1f}%)")
            datasets_ok = False

    if not datasets_ok:
        print("\n⚠️  Some datasets have extreme class imbalance")
        print("   Model may learn dataset-specific biases")

    # Per-fly variance
    print("\nPER-FLY VARIANCE:")
    fly_means = df.groupby('fly')['prediction'].mean()
    fly_var = fly_means.std()
    print(f"  Mean approach rate per fly: {fly_means.mean():.3f}")
    print(f"  Std dev across flies: {fly_var:.3f}")

    if fly_var < 0.05:
        print("  ⚠️  WARNING: All flies behave identically!")
        print("     No individual differences - suspicious!")
        test_passed = False
    else:
        print(f"  ✅ Flies show natural variation ({fly_var:.3f})")

    # Check unique values
    print("\nLABEL VALUES:")
    unique_vals = sorted(df['prediction'].unique())
    print(f"  Unique values: {unique_vals}")

    if len(unique_vals) == 1:
        print("  ❌ CRITICAL: Only ONE unique label value!")
        print("     All trials have same outcome - task is trivial!")
        test_passed = False
    elif not all(v in [0, 0.0, 1, 1.0] for v in unique_vals):
        print(f"  ⚠️  WARNING: Non-binary labels detected: {unique_vals}")
        print("     Expected only 0 and 1")

    print("="*70)

    # Final verdict
    if test_passed:
        print("\n✅ TEST 1 PASSED: Data has sufficient variance")
        print("   Task is not trivially easy")
        print("   100% accuracy cannot be explained by class imbalance alone")
        return True
    else:
        print("\n❌ TEST 1 FAILED: Task may be trivially easy")
        print("   ROOT CAUSE: Labels lack variance or extreme imbalance")
        print("   RECOMMENDATION: Verify behavioral data source")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        csv_path = '/home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv'
        print(f"Using default path: {csv_path}")
    else:
        csv_path = sys.argv[1]

    passed = test_class_distribution(csv_path)
    sys.exit(0 if passed else 1)
