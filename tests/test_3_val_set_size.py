#!/usr/bin/env python
"""Test 3: Check validation set size.

This test checks if validation sets are large enough for reliable accuracy estimates.
If val set < 100 trials, 100% accuracy could be chance.
"""

import json
import sys
from pathlib import Path

def test_val_set_size(results_path):
    """Check if validation sets are large enough."""

    print("="*70)
    print("TEST 3: VALIDATION SET SIZE ANALYSIS")
    print("="*70)

    # Load training results
    results_file = Path(results_path) / 'results.json'

    if not results_file.exists():
        print(f"\n❌ ERROR: Results file not found: {results_file}")
        return False

    with open(results_file, 'r') as f:
        results = json.load(f)

    test_passed = True
    total_val_trials = 0

    for fold_idx, fold_data in enumerate(results['fold_results']):
        print(f"\nFold {fold_idx + 1}:")

        # Estimate validation size
        if 'val_flies' in fold_data:
            n_val_flies = len(fold_data['val_flies'])
            # Assume ~10 trials per fly (typical for this data)
            estimated_val_trials = n_val_flies * 10

            print(f"  Val flies: {n_val_flies}")
            print(f"  Estimated val trials: ~{estimated_val_trials}")

            total_val_trials += estimated_val_trials

            if estimated_val_trials < 50:
                print(f"  ❌ CRITICAL: Validation set is TOO SMALL!")
                print(f"     Getting 100% on <50 samples is easy")
                test_passed = False
            elif estimated_val_trials < 100:
                print(f"  ⚠️  WARNING: Validation set is small")
                print(f"     100% accuracy on <100 samples is not unusual")
                test_passed = False
            else:
                print(f"  ✅ Validation set size is adequate")
        else:
            print("  ⚠️  Cannot estimate size (no fly tracking)")
            test_passed = False

        # Check actual validation accuracy
        val_acc = fold_data.get('best_val_acc', 0)
        print(f"  Val accuracy: {val_acc:.1%}")

    avg_val_trials = total_val_trials / len(results['fold_results'])
    print(f"\nAverage validation trials per fold: ~{avg_val_trials:.0f}")

    print("="*70)

    # Final verdict
    if not test_passed:
        print("\n❌ TEST 3 FAILED: Validation sets may be too small")
        print("   ROOT CAUSE: Sample size insufficient for reliable estimates")
        print("   100% accuracy on <100 samples is not unusual")
        print("\n   RECOMMENDATIONS:")
        print("   1. Use fewer folds (e.g., 3-fold instead of 5-fold)")
        print("   2. Collect more data")
        print("   3. Accept that results may be noisy")
        return False
    else:
        print("\n✅ TEST 3 PASSED: Validation sets are adequate size")
        print("   Sample sizes are sufficient for reliable accuracy")
        print("   100% accuracy cannot be explained by small sample size")
        return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        results_path = 'results/ccbpn_recurrent_FIXED'
        print(f"Using default path: {results_path}")
    else:
        results_path = sys.argv[1]

    passed = test_val_set_size(results_path)
    sys.exit(0 if passed else 1)
