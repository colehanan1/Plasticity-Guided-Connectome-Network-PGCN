#!/usr/bin/env python
"""Test 2: Verify cross-validation splits for data leakage.

This test checks if the same flies appear in both train and validation sets.
If yes, model can memorize fly-specific patterns → 100% accuracy.
"""

import json
import sys
from pathlib import Path

def test_cv_splits(results_path):
    """Check for train/val fly overlap in cross-validation."""

    print("="*70)
    print("TEST 2: CROSS-VALIDATION SPLIT ANALYSIS")
    print("="*70)

    # Load training results
    results_file = Path(results_path) / 'results.json'

    if not results_file.exists():
        print(f"\n❌ ERROR: Results file not found: {results_file}")
        print("   Make sure you've trained the model first!")
        return False

    with open(results_file, 'r') as f:
        results = json.load(f)

    all_flies = set()
    leakage_detected = False
    test_passed = True

    for fold_idx, fold_data in enumerate(results['fold_results']):
        print(f"\nFold {fold_idx + 1}:")

        # Check if fold tracking exists
        if 'train_flies' not in fold_data or 'val_flies' not in fold_data:
            print("  ⚠️  WARNING: Fly IDs not tracked in results!")
            print("  Cannot verify train/val separation.")
            print("  This is a problem - we need to track which flies are in each split.")
            test_passed = False
            continue

        train_flies = set(fold_data['train_flies'])
        val_flies = set(fold_data['val_flies'])
        overlap = train_flies & val_flies

        print(f"  Train flies: {len(train_flies)}")
        print(f"  Val flies: {len(val_flies)}")
        print(f"  Overlap: {len(overlap)}")

        if len(overlap) > 0:
            print(f"  ❌ DATA LEAKAGE! Overlapping flies: {list(overlap)[:5]}...")
            leakage_detected = True
            test_passed = False
        else:
            print(f"  ✅ No overlap (good)")

        all_flies.update(train_flies)
        all_flies.update(val_flies)

    print(f"\nTotal unique flies across all folds: {len(all_flies)}")

    print("="*70)

    # Final verdict
    if leakage_detected:
        print("\n❌ TEST 2 FAILED: Data leakage detected!")
        print("   ROOT CAUSE: Same flies appear in both train and validation")
        print("   Model can memorize fly-specific patterns → 100%")
        print("\n   RECOMMENDATION: Fix GroupKFold implementation")
        print("   - Ensure flies are used as groups, not trials")
        print("   - Verify no overlap between train_flies and val_flies")
        return False
    elif not test_passed:
        print("\n⚠️  TEST 2 INCONCLUSIVE: Fly tracking not implemented")
        print("   Cannot verify if CV splits are proper")
        print("\n   RECOMMENDATION: Add fly tracking to training script")
        print("   - Store train_flies and val_flies in fold_results")
        print("   - This enables verification of proper CV")
        return False
    else:
        print("\n✅ TEST 2 PASSED: No cross-validation leakage detected")
        print("   Train and val flies are properly separated")
        print("   100% accuracy cannot be explained by CV leakage")
        return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        results_path = 'results/ccbpn_recurrent_FIXED'
        print(f"Using default path: {results_path}")
    else:
        results_path = sys.argv[1]

    passed = test_cv_splits(results_path)
    sys.exit(0 if passed else 1)
