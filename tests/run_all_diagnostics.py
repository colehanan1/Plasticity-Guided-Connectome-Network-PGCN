#!/usr/bin/env python
"""Master diagnostic runner for debugging 100% accuracy.

Runs all 4 diagnostic tests in sequence and provides comprehensive report.

Usage:
    python tests/run_all_diagnostics.py

or:
    python tests/run_all_diagnostics.py \\
        --data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \\
        --results results/ccbpn_recurrent_FIXED
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_test(test_script, *args):
    """Run a diagnostic test and return result."""
    cmd = [sys.executable, str(test_script)] + list(args)
    print(f"\n🔬 Running: {test_script.name}")
    print("-" * 70)

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='/home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv',
                       help='Path to behavioral data CSV')
    parser.add_argument('--results', default='results/ccbpn_recurrent_FIXED',
                       help='Path to training results directory')

    args = parser.parse_args()

    tests_dir = Path(__file__).parent
    results = {}

    print("="*70)
    print(" DEEP DIAGNOSTIC SUITE - DEBUGGING 100% ACCURACY")
    print("="*70)
    print(f"\nBehavioral data: {args.data}")
    print(f"Training results: {args.results}")
    print(f"\nRunning 4 diagnostic tests...")

    # Test 1: Class Distribution
    print("\n" + "="*70)
    print("TEST 1: CLASS DISTRIBUTION")
    print("="*70)
    test1_passed = run_test(tests_dir / 'test_1_class_distribution.py', args.data)
    results['test1_class_distribution'] = test1_passed

    # Test 2: CV Splits
    print("\n" + "="*70)
    print("TEST 2: CROSS-VALIDATION SPLITS")
    print("="*70)
    test2_passed = run_test(tests_dir / 'test_2_cv_splits.py', args.results)
    results['test2_cv_splits'] = test2_passed

    # Test 3: Validation Size
    print("\n" + "="*70)
    print("TEST 3: VALIDATION SET SIZE")
    print("="*70)
    test3_passed = run_test(tests_dir / 'test_3_val_set_size.py', args.results)
    results['test3_val_size'] = test3_passed

    # Test 4: Scrambled Labels (creates dataset, doesn't train)
    print("\n" + "="*70)
    print("TEST 4: SCRAMBLED LABELS SETUP")
    print("="*70)
    test4_setup = run_test(tests_dir / 'test_4_scrambled_labels.py', args.data)
    results['test4_scrambled_setup'] = test4_setup

    # Final Report
    print("\n" + "="*70)
    print(" DIAGNOSTIC RESULTS SUMMARY")
    print("="*70)

    all_passed = all([results['test1_class_distribution'],
                     results['test2_cv_splits'],
                     results['test3_val_size']])

    print("\nTest Results:")
    print(f"  Test 1 (Class Distribution): {'✅ PASS' if results['test1_class_distribution'] else '❌ FAIL'}")
    print(f"  Test 2 (CV Splits):           {'✅ PASS' if results['test2_cv_splits'] else '❌ FAIL'}")
    print(f"  Test 3 (Val Size):            {'✅ PASS' if results['test3_val_size'] else '❌ FAIL'}")
    print(f"  Test 4 (Scrambled Labels):    {'✅ SETUP' if results['test4_scrambled_setup'] else '❌ ERROR'} (needs training)")

    print("\n" + "="*70)
    print(" RECOMMENDATIONS")
    print("="*70)

    if not results['test1_class_distribution']:
        print("\n❌ Test 1 FAILED - Class distribution issue")
        print("   ROOT CAUSE: Labels lack variance or extreme class imbalance")
        print("   ACTION: Verify behavioral data source is correct")
        print("   - Check if 'prediction' column has the right values")
        print("   - Ensure data includes both approach AND avoid trials")

    if not results['test2_cv_splits']:
        print("\n❌ Test 2 FAILED - Cross-validation leakage")
        print("   ROOT CAUSE: Same flies in train and validation")
        print("   ACTION: Fix GroupKFold implementation")
        print("   - Ensure flies (not trials) are used as groups")
        print("   - Add tracking of train_flies and val_flies to results")

    if not results['test3_val_size']:
        print("\n❌ Test 3 FAILED - Validation sets too small")
        print("   ROOT CAUSE: Sample size insufficient")
        print("   ACTION: Use fewer folds or collect more data")
        print("   - Try 3-fold CV instead of 5-fold")
        print("   - Or accept that results may be noisy with small samples")

    if all_passed:
        print("\n✅ Tests 1-3 ALL PASSED!")
        print("   Data quality is good, CV is proper, sample size adequate")
        print("\n   NEXT CRITICAL STEP:")
        print("   🔬 Test 4: Train on SCRAMBLED labels")
        print("\n   A scrambled dataset has been created.")
        print("   Now train on it using the command shown above.")
        print("\n   If you get >70% accuracy on scrambled labels:")
        print("   → Model is memorizing metadata (fly ID, dataset ID, etc.)")
        print("   → Need to strip metadata from model inputs")
        print("\n   If you get ~50% accuracy on scrambled labels:")
        print("   → Model is learning from odor patterns (good!)")
        print("   → 100% on real data suggests a subtle bug")
        print("   → Try baseline CCBPN without LSTM")

    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)

    if not all_passed:
        print("\n1. Fix failing tests first")
        print("2. Don't re-train until tests pass")
        print("3. Systematic debugging is key")
    else:
        print("\n1. Train on scrambled labels (command shown in Test 4)")
        print("2. Report back the validation accuracy")
        print("3. If >70%: Strip metadata from inputs")
        print("4. If ~50%: Try baseline CCBPN (no LSTM)")

    print("\n" + "="*70)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
