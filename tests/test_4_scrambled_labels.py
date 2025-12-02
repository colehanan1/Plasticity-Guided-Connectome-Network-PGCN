#!/usr/bin/env python
"""Test 4: Scrambled labels sanity check.

This test creates a version of the data with randomly shuffled labels.
If model gets >70% accuracy on scrambled data, it's memorizing metadata
(fly ID, dataset ID, trial order) instead of learning from odor patterns.

CRITICAL: This is the most important test!
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def create_scrambled_dataset(csv_path, output_path=None):
    """Create dataset with scrambled labels for sanity check."""

    print("="*70)
    print("TEST 4: SCRAMBLED LABELS SANITY CHECK")
    print("="*70)

    # Load data
    df = pd.read_csv(csv_path)

    original_dist = df['prediction'].value_counts()
    print(f"\nORIGINAL LABEL DISTRIBUTION:")
    print(f"  Approach (1): {original_dist.get(1, 0)} ({original_dist.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Avoid (0): {original_dist.get(0, 0)} ({original_dist.get(0, 0)/len(df)*100:.1f}%)")

    # SCRAMBLE labels randomly (50/50 split)
    np.random.seed(42)  # Reproducible
    df['prediction'] = np.random.randint(0, 2, size=len(df))

    scrambled_dist = df['prediction'].value_counts()
    print(f"\nSCRAMBLED LABEL DISTRIBUTION:")
    print(f"  Approach (1): {scrambled_dist.get(1, 0)} ({scrambled_dist.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Avoid (0): {scrambled_dist.get(0, 0)} ({scrambled_dist.get(0, 0)/len(df)*100:.1f}%)")

    # Save scrambled version
    if output_path is None:
        input_path = Path(csv_path)
        output_path = input_path.parent / (input_path.stem + '_SCRAMBLED.csv')
    else:
        output_path = Path(output_path)

    df.to_csv(output_path, index=False)

    print(f"\n✅ Scrambled dataset created: {output_path}")
    print(f"   Total trials: {len(df)}")
    print(f"   Labels are now RANDOM (no pattern)")

    print("\n" + "="*70)
    print("NEXT STEP: Train model on scrambled data")
    print("="*70)
    print("\nRun this command:")
    print(f"\n  python src/scripts/train_ccbpn_recurrent.py \\")
    print(f"    --behavioral-data {output_path} \\")
    print(f"    --cache-dir data/cache \\")
    print(f"    --output-dir results/ccbpn_SCRAMBLED \\")
    print(f"    --epochs 50 \\")
    print(f"    --context-dim 64 \\")
    print(f"    --lr 0.001 \\")
    print(f"    --n-folds 5")

    print("\n" + "="*70)
    print("EXPECTED RESULTS:")
    print("="*70)
    print("\n✅ GOOD (Model is learning from odor patterns):")
    print("   - Validation accuracy: 45-55% (random guessing)")
    print("   - Model cannot predict random labels")
    print("   - This proves model is NOT memorizing metadata")

    print("\n❌ BAD (Model is memorizing metadata):")
    print("   - Validation accuracy: >70%")
    print("   - Model predicts 'random' labels better than chance")
    print("   - ROOT CAUSE: Model has access to informative metadata")
    print("   - SOLUTION: Strip fly_id, dataset_id from inputs")

    print("\n⚠️  CONCERNING (Somewhere in between):")
    print("   - Validation accuracy: 55-70%")
    print("   - Suggests partial metadata leakage")
    print("   - Needs investigation")

    print("\n" + "="*70)
    print(f"\nScrambled dataset saved to: {output_path}")
    print("Now train on this and report back the accuracy!")
    print("="*70)

    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        csv_path = '/home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv'
        print(f"Using default path: {csv_path}")
    else:
        csv_path = sys.argv[1]

    create_scrambled_dataset(csv_path)
    sys.exit(0)
