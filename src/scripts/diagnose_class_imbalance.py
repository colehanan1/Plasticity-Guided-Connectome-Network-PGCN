#!/usr/bin/env python3
"""Diagnostic script to check class imbalance in behavioral dataset.

This script analyzes the distribution of approach/avoid labels to determine
if the model is simply learning to predict the majority class.

Usage
-----
    python src/scripts/diagnose_class_imbalance.py

Output
------
Prints class distribution statistics and per-dataset breakdowns to verify
whether validation accuracy matches the majority class baseline.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pgcn.data.behavioral_data import load_behavioral_dataframe


def main():
    """Analyze class distribution in behavioral dataset."""

    # Load behavioral data
    print("Loading behavioral data...")
    df = load_behavioral_dataframe(validate=True)

    print(f"\nDataset summary:")
    print(f"  Total trials: {len(df)}")
    print(f"  Total flies: {df['fly'].nunique()}")
    print(f"  Total datasets: {df['dataset'].nunique()}")

    # Overall class distribution
    print("\n" + "="*60)
    print("OVERALL CLASS DISTRIBUTION")
    print("="*60)

    class_counts = df['prediction'].value_counts()
    class_props = df['prediction'].value_counts(normalize=True)

    print(f"\nClass counts:")
    print(f"  Approach (1): {class_counts.get(1, 0):4d} trials ({class_props.get(1, 0):.1%})")
    print(f"  Avoid (0):    {class_counts.get(0, 0):4d} trials ({class_props.get(0, 0):.1%})")

    majority_baseline = class_props.max()
    print(f"\n⚠️  Majority class baseline: {majority_baseline:.1%}")
    print(f"    (Model can achieve this accuracy by always predicting majority class)")

    # Per-dataset breakdown
    print("\n" + "="*60)
    print("PER-DATASET BREAKDOWN")
    print("="*60)

    dataset_stats = []
    for dataset in sorted(df['dataset'].unique()):
        subset = df[df['dataset'] == dataset]
        n_trials = len(subset)
        approach_rate = subset['prediction'].mean()
        avoid_rate = 1 - approach_rate
        n_flies = subset['fly'].nunique()

        dataset_stats.append({
            'dataset': dataset,
            'n_trials': n_trials,
            'n_flies': n_flies,
            'approach_rate': approach_rate,
            'avoid_rate': avoid_rate,
        })

        print(f"\n{dataset}:")
        print(f"  Trials: {n_trials:3d} ({n_trials/len(df):.1%} of total)")
        print(f"  Flies:  {n_flies:3d}")
        print(f"  Approach: {approach_rate:.1%}")
        print(f"  Avoid:    {avoid_rate:.1%}")

    # Check if imbalance is consistent across datasets
    print("\n" + "="*60)
    print("IMBALANCE ANALYSIS")
    print("="*60)

    approach_rates = [s['approach_rate'] for s in dataset_stats]
    mean_approach = np.mean(approach_rates)
    std_approach = np.std(approach_rates)

    print(f"\nApproach rate across datasets:")
    print(f"  Mean: {mean_approach:.1%} ± {std_approach:.1%}")
    print(f"  Range: {min(approach_rates):.1%} - {max(approach_rates):.1%}")

    if mean_approach > 0.6 or mean_approach < 0.4:
        print(f"\n⚠️  WARNING: Strong class imbalance detected!")
        print(f"     Model may be learning to predict majority class only.")
        print(f"     Validation accuracy ~{majority_baseline:.1%} suggests this is happening.")
        print(f"\n💡 SOLUTION: Use class-balanced loss function (see Fix #1)")
    else:
        print(f"\n✓ Classes are relatively balanced (40-60% range)")

    # Per-trial-type breakdown
    print("\n" + "="*60)
    print("PER-TRIAL-TYPE BREAKDOWN")
    print("="*60)

    # Check if trial_label exists
    if 'trial_label' in df.columns:
        for trial_type in ['training', 'testing']:
            mask = df['trial_label'].astype(str).str.contains(trial_type, na=False)
            subset = df[mask]
            if len(subset) > 0:
                approach_rate = subset['prediction'].mean()
                print(f"\n{trial_type.capitalize()} trials:")
                print(f"  Count: {len(subset)}")
                print(f"  Approach rate: {approach_rate:.1%}")

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    if majority_baseline > 0.7:
        print(f"\n🔴 CRITICAL: Majority baseline ({majority_baseline:.1%}) is HIGH")
        print(f"   Your validation accuracy (~73.7%) is suspiciously close to this!")
        print(f"   Model is likely NOT learning odor-specific patterns.")
        print(f"\n   Next steps:")
        print(f"   1. Run diagnose_model_predictions.py to verify model outputs")
        print(f"   2. Implement class-balanced loss (Fix #1)")
        print(f"   3. Retrain and check if accuracy improves beyond baseline")
    elif majority_baseline > 0.6:
        print(f"\n🟡 WARNING: Moderate class imbalance ({majority_baseline:.1%})")
        print(f"   Consider using class-balanced loss for better performance")
    else:
        print(f"\n🟢 OK: Classes are balanced ({majority_baseline:.1%} baseline)")
        print(f"   Class imbalance is NOT the primary issue")


if __name__ == "__main__":
    main()
