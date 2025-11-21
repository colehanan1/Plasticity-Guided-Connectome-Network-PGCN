#!/usr/bin/env python3
"""Filter control datasets from behavioral CSV for clean CCBPN training.

This script removes control datasets (non-rewarded trials) from behavioral data,
keeping only conditioned/reward-paired trials for cleaner model training.

Control datasets typically show low approach rates (~10-15%) because flies were
not conditioned to associate odors with reward. Including these in training can
confuse the model.

Usage:
    python scripts/filter_control_data.py \
        --input ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \
        --output ~/Documents/cole/Data/Opto/Combined/model_predictions_conditioned_only.csv
"""

import pandas as pd
import argparse
from pathlib import Path


def filter_control_datasets(input_csv: str, output_csv: str):
    """Remove control datasets from behavioral data.

    Args:
        input_csv: Path to original model_predictions.csv
        output_csv: Path to save filtered CSV (conditioned trials only)
    """
    df = pd.read_csv(input_csv)

    print("=" * 70)
    print("Control Dataset Filter for CCBPN Training")
    print("=" * 70)
    print()
    print(f"Original dataset: {len(df)} trials from {len(df['dataset'].unique())} datasets")
    print()
    print("Datasets:")
    for dataset in sorted(df['dataset'].unique()):
        n_trials = len(df[df['dataset'] == dataset])
        approach_rate = df[df['dataset'] == dataset]['prediction'].mean()
        print(f"  {dataset:25s}: {n_trials:4d} trials, {approach_rate:6.1%} approach")

    # Filter out control datasets (keep only conditioned/reward-paired trials)
    # Control keywords: datasets with "control" or "AIR" in name
    control_keywords = ['control', 'AIR']
    df_filtered = df[~df['dataset'].str.contains('|'.join(control_keywords), case=False)]

    print()
    print("=" * 70)
    print(f"Filtered dataset: {len(df_filtered)} trials from {len(df_filtered['dataset'].unique())} datasets")
    print(f"Removed {len(df) - len(df_filtered)} control trials ({(len(df) - len(df_filtered)) / len(df):.1%})")
    print("=" * 70)
    print()

    print("Remaining datasets (conditioned only):")
    for dataset in sorted(df_filtered['dataset'].unique()):
        n_trials = len(df_filtered[df_filtered['dataset'] == dataset])
        approach_rate = df_filtered[df_filtered['dataset'] == dataset]['prediction'].mean()
        print(f"  {dataset:25s}: {n_trials:4d} trials, {approach_rate:6.1%} approach")

    # Save filtered data
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)

    print()
    print(f"✓ Saved filtered data to: {output_path}")
    print()
    print("Use this file for CCBPN training:")
    print(f"  python src/scripts/train_ccbpn.py \\")
    print(f"      --behavioral_data {output_path} \\")
    print(f"      --dataset_mapping configs/dataset_to_odor_mapping.yaml \\")
    print(f"      --task odor_discrimination \\")
    print(f"      --epochs 100")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filter control datasets from behavioral CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter control datasets
  python scripts/filter_control_data.py \\
      --input ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \\
      --output ~/Documents/cole/Data/Opto/Combined/model_predictions_conditioned_only.csv

  # This will remove datasets containing 'control' or 'AIR' in the name:
  #   - Benz_control
  #   - hex_control
  #   - EB_control
  #   - opto_AIR
  #
  # And keep only conditioned datasets:
  #   - opto_benz
  #   - opto_hex
  #   - opto_EB
        """
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV path (model_predictions.csv)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV path (e.g., model_predictions_conditioned_only.csv)'
    )
    args = parser.parse_args()

    filter_control_datasets(args.input, args.output)
