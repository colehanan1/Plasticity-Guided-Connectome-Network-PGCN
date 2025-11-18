#!/usr/bin/env python3
"""Helper script to verify DoOR coverage for experimental odors.

This script helps you:
1. Check which datasets exist in your behavioral CSV
2. Verify that experimental odors are in DoOR database
3. Generate a template dataset-to-odor mapping file

Usage
-----
# Basic check:
python src/scripts/verify_door_coverage.py \
    --behavioral_csv /path/to/model_predictions.csv \
    --cache_dir data/cache

# Generate mapping template:
python src/scripts/verify_door_coverage.py \
    --behavioral_csv /path/to/model_predictions.csv \
    --cache_dir data/cache \
    --generate_template configs/dataset_to_odor_mapping.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pgcn.data.behavioral_data import load_behavioral_dataframe
from pgcn.data.door_integration import DoORIntegration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify DoOR coverage for CCBPN training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--behavioral_csv",
        type=str,
        required=True,
        help="Path to model_predictions.csv"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cache",
        help="Path to FlyWire cache directory"
    )

    parser.add_argument(
        "--generate_template",
        type=str,
        default=None,
        help="Generate mapping template YAML at this path"
    )

    parser.add_argument(
        "--test_odors",
        type=str,
        nargs='+',
        default=['hexanol', 'ethyl_butyrate', 'benzaldehyde', '3-octanol',
                'citral', 'linalool', 'apple_cider_vinegar'],
        help="List of experimental odors to test"
    )

    return parser.parse_args()


def check_behavioral_csv_structure(csv_path: str):
    """Analyze behavioral CSV structure."""
    print("\n" + "="*60)
    print("BEHAVIORAL CSV STRUCTURE")
    print("="*60)

    df = load_behavioral_dataframe(csv_path)

    print(f"✓ Loaded {len(df)} trials")
    print(f"✓ Columns: {list(df.columns)}")

    # Check for odor-related columns
    has_odor_column = 'odor_identity' in df.columns or 'odor' in df.columns
    if has_odor_column:
        print("✓ Found explicit odor column")
        if 'odor_identity' in df.columns:
            unique_odors = df['odor_identity'].unique()
            print(f"  Unique odors: {unique_odors}")
    else:
        print("⚠️  No explicit odor column - will infer from dataset name")

    # Analyze datasets
    print(f"\n✓ Unique datasets: {df['dataset'].nunique()}")
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        n_flies = dataset_df['fly'].nunique()
        n_trials = len(dataset_df)

        # Count trial types
        if 'trial_type' in df.columns:
            training = len(dataset_df[dataset_df['trial_type'] == 'training'])
            testing = len(dataset_df[dataset_df['trial_type'] == 'testing'])
            print(f"  {dataset}: {n_flies} flies, {n_trials} trials "
                  f"(training={training}, testing={testing})")
        else:
            print(f"  {dataset}: {n_flies} flies, {n_trials} trials")

    # Analyze trial labels
    print(f"\n✓ Unique trial labels: {df['trial_label'].nunique()}")
    trial_label_examples = df['trial_label'].unique()[:10]
    print(f"  Examples: {trial_label_examples}")

    return df


def check_door_coverage(cache_dir: str, test_odors: list):
    """Check DoOR coverage for experimental odors."""
    print("\n" + "="*60)
    print("DoOR DATABASE COVERAGE")
    print("="*60)

    try:
        door = DoORIntegration(cache_dir=Path(cache_dir))
        print(f"✓ DoOR database loaded: {len(door.door_data)} odorants")

        print(f"\n✓ Testing {len(test_odors)} experimental odors:")

        covered_odors = []
        missing_odors = []

        for odor in test_odors:
            # Try to get PN activity pattern
            pn_activity = door.odor_to_pn_activity(odor, n_pn=100)
            n_active = np.sum(pn_activity > 0.1)

            if n_active > 0:
                print(f"  ✓ {odor:25s} → {n_active:3d} active PNs")
                covered_odors.append(odor)
            else:
                print(f"  ✗ {odor:25s} → NOT IN DoOR")
                missing_odors.append(odor)

        print(f"\n✓ Coverage: {len(covered_odors)}/{len(test_odors)} odors found in DoOR")

        if missing_odors:
            print(f"\n⚠️  Missing odors: {missing_odors}")
            print("   These odors will have zero PN activity patterns!")
            print("   Consider using chemical similarity approximation or adding to DoOR manually.")

        # Check odor distinctiveness
        if len(covered_odors) >= 2:
            print(f"\n✓ Checking odor distinctiveness...")
            for i, odor1 in enumerate(covered_odors[:5]):
                for odor2 in covered_odors[i+1:6]:
                    similarity = door.get_odor_similarity(odor1, odor2, n_pn=100)
                    status = "✓" if similarity < 0.9 else "⚠️"
                    print(f"  {status} {odor1:15s} vs {odor2:15s}: correlation = {similarity:.2f}")

        return door, covered_odors, missing_odors

    except Exception as e:
        print(f"✗ Error loading DoOR: {e}")
        print("\nTroubleshooting:")
        print("1. Check that FlyWire cache exists at:", cache_dir)
        print("2. Download DoOR database:")
        print("   wget https://github.com/ropensci/DoOR.data/raw/master/data/door_response_matrix.csv")
        print(f"   mv door_response_matrix.csv {cache_dir}/")
        return None, [], []


def generate_mapping_template(
    behavioral_csv: str,
    output_path: str,
    covered_odors: list,
    missing_odors: list
):
    """Generate dataset-to-odor mapping template."""
    print("\n" + "="*60)
    print("GENERATING MAPPING TEMPLATE")
    print("="*60)

    df = load_behavioral_dataframe(behavioral_csv)
    datasets = df['dataset'].unique()

    template_lines = []
    template_lines.append("# Dataset-to-Odor Mapping for CCBPN Training")
    template_lines.append("# Generated by verify_door_coverage.py")
    template_lines.append("#")
    template_lines.append("# INSTRUCTIONS:")
    template_lines.append("# 1. For each dataset below, fill in the actual odor sequences")
    template_lines.append("# 2. Training trials: Usually same odor repeated (CS+ conditioning)")
    template_lines.append("# 3. Testing trials: Mix of CS+ and CS- odors in order")
    template_lines.append("#")
    template_lines.append(f"# Available odors in DoOR: {covered_odors}")
    if missing_odors:
        template_lines.append(f"# ⚠️  Missing from DoOR (will be zero): {missing_odors}")
    template_lines.append("")

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        template_lines.append(f"# Dataset: {dataset}")

        # Get trial counts
        if 'trial_type' in df.columns:
            training_trials = dataset_df[dataset_df['trial_type'] == 'training']
            testing_trials = dataset_df[dataset_df['trial_type'] == 'testing']

            n_training = len(training_trials['trial_label'].unique())
            n_testing = len(testing_trials['trial_label'].unique())
        else:
            # Try to infer from trial_label patterns
            trial_labels = dataset_df['trial_label'].unique()
            training_labels = [t for t in trial_labels if 'training' in str(t)]
            testing_labels = [t for t in trial_labels if 'testing' in str(t)]

            n_training = len(training_labels)
            n_testing = len(testing_labels)

        template_lines.append(f"{dataset}:")

        # Infer likely odor from dataset name
        inferred_odor = None
        if 'benz' in dataset.lower():
            inferred_odor = 'benzaldehyde'
        elif 'hex' in dataset.lower():
            inferred_odor = 'hexanol'
        elif 'eb' in dataset.lower():
            inferred_odor = 'ethyl_butyrate'
        elif 'oct' in dataset.lower():
            inferred_odor = '3-octanol'
        elif 'citral' in dataset.lower():
            inferred_odor = 'citral'
        elif 'linalool' in dataset.lower():
            inferred_odor = 'linalool'

        # Training trials
        template_lines.append(f"  training_trials:")
        if inferred_odor:
            template_lines.append(f"    # INFERRED CS+: {inferred_odor} (verify this!)")
            template_lines.append(f"    # FILL IN: List {n_training} odors, one per training trial")
            for i in range(min(n_training, 5)):
                template_lines.append(f"    - {inferred_odor}  # training_{i+1}")
            if n_training > 5:
                template_lines.append(f"    # ... add {n_training - 5} more training trials")
        else:
            template_lines.append(f"    # FILL IN: List {n_training} odors, one per training trial")
            for i in range(min(n_training, 3)):
                template_lines.append(f"    - FILL_IN_ODOR  # training_{i+1}")

        # Testing trials
        template_lines.append(f"  testing_trials:")
        if inferred_odor:
            template_lines.append(f"    # FILL IN: Mix of CS+ ({inferred_odor}) and CS- odors")
            template_lines.append(f"    # List {n_testing} odors in order presented")
            template_lines.append(f"    - {inferred_odor}  # testing_1 (CS+ example)")
            template_lines.append(f"    - FILL_IN_CS_MINUS  # testing_2 (CS- example)")
            if n_testing > 2:
                template_lines.append(f"    # ... add {n_testing - 2} more testing trials")
        else:
            template_lines.append(f"    # FILL IN: List {n_testing} odors, one per testing trial")
            for i in range(min(n_testing, 2)):
                template_lines.append(f"    - FILL_IN_ODOR  # testing_{i+1}")

        template_lines.append("")

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(template_lines))

    print(f"✓ Template generated: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Edit {output_path} to fill in actual odor sequences")
    print(f"2. Verify YAML syntax: python -c \"import yaml; yaml.safe_load(open('{output_path}'))\"")
    print(f"3. Train CCBPN with real odor data!")


def main():
    args = parse_args()

    print("CCBPN DoOR Coverage Verification Tool")
    print("="*60)

    # Step 1: Check behavioral CSV structure
    df = check_behavioral_csv_structure(args.behavioral_csv)

    # Step 2: Check DoOR coverage
    door, covered, missing = check_door_coverage(args.cache_dir, args.test_odors)

    # Step 3: Generate template if requested
    if args.generate_template and door is not None:
        generate_mapping_template(
            args.behavioral_csv,
            args.generate_template,
            covered,
            missing
        )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Behavioral CSV: {len(df)} trials across {df['dataset'].nunique()} datasets")
    print(f"✓ DoOR coverage: {len(covered)}/{len(args.test_odors)} odors")

    if missing:
        print(f"\n⚠️  WARNING: {len(missing)} odors not in DoOR!")
        print(f"   Missing: {missing}")
        print(f"   These trials will have ZERO odor patterns!")

    if len(covered) < len(args.test_odors) * 0.7:
        print("\n⚠️  CRITICAL: Less than 70% odor coverage!")
        print("   Training will likely fail. Action required:")
        print("   1. Download full DoOR database")
        print("   2. Use chemical similarity approximation for missing odors")
        print("   3. Or exclude trials with missing odors")
        sys.exit(1)

    print("\n✓ Ready to proceed with CCBPN training!")
    print("\nNext steps:")
    print(f"1. Fill in dataset-to-odor mapping: {args.generate_template or 'configs/dataset_to_odor_mapping.yaml'}")
    print(f"2. Run training: python src/scripts/train_ccbpn.py --behavioral_csv {args.behavioral_csv}")


if __name__ == "__main__":
    main()
