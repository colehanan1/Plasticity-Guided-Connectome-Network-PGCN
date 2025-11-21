#!/usr/bin/env python3
"""Diagnostic script to analyze model prediction diversity.

This script loads a trained CCBPN model and checks if it produces diverse
odor-specific predictions or just memorizes a single output (e.g., "always approach").

Usage
-----
    python src/scripts/diagnose_model_predictions.py \
        --model results/ccbpn_final/ccbpn_odor_discrimination_best.pt

Output
------
Prints prediction distribution statistics and per-odor prediction rates.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pgcn.models.ccbpn import ConnectomeConstrainedBehavioralPredictor
from pgcn.data.behavioral_data import load_behavioral_dataframe
from pgcn.data.door_integration import DoORIntegration


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze CCBPN model prediction diversity"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/ccbpn_odor_discrimination_best.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cache",
        help="Path to FlyWire cache directory"
    )
    parser.add_argument(
        "--dataset_mapping",
        type=str,
        default="configs/dataset_to_odor_mapping.yaml",
        help="Path to dataset-to-odor mapping YAML"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=50,
        help="Sequence length for odor patterns"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    )
    return parser.parse_args()


def prepare_test_data(cache_dir, dataset_mapping_path, sequence_length, device):
    """Prepare test data for model evaluation."""
    from train_ccbpn import prepare_behavioral_data

    return prepare_behavioral_data(
        behavioral_csv=None,
        dataset_mapping_path=dataset_mapping_path,
        cache_dir=cache_dir,
        sequence_length=sequence_length,
        device=device,
    )


def main():
    """Run model prediction diagnostics."""
    args = parse_args()

    print("="*60)
    print("CCBPN MODEL PREDICTION DIAGNOSTICS")
    print("="*60)

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print(f"   Have you trained a model yet?")
        print(f"   Run: python src/scripts/train_ccbpn.py")
        return

    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=args.device)

    model = ConnectomeConstrainedBehavioralPredictor(
        cache_dir=args.cache_dir,
        behavioral_task='odor_discrimination',
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    print(f"✓ Model loaded (best val acc: {checkpoint.get('best_val_acc', 0):.1%})")

    # Prepare test data
    print(f"\nPreparing test data...")
    odor_sequences, dopamine_signals, behavioral_labels, groups = prepare_test_data(
        cache_dir=args.cache_dir,
        dataset_mapping_path=args.dataset_mapping,
        sequence_length=args.sequence_length,
        device=args.device,
    )

    # Resize odor sequences if needed
    if odor_sequences.shape[2] != model.n_pn:
        new_odor_sequences = torch.zeros(
            odor_sequences.shape[0],
            odor_sequences.shape[1],
            model.n_pn,
            device=args.device
        )
        min_pn = min(odor_sequences.shape[2], model.n_pn)
        new_odor_sequences[:, :, :min_pn] = odor_sequences[:, :, :min_pn]
        odor_sequences = new_odor_sequences

    n_trials = len(behavioral_labels)
    print(f"✓ Prepared {n_trials} trials")

    # Get model predictions
    print(f"\nGenerating predictions...")
    with torch.no_grad():
        outputs = model(odor_sequences, dopamine_signals, return_intermediates=False)
        predicted_probs = outputs['behavioral_output'][:, -1].cpu().numpy()  # Final time step
        predicted_labels = (predicted_probs > 0.5).astype(int)

    # Compute accuracy
    accuracy = np.mean(predicted_labels == behavioral_labels)
    print(f"✓ Overall accuracy: {accuracy:.1%}")

    # Analyze prediction distribution
    print("\n" + "="*60)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*60)

    n_approach = np.sum(predicted_labels == 1)
    n_avoid = np.sum(predicted_labels == 0)

    print(f"\nPredicted class distribution:")
    print(f"  Approach (1): {n_approach:4d} ({n_approach/n_trials:.1%})")
    print(f"  Avoid (0):    {n_avoid:4d} ({n_avoid/n_trials:.1%})")

    # Check if model is just predicting one class
    if n_approach > 0.95 * n_trials:
        print(f"\n🔴 CRITICAL: Model predicts 'approach' for {n_approach/n_trials:.1%} of trials!")
        print(f"   Model has NOT learned odor-specific patterns.")
        print(f"   It's just memorizing the majority class.")
    elif n_avoid > 0.95 * n_trials:
        print(f"\n🔴 CRITICAL: Model predicts 'avoid' for {n_avoid/n_trials:.1%} of trials!")
        print(f"   Model has NOT learned odor-specific patterns.")
    elif min(n_approach, n_avoid) < 0.1 * n_trials:
        print(f"\n🟡 WARNING: Model is biased toward one class")
        print(f"   Predictions are not well-balanced")
    else:
        print(f"\n🟢 OK: Model produces diverse predictions")

    # Analyze prediction probabilities
    print("\n" + "="*60)
    print("PREDICTION PROBABILITY ANALYSIS")
    print("="*60)

    print(f"\nPrediction probabilities (P(approach)):")
    print(f"  Mean: {np.mean(predicted_probs):.3f}")
    print(f"  Std:  {np.std(predicted_probs):.3f}")
    print(f"  Min:  {np.min(predicted_probs):.3f}")
    print(f"  Max:  {np.max(predicted_probs):.3f}")

    # Check if probabilities are all near 0 or 1 (overconfident)
    extreme_probs = np.sum((predicted_probs < 0.1) | (predicted_probs > 0.9)) / n_trials

    if np.std(predicted_probs) < 0.05:
        print(f"\n🔴 CRITICAL: Probabilities have very low variance ({np.std(predicted_probs):.3f})")
        print(f"   Model outputs nearly constant predictions!")
    elif extreme_probs > 0.9:
        print(f"\n🟡 WARNING: {extreme_probs:.1%} of predictions are extreme (< 0.1 or > 0.9)")
        print(f"   Model is overconfident (may be memorizing)")
    else:
        print(f"\n🟢 OK: Probabilities show reasonable diversity")

    # Analyze per-odor predictions
    print("\n" + "="*60)
    print("PER-ODOR PREDICTION ANALYSIS")
    print("="*60)

    # Load odor mapping to get trial→odor assignments
    df = load_behavioral_dataframe(validate=False)

    with open(args.dataset_mapping, 'r') as f:
        dataset_mapping = yaml.safe_load(f)

    # Map trials to odors
    trial_odors = []
    for idx, row in df.iterrows():
        dataset = row['dataset']
        trial_label = row['trial_label']

        # Parse trial type
        if 'training' in str(trial_label):
            trial_type = 'training_trials'
            trial_num = int(str(trial_label).split('_')[-1]) - 1
        elif 'testing' in str(trial_label):
            trial_type = 'testing_trials'
            trial_num = int(str(trial_label).split('_')[-1]) - 1
        else:
            trial_odors.append("unknown")
            continue

        # Get odor name
        if dataset in dataset_mapping:
            trials_list = dataset_mapping[dataset].get(trial_type, [])
            if trial_num < len(trials_list):
                odor_name = trials_list[trial_num]
            else:
                odor_name = "unknown"
        else:
            odor_name = "unknown"

        trial_odors.append(odor_name)

    trial_odors = np.array(trial_odors)

    # Analyze predictions per odor
    unique_odors = sorted(set(trial_odors) - {'unknown'})

    print(f"\nPredictions by odor:")
    odor_stats = []

    for odor in unique_odors:
        mask = trial_odors == odor
        n_odor = np.sum(mask)

        if n_odor == 0:
            continue

        odor_preds = predicted_labels[mask]
        odor_probs = predicted_probs[mask]
        odor_labels = behavioral_labels[mask]

        approach_rate = np.mean(odor_preds)
        true_approach_rate = np.mean(odor_labels)
        accuracy_odor = np.mean(odor_preds == odor_labels)

        odor_stats.append({
            'odor': odor,
            'n_trials': n_odor,
            'pred_approach_rate': approach_rate,
            'true_approach_rate': true_approach_rate,
            'accuracy': accuracy_odor,
            'prob_std': np.std(odor_probs),
        })

        print(f"\n  {odor}:")
        print(f"    Trials: {n_odor}")
        print(f"    Predicted approach rate: {approach_rate:.1%}")
        print(f"    True approach rate:      {true_approach_rate:.1%}")
        print(f"    Accuracy: {accuracy_odor:.1%}")
        print(f"    Prob std: {np.std(odor_probs):.3f}")

    # Check if model differentiates odors
    print("\n" + "="*60)
    print("ODOR DIFFERENTIATION ANALYSIS")
    print("="*60)

    pred_approach_rates = [s['pred_approach_rate'] for s in odor_stats]
    prob_stds = [s['prob_std'] for s in odor_stats]

    if len(pred_approach_rates) > 1:
        range_approach = max(pred_approach_rates) - min(pred_approach_rates)
        mean_prob_std = np.mean(prob_stds)

        print(f"\nApproach rate range across odors: {range_approach:.1%}")
        print(f"Mean probability std per odor: {mean_prob_std:.3f}")

        if range_approach < 0.1:
            print(f"\n🔴 CRITICAL: Model predictions are nearly IDENTICAL across odors!")
            print(f"   Range is only {range_approach:.1%}")
            print(f"   Model is NOT learning odor-specific patterns.")
        elif range_approach < 0.3:
            print(f"\n🟡 WARNING: Low variation across odors ({range_approach:.1%})")
            print(f"   Model may not be fully utilizing odor information")
        else:
            print(f"\n🟢 OK: Model differentiates odors (range: {range_approach:.1%})")

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    issues_found = []

    if n_approach > 0.95 * n_trials or n_avoid > 0.95 * n_trials:
        issues_found.append("Model predicts single class for >95% of trials")

    if np.std(predicted_probs) < 0.05:
        issues_found.append("Output probabilities have very low variance")

    if len(pred_approach_rates) > 1 and (max(pred_approach_rates) - min(pred_approach_rates)) < 0.1:
        issues_found.append("Predictions identical across different odors")

    if issues_found:
        print("\n🔴 ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")

        print("\n💡 RECOMMENDED FIXES:")
        print("   1. Implement class-balanced loss (Fix #1)")
        print("   2. Increase KC sparsity to 10% (Fix #2)")
        print("   3. Add learning rate scheduler (Fix #3)")
        print("   4. Add dropout regularization (Fix #4)")
        print("\n   Run: python src/scripts/train_ccbpn.py --kc_sparsity 0.10 --use_class_weights")
    else:
        print("\n🟢 Model appears to be learning odor-specific patterns")
        print("   Prediction diversity looks reasonable")


if __name__ == "__main__":
    main()
