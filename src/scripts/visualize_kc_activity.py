#!/usr/bin/env python3
"""Diagnostic script to visualize KC activity patterns for different odors.

This script loads a trained CCBPN model and visualizes the internal KC
representations to check if different odors produce distinct activity patterns.

Usage
-----
    python src/scripts/visualize_kc_activity.py \
        --model results/ccbpn_final/ccbpn_odor_discrimination_best.pt \
        --output results/diagnostics

Output
------
Generates visualization plots:
- kc_activity_pca.png: PCA projection of KC activity by odor
- kc_activity_heatmap.png: Heatmap of KC responses to each odor
- kc_sparsity_hist.png: Distribution of KC sparsity across trials
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pgcn.models.ccbpn import ConnectomeConstrainedBehavioralPredictor
from pgcn.data.behavioral_data import load_behavioral_dataframe


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize CCBPN KC activity patterns"
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
        "--output_dir",
        type=str,
        default="results/diagnostics",
        help="Output directory for plots"
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


def get_trial_odors(dataset_mapping_path):
    """Map each trial to its odor name."""
    df = load_behavioral_dataframe(validate=False)

    with open(dataset_mapping_path, 'r') as f:
        dataset_mapping = yaml.safe_load(f)

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

    return np.array(trial_odors)


def main():
    """Run KC activity visualization diagnostics."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("CCBPN KC ACTIVITY VISUALIZATION")
    print("="*60)

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print(f"   Have you trained a model yet?")
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

    print(f"✓ Model loaded (n_pn={model.n_pn}, n_kc={model.n_kc}, n_mbon={model.n_mbon})")

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

    # Get trial odor labels
    print(f"\nMapping trials to odors...")
    trial_odors = get_trial_odors(args.dataset_mapping)
    unique_odors = sorted(set(trial_odors) - {'unknown'})
    print(f"✓ Found {len(unique_odors)} unique odors")

    # Extract KC activity patterns
    print(f"\nExtracting KC activity patterns...")
    kc_activities = []

    with torch.no_grad():
        for trial_idx in range(n_trials):
            odor_seq = odor_sequences[trial_idx:trial_idx+1]
            dopamine = dopamine_signals[trial_idx:trial_idx+1]

            # Forward pass with intermediates
            outputs = model(odor_seq, dopamine, return_intermediates=True)

            # Get KC activity at odor peak (t=20, middle of odor presentation)
            kc_act = outputs['kc_activity'][0, 20, :].cpu().numpy()  # (n_kc,)
            kc_activities.append(kc_act)

    kc_activities = np.array(kc_activities)  # (n_trials, n_kc)
    print(f"✓ Extracted KC activity: {kc_activities.shape}")

    # Compute KC sparsity statistics
    print("\n" + "="*60)
    print("KC SPARSITY ANALYSIS")
    print("="*60)

    kc_sparsity = np.mean(kc_activities > 0.1, axis=1)  # Fraction active per trial
    print(f"\nKC sparsity (fraction active):")
    print(f"  Mean: {np.mean(kc_sparsity):.1%} ± {np.std(kc_sparsity):.1%}")
    print(f"  Range: {np.min(kc_sparsity):.1%} - {np.max(kc_sparsity):.1%}")
    print(f"  Target: 5.0% (biological)")

    if np.mean(kc_sparsity) < 0.03:
        print(f"\n🔴 WARNING: KC sparsity is very low ({np.mean(kc_sparsity):.1%})")
        print(f"   Model may not have enough active KCs to represent odors")
    elif np.mean(kc_sparsity) > 0.15:
        print(f"\n🟡 WARNING: KC sparsity is high ({np.mean(kc_sparsity):.1%})")
        print(f"   Sparsity constraint may not be working properly")
    else:
        print(f"\n🟢 OK: KC sparsity is biologically plausible")

    # Plot 1: KC sparsity histogram
    print(f"\nGenerating sparsity histogram...")
    plt.figure(figsize=(8, 6))
    plt.hist(kc_sparsity * 100, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(5.0, color='red', linestyle='--', linewidth=2, label='Target (5%)')
    plt.axvline(np.mean(kc_sparsity) * 100, color='blue', linestyle='--',
                linewidth=2, label=f'Mean ({np.mean(kc_sparsity):.1%})')
    plt.xlabel('KC Sparsity (%)')
    plt.ylabel('Number of Trials')
    plt.title('Distribution of KC Sparsity Across Trials')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'kc_sparsity_hist.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'kc_sparsity_hist.png'}")

    # Plot 2: PCA visualization
    print(f"\nPerforming PCA...")
    pca = PCA(n_components=2)
    kc_2d = pca.fit_transform(kc_activities)

    explained_var = pca.explained_variance_ratio_
    print(f"✓ PCA explained variance: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}")

    plt.figure(figsize=(12, 10))

    # Create color palette
    colors = sns.color_palette('husl', len(unique_odors))
    odor_colors = {odor: colors[i] for i, odor in enumerate(unique_odors)}

    # Plot each odor
    for odor in unique_odors:
        mask = trial_odors == odor
        plt.scatter(
            kc_2d[mask, 0],
            kc_2d[mask, 1],
            c=[odor_colors[odor]],
            label=odor,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
    plt.title('KC Activity Representations (PCA)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'kc_activity_pca.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'kc_activity_pca.png'}")

    # Plot 3: Average KC response heatmap per odor
    print(f"\nGenerating KC response heatmap...")

    # Compute average KC response per odor
    odor_avg_kc = np.zeros((len(unique_odors), kc_activities.shape[1]))
    for i, odor in enumerate(unique_odors):
        mask = trial_odors == odor
        odor_avg_kc[i] = np.mean(kc_activities[mask], axis=0)

    # Find most variable KCs (top 100)
    kc_variance = np.var(odor_avg_kc, axis=0)
    top_kc_indices = np.argsort(kc_variance)[-100:]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        odor_avg_kc[:, top_kc_indices].T,
        cmap='viridis',
        cbar_kws={'label': 'KC Activity'},
        xticklabels=unique_odors,
        yticklabels=False,
        vmin=0,
        vmax=1,
    )
    plt.xlabel('Odor', fontsize=12)
    plt.ylabel('KC Index (top 100 most variable)', fontsize=12)
    plt.title('Average KC Response per Odor', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'kc_activity_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'kc_activity_heatmap.png'}")

    # Analyze odor separability
    print("\n" + "="*60)
    print("ODOR SEPARABILITY ANALYSIS")
    print("="*60)

    # Compute pairwise distances between odor centroids in KC space
    from scipy.spatial.distance import pdist, squareform

    odor_centroids = np.zeros((len(unique_odors), kc_activities.shape[1]))
    for i, odor in enumerate(unique_odors):
        mask = trial_odors == odor
        odor_centroids[i] = np.mean(kc_activities[mask], axis=0)

    pairwise_dists = squareform(pdist(odor_centroids, 'euclidean'))

    print(f"\nPairwise odor distances (KC space):")
    print(f"  Mean: {np.mean(pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)]):.3f}")
    print(f"  Std:  {np.std(pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)]):.3f}")

    # Find most similar and most different odor pairs
    triu_indices = np.triu_indices_from(pairwise_dists, k=1)
    flat_dists = pairwise_dists[triu_indices]
    flat_pairs = list(zip(triu_indices[0], triu_indices[1]))

    most_similar_idx = np.argmin(flat_dists)
    most_different_idx = np.argmax(flat_dists)

    most_similar_pair = (unique_odors[flat_pairs[most_similar_idx][0]],
                         unique_odors[flat_pairs[most_similar_idx][1]])
    most_different_pair = (unique_odors[flat_pairs[most_different_idx][0]],
                          unique_odors[flat_pairs[most_different_idx][1]])

    print(f"\nMost similar odors:")
    print(f"  {most_similar_pair[0]} <-> {most_similar_pair[1]}")
    print(f"  Distance: {flat_dists[most_similar_idx]:.3f}")

    print(f"\nMost different odors:")
    print(f"  {most_different_pair[0]} <-> {most_different_pair[1]}")
    print(f"  Distance: {flat_dists[most_different_idx]:.3f}")

    # Check if odors cluster together
    mean_within_odor_dist = []
    for odor in unique_odors:
        mask = trial_odors == odor
        odor_kcs = kc_activities[mask]
        if len(odor_kcs) > 1:
            within_dists = pdist(odor_kcs, 'euclidean')
            mean_within_odor_dist.append(np.mean(within_dists))

    mean_within = np.mean(mean_within_odor_dist)
    mean_between = np.mean(flat_dists)

    print(f"\nWithin-odor vs. between-odor distances:")
    print(f"  Mean within-odor distance:  {mean_within:.3f}")
    print(f"  Mean between-odor distance: {mean_between:.3f}")
    print(f"  Ratio (between/within):     {mean_between/mean_within:.2f}x")

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    issues_found = []

    if mean_between / mean_within < 1.5:
        issues_found.append("Odors are NOT well-separated in KC space (ratio < 1.5x)")

    if explained_var[0] + explained_var[1] < 0.3:
        issues_found.append(f"Low PCA variance explained ({explained_var[0]+explained_var[1]:.1%})")

    if np.mean(kc_sparsity) < 0.03 or np.mean(kc_sparsity) > 0.15:
        issues_found.append(f"KC sparsity outside biological range ({np.mean(kc_sparsity):.1%})")

    if issues_found:
        print("\n🔴 ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")

        print("\n💡 INTERPRETATION:")
        if mean_between / mean_within < 1.5:
            print("   - KC representations are NOT odor-specific")
            print("   - Model is not learning distinct patterns for different odors")
            print("   - This explains why behavioral predictions are similar across odors")

        print("\n💡 RECOMMENDED FIXES:")
        print("   1. Increase KC sparsity to 10% (more active KCs per trial)")
        print("   2. Use class-balanced loss (force model to differentiate odors)")
        print("   3. Add learning rate scheduler (help model escape local minimum)")
    else:
        print("\n🟢 KC representations look good!")
        print("   - Odors are well-separated in KC space")
        print("   - Model is learning odor-specific patterns")

    print(f"\n{'='*60}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
