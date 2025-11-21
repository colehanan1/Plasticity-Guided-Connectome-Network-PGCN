#!/usr/bin/env python3
"""
Visualize KC Sparsity Capacity Test Results

Generates comparison plots across 10%, 15%, and 20% KC sparsity experiments.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(metrics_file):
    """Load metrics JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)

def extract_fold_metrics(metrics):
    """Extract validation accuracies and epochs for all folds."""
    val_accs = []
    epochs_stopped = []

    for fold_result in metrics['fold_results']:
        val_accs.append(fold_result['best_val_acc'] * 100)  # Convert to percentage
        epochs_stopped.append(len(fold_result['metrics']))

    return val_accs, epochs_stopped

def plot_sparsity_comparison(output_dir):
    """Generate comparison plots for sparsity experiments."""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics from all three experiments
    # Note: Baseline (10%) metrics should be from previous run
    baseline_file = "results/ccbpn_balanced/ccbpn_odor_discrimination_metrics.json"  # Adjust if different
    exp15_file = "results/ccbpn_sparsity15/ccbpn_odor_discrimination_metrics.json"
    exp20_file = "results/ccbpn_sparsity20/ccbpn_odor_discrimination_metrics.json"

    # Load data
    print("Loading metrics...")
    try:
        baseline = load_metrics(baseline_file)
        baseline_accs, baseline_epochs = extract_fold_metrics(baseline)
    except FileNotFoundError:
        print(f"Warning: Baseline metrics not found at {baseline_file}")
        print("Using fallback baseline data from report")
        baseline_accs = [69.2, 73.3, 68.8, 67.9, 73.3]
        baseline_epochs = [23, 49, 22, 46, 23]

    exp15 = load_metrics(exp15_file)
    exp20 = load_metrics(exp20_file)

    exp15_accs, exp15_epochs = extract_fold_metrics(exp15)
    exp20_accs, exp20_epochs = extract_fold_metrics(exp20)

    # =====================================================================
    # Plot 1: Validation Accuracy by Fold
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    folds = np.arange(1, 6)
    width = 0.25

    ax.bar(folds - width, baseline_accs, width, label='10% Sparsity (200 KCs)',
           color='#2E86AB', alpha=0.8)
    ax.bar(folds, exp15_accs, width, label='15% Sparsity (300 KCs)',
           color='#A23B72', alpha=0.8)
    ax.bar(folds + width, exp20_accs, width, label='20% Sparsity (400 KCs)',
           color='#F18F01', alpha=0.8)

    # Add horizontal line at baseline average
    baseline_mean = np.mean(baseline_accs)
    ax.axhline(baseline_mean, color='#2E86AB', linestyle='--', alpha=0.5,
               label=f'Baseline Mean: {baseline_mean:.1f}%')
    ax.axhline(np.mean(exp15_accs), color='#A23B72', linestyle='--', alpha=0.5,
               label=f'15% Mean: {np.mean(exp15_accs):.1f}%')
    ax.axhline(np.mean(exp20_accs), color='#F18F01', linestyle='--', alpha=0.5,
               label=f'20% Mean: {np.mean(exp20_accs):.1f}%')

    # Add majority class baseline
    ax.axhline(67.7, color='red', linestyle=':', alpha=0.7,
               label='Majority Class Baseline: 67.7%')

    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('KC Sparsity Capacity Test: Validation Accuracy by Fold',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.set_ylim(60, 85)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sparsity_comparison_by_fold.png', dpi=300)
    print(f"Saved: {output_dir / 'sparsity_comparison_by_fold.png'}")
    plt.close()

    # =====================================================================
    # Plot 2: Average Performance with Error Bars
    # =====================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    sparsities = ['10%\n(200 KCs)', '15%\n(300 KCs)', '20%\n(400 KCs)']
    means = [np.mean(baseline_accs), np.mean(exp15_accs), np.mean(exp20_accs)]
    stds = [np.std(baseline_accs), np.std(exp15_accs), np.std(exp20_accs)]
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    x_pos = np.arange(len(sparsities))
    bars = ax.bar(x_pos, means, color=colors, alpha=0.8, width=0.6)
    ax.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='black',
                capsize=10, capthick=2, linewidth=2)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.5, f'{mean:.1f}%\n±{std:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add baseline reference line
    ax.axhline(67.7, color='red', linestyle=':', alpha=0.7, linewidth=2,
               label='Majority Class Baseline (67.7%)')

    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_xlabel('KC Sparsity Configuration', fontsize=12)
    ax.set_title('KC Capacity Test: Average Performance Across All Folds',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sparsities, fontsize=11)
    ax.set_ylim(60, 80)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sparsity_comparison_average.png', dpi=300)
    print(f"Saved: {output_dir / 'sparsity_comparison_average.png'}")
    plt.close()

    # =====================================================================
    # Plot 3: Convergence Speed (Epochs to Early Stopping)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    folds = np.arange(1, 6)
    width = 0.25

    ax.bar(folds - width, baseline_epochs, width, label='10% Sparsity',
           color='#2E86AB', alpha=0.8)
    ax.bar(folds, exp15_epochs, width, label='15% Sparsity',
           color='#A23B72', alpha=0.8)
    ax.bar(folds + width, exp20_epochs, width, label='20% Sparsity',
           color='#F18F01', alpha=0.8)

    # Add mean lines
    ax.axhline(np.mean(baseline_epochs), color='#2E86AB', linestyle='--', alpha=0.5,
               label=f'10% Mean: {np.mean(baseline_epochs):.0f} epochs')
    ax.axhline(np.mean(exp15_epochs), color='#A23B72', linestyle='--', alpha=0.5,
               label=f'15% Mean: {np.mean(exp15_epochs):.0f} epochs')
    ax.axhline(np.mean(exp20_epochs), color='#F18F01', linestyle='--', alpha=0.5,
               label=f'20% Mean: {np.mean(exp20_epochs):.0f} epochs')

    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('Epochs Until Early Stopping', fontsize=12)
    ax.set_title('KC Sparsity Capacity Test: Convergence Speed',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sparsity_comparison_convergence.png', dpi=300)
    print(f"Saved: {output_dir / 'sparsity_comparison_convergence.png'}")
    plt.close()

    # =====================================================================
    # Plot 4: Learning Curves (Sample Fold 3 - Best Performance)
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Get Fold 3 metrics for all experiments
    baseline_fold3 = baseline_accs[2] if len(baseline_accs) > 2 else None
    exp15_fold3 = exp15['fold_results'][2]['metrics']
    exp20_fold3 = exp20['fold_results'][2]['metrics']

    # Extract epochs and metrics
    exp15_epochs_list = [m['epoch'] for m in exp15_fold3]
    exp15_train_acc = [m['train_acc'] * 100 for m in exp15_fold3]
    exp15_val_acc = [m['val_acc'] * 100 for m in exp15_fold3]
    exp15_train_loss = [m['train_loss'] for m in exp15_fold3]
    exp15_val_loss = [m['val_loss'] for m in exp15_fold3]

    exp20_epochs_list = [m['epoch'] for m in exp20_fold3]
    exp20_train_acc = [m['train_acc'] * 100 for m in exp20_fold3]
    exp20_val_acc = [m['val_acc'] * 100 for m in exp20_fold3]
    exp20_train_loss = [m['train_loss'] for m in exp20_fold3]
    exp20_val_loss = [m['val_loss'] for m in exp20_fold3]

    # Plot accuracy
    ax1.plot(exp15_epochs_list, exp15_train_acc, 'o-', color='#A23B72',
             alpha=0.7, label='15% Train', linewidth=2)
    ax1.plot(exp15_epochs_list, exp15_val_acc, 's-', color='#A23B72',
             alpha=0.9, label='15% Val', linewidth=2)
    ax1.plot(exp20_epochs_list, exp20_train_acc, 'o-', color='#F18F01',
             alpha=0.7, label='20% Train', linewidth=2)
    ax1.plot(exp20_epochs_list, exp20_val_acc, 's-', color='#F18F01',
             alpha=0.9, label='20% Val', linewidth=2)

    ax1.axhline(67.7, color='red', linestyle=':', alpha=0.7,
                label='Majority Class (67.7%)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Fold 3 Learning Curves: Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot loss
    ax2.plot(exp15_epochs_list, exp15_train_loss, 'o-', color='#A23B72',
             alpha=0.7, label='15% Train', linewidth=2)
    ax2.plot(exp15_epochs_list, exp15_val_loss, 's-', color='#A23B72',
             alpha=0.9, label='15% Val', linewidth=2)
    ax2.plot(exp20_epochs_list, exp20_train_loss, 'o-', color='#F18F01',
             alpha=0.7, label='20% Train', linewidth=2)
    ax2.plot(exp20_epochs_list, exp20_val_loss, 's-', color='#F18F01',
             alpha=0.9, label='20% Val', linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Fold 3 Learning Curves: Loss', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sparsity_comparison_learning_curves.png', dpi=300)
    print(f"Saved: {output_dir / 'sparsity_comparison_learning_curves.png'}")
    plt.close()

    # =====================================================================
    # Plot 5: Summary Box Plot
    # =====================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [baseline_accs, exp15_accs, exp20_accs]
    positions = [1, 2, 3]
    colors_box = ['#2E86AB', '#A23B72', '#F18F01']

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=False,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, (d, pos) in enumerate(zip(data, positions)):
        x = np.random.normal(pos, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, color='black', s=50, zorder=3)

    # Add baseline reference
    ax.axhline(67.7, color='red', linestyle=':', alpha=0.7, linewidth=2,
               label='Majority Class Baseline (67.7%)')

    ax.set_xticks(positions)
    ax.set_xticklabels(['10% Sparsity\n(200 KCs)', '15% Sparsity\n(300 KCs)',
                        '20% Sparsity\n(400 KCs)'], fontsize=11)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('KC Capacity Test: Distribution of Validation Accuracy',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(60, 85)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sparsity_comparison_boxplot.png', dpi=300)
    print(f"Saved: {output_dir / 'sparsity_comparison_boxplot.png'}")
    plt.close()

    print(f"\n✅ All visualizations saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - sparsity_comparison_by_fold.png (validation accuracy per fold)")
    print("  - sparsity_comparison_average.png (mean ± std across folds)")
    print("  - sparsity_comparison_convergence.png (epochs until early stopping)")
    print("  - sparsity_comparison_learning_curves.png (Fold 3 training dynamics)")
    print("  - sparsity_comparison_boxplot.png (distribution summary)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize KC sparsity comparison")
    parser.add_argument("--output_dir", type=str, default="results/sparsity_comparison",
                        help="Directory to save plots")
    args = parser.parse_args()

    plot_sparsity_comparison(args.output_dir)
