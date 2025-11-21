#!/usr/bin/env python3
"""Compare CCBPN training results before and after fixes.

This script loads training metrics from two different runs (baseline and improved)
and generates a comprehensive comparison report showing the impact of fixes:
- Class-balanced loss
- Increased KC sparsity (5% → 10%)
- Learning rate scheduler
- Dropout regularization

Usage
-----
    python src/scripts/compare_training_results.py \
        --baseline results/ccbpn_baseline/ccbpn_odor_discrimination_metrics.json \
        --improved results/ccbpn_fixed/ccbpn_odor_discrimination_metrics.json \
        --output results/diagnostics/comparison_report.txt

Output
------
Prints comparison statistics and generates plots showing:
- Loss curves (baseline vs. improved)
- Accuracy curves (baseline vs. improved)
- Convergence speed comparison
- Final performance metrics
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare CCBPN training results before/after fixes"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline metrics JSON file"
    )
    parser.add_argument(
        "--improved",
        type=str,
        required=True,
        help="Path to improved metrics JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/diagnostics",
        help="Output directory for comparison plots and report"
    )
    return parser.parse_args()


def load_metrics(json_path):
    """Load training metrics from JSON file."""
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def extract_fold_curves(metrics):
    """Extract loss and accuracy curves across all folds."""
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []

    for fold_result in metrics['fold_results']:
        fold_metrics = fold_result['metrics']

        train_losses = [m['train_loss'] for m in fold_metrics]
        val_losses = [m['val_loss'] for m in fold_metrics]
        train_accs = [m['train_acc'] for m in fold_metrics]
        val_accs = [m['val_acc'] for m in fold_metrics]

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs)

    return {
        'train_loss': np.array(all_train_losses),
        'val_loss': np.array(all_val_losses),
        'train_acc': np.array(all_train_accs),
        'val_acc': np.array(all_val_accs),
    }


def plot_comparison(baseline_curves, improved_curves, output_dir):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Baseline loss
    baseline_train_mean = baseline_curves['train_loss'].mean(axis=0)
    baseline_train_std = baseline_curves['train_loss'].std(axis=0)
    baseline_val_mean = baseline_curves['val_loss'].mean(axis=0)
    baseline_val_std = baseline_curves['val_loss'].std(axis=0)

    epochs = np.arange(1, len(baseline_train_mean) + 1)

    ax1.plot(epochs, baseline_train_mean, 'b-', label='Train', linewidth=2)
    ax1.fill_between(epochs,
                      baseline_train_mean - baseline_train_std,
                      baseline_train_mean + baseline_train_std,
                      alpha=0.2, color='b')
    ax1.plot(epochs, baseline_val_mean, 'r-', label='Val', linewidth=2)
    ax1.fill_between(epochs,
                      baseline_val_mean - baseline_val_std,
                      baseline_val_mean + baseline_val_std,
                      alpha=0.2, color='r')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('BASELINE: Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Improved loss
    improved_train_mean = improved_curves['train_loss'].mean(axis=0)
    improved_train_std = improved_curves['train_loss'].std(axis=0)
    improved_val_mean = improved_curves['val_loss'].mean(axis=0)
    improved_val_std = improved_curves['val_loss'].std(axis=0)

    epochs_improved = np.arange(1, len(improved_train_mean) + 1)

    ax2.plot(epochs_improved, improved_train_mean, 'b-', label='Train', linewidth=2)
    ax2.fill_between(epochs_improved,
                      improved_train_mean - improved_train_std,
                      improved_train_mean + improved_train_std,
                      alpha=0.2, color='b')
    ax2.plot(epochs_improved, improved_val_mean, 'r-', label='Val', linewidth=2)
    ax2.fill_between(epochs_improved,
                      improved_val_mean - improved_val_std,
                      improved_val_mean + improved_val_std,
                      alpha=0.2, color='r')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('IMPROVED: Loss Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'loss_comparison.png'}")

    # Plot 2: Accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Baseline accuracy
    baseline_train_acc_mean = baseline_curves['train_acc'].mean(axis=0)
    baseline_train_acc_std = baseline_curves['train_acc'].std(axis=0)
    baseline_val_acc_mean = baseline_curves['val_acc'].mean(axis=0)
    baseline_val_acc_std = baseline_curves['val_acc'].std(axis=0)

    ax1.plot(epochs, baseline_train_acc_mean * 100, 'b-', label='Train', linewidth=2)
    ax1.fill_between(epochs,
                      (baseline_train_acc_mean - baseline_train_acc_std) * 100,
                      (baseline_train_acc_mean + baseline_train_acc_std) * 100,
                      alpha=0.2, color='b')
    ax1.plot(epochs, baseline_val_acc_mean * 100, 'r-', label='Val', linewidth=2)
    ax1.fill_between(epochs,
                      (baseline_val_acc_mean - baseline_val_acc_std) * 100,
                      (baseline_val_acc_mean + baseline_val_acc_std) * 100,
                      alpha=0.2, color='r')

    ax1.axhline(65.7, color='gray', linestyle='--', linewidth=1.5,
                label='Majority baseline (65.7%)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('BASELINE: Accuracy Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([60, 100])

    # Improved accuracy
    improved_train_acc_mean = improved_curves['train_acc'].mean(axis=0)
    improved_train_acc_std = improved_curves['train_acc'].std(axis=0)
    improved_val_acc_mean = improved_curves['val_acc'].mean(axis=0)
    improved_val_acc_std = improved_curves['val_acc'].std(axis=0)

    ax2.plot(epochs_improved, improved_train_acc_mean * 100, 'b-', label='Train', linewidth=2)
    ax2.fill_between(epochs_improved,
                      (improved_train_acc_mean - improved_train_acc_std) * 100,
                      (improved_train_acc_mean + improved_train_acc_std) * 100,
                      alpha=0.2, color='b')
    ax2.plot(epochs_improved, improved_val_acc_mean * 100, 'r-', label='Val', linewidth=2)
    ax2.fill_between(epochs_improved,
                      (improved_val_acc_mean - improved_val_acc_std) * 100,
                      (improved_val_acc_mean + improved_val_acc_std) * 100,
                      alpha=0.2, color='r')

    ax2.axhline(65.7, color='gray', linestyle='--', linewidth=1.5,
                label='Majority baseline (65.7%)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('IMPROVED: Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([60, 100])

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'accuracy_comparison.png'}")

    # Plot 3: Direct comparison (overlay)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss overlay
    ax1.plot(epochs, baseline_train_mean, 'b--', label='Baseline Train', linewidth=2, alpha=0.7)
    ax1.plot(epochs, baseline_val_mean, 'r--', label='Baseline Val', linewidth=2, alpha=0.7)
    ax1.plot(epochs_improved, improved_train_mean, 'b-', label='Improved Train', linewidth=2)
    ax1.plot(epochs_improved, improved_val_mean, 'r-', label='Improved Val', linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss: Baseline vs. Improved', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Accuracy overlay
    ax2.plot(epochs, baseline_val_acc_mean * 100, 'r--', label='Baseline Val', linewidth=2, alpha=0.7)
    ax2.plot(epochs_improved, improved_val_acc_mean * 100, 'r-', label='Improved Val', linewidth=2)
    ax2.axhline(65.7, color='gray', linestyle='--', linewidth=1.5,
                label='Majority baseline')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy: Baseline vs. Improved', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([60, 100])

    plt.tight_layout()
    plt.savefig(output_dir / 'overlay_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'overlay_comparison.png'}")


def print_comparison_report(baseline_metrics, improved_metrics, output_dir):
    """Print and save comprehensive comparison report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []

    def add_line(line=""):
        report_lines.append(line)
        print(line)

    add_line("=" * 80)
    add_line("CCBPN TRAINING COMPARISON REPORT")
    add_line("=" * 80)

    # Extract final metrics
    baseline_best = baseline_metrics['best_val_acc']
    improved_best = improved_metrics['best_val_acc']

    baseline_folds = [f['best_val_acc'] for f in baseline_metrics['fold_results']]
    improved_folds = [f['best_val_acc'] for f in improved_metrics['fold_results']]

    baseline_mean = np.mean(baseline_folds)
    baseline_std = np.std(baseline_folds)
    improved_mean = np.mean(improved_folds)
    improved_std = np.std(improved_folds)

    # Configuration comparison
    add_line("\n" + "-" * 80)
    add_line("CONFIGURATION COMPARISON")
    add_line("-" * 80)

    baseline_args = baseline_metrics['args']
    improved_args = improved_metrics['args']

    add_line(f"\n{'Parameter':<25s} {'Baseline':<20s} {'Improved':<20s} {'Change':<15s}")
    add_line("-" * 80)

    configs = [
        ('KC Sparsity', 'kc_sparsity', '{:.1%}'),
        ('Dropout Rate', 'dropout', '{:.1%}'),
        ('Learning Rate', 'learning_rate', '{:.4f}'),
        ('Class Weights', 'use_class_weights', '{}'),
        ('LR Scheduler', 'use_lr_scheduler', '{}'),
        ('Epochs', 'epochs', '{}'),
    ]

    for label, key, fmt in configs:
        baseline_val = baseline_args.get(key, 'N/A')
        improved_val = improved_args.get(key, 'N/A')

        if baseline_val != 'N/A' and improved_val != 'N/A':
            baseline_str = fmt.format(baseline_val)
            improved_str = fmt.format(improved_val)
        else:
            baseline_str = str(baseline_val)
            improved_str = str(improved_val)

        if baseline_val != improved_val:
            change = "✓ CHANGED"
        else:
            change = ""

        add_line(f"{label:<25s} {baseline_str:<20s} {improved_str:<20s} {change:<15s}")

    # Performance comparison
    add_line("\n" + "-" * 80)
    add_line("PERFORMANCE COMPARISON")
    add_line("-" * 80)

    add_line(f"\nMajority class baseline: 65.7%")
    add_line(f"\nBaseline (before fixes):")
    add_line(f"  Average validation accuracy: {baseline_mean:.1%} ± {baseline_std:.1%}")
    add_line(f"  Best validation accuracy:     {baseline_best:.1%}")
    add_line(f"  Improvement over baseline:    {(baseline_mean - 0.657)*100:+.1f} pp")

    add_line(f"\nImproved (after fixes):")
    add_line(f"  Average validation accuracy: {improved_mean:.1%} ± {improved_std:.1%}")
    add_line(f"  Best validation accuracy:     {improved_best:.1%}")
    add_line(f"  Improvement over baseline:    {(improved_mean - 0.657)*100:+.1f} pp")

    add_line(f"\nNet improvement:")
    add_line(f"  Δ Average accuracy: {(improved_mean - baseline_mean)*100:+.1f} pp")
    add_line(f"  Δ Best accuracy:    {(improved_best - baseline_best)*100:+.1f} pp")
    add_line(f"  Relative improvement: {((improved_mean - baseline_mean) / baseline_mean)*100:+.1f}%")

    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(baseline_folds, improved_folds)
    add_line(f"\nStatistical significance (t-test):")
    add_line(f"  t-statistic: {t_stat:.3f}")
    add_line(f"  p-value:     {p_value:.4f}")
    if p_value < 0.05:
        add_line(f"  Result: ✓ SIGNIFICANT (p < 0.05)")
    else:
        add_line(f"  Result: ✗ Not significant (p >= 0.05)")

    # Per-fold breakdown
    add_line("\n" + "-" * 80)
    add_line("PER-FOLD BREAKDOWN")
    add_line("-" * 80)

    add_line(f"\n{'Fold':<10s} {'Baseline':<15s} {'Improved':<15s} {'Δ':<15s}")
    add_line("-" * 80)
    for i, (b_acc, i_acc) in enumerate(zip(baseline_folds, improved_folds), 1):
        delta = (i_acc - b_acc) * 100
        add_line(f"Fold {i:<5d} {b_acc:.1%}{'':<8s} {i_acc:.1%}{'':<8s} {delta:+.1f} pp")

    # Convergence analysis
    add_line("\n" + "-" * 80)
    add_line("CONVERGENCE ANALYSIS")
    add_line("-" * 80)

    baseline_curves = extract_fold_curves(baseline_metrics)
    improved_curves = extract_fold_curves(improved_metrics)

    baseline_final_loss = baseline_curves['train_loss'][:, -1].mean()
    improved_final_loss = improved_curves['train_loss'][:, -1].mean()

    add_line(f"\nFinal training loss:")
    add_line(f"  Baseline: {baseline_final_loss:.4f}")
    add_line(f"  Improved: {improved_final_loss:.4f}")
    add_line(f"  Δ: {improved_final_loss - baseline_final_loss:+.4f} ({((improved_final_loss - baseline_final_loss) / baseline_final_loss)*100:+.1f}%)")

    # Summary
    add_line("\n" + "=" * 80)
    add_line("SUMMARY")
    add_line("=" * 80)

    if improved_mean > baseline_mean + 0.05:
        add_line("\n🟢 MAJOR IMPROVEMENT")
        add_line(f"   Fixes resulted in {(improved_mean - baseline_mean)*100:.1f} pp accuracy gain!")
    elif improved_mean > baseline_mean + 0.02:
        add_line("\n🟡 MODERATE IMPROVEMENT")
        add_line(f"   Fixes resulted in {(improved_mean - baseline_mean)*100:.1f} pp accuracy gain.")
    elif improved_mean > baseline_mean:
        add_line("\n🟡 SLIGHT IMPROVEMENT")
        add_line(f"   Fixes resulted in {(improved_mean - baseline_mean)*100:.1f} pp accuracy gain.")
    else:
        add_line("\n🔴 NO IMPROVEMENT")
        add_line(f"   Fixes did not improve performance.")

    add_line("\nFixes applied:")
    if improved_args.get('use_class_weights', False):
        add_line("  ✓ Class-balanced loss")
    if improved_args.get('kc_sparsity', 0.05) > 0.05:
        add_line(f"  ✓ Increased KC sparsity ({baseline_args.get('kc_sparsity', 0.05):.1%} → {improved_args.get('kc_sparsity', 0.10):.1%})")
    if improved_args.get('use_lr_scheduler', False):
        add_line("  ✓ Learning rate scheduler")
    if improved_args.get('dropout', 0.0) > 0:
        add_line(f"  ✓ Dropout regularization ({improved_args.get('dropout', 0.0):.1%})")

    add_line("\n" + "=" * 80)

    # Save report
    report_path = output_dir / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n✓ Report saved to: {report_path}")


def main():
    """Main comparison pipeline."""
    args = parse_args()

    # Load metrics
    print("Loading metrics...")
    baseline_metrics = load_metrics(args.baseline)
    improved_metrics = load_metrics(args.improved)

    # Extract curves
    print("\nExtracting training curves...")
    baseline_curves = extract_fold_curves(baseline_metrics)
    improved_curves = extract_fold_curves(improved_metrics)

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(baseline_curves, improved_curves, args.output_dir)

    # Print report
    print("\n" + "=" * 80)
    print_comparison_report(baseline_metrics, improved_metrics, args.output_dir)


if __name__ == "__main__":
    main()
