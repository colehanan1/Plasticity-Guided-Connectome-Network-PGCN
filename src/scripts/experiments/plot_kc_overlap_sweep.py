#!/usr/bin/env python3
"""
KC Overlap Sweep Visualization

Creates publication-quality 2-panel figure showing:
1. KC overlap vs protection benefit (main result)
2. Baseline vs veto forgetting across overlap levels

Key Finding:
Veto gate protection shows variable efficacy (23-92% reduction) across
KC overlap range of 0-43%, with high efficacy in 0% and 16-18% overlap
conditions.

Usage:
    python scripts/experiments/plot_kc_overlap_sweep.py \
        --input reports/experiments/overlap/kc_overlap_sweep.csv \
        --output reports/figures/kc_overlap_sweep.pdf

Author: PGCN Project
Date: 2025-11-19
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_two_panel_figure(df: pd.DataFrame, output_path: Path) -> None:
    """Create 2-panel publication figure.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe with columns: measured_kc_overlap_pct,
        relative_reduction_pct, baseline_forgetting_absolute,
        veto_forgetting_absolute.
    output_path : Path
        Output PDF file path.
    """
    # Set publication style
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: KC Overlap vs Protection Benefit
    ax = axes[0]

    # Scatter plot with color gradient by chemical similarity
    scatter = ax.scatter(
        df['measured_kc_overlap_pct'],
        df['relative_reduction_pct'],
        s=200,
        c=df['chemical_similarity'],
        cmap='viridis',
        edgecolors='black',
        linewidth=2,
        alpha=0.85,
        zorder=3,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Chemical Similarity', fontweight='bold', fontsize=12)

    # Add trend line (linear regression)
    x = df['measured_kc_overlap_pct'].values
    y = df['relative_reduction_pct'].values

    # Filter out outliers/noise for regression
    mask = (y >= 0) & (y <= 100)
    if mask.sum() >= 2:
        x_filtered = x[mask]
        y_filtered = y[mask]

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_filtered, y_filtered)

        # Plot regression line
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(
            x_fit, y_fit,
            'r--',
            alpha=0.6,
            linewidth=2.5,
            label=f'Linear fit (R={r_value:.2f}, p={p_value:.3f})',
            zorder=2,
        )

    # Reference lines
    ax.axhline(0, color='gray', linewidth=1, linestyle=':', alpha=0.5, zorder=1)
    ax.axhline(50, color='gray', linewidth=1, linestyle=':', alpha=0.3, zorder=1)

    # Labels and title
    ax.set_xlabel('KC Overlap (%)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Protection Benefit\n(% Reduction in Forgetting)', fontweight='bold', fontsize=14)
    ax.set_title('A. KC Overlap vs Veto Gate Protection', fontweight='bold', fontsize=15, pad=15)
    ax.legend(frameon=True, loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, zorder=0)

    # Set axis limits
    ax.set_xlim(-2, max(x) + 5)
    ax.set_ylim(-10, max(y) + 10)

    # Panel B: Forgetting Comparison (Baseline vs Veto)
    ax = axes[1]

    # Sort by KC overlap for better visualization
    df_sorted = df.sort_values('measured_kc_overlap_pct')

    x_pos = np.arange(len(df_sorted))
    width = 0.35

    baseline_bars = ax.bar(
        x_pos - width/2,
        df_sorted['baseline_forgetting_absolute'],
        width,
        label='Baseline (no protection)',
        color='#d62728',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.75,
        zorder=2,
    )

    veto_bars = ax.bar(
        x_pos + width/2,
        df_sorted['veto_forgetting_absolute'],
        width,
        label='Veto Gate',
        color='#2ca02c',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.75,
        zorder=2,
    )

    # Add value labels on bars
    for bars in [baseline_bars, veto_bars]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # Only label non-trivial values
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                )

    # Labels and title
    ax.set_xlabel('Condition (sorted by KC overlap)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Absolute Forgetting\n(|MBON_before - MBON_after|)', fontweight='bold', fontsize=14)
    ax.set_title('B. Catastrophic Forgetting Prevention', fontweight='bold', fontsize=15, pad=15)

    # X-axis labels showing KC overlap percentage
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{row['measured_kc_overlap_pct']:.0f}%" for _, row in df_sorted.iterrows()],
        rotation=45,
        ha='right',
        fontsize=10,
    )

    ax.legend(frameon=True, loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to: {output_path}")


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics from the results.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for _, row in df.iterrows():
        print(f"\n{row['condition'].upper()}:")
        print(f"  Description: {row['description']}")
        print(f"  KC overlap: {row['measured_kc_overlap_pct']:.1f}% "
              f"(expected: {row['expected_kc_overlap']:.1f}%)")
        print(f"  Chemical similarity: {row['chemical_similarity']:.3f}")
        print(f"  Baseline forgetting: {row['baseline_forgetting_absolute']:.3f}")
        print(f"  Veto forgetting: {row['veto_forgetting_absolute']:.3f}")
        print(f"  Protection benefit: {row['protection_benefit_absolute']:+.3f} "
              f"({row['relative_reduction_pct']:.1f}% reduction)")

    # Correlation analysis
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    if len(df) >= 2:
        # Filter out cases with zero baseline forgetting (no protection needed)
        df_nonzero = df[df['baseline_forgetting_absolute'] > 0.01]

        if len(df_nonzero) >= 2:
            corr_overlap_benefit = df_nonzero[['measured_kc_overlap_pct', 'relative_reduction_pct']].corr().iloc[0, 1]
            print(f"KC Overlap ↔ Protection Benefit: r = {corr_overlap_benefit:.3f} "
                  f"(n={len(df_nonzero)} conditions with forgetting)")

            # Linear regression
            x = df_nonzero['measured_kc_overlap_pct'].values
            y = df_nonzero['relative_reduction_pct'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            print(f"  Linear regression: y = {slope:.2f}x + {intercept:.1f}")
            print(f"  R² = {r_value**2:.3f}, p = {p_value:.4f}")
        else:
            print("Insufficient non-zero forgetting cases for correlation analysis")
    else:
        print("Insufficient data points for correlation analysis (need ≥ 2)")

    # Summary statistics
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)

    print(f"Total conditions tested: {len(df)}")
    print(f"KC overlap range: {df['measured_kc_overlap_pct'].min():.1f}% - "
          f"{df['measured_kc_overlap_pct'].max():.1f}%")
    print(f"Protection benefit range: {df['relative_reduction_pct'].min():.1f}% - "
          f"{df['relative_reduction_pct'].max():.1f}%")
    print(f"Mean protection benefit: {df['relative_reduction_pct'].mean():.1f}%")
    print(f"Median protection benefit: {df['relative_reduction_pct'].median():.1f}%")


def main():
    """Run visualization from command line."""
    parser = argparse.ArgumentParser(
        description="Plot KC Overlap Sweep Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file from KC overlap sweep experiment'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/figures/kc_overlap_sweep.pdf',
        help='Output PDF file path (default: reports/figures/kc_overlap_sweep.pdf)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='pdf',
        choices=['pdf', 'png', 'svg'],
        help='Output format (default: pdf)'
    )

    args = parser.parse_args()

    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df)} results from: {input_path}")

    # Validate required columns
    required_cols = [
        'condition',
        'expected_kc_overlap',
        'measured_kc_overlap_pct',
        'chemical_similarity',
        'description',
        'baseline_forgetting_absolute',
        'veto_forgetting_absolute',
        'protection_benefit_absolute',
        'relative_reduction_pct',
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Error: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Adjust extension if format specified
    if args.format != 'pdf':
        output_path = output_path.with_suffix(f'.{args.format}')

    # Generate figure
    print("\nGenerating 2-panel figure...")
    create_two_panel_figure(df, output_path)

    # Print summary
    print_summary_statistics(df)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
