#!/usr/bin/env python3
"""
KC Overlap vs Veto Protection Visualization

Creates publication-quality 3-panel figure showing:
1. Chemical similarity vs KC overlap (validation of sparse coding hypothesis)
2. KC overlap vs protection benefit (main result)
3. Forgetting comparison across similarity levels (bar chart)

Author: PGCN Project
Date: 2025-11-18

Usage:
    python scripts/experiments/plot_overlap_vs_protection.py \
        --input reports/experiments/overlap/kc_overlap_results.csv \
        --output reports/figures/kc_overlap_vs_protection.pdf
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_three_panel_figure(df: pd.DataFrame, output_path: Path) -> None:
    """Create 3-panel publication figure.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe with columns: chemical_similarity, kc_overlap_pct,
        protection_benefit_pct, baseline_forgetting_pct, veto_forgetting_pct.
    output_path : Path
        Output PDF file path.
    """
    # Set publication style
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel A: Chemical Similarity vs KC Overlap
    ax = axes[0]
    ax.scatter(
        df['chemical_similarity'] * 100,
        df['kc_overlap_pct'],
        s=150,
        c=df['kc_overlap_pct'],
        cmap='viridis',
        edgecolors='black',
        linewidth=1.5,
        alpha=0.8,
    )

    # Add linear fit
    x = df['chemical_similarity'].values * 100
    y = df['kc_overlap_pct'].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, p(x_fit), 'r--', alpha=0.5, linewidth=2, label=f'Linear fit (R²={np.corrcoef(x, y)[0,1]**2:.2f})')

    # Annotate points
    for _, row in df.iterrows():
        ax.annotate(
            row['similarity_level'].upper(),
            xy=(row['chemical_similarity'] * 100, row['kc_overlap_pct']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
        )

    ax.set_xlabel('Chemical Similarity (%)', fontweight='bold')
    ax.set_ylabel('KC Overlap (%)', fontweight='bold')
    ax.set_title('A. Sparse Coding Validation', fontweight='bold', fontsize=14)
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel B: KC Overlap vs Protection Benefit
    ax = axes[1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    for idx, (_, row) in enumerate(df.iterrows()):
        ax.bar(
            row['kc_overlap_pct'],
            row['protection_benefit_pct'],
            width=3,
            color=colors[idx],
            edgecolor='black',
            linewidth=1.5,
            alpha=0.7,
            label=f"{row['similarity_level'].upper()} ({row['kc_overlap_pct']:.1f}% overlap)",
        )

    # Add expectation line (y = x, meaning benefit scales with overlap)
    max_overlap = df['kc_overlap_pct'].max()
    if max_overlap > 0:
        ax.plot([0, max_overlap], [0, max_overlap], 'k--', alpha=0.3, linewidth=2, label='Expected (y=x)')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
    ax.set_xlabel('KC Overlap (%)', fontweight='bold')
    ax.set_ylabel('Protection Benefit (%)', fontweight='bold')
    ax.set_title('B. KC Overlap vs Protection Efficacy', fontweight='bold', fontsize=14)
    ax.legend(frameon=True, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Forgetting Comparison (Baseline vs Veto)
    ax = axes[2]
    x_pos = np.arange(len(df))
    width = 0.35

    baseline_bars = ax.bar(
        x_pos - width/2,
        df['baseline_forgetting_pct'],
        width,
        label='Baseline (no protection)',
        color='#d62728',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7,
    )

    veto_bars = ax.bar(
        x_pos + width/2,
        df['veto_forgetting_pct'],
        width,
        label='Veto Gate',
        color='#2ca02c',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7,
    )

    # Add value labels on bars
    for bars in [baseline_bars, veto_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=9,
                fontweight='bold',
            )

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
    ax.set_xlabel('Similarity Level', fontweight='bold')
    ax.set_ylabel('Forgetting (%)', fontweight='bold')
    ax.set_title('C. Catastrophic Forgetting Prevention', fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([level.upper() for level in df['similarity_level']])
    ax.legend(frameon=True, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

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
        print(f"\n{row['similarity_level'].upper()} SIMILARITY:")
        print(f"  Chemical similarity: {row['chemical_similarity']:.3f}")
        print(f"  KC overlap: {row['kc_overlap_pct']:.1f}% ({row['kc_overlap_count']} KCs)")
        print(f"  Jaccard index: {row['jaccard_index']:.3f}")
        print(f"  Baseline forgetting: {row['baseline_forgetting_pct']:.1f}%")
        print(f"  Veto forgetting: {row['veto_forgetting_pct']:.1f}%")
        print(f"  Protection benefit: {row['protection_benefit_pct']:+.1f}%")
        print(f"  Relative reduction: {row['relative_reduction']:.1%}")

    # Correlation analysis
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    if len(df) >= 2:
        corr_chem_overlap = df[['chemical_similarity', 'kc_overlap_pct']].corr().iloc[0, 1]
        corr_overlap_benefit = df[['kc_overlap_pct', 'protection_benefit_pct']].corr().iloc[0, 1]

        print(f"Chemical Similarity ↔ KC Overlap: r = {corr_chem_overlap:.3f}")
        print(f"KC Overlap ↔ Protection Benefit: r = {corr_overlap_benefit:.3f}")
    else:
        print("Insufficient data points for correlation analysis (need ≥ 2)")


def main():
    """Run visualization from command line."""
    parser = argparse.ArgumentParser(
        description="Plot KC Overlap vs Protection Benefit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file from kc_overlap_vs_veto_efficacy.py'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/figures/kc_overlap_vs_protection.pdf',
        help='Output PDF file path (default: reports/figures/kc_overlap_vs_protection.pdf)'
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
        'similarity_level',
        'chemical_similarity',
        'kc_overlap_pct',
        'kc_overlap_count',
        'jaccard_index',
        'baseline_forgetting_pct',
        'veto_forgetting_pct',
        'protection_benefit_pct',
        'relative_reduction',
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Error: Missing required columns: {missing}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Adjust extension if format specified
    if args.format != 'pdf':
        output_path = output_path.with_suffix(f'.{args.format}')

    # Generate figure
    print("\nGenerating 3-panel figure...")
    create_three_panel_figure(df, output_path)

    # Print summary
    print_summary_statistics(df)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
