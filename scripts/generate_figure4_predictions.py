#!/usr/bin/env python3
"""
Generate Figure 4 Predictions from Realistic Training.

This script takes the test results from realistic_behavioral_training.py
and generates predictions for Figure 4 (Behavioral Validation), comparing
model predictions to observed fly behavior.

Workflow:
1. Load test results from realistic training protocol
2. Calculate response rates per odor
3. Compare with observed behavioral data
4. Generate Figure 4 with observed vs. predicted bars
5. Calculate R² goodness-of-fit metric

Author: PGCN Enhancement
Date: 2025-11-11
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Observed behavioral data from real fly experiments
# Source: Cole Hanan's experiments (from generate_publication_figures.py)
OBSERVED_DATA = {
    'benzaldehyde': {'response_rate': 0.21, 'n': 48},
    '1-hexanol': {'response_rate': 0.65, 'n': 51},
    'ethyl_butyrate': {'response_rate': 0.50, 'n': 45},
    '3-octanol': {'response_rate': 0.44, 'n': 47},
    'linalool': {'response_rate': 0.31, 'n': 49},
}


def load_test_results(results_dir: Path) -> pd.DataFrame:
    """
    Load test results from realistic training protocol.

    Args:
        results_dir: Directory containing test_results.csv

    Returns:
        pd.DataFrame: Test trial results
    """
    test_results_path = results_dir / 'test_results.csv'

    if not test_results_path.exists():
        raise FileNotFoundError(
            f"Test results not found: {test_results_path}\n"
            f"Please run realistic_behavioral_training.py first!"
        )

    df = pd.read_csv(test_results_path)
    print(f"✓ Loaded {len(df)} test trials from {test_results_path}")

    return df


def calculate_predicted_responses(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate predicted response rates from test trials.

    Args:
        test_df: Test results DataFrame

    Returns:
        pd.DataFrame: Per-odor predicted response rates
    """
    # Group by odor and calculate response rate
    predictions = test_df.groupby('odor').agg({
        'response': ['mean', 'sum', 'count'],
        'peak_mbon': 'mean',
        'mean_mbon': 'mean'
    }).round(4)

    predictions.columns = ['predicted_response', 'n_responses', 'n_tests', 'avg_peak_mbon', 'avg_mean_mbon']
    predictions = predictions.reset_index()

    print(f"\n✓ Calculated predictions for {len(predictions)} odors")
    print(predictions[['odor', 'predicted_response', 'n_tests']].to_string(index=False))

    return predictions


def match_with_observed(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match predicted responses with observed behavioral data.

    Args:
        predictions_df: Predicted response rates

    Returns:
        pd.DataFrame: Matched observed and predicted data
    """
    # Standardize odor names for matching
    odor_name_map = {
        'benzaldehyde': 'Benzaldehyde',
        '1-hexanol': 'Hexanol',
        'ethyl_butyrate': 'Ethyl Butyrate',
        '3-octanol': '3-Octanol',
        'linalool': 'Linalool'
    }

    matched_data = []

    for odor_raw, display_name in odor_name_map.items():
        # Get predicted value
        pred_row = predictions_df[predictions_df['odor'] == odor_raw]

        if len(pred_row) == 0:
            print(f"  Warning: No prediction for {odor_raw}, skipping")
            continue

        predicted = pred_row.iloc[0]['predicted_response']

        # Get observed value
        if odor_raw in OBSERVED_DATA:
            observed = OBSERVED_DATA[odor_raw]['response_rate']
            n_flies = OBSERVED_DATA[odor_raw]['n']
        else:
            print(f"  Warning: No observed data for {odor_raw}, skipping")
            continue

        matched_data.append({
            'odor': display_name,
            'odor_raw': odor_raw,
            'observed': observed,
            'predicted': predicted,
            'n_flies': n_flies
        })

    matched_df = pd.DataFrame(matched_data)

    print(f"\n✓ Matched {len(matched_df)} odors with observed data")

    return matched_df


def calculate_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).

    R² measures how well predictions explain variance in observed data.
    R² = 1 - (SS_res / SS_tot)

    Args:
        observed: Observed response rates
        predicted: Predicted response rates

    Returns:
        float: R² value (0 to 1)
    """
    # Sum of squared residuals
    ss_res = np.sum((observed - predicted) ** 2)

    # Total sum of squares
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)

    # R-squared
    if ss_tot == 0:
        return 0.0

    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def generate_figure4(
    matched_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Generate Figure 4: Observed vs. Predicted Response.

    Creates a bar chart comparing observed fly behavior to model predictions.

    Args:
        matched_df: DataFrame with observed and predicted values
        output_path: Path to save figure
    """
    print("\n📊 Generating Figure 4...")

    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['figure.dpi'] = 300

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(matched_df))
    width = 0.35

    # Bars
    bars1 = ax.bar(
        x - width/2,
        matched_df['observed'],
        width,
        label='Observed (behavior)',
        color='gray',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7
    )

    bars2 = ax.bar(
        x + width/2,
        matched_df['predicted'],
        width,
        label='Predicted (connectome)',
        color='#C44E52',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.6
    )

    # Labels and title
    ax.set_ylabel('Response Rate', fontsize=14, fontweight='bold')
    ax.set_xlabel('Odor (10% dilution)', fontsize=14, fontweight='bold')
    ax.set_title('Observed vs. Predicted Response\n(Realistic Training Protocol)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(matched_df['odor'], fontsize=11)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Calculate and display R²
    r_squared = calculate_r_squared(
        matched_df['observed'].values,
        matched_df['predicted'].values
    )

    ax.text(
        0.05, 0.95,
        f'$R^2 = {r_squared:.3f}$',
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='lightgreen',
            alpha=0.8,
            edgecolor='black',
            linewidth=2
        )
    )

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')

    print(f"  ✓ Saved figure: {output_path}")
    print(f"  ✓ Saved PDF: {output_path.with_suffix('.pdf')}")

    plt.close()


def generate_comparison_report(
    matched_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Generate detailed comparison report.

    Args:
        matched_df: Matched observed/predicted data
        output_path: Path to save report
    """
    print("\n📝 Generating comparison report...")

    # Calculate metrics
    r_squared = calculate_r_squared(
        matched_df['observed'].values,
        matched_df['predicted'].values
    )

    pearson_r = np.corrcoef(
        matched_df['observed'].values,
        matched_df['predicted'].values
    )[0, 1]

    rmse = np.sqrt(np.mean(
        (matched_df['observed'].values - matched_df['predicted'].values) ** 2
    ))

    mae = np.mean(np.abs(
        matched_df['observed'].values - matched_df['predicted'].values
    ))

    # Create report
    report = []
    report.append("=" * 70)
    report.append("BEHAVIORAL VALIDATION REPORT")
    report.append("Realistic Training Protocol vs. Observed Behavior")
    report.append("=" * 70)
    report.append("")

    report.append("GOODNESS-OF-FIT METRICS:")
    report.append(f"  R² (coefficient of determination): {r_squared:.4f}")
    report.append(f"  Pearson r (correlation):            {pearson_r:.4f}")
    report.append(f"  RMSE (root mean square error):      {rmse:.4f}")
    report.append(f"  MAE (mean absolute error):          {mae:.4f}")
    report.append("")

    report.append("PER-ODOR COMPARISON:")
    report.append("-" * 70)
    report.append(f"{'Odor':<20s} {'Observed':>10s} {'Predicted':>10s} {'Error':>10s} {'% Error':>10s}")
    report.append("-" * 70)

    for _, row in matched_df.iterrows():
        error = row['predicted'] - row['observed']
        pct_error = (error / row['observed']) * 100 if row['observed'] != 0 else 0

        report.append(
            f"{row['odor']:<20s} "
            f"{row['observed']:>10.3f} "
            f"{row['predicted']:>10.3f} "
            f"{error:>10.3f} "
            f"{pct_error:>10.1f}%"
        )

    report.append("-" * 70)
    report.append("")

    report.append("INTERPRETATION:")
    if r_squared >= 0.90:
        report.append(f"  ✓ EXCELLENT: R² = {r_squared:.3f} indicates high predictive accuracy")
    elif r_squared >= 0.75:
        report.append(f"  ✓ GOOD: R² = {r_squared:.3f} indicates reasonable predictive accuracy")
    elif r_squared >= 0.50:
        report.append(f"  • MODERATE: R² = {r_squared:.3f} indicates moderate predictive accuracy")
    else:
        report.append(f"  ✗ POOR: R² = {r_squared:.3f} indicates low predictive accuracy")

    report.append("")
    report.append("=" * 70)

    # Write report
    report_text = "\n".join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  ✓ Saved report: {output_path}")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Figure 4 predictions from realistic training'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results/realistic_training'),
        help='Directory with test results from realistic training'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/figure4_validation'),
        help='Output directory for figure and reports'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FIGURE 4 PREDICTION GENERATOR")
    print("=" * 70)
    print(f"Results: {args.results_dir}")
    print(f"Output:  {args.output_dir}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load test results
    test_df = load_test_results(args.results_dir)

    # Step 2: Calculate predictions
    predictions_df = calculate_predicted_responses(test_df)

    # Save predictions
    predictions_path = args.output_dir / 'predicted_responses.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\n✓ Saved predictions: {predictions_path}")

    # Step 3: Match with observed data
    matched_df = match_with_observed(predictions_df)

    # Save matched data
    matched_path = args.output_dir / 'observed_vs_predicted.csv'
    matched_df.to_csv(matched_path, index=False)
    print(f"✓ Saved matched data: {matched_path}")

    # Step 4: Generate Figure 4
    figure_path = args.output_dir / 'fig4_behavioral_validation_realistic.png'
    generate_figure4(matched_df, figure_path)

    # Step 5: Generate comparison report
    report_path = args.output_dir / 'behavioral_validation_report.txt'
    generate_comparison_report(matched_df, report_path)

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  • {predictions_path}")
    print(f"  • {matched_path}")
    print(f"  • {figure_path}")
    print(f"  • {figure_path.with_suffix('.pdf')}")
    print(f"  • {report_path}")
    print()


if __name__ == '__main__':
    main()