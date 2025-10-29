#!/usr/bin/env python3
"""
Standalone script to run statistical tests on existing cross-validation artifacts.

This script allows re-running statistical analysis without re-training models,
useful for:
- Experimenting with different statistical parameters
- Adding new statistical tests to existing results
- Generating reports with different significance levels

Usage:
    python analysis/run_statistical_tests.py --artifacts-dir artifacts/cross_validation
    python analysis/run_statistical_tests.py --artifacts-dir artifacts/cross_validation --report-prefix week4
    python analysis/run_statistical_tests.py --artifacts-dir artifacts/cross_validation --n-permutations 10000

Requirements:
    - Cross-validation artifacts must exist in the specified directory
    - Expects fold_*.json files and {prefix}_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from statistical_tests import run_all_statistical_tests


def load_fold_results(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """
    Load fold results from individual fold JSON files.

    Args:
        artifacts_dir: Directory containing fold_*.json files

    Returns:
        List of fold result dictionaries

    Raises:
        FileNotFoundError: If no fold files found
        json.JSONDecodeError: If JSON files are malformed
    """
    fold_files = sorted(artifacts_dir.glob("fold_*.json"))

    if not fold_files:
        raise FileNotFoundError(
            f"No fold_*.json files found in {artifacts_dir}. "
            "Make sure cross-validation has been run first."
        )

    fold_results = []
    for fold_file in fold_files:
        with fold_file.open("r", encoding="utf-8") as f:
            fold_data = json.load(f)
            fold_results.append(fold_data)

    print(f"Loaded {len(fold_results)} fold results from {artifacts_dir}")
    return fold_results


def load_aggregate_report(artifacts_dir: Path, report_prefix: str) -> Optional[Dict[str, Any]]:
    """
    Load aggregate cross-validation report.

    Args:
        artifacts_dir: Directory containing report JSON
        report_prefix: Prefix for report filename (e.g., "week4")

    Returns:
        Aggregate report dictionary, or None if not found
    """
    report_path = artifacts_dir / f"{report_prefix}_report.json"

    if not report_path.exists():
        print(f"Warning: Aggregate report not found at {report_path}")
        return None

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    print(f"Loaded aggregate report from {report_path}")
    return report


def extract_chemical_similarity_data(
    fold_results: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Extract chemical similarity and behavioral response data from fold results.

    This function attempts to extract paired chemical similarity and behavioral
    response data from the generalization results in each fold. If the data
    structure doesn't contain sufficient information, returns None.

    Args:
        fold_results: List of fold result dictionaries

    Returns:
        Dictionary with "similarities", "response_rates", "prediction_probabilities"
        or None if data cannot be extracted
    """
    similarities = []
    response_rates = []
    prediction_probabilities = []

    for fold_result in fold_results:
        if "generalisation" not in fold_result:
            continue

        for gen_entry in fold_result["generalisation"]:
            # Chemical similarity would need to be stored in CV results
            # For now, we'll return None since it's not in the current structure
            # This can be enhanced when chemical similarity is added to CV output
            pass

    # Current CV output doesn't include chemical similarity data
    # Return None for now - this will skip correlation analysis
    return None


def print_statistical_summary(report: Dict[str, Any]) -> None:
    """
    Print human-readable summary of statistical test results.

    Args:
        report: Statistical report dictionary from run_all_statistical_tests()
    """
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 80)

    # Metadata
    metadata = report["metadata"]
    print(f"\nMetadata:")
    print(f"  Folds: {metadata['n_folds']}")
    print(f"  Permutations: {metadata['n_permutations']}")
    print(f"  Bootstrap samples: {metadata['n_bootstrap_samples']}")
    print(f"  Chance level: {metadata['chance_level']:.2f}")

    # Permutation tests vs chance
    print(f"\n{'─' * 80}")
    print("PERMUTATION TESTS VS CHANCE LEVEL")
    print(f"{'─' * 80}")

    vs_chance = report["permutation_tests"]["vs_chance"]
    for metric_name, results in vs_chance.items():
        sig_05 = "✓" if results["significant_at_0.05"] else "✗"
        sig_01 = "✓✓" if results["significant_at_0.01"] else ""

        print(f"\n{metric_name}:")
        print(f"  Observed: {results['observed_mean']:.4f}")
        print(f"  Chance: {results['chance_level']:.4f}")
        print(f"  p-value (one-tailed): {results['p_value_one_tailed']:.4f} {sig_05} {sig_01}")
        print(f"  p-value (two-tailed): {results['p_value_two_tailed']:.4f}")

        # Effect size
        if metric_name in report["effect_sizes"]["vs_chance"]:
            effect = report["effect_sizes"]["vs_chance"][metric_name]
            print(f"  Cohen's d: {effect['cohens_d']:.4f} ({effect['interpretation']})")

    # Confidence intervals
    print(f"\n{'─' * 80}")
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"{'─' * 80}")

    cis = report["confidence_intervals"]
    for metric_name, ci_data in cis.items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {ci_data['mean']:.4f}")
        print(f"  95% CI: [{ci_data['ci_95'][0]:.4f}, {ci_data['ci_95'][1]:.4f}]")
        print(f"  99% CI: [{ci_data['ci_99'][0]:.4f}, {ci_data['ci_99'][1]:.4f}]")

    # Between-condition tests
    if report["permutation_tests"]["between_conditions"]:
        print(f"\n{'─' * 80}")
        print("PERMUTATION TESTS BETWEEN CONDITIONS")
        print(f"{'─' * 80}")

        between = report["permutation_tests"]["between_conditions"]
        for comparison_name, metrics in between.items():
            print(f"\n{comparison_name}:")
            for metric_name, results in metrics.items():
                sig_05 = "✓" if results["significant_at_0.05"] else "✗"
                sig_01 = "✓✓" if results["significant_at_0.01"] else ""

                print(f"  {metric_name}:")
                print(f"    Mean difference: {results['mean_difference']:.4f}")
                print(f"    p-value (two-tailed): {results['p_value_two_tailed']:.4f} {sig_05} {sig_01}")

                # Effect size
                effect_key = f"between_conditions.{comparison_name}.{metric_name}"
                if comparison_name in report["effect_sizes"]["between_conditions"]:
                    if metric_name in report["effect_sizes"]["between_conditions"][comparison_name]:
                        effect = report["effect_sizes"]["between_conditions"][comparison_name][metric_name]
                        print(f"    Cohen's d: {effect['cohens_d']:.4f} ({effect['interpretation']})")
                        if effect["eta_squared"] is not None:
                            print(f"    Eta-squared: {effect['eta_squared']:.4f} ({effect['eta_squared_interpretation']})")

    # Correlation analysis
    if report["correlations"]:
        print(f"\n{'─' * 80}")
        print("CHEMICAL SIMILARITY CORRELATION ANALYSIS")
        print(f"{'─' * 80}")

        for corr_name, corr_data in report["correlations"].items():
            sig_05 = "✓" if corr_data["significant_at_0.05"] else "✗"
            sig_01 = "✓✓" if corr_data["significant_at_0.01"] else ""

            print(f"\n{corr_name}:")
            print(f"  Pearson r: {corr_data['pearson_r']:.4f} (p={corr_data['pearson_p']:.4f}) {sig_05} {sig_01}")
            print(f"  Spearman ρ: {corr_data['spearman_rho']:.4f} (p={corr_data['spearman_p']:.4f})")
            print(f"  95% CI: [{corr_data['ci_95'][0]:.4f}, {corr_data['ci_95'][1]:.4f}]")
            print(f"  99% CI: [{corr_data['ci_99'][0]:.4f}, {corr_data['ci_99'][1]:.4f}]")
            print(f"  N samples: {corr_data['n_samples']}")

    print(f"\n{'=' * 80}")
    print("Legend: ✓ = p < 0.05, ✓✓ = p < 0.01")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run statistical tests on cross-validation artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python analysis/run_statistical_tests.py --artifacts-dir artifacts/cross_validation

  # Specify report prefix
  python analysis/run_statistical_tests.py \\
    --artifacts-dir artifacts/cross_validation \\
    --report-prefix week4

  # Use more permutations for higher precision
  python analysis/run_statistical_tests.py \\
    --artifacts-dir artifacts/cross_validation \\
    --n-permutations 10000 \\
    --n-bootstrap-samples 10000
        """
    )

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing cross-validation artifacts (fold_*.json files)"
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="week4",
        help="Prefix for report files (default: week4)"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
        help="Number of permutation resamples (default: 5000)"
    )
    parser.add_argument(
        "--n-bootstrap-samples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples (default: 5000)"
    )
    parser.add_argument(
        "--chance-level",
        type=float,
        default=0.52,
        help="Baseline chance performance (default: 0.52)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--no-print-summary",
        action="store_true",
        help="Skip printing summary to console"
    )

    args = parser.parse_args()

    # Validate artifacts directory exists
    if not args.artifacts_dir.exists():
        print(f"Error: Artifacts directory not found: {args.artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    # Load fold results
    try:
        fold_results = load_fold_results(args.artifacts_dir)
    except Exception as e:
        print(f"Error loading fold results: {e}", file=sys.stderr)
        sys.exit(1)

    # Load aggregate report (optional, for additional context)
    aggregate_report = load_aggregate_report(args.artifacts_dir, args.report_prefix)

    # Extract chemical similarity data (if available)
    chemical_similarity_data = extract_chemical_similarity_data(fold_results)
    if chemical_similarity_data is None:
        print("Note: Chemical similarity correlation analysis will be skipped (data not found in artifacts)")

    # Run statistical tests
    print(f"\nRunning statistical tests with {args.n_permutations} permutations and {args.n_bootstrap_samples} bootstrap samples...")

    statistical_report = run_all_statistical_tests(
        fold_results=fold_results,
        chance_level=args.chance_level,
        n_permutations=args.n_permutations,
        n_bootstrap_samples=args.n_bootstrap_samples,
        chemical_similarity_data=chemical_similarity_data,
        random_seed=args.random_seed
    )

    # Save statistical report (convert numpy types to native Python types)
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    output_path = args.artifacts_dir / f"{args.report_prefix}_statistical_report.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(statistical_report), f, indent=2)

    print(f"\nStatistical report saved to: {output_path}")

    # Print summary
    if not args.no_print_summary:
        print_statistical_summary(statistical_report)

    print(f"\n✓ Statistical analysis complete!")


if __name__ == "__main__":
    main()
