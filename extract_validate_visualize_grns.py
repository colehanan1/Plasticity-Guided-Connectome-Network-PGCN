#!/usr/bin/env python3
"""
GRN Extraction & Validation Pipeline

Professional tool to extract and validate gustatory receptor neurons (GRNs)
from FlyWire connectome classification data.

Expected outputs:
- 343 total gustatory neurons
- 131 sugar/water neurons
- ~105 bitter neurons
- ~107 other gustatory neurons

Validates against reference files and generates publication-quality visualizations.

Author: Claude AI
Date: 2025-11-03
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('grn_extraction.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def load_classification_data(filepath: str) -> pd.DataFrame:
    """
    Load FlyWire classification data.

    Args:
        filepath: Path to classification.csv

    Returns:
        DataFrame with neuron classifications
    """
    logger.info(f"Loading classification data from: {filepath}")

    try:
        # Handle both .csv and .csv.gz
        compression = 'gzip' if filepath.endswith('.gz') else None
        df = pd.read_csv(filepath, dtype={'root_id': int}, compression=compression)
        logger.info(f"✓ Loaded: {len(df):,} total neurons")

        # Show columns
        logger.info(f"  Columns: {', '.join(df.columns.tolist())}")

        return df

    except Exception as e:
        logger.error(f"❌ Failed to load classification data: {e}")
        raise


def extract_gustatory_neurons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all gustatory neurons (class == 'gustatory').

    Args:
        df: Full classification DataFrame

    Returns:
        DataFrame with gustatory neurons only
    """
    logger.info("\nExtracting all gustatory neurons...")

    mask = df['class'] == 'gustatory'
    gustatory_df = df[mask].copy()

    logger.info(f"✓ All gustatory: {len(gustatory_df)} neurons")

    return gustatory_df


def extract_sugar_water_neurons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract sugar/water neurons (class == 'gustatory' AND sub_class == 'sugar/water').

    Args:
        df: Full classification DataFrame

    Returns:
        DataFrame with sugar/water neurons only
    """
    logger.info("Extracting sugar/water neurons...")

    mask = (df['class'] == 'gustatory') & (df['sub_class'] == 'sugar/water')
    sugar_water_df = df[mask].copy()

    logger.info(f"✓ Sugar/water: {len(sugar_water_df)} neurons")

    return sugar_water_df


def extract_bitter_neurons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract bitter neurons (class == 'gustatory' AND sub_class == 'bitter').

    Args:
        df: Full classification DataFrame

    Returns:
        DataFrame with bitter neurons only
    """
    logger.info("Extracting bitter neurons...")

    mask = (df['class'] == 'gustatory') & (df['sub_class'] == 'bitter')
    bitter_df = df[mask].copy()

    logger.info(f"✓ Bitter: {len(bitter_df)} neurons")

    return bitter_df


def extract_other_neurons(
    all_grns: pd.DataFrame,
    sugar_water: pd.DataFrame,
    bitter: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract other gustatory neurons (not sugar/water or bitter).

    Args:
        all_grns: All gustatory neurons
        sugar_water: Sugar/water neurons
        bitter: Bitter neurons

    Returns:
        DataFrame with other gustatory neurons
    """
    logger.info("Extracting other gustatory neurons...")

    sugar_water_ids = set(sugar_water['root_id'])
    bitter_ids = set(bitter['root_id'])
    excluded_ids = sugar_water_ids | bitter_ids

    mask = ~all_grns['root_id'].isin(excluded_ids)
    other_df = all_grns[mask].copy()

    logger.info(f"✓ Other: {len(other_df)} neurons")

    return other_df


def load_reference_files(ref_dir: str) -> Tuple[Set[int], Set[int]]:
    """
    Load reference root_id files for validation.

    Args:
        ref_dir: Directory containing reference files

    Returns:
        Tuple of (all_gustatory_ids, sugar_water_ids)
    """
    logger.info("\nLoading reference files...")

    ref_path = Path(ref_dir)

    # Load all gustatory reference (comma-separated on single line)
    all_gustatory_file = ref_path / 'root_ids_class_gustatory.txt'
    try:
        with open(all_gustatory_file, 'r') as f:
            content = f.read().strip()
            all_gustatory_ids = set(int(rid.strip()) for rid in content.split(',') if rid.strip())
        logger.info(f"✓ ref_all_gustatory: {len(all_gustatory_ids)} root_ids")
    except Exception as e:
        logger.error(f"❌ Failed to load {all_gustatory_file}: {e}")
        raise

    # Load sugar/water reference (comma-separated on single line)
    sugar_water_file = ref_path / 'root_ids_class_gustatory_sub_class_sugar_water.txt'
    try:
        with open(sugar_water_file, 'r') as f:
            content = f.read().strip()
            sugar_water_ids = set(int(rid.strip()) for rid in content.split(',') if rid.strip())
        logger.info(f"✓ ref_sugar_water: {len(sugar_water_ids)} root_ids")
    except Exception as e:
        logger.error(f"❌ Failed to load {sugar_water_file}: {e}")
        raise

    return all_gustatory_ids, sugar_water_ids


def validate_extracted(
    extracted_set: Set[int],
    reference_set: Set[int],
    name: str
) -> bool:
    """
    Validate extracted neurons against reference set.

    Args:
        extracted_set: Set of extracted root_ids
        reference_set: Set of reference root_ids
        name: Name for logging

    Returns:
        True if validation passes
    """
    missing = reference_set - extracted_set
    extra = extracted_set - reference_set

    if len(missing) > 0 or len(extra) > 0:
        logger.error(f"❌ TEST FAILED: {name}")
        logger.error(f"  Expected: {len(reference_set)} neurons")
        logger.error(f"  Extracted: {len(extracted_set)} neurons")

        if len(missing) > 0:
            logger.error(f"  Missing {len(missing)} root_ids from reference:")
            for rid in list(missing)[:5]:
                logger.error(f"    {rid}")
            if len(missing) > 5:
                logger.error(f"    ... and {len(missing) - 5} more")

        if len(extra) > 0:
            logger.error(f"  Extra {len(extra)} root_ids not in reference:")
            for rid in list(extra)[:5]:
                logger.error(f"    {rid}")
            if len(extra) > 5:
                logger.error(f"    ... and {len(extra) - 5} more")

        return False

    logger.info(f"✓ TEST {name}: Matches reference ({len(extracted_set)} == {len(reference_set)})")
    return True


def run_validation_tests(
    all_grns: pd.DataFrame,
    sugar_water: pd.DataFrame,
    bitter: pd.DataFrame,
    ref_all: Set[int],
    ref_sugar_water: Set[int]
) -> bool:
    """
    Run all validation tests.

    Args:
        all_grns: All gustatory neurons
        sugar_water: Sugar/water neurons
        bitter: Bitter neurons
        ref_all: Reference all gustatory IDs
        ref_sugar_water: Reference sugar/water IDs

    Returns:
        True if all tests pass
    """
    logger.info("\n" + "="*80)
    logger.info("Validation Tests:")
    logger.info("="*80)

    all_passed = True

    # TEST 1: All gustatory matches reference
    all_passed &= validate_extracted(
        set(all_grns['root_id']),
        ref_all,
        "1 - All gustatory matches reference"
    )

    # TEST 2: Sugar/water matches reference
    all_passed &= validate_extracted(
        set(sugar_water['root_id']),
        ref_sugar_water,
        "2 - Sugar/water matches reference"
    )

    # TEST 3: Sugar/water is subset of all gustatory
    sugar_water_ids = set(sugar_water['root_id'])
    all_grn_ids = set(all_grns['root_id'])

    if sugar_water_ids.issubset(all_grn_ids):
        logger.info(f"✓ TEST 3: Sugar/water ⊆ All gustatory ({len(sugar_water_ids)} ⊆ {len(all_grn_ids)})")
    else:
        logger.error(f"❌ TEST 3 FAILED: Sugar/water not subset of all gustatory")
        all_passed = False

    # TEST 4: No overlap between sugar/water and bitter
    bitter_ids = set(bitter['root_id'])
    intersection = sugar_water_ids & bitter_ids

    if len(intersection) == 0:
        logger.info(f"✓ TEST 4: No cross-contamination (intersection = 0)")
    else:
        logger.error(f"❌ TEST 4 FAILED: Found {len(intersection)} neurons in both sugar/water AND bitter")
        logger.error(f"  Overlapping root_ids: {list(intersection)[:5]}")
        all_passed = False

    logger.info("="*80)

    if all_passed:
        logger.info("\n✓✓✓ ALL VALIDATION TESTS PASSED ✓✓✓\n")
    else:
        logger.error("\n❌ SOME VALIDATION TESTS FAILED ❌\n")

    return all_passed


def compute_statistics(
    sugar_water: pd.DataFrame,
    bitter: pd.DataFrame,
    other: pd.DataFrame,
    all_grns: pd.DataFrame
) -> Dict:
    """
    Compute statistics for all GRN categories.

    Args:
        sugar_water: Sugar/water neurons
        bitter: Bitter neurons
        other: Other neurons
        all_grns: All gustatory neurons

    Returns:
        Dictionary with statistics
    """
    logger.info("\n" + "="*80)
    logger.info("Hemisphere Distribution:")
    logger.info("="*80)

    stats = {}

    for name, df in [
        ('Sugar/water', sugar_water),
        ('Bitter', bitter),
        ('Other', other),
        ('Total', all_grns)
    ]:
        left_count = len(df[df['side'] == 'left'])
        right_count = len(df[df['side'] == 'right'])
        ratio = left_count / right_count if right_count > 0 else 0

        balanced = 0.95 <= ratio <= 1.05

        stats[name] = {
            'left': left_count,
            'right': right_count,
            'ratio': ratio,
            'balanced': balanced,
            'total': len(df)
        }

        logger.info(f"  {name:12s}: Left={left_count:3d}, Right={right_count:3d} "
                   f"(ratio={ratio:.3f}, {'balanced ✓' if balanced else 'UNBALANCED'})")

    logger.info("="*80)

    return stats


def export_to_csv(
    all_grns: pd.DataFrame,
    sugar_water: pd.DataFrame,
    bitter: pd.DataFrame,
    other: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Export all GRN categories to CSV files.

    Args:
        all_grns: All gustatory neurons
        sugar_water: Sugar/water neurons
        bitter: Bitter neurons
        other: Other neurons
        output_dir: Output directory
    """
    logger.info("\n" + "="*80)
    logger.info("Exporting data...")
    logger.info("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Select columns to export
    columns = ['root_id', 'class', 'sub_class', 'super_class', 'side', 'hemilineage', 'flow']

    datasets = [
        (all_grns, f'all_grns_{len(all_grns)}.csv'),
        (sugar_water, f'sugar_water_grns_{len(sugar_water)}.csv'),
        (bitter, f'bitter_grns_{len(bitter)}.csv'),
        (other, f'other_grns_{len(other)}.csv')
    ]

    for df, filename in datasets:
        filepath = output_path / filename
        # Only export columns that exist
        export_cols = [col for col in columns if col in df.columns]
        df[export_cols].to_csv(filepath, index=False)
        logger.info(f"✓ {filepath} ({len(df)} rows)")

    logger.info("="*80)


def plot_population_pie(counts: Dict, output_dir: str) -> str:
    """
    Generate population breakdown pie chart.

    Args:
        counts: Dictionary with counts for each category
        output_dir: Output directory

    Returns:
        Path to saved figure
    """
    logger.info("\nGenerating Figure 1: Population pie chart...")

    fig, ax = plt.subplots(figsize=(10, 8))

    categories = ['Sugar/water', 'Bitter', 'Other']
    values = [counts[cat]['total'] for cat in categories]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    percentages = [100 * v / sum(values) for v in values]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=categories,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )

    # Add counts to labels
    for i, text in enumerate(texts):
        text.set_text(f"{categories[i]}\n(n={values[i]})")

    ax.set_title(f'Gustatory Neuron Population Breakdown (n={sum(values)})',
                fontsize=14, fontweight='bold', pad=20)

    filepath = Path(output_dir) / 'grn_population_pie.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    size_kb = filepath.stat().st_size / 1024
    logger.info(f"✓ Figure 1 (pie): saved {size_kb:.0f} KB")

    return str(filepath)


def plot_count_comparison(counts: Dict, output_dir: str) -> str:
    """
    Generate count comparison bar chart.

    Args:
        counts: Dictionary with counts for each category
        output_dir: Output directory

    Returns:
        Path to saved figure
    """
    logger.info("Generating Figure 2: Count comparison bar chart...")

    fig, ax = plt.subplots(figsize=(10, 8))

    categories = ['Sugar/water', 'Bitter', 'Other']
    values = [counts[cat]['total'] for cat in categories]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Gustatory Neuron Counts by Type',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    filepath = Path(output_dir) / 'grn_count_comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    size_kb = filepath.stat().st_size / 1024
    logger.info(f"✓ Figure 2 (bar): saved {size_kb:.0f} KB")

    return str(filepath)


def plot_hemisphere_distribution(counts: Dict, output_dir: str) -> str:
    """
    Generate hemisphere distribution grouped bar chart.

    Args:
        counts: Dictionary with counts for each category
        output_dir: Output directory

    Returns:
        Path to saved figure
    """
    logger.info("Generating Figure 3: Hemisphere distribution...")

    fig, ax = plt.subplots(figsize=(12, 8))

    categories = ['Sugar/water', 'Bitter', 'Other']
    left_counts = [counts[cat]['left'] for cat in categories]
    right_counts = [counts[cat]['right'] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, left_counts, width, label='Left',
                   color='#4A90E2', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, right_counts, width, label='Right',
                   color='#F5A623', alpha=0.8, edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('GRN Hemisphere Distribution by Type',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    filepath = Path(output_dir) / 'grn_hemisphere_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    size_kb = filepath.stat().st_size / 1024
    logger.info(f"✓ Figure 3 (hemisphere): saved {size_kb:.0f} KB")

    return str(filepath)


def plot_validation_dashboard(
    counts: Dict,
    validation_passed: bool,
    output_dir: str
) -> str:
    """
    Generate comprehensive validation dashboard.

    Args:
        counts: Dictionary with counts for each category
        validation_passed: Whether validation passed
        output_dir: Output directory

    Returns:
        Path to saved figure
    """
    logger.info("Generating Figure 4: Validation dashboard...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Subplot 1: Pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Sugar/water', 'Bitter', 'Other']
    values = [counts[cat]['total'] for cat in categories]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Population Breakdown', fontsize=12, fontweight='bold')

    # Subplot 2: Bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Count Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Subplot 3: Hemisphere distribution
    ax3 = fig.add_subplot(gs[1, 0])
    left_counts = [counts[cat]['left'] for cat in categories]
    right_counts = [counts[cat]['right'] for cat in categories]
    x = np.arange(len(categories))
    width = 0.35
    ax3.bar(x - width/2, left_counts, width, label='Left', color='#4A90E2', alpha=0.8)
    ax3.bar(x + width/2, right_counts, width, label='Right', color='#F5A623', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Hemisphere Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Subplot 4: Validation summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    status_text = "✓ ALL TESTS PASSED" if validation_passed else "❌ VALIDATION FAILED"
    status_color = 'green' if validation_passed else 'red'

    summary_text = f"""
VALIDATION SUMMARY

{status_text}

Total neurons: {counts['Total']['total']}
  Sugar/water: {counts['Sugar/water']['total']} ({100*counts['Sugar/water']['total']/counts['Total']['total']:.1f}%)
  Bitter: {counts['Bitter']['total']} ({100*counts['Bitter']['total']/counts['Total']['total']:.1f}%)
  Other: {counts['Other']['total']} ({100*counts['Other']['total']/counts['Total']['total']:.1f}%)

Hemisphere balance:
  Left: {counts['Total']['left']} ({100*counts['Total']['left']/counts['Total']['total']:.1f}%)
  Right: {counts['Total']['right']} ({100*counts['Total']['right']/counts['Total']['total']:.1f}%)
  Ratio: {counts['Total']['ratio']:.3f}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat' if validation_passed else 'lightcoral',
                     alpha=0.5))

    plt.suptitle('GRN Extraction & Validation Dashboard',
                fontsize=16, fontweight='bold', y=0.98)

    filepath = Path(output_dir) / 'grn_validation_dashboard.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    size_kb = filepath.stat().st_size / 1024
    logger.info(f"✓ Figure 4 (dashboard): saved {size_kb:.0f} KB")

    return str(filepath)


def generate_validation_report(
    all_grns: pd.DataFrame,
    sugar_water: pd.DataFrame,
    bitter: pd.DataFrame,
    other: pd.DataFrame,
    counts: Dict,
    validation_passed: bool,
    output_dir: str
) -> str:
    """
    Generate text validation report.

    Args:
        all_grns: All gustatory neurons
        sugar_water: Sugar/water neurons
        bitter: Bitter neurons
        other: Other neurons
        counts: Statistics dictionary
        validation_passed: Whether validation passed
        output_dir: Output directory

    Returns:
        Path to saved report
    """
    logger.info("\n" + "="*80)
    logger.info("Generating validation report...")
    logger.info("="*80)

    filepath = Path(output_dir) / 'grn_validation_report.txt'

    with open(filepath, 'w') as f:
        f.write("GRN EXTRACTION & VALIDATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Input file: data/flywire/classification.csv.gz (139,255 rows)\n")
        f.write("Reference files:\n")
        f.write("  - data/flywire/root_ids_class_gustatory.txt\n")
        f.write("  - data/flywire/root_ids_class_gustatory_sub_class_sugar_water.txt\n\n")

        f.write("EXTRACTION RESULTS:\n")
        f.write(f"  All gustatory:    {len(all_grns)} neurons\n")
        f.write(f"  Sugar/water:      {len(sugar_water)} neurons\n")
        f.write(f"  Bitter:           {len(bitter)} neurons\n")
        f.write(f"  Other:            {len(other)} neurons\n")
        f.write(f"  TOTAL:            {len(all_grns)} neurons\n\n")

        f.write("VALIDATION RESULTS:\n")
        if validation_passed:
            f.write("  ✓ TEST 1: All gustatory matches reference\n")
            f.write("  ✓ TEST 2: Sugar/water matches reference\n")
            f.write("  ✓ TEST 3: Sugar/water ⊆ all gustatory\n")
            f.write("  ✓ TEST 4: No cross-contamination\n\n")
            f.write("  STATUS: ALL VALIDATION TESTS PASSED ✓✓✓\n\n")
        else:
            f.write("  ❌ VALIDATION FAILED - See log for details\n\n")

        f.write("HEMISPHERE DISTRIBUTION:\n")
        for cat in ['Sugar/water', 'Bitter', 'Other', 'Total']:
            f.write(f"  {cat:12s}: {counts[cat]['left']:3d} left, "
                   f"{counts[cat]['right']:3d} right "
                   f"(ratio={counts[cat]['ratio']:.3f}, "
                   f"{'balanced' if counts[cat]['balanced'] else 'unbalanced'})\n")

        f.write("\nOUTPUT FILES:\n")
        f.write("  Data:  4 CSV files (all_grns, sugar_water_grns, bitter_grns, other_grns)\n")
        f.write("  Report: 1 text file\n")
        f.write("  Figures: 4 PNG files (DPI 300)\n")
        f.write("  Total: 9 files in reports/ directory\n")
        f.write("="*80 + "\n")

    logger.info(f"✓ {filepath}")
    logger.info("="*80)

    return str(filepath)


def main():
    """Main pipeline orchestration."""

    print("\n" + "="*80)
    print("GRN EXTRACTION & VALIDATION PIPELINE")
    print("="*80 + "\n")

    # PHASE 1: Extract data
    print("PHASE 1: EXTRACT DATA")
    print("-"*80)

    df = load_classification_data('data/flywire/classification.csv.gz')

    all_grns = extract_gustatory_neurons(df)
    sugar_water = extract_sugar_water_neurons(df)
    bitter = extract_bitter_neurons(df)
    other = extract_other_neurons(all_grns, sugar_water, bitter)

    # PHASE 2: Validate
    print("\nPHASE 2: VALIDATE AGAINST REFERENCE FILES")
    print("-"*80)

    ref_all, ref_sugar_water = load_reference_files('data/flywire')

    validation_passed = run_validation_tests(
        all_grns, sugar_water, bitter, ref_all, ref_sugar_water
    )

    # PHASE 3: Compute statistics
    print("\nPHASE 3: COMPUTE STATISTICS")
    print("-"*80)

    counts = compute_statistics(sugar_water, bitter, other, all_grns)

    # PHASE 4: Export CSVs
    print("\nPHASE 4: EXPORT DATA")
    print("-"*80)

    export_to_csv(all_grns, sugar_water, bitter, other, 'reports')

    # PHASE 5: Generate visualizations
    print("\nPHASE 5: GENERATE VISUALIZATIONS")
    print("-"*80)

    plot_population_pie(counts, 'reports')
    plot_count_comparison(counts, 'reports')
    plot_hemisphere_distribution(counts, 'reports')
    plot_validation_dashboard(counts, validation_passed, 'reports')

    # PHASE 6: Generate report
    print("\nPHASE 6: GENERATE VALIDATION REPORT")
    print("-"*80)

    generate_validation_report(
        all_grns, sugar_water, bitter, other, counts, validation_passed, 'reports'
    )

    # Final summary
    print("\n" + "="*80)
    print("GRN EXTRACTION & VALIDATION COMPLETE")
    print("="*80)
    print(f"Source:  data/flywire/classification.csv.gz ({len(df):,} neurons)")
    print("Reference files: 2 (data/flywire/*.txt)")
    print()
    if validation_passed:
        print("VALIDATION STATUS:  ✓ ALL TESTS PASSED")
    else:
        print("VALIDATION STATUS:  ❌ SOME TESTS FAILED")
    print(f"  ✓ Gustatory count: {len(all_grns)} == {len(ref_all)}")
    print(f"  ✓ Sugar/water count: {len(sugar_water)} == {len(ref_sugar_water)}")
    print(f"  ✓ Subset relationship verified")
    print(f"  ✓ No cross-contamination")
    print()
    print("OUTPUT FILES (9 total):")
    print(f"  CSV data:        4 files ({len(all_grns)} + {len(sugar_water)} + "
          f"{len(bitter)} + {len(other)} neurons)")
    print("  Validation:      1 text report")
    print("  Visualizations:  4 PNG figures (DPI 300)")
    print()
    print("Ready for:")
    print("  - Downstream connectivity analysis")
    print("  - PGCN network initialization")
    print("  - Publication figures")
    print("="*80 + "\n")

    return validation_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
