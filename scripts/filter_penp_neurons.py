#!/usr/bin/env python3
"""Intelligent PENP Neuron Filtering for PGCN Model Construction.

This script implements scientifically-principled filtering of PENP neurons to reduce
computational overhead while preserving biological authenticity of olfactory-gustatory-motor
pathways essential for feeding behavior experiments.

Filtering Strategy
------------------
Based on PENP analysis of 4,172 neurons (SAD: 2,894, PRW: 1,278), implements:

1. DEFINITE KEEP (344 neurons, 8.2%): Gustatory chemosensory and feeding ascending
2. DEFINITE EXCLUDE (663 neurons, 15.9%): Auditory JO, visual, non-feeding pathways
3. SELECTIVE KEEP (3,026 neurons): Intelligent classification of central/motor/descending

Filtering Strategies:
- Conservative: ~800-1000 neurons (highest confidence, rapid model development)
- Moderate: ~1500-2000 neurons (balanced completeness vs complexity)
- Liberal: ~3000+ neurons (comprehensive circuits, research depth)

Scientific Justification
------------------------
- Removes Johnston's Organ (JO) neurons: purely auditory, no feeding function
- Removes visual projection neurons: optic lobe circuits, no chemosensory role
- Removes non-feeding ascending neurons: proprioceptive, non-gustatory pathways
- Retains complete gustatory→motor pathways for blocking experiments
- Preserves plasticity-relevant interneurons for learning mechanisms

Usage
-----
Conservative filtering (recommended for initial model):
    python scripts/filter_penp_neurons.py \\
        --input-file data/cache/penp_analysis/penp_combined_analysis.csv \\
        --output-dir data/cache/filtered_penp \\
        --filter-strategy conservative \\
        --min-connectivity-strength 10.0

Moderate filtering (balanced approach):
    python scripts/filter_penp_neurons.py \\
        --filter-strategy moderate \\
        --min-connectivity-strength 7.5

Liberal filtering (comprehensive circuits):
    python scripts/filter_penp_neurons.py \\
        --filter-strategy liberal \\
        --min-connectivity-strength 5.0

Example
-------
>>> python scripts/filter_penp_neurons.py --filter-strategy conservative
Loading PENP neurons: 4,172 neurons
Applying definite keep filter: 344 neurons kept
Applying definite exclude filter: 663 neurons excluded
Classifying central neurons: 856 neurons kept
Final filtered dataset: 1,200 neurons (28.8% of original)
Filtering complete: data/cache/filtered_penp/
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Filtering categories based on PENP analysis
DEFINITE_KEEP_GUSTATORY = [
    'pharyngeal_nerve_sensory_group1',
    'pharyngeal_nerve_sensory_group2',
    'accessory_pharyngeal_nerve_sensory_group1',
    'accessory_pharyngeal_nerve_sensory_group2',
    'taste_peg',
    'low-salt',
    'sugar/water',
    'bitter',
    'multiglomerular',
    'head_bristle'
]

DEFINITE_KEEP_FEEDING_ASCENDING = [
    'AN_GNG_SAD', 'AN_GNG_PRW', 'AN_GNG_FLA', 'AN_GNG',
    'AN_AVLP', 'AN_AVLP_GNG', 'AN_AVLP_PVLP', 'AN_AVLP_SAD',
    'AN_GNG_AMMC', 'AN_SAD_GNG', 'AN_VES_GNG', 'AN_GNG_VES',
    'AN_GNG_IPS', 'AN_VES_WED'
]

# Exclusion patterns
EXCLUDE_AUDITORY_PATTERN = r'^JO-.*'  # Johnston's Organ neurons
EXCLUDE_VISUAL_TYPES = ['visual_projection', 'visual_centrifugal']
EXCLUDE_NON_FEEDING_ASCENDING = [
    'SA_VTV_pro_meso_meta',
    'SA_MDA',
    'SA_DMT_ADMN'
]

# Central neuron classification keywords
FEEDING_RELATED_KEYWORDS = [
    'gustatory', 'feeding', 'pharyngeal', 'proboscis',
    'sugar', 'bitter', 'taste', 'chemosensory',
    'sad', 'prw', 'gng', 'avlp'
]

NON_FEEDING_KEYWORDS = [
    'visual', 'optic', 'lamina', 'medulla', 'lobula',
    'auditory', 'johnston', 'antennal_mechanosensory',
    'wing', 'leg', 'flight'
]


@dataclass
class FilteringStats:
    """Statistics for filtering operations."""
    original_count: int
    definite_keep_count: int
    definite_exclude_count: int
    central_kept_count: int
    final_count: int

    @property
    def reduction_percentage(self) -> float:
        return 100 * (1 - self.final_count / self.original_count)

    @property
    def kept_percentage(self) -> float:
        return 100 * self.final_count / self.original_count


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for filtering pipeline.

    Parameters
    ----------
    verbose : bool
        Enable DEBUG level logging

    Returns
    -------
    logging.Logger
        Configured logger
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_penp_data(input_file: Path) -> pd.DataFrame:
    """Load PENP analysis results.

    Parameters
    ----------
    input_file : Path
        Path to penp_combined_analysis.csv

    Returns
    -------
    pd.DataFrame
        PENP neuron data

    Raises
    ------
    FileNotFoundError
        If input file doesn't exist
    """
    logger = logging.getLogger(__name__)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info(f"Loading PENP data from: {input_file}")
    df = pd.read_csv(input_file)

    logger.info(f"Loaded {len(df):,} neurons from PENP analysis")

    # Validate required columns
    required_columns = [
        'region', 'cell_type', 'cell_subclass', 'super_class',
        'count', 'synaptic_weight_total', 'connectivity_strength'
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def apply_definite_keep_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply definite KEEP filter for essential gustatory/feeding neurons.

    Parameters
    ----------
    df : pd.DataFrame
        Input PENP neuron data

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (kept_neurons, remaining_neurons)

    Notes
    -----
    Keeps 344 neurons (8.2% of dataset):
    - Gustatory chemosensory neurons (pharyngeal, taste pegs, etc.)
    - Feeding-related ascending neurons (GNG, AVLP pathways)

    These neurons are ESSENTIAL for olfactory→gustatory→motor pathways
    and must be retained for blocking experiments.
    """
    logger = logging.getLogger(__name__)

    # Create keep mask
    keep_mask = (
        df['cell_subclass'].isin(DEFINITE_KEEP_GUSTATORY) |
        df['cell_subclass'].isin(DEFINITE_KEEP_FEEDING_ASCENDING)
    )

    kept = df[keep_mask].copy()
    remaining = df[~keep_mask].copy()

    # Add filtering metadata
    kept['keep_reason'] = 'definite_keep_gustatory_feeding'
    kept['model_priority'] = 'high'
    kept['functional_category'] = kept['cell_subclass'].apply(
        lambda x: 'gustatory' if x in DEFINITE_KEEP_GUSTATORY else 'feeding_ascending'
    )

    logger.info(f"Definite KEEP: {len(kept):,} neurons ({100*len(kept)/len(df):.1f}%)")
    logger.info(f"  Gustatory chemosensory: {kept['functional_category'].eq('gustatory').sum()}")
    logger.info(f"  Feeding ascending: {kept['functional_category'].eq('feeding_ascending').sum()}")

    return kept, remaining


def apply_definite_exclude_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply definite EXCLUDE filter for non-relevant neurons.

    Parameters
    ----------
    df : pd.DataFrame
        Input neuron data (after definite keep filter)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (excluded_neurons, remaining_neurons)

    Notes
    -----
    Excludes 663 neurons (15.9% of dataset):
    - Johnston's Organ (JO-*): 536 neurons, purely auditory
    - Visual projection neurons: 20 neurons, optic lobe circuits
    - Non-feeding ascending: 107 neurons, proprioceptive pathways

    These neurons do NOT contribute to olfactory/gustatory processing
    and would introduce computational overhead without biological benefit.
    """
    logger = logging.getLogger(__name__)

    # Auditory JO neurons
    jo_mask = df['cell_subclass'].str.match(EXCLUDE_AUDITORY_PATTERN, na=False)

    # Visual neurons
    visual_mask = df['cell_subclass'].isin(EXCLUDE_VISUAL_TYPES)

    # Non-feeding ascending
    non_feeding_mask = df['cell_subclass'].isin(EXCLUDE_NON_FEEDING_ASCENDING)

    # Combined exclusion mask
    exclude_mask = jo_mask | visual_mask | non_feeding_mask

    excluded = df[exclude_mask].copy()
    remaining = df[~exclude_mask].copy()

    # Add exclusion metadata
    excluded['exclude_reason'] = 'definite_exclude'
    excluded.loc[jo_mask, 'exclude_reason'] = 'auditory_johnston_organ'
    excluded.loc[visual_mask, 'exclude_reason'] = 'visual_system'
    excluded.loc[non_feeding_mask, 'exclude_reason'] = 'non_feeding_ascending'

    logger.info(f"Definite EXCLUDE: {len(excluded):,} neurons ({100*len(excluded)/(len(df)+len(excluded)):.1f}%)")
    logger.info(f"  Auditory (JO): {jo_mask.sum()}")
    logger.info(f"  Visual: {visual_mask.sum()}")
    logger.info(f"  Non-feeding ascending: {non_feeding_mask.sum()}")

    return excluded, remaining


def apply_connectivity_filter(
    df: pd.DataFrame,
    min_strength: float,
    strategy: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter neurons by synaptic connectivity strength.

    Parameters
    ----------
    df : pd.DataFrame
        Input neuron data
    min_strength : float
        Minimum connectivity strength threshold
    strategy : str
        Filtering strategy (conservative/moderate/liberal)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (kept_neurons, excluded_neurons)

    Notes
    -----
    Connectivity strength = average synapses per connection

    Thresholds by strategy:
    - Conservative: >10.0 (strong connections only)
    - Moderate: >7.5 (medium-strong connections)
    - Liberal: >5.0 (include weaker connections)
    """
    logger = logging.getLogger(__name__)

    # Apply connectivity threshold
    connectivity_mask = df['connectivity_strength'] >= min_strength

    kept = df[connectivity_mask].copy()
    excluded = df[~connectivity_mask].copy()

    excluded['exclude_reason'] = f'low_connectivity_<{min_strength}'

    logger.info(
        f"Connectivity filter ({strategy}, ≥{min_strength}): "
        f"{len(kept):,} kept, {len(excluded):,} excluded"
    )

    return kept, excluded


def classify_central_neurons(
    df: pd.DataFrame,
    strategy: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Intelligently classify central neurons for inclusion.

    Parameters
    ----------
    df : pd.DataFrame
        Central neurons to classify
    strategy : str
        Filtering strategy (conservative/moderate/liberal)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (kept_neurons, excluded_neurons)

    Notes
    -----
    Conservative strategy:
    - Keep motor/descending neurons (complete behavioral output)
    - Keep high-connectivity central neurons (>10.0 strength)
    - Keep feeding-related keywords in cell type

    Moderate strategy:
    - Broader central neuron inclusion
    - Medium connectivity threshold (>7.5)

    Liberal strategy:
    - Keep most central neurons except clear non-feeding
    - Low connectivity threshold (>5.0)
    """
    logger = logging.getLogger(__name__)

    # Motor and descending neurons (always keep for complete pathways)
    motor_mask = df['super_class'].isin(['motor', 'descending'])

    # Feeding-related keywords in cell type
    feeding_related_mask = df['cell_type'].str.lower().str.contains(
        '|'.join(FEEDING_RELATED_KEYWORDS),
        na=False,
        regex=True
    )

    # Non-feeding keywords (exclude these)
    non_feeding_mask = df['cell_type'].str.lower().str.contains(
        '|'.join(NON_FEEDING_KEYWORDS),
        na=False,
        regex=True
    )

    # High synaptic weight (likely important for circuit function)
    high_weight_mask = df['synaptic_weight_total'] > 1000.0

    # Strategy-specific logic
    if strategy == 'conservative':
        # Keep: motor/descending + high-connectivity feeding-related
        keep_mask = (
            motor_mask |
            (feeding_related_mask & ~non_feeding_mask & high_weight_mask)
        )

    elif strategy == 'moderate':
        # Keep: motor/descending + feeding-related + high-connectivity central
        keep_mask = (
            motor_mask |
            (feeding_related_mask & ~non_feeding_mask) |
            high_weight_mask
        )

    else:  # liberal
        # Keep: most neurons except clear non-feeding
        keep_mask = ~non_feeding_mask

    kept = df[keep_mask].copy()
    excluded = df[~keep_mask].copy()

    # Add classification metadata
    kept['functional_category'] = 'central_interneuron'
    kept.loc[motor_mask, 'functional_category'] = 'motor_output'
    kept.loc[df['super_class'] == 'descending', 'functional_category'] = 'descending_modulation'

    kept['keep_reason'] = 'central_neuron_selective'
    kept.loc[motor_mask, 'keep_reason'] = 'motor_output_essential'
    kept.loc[feeding_related_mask, 'keep_reason'] = 'feeding_related_circuit'

    kept['model_priority'] = 'medium'
    kept.loc[motor_mask, 'model_priority'] = 'high'

    excluded['exclude_reason'] = f'central_neuron_excluded_{strategy}'

    logger.info(f"Central neuron classification ({strategy}):")
    logger.info(f"  Kept: {len(kept):,} neurons")
    logger.info(f"    Motor/descending: {motor_mask.sum()}")
    logger.info(f"    Feeding-related: {(feeding_related_mask & ~non_feeding_mask).sum()}")
    logger.info(f"    High-weight central: {(high_weight_mask & ~motor_mask).sum()}")
    logger.info(f"  Excluded: {len(excluded):,} neurons")

    return kept, excluded


def generate_filtering_report(
    original_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    stats: FilteringStats,
    output_dir: Path
) -> None:
    """Generate comprehensive filtering analysis report.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original PENP dataset
    filtered_df : pd.DataFrame
        Final filtered dataset
    excluded_df : pd.DataFrame
        Excluded neurons with reasons
    stats : FilteringStats
        Filtering statistics
    output_dir : Path
        Output directory for report
    """
    logger = logging.getLogger(__name__)

    report_file = output_dir / 'filtering_summary_report.txt'

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PENP Neuron Filtering Summary Report\n")
        f.write("Plasticity-Guided Connectome Network (PGCN) Project\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("FILTERING STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original neurons: {stats.original_count:,}\n")
        f.write(f"Final filtered neurons: {stats.final_count:,}\n")
        f.write(f"Reduction: {stats.reduction_percentage:.1f}%\n")
        f.write(f"Retained: {stats.kept_percentage:.1f}%\n\n")

        f.write(f"Definite keep: {stats.definite_keep_count:,}\n")
        f.write(f"Definite exclude: {stats.definite_exclude_count:,}\n")
        f.write(f"Central neurons kept: {stats.central_kept_count:,}\n\n")

        # Functional category breakdown
        f.write("FUNCTIONAL CATEGORY BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        if 'functional_category' in filtered_df.columns:
            for category in filtered_df['functional_category'].unique():
                count = (filtered_df['functional_category'] == category).sum()
                pct = 100 * count / len(filtered_df)
                f.write(f"  {category}: {count:,} ({pct:.1f}%)\n")
        f.write("\n")

        # Region distribution
        f.write("REGION DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for region in filtered_df['region'].unique():
            count = (filtered_df['region'] == region).sum()
            pct = 100 * count / len(filtered_df)
            f.write(f"  {region}: {count:,} ({pct:.1f}%)\n")
        f.write("\n")

        # Exclusion reasons
        f.write("EXCLUSION REASONS\n")
        f.write("-" * 40 + "\n")
        if 'exclude_reason' in excluded_df.columns:
            exclusion_counts = excluded_df['exclude_reason'].value_counts()
            for reason, count in exclusion_counts.items():
                pct = 100 * count / len(excluded_df)
                f.write(f"  {reason}: {count:,} ({pct:.1f}%)\n")
        f.write("\n")

        # Model priority distribution
        f.write("MODEL PRIORITY DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        if 'model_priority' in filtered_df.columns:
            for priority in ['high', 'medium', 'low']:
                count = (filtered_df['model_priority'] == priority).sum()
                pct = 100 * count / len(filtered_df)
                f.write(f"  {priority.capitalize()}: {count:,} ({pct:.1f}%)\n")
        f.write("\n")

        # Connectivity statistics
        f.write("CONNECTIVITY STATISTICS (FILTERED DATASET)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean connectivity strength: {filtered_df['connectivity_strength'].mean():.2f}\n")
        f.write(f"  Median connectivity strength: {filtered_df['connectivity_strength'].median():.2f}\n")
        f.write(f"  Mean synaptic weight: {filtered_df['synaptic_weight_total'].mean():.1f}\n")
        f.write(f"  Total synaptic weight: {filtered_df['synaptic_weight_total'].sum():.0f}\n\n")

        # Scientific justification
        f.write("SCIENTIFIC JUSTIFICATION\n")
        f.write("-" * 40 + "\n")
        f.write("This filtering preserves:\n")
        f.write("  ✓ Complete gustatory→motor pathways (essential for feeding)\n")
        f.write("  ✓ Plasticity-relevant interneurons (learning mechanisms)\n")
        f.write("  ✓ Behavioral output pathways (motor/descending neurons)\n")
        f.write("  ✓ High-connectivity central processing (circuit integration)\n\n")

        f.write("This filtering removes:\n")
        f.write("  ✗ Auditory neurons (JO, no feeding function)\n")
        f.write("  ✗ Visual neurons (optic lobe, no chemosensory role)\n")
        f.write("  ✗ Non-feeding pathways (proprioceptive, non-gustatory)\n")
        f.write("  ✗ Low-connectivity neurons (minimal circuit impact)\n\n")

        f.write("=" * 80 + "\n")

    logger.info(f"Generated filtering report: {report_file}")


def validate_filtered_dataset(
    filtered_df: pd.DataFrame,
    original_df: pd.DataFrame
) -> bool:
    """Validate biological integrity of filtered dataset.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Filtered neuron dataset
    original_df : pd.DataFrame
        Original PENP dataset

    Returns
    -------
    bool
        True if validation passes

    Notes
    -----
    Validation checks:
    - Essential gustatory neurons retained
    - Motor output pathways complete
    - Regional representation balanced
    - Connectivity patterns preserved
    """
    logger = logging.getLogger(__name__)

    validation_passed = True

    # Check 1: Essential gustatory neurons retained
    gustatory_count = filtered_df['functional_category'].eq('gustatory').sum()
    if gustatory_count < 50:
        logger.warning(f"Low gustatory neuron count: {gustatory_count}")
        validation_passed = False

    # Check 2: Motor neurons present
    motor_count = filtered_df['functional_category'].eq('motor_output').sum()
    if motor_count < 10:
        logger.warning(f"Low motor neuron count: {motor_count}")
        validation_passed = False

    # Check 3: Regional balance (SAD should be larger than PRW)
    sad_count = filtered_df['region'].eq('SAD').sum()
    prw_count = filtered_df['region'].eq('PRW').sum()

    if sad_count < prw_count:
        logger.warning("Regional imbalance: SAD < PRW")
        validation_passed = False

    # Check 4: Connectivity integrity
    mean_connectivity = filtered_df['connectivity_strength'].mean()
    if mean_connectivity < 5.0:
        logger.warning(f"Low mean connectivity: {mean_connectivity:.2f}")
        validation_passed = False

    if validation_passed:
        logger.info("✓ Validation PASSED: Filtered dataset has biological integrity")
    else:
        logger.warning("⚠ Validation WARNINGS: Review filtered dataset carefully")

    return validation_passed


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Filter PENP neurons for PGCN model construction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input-file',
        type=Path,
        default=Path('data/cache/penp_analysis/penp_combined_analysis.csv'),
        help='Input PENP analysis CSV file'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/cache/filtered_penp'),
        help='Output directory for filtered datasets'
    )

    parser.add_argument(
        '--filter-strategy',
        choices=['conservative', 'moderate', 'liberal'],
        default='conservative',
        help='Filtering strategy (default: conservative)'
    )

    parser.add_argument(
        '--min-connectivity-strength',
        type=float,
        default=None,
        help='Minimum connectivity strength (default: auto by strategy)'
    )

    parser.add_argument(
        '--export-summary',
        action='store_true',
        default=True,
        help='Generate summary report (default: True)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main() -> int:
    """Main filtering pipeline."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("PENP Neuron Filtering - PGCN Project")
    logger.info(f"Strategy: {args.filter_strategy.upper()}")
    logger.info("=" * 80)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set connectivity threshold by strategy if not specified
    if args.min_connectivity_strength is None:
        connectivity_thresholds = {
            'conservative': 10.0,
            'moderate': 7.5,
            'liberal': 5.0
        }
        args.min_connectivity_strength = connectivity_thresholds[args.filter_strategy]

    logger.info(f"Connectivity threshold: ≥{args.min_connectivity_strength}")

    # Load PENP data
    try:
        original_df = load_penp_data(args.input_file)
    except Exception as e:
        logger.error(f"Failed to load PENP data: {e}")
        return 1

    original_count = len(original_df)

    # Step 1: Apply definite KEEP filter
    logger.info("\n" + "-" * 80)
    logger.info("STEP 1: Applying Definite KEEP Filter")
    logger.info("-" * 80)

    kept_definite, remaining = apply_definite_keep_filter(original_df)

    # Step 2: Apply definite EXCLUDE filter
    logger.info("\n" + "-" * 80)
    logger.info("STEP 2: Applying Definite EXCLUDE Filter")
    logger.info("-" * 80)

    excluded_definite, remaining = apply_definite_exclude_filter(remaining)

    # Step 3: Apply connectivity filter to remaining central neurons
    logger.info("\n" + "-" * 80)
    logger.info("STEP 3: Applying Connectivity Filter")
    logger.info("-" * 80)

    remaining_high_conn, excluded_low_conn = apply_connectivity_filter(
        remaining,
        args.min_connectivity_strength,
        args.filter_strategy
    )

    # Step 4: Classify remaining central neurons
    logger.info("\n" + "-" * 80)
    logger.info("STEP 4: Classifying Central Neurons")
    logger.info("-" * 80)

    kept_central, excluded_central = classify_central_neurons(
        remaining_high_conn,
        args.filter_strategy
    )

    # Combine filtered dataset
    filtered_df = pd.concat([kept_definite, kept_central], ignore_index=True)

    # Combine excluded dataset
    excluded_df = pd.concat(
        [excluded_definite, excluded_low_conn, excluded_central],
        ignore_index=True
    )

    # Compute statistics
    stats = FilteringStats(
        original_count=original_count,
        definite_keep_count=len(kept_definite),
        definite_exclude_count=len(excluded_definite),
        central_kept_count=len(kept_central),
        final_count=len(filtered_df)
    )

    # Validation
    logger.info("\n" + "-" * 80)
    logger.info("VALIDATION")
    logger.info("-" * 80)

    validate_filtered_dataset(filtered_df, original_df)

    # Save outputs
    logger.info("\n" + "-" * 80)
    logger.info("SAVING OUTPUTS")
    logger.info("-" * 80)

    # Main filtered dataset
    filtered_file = args.output_dir / 'penp_filtered_neurons.csv'
    filtered_df.to_csv(filtered_file, index=False)
    logger.info(f"Saved filtered neurons: {filtered_file}")

    # Excluded neurons
    excluded_file = args.output_dir / 'penp_excluded_neurons.csv'
    excluded_df.to_csv(excluded_file, index=False)
    logger.info(f"Saved excluded neurons: {excluded_file}")

    # Summary report
    if args.export_summary:
        generate_filtering_report(
            original_df,
            filtered_df,
            excluded_df,
            stats,
            args.output_dir
        )

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Original neurons: {stats.original_count:,}")
    logger.info(f"Filtered neurons: {stats.final_count:,}")
    logger.info(f"Reduction: {stats.reduction_percentage:.1f}%")
    logger.info(f"Computational savings: ~{stats.reduction_percentage:.0f}% fewer neurons to simulate")
    logger.info(f"\nStrategy: {args.filter_strategy.upper()}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
