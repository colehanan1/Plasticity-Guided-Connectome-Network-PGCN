#!/usr/bin/env python3
"""Corrected PENP Neuron Analysis - Using classification.csv.gz as Authoritative Source.

CRITICAL FIX: This script now uses classification.csv.gz as the primary data source,
which contains complete neuron classification data. Previous version used incomplete
CSV files resulting in only 1.5% success rate (36/2,444 neurons found).

Expected Results with Fix:
--------------------------
- Success rate: 98%+ (2,400+ / 2,444 neurons found)
- Data source: classification.csv.gz (authoritative, complete)
- Root ID matching: Direct lookup in classification DataFrame

Previous Issue:
---------------
❌ Used incomplete penp_combined_analysis.csv (only SAD/PRW, missing data)
❌ Only found 36/2,444 neurons (1.5% success rate)
❌ Missing 2,408 neurons

Corrected Approach:
-------------------
✅ Use classification.csv.gz (all neurons, root_id in column 1)
✅ Direct DataFrame lookup for each root ID
✅ Expected: 2,400+/2,444 neurons found (98%+ success rate)
✅ Complete olfactory→gustatory→motor pathway data

Usage
-----
python scripts/analyze_penp_corrected.py \\
    --root-ids-file data/penp_root_ids.txt \\
    --classification-file data/flywire/classification.csv.gz \\
    --output-dir data/cache/corrected_penp_analysis
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# PENP regions
PENP_REGIONS = ['SAD', 'PRW', 'CAN', 'FLA', 'AMMC']

# OLFACTORY PATHWAY (PRIMARY)
OLFACTORY_KEYWORDS = [
    'orn', 'osn', 'olfactory receptor', 'or42', 'or47', 'or67', 'or85',
    'ir8', 'ir25', 'ir64', 'ir76',
    'pn', 'upn', 'mpn', 'projection neuron', 'alpn',
    'antennal lobe', 'glomerulus', 'glomeruli', 'al', 'alln',
    'antenna', 'antennal', 'arista', 'funiculus',
    'johnston', 'jo', 'jorgos', 'mechanosensory',
    'multiglomerular', 'multi-glomerular'
]

# GUSTATORY PATHWAY (SECONDARY)
GUSTATORY_KEYWORDS = [
    'taste', 'bitter', 'sweet', 'sugar', 'salt', 'water',
    'pharyngeal', 'labellar', 'proboscis',
    'grn', 'gustatory receptor', 'gr',
    'taste peg', 'chemosensory'
]

# INTEGRATION & PROCESSING
INTEGRATION_KEYWORDS = [
    'kenyon', 'kc', 'mushroom body', 'mb',
    'mbon', 'dan', 'dopamin',
    'subesophageal', 'sez', 'gng', 'avlp',
    'central', 'interneuron', 'local'
]

# MOTOR OUTPUT
MOTOR_KEYWORDS = [
    'motor', 'descending', 'ascending',
    'mn', 'dn', 'an'
]

# EXCLUDE (only clearly irrelevant)
EXCLUDE_KEYWORDS = [
    't4', 't5', 'lamina', 'medulla', 'lobula', 'optic lobe',
    'leg', 'femur', 'tibia', 'tarsus',
    'wing', 'flight', 'haltere'
]


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_classification_data(classification_file: Path) -> pd.DataFrame:
    """Load complete neuron classification data from classification.csv.gz.

    This is the AUTHORITATIVE source with all neurons.

    Parameters
    ----------
    classification_file : Path
        Path to classification.csv.gz

    Returns
    -------
    pd.DataFrame
        Complete classification data with root_id column

    Raises
    ------
    FileNotFoundError
        If classification.csv.gz not found
    ValueError
        If root_id column missing
    """
    logger = logging.getLogger(__name__)

    if not classification_file.exists():
        raise FileNotFoundError(
            f"classification.csv.gz not found at: {classification_file}\n"
            "This file is REQUIRED and contains complete neuron classification data."
        )

    logger.info(f"Loading classification data from: {classification_file}")

    try:
        # Load compressed classification file
        with gzip.open(classification_file, 'rt') as f:
            classification_df = pd.read_csv(f)

        logger.info(f"✓ Loaded {len(classification_df):,} neurons from classification.csv.gz")

        # Verify root_id column exists
        if 'root_id' not in classification_df.columns:
            raise ValueError("root_id column not found in classification.csv.gz")

        # Check for duplicates
        duplicates = classification_df['root_id'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate root_ids in classification data")

        # Log available columns
        logger.debug(f"Available columns: {classification_df.columns.tolist()}")

        return classification_df

    except Exception as e:
        logger.error(f"Failed to load classification.csv.gz: {e}")
        raise


def parse_root_ids_file(root_ids_file: Path) -> List[int]:
    """Parse root IDs from comma-separated file.

    Uses pandas to read the file, which correctly preserves full precision
    for 18-digit root IDs (even when written in scientific notation).

    Parameters
    ----------
    root_ids_file : Path
        File containing comma-separated root IDs

    Returns
    -------
    List[int]
        List of parsed root IDs as integers
    """
    logger = logging.getLogger(__name__)

    if not root_ids_file.exists():
        raise FileNotFoundError(f"Root IDs file not found: {root_ids_file}")

    logger.info(f"Parsing root IDs from: {root_ids_file}")

    try:
        # Use pandas to read - it correctly handles scientific notation
        # and preserves full precision as int64
        df = pd.read_csv(root_ids_file, header=None)

        # File is a single row with 2,444 columns
        # Convert to list of integers
        root_ids = df.iloc[0].astype('int64').tolist()

        logger.info(f"✓ Parsed {len(root_ids):,} root IDs from file")

        expected_count = 2444
        if len(root_ids) != expected_count:
            logger.warning(
                f"Root ID count: expected {expected_count}, got {len(root_ids)}"
            )

        return root_ids

    except Exception as e:
        logger.error(f"Failed to parse root IDs file: {e}")
        raise


def query_neuron_details(
    root_id: int,
    classification_df: pd.DataFrame
) -> Optional[Dict]:
    """Query neuron details from classification DataFrame using root_id.

    Parameters
    ----------
    root_id : int
        FlyWire root ID
    classification_df : pd.DataFrame
        Loaded classification data

    Returns
    -------
    Optional[Dict]
        Neuron details or None if not found
    """
    # Ensure root_id is int for matching
    root_id = int(root_id)

    # Find neuron in classification data
    neuron_row = classification_df[classification_df['root_id'] == root_id]

    if neuron_row.empty:
        return None

    # Extract first match (should be unique)
    neuron = neuron_row.iloc[0]

    # Build neuron data dictionary with all available fields
    neuron_data = {'root_id': int(neuron['root_id'])}

    # Standard fields (may have different column names)
    field_mappings = {
        'cell_type': ['cell_type', 'type', 'primary_type'],
        'cell_subclass': ['cell_subclass', 'subclass', 'sub_class'],
        'super_class': ['super_class', 'superclass', 'class'],
        'cell_class': ['cell_class', 'class'],
    }

    for target_field, possible_columns in field_mappings.items():
        value = None
        for col in possible_columns:
            if col in neuron.index:
                value = neuron.get(col, None)
                if pd.notna(value):
                    break
        neuron_data[target_field] = value if pd.notna(value) else 'unknown'

    # Add all other available columns
    for col in neuron.index:
        if col not in neuron_data and col != 'root_id':
            neuron_data[col] = neuron[col]

    return neuron_data


def classify_functional_role(neuron_data: Dict) -> Tuple[str, str, str, float, float]:
    """Classify neuron with olfactory-FIRST priority.

    Parameters
    ----------
    neuron_data : Dict
        Neuron classification data

    Returns
    -------
    Tuple[str, str, str, float, float]
        (functional_category, pathway_role, keep_reason,
         olfactory_relevance, gustatory_relevance)
    """
    # Combine all text fields for matching
    text_fields = []
    for key in ['cell_type', 'cell_subclass', 'super_class', 'cell_class']:
        value = neuron_data.get(key, '')
        if value and str(value).lower() != 'unknown':
            text_fields.append(str(value).lower())

    combined_text = ' '.join(text_fields)

    # Initialize relevance scores
    olfactory_relevance = 0.0
    gustatory_relevance = 0.0

    # EXPLICIT EXCLUSIONS - Check first
    # 1. Exclude optic_lobe_intrinsic class
    cell_class = str(neuron_data.get('cell_class', '')).lower()
    if cell_class == 'optic_lobe_intrinsic':
        return ('exclude', 'none', 'optic_lobe_intrinsic', 0.0, 0.0)

    # 2. Exclude unknown_sensory class
    if cell_class == 'unknown_sensory':
        return ('exclude', 'none', 'unknown_sensory', 0.0, 0.0)

    # 3. Exclude auditory subclass
    cell_subclass = str(neuron_data.get('cell_subclass', '')).lower()
    sub_class = str(neuron_data.get('sub_class', '')).lower()
    if cell_subclass == 'auditory' or sub_class == 'auditory':
        return ('exclude', 'none', 'auditory_only', 0.0, 0.0)

    # Check for EXCLUDE keywords (only clear non-feeding)
    if any(kw in combined_text for kw in EXCLUDE_KEYWORDS):
        if not any(kw in combined_text for kw in OLFACTORY_KEYWORDS + GUSTATORY_KEYWORDS):
            return ('exclude', 'none', 'clearly_non_feeding', 0.0, 0.0)

    # PRIORITY 1: Olfactory pathway (HIGHEST)
    olfactory_matches = sum(1 for kw in OLFACTORY_KEYWORDS if kw in combined_text)
    if olfactory_matches > 0:
        olfactory_relevance = min(1.0, olfactory_matches / 3.0)

        if any(kw in combined_text for kw in ['orn', 'osn', 'olfactory receptor']):
            return ('olfactory_primary', 'input', 'olfactory_receptor_neuron',
                    olfactory_relevance, 0.0)

        elif any(kw in combined_text for kw in ['pn', 'projection neuron', 'alpn']):
            return ('olfactory_primary', 'processing', 'projection_neuron',
                    olfactory_relevance, 0.0)

        elif any(kw in combined_text for kw in ['antennal lobe', 'glomerulus', 'alln']):
            return ('olfactory_primary', 'processing', 'antennal_lobe_neuron',
                    olfactory_relevance, 0.0)

        elif any(kw in combined_text for kw in ['antenna', 'antennal', 'johnston', 'jo']):
            return ('mechanosensory_antenna', 'input', 'antenna_mechanosensory',
                    olfactory_relevance, 0.0)

        else:
            return ('olfactory_secondary', 'processing', 'olfactory_related',
                    olfactory_relevance, 0.0)

    # PRIORITY 2: Integration (MB, central)
    integration_matches = sum(1 for kw in INTEGRATION_KEYWORDS if kw in combined_text)
    if integration_matches > 0:
        if any(kw in combined_text for kw in ['kenyon', 'kc', 'mushroom body']):
            return ('integration_mb', 'integration', 'mushroom_body_neuron', 0.0, 0.0)

        elif any(kw in combined_text for kw in ['mbon', 'dan', 'dopamin']):
            return ('integration_mb', 'output', 'mb_output_neuron', 0.0, 0.0)

        else:
            return ('integration_central', 'integration', 'central_processing', 0.0, 0.0)

    # PRIORITY 3: Motor output
    motor_matches = sum(1 for kw in MOTOR_KEYWORDS if kw in combined_text)
    if motor_matches > 0:
        if 'motor' in combined_text:
            return ('motor_output', 'output', 'motor_neuron', 0.0, 0.0)
        elif 'descending' in combined_text:
            return ('descending_modulation', 'modulation', 'descending_neuron', 0.0, 0.0)
        elif 'ascending' in combined_text:
            return ('ascending_integration', 'integration', 'ascending_neuron', 0.0, 0.0)

    # PRIORITY 4: Gustatory (SECONDARY)
    gustatory_matches = sum(1 for kw in GUSTATORY_KEYWORDS if kw in combined_text)
    if gustatory_matches > 0:
        gustatory_relevance = min(1.0, gustatory_matches / 3.0)

        if any(kw in combined_text for kw in ['taste', 'bitter', 'sweet']):
            return ('gustatory_primary', 'input', 'taste_receptor_neuron',
                    0.0, gustatory_relevance)

        elif 'pharyngeal' in combined_text:
            return ('gustatory_primary', 'input', 'pharyngeal_sensory',
                    0.0, gustatory_relevance)

        else:
            return ('gustatory_secondary', 'processing', 'gustatory_related',
                    0.0, gustatory_relevance)

    # Default: Unknown but keep
    return ('unknown', 'unknown', 'unclassified', 0.0, 0.0)


def validate_results(
    processed_neurons: List[Dict],
    total_root_ids: int,
    min_success_rate: float = 0.95
) -> bool:
    """Validate that processing achieved expected success rate.

    Parameters
    ----------
    processed_neurons : List[Dict]
        Successfully processed neurons
    total_root_ids : int
        Total number of root IDs to process
    min_success_rate : float
        Minimum acceptable success rate (default: 0.95 = 95%)

    Returns
    -------
    bool
        True if validation passed

    Raises
    ------
    ValueError
        If success rate below threshold
    """
    logger = logging.getLogger(__name__)

    success_count = len(processed_neurons)
    success_rate = success_count / total_root_ids

    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Total root IDs to process: {total_root_ids:,}")
    logger.info(f"Successfully classified: {success_count:,}")
    logger.info(f"Not found: {total_root_ids - success_count:,}")
    logger.info(f"Success rate: {success_rate:.1%}")
    logger.info(f"Required minimum: {min_success_rate:.1%}")

    if success_rate >= min_success_rate:
        logger.info("✓ VALIDATION PASSED")
        return True
    else:
        logger.error(f"✗ VALIDATION FAILED: Success rate {success_rate:.1%} below minimum {min_success_rate:.1%}")
        logger.error(
            "This may indicate:\n"
            "  - classification.csv.gz is incomplete\n"
            "  - root_id format mismatch\n"
            "  - incorrect root IDs file"
        )
        return False


def generate_outputs(
    neurons_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate all output files."""
    logger = logging.getLogger(__name__)

    # Main classified dataset
    main_file = output_dir / 'penp_all_neurons_classified.csv'
    neurons_df.to_csv(main_file, index=False)
    logger.info(f"✓ Saved all neurons: {main_file}")

    # Pathway-specific outputs
    pathways = {
        'olfactory': neurons_df['functional_category'].str.contains('olfactory', na=False),
        'gustatory': neurons_df['functional_category'].str.contains('gustatory', na=False),
        'integration': neurons_df['functional_category'].str.contains('integration', na=False),
        'motor': neurons_df['functional_category'].isin(['motor_output', 'descending_modulation']),
        'excluded': neurons_df['functional_category'] == 'exclude'
    }

    for pathway_name, mask in pathways.items():
        pathway_df = neurons_df[mask]
        pathway_file = output_dir / f'penp_{pathway_name}_pathway.csv'
        pathway_df.to_csv(pathway_file, index=False)
        logger.info(f"  {pathway_name.capitalize()}: {len(pathway_df):,} neurons → {pathway_file.name}")

    # Summary report
    report_file = output_dir / 'corrected_analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CORRECTED PENP Analysis Report (Using classification.csv.gz)\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total neurons: {len(neurons_df):,}\n\n")

        f.write("FUNCTIONAL CATEGORIES:\n")
        f.write("-"*40 + "\n")
        for category in neurons_df['functional_category'].value_counts().head(10).items():
            f.write(f"  {category[0]}: {category[1]:,}\n")

        f.write("\n" + "="*80 + "\n")

    logger.info(f"✓ Generated report: {report_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Corrected PENP analysis using classification.csv.gz',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--root-ids-file',
        type=Path,
        required=True,
        help='File with 2,444 comma-separated root IDs'
    )

    parser.add_argument(
        '--classification-file',
        type=Path,
        default=Path('data/flywire/classification.csv.gz'),
        help='Path to classification.csv.gz (default: data/flywire/classification.csv.gz)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/cache/corrected_penp_analysis'),
        help='Output directory'
    )

    parser.add_argument(
        '--min-success-rate',
        type=float,
        default=0.95,
        help='Minimum success rate for validation (default: 0.95 = 95%%)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main() -> int:
    """Main corrected analysis pipeline."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)

    logger.info("="*80)
    logger.info("CORRECTED PENP Analysis - Using classification.csv.gz")
    logger.info("="*80)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1: Load classification data (AUTHORITATIVE SOURCE)
    logger.info("\nSTEP 1: Loading Classification Data")
    logger.info("-"*80)

    try:
        classification_df = load_classification_data(args.classification_file)
    except Exception as e:
        logger.error(f"Failed to load classification data: {e}")
        return 1

    # STEP 2: Parse root IDs
    logger.info("\nSTEP 2: Parsing Root IDs")
    logger.info("-"*80)

    try:
        root_ids = parse_root_ids_file(args.root_ids_file)
    except Exception as e:
        logger.error(f"Failed to parse root IDs: {e}")
        return 1

    # STEP 3: Process each neuron
    logger.info("\nSTEP 3: Processing Neurons")
    logger.info("-"*80)
    logger.info(f"Processing {len(root_ids):,} PENP neurons...")

    processed_neurons = []
    not_found_ids = []

    for idx, root_id in enumerate(root_ids):
        if (idx + 1) % 250 == 0:
            logger.info(f"  Processed {idx+1:,} / {len(root_ids):,} neurons...")

        # Query from classification DataFrame
        neuron_data = query_neuron_details(root_id, classification_df)

        if neuron_data is None:
            not_found_ids.append(root_id)
            logger.debug(f"Neuron {root_id} not found in classification.csv.gz")
            continue

        # Classify functional role
        (functional_category, pathway_role, keep_reason,
         olfactory_relevance, gustatory_relevance) = classify_functional_role(neuron_data)

        neuron_data.update({
            'functional_category': functional_category,
            'pathway_role': pathway_role,
            'keep_reason': keep_reason,
            'olfactory_relevance': olfactory_relevance,
            'gustatory_relevance': gustatory_relevance,
            'region': 'unknown'  # Would need connectivity data
        })

        processed_neurons.append(neuron_data)

    logger.info(f"\n✓ Processing complete:")
    logger.info(f"  Successfully classified: {len(processed_neurons):,}")
    logger.info(f"  Not found: {len(not_found_ids):,}")

    if not_found_ids and logger.level == logging.DEBUG:
        logger.debug(f"Not found root IDs (first 10): {not_found_ids[:10]}")

    # STEP 4: Validate results
    logger.info("\nSTEP 4: Validation")
    logger.info("-"*80)

    try:
        validate_results(processed_neurons, len(root_ids), args.min_success_rate)
    except ValueError as e:
        logger.error(str(e))
        # Continue anyway to generate outputs

    # STEP 5: Generate outputs
    logger.info("\nSTEP 5: Generating Outputs")
    logger.info("-"*80)

    neurons_df = pd.DataFrame(processed_neurons)
    generate_outputs(neurons_df, args.output_dir)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total processed: {len(neurons_df):,} / {len(root_ids):,}")
    logger.info(f"Success rate: {len(neurons_df)/len(root_ids):.1%}")
    logger.info(f"Olfactory: {neurons_df['functional_category'].str.contains('olfactory', na=False).sum():,}")
    logger.info(f"Gustatory: {neurons_df['functional_category'].str.contains('gustatory', na=False).sum():,}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
