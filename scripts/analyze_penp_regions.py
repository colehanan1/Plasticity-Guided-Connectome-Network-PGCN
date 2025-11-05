#!/usr/bin/env python3
"""Periesophageal Neuropil (PENP) Region Analysis for PGCN.

This script performs comprehensive analysis of neurons in all PENP regions
(SAD, PRW, CAN, FLA, AMMC) from FlyWire connectome data, generating detailed
CSV reports for model selection and circuit analysis.

PENP regions are critical for sensory processing in Drosophila:
- SAD (Saddle): Mechanosensory and gustatory processing
- PRW (Periesophageal Region Wing): Motor control integration
- CAN (Canal): Mechanosensory pathways
- FLA (Flange): Proprioceptive and mechanosensory
- AMMC (Antennal Mechanosensory and Motor Center): Mechanosensory hub

Scientific References:
- Dorkenwald et al. (2024). Neuronal wiring diagram of Drosophila.
- Hampel et al. (2015). Drosophila br

ain: AMMC and mechanosensory pathways.
- Matsuo et al. (2016). Organization of projection neurons in Drosophila.

Usage
-----
Basic analysis:
    python scripts/analyze_penp_regions.py

Full analysis with custom parameters:
    python scripts/analyze_penp_regions.py \\
        --dataset-dir data/flywire \\
        --output-dir data/cache/penp_analysis \\
        --min-synapses 5 \\
        --filter-sensory-types \\
        --generate-summary

Example
-------
>>> python scripts/analyze_penp_regions.py --min-synapses 10 --filter-sensory-types
Processing PENP regions: SAD, PRW, CAN, FLA, AMMC
Found 2,456 neurons across all PENP regions
Generating individual region reports...
Analysis complete: data/cache/penp_analysis/
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loaders.flywire_local import FlyWireLocalDataLoader
from data_loaders.neuron_classification import (
    get_pn_neurons,
    extract_neurotransmitter_info,
)


# PENP region definitions
PENP_REGIONS = ['SAD', 'PRW', 'CAN', 'FLA', 'AMMC']

# Sensory modality keywords for filtering
OLFACTORY_KEYWORDS = [
    'olfactory', 'orn', 'osn', 'or42', 'or47', 'ir8', 'ir25',
    'antennal', 'glomerul', 'alpn', 'projection neuron'
]

GUSTATORY_KEYWORDS = [
    'gustatory', 'grn', 'sugar', 'bitter', 'gr', 'taste',
    'pharyngeal', 'labellar', 'proboscis'
]

MECHANOSENSORY_KEYWORDS = [
    'mechanosensory', 'johnston', 'chordotonal', 'jorgos',
    'ammc', 'bristle', 'propriocept', 'stretch'
]

# Visual/auditory keywords to exclude
EXCLUDE_KEYWORDS = [
    'visual', 'optic', 'lamina', 'medulla', 'lobula',
    'retinal', 'photoreceptor', 'auditory'
]


@dataclass
class PENPNeuron:
    """Data structure for PENP region neuron."""
    root_id: int
    cell_type: str
    cell_subclass: str
    super_class: str
    region: str
    neurotransmitter: str
    hemisphere: str
    input_synapses: int
    output_synapses: int
    total_synaptic_weight: float
    sensory_modality: Optional[str] = None


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for PENP analysis.

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


def load_penp_neurons(
    loader: FlyWireLocalDataLoader,
    regions: List[str] = PENP_REGIONS,
    min_synapses: int = 5
) -> pd.DataFrame:
    """Query FlyWire for neurons in PENP regions.

    Implements the FlyWire query:
    {{input_neuropils:SAD || input_neuropils:PRW || ...}} &&
    {{output_neuropils:SAD || output_neuropils:PRW || ...}}

    Parameters
    ----------
    loader : FlyWireLocalDataLoader
        Initialized FlyWire data loader
    regions : List[str]
        PENP regions to query (default: all 5)
    min_synapses : int
        Minimum synapse count threshold

    Returns
    -------
    pd.DataFrame
        Neurons with connectivity in PENP regions

    Notes
    -----
    Neurons must have both input AND output synapses in PENP regions
    to be considered intrinsic to the periesophageal network.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading PENP neurons for regions: {', '.join(regions)}")

    # Load connections filtered by PENP neuropils
    connections = loader.load_connections(
        neuropil_filter=regions,
        min_synapses=min_synapses
    )

    logger.info(f"Loaded {len(connections):,} connections in PENP regions")

    # Find neurons with both input and output in PENP
    presynaptic_neurons = set(connections['pre_root_id'].unique())
    postsynaptic_neurons = set(connections['post_root_id'].unique())

    # Neurons intrinsic to PENP (both input and output)
    penp_neurons = presynaptic_neurons & postsynaptic_neurons

    logger.info(f"Found {len(penp_neurons):,} neurons with PENP connectivity")

    # Create dataframe with neuron IDs and their PENP regions
    neuron_regions = defaultdict(set)

    for _, row in connections.iterrows():
        if row['pre_root_id'] in penp_neurons:
            neuron_regions[row['pre_root_id']].add(row['neuropil'])
        if row['post_root_id'] in penp_neurons:
            neuron_regions[row['post_root_id']].add(row['neuropil'])

    penp_df = pd.DataFrame([
        {'root_id': neuron_id, 'penp_regions': ','.join(sorted(regions_set))}
        for neuron_id, regions_set in neuron_regions.items()
    ])

    return penp_df


def classify_sensory_modalities(
    neurons_df: pd.DataFrame,
    cell_types_df: pd.DataFrame,
    filter_sensory: bool = True
) -> pd.DataFrame:
    """Classify neurons by sensory modality (olfactory/gustatory/mechanosensory).

    Parameters
    ----------
    neurons_df : pd.DataFrame
        PENP neurons to classify
    cell_types_df : pd.DataFrame
        Cell type annotations from FlyWire
    filter_sensory : bool
        If True, exclude purely visual/auditory neurons

    Returns
    -------
    pd.DataFrame
        Neurons with sensory_modality classification

    Notes
    -----
    Classification based on cell type keywords:
    - Olfactory: ORN, OSN, olfactory PN, etc.
    - Gustatory: GRN, sugar, bitter, taste neurons
    - Mechanosensory: JO neurons, chordotonal, proprioceptors
    """
    logger = logging.getLogger(__name__)

    # Merge with cell types
    merged = neurons_df.merge(
        cell_types_df[['root_id', 'cell_type']],
        on='root_id',
        how='left'
    )

    merged['sensory_modality'] = pd.NA

    # Classify by keywords
    for idx, row in merged.iterrows():
        cell_type_lower = str(row['cell_type']).lower()

        # Check for exclusions first
        if filter_sensory and any(kw in cell_type_lower for kw in EXCLUDE_KEYWORDS):
            continue

        # Check for olfactory
        if any(kw in cell_type_lower for kw in OLFACTORY_KEYWORDS):
            merged.at[idx, 'sensory_modality'] = 'olfactory'
        # Check for gustatory
        elif any(kw in cell_type_lower for kw in GUSTATORY_KEYWORDS):
            merged.at[idx, 'sensory_modality'] = 'gustatory'
        # Check for mechanosensory
        elif any(kw in cell_type_lower for kw in MECHANOSENSORY_KEYWORDS):
            merged.at[idx, 'sensory_modality'] = 'mechanosensory'
        else:
            merged.at[idx, 'sensory_modality'] = 'other'

    # Filter if requested
    if filter_sensory:
        original_count = len(merged)
        merged = merged[merged['sensory_modality'].notna()]
        logger.info(
            f"Filtered to sensory neurons: {len(merged):,} / {original_count:,} "
            f"({len(merged)/original_count:.1%})"
        )

    return merged


def analyze_region_composition(
    neurons_df: pd.DataFrame,
    region: str,
    classification_df: pd.DataFrame,
    connections_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate detailed cell type statistics for a PENP region.

    Parameters
    ----------
    neurons_df : pd.DataFrame
        Neurons in this region
    region : str
        Region name (SAD, PRW, etc.)
    classification_df : pd.DataFrame
        Cell classification data (super_class, sub_class)
    connections_df : pd.DataFrame
        Connectivity data for synapse counts

    Returns
    -------
    pd.DataFrame
        Region composition statistics

    Notes
    -----
    Output schema:
    - region, cell_type, cell_subclass, super_class
    - count, percentage_of_region
    - neurotransmitter, hemisphere
    - synaptic_weight_total, connectivity_strength
    """
    logger = logging.getLogger(__name__)

    # Filter neurons for this region
    region_neurons = neurons_df[
        neurons_df['penp_regions'].str.contains(region, na=False)
    ].copy()

    logger.info(f"Analyzing {region}: {len(region_neurons):,} neurons")

    # Merge with classification
    region_neurons = region_neurons.merge(
        classification_df[['root_id', 'super_class', 'sub_class']],
        on='root_id',
        how='left'
    )

    # Compute connectivity metrics
    connectivity_metrics = compute_connectivity_metrics(
        region_neurons['root_id'].tolist(),
        connections_df,
        region
    )

    region_neurons = region_neurons.merge(
        connectivity_metrics,
        on='root_id',
        how='left'
    )

    # Group by cell type/subclass
    composition = region_neurons.groupby(
        ['cell_type', 'sub_class', 'super_class'],
        dropna=False
    ).agg({
        'root_id': 'count',
        'synaptic_weight_total': 'sum',
        'connectivity_strength': 'mean'
    }).reset_index()

    composition.columns = [
        'cell_type', 'cell_subclass', 'super_class',
        'count', 'synaptic_weight_total', 'connectivity_strength'
    ]

    # Add region and percentage
    composition['region'] = region
    composition['percentage_of_region'] = (
        100 * composition['count'] / composition['count'].sum()
    )

    # Reorder columns
    composition = composition[[
        'region', 'cell_type', 'cell_subclass', 'super_class',
        'count', 'percentage_of_region',
        'synaptic_weight_total', 'connectivity_strength'
    ]]

    # Sort by count descending
    composition = composition.sort_values('count', ascending=False)

    return composition


def compute_connectivity_metrics(
    neuron_ids: List[int],
    connections_df: pd.DataFrame,
    region: Optional[str] = None
) -> pd.DataFrame:
    """Compute synaptic weight and connectivity strength metrics.

    Parameters
    ----------
    neuron_ids : List[int]
        Neuron root IDs to analyze
    connections_df : pd.DataFrame
        Connection table with syn_count
    region : Optional[str]
        Filter connections to specific neuropil

    Returns
    -------
    pd.DataFrame
        Connectivity metrics per neuron

    Notes
    -----
    Metrics computed:
    - synaptic_weight_total: Sum of all synapse counts
    - connectivity_strength: Average synapses per connection
    - input_synapses: Total incoming synapses
    - output_synapses: Total outgoing synapses
    """
    neuron_set = set(neuron_ids)

    # Filter connections
    if region:
        conn_filtered = connections_df[
            connections_df['neuropil'] == region
        ].copy()
    else:
        conn_filtered = connections_df.copy()

    # Compute input synapses
    input_syn = conn_filtered[
        conn_filtered['post_root_id'].isin(neuron_set)
    ].groupby('post_root_id')['syn_count'].sum().reset_index()
    input_syn.columns = ['root_id', 'input_synapses']

    # Compute output synapses
    output_syn = conn_filtered[
        conn_filtered['pre_root_id'].isin(neuron_set)
    ].groupby('pre_root_id')['syn_count'].sum().reset_index()
    output_syn.columns = ['root_id', 'output_synapses']

    # Compute connectivity strength (avg synapses per connection)
    conn_strength_pre = conn_filtered[
        conn_filtered['pre_root_id'].isin(neuron_set)
    ].groupby('pre_root_id')['syn_count'].mean().reset_index()
    conn_strength_pre.columns = ['root_id', 'connectivity_strength']

    # Merge metrics
    metrics = pd.DataFrame({'root_id': neuron_ids})

    for df in [input_syn, output_syn, conn_strength_pre]:
        metrics = metrics.merge(df, on='root_id', how='left')

    # Fill NaN with 0
    metrics = metrics.fillna(0)

    # Total synaptic weight
    metrics['synaptic_weight_total'] = (
        metrics['input_synapses'] + metrics['output_synapses']
    )

    return metrics


def generate_csv_reports(
    penp_analysis: Dict[str, pd.DataFrame],
    output_dir: Path,
    combined_analysis: pd.DataFrame
) -> None:
    """Generate structured CSV output files.

    Parameters
    ----------
    penp_analysis : Dict[str, pd.DataFrame]
        Analysis results per region
    output_dir : Path
        Output directory
    combined_analysis : pd.DataFrame
        Combined PENP analysis

    Notes
    -----
    Generates:
    - individual_regions/{region}_neurons.csv for each PENP region
    - penp_combined_analysis.csv
    - penp_connectivity_matrix.csv (if connectivity data available)
    """
    logger = logging.getLogger(__name__)

    # Create output directories
    individual_dir = output_dir / 'individual_regions'
    individual_dir.mkdir(parents=True, exist_ok=True)

    # Save individual region reports
    for region, analysis_df in penp_analysis.items():
        output_file = individual_dir / f"{region.lower()}_neurons.csv"
        analysis_df.to_csv(output_file, index=False)
        logger.info(f"Saved {region} analysis: {output_file}")

    # Save combined analysis
    combined_file = output_dir / 'penp_combined_analysis.csv'
    combined_analysis.to_csv(combined_file, index=False)
    logger.info(f"Saved combined analysis: {combined_file}")


def generate_summary_report(
    penp_analysis: Dict[str, pd.DataFrame],
    combined_analysis: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate human-readable summary report.

    Parameters
    ----------
    penp_analysis : Dict[str, pd.DataFrame]
        Analysis results per region
    combined_analysis : pd.DataFrame
        Combined PENP analysis
    output_dir : Path
        Output directory
    """
    summary_file = output_dir / 'penp_summary_report.txt'

    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PENP Region Analysis Summary Report\n")
        f.write("Plasticity-Guided Connectome Network (PGCN) Project\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        total_neurons = combined_analysis['count'].sum()
        f.write(f"Total PENP neurons: {total_neurons:,}\n")
        f.write(f"Regions analyzed: {', '.join(PENP_REGIONS)}\n")
        f.write(f"Unique cell types: {combined_analysis['cell_type'].nunique()}\n\n")

        # Per-region statistics
        f.write("PER-REGION BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for region in PENP_REGIONS:
            if region in penp_analysis:
                region_df = penp_analysis[region]
                f.write(f"\n{region}:\n")
                f.write(f"  Neurons: {region_df['count'].sum():,}\n")
                f.write(f"  Cell types: {region_df['cell_type'].nunique()}\n")
                f.write(f"  Top 3 types:\n")
                for idx, row in region_df.head(3).iterrows():
                    f.write(
                        f"    - {row['cell_type']}: "
                        f"{row['count']} ({row['percentage_of_region']:.1f}%)\n"
                    )

        # Sensory modality distribution
        if 'sensory_modality' in combined_analysis.columns:
            f.write("\nSENSORY MODALITY DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            for modality in ['olfactory', 'gustatory', 'mechanosensory', 'other']:
                count = combined_analysis[
                    combined_analysis.get('sensory_modality') == modality
                ]['count'].sum()
                if count > 0:
                    pct = 100 * count / total_neurons
                    f.write(f"  {modality.capitalize()}: {count:,} ({pct:.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")

    logging.getLogger(__name__).info(f"Generated summary report: {summary_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze PENP region neurons from FlyWire connectome',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('data/flywire'),
        help='FlyWire dataset directory (default: data/flywire)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/cache/penp_analysis'),
        help='Output directory for analysis results (default: data/cache/penp_analysis)'
    )

    parser.add_argument(
        '--min-synapses',
        type=int,
        default=5,
        help='Minimum synapse threshold for connections (default: 5)'
    )

    parser.add_argument(
        '--filter-sensory-types',
        action='store_true',
        help='Filter for olfactory/gustatory/mechanosensory neurons only'
    )

    parser.add_argument(
        '--regions',
        nargs='+',
        choices=PENP_REGIONS,
        default=PENP_REGIONS,
        help='PENP regions to analyze (default: all)'
    )

    parser.add_argument(
        '--generate-summary',
        action='store_true',
        default=True,
        help='Generate human-readable summary report'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main() -> int:
    """Main analysis pipeline."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("PENP Region Analysis - PGCN Project")
    logger.info("=" * 80)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Initialize FlyWire data loader
    logger.info("Initializing FlyWire data loader...")
    try:
        loader = FlyWireLocalDataLoader(dataset_dir=args.dataset_dir)
    except Exception as e:
        logger.error(f"Failed to initialize data loader: {e}")
        return 1

    # Load PENP neurons
    logger.info(f"\nQuerying PENP regions: {', '.join(args.regions)}")
    try:
        penp_neurons = load_penp_neurons(
            loader,
            regions=args.regions,
            min_synapses=args.min_synapses
        )
    except Exception as e:
        logger.error(f"Failed to load PENP neurons: {e}")
        return 1

    if len(penp_neurons) == 0:
        logger.warning("No PENP neurons found with specified criteria")
        return 0

    # Load cell type and classification data
    logger.info("\nLoading cell type annotations...")
    try:
        cell_types = loader.load_cell_types()
        classification = loader.load_classification()
        connections = loader.load_connections(
            neuropil_filter=args.regions,
            min_synapses=args.min_synapses
        )
    except Exception as e:
        logger.error(f"Failed to load annotations: {e}")
        return 1

    # Classify by sensory modality
    if args.filter_sensory_types:
        logger.info("\nClassifying by sensory modality...")
        penp_neurons = classify_sensory_modalities(
            penp_neurons,
            cell_types,
            filter_sensory=True
        )

    # Analyze each region
    logger.info("\nAnalyzing individual PENP regions...")
    penp_analysis = {}

    for region in args.regions:
        try:
            region_analysis = analyze_region_composition(
                penp_neurons,
                region,
                classification,
                connections
            )
            penp_analysis[region] = region_analysis
            logger.info(
                f"  {region}: {region_analysis['count'].sum():,} neurons, "
                f"{len(region_analysis)} cell types"
            )
        except Exception as e:
            logger.error(f"Failed to analyze region {region}: {e}")
            continue

    # Generate combined analysis
    logger.info("\nGenerating combined PENP analysis...")
    if penp_analysis:
        combined_analysis = pd.concat(
            penp_analysis.values(),
            ignore_index=True
        )
    else:
        logger.error("No region analyses completed")
        return 1

    # Generate CSV reports
    logger.info("\nGenerating CSV reports...")
    try:
        generate_csv_reports(
            penp_analysis,
            args.output_dir,
            combined_analysis
        )
    except Exception as e:
        logger.error(f"Failed to generate CSV reports: {e}")
        return 1

    # Generate summary report
    if args.generate_summary:
        logger.info("\nGenerating summary report...")
        try:
            generate_summary_report(
                penp_analysis,
                combined_analysis,
                args.output_dir
            )
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total neurons analyzed: {len(penp_neurons):,}")
    logger.info(f"Regions: {', '.join(args.regions)}")
    logger.info(f"Unique cell types: {combined_analysis['cell_type'].nunique()}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
