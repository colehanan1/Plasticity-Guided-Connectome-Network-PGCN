"""
Analyze OR7a Pathway Baseline Behavior: Blocking & Cross-Transfer

This script analyzes baseline behavioral data to validate the OR7a pathway
hypothesis:
1. Benzaldehyde training → Benzaldehyde test = BLOCKED (learned suppression)
2. Benzaldehyde training → Hexanol test = ENHANCED (cross-transfer)

Usage:
    # With behavioral data
    python scripts/analyze_or7a_baseline.py --behavioral-data data/behavioral/baseline.csv

    # Without behavioral data (pathway + DoOR analysis only)
    python scripts/analyze_or7a_baseline.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pgcn.analysis.or7a_pathway import OR7aPathwayAnalyzer
from src.pgcn.door.door_data_manager import DoORDataManager
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze OR7a pathway baseline behavioral patterns"
    )

    parser.add_argument(
        '--flywire-data',
        type=Path,
        default=Path('data/flywire'),
        help='FlyWire data directory'
    )

    parser.add_argument(
        '--pathway-results',
        type=Path,
        default=Path('results/or7a_complete_pathway'),
        help='OR7a pathway analysis results directory'
    )

    parser.add_argument(
        '--behavioral-data',
        type=Path,
        default=None,
        help='Behavioral data CSV (optional)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/or7a_baseline_analysis'),
        help='Output directory'
    )

    parser.add_argument(
        '--odor-panel',
        nargs='+',
        default=['benzaldehyde', '1-hexanol', 'ethyl butyrate', 'linalool', '3-octanol'],
        help='Odor panel to analyze'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("OR7a Baseline Behavior Analysis")
    logger.info("="*80)

    # Initialize DoOR manager
    logger.info("\nInitializing DoOR database access...")
    try:
        door_manager = DoORDataManager(method='rpy2')
        logger.info("✓ DoOR manager initialized")
    except Exception as e:
        logger.warning(f"Could not initialize DoOR with rpy2: {e}")
        logger.info("Attempting CSV fallback...")
        door_manager = DoORDataManager(method='csv')

    # Create analyzer
    analyzer = OR7aPathwayAnalyzer(
        flywire_data_dir=args.flywire_data,
        pathway_results_dir=args.pathway_results,
        door_manager=door_manager,
        output_dir=args.output_dir
    )

    # Load OR7a data
    logger.info("\nLoading OR7a neuron data...")
    or7a_neurons = analyzer.load_or7a_data()

    # Load pathway data
    logger.info("\nLoading pathway analysis results...")
    analyzer.load_pathway_data()

    # Load DoOR responses for odor panel
    logger.info(f"\nLoading DoOR responses for {len(args.odor_panel)} odors...")
    door_responses = analyzer.load_door_responses(odor_panel=args.odor_panel)

    # Save DoOR responses
    door_output = args.output_dir / 'or7a_door_responses.csv'
    door_responses.to_csv(door_output, index=False)
    logger.info(f"✓ Saved DoOR responses to {door_output}")

    # Load behavioral data if provided
    behavioral_data = None
    if args.behavioral_data and args.behavioral_data.exists():
        logger.info(f"\nLoading behavioral data from {args.behavioral_data}")
        behavioral_data = pd.read_csv(args.behavioral_data)
        logger.info(f"✓ Loaded {len(behavioral_data)} behavioral observations")
    else:
        logger.info("\nNo behavioral data provided - analyzing pathway + DoOR only")

    # Analyze baseline
    logger.info("\n" + "="*80)
    logger.info("BASELINE ANALYSIS")
    logger.info("="*80)

    baseline = analyzer.analyze_baseline_behavior(behavioral_data)

    # Print key findings
    logger.info("\n" + "-"*80)
    logger.info("KEY FINDINGS")
    logger.info("-"*80)

    if 'pathway_architecture' in baseline:
        arch = baseline['pathway_architecture']

        if 'critical_bottleneck' in arch:
            bn = arch['critical_bottleneck']
            logger.info(f"\n**CRITICAL BOTTLENECK IDENTIFIED**")
            logger.info(f"  Transition: {bn['transition']}")
            logger.info(f"  {bn['source_neurons']} → {bn['target_neurons']} neurons")
            logger.info(f"  Expansion ratio: {bn['expansion_ratio']:.4f}")
            logger.info(f"  Severity: {bn['severity']}")

        if 'or7a_pn_strength' in arch:
            pn = arch['or7a_pn_strength']
            logger.info(f"\n**OR7a→PN CONNECTION STRENGTH**")
            logger.info(f"  Total synapses: {pn['total_synapses']:,}")
            logger.info(f"  Mean synapses/connection: {pn['mean_synapses']:.1f}")
            logger.info(f"  Unique PNs: {pn['unique_pns']}")
            logger.info(f"  → Just {pn['unique_pns']} DL5_adPN neurons control entire OR7a pathway!")

    if 'door_predictions' in baseline:
        door = baseline['door_predictions']

        logger.info(f"\n**DoOR RECEPTOR PREDICTIONS**")

        if 'benzaldehyde_response' in door:
            logger.info(f"  Benzaldehyde Or7a response: {door['benzaldehyde_response']:.3f}")

        if 'hexanol_response' in door:
            logger.info(f"  Hexanol Or7a response: {door['hexanol_response']:.3f}")

        if 'predicted_cross_transfer' in door:
            logger.info(f"  Predicted cross-transfer: {door['predicted_cross_transfer']:.1%}")
            logger.info(f"  → Hexanol activates OR7a at {door['predicted_cross_transfer']:.1%} of benzaldehyde level")

        if 'control_responses' in door:
            logger.info(f"\n  Control odor responses:")
            for odor, response in door['control_responses'].items():
                logger.info(f"    {odor}: {response:.3f}")

    if 'behavioral_patterns' in baseline:
        behav = baseline['behavioral_patterns']

        logger.info(f"\n**BEHAVIORAL PATTERNS**")

        if 'blocking_magnitude' in behav:
            logger.info(f"  Blocking magnitude: {behav['blocking_magnitude']:.3f}")

        if 'hexanol_transfer' in behav:
            logger.info(f"  Hexanol cross-transfer: {behav['hexanol_transfer']:.3f}")

        if 'transfer_enhancement' in behav:
            logger.info(f"  Transfer enhancement: {behav['transfer_enhancement']:.3f}")

    if 'door_behavior_correlation' in baseline:
        corr = baseline['door_behavior_correlation']

        logger.info(f"\n**DoOR-BEHAVIOR CORRELATION**")
        logger.info(f"  Pearson r: {corr['pearson_r']:.3f}")
        logger.info(f"  P-value: {corr['p_value']:.3e}")
        logger.info(f"  N odors: {corr['n_odors']}")

    # Save complete analysis
    import json
    output_path = args.output_dir / 'or7a_baseline_complete.json'
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    logger.info(f"\n✓ Saved complete analysis to {output_path}")

    logger.info("\n" + "="*80)
    logger.info("Baseline analysis complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
