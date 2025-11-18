"""
Predict OR7a Suppression Experimental Outcomes

This script generates quantitative predictions for OR7a suppression experiments:
1. Expected loss of benzaldehyde blocking
2. Expected loss of hexanol cross-transfer
3. Control odor predictions (no effect expected)

Based on:
- FlyWire connectomics (41 OR7a → 2 DL5_adPN bottleneck)
- DoOR receptor responses (benzaldehyde 0.551, hexanol 0.139)
- Pathway architecture analysis

Usage:
    python scripts/predict_or7a_suppression.py --output results/or7a_suppression_predictions/
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
        description="Predict OR7a pathway suppression experimental outcomes"
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
        help='OR7a pathway analysis results'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/or7a_suppression_predictions'),
        help='Output directory'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("OR7a Suppression Predictions")
    logger.info("="*80)

    # Initialize DoOR
    logger.info("\nInitializing DoOR database...")
    try:
        door_manager = DoORDataManager(method='rpy2')
    except:
        door_manager = DoORDataManager(method='csv')

    # Create analyzer
    analyzer = OR7aPathwayAnalyzer(
        flywire_data_dir=args.flywire_data,
        pathway_results_dir=args.pathway_results,
        door_manager=door_manager,
        output_dir=args.output_dir
    )

    # Load data
    logger.info("\nLoading pathway and DoOR data...")
    analyzer.load_pathway_data()
    analyzer.load_door_responses()

    # Generate predictions
    logger.info("\n" + "="*80)
    logger.info("SUPPRESSION EFFECT PREDICTIONS")
    logger.info("="*80)

    predictions = analyzer.predict_suppression_effects()

    # Print predictions
    logger.info("\n" + "-"*80)
    logger.info("PREDICTED OUTCOMES")
    logger.info("-"*80)

    if 'pathway_elimination' in predictions:
        elim = predictions['pathway_elimination']
        logger.info(f"\n**PATHWAY ELIMINATION ANALYSIS**")
        logger.info(f"  OR7a→PN synapses: {elim['or7a_pn_synapses']:,}")
        logger.info(f"  Unique PNs: {elim['unique_pns']}")
        logger.info(f"  Bottleneck severity: {elim['bottleneck_severity']}")

        if elim['unique_pns'] <= 3:
            logger.info(f"\n  → PREDICTION: Complete pathway elimination possible")
            logger.info(f"  → Suppressing {elim['unique_pns']} DL5_adPN neurons should block entire OR7a pathway!")

    if 'blocking_reduction' in predictions:
        blocking = predictions['blocking_reduction']
        logger.info(f"\n**BENZALDEHYDE BLOCKING PREDICTION**")
        logger.info(f"  Mechanism: {blocking['mechanism']}")
        logger.info(f"  Expected change: {blocking['expected_change']}")
        logger.info(f"  Quantitative basis: Or7a response = {blocking['quantitative']:.3f}")
        logger.info(f"\n  → PREDICTION: OR7a suppression eliminates blocking")
        logger.info(f"  → Learning should recover from ~0.2 to ~0.7 (normal levels)")

    if 'transfer_reduction' in predictions:
        transfer = predictions['transfer_reduction']
        logger.info(f"\n**HEXANOL CROSS-TRANSFER PREDICTION**")
        logger.info(f"  Mechanism: {transfer['mechanism']}")
        logger.info(f"  Expected change: {transfer['expected_change']}")
        logger.info(f"  Quantitative basis: Hexanol/Benzaldehyde ratio = {transfer['quantitative']:.1%}")
        logger.info(f"\n  → PREDICTION: OR7a suppression eliminates {transfer['quantitative']:.1%} of cross-transfer")
        logger.info(f"  → Transfer should drop from ~0.5 to ~0.2 (baseline level)")

    if 'experimental_targets' in predictions:
        targets = predictions['experimental_targets']
        logger.info(f"\n**EXPERIMENTAL TARGETING RECOMMENDATIONS**")
        logger.info(f"  Strategy: {targets['targeting_recommendation']}")
        logger.info(f"  Primary targets (DL5_adPN root IDs):")

        for i, (root_id, score) in enumerate(zip(targets['primary_pns'], targets['importance_scores']), 1):
            logger.info(f"    {i}. {root_id} (importance: {score:.0f})")

    # Generate experimental protocol table
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENTAL PREDICTIONS TABLE")
    logger.info("="*80)

    exp_table = []

    # Control condition
    exp_table.append({
        'condition': 'Wildtype Control',
        'genotype': 'w1118',
        'training': 'benzaldehyde',
        'test': 'benzaldehyde',
        'predicted_LI': 0.2,
        'interpretation': 'Strong blocking (low learning)'
    })

    exp_table.append({
        'condition': 'Wildtype Control',
        'genotype': 'w1118',
        'training': 'benzaldehyde',
        'test': '1-hexanol',
        'predicted_LI': 0.5,
        'interpretation': 'Cross-transfer (enhanced)'
    })

    # OR7a suppression
    exp_table.append({
        'condition': 'OR7a Suppressed',
        'genotype': 'Or7a-GAL4 > UAS-Kir2.1',
        'training': 'benzaldehyde',
        'test': 'benzaldehyde',
        'predicted_LI': 0.7,
        'interpretation': 'Blocking eliminated (restored learning)'
    })

    exp_table.append({
        'condition': 'OR7a Suppressed',
        'genotype': 'Or7a-GAL4 > UAS-Kir2.1',
        'training': 'benzaldehyde',
        'test': '1-hexanol',
        'predicted_LI': 0.2,
        'interpretation': 'Cross-transfer eliminated'
    })

    # Control odors
    exp_table.append({
        'condition': 'OR7a Suppressed',
        'genotype': 'Or7a-GAL4 > UAS-Kir2.1',
        'training': 'ethyl butyrate',
        'test': 'ethyl butyrate',
        'predicted_LI': 0.7,
        'interpretation': 'Normal (OR7a not involved)'
    })

    exp_df = pd.DataFrame(exp_table)

    # Print table
    logger.info("\n" + exp_df.to_string(index=False))

    # Save predictions
    output_file = args.output_dir / 'or7a_suppression_predictions.csv'
    exp_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved predictions table to {output_file}")

    # Save complete predictions
    import json
    json_file = args.output_dir / 'or7a_suppression_predictions_complete.json'
    with open(json_file, 'w') as f:
        json.dump(predictions, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    logger.info(f"✓ Saved complete predictions to {json_file}")

    logger.info("\n" + "="*80)
    logger.info("Suppression predictions complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
