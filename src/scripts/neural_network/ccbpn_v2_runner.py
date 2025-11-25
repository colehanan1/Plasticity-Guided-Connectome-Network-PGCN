#!/usr/bin/env python3
"""
CCBPN v2.0 Runner Script

Self-contained CLI for training and evaluating the complete B2 v2.0 model.

Usage:
    python ccbpn_v2_runner.py --pgcn-cache data/cache --n-trials 50 --output results/v2.json

Options:
    --pgcn-cache: Path to PGCN cache directory
    --n-trials: Number of training trials per odor (default: 50)
    --output: Output JSON file for results
    --verbose: Enable detailed logging
    --verify-shapes: Print connectivity shapes and exit
    --compare-to-b1: Compare ablation prediction to B1 model
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.neural_network.ccbpn_v2_full import (
    CCBPN_V2,
    CCBPN_V2_Config,
    train_ccbpn_v2
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def verify_shapes(config: CCBPN_V2_Config):
    """
    Verify connectivity matrix shapes and print summary.

    Args:
        config: Model configuration
    """
    logger.info("=" * 80)
    logger.info("VERIFYING CONNECTIVITY SHAPES")
    logger.info("=" * 80)

    try:
        network = CCBPN_V2(config, seed=42)

        logger.info(f"\n✓ Successfully loaded network")
        logger.info(f"\nConnectivity shapes:")
        logger.info(f"  PN→KC: {network.W_pn_kc.shape}")
        logger.info(f"  KC→MBON: {network.W_kc_mbon_dense.shape}")
        logger.info(f"  DAN→KC: {network.W_dan_kc.shape}")
        logger.info(f"  DAN→MBON: {network.W_dan_mbon.shape}")

        logger.info(f"\nNeuron counts:")
        logger.info(f"  PNs: {len(network.pn_ids)}")
        logger.info(f"  KCs: {len(network.kc_ids)}")
        logger.info(f"  MBONs: {len(network.mbon_ids)}")
        logger.info(f"  DANs: {len(network.dan_ids)}")

        logger.info(f"\nPathway identification:")
        logger.info(f"  Or7a PNs: {len(network.or7a_indices)} indices")
        logger.info(f"  Or67b PNs: {len(network.or67b_indices)} indices")

        logger.info(f"\n✓ All shapes valid, no errors")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"\n✗ Shape verification failed: {e}")
        return False


def run_training(config: CCBPN_V2_Config, n_trials_per_odor: int,
                output_path: Path, compare_to_b1: bool):
    """
    Run complete training protocol.

    Args:
        config: Model configuration
        n_trials_per_odor: Trials per odor
        output_path: Path for output JSON
        compare_to_b1: Whether to compare to B1 model
    """
    logger.info("=" * 80)
    logger.info("RUNNING CCBPN V2.0 TRAINING")
    logger.info("=" * 80)

    # Initialize and train
    network = CCBPN_V2(config, seed=42)
    trial_df = train_ccbpn_v2(network, n_trials_per_odor=n_trials_per_odor)

    # Extract final results
    benz_trials = trial_df[trial_df['odor'] == 'benzaldehyde']
    hex_trials = trial_df[trial_df['odor'] == 'hexanol']

    benz_final = float(benz_trials['approach_pred'].iloc[-1])
    hex_final = float(hex_trials['approach_pred'].iloc[-1])

    # Ablation prediction
    ablation_pred = network.predict_ablation()

    # Prepare output
    results = {
        'model': 'CCBPN v2.0 (Full)',
        'phases_integrated': [
            'Phase 1: FlyWire connectivity',
            'Phase 2: Antennal lobe circuits',
            'Phase 3: MBON opponent coding',
            'Phase 4: RPE dopamine'
        ],
        'connectivity': {
            'n_pn': len(network.pn_ids),
            'n_kc': len(network.kc_ids),
            'n_mbon': len(network.mbon_ids),
            'n_dan': len(network.dan_ids),
            'or7a_pns': len(network.or7a_indices),
            'or67b_pns': len(network.or67b_indices),
            'pn_to_kc_synapses': int(network.W_pn_kc.nnz),
            'kc_to_mbon_synapses': int(network.W_kc_mbon_dense.size),
        },
        'training': {
            'n_trials_per_odor': n_trials_per_odor,
            'benzaldehyde': {
                'initial': float(benz_trials['approach_pred'].iloc[0]),
                'final': benz_final,
                'target': 0.21,
                'error_pct': abs(benz_final - 0.21) / 0.21 * 100
            },
            'hexanol': {
                'initial': float(hex_trials['approach_pred'].iloc[0]),
                'final': hex_final,
                'target': 0.76,
                'error_pct': abs(hex_final - 0.76) / 0.76 * 100
            }
        },
        'ablation': {
            'b2_v2_prediction': float(ablation_pred),
            'b1_prediction': 0.744,
            'difference_pp': float(abs(ablation_pred - 0.744) * 100),
            'converged': bool(abs(ablation_pred - 0.744) < 0.10)
        }
    }

    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n✓ Training complete:")
    logger.info(f"  Benzaldehyde: {benz_final:.2%} (target: 21%, error: {results['training']['benzaldehyde']['error_pct']:.1f}%)")
    logger.info(f"  Hexanol:      {hex_final:.2%} (target: 76%, error: {results['training']['hexanol']['error_pct']:.1f}%)")

    logger.info(f"\n✓ Ablation prediction:")
    logger.info(f"  B2 v2.0: {ablation_pred:.1%}")
    if compare_to_b1:
        logger.info(f"  B1:      74.4%")
        logger.info(f"  Diff:    {results['ablation']['difference_pp']:.1f} pp")
        if results['ablation']['converged']:
            logger.info(f"  Status:  ✓ CONVERGED (within 10pp)")
        else:
            logger.info(f"  Status:  ⚠ Check parameters")

    logger.info(f"\n✓ Saved results: {output_path}")
    logger.info("=" * 80 + "\n")

    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CCBPN v2.0 Runner: Train and evaluate complete model"
    )

    parser.add_argument(
        '--pgcn-cache',
        type=str,
        default='data/cache',
        help='Path to PGCN cache directory (default: data/cache)'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of training trials per odor (default: 50)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/ccbpn_v2/results.json',
        help='Output JSON file for results (default: results/ccbpn_v2/results.json)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable detailed logging (DEBUG level)'
    )

    parser.add_argument(
        '--verify-shapes',
        action='store_true',
        help='Verify connectivity shapes and exit without training'
    )

    parser.add_argument(
        '--compare-to-b1',
        action='store_true',
        help='Compare ablation prediction to B1 model'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build config
    config = CCBPN_V2_Config(cache_dir=args.pgcn_cache)

    # Verify shapes only
    if args.verify_shapes:
        success = verify_shapes(config)
        sys.exit(0 if success else 1)

    # Run training
    output_path = Path(args.output)
    results = run_training(
        config=config,
        n_trials_per_odor=args.n_trials,
        output_path=output_path,
        compare_to_b1=args.compare_to_b1
    )

    logger.info("🎉 CCBPN v2.0 execution complete!")


if __name__ == '__main__':
    main()
