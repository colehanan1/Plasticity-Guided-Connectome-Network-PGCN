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

import numpy as np
from scipy import sparse
from typing import Iterable, List

from scripts.neural_network.ccbpn_v2_full import (
    CCBPN_V2,
    CCBPN_V2_Config,
    train_ccbpn_v2
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def verify_shapes(config: CCBPN_V2_Config, seed: int = 42):
    """
    Verify connectivity matrix shapes and print summary.

    Args:
        config: Model configuration
    """
    logger.info("=" * 80)
    logger.info("VERIFYING CONNECTIVITY SHAPES")
    logger.info("=" * 80)

    try:
        network = CCBPN_V2(config, seed=seed)

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
                output_path: Path, compare_to_b1: bool, seed: int):
    """
    Run complete training protocol.

    Args:
        config: Model configuration
        n_trials_per_odor: Trials per odor
        output_path: Path for output JSON
        compare_to_b1: Whether to compare to B1 model
        seed: Random seed for initialization
    """
    logger.info("=" * 80)
    logger.info("RUNNING CCBPN V2.0 TRAINING")
    logger.info("=" * 80)
    logger.info(f"Using seed: {seed}")

    # Initialize and train
    network = CCBPN_V2(config, seed=seed)
    trial_df = train_ccbpn_v2(network, n_trials_per_odor=n_trials_per_odor)

    # Extract final results
    benz_trials = trial_df[trial_df['odor'] == 'benzaldehyde']
    hex_trials = trial_df[trial_df['odor'] == 'hexanol']

    benz_final = float(benz_trials['approach_pred'].iloc[-1])
    hex_final = float(hex_trials['approach_pred'].iloc[-1])

    # Ablation prediction
    ablation_pred = network.predict_ablation()

    def _count_synapses(matrix) -> int:
        """Return nonzero count for sparse or dense matrices."""
        if sparse.issparse(matrix):
            return int(matrix.nnz)
        return int(np.count_nonzero(matrix))

    # Prepare output
    results = {
        'model': 'CCBPN v2.0 (Full)',
        'seed': seed,
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
            'pn_to_kc_synapses': _count_synapses(network.W_pn_kc),
            'kc_to_mbon_synapses': _count_synapses(
                network.W_kc_mbon if hasattr(network, "W_kc_mbon") else network.W_kc_mbon_dense
            ),
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


def run_seed_sweep(config: CCBPN_V2_Config, n_trials_per_odor: int,
                   output_path: Path, compare_to_b1: bool,
                   seeds: Iterable[int]) -> List[dict]:
    """
    Run multiple seeds in one invocation and summarize metrics.
    """
    all_results = []

    for seed in seeds:
        # Derive per-seed output path to avoid overwrite
        seed_output = output_path.with_name(f"{output_path.stem}_seed{seed}{output_path.suffix}")
        results = run_training(
            config=config,
            n_trials_per_odor=n_trials_per_odor,
            output_path=seed_output,
            compare_to_b1=compare_to_b1,
            seed=seed
        )
        all_results.append(results)

    # Aggregate
    benz = np.array([r['training']['benzaldehyde']['final'] for r in all_results])
    hexn = np.array([r['training']['hexanol']['final'] for r in all_results])
    abla = np.array([r['ablation']['b2_v2_prediction'] for r in all_results])

    def summarize(name, arr):
        logger.info(f"{name}: {arr.mean()*100:.2f} ± {arr.std(ddof=1)*100:.2f}% (n={len(arr)})")

    logger.info("\n" + "=" * 80)
    logger.info("MULTI-SEED SUMMARY")
    logger.info("=" * 80)
    summarize("Benzaldehyde", benz)
    summarize("Hexanol", hexn)
    summarize("Ablation", abla)
    logger.info("=" * 80 + "\n")

    return all_results


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

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for initialization (default: 42)'
    )

    parser.add_argument(
        '--seed-sweep',
        action='store_true',
        help='Run a 10-seed sweep (42-51) and summarize results'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build config
    config = CCBPN_V2_Config(cache_dir=args.pgcn_cache)

    # Verify shapes only
    if args.verify_shapes:
        success = verify_shapes(config, seed=args.seed)
        sys.exit(0 if success else 1)

    # Run training
    output_path = Path(args.output)
    if args.seed_sweep:
        seeds = range(42, 52)  # inclusive 42-51
        results = run_seed_sweep(
            config=config,
            n_trials_per_odor=args.n_trials,
            output_path=output_path,
            compare_to_b1=args.compare_to_b1,
            seeds=seeds
        )
    else:
        results = run_training(
            config=config,
            n_trials_per_odor=args.n_trials,
            output_path=output_path,
            compare_to_b1=args.compare_to_b1,
            seed=args.seed
        )

    logger.info("🎉 CCBPN v2.0 execution complete!")


if __name__ == '__main__':
    main()
