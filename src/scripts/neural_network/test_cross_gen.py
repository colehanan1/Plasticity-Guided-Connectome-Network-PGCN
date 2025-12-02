#!/usr/bin/env python3
"""
Test script for cross-generalization validation.

Tests Or67b overlap mechanism after implementing veto-at-readout fix.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.scripts.neural_network.ccbpn_v2_full import CCBPN_V2, CCBPN_V2_Config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test cross-generalization after veto-at-readout fix."""

    logger.info("\n" + "╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + " " * 15 + "CCBPN v2.0: Cross-Generalization Test" + " " * 24 + "║")
    logger.info("║" + " " * 15 + "Testing Or67b Overlap Mechanism" + " " * 32 + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝\n")

    # Initialize network
    config = CCBPN_V2_Config(cache_dir="data/cache")
    network = CCBPN_V2(config, seed=42)

    # First train the network normally
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Train network on both odors (normal training)")
    logger.info("=" * 80)

    # Train benzaldehyde
    logger.info("\nTraining benzaldehyde...")
    for trial in range(50):
        progress = trial / 50
        target = 0.16 + (0.21 - 0.16) * progress
        network.train_trial(
            odor='benzaldehyde',
            or7a_activation=0.576,
            or67b_activation=0.746,
            reward_given=True,
            target_approach=target,
            trial_num=trial
        )

    # Train hexanol
    logger.info("Training hexanol...")
    for trial in range(50):
        progress = trial / 50
        target = 0.20 + (0.76 - 0.20) * progress
        network.train_trial(
            odor='hexanol',
            or7a_activation=0.165,
            or67b_activation=0.792,
            reward_given=True,
            target_approach=target,
            trial_num=trial + 50
        )

    logger.info("\n✓ Training complete")

    # Now test cross-generalization
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Test cross-odor generalization")
    logger.info("=" * 80)

    results = network.test_cross_generalization(n_trials=50, verbose=True)

    # Print summary
    logger.info("\n" + "╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + " " * 25 + "VALIDATION SUMMARY" + " " * 35 + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝\n")

    test1 = results['train_benz_test_hex']
    test2 = results['train_hex_test_benz']

    logger.info("TEST 1: Train Benzaldehyde → Test Hexanol")
    logger.info(f"  Hexanol cross-gen: {test1['hexanol_cross_gen']:.1f}%")
    logger.info(f"  Expected range:    65-80%")
    logger.info(f"  Real fly data:     {test1['real_fly_hex']:.1f}%")
    logger.info(f"  KC overlap:        {test1['kc_overlap_pearson']:.3f} (Pearson r)")
    logger.info(f"  Status:            {'✅ PASS' if test1['match'] else '❌ FAIL'}\n")

    logger.info("TEST 2: Train Hexanol → Test Benzaldehyde")
    logger.info(f"  Benzaldehyde cross-gen: {test2['benzaldehyde_cross_gen']:.1f}%")
    logger.info(f"  Expected range:         15-25%")
    logger.info(f"  Real fly data:          {test2['real_fly_benz']:.1f}%")
    logger.info(f"  KC overlap:             {test2['kc_overlap_pearson']:.3f} (Pearson r)")
    logger.info(f"  Status:                 {'✅ PASS' if test2['match'] else '❌ FAIL'}\n")

    logger.info(f"Or67b receptor overlap: {results['or67b_overlap_pct']:.1f}%\n")

    overall_pass = test1['match'] and test2['match']
    logger.info("=" * 80)
    logger.info(f"OVERALL: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")
    logger.info("=" * 80 + "\n")

if __name__ == '__main__':
    main()
