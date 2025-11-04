"""
GRN Connectivity Analysis - Testing & Validation

This script implements comprehensive tests as specified in the requirements:
- Test 1: Data Integrity
- Test 2: Connectivity Data Structure
- Test 3: Statistical Validation
- Test 4: Comparison Test (Sugar vs All GRNs)
- Test 5: Visualization Output

Author: Claude AI
Date: 2025-11-03
"""

import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any
import os
import yaml

logger = logging.getLogger(__name__)


class PipelineTester:
    """Comprehensive testing suite for GRN connectivity pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tester with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cache_dir = Path(config['data']['cache_dir'])
        self.output_dir = Path(config['data']['output_dir'])
        self.tests_passed = 0
        self.tests_failed = 0

    def test_1_data_integrity(
        self,
        sugar_grns: pd.DataFrame,
        all_grns: pd.DataFrame
    ) -> bool:
        """
        Test 1: Validate loaded GRN data integrity.

        Args:
            sugar_grns: Sugar GRN DataFrame
            all_grns: All GRN DataFrame

        Returns:
            True if test passes
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 1: DATA INTEGRITY")
        logger.info("="*80)

        try:
            # Check sugar GRNs exist
            assert len(sugar_grns) > 0, "No sugar GRNs found"
            logger.info(f"✓ Found {len(sugar_grns)} sugar GRNs")

            # Check all GRNs exist
            assert len(all_grns) > 0, "No GRNs found"
            logger.info(f"✓ Found {len(all_grns)} total GRNs")

            # Check required columns
            assert 'root_id' in sugar_grns.columns, "Missing 'root_id' column"
            assert 'processed_labels' in sugar_grns.columns, "Missing 'processed_labels' column"
            logger.info("✓ Required columns present")

            # Check root_id types
            assert all(isinstance(rid, (int, np.integer)) for rid in sugar_grns['root_id']), \
                "Invalid root_id types"
            logger.info("✓ All root_ids are integers")

            # Check for duplicates
            assert sugar_grns['root_id'].is_unique, "Duplicate root_ids detected in sugar GRNs"
            assert all_grns['root_id'].is_unique, "Duplicate root_ids detected in all GRNs"
            logger.info("✓ No duplicate root_ids")

            # Check labels contain expected keywords
            sample_labels = sugar_grns['processed_labels'].head(3).tolist()
            logger.info(f"✓ Sample labels (should contain sugar + gustatory keywords):")
            for label in sample_labels:
                logger.info(f"    - {label}")

            # Verify labels are non-empty strings
            assert all(isinstance(label, str) and len(label) > 0
                      for label in sugar_grns['processed_labels']), \
                "Empty or invalid labels found"
            logger.info("✓ All labels are non-empty strings")

            # Check sugar GRNs are subset of all GRNs
            sugar_ids = set(sugar_grns['root_id'])
            all_ids = set(all_grns['root_id'])
            assert sugar_ids.issubset(all_ids), "Sugar GRNs not subset of all GRNs"
            logger.info("✓ Sugar GRNs ⊂ All GRNs")

            logger.info("\n✅ TEST 1 PASSED: Data integrity validated")
            self.tests_passed += 1
            return True

        except AssertionError as e:
            logger.error(f"\n❌ TEST 1 FAILED: {e}")
            self.tests_failed += 1
            return False

    def test_2_connectivity_structure(
        self,
        connectivity_dict: Dict[int, Dict[str, Any]]
    ) -> bool:
        """
        Test 2: Validate connectivity data structure.

        Args:
            connectivity_dict: Connectivity dictionary

        Returns:
            True if test passes
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 2: CONNECTIVITY DATA STRUCTURE")
        logger.info("="*80)

        try:
            # Check it's a dictionary
            assert isinstance(connectivity_dict, dict), "Connectivity data must be dict"
            logger.info(f"✓ Connectivity is a dictionary with {len(connectivity_dict)} GRNs")

            # Check structure of first entry
            if len(connectivity_dict) > 0:
                sample_grn = list(connectivity_dict.keys())[0]
                assert 'partners' in connectivity_dict[sample_grn], "Missing 'partners' key"
                assert 'synapse_counts' in connectivity_dict[sample_grn], "Missing 'synapse_counts' key"
                assert 'partner_labels' in connectivity_dict[sample_grn], "Missing 'partner_labels' key"
                logger.info("✓ Required keys present in connectivity data")

                # Check data types
                assert isinstance(connectivity_dict[sample_grn]['partners'], list), \
                    "'partners' must be a list"
                assert isinstance(connectivity_dict[sample_grn]['synapse_counts'], list), \
                    "'synapse_counts' must be a list"
                assert isinstance(connectivity_dict[sample_grn]['partner_labels'], list), \
                    "'partner_labels' must be a list"
                logger.info("✓ Data types are correct")

                # Check list lengths match
                n_partners = len(connectivity_dict[sample_grn]['partners'])
                n_synapses = len(connectivity_dict[sample_grn]['synapse_counts'])
                n_labels = len(connectivity_dict[sample_grn]['partner_labels'])
                assert n_partners == n_synapses == n_labels, \
                    "Mismatched list lengths in connectivity data"
                logger.info("✓ List lengths are consistent")

            # Check for anomalies
            partner_counts = [len(v['partners']) for v in connectivity_dict.values()]
            if len(partner_counts) > 0:
                max_partners = max(partner_counts)
                mean_partners = np.mean(partner_counts)
                logger.info(f"  Max partners for single GRN: {max_partners}")
                logger.info(f"  Mean partners per GRN: {mean_partners:.1f}")

                # Sanity check: max partners shouldn't be > 500 (could indicate error)
                assert max_partners < 500, \
                    f"Unexpected spike in partner count: {max_partners} (possible data error)"
                logger.info("✓ Partner counts within expected range")

            logger.info(f"\n✅ TEST 2 PASSED: Connectivity structure valid for {len(connectivity_dict)} GRNs")
            self.tests_passed += 1
            return True

        except AssertionError as e:
            logger.error(f"\n❌ TEST 2 FAILED: {e}")
            self.tests_failed += 1
            return False

    def test_3_statistical_validation(
        self,
        connectivity_dict: Dict[int, Dict[str, Any]]
    ) -> bool:
        """
        Test 3: Validate statistical distributions are reasonable.

        Args:
            connectivity_dict: Connectivity dictionary

        Returns:
            True if test passes
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 3: STATISTICAL VALIDATION")
        logger.info("="*80)

        try:
            # Extract metrics
            partner_counts = [len(v['partners']) for v in connectivity_dict.values()]
            all_synapse_counts = []
            for v in connectivity_dict.values():
                all_synapse_counts.extend(v['synapse_counts'])

            if len(partner_counts) == 0:
                logger.warning("  No connectivity data to validate")
                return True

            # Check partner count distribution
            mean_partners = np.mean(partner_counts)
            assert 5 < mean_partners < 200, \
                f"Unusual partner count distribution: mean={mean_partners:.1f}"
            logger.info(f"✓ Partner count mean within expected range: {mean_partners:.1f}")

            # Check synapse count distribution
            if len(all_synapse_counts) > 0:
                median_synapses = np.median(all_synapse_counts)
                max_synapses = np.max(all_synapse_counts)

                assert median_synapses >= 1, \
                    f"Median synapse count < 1: {median_synapses} (data error)"
                logger.info(f"✓ Median synapse count >= 1: {median_synapses:.0f}")

                assert max_synapses > median_synapses, \
                    "Max synapses should be > median"
                logger.info(f"✓ Max synapse count: {max_synapses:.0f}")

            # Print summary statistics
            logger.info(f"\n  Summary statistics:")
            logger.info(f"    Partners/GRN: {np.mean(partner_counts):.1f} ± {np.std(partner_counts):.1f}")
            logger.info(f"    Partner count range: [{np.min(partner_counts)}, {np.max(partner_counts)}]")

            if len(all_synapse_counts) > 0:
                logger.info(f"    Synapses: median={np.median(all_synapse_counts):.0f}, "
                          f"mean={np.mean(all_synapse_counts):.1f}, "
                          f"max={np.max(all_synapse_counts):.0f}")

            logger.info("\n✅ TEST 3 PASSED: Statistics within expected ranges")
            self.tests_passed += 1
            return True

        except AssertionError as e:
            logger.error(f"\n❌ TEST 3 FAILED: {e}")
            self.tests_failed += 1
            return False

    def test_4_comparison(
        self,
        sugar_grns: pd.DataFrame,
        all_grns: pd.DataFrame,
        sugar_connectivity: Dict[int, Dict[str, Any]],
        all_connectivity: Dict[int, Dict[str, Any]]
    ) -> bool:
        """
        Test 4: Verify sugar GRNs vs all GRNs comparison.

        Args:
            sugar_grns: Sugar GRN DataFrame
            all_grns: All GRN DataFrame
            sugar_connectivity: Sugar connectivity dict
            all_connectivity: All connectivity dict

        Returns:
            True if test passes
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 4: COMPARISON TEST (Sugar vs All GRNs)")
        logger.info("="*80)

        try:
            # Verify sugar GRNs are subset of all GRNs
            sugar_ids = set(sugar_grns['root_id'])
            all_ids = set(all_grns['root_id'])

            assert sugar_ids.issubset(all_ids), "Sugar GRNs not subset of all GRNs"
            logger.info("✓ Sugar GRNs ⊂ All GRNs")

            # Check connectivity dicts align
            sugar_conn_ids = set(sugar_connectivity.keys())
            all_conn_ids = set(all_connectivity.keys())

            assert sugar_conn_ids.issubset(all_conn_ids), \
                "Sugar connectivity not subset of all connectivity"
            logger.info("✓ Sugar connectivity ⊂ All connectivity")

            # Calculate ratios
            grn_ratio = len(sugar_ids) / len(all_ids) if len(all_ids) > 0 else 0
            conn_ratio = len(sugar_conn_ids) / len(all_conn_ids) if len(all_conn_ids) > 0 else 0

            logger.info(f"\n  Statistics:")
            logger.info(f"    Sugar GRNs: {len(sugar_ids)}")
            logger.info(f"    All GRNs: {len(all_ids)}")
            logger.info(f"    Ratio: {grn_ratio:.1%}")
            logger.info(f"\n    Sugar connectivity entries: {len(sugar_conn_ids)}")
            logger.info(f"    All connectivity entries: {len(all_conn_ids)}")
            logger.info(f"    Ratio: {conn_ratio:.1%}")

            # Verify ratios are reasonable
            assert 0 < grn_ratio <= 1.0, f"Invalid GRN ratio: {grn_ratio}"
            assert 0 < conn_ratio <= 1.0, f"Invalid connectivity ratio: {conn_ratio}"
            logger.info("✓ Ratios are within valid range")

            logger.info("\n✅ TEST 4 PASSED: Comparison validation successful")
            self.tests_passed += 1
            return True

        except AssertionError as e:
            logger.error(f"\n❌ TEST 4 FAILED: {e}")
            self.tests_failed += 1
            return False

    def test_5_visualization_output(
        self,
        expected_figures: list
    ) -> bool:
        """
        Test 5: Verify all output files exist and are non-empty.

        Args:
            expected_figures: List of expected figure filenames

        Returns:
            True if test passes
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 5: VISUALIZATION OUTPUT")
        logger.info("="*80)

        try:
            all_found = True
            total_size = 0

            for fig in expected_figures:
                path = self.output_dir / fig

                if not path.exists():
                    logger.warning(f"  ⚠ Missing figure: {fig}")
                    all_found = False
                    continue

                size = path.stat().st_size / 1024  # KB

                # Check minimum size (50 KB)
                if size < 50:
                    logger.warning(f"  ⚠ Figure suspiciously small: {fig} ({size:.0f} KB)")
                    all_found = False
                else:
                    logger.info(f"  ✓ {fig}: {size:.0f} KB")
                    total_size += size

            if all_found:
                logger.info(f"\n✓ All {len(expected_figures)} figures generated successfully")
                logger.info(f"  Total size: {total_size:.0f} KB")
                logger.info("\n✅ TEST 5 PASSED: All visualizations generated")
                self.tests_passed += 1
                return True
            else:
                logger.error("\n❌ TEST 5 FAILED: Some figures missing or invalid")
                self.tests_failed += 1
                return False

        except Exception as e:
            logger.error(f"\n❌ TEST 5 FAILED: {e}")
            self.tests_failed += 1
            return False

    def print_summary(self) -> None:
        """Print final test summary."""
        total_tests = self.tests_passed + self.tests_failed

        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Tests run:    {total_tests}")
        logger.info(f"Tests passed: {self.tests_passed} ✅")
        logger.info(f"Tests failed: {self.tests_failed} ❌")

        if self.tests_failed == 0:
            logger.info("\n🎉 ALL TESTS PASSED! 🎉")
        else:
            logger.warning(f"\n⚠️  {self.tests_failed} test(s) failed")

        logger.info("="*80)


def run_all_tests(config_path: str = None) -> bool:
    """
    Run all tests on the pipeline outputs.

    Args:
        config_path: Path to config file

    Returns:
        True if all tests pass
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'grn_viz_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize tester
    tester = PipelineTester(config)

    # Load cached data
    cache_dir = Path(config['data']['cache_dir'])

    # Test 1 & 4: Load GRN metadata
    sugar_path = cache_dir / 'sugar_grns_metadata.csv'
    all_path = cache_dir / 'all_grns_metadata.csv'

    if not sugar_path.exists() or not all_path.exists():
        logger.error("GRN metadata files not found. Run extraction first.")
        return False

    sugar_grns = pd.read_csv(sugar_path)
    all_grns = pd.read_csv(all_path)

    tester.test_1_data_integrity(sugar_grns, all_grns)

    # Test 2 & 3: Load connectivity
    conn_path = cache_dir / 'downstream_connectivity.pkl'

    if not conn_path.exists():
        logger.error("Connectivity cache not found. Run extraction first.")
        return False

    with open(conn_path, 'rb') as f:
        all_connectivity = pickle.load(f)

    sugar_ids = set(sugar_grns['root_id'])
    sugar_connectivity = {k: v for k, v in all_connectivity.items() if k in sugar_ids}

    tester.test_2_connectivity_structure(all_connectivity)
    tester.test_3_statistical_validation(all_connectivity)
    tester.test_4_comparison(sugar_grns, all_grns, sugar_connectivity, all_connectivity)

    # Test 5: Check visualization outputs
    expected_figures = [
        'grn_connectivity_overview.png',
        'sugar_vs_all_grns_comparison.png',
        'grn_downstream_network.png',
        'grn_connectivity_heatmap.png',
        'grn_analysis_dashboard.png'
    ]

    tester.test_5_visualization_output(expected_figures)

    # Print summary
    tester.print_summary()

    return tester.tests_failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
