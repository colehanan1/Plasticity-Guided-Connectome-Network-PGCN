"""
GRN Connectivity Analyzer

This module handles:
1. Computing per-GRN and population-level statistics
2. Comparing sugar GRNs vs all GRNs
3. Identifying convergence patterns
4. Generating summary statistics and tables

Author: Claude AI
Date: 2025-11-03
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
from scipy import stats
from collections import Counter

logger = logging.getLogger(__name__)


class GRNConnectivityAnalyzer:
    """Analyze downstream connectivity patterns for GRNs."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyzer with configuration.

        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.output_dir = Path(config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Analysis parameters
        self.min_synapses = config['analysis']['min_synapses']
        self.convergence_threshold = config['analysis']['convergence_threshold']
        self.top_n = config['analysis']['top_n_partners_display']

        # Statistical parameters
        self.alpha = config['statistics']['alpha']
        self.confidence = config['statistics']['confidence_level']

        logger.info("GRN Connectivity Analyzer initialized")
        logger.info(f"Output directory: {self.output_dir}")

    def compute_connectivity_statistics(
        self,
        connectivity_dict: Dict[int, Dict[str, Any]],
        grn_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for GRN connectivity.

        Args:
            connectivity_dict: Connectivity data from extractor
            grn_df: DataFrame with GRN metadata

        Returns:
            Dictionary with various statistics
        """
        logger.info("Computing connectivity statistics...")

        stats_dict = {
            'per_grn': [],
            'population': {},
            'partner_analysis': {},
            'convergence': {}
        }

        # Per-GRN statistics
        logger.info("  Computing per-GRN statistics...")
        for grn_id, conn_data in connectivity_dict.items():
            partners = conn_data['partners']
            synapse_counts = conn_data['synapse_counts']

            grn_stats = {
                'root_id': grn_id,
                'n_partners': len(partners),
                'total_synapses': sum(synapse_counts),
                'mean_synapses': np.mean(synapse_counts) if len(synapse_counts) > 0 else 0,
                'median_synapses': np.median(synapse_counts) if len(synapse_counts) > 0 else 0,
                'std_synapses': np.std(synapse_counts) if len(synapse_counts) > 0 else 0,
                'max_synapses': max(synapse_counts) if len(synapse_counts) > 0 else 0,
                'min_synapses': min(synapse_counts) if len(synapse_counts) > 0 else 0
            }
            stats_dict['per_grn'].append(grn_stats)

        # Convert to DataFrame for easier manipulation
        per_grn_df = pd.DataFrame(stats_dict['per_grn'])

        # Population statistics
        logger.info("  Computing population statistics...")
        stats_dict['population'] = {
            'n_grns': len(connectivity_dict),
            'total_connections': per_grn_df['n_partners'].sum(),
            'total_synapses': per_grn_df['total_synapses'].sum(),
            'partners_per_grn': {
                'min': per_grn_df['n_partners'].min(),
                'max': per_grn_df['n_partners'].max(),
                'mean': per_grn_df['n_partners'].mean(),
                'median': per_grn_df['n_partners'].median(),
                'std': per_grn_df['n_partners'].std()
            },
            'synapses_per_connection': {
                'min': per_grn_df['min_synapses'].min(),
                'max': per_grn_df['max_synapses'].max(),
                'mean': per_grn_df['mean_synapses'].mean(),
                'median': per_grn_df['median_synapses'].median()
            }
        }

        # Partner analysis
        logger.info("  Analyzing downstream partners...")
        all_partners = []
        partner_label_counts = Counter()

        for conn_data in connectivity_dict.values():
            all_partners.extend(conn_data['partners'])
            partner_label_counts.update(conn_data['partner_labels'])

        unique_partners = set(all_partners)
        stats_dict['partner_analysis'] = {
            'unique_partners': len(unique_partners),
            'total_connections': len(all_partners),
            'partner_type_distribution': dict(partner_label_counts)
        }

        # Convergence analysis
        logger.info("  Computing convergence patterns...")
        partner_counter = Counter(all_partners)
        convergent_partners = {
            partner: count
            for partner, count in partner_counter.items()
            if count >= self.convergence_threshold
        }

        stats_dict['convergence'] = {
            'n_convergent_partners': len(convergent_partners),
            'avg_grns_per_target': np.mean(list(partner_counter.values())),
            'max_convergence': max(partner_counter.values()) if partner_counter else 0,
            'top_convergent_targets': dict(
                sorted(convergent_partners.items(), key=lambda x: x[1], reverse=True)[:self.top_n]
            )
        }

        # Print summary
        self._print_statistics_summary(stats_dict)

        return stats_dict

    def compare_sugar_vs_all(
        self,
        sugar_connectivity: Dict[int, Dict[str, Any]],
        all_connectivity: Dict[int, Dict[str, Any]],
        sugar_grn_df: pd.DataFrame,
        all_grn_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Statistical comparison of sugar GRNs vs all GRNs.

        Args:
            sugar_connectivity: Connectivity data for sugar GRNs
            all_connectivity: Connectivity data for all GRNs
            sugar_grn_df: Metadata for sugar GRNs
            all_grn_df: Metadata for all GRNs

        Returns:
            DataFrame with comparison statistics
        """
        logger.info("Comparing sugar GRNs vs all GRNs...")

        # Extract metrics for comparison
        sugar_partner_counts = [
            len(conn['partners']) for conn in sugar_connectivity.values()
        ]
        all_partner_counts = [
            len(conn['partners']) for conn in all_connectivity.values()
        ]

        sugar_synapse_counts = np.concatenate([
            conn['synapse_counts'] for conn in sugar_connectivity.values()
        ])
        all_synapse_counts = np.concatenate([
            conn['synapse_counts'] for conn in all_connectivity.values()
        ])

        # Statistical tests
        logger.info("  Running Mann-Whitney U test for partner counts...")
        partner_stat, partner_pval = stats.mannwhitneyu(
            sugar_partner_counts,
            all_partner_counts,
            alternative='two-sided'
        )

        logger.info("  Running Mann-Whitney U test for synapse counts...")
        synapse_stat, synapse_pval = stats.mannwhitneyu(
            sugar_synapse_counts,
            all_synapse_counts,
            alternative='two-sided'
        )

        # Effect sizes (Cohen's d)
        partner_effect_size = self._cohens_d(sugar_partner_counts, all_partner_counts)
        synapse_effect_size = self._cohens_d(sugar_synapse_counts, all_synapse_counts)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Metric': [
                'N (GRNs)',
                'Partners per GRN (mean)',
                'Partners per GRN (median)',
                'Partners per GRN (std)',
                'Synapses per connection (mean)',
                'Synapses per connection (median)',
                'Synapses per connection (std)',
                'Partner count p-value',
                'Partner count effect size',
                'Synapse count p-value',
                'Synapse count effect size'
            ],
            'Sugar GRNs': [
                len(sugar_connectivity),
                np.mean(sugar_partner_counts),
                np.median(sugar_partner_counts),
                np.std(sugar_partner_counts),
                np.mean(sugar_synapse_counts),
                np.median(sugar_synapse_counts),
                np.std(sugar_synapse_counts),
                partner_pval,
                partner_effect_size,
                synapse_pval,
                synapse_effect_size
            ],
            'All GRNs': [
                len(all_connectivity),
                np.mean(all_partner_counts),
                np.median(all_partner_counts),
                np.std(all_partner_counts),
                np.mean(all_synapse_counts),
                np.median(all_synapse_counts),
                np.std(all_synapse_counts),
                partner_pval,
                partner_effect_size,
                synapse_pval,
                synapse_effect_size
            ]
        })

        # Print comparison summary
        logger.info("\n" + "="*80)
        logger.info("SUGAR GRN vs ALL GRN COMPARISON")
        logger.info("="*80)
        logger.info(f"Sugar GRNs: {len(sugar_connectivity)}")
        logger.info(f"All GRNs: {len(all_connectivity)}")
        logger.info(f"")
        logger.info(f"Partners per GRN:")
        logger.info(f"  Sugar: {np.mean(sugar_partner_counts):.1f} ± {np.std(sugar_partner_counts):.1f}")
        logger.info(f"  All:   {np.mean(all_partner_counts):.1f} ± {np.std(all_partner_counts):.1f}")
        logger.info(f"  p-value: {partner_pval:.4f} {'***' if partner_pval < 0.001 else '**' if partner_pval < 0.01 else '*' if partner_pval < 0.05 else 'ns'}")
        logger.info(f"  Effect size (Cohen's d): {partner_effect_size:.3f}")
        logger.info(f"")
        logger.info(f"Synapses per connection:")
        logger.info(f"  Sugar: {np.mean(sugar_synapse_counts):.1f} ± {np.std(sugar_synapse_counts):.1f}")
        logger.info(f"  All:   {np.mean(all_synapse_counts):.1f} ± {np.std(all_synapse_counts):.1f}")
        logger.info(f"  p-value: {synapse_pval:.4f} {'***' if synapse_pval < 0.001 else '**' if synapse_pval < 0.01 else '*' if synapse_pval < 0.05 else 'ns'}")
        logger.info(f"  Effect size (Cohen's d): {synapse_effect_size:.3f}")
        logger.info("="*80)

        return comparison_df

    def generate_partner_table(
        self,
        connectivity_dict: Dict[int, Dict[str, Any]],
        top_n: int = None
    ) -> pd.DataFrame:
        """
        Generate table of top downstream partners.

        Args:
            connectivity_dict: Connectivity data
            top_n: Number of top partners to include (None = all)

        Returns:
            DataFrame with partner information
        """
        logger.info(f"Generating partner table (top {top_n if top_n else 'all'})...")

        # Aggregate partners across all GRNs
        partner_data = {}

        for grn_id, conn_data in connectivity_dict.items():
            for i, partner_id in enumerate(conn_data['partners']):
                if partner_id not in partner_data:
                    partner_data[partner_id] = {
                        'partner_id': partner_id,
                        'n_presynaptic_grns': 0,
                        'total_synapses': 0,
                        'mean_synapses': 0,
                        'labels': []
                    }

                partner_data[partner_id]['n_presynaptic_grns'] += 1
                partner_data[partner_id]['total_synapses'] += conn_data['synapse_counts'][i]
                partner_data[partner_id]['labels'].append(conn_data['partner_labels'][i])

        # Create DataFrame
        partner_df = pd.DataFrame(list(partner_data.values()))

        # Compute mean synapses
        partner_df['mean_synapses'] = (
            partner_df['total_synapses'] / partner_df['n_presynaptic_grns']
        )

        # Get most common label for each partner
        partner_df['primary_label'] = partner_df['labels'].apply(
            lambda x: Counter(x).most_common(1)[0][0] if x else 'Unknown'
        )
        partner_df = partner_df.drop(columns=['labels'])

        # Sort by number of presynaptic GRNs (convergence)
        partner_df = partner_df.sort_values('n_presynaptic_grns', ascending=False)

        # Filter to top N if specified
        if top_n:
            partner_df = partner_df.head(top_n)

        logger.info(f"✓ Generated partner table with {len(partner_df)} entries")

        return partner_df

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def _print_statistics_summary(self, stats_dict: Dict[str, Any]) -> None:
        """
        Print formatted summary of statistics.

        Args:
            stats_dict: Statistics dictionary
        """
        pop = stats_dict['population']
        partner = stats_dict['partner_analysis']
        conv = stats_dict['convergence']

        logger.info("\n" + "="*80)
        logger.info("CONNECTIVITY STATISTICS SUMMARY")
        logger.info("="*80)
        logger.info(f"✓ GRNs analyzed: {pop['n_grns']}")
        logger.info(f"✓ Total connections: {pop['total_connections']:,}")
        logger.info(f"✓ Total synapses: {pop['total_synapses']:,}")
        logger.info(f"")
        logger.info(f"Partners per GRN:")
        logger.info(f"  min={pop['partners_per_grn']['min']:.0f}, "
                   f"max={pop['partners_per_grn']['max']:.0f}, "
                   f"mean={pop['partners_per_grn']['mean']:.1f}, "
                   f"median={pop['partners_per_grn']['median']:.0f}")
        logger.info(f"")
        logger.info(f"Synapses per connection:")
        logger.info(f"  min={pop['synapses_per_connection']['min']:.0f}, "
                   f"max={pop['synapses_per_connection']['max']:.0f}, "
                   f"mean={pop['synapses_per_connection']['mean']:.1f}, "
                   f"median={pop['synapses_per_connection']['median']:.0f}")
        logger.info(f"")
        logger.info(f"✓ Unique downstream partners: {partner['unique_partners']}")
        logger.info(f"✓ Partner type distribution:")
        for ptype, count in sorted(partner['partner_type_distribution'].items(),
                                   key=lambda x: x[1], reverse=True):
            pct = 100 * count / partner['total_connections']
            logger.info(f"    - {ptype}: {count} ({pct:.1f}%)")
        logger.info(f"")
        logger.info(f"✓ Convergence analysis:")
        logger.info(f"  Convergent partners (≥{self.convergence_threshold} GRNs): "
                   f"{conv['n_convergent_partners']}")
        logger.info(f"  Avg GRNs per target: {conv['avg_grns_per_target']:.2f}")
        logger.info(f"  Max convergence: {conv['max_convergence']} GRNs")
        logger.info("="*80)

    def save_statistics(
        self,
        stats_dict: Dict[str, Any],
        comparison_df: pd.DataFrame,
        partner_df: pd.DataFrame,
        prefix: str = ""
    ) -> None:
        """
        Save all statistics to CSV files.

        Args:
            stats_dict: Statistics dictionary
            comparison_df: Comparison DataFrame
            partner_df: Partner table DataFrame
            prefix: Optional prefix for filenames
        """
        logger.info("Saving statistics to files...")

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Save per-GRN statistics
        per_grn_path = self.output_dir / f'{prefix}per_grn_stats_{timestamp}.csv'
        per_grn_df = pd.DataFrame(stats_dict['per_grn'])
        per_grn_df.to_csv(per_grn_path, index=False)
        logger.info(f"✓ Saved per-GRN statistics: {per_grn_path}")

        # Save comparison
        comparison_path = self.output_dir / f'{prefix}sugar_vs_all_comparison_{timestamp}.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"✓ Saved comparison: {comparison_path}")

        # Save partner table
        partner_path = self.output_dir / f'{prefix}downstream_partners_{timestamp}.csv'
        partner_df.to_csv(partner_path, index=False)
        logger.info(f"✓ Saved partner table: {partner_path}")

        # Save population stats as text summary
        summary_path = self.output_dir / f'{prefix}connectivity_summary_{timestamp}.txt'
        with open(summary_path, 'w') as f:
            f.write("GRN DOWNSTREAM CONNECTIVITY SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            pop = stats_dict['population']
            f.write(f"Population Statistics:\n")
            f.write(f"  N GRNs: {pop['n_grns']}\n")
            f.write(f"  Total connections: {pop['total_connections']:,}\n")
            f.write(f"  Total synapses: {pop['total_synapses']:,}\n\n")

            f.write(f"Partners per GRN:\n")
            for key, val in pop['partners_per_grn'].items():
                f.write(f"  {key}: {val:.2f}\n")

            f.write(f"\nSynapses per connection:\n")
            for key, val in pop['synapses_per_connection'].items():
                f.write(f"  {key}: {val:.2f}\n")

            partner = stats_dict['partner_analysis']
            f.write(f"\nPartner Analysis:\n")
            f.write(f"  Unique partners: {partner['unique_partners']}\n")
            f.write(f"  Total connections: {partner['total_connections']}\n")

            conv = stats_dict['convergence']
            f.write(f"\nConvergence:\n")
            f.write(f"  Convergent partners: {conv['n_convergent_partners']}\n")
            f.write(f"  Avg GRNs per target: {conv['avg_grns_per_target']:.2f}\n")
            f.write(f"  Max convergence: {conv['max_convergence']}\n")

        logger.info(f"✓ Saved summary: {summary_path}")


def main():
    """Example usage of the analyzer."""
    import yaml
    import pickle
    from pathlib import Path

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'grn_viz_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize analyzer
    analyzer = GRNConnectivityAnalyzer(config)

    # Load cached connectivity data
    cache_dir = Path(config['data']['cache_dir'])
    conn_path = cache_dir / 'downstream_connectivity.pkl'

    if not conn_path.exists():
        logger.error("Connectivity cache not found. Run extractor first.")
        return

    with open(conn_path, 'rb') as f:
        connectivity_dict = pickle.load(f)

    # Load GRN metadata
    sugar_path = cache_dir / 'sugar_grns_metadata.csv'
    all_path = cache_dir / 'all_grns_metadata.csv'

    sugar_grns = pd.read_csv(sugar_path)
    all_grns = pd.read_csv(all_path)

    # Filter connectivity to sugar vs all
    sugar_ids = set(sugar_grns['root_id'])
    sugar_connectivity = {k: v for k, v in connectivity_dict.items() if k in sugar_ids}
    all_connectivity = connectivity_dict

    print("\n" + "="*80)
    print("PHASE 3: ANALYZE CONNECTIVITY")
    print("="*80)

    # Compute statistics
    sugar_stats = analyzer.compute_connectivity_statistics(sugar_connectivity, sugar_grns)
    all_stats = analyzer.compute_connectivity_statistics(all_connectivity, all_grns)

    # Compare sugar vs all
    comparison_df = analyzer.compare_sugar_vs_all(
        sugar_connectivity, all_connectivity, sugar_grns, all_grns
    )

    # Generate partner tables
    partner_df = analyzer.generate_partner_table(sugar_connectivity, top_n=30)

    # Save results
    analyzer.save_statistics(sugar_stats, comparison_df, partner_df, prefix="sugar_")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
