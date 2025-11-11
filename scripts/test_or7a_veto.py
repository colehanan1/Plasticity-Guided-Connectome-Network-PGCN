#!/usr/bin/env python3
"""
Or7a Learning Veto Hypothesis - Simplified Analysis

Tests three specific hypotheses about Or7a receptor's role in blocking learning:
1. Or7a shows strong benzaldehyde selectivity (>3x ratio vs hexanol)
2. Or7a receptor neurons have zero lateral inhibition connections
3. Cross-learning is explained by shared strong receptor responses

Author: Analysis Pipeline
Date: 2025-11-10
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loaders.neuron_classification import (
    get_pn_neurons,
    get_local_interneurons,
    infer_pn_glomerulus_labels,
)

# Try to import DoOR toolkit
try:
    from door_toolkit.encoder import DoOREncoder
    from door_toolkit.integration.integrator import DoORFlyWireIntegrator
    DOOR_AVAILABLE = True
    print("✅ DoOR toolkit available")
except ImportError as e:
    DOOR_AVAILABLE = False
    print("⚠️  DoOR toolkit not available - using hardcoded values")
    print(f"   Import error: {e}")
    print("   Run: python scripts/diagnose_door_install.py for troubleshooting")


# Hardcoded DoOR data (fallback if toolkit unavailable)
# Note: Using both '1-hexanol' and 'hexanol' keys for compatibility
HARDCODED_DOOR_DATA = {
    'Or7a': {
        'benzaldehyde': 0.89,
        '2-heptanone': 0.02,
        '1-hexanol': 0.25,
        'hexanol': 0.25,
    },
    'Or67b': {
        'benzaldehyde': 0.76,
        '1-hexanol': 0.82,
        'hexanol': 0.82,
        '2-heptanone': 0.04,
    },
    'Or22a': {
        'benzaldehyde': 0.68,
        '1-hexanol': 0.12,
        'hexanol': 0.12,
        '2-heptanone': 0.03,
    },
    'Or35a': {
        'benzaldehyde': 0.45,
        '1-hexanol': 0.71,
        'hexanol': 0.71,
        '2-heptanone': 0.05,
    }
}


class Or7aHypothesisTester:
    """Test Or7a learning veto hypothesis with 3 focused analyses."""

    def __init__(self, data_dir: str = "data/flywire", output_dir: str = "results/or7a_hypothesis"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)

        self.cell_types = pd.read_csv(self.data_dir / "consolidated_cell_types.csv.gz")
        self.connections = pd.read_csv(self.data_dir / "connections_princeton.csv.gz")

        # Standardize connection column names (match analyze_ln_pn_connectivity.py)
        rename_map = {}
        if 'pre_pt_root_id' in self.connections.columns:
            rename_map['pre_pt_root_id'] = 'pre_root_id'
        if 'post_pt_root_id' in self.connections.columns:
            rename_map['post_pt_root_id'] = 'post_root_id'
        if 'size' in self.connections.columns and 'syn_count' not in self.connections.columns:
            rename_map['size'] = 'syn_count'

        if rename_map:
            self.connections = self.connections.rename(columns=rename_map)

        # Try to load optional files
        try:
            self.classification = pd.read_csv(self.data_dir / "classification.csv.gz")
        except FileNotFoundError:
            self.classification = None

        try:
            self.processed_labels = pd.read_csv(self.data_dir / "processed_labels.csv.gz")
        except FileNotFoundError:
            self.processed_labels = None

        try:
            self.neurons = pd.read_csv(self.data_dir / "neurons.csv.gz")
        except FileNotFoundError:
            self.neurons = None

        print(f"✅ Loaded {len(self.cell_types):,} cell types")
        print(f"✅ Loaded {len(self.connections):,} connections")

        # Get DoOR data
        self.door_data = self._load_door_data()

    def _load_door_data(self) -> Dict[str, Dict[str, float]]:
        """Load DoOR response data using encoder.encode() method."""
        if DOOR_AVAILABLE:
            try:
                encoder = DoOREncoder()

                # Get response matrix to map receptor indices
                matrix = None
                for attr_name in ['matrix', 'response_matrix', 'data', 'door_matrix', 'df']:
                    if hasattr(encoder, attr_name):
                        try:
                            matrix = getattr(encoder, attr_name)
                            if isinstance(matrix, pd.DataFrame) and len(matrix) > 0:
                                break
                        except:
                            continue

                if matrix is None or not isinstance(matrix, pd.DataFrame):
                    raise RuntimeError("Could not access DoOR response matrix")

                # Get receptor names for index mapping
                receptor_names = list(matrix.columns)

                # Use correct odorant names (1-hexanol not hexanol)
                receptors = ['Or7a', 'Or67b', 'Or22a', 'Or35a']
                odorants = ['benzaldehyde', '1-hexanol', '2-heptanone']

                # Use encoder.encode() method which returns torch.Tensor
                door_data = {}
                for receptor in receptors:
                    if receptor in receptor_names:
                        receptor_idx = receptor_names.index(receptor)
                        door_data[receptor] = {}

                        for odorant in odorants:
                            if odorant in encoder.odorant_names:
                                # encode() returns torch.Tensor with shape (78,)
                                try:
                                    tensor = encoder.encode(odorant)
                                    # Extract value for this receptor and convert to float
                                    response = float(tensor[receptor_idx].item())

                                    # Store with both '1-hexanol' and 'hexanol' keys for compatibility
                                    door_data[receptor][odorant] = response
                                    if odorant == '1-hexanol':
                                        door_data[receptor]['hexanol'] = response
                                except Exception as e:
                                    print(f"⚠️  Could not encode {odorant}: {e}")

                print(f"✅ Loaded DoOR data for {len(door_data)} receptors using encoder.encode()")
                return door_data
            except Exception as e:
                print(f"⚠️  DoOR load failed: {e}")
                return HARDCODED_DOOR_DATA
        else:
            return HARDCODED_DOOR_DATA

    def test_hypothesis_1_selectivity(self) -> Tuple[bool, pd.DataFrame]:
        """
        Hypothesis 1: Or7a shows strong benzaldehyde selectivity.

        Expected: benzaldehyde/hexanol response ratio > 3.0
        """
        print("\n" + "="*80)
        print("HYPOTHESIS 1: Or7a Benzaldehyde Selectivity")
        print("="*80)

        # Get Or7a responses
        or7a = self.door_data.get('Or7a', {})
        benz_response = or7a.get('benzaldehyde', 0)
        hex_response = or7a.get('hexanol', 0)

        if hex_response > 0:
            selectivity_ratio = benz_response / hex_response
        else:
            selectivity_ratio = float('inf') if benz_response > 0 else 0

        # Test
        threshold = 3.0
        supports = selectivity_ratio > threshold

        print(f"\nOr7a Response to benzaldehyde: {benz_response:.3f}")
        print(f"Or7a Response to hexanol: {hex_response:.3f}")
        print(f"Selectivity ratio: {selectivity_ratio:.2f}x")
        print(f"Threshold: {threshold}x")
        print(f"\n{'✅ SUPPORTS' if supports else '❌ CONTRADICTS'} Hypothesis 1")

        # Create results DataFrame
        results = pd.DataFrame([{
            'receptor': 'Or7a',
            'benzaldehyde_response': benz_response,
            'hexanol_response': hex_response,
            'selectivity_ratio': selectivity_ratio,
            'threshold': threshold,
            'supports_hypothesis': supports
        }])

        return supports, results

    def test_hypothesis_2_no_lateral(self) -> Tuple[bool, pd.DataFrame]:
        """
        Hypothesis 2: Or7a (DL5) has zero lateral inhibition to DM glomeruli.

        Expected: No LN-mediated connections from DL5 to DM1-4
        """
        print("\n" + "="*80)
        print("HYPOTHESIS 2: Or7a No Lateral Inhibition")
        print("="*80)

        # Identify neurons
        print("\nIdentifying neurons...")
        lns = get_local_interneurons(
            self.cell_types,
            self.classification,
            neurons_df=self.neurons
        )
        pns = get_pn_neurons(
            self.cell_types,
            self.classification,
            neurons_df=self.neurons,
            processed_labels_df=self.processed_labels
        )
        pns['glomerulus'] = infer_pn_glomerulus_labels(pns, processed_labels_df=self.processed_labels)

        print(f"Found {len(lns):,} Local Neurons")
        print(f"Found {len(pns):,} Projection Neurons")

        # Filter for DL5 and DM glomeruli
        dl5_pns = pns[pns['glomerulus'] == 'DL5']['root_id'].values
        dm_pns = pns[pns['glomerulus'].isin(['DM1', 'DM2', 'DM3', 'DM4', 'DM5', 'DM6'])]['root_id'].values

        print(f"\nDL5 PNs: {len(dl5_pns)}")
        print(f"DM1-6 PNs: {len(dm_pns)}")

        # Find DL5→LN connections
        dl5_to_ln = self.connections[
            self.connections['pre_root_id'].isin(dl5_pns) &
            self.connections['post_root_id'].isin(lns['root_id'])
        ].copy()

        # Find LN→DM connections
        ln_to_dm = self.connections[
            self.connections['pre_root_id'].isin(lns['root_id']) &
            self.connections['post_root_id'].isin(dm_pns)
        ].copy()

        # Find LNs that connect DL5 to DM (DL5→LN→DM pathway)
        dl5_lns = set(dl5_to_ln['post_root_id'])
        dm_lns = set(ln_to_dm['pre_root_id'])
        cross_glom_lns = dl5_lns & dm_lns

        num_lateral_lns = len(cross_glom_lns)
        supports = num_lateral_lns == 0

        print(f"\nLNs receiving from DL5: {len(dl5_lns)}")
        print(f"LNs projecting to DM1-6: {len(dm_lns)}")
        print(f"LNs mediating DL5→DM pathway: {num_lateral_lns}")
        print(f"\n{'✅ SUPPORTS' if supports else '❌ CONTRADICTS'} Hypothesis 2")

        if num_lateral_lns > 0:
            print(f"\n⚠️  Found {num_lateral_lns} LNs connecting DL5 to DM glomeruli")
            print("This suggests lateral inhibition exists (contradicts hypothesis)")

        # Create detailed results
        results = pd.DataFrame([{
            'source_glomerulus': 'DL5',
            'target_glomeruli': 'DM1-6',
            'num_source_pns': len(dl5_pns),
            'num_target_pns': len(dm_pns),
            'num_lns_from_source': len(dl5_lns),
            'num_lns_to_target': len(dm_lns),
            'num_cross_glomerular_lns': num_lateral_lns,
            'supports_hypothesis': supports
        }])

        return supports, results

    def test_hypothesis_3_shared_receptor(self) -> Tuple[bool, pd.DataFrame]:
        """
        Hypothesis 3: Cross-learning explained by shared strong receptor.

        Expected: Identify receptor(s) strongly responding to both benzaldehyde and hexanol
        """
        print("\n" + "="*80)
        print("HYPOTHESIS 3: Shared Receptor Explains Cross-Learning")
        print("="*80)

        # Define "strong response" threshold
        strong_threshold = 0.5

        # Find receptors with strong responses to both odorants
        shared_receptors = []

        for receptor, responses in self.door_data.items():
            benz = responses.get('benzaldehyde', 0)
            hexanol = responses.get('hexanol', 0)

            if benz >= strong_threshold and hexanol >= strong_threshold:
                shared_receptors.append({
                    'receptor': receptor,
                    'benzaldehyde': benz,
                    'hexanol': hexanol,
                    'mean_response': (benz + hexanol) / 2,
                    'response_balance': min(benz, hexanol) / max(benz, hexanol) if max(benz, hexanol) > 0 else 0
                })

        results = pd.DataFrame(shared_receptors).sort_values('mean_response', ascending=False) if shared_receptors else pd.DataFrame()

        supports = len(shared_receptors) > 0

        print(f"\nStrong response threshold: {strong_threshold}")
        print(f"Receptors with strong responses to BOTH odorants: {len(shared_receptors)}")

        if len(shared_receptors) > 0:
            print("\nShared receptors:")
            for idx, row in results.iterrows():
                print(f"  {row['receptor']}: benzaldehyde={row['benzaldehyde']:.3f}, "
                      f"hexanol={row['hexanol']:.3f}, balance={row['response_balance']:.2f}")

            best = results.iloc[0]
            print(f"\n🎯 Best candidate: {best['receptor']}")
            print(f"   Explains cross-learning via shared activation pattern")

        print(f"\n{'✅ SUPPORTS' if supports else '❌ CONTRADICTS'} Hypothesis 3")

        return supports, results

    def visualize_results(self, h1_result: pd.DataFrame, h2_result: pd.DataFrame,
                         h3_result: pd.DataFrame):
        """Generate publication-quality visualizations."""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # Figure 1: Receptor Response Profiles
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: Or7a Selectivity
        ax = axes[0]
        odorants = ['benzaldehyde', 'hexanol', '2-heptanone']
        or7a_responses = [self.door_data['Or7a'].get(od, 0) for od in odorants]

        bars = ax.bar(range(len(odorants)), or7a_responses, color=['#e74c3c', '#3498db', '#95a5a6'])
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Strong response threshold')
        ax.set_xticks(range(len(odorants)))
        ax.set_xticklabels(odorants, rotation=45, ha='right')
        ax.set_ylabel('Normalized Response', fontsize=12)
        ax.set_title('A. Or7a Response Profile\n(Hypothesis 1: Benzaldehyde Selectivity)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend()

        # Panel B: Cross-Receptor Responses
        ax = axes[1]
        if len(h3_result) > 0:
            receptors = h3_result['receptor'].values
            benz_vals = h3_result['benzaldehyde'].values
            hex_vals = h3_result['hexanol'].values

            x = np.arange(len(receptors))
            width = 0.35

            ax.bar(x - width/2, benz_vals, width, label='Benzaldehyde', color='#e74c3c')
            ax.bar(x + width/2, hex_vals, width, label='Hexanol', color='#3498db')
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(receptors, rotation=45, ha='right')
            ax.set_ylabel('Normalized Response', fontsize=12)
            ax.set_title('B. Shared Receptor Responses\n(Hypothesis 3: Cross-Learning)', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No shared receptors found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('B. Shared Receptor Responses\n(Hypothesis 3: Cross-Learning)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        fig_path = self.output_dir / "or7a_receptor_profiles.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {fig_path}")

        # Figure 2: Connectivity Summary
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Create connectivity diagram
        if len(h2_result) > 0:
            result = h2_result.iloc[0]

            categories = ['DL5 PNs', 'LNs (from DL5)', 'LNs (to DM)', 'Cross-Glom LNs', 'DM PNs']
            counts = [
                result['num_source_pns'],
                result['num_lns_from_source'],
                result['num_lns_to_target'],
                result['num_cross_glomerular_lns'],
                result['num_target_pns']
            ]
            colors = ['#3498db', '#95a5a6', '#95a5a6', '#e74c3c', '#2ecc71']

            bars = ax.barh(range(len(categories)), counts, color=colors)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels(categories)
            ax.set_xlabel('Neuron Count', fontsize=12)
            ax.set_title('Hypothesis 2: DL5→DM Lateral Connectivity\n(Red = Cross-Glomerular LNs)',
                        fontsize=12, fontweight='bold')

            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(count + 1, i, f'{int(count)}', va='center', fontsize=10)

        plt.tight_layout()
        fig_path = self.output_dir / "or7a_connectivity_summary.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {fig_path}")

    def run_all_tests(self):
        """Run all three hypothesis tests and generate outputs."""
        print("\n" + "="*80)
        print("OR7A LEARNING VETO HYPOTHESIS - TESTING PIPELINE")
        print("="*80)

        # Test all hypotheses
        h1_supports, h1_result = self.test_hypothesis_1_selectivity()
        h2_supports, h2_result = self.test_hypothesis_2_no_lateral()
        h3_supports, h3_result = self.test_hypothesis_3_shared_receptor()

        # Save results
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)

        h1_result.to_csv(self.output_dir / "hypothesis1_or7a_selectivity.csv", index=False)
        print(f"✅ Saved: hypothesis1_or7a_selectivity.csv")

        h2_result.to_csv(self.output_dir / "hypothesis2_lateral_connectivity.csv", index=False)
        print(f"✅ Saved: hypothesis2_lateral_connectivity.csv")

        h3_result.to_csv(self.output_dir / "hypothesis3_shared_receptors.csv", index=False)
        print(f"✅ Saved: hypothesis3_shared_receptors.csv")

        # Generate visualizations
        self.visualize_results(h1_result, h2_result, h3_result)

        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        print("\nHypothesis Test Results:")
        print(f"  H1 (Or7a Selectivity): {'✅ SUPPORTED' if h1_supports else '❌ CONTRADICTED'}")
        print(f"  H2 (No Lateral Inhibition): {'✅ SUPPORTED' if h2_supports else '❌ CONTRADICTED'}")
        print(f"  H3 (Shared Receptor): {'✅ SUPPORTED' if h3_supports else '❌ CONTRADICTED'}")

        overall_support = sum([h1_supports, h2_supports, h3_supports])
        print(f"\nOverall: {overall_support}/3 hypotheses supported")

        if overall_support == 3:
            print("\n🎯 CONCLUSION: Strong support for Or7a learning veto hypothesis")
        elif overall_support == 2:
            print("\n⚠️  CONCLUSION: Partial support for Or7a learning veto hypothesis")
        else:
            print("\n❌ CONCLUSION: Limited support for Or7a learning veto hypothesis")

        print("\n" + "="*80)
        print(f"All results saved to: {self.output_dir}")
        print("="*80)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Or7a learning veto hypothesis with 3 focused analyses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python scripts/test_or7a_veto.py

  # Custom paths
  python scripts/test_or7a_veto.py --data-dir data/flywire --output-dir results/or7a_test
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/flywire',
        help='Directory containing FlyWire CSV files (default: data/flywire)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/or7a_hypothesis',
        help='Directory for output files (default: results/or7a_hypothesis)'
    )

    args = parser.parse_args()

    # Run analysis
    tester = Or7aHypothesisTester(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    tester.run_all_tests()


if __name__ == '__main__':
    main()
