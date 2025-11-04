"""
GRN Connectivity Visualization

This module handles:
1. Bar chart of downstream partners per GRN
2. Comparison plots (violin/box) for sugar vs all GRNs
3. Force-directed network graph
4. Hierarchical heatmap with clustering
5. Statistical summary dashboard

Author: Claude AI
Date: 2025-11-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


class GRNConnectivityVisualizer:
    """Generate publication-ready visualizations of GRN connectivity."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualizer with configuration.

        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.output_dir = Path(config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization settings
        self.dpi = config['visualization']['figure_dpi']
        self.style = config['visualization']['figure_style']
        self.palette = config['visualization']['color_palette']
        self.log_scale = config['visualization']['synapse_log_scale']
        self.heatmap_cmap = config['visualization']['heatmap_cmap']

        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette(self.palette)

        # Font settings
        plt.rcParams['font.family'] = config['visualization']['font_family']
        plt.rcParams['font.size'] = config['visualization']['font_size_labels']
        plt.rcParams['axes.titlesize'] = config['visualization']['font_size_title']
        plt.rcParams['legend.fontsize'] = config['visualization']['font_size_legend']

        logger.info("GRN Connectivity Visualizer initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"DPI: {self.dpi}")

    def plot_grn_summary(
        self,
        connectivity_dict: Dict[int, Dict[str, Any]],
        grn_df: pd.DataFrame,
        output_filename: str = 'grn_connectivity_overview.png'
    ) -> str:
        """
        Generate Figure 1: Bar chart of downstream partners per GRN.

        Args:
            connectivity_dict: Connectivity data
            grn_df: GRN metadata DataFrame
            output_filename: Output filename

        Returns:
            Path to saved figure
        """
        logger.info("Generating GRN summary bar chart...")

        fig_size = self.config['visualization']['figure_sizes']['overview']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, height_ratios=[2, 1])

        # Prepare data
        data = []
        for grn_id, conn in connectivity_dict.items():
            n_partners = len(conn['partners'])
            total_synapses = sum(conn['synapse_counts'])
            category = grn_df[grn_df['root_id'] == grn_id]['grn_category'].values
            category = category[0] if len(category) > 0 else 'unknown'

            data.append({
                'grn_id': grn_id,
                'n_partners': n_partners,
                'total_synapses': total_synapses,
                'category': category
            })

        df = pd.DataFrame(data).sort_values('n_partners', ascending=False)

        # Color map for categories
        category_colors = {
            'sugar': '#FF6B6B',
            'bitter': '#4ECDC4',
            'water': '#45B7D1',
            'other': '#95A5A6'
        }

        colors = [category_colors.get(cat, '#95A5A6') for cat in df['category']]

        # Plot 1: Number of partners
        bars1 = ax1.bar(range(len(df)), df['n_partners'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('GRN (sorted by partner count)', fontweight='bold')
        ax1.set_ylabel('Number of Downstream Partners', fontweight='bold')
        ax1.set_title('Downstream Connectivity: Partners per GRN', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(axis='y', alpha=0.3)

        # Add category legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat.capitalize())
                          for cat, color in category_colors.items()]
        ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

        # Plot 2: Total synapses
        bars2 = ax2.bar(range(len(df)), df['total_synapses'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('GRN (sorted by partner count)', fontweight='bold')
        ax2.set_ylabel('Total Synapses', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")

        return str(output_path)

    def plot_comparison(
        self,
        sugar_connectivity: Dict[int, Dict[str, Any]],
        all_connectivity: Dict[int, Dict[str, Any]],
        comparison_df: pd.DataFrame,
        output_filename: str = 'sugar_vs_all_grns_comparison.png'
    ) -> str:
        """
        Generate Figure 2: Sugar GRN vs All GRN comparison.

        Args:
            sugar_connectivity: Sugar GRN connectivity
            all_connectivity: All GRN connectivity
            comparison_df: Statistical comparison DataFrame
            output_filename: Output filename

        Returns:
            Path to saved figure
        """
        logger.info("Generating sugar vs all GRNs comparison plots...")

        fig_size = self.config['visualization']['figure_sizes']['comparison']
        fig, axes = plt.subplots(2, 2, figsize=fig_size)

        # Extract data
        sugar_partners = [len(c['partners']) for c in sugar_connectivity.values()]
        all_partners = [len(c['partners']) for c in all_connectivity.values()]

        sugar_synapses = np.concatenate([c['synapse_counts'] for c in sugar_connectivity.values()])
        all_synapses = np.concatenate([c['synapse_counts'] for c in all_connectivity.values()])

        # Prepare data for seaborn
        partner_data = pd.DataFrame({
            'Count': sugar_partners + all_partners,
            'Group': ['Sugar GRNs'] * len(sugar_partners) + ['All GRNs'] * len(all_partners)
        })

        synapse_data = pd.DataFrame({
            'Count': np.concatenate([sugar_synapses, all_synapses]),
            'Group': ['Sugar GRNs'] * len(sugar_synapses) + ['All GRNs'] * len(all_synapses)
        })

        # Plot 1: Violin plot of partner counts
        sns.violinplot(data=partner_data, x='Group', y='Count', ax=axes[0, 0], palette='Set2')
        axes[0, 0].set_title('Distribution of Downstream Partners', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Partners', fontweight='bold')
        axes[0, 0].set_xlabel('')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Add p-value annotation
        pval = comparison_df[comparison_df['Metric'] == 'Partner count p-value']['Sugar GRNs'].values[0]
        axes[0, 0].text(0.5, 0.95, f'p = {pval:.4f}', transform=axes[0, 0].transAxes,
                       ha='center', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Box plot of partner counts
        sns.boxplot(data=partner_data, x='Group', y='Count', ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('Partner Count Distribution (Box Plot)', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Partners', fontweight='bold')
        axes[0, 1].set_xlabel('')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Plot 3: Violin plot of synapse counts
        if self.log_scale and len(synapse_data[synapse_data['Count'] > 0]) > 0:
            synapse_data_plot = synapse_data[synapse_data['Count'] > 0].copy()
            synapse_data_plot['Count'] = np.log10(synapse_data_plot['Count'])
            ylabel = 'Synapse Count (log10)'
        else:
            synapse_data_plot = synapse_data
            ylabel = 'Synapse Count'

        sns.violinplot(data=synapse_data_plot, x='Group', y='Count', ax=axes[1, 0], palette='Set2')
        axes[1, 0].set_title('Distribution of Synapse Counts', fontweight='bold')
        axes[1, 0].set_ylabel(ylabel, fontweight='bold')
        axes[1, 0].set_xlabel('')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Add p-value annotation
        syn_pval = comparison_df[comparison_df['Metric'] == 'Synapse count p-value']['Sugar GRNs'].values[0]
        axes[1, 0].text(0.5, 0.95, f'p = {syn_pval:.4f}', transform=axes[1, 0].transAxes,
                       ha='center', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 4: Cumulative distribution
        axes[1, 1].hist(sugar_partners, bins=20, alpha=0.6, label='Sugar GRNs',
                       cumulative=True, density=True, histtype='step', linewidth=2)
        axes[1, 1].hist(all_partners, bins=20, alpha=0.6, label='All GRNs',
                       cumulative=True, density=True, histtype='step', linewidth=2)
        axes[1, 1].set_title('Cumulative Distribution of Partners', fontweight='bold')
        axes[1, 1].set_xlabel('Number of Partners', fontweight='bold')
        axes[1, 1].set_ylabel('Cumulative Probability', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.suptitle('Sugar GRNs vs All GRNs: Statistical Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")

        return str(output_path)

    def plot_network_graph(
        self,
        connectivity_dict: Dict[int, Dict[str, Any]],
        grn_df: pd.DataFrame,
        max_nodes: int = None,
        output_filename: str = 'grn_downstream_network.png'
    ) -> str:
        """
        Generate Figure 3: Force-directed connectivity network.

        Args:
            connectivity_dict: Connectivity data
            grn_df: GRN metadata
            max_nodes: Maximum nodes to display (None = all)
            output_filename: Output filename

        Returns:
            Path to saved figure
        """
        logger.info("Generating network graph...")

        if max_nodes is None:
            max_nodes = self.config['visualization']['network']['max_nodes']

        fig_size = self.config['visualization']['figure_sizes']['network']
        fig, ax = plt.subplots(figsize=fig_size)

        # Build graph
        G = nx.DiGraph()

        # Add GRN nodes
        for grn_id in connectivity_dict.keys():
            G.add_node(f"GRN_{grn_id}", node_type='grn', root_id=grn_id)

        # Collect all partners and their connection counts
        partner_connections = {}
        for grn_id, conn in connectivity_dict.items():
            for i, partner_id in enumerate(conn['partners']):
                if partner_id not in partner_connections:
                    partner_connections[partner_id] = {
                        'count': 0,
                        'total_synapses': 0,
                        'label': conn['partner_labels'][i]
                    }
                partner_connections[partner_id]['count'] += 1
                partner_connections[partner_id]['total_synapses'] += conn['synapse_counts'][i]

        # Sort partners by connection count and limit
        top_partners = sorted(partner_connections.items(),
                            key=lambda x: x[1]['count'],
                            reverse=True)[:max_nodes]

        top_partner_ids = {p[0] for p in top_partners}

        # Add partner nodes
        for partner_id, info in top_partners:
            G.add_node(f"P_{partner_id}",
                      node_type='partner',
                      root_id=partner_id,
                      label=info['label'])

        # Add edges
        edge_weights = []
        for grn_id, conn in connectivity_dict.items():
            for i, partner_id in enumerate(conn['partners']):
                if partner_id in top_partner_ids:
                    weight = conn['synapse_counts'][i]
                    G.add_edge(f"GRN_{grn_id}", f"P_{partner_id}", weight=weight)
                    edge_weights.append(weight)

        logger.info(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if G.number_of_nodes() == 0:
            logger.warning("No nodes to plot!")
            plt.close()
            return ""

        # Layout
        layout_type = self.config['visualization']['network_layout']
        if layout_type == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout_type == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.circular_layout(G)

        # Node colors and sizes
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'grn':
                node_colors.append('#FF6B6B')  # Red for GRNs
                node_sizes.append(300)
            else:
                label = G.nodes[node].get('label', 'Unknown')
                if label == 'KC':
                    node_colors.append('#4ECDC4')
                elif label == 'MBON':
                    node_colors.append('#95E1D3')
                elif label == 'LH':
                    node_colors.append('#F38181')
                else:
                    node_colors.append('#95A5A6')

                # Size by in-degree
                in_degree = G.in_degree(node)
                node_sizes.append(100 + in_degree * 30)

        # Edge widths (log scale)
        if edge_weights:
            max_weight = max(edge_weights)
            min_weight = min(edge_weights)
            edge_widths = []
            for u, v in G.edges():
                weight = G[u][v]['weight']
                if self.log_scale:
                    normalized = (np.log10(weight + 1) - np.log10(min_weight + 1)) / \
                               (np.log10(max_weight + 1) - np.log10(min_weight + 1) + 1e-10)
                else:
                    normalized = (weight - min_weight) / (max_weight - min_weight + 1e-10)
                edge_widths.append(0.5 + normalized * 3)
        else:
            edge_widths = [1.0] * G.number_of_edges()

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                              alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3,
                              arrows=True, arrowsize=10, ax=ax)

        # Labels for GRNs only (too crowded otherwise)
        grn_labels = {node: f"{G.nodes[node]['root_id']}"
                     for node in G.nodes() if G.nodes[node]['node_type'] == 'grn'}
        nx.draw_networkx_labels(G, pos, labels=grn_labels, font_size=6, ax=ax)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='GRNs'),
            Patch(facecolor='#4ECDC4', label='KC'),
            Patch(facecolor='#95E1D3', label='MBON'),
            Patch(facecolor='#F38181', label='LH'),
            Patch(facecolor='#95A5A6', label='Other')
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

        ax.set_title('GRN Downstream Connectivity Network', fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")

        return str(output_path)

    def plot_heatmap_connectivity(
        self,
        connectivity_dict: Dict[int, Dict[str, Any]],
        grn_df: pd.DataFrame,
        top_n_partners: int = 30,
        output_filename: str = 'grn_connectivity_heatmap.png'
    ) -> str:
        """
        Generate Figure 4: Heatmap with hierarchical clustering.

        Args:
            connectivity_dict: Connectivity data
            grn_df: GRN metadata
            top_n_partners: Number of top partners to display
            output_filename: Output filename

        Returns:
            Path to saved figure
        """
        logger.info(f"Generating connectivity heatmap (top {top_n_partners} partners)...")

        fig_size = self.config['visualization']['figure_sizes']['heatmap']

        # Build connectivity matrix
        # First, identify top partners by total convergence
        partner_totals = {}
        for conn in connectivity_dict.values():
            for i, partner_id in enumerate(conn['partners']):
                if partner_id not in partner_totals:
                    partner_totals[partner_id] = 0
                partner_totals[partner_id] += conn['synapse_counts'][i]

        top_partners = sorted(partner_totals.items(), key=lambda x: x[1], reverse=True)[:top_n_partners]
        top_partner_ids = [p[0] for p in top_partners]

        # Build matrix: rows = GRNs, columns = partners
        grn_ids = sorted(connectivity_dict.keys())
        matrix = np.zeros((len(grn_ids), len(top_partner_ids)))

        for i, grn_id in enumerate(grn_ids):
            conn = connectivity_dict[grn_id]
            for j, partner_id in enumerate(top_partner_ids):
                if partner_id in conn['partners']:
                    idx = conn['partners'].index(partner_id)
                    matrix[i, j] = conn['synapse_counts'][idx]

        # Log transform if configured
        if self.log_scale:
            matrix_plot = np.log10(matrix + 1)
            cbar_label = 'Synapse Count (log10)'
        else:
            matrix_plot = matrix
            cbar_label = 'Synapse Count'

        # Create DataFrame for better labeling
        matrix_df = pd.DataFrame(
            matrix_plot,
            index=[f"GRN_{gid}" for gid in grn_ids],
            columns=[f"P_{pid}" for pid in top_partner_ids]
        )

        # Cluster if configured
        cluster_rows = self.config['visualization']['heatmap']['cluster_rows']
        cluster_cols = self.config['visualization']['heatmap']['cluster_cols']

        # Create clustermap
        g = sns.clustermap(
            matrix_df,
            cmap=self.heatmap_cmap,
            figsize=fig_size,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            cbar_kws={'label': cbar_label},
            linewidths=0.5,
            linecolor='gray',
            dendrogram_ratio=0.1,
            yticklabels=True,
            xticklabels=True
        )

        g.fig.suptitle('GRN to Partner Connectivity Heatmap', fontsize=14, fontweight='bold', y=0.98)

        # Adjust tick label sizes
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)

        # Save figure
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")

        return str(output_path)

    def plot_analysis_dashboard(
        self,
        connectivity_dict: Dict[int, Dict[str, Any]],
        stats_dict: Dict[str, Any],
        grn_df: pd.DataFrame,
        output_filename: str = 'grn_analysis_dashboard.png'
    ) -> str:
        """
        Generate Figure 5: Statistical summary dashboard.

        Args:
            connectivity_dict: Connectivity data
            stats_dict: Statistics dictionary from analyzer
            grn_df: GRN metadata
            output_filename: Output filename

        Returns:
            Path to saved figure
        """
        logger.info("Generating analysis dashboard...")

        fig_size = self.config['visualization']['figure_sizes']['dashboard']
        fig = plt.figure(figsize=fig_size)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Collect data
        partner_counts = [len(c['partners']) for c in connectivity_dict.values()]
        all_synapses = np.concatenate([c['synapse_counts'] for c in connectivity_dict.values()])
        partner_labels = []
        for c in connectivity_dict.values():
            partner_labels.extend(c['partner_labels'])

        # Plot 1: Histogram of partner counts
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(partner_counts, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Number of Partners')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Partner Counts', fontweight='bold')
        ax1.grid(alpha=0.3)

        # Plot 2: Histogram of synapse counts
        ax2 = fig.add_subplot(gs[0, 1])
        if self.log_scale:
            bins = np.logspace(0, np.log10(max(all_synapses) + 1), 30)
            ax2.hist(all_synapses, bins=bins, color='#FF6B6B', alpha=0.7, edgecolor='black')
            ax2.set_xscale('log')
            ax2.set_xlabel('Synapse Count (log scale)')
        else:
            ax2.hist(all_synapses, bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Synapse Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Synapse Counts', fontweight='bold')
        ax2.grid(alpha=0.3)

        # Plot 3: Box plot by category
        ax3 = fig.add_subplot(gs[0, 2])
        category_data = []
        categories = []
        for grn_id, conn in connectivity_dict.items():
            cat = grn_df[grn_df['root_id'] == grn_id]['grn_category'].values
            cat = cat[0] if len(cat) > 0 else 'unknown'
            category_data.append(len(conn['partners']))
            categories.append(cat)

        cat_df = pd.DataFrame({'Partners': category_data, 'Category': categories})
        sns.boxplot(data=cat_df, x='Category', y='Partners', ax=ax3, palette='Set2')
        ax3.set_xlabel('GRN Category')
        ax3.set_ylabel('Number of Partners')
        ax3.set_title('Partners by GRN Category', fontweight='bold')
        ax3.grid(alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # Plot 4: Cumulative distribution of partners
        ax4 = fig.add_subplot(gs[1, :])
        sorted_partners = np.sort(partner_counts)
        cumulative = np.arange(1, len(sorted_partners) + 1) / len(sorted_partners)
        ax4.plot(sorted_partners, cumulative, linewidth=2, color='#4ECDC4')
        ax4.fill_between(sorted_partners, 0, cumulative, alpha=0.3, color='#4ECDC4')
        ax4.set_xlabel('Number of Partners')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution of Partner Counts', fontweight='bold')
        ax4.grid(alpha=0.3)

        # Plot 5: Pie chart of partner types
        ax5 = fig.add_subplot(gs[2, 0])
        label_counts = pd.Series(partner_labels).value_counts()
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3', '#F38181', '#95A5A6']
        ax5.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
               colors=colors_pie[:len(label_counts)], startangle=90)
        ax5.set_title('Partner Type Distribution', fontweight='bold')

        # Plot 6: Statistics text box
        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.axis('off')

        pop = stats_dict['population']
        partner = stats_dict['partner_analysis']
        conv = stats_dict['convergence']

        stats_text = f"""
CONNECTIVITY STATISTICS

GRNs analyzed: {pop['n_grns']}
Total connections: {pop['total_connections']:,}
Total synapses: {pop['total_synapses']:,}

Partners per GRN:
  Mean: {pop['partners_per_grn']['mean']:.1f}
  Median: {pop['partners_per_grn']['median']:.0f}
  Range: [{pop['partners_per_grn']['min']:.0f}, {pop['partners_per_grn']['max']:.0f}]

Synapses per connection:
  Mean: {pop['synapses_per_connection']['mean']:.1f}
  Median: {pop['synapses_per_connection']['median']:.0f}

Unique downstream partners: {partner['unique_partners']}

Convergence:
  Convergent partners: {conv['n_convergent_partners']}
  Max convergence: {conv['max_convergence']} GRNs
        """

        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('GRN Downstream Connectivity: Analysis Dashboard',
                    fontsize=16, fontweight='bold', y=0.995)

        # Save figure
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")

        return str(output_path)


def main():
    """Example usage of the visualizer."""
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

    # Initialize visualizer
    viz = GRNConnectivityVisualizer(config)

    # Load cached data
    cache_dir = Path(config['data']['cache_dir'])
    conn_path = cache_dir / 'downstream_connectivity.pkl'
    sugar_path = cache_dir / 'sugar_grns_metadata.csv'

    if not conn_path.exists() or not sugar_path.exists():
        logger.error("Required cache files not found. Run extractor and analyzer first.")
        return

    with open(conn_path, 'rb') as f:
        connectivity_dict = pickle.load(f)

    grn_df = pd.read_csv(sugar_path)

    # Filter to sugar GRNs
    sugar_ids = set(grn_df['root_id'])
    sugar_connectivity = {k: v for k, v in connectivity_dict.items() if k in sugar_ids}

    print("\n" + "="*80)
    print("PHASE 4: GENERATE VISUALIZATIONS")
    print("="*80)

    # Generate all figures
    viz.plot_grn_summary(sugar_connectivity, grn_df)
    # viz.plot_comparison() # Needs comparison_df
    viz.plot_network_graph(sugar_connectivity, grn_df)
    viz.plot_heatmap_connectivity(sugar_connectivity, grn_df)
    # viz.plot_analysis_dashboard() # Needs stats_dict

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
