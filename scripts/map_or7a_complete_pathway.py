"""
Complete OR7a Circuit Pathway Mapping: OR7a → PN → KC → MBON → Behavior

This script traces the complete olfactory learning pathway from OR7a olfactory
receptor neurons through projection neurons, Kenyon cells, mushroom body output
neurons, and ultimately to behavioral control circuits.

Multi-level circuit architecture:
    OR7a (41) → DL5_adPN (2) → KC (~200-500) → MBON (~20-50) → Behavior

Usage:
    # Full pathway analysis
    python scripts/map_or7a_complete_pathway.py --data-source local

    # Focus on specific levels
    python scripts/map_or7a_complete_pathway.py --max-level 3

    # Custom output
    python scripts/map_or7a_complete_pathway.py --output-dir results/complete_pathway/
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Visualization settings
sns.set_theme(context="talk", style="whitegrid", palette="deep")

# Default paths
DEFAULT_OR7A_DATA = Path("data/flywire/search_results_or7a.csv")
DEFAULT_CONNECTIONS_DATA = Path("data/flywire/connections_princeton.csv.gz")
DEFAULT_CELL_TYPES_DATA = Path("data/flywire/consolidated_cell_types.csv.gz")
DEFAULT_OUTPUT_DIR = Path("results/or7a_complete_pathway")

# Known DL5_adPN projection neurons (from previous analysis)
DL5_ADPN_ROOT_IDS = [720575940639080700, 720575940617207200]

# Cell type patterns for circuit levels
CELL_TYPE_PATTERNS = {
    'OR7a': ['ORN_DL5', 'Or7a'],
    'PN': ['DL5_adPN', 'DL5_lPN', 'DL5'],
    'KC': ['KC', 'KCab', 'KCg'],
    'MBON': ['MBON'],
    'Motor': ['DNp', 'DNa', 'DNb', 'MDN'],
    'Descending': ['DN'],
    'Central_Complex': ['FB', 'EB', 'PB', 'NO']
}

# Circuit level definitions
CIRCUIT_LEVELS = {
    0: 'OR7a_ORN',
    1: 'DL5_PN',
    2: 'Kenyon_Cell',
    3: 'MBON',
    4: 'Behavioral_Output'
}


@dataclass
class CircuitLevel:
    """Represents one level of the olfactory circuit."""
    level: int
    name: str
    root_ids: List[int] = field(default_factory=list)
    cell_types: List[str] = field(default_factory=list)
    neuron_count: int = 0
    connections_to_next: Optional[pd.DataFrame] = None
    convergence_ratio: Optional[float] = None
    mean_synapses: Optional[float] = None


@dataclass
class CompletePathwayMapper:
    """
    Maps the complete OR7a olfactory pathway through multiple circuit levels.

    Parameters
    ----------
    or7a_data_path : Path
        Path to OR7a neuron CSV file
    data_source : str
        Either 'local' (use local connection files) or 'api' (query FlyWire API)
    connections_path : Optional[Path]
        Path to local connections CSV
    cell_types_path : Optional[Path]
        Path to local cell types CSV
    output_dir : Path
        Directory for output files
    min_synapses : int
        Minimum synapse threshold for connections
    max_levels : int
        Maximum circuit levels to trace (1-5)
    """

    or7a_data_path: Path = DEFAULT_OR7A_DATA
    data_source: str = "local"
    connections_path: Optional[Path] = None
    cell_types_path: Optional[Path] = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_synapses: int = 3
    max_levels: int = 5

    # Internal state
    _connections_df: Optional[pd.DataFrame] = None
    _cell_types_df: Optional[pd.DataFrame] = None
    _circuit_levels: Dict[int, CircuitLevel] = field(default_factory=dict)
    _complete_pathway: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Initialize paths and validate configuration."""
        self.or7a_data_path = Path(self.or7a_data_path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.connections_path is None:
            self.connections_path = DEFAULT_CONNECTIONS_DATA
        if self.cell_types_path is None:
            self.cell_types_path = DEFAULT_CELL_TYPES_DATA

    def load_data_sources(self):
        """Load all required data sources."""
        logger.info("Loading data sources...")

        # Load connections
        self._connections_df = self._load_connections()

        # Load cell types
        self._cell_types_df = self._load_cell_types()

        logger.info(f"Data loaded: {len(self._connections_df):,} connections, "
                   f"{len(self._cell_types_df):,} cell type annotations")

    def _load_connections(self) -> pd.DataFrame:
        """Load connection data."""
        conn_path = Path(self.connections_path)
        if not conn_path.exists():
            raise FileNotFoundError(f"Connections file not found: {conn_path}")

        logger.info(f"Loading connections from {conn_path}")

        # Try standard column names first
        try:
            df = pd.read_csv(
                conn_path,
                usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"],
                dtype={"pre_root_id": "int64", "post_root_id": "int64", "syn_count": "int32"}
            )
        except ValueError:
            # Alternative format
            df = pd.read_csv(conn_path)
            if 'pre_pt_root_id' in df.columns:
                df = df.rename(columns={'pre_pt_root_id': 'pre_root_id'})
            if 'post_pt_root_id' in df.columns:
                df = df.rename(columns={'post_pt_root_id': 'post_root_id'})
            if 'size' in df.columns and 'syn_count' not in df.columns:
                df = df.rename(columns={'size': 'syn_count'})

            keep_cols = ['pre_root_id', 'post_root_id', 'syn_count']
            if 'neuropil' in df.columns:
                keep_cols.append('neuropil')
            df = df[keep_cols]

        logger.info(f"Loaded {len(df):,} connections")
        return df

    def _load_cell_types(self) -> pd.DataFrame:
        """Load cell type annotations."""
        cell_path = Path(self.cell_types_path)

        if not cell_path.exists():
            logger.warning(f"Cell types file not found: {cell_path}")
            return pd.DataFrame(columns=["root_id", "cell_type"])

        logger.info(f"Loading cell types from {cell_path}")

        try:
            df = pd.read_csv(cell_path, usecols=["root_id", "cell_type"])
        except ValueError:
            df = pd.read_csv(cell_path)
            if 'primary_type' in df.columns:
                df = df.rename(columns={'primary_type': 'cell_type'})
            df = df[['root_id', 'cell_type']]

        logger.info(f"Loaded cell type info for {len(df):,} neurons")
        return df

    def identify_cell_category(self, cell_type: str) -> str:
        """Identify which circuit category a cell type belongs to."""
        if pd.isna(cell_type):
            return 'Unknown'

        cell_type_str = str(cell_type)

        for category, patterns in CELL_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in cell_type_str.lower():
                    return category

        return 'Other'

    def get_outputs_for_neurons(
        self,
        root_ids: List[int],
        level_name: str
    ) -> pd.DataFrame:
        """Get all output connections for a set of neurons."""
        logger.info(f"Querying outputs for {len(root_ids)} {level_name} neurons...")

        # Filter connections
        outputs = self._connections_df[
            self._connections_df['pre_root_id'].isin(root_ids)
        ].copy()

        # Apply synapse threshold
        outputs = outputs[outputs['syn_count'] >= self.min_synapses]

        logger.info(f"Found {len(outputs):,} output connections (≥{self.min_synapses} synapses)")

        # Add cell type information
        outputs = outputs.merge(
            self._cell_types_df,
            left_on='post_root_id',
            right_on='root_id',
            how='left',
            suffixes=('', '_target')
        )

        # Add functional categories
        outputs['target_category'] = outputs['cell_type'].apply(self.identify_cell_category)

        return outputs

    def map_circuit_level(
        self,
        level_num: int,
        source_root_ids: List[int],
        level_name: str
    ) -> CircuitLevel:
        """Map one level of the circuit."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Level {level_num}: {level_name}")
        logger.info(f"{'='*80}")

        # Get outputs from source neurons
        outputs = self.get_outputs_for_neurons(source_root_ids, level_name)

        if len(outputs) == 0:
            logger.warning(f"No outputs found for level {level_num}")
            return CircuitLevel(
                level=level_num,
                name=level_name,
                root_ids=[],
                neuron_count=0
            )

        # Get unique target neurons
        target_root_ids = outputs['post_root_id'].unique().tolist()
        target_cell_types = outputs.groupby('post_root_id')['cell_type'].first().tolist()

        # Calculate statistics
        convergence = len(source_root_ids) / len(target_root_ids) if len(target_root_ids) > 0 else 0
        mean_syn = outputs['syn_count'].mean()

        # Show category distribution
        category_dist = outputs.groupby('target_category').agg({
            'post_root_id': 'nunique',
            'syn_count': ['sum', 'mean']
        }).reset_index()
        category_dist.columns = ['category', 'num_neurons', 'total_synapses', 'mean_synapses']
        category_dist = category_dist.sort_values('total_synapses', ascending=False)

        logger.info(f"\nTarget Distribution:")
        logger.info(f"  Total targets: {len(target_root_ids)}")
        logger.info(f"  Convergence ratio: {convergence:.3f}")
        logger.info(f"  Mean synapses/connection: {mean_syn:.1f}")
        logger.info(f"\n  By category:")
        for _, row in category_dist.head(10).iterrows():
            logger.info(f"    {row['category']:20s}: {int(row['num_neurons']):4d} neurons, "
                       f"{int(row['total_synapses']):6d} synapses")

        circuit_level = CircuitLevel(
            level=level_num,
            name=level_name,
            root_ids=target_root_ids,
            cell_types=target_cell_types,
            neuron_count=len(target_root_ids),
            connections_to_next=outputs,
            convergence_ratio=convergence,
            mean_synapses=mean_syn
        )

        return circuit_level

    def trace_complete_pathway(self):
        """Trace the complete OR7a pathway through all levels."""
        logger.info("\n" + "="*80)
        logger.info("COMPLETE OR7a PATHWAY MAPPING")
        logger.info("="*80)

        # Load data
        self.load_data_sources()

        # Level 0: OR7a neurons
        logger.info("\nLevel 0: OR7a Olfactory Receptor Neurons")
        or7a_df = pd.read_csv(self.or7a_data_path)
        or7a_root_ids = or7a_df['root_id'].tolist()

        self._circuit_levels[0] = CircuitLevel(
            level=0,
            name='OR7a_ORN',
            root_ids=or7a_root_ids,
            cell_types=['ORN_DL5'] * len(or7a_root_ids),
            neuron_count=len(or7a_root_ids)
        )
        logger.info(f"  Starting with {len(or7a_root_ids)} OR7a neurons")

        # Level 1: Projection Neurons (DL5_adPN)
        if self.max_levels >= 1:
            level_1 = self.map_circuit_level(
                level_num=1,
                source_root_ids=or7a_root_ids,
                level_name='DL5_Projection_Neurons'
            )
            self._circuit_levels[1] = level_1

            # Filter to actual PNs
            pn_outputs = level_1.connections_to_next
            if pn_outputs is not None:
                pn_mask = pn_outputs['target_category'] == 'PN'
                pn_root_ids = pn_outputs[pn_mask]['post_root_id'].unique().tolist()
                logger.info(f"\n  Identified {len(pn_root_ids)} DL5 projection neurons")

                # Update level 1 to PNs only
                level_1.root_ids = pn_root_ids
                level_1.neuron_count = len(pn_root_ids)
            else:
                pn_root_ids = []

        # Level 2: Kenyon Cells
        if self.max_levels >= 2 and len(pn_root_ids) > 0:
            level_2 = self.map_circuit_level(
                level_num=2,
                source_root_ids=pn_root_ids,
                level_name='Kenyon_Cells'
            )
            self._circuit_levels[2] = level_2

            # Filter to actual KCs
            kc_outputs = level_2.connections_to_next
            if kc_outputs is not None:
                kc_mask = kc_outputs['target_category'] == 'KC'
                kc_root_ids = kc_outputs[kc_mask]['post_root_id'].unique().tolist()
                logger.info(f"\n  Identified {len(kc_root_ids)} Kenyon cells")

                # Update level 2 to KCs only
                level_2.root_ids = kc_root_ids
                level_2.neuron_count = len(kc_root_ids)
            else:
                kc_root_ids = []

        # Level 3: MBONs
        if self.max_levels >= 3 and len(kc_root_ids) > 0:
            level_3 = self.map_circuit_level(
                level_num=3,
                source_root_ids=kc_root_ids,
                level_name='MBONs'
            )
            self._circuit_levels[3] = level_3

            # Filter to actual MBONs
            mbon_outputs = level_3.connections_to_next
            if mbon_outputs is not None:
                mbon_mask = mbon_outputs['target_category'] == 'MBON'
                mbon_root_ids = mbon_outputs[mbon_mask]['post_root_id'].unique().tolist()
                logger.info(f"\n  Identified {len(mbon_root_ids)} MBONs")

                # Update level 3 to MBONs only
                level_3.root_ids = mbon_root_ids
                level_3.neuron_count = len(mbon_root_ids)
            else:
                mbon_root_ids = []

        # Level 4: Behavioral outputs
        if self.max_levels >= 4 and len(mbon_root_ids) > 0:
            level_4 = self.map_circuit_level(
                level_num=4,
                source_root_ids=mbon_root_ids,
                level_name='Behavioral_Outputs'
            )
            self._circuit_levels[4] = level_4

        logger.info("\n" + "="*80)
        logger.info("Pathway mapping complete!")
        logger.info("="*80)

    def build_complete_pathway_dataframe(self) -> pd.DataFrame:
        """Build a single dataframe with all pathway connections."""
        all_connections = []

        for level_num in sorted(self._circuit_levels.keys()):
            level = self._circuit_levels[level_num]

            if level.connections_to_next is not None:
                conns = level.connections_to_next.copy()
                conns['source_level'] = level_num
                conns['source_level_name'] = level.name
                conns['target_level'] = level_num + 1
                conns['target_level_name'] = CIRCUIT_LEVELS.get(level_num + 1, 'Unknown')

                all_connections.append(conns)

        if not all_connections:
            return pd.DataFrame()

        pathway_df = pd.concat(all_connections, ignore_index=True)
        self._complete_pathway = pathway_df

        return pathway_df

    def generate_pathway_summary(self) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive pathway summary statistics."""
        summaries = {}

        # 1. Summary by level
        level_summary = []
        for level_num in sorted(self._circuit_levels.keys()):
            level = self._circuit_levels[level_num]
            level_summary.append({
                'level': level_num,
                'level_name': level.name,
                'neuron_count': level.neuron_count,
                'convergence_ratio': level.convergence_ratio or 0,
                'mean_synapses_per_connection': level.mean_synapses or 0
            })

        summaries['by_level'] = pd.DataFrame(level_summary)

        # 2. Connection statistics between levels
        if self._complete_pathway is not None:
            conn_stats = self._complete_pathway.groupby(['source_level', 'target_level']).agg({
                'pre_root_id': 'nunique',
                'post_root_id': 'nunique',
                'syn_count': ['sum', 'mean', 'median', 'std']
            }).reset_index()
            conn_stats.columns = ['source_level', 'target_level', 'source_neurons',
                                 'target_neurons', 'total_synapses', 'mean_synapses',
                                 'median_synapses', 'std_synapses']
            summaries['connections'] = conn_stats

        # 3. Target category distributions
        if self._complete_pathway is not None:
            category_dist = self._complete_pathway.groupby(['source_level', 'target_category']).agg({
                'post_root_id': 'nunique',
                'syn_count': ['sum', 'mean']
            }).reset_index()
            category_dist.columns = ['source_level', 'category', 'num_neurons',
                                    'total_synapses', 'mean_synapses']
            summaries['categories'] = category_dist

        # 4. Bottleneck analysis
        bottlenecks = []
        for i in range(len(self._circuit_levels) - 1):
            if i in self._circuit_levels and i+1 in self._circuit_levels:
                source_count = self._circuit_levels[i].neuron_count
                target_count = self._circuit_levels[i+1].neuron_count

                if source_count > 0:
                    expansion_ratio = target_count / source_count
                    bottlenecks.append({
                        'transition': f"{CIRCUIT_LEVELS[i]} → {CIRCUIT_LEVELS[i+1]}",
                        'source_neurons': source_count,
                        'target_neurons': target_count,
                        'expansion_ratio': expansion_ratio,
                        'bottleneck_severity': 'High' if expansion_ratio < 0.5 else
                                             ('Medium' if expansion_ratio < 1.0 else 'Low')
                    })

        summaries['bottlenecks'] = pd.DataFrame(bottlenecks)

        return summaries

    def identify_critical_targets(self) -> pd.DataFrame:
        """Identify critical neurons for experimental targeting."""
        if self._complete_pathway is None:
            return pd.DataFrame()

        # Calculate importance scores for each neuron
        neuron_importance = defaultdict(lambda: {'total_synapses': 0, 'connections': 0, 'levels': set()})

        for _, conn in self._complete_pathway.iterrows():
            post_id = conn['post_root_id']
            neuron_importance[post_id]['total_synapses'] += conn['syn_count']
            neuron_importance[post_id]['connections'] += 1
            neuron_importance[post_id]['levels'].add(conn['target_level'])

        # Convert to dataframe
        importance_records = []
        for root_id, stats in neuron_importance.items():
            # Get cell type
            cell_info = self._cell_types_df[self._cell_types_df['root_id'] == root_id]
            cell_type = cell_info['cell_type'].iloc[0] if len(cell_info) > 0 else 'Unknown'

            importance_records.append({
                'root_id': root_id,
                'cell_type': cell_type,
                'category': self.identify_cell_category(cell_type),
                'total_synapses': stats['total_synapses'],
                'num_connections': stats['connections'],
                'num_levels': len(stats['levels']),
                'importance_score': stats['total_synapses'] * stats['connections']
            })

        targets_df = pd.DataFrame(importance_records)
        targets_df = targets_df.sort_values('importance_score', ascending=False)

        return targets_df

    def visualize_complete_pathway(self):
        """Create comprehensive pathway visualization."""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # 1. Pathway diagram
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_pathway_diagram(ax1)

        # 2. Neuron counts by level
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_neuron_counts(ax2)

        # 3. Connection strengths
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_connection_strengths(ax3)

        # 4. Category distribution
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_category_distribution(ax4)

        # 5. Convergence/divergence
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_convergence(ax5)

        # 6. Synapse distribution
        ax6 = fig.add_subplot(gs[2, 1:])
        self._plot_synapse_distribution(ax6)

        plt.suptitle('Complete OR7a Pathway Analysis: ORN → PN → KC → MBON → Behavior',
                    fontsize=16, fontweight='bold')

        output_path = self.output_dir / 'or7a_complete_pathway_analysis.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved pathway visualization to {output_path}")

        return fig

    def _plot_pathway_diagram(self, ax):
        """Plot simplified pathway diagram."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')

        # Plot each level
        level_positions = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

        for level_num, level in self._circuit_levels.items():
            x = level_positions.get(level_num, level_num * 2)
            y = 3

            # Draw box
            box = FancyBboxPatch(
                (x - 0.4, y - 0.5), 0.8, 1,
                boxstyle="round,pad=0.1",
                facecolor=colors[level_num % len(colors)],
                edgecolor='black',
                linewidth=2,
                alpha=0.7
            )
            ax.add_patch(box)

            # Add label
            ax.text(x, y + 0.8, level.name.replace('_', '\n'),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(x, y, f'n={level.neuron_count}',
                   ha='center', va='center', fontsize=12, fontweight='bold')

            # Draw connection arrow
            if level_num < len(self._circuit_levels) - 1 and level_num + 1 in level_positions:
                x_next = level_positions[level_num + 1]
                ax.annotate('', xy=(x_next - 0.4, y), xytext=(x + 0.4, y),
                          arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

                # Add synapse count if available
                if level.mean_synapses is not None:
                    mid_x = (x + x_next) / 2
                    ax.text(mid_x, y - 0.3, f'~{level.mean_synapses:.0f} syn',
                           ha='center', va='top', fontsize=8, style='italic')

        ax.set_title('OR7a Circuit Architecture', fontsize=14, fontweight='bold', pad=20)

    def _plot_neuron_counts(self, ax):
        """Plot neuron counts at each level."""
        levels = []
        counts = []
        names = []

        for level_num in sorted(self._circuit_levels.keys()):
            level = self._circuit_levels[level_num]
            levels.append(level_num)
            counts.append(level.neuron_count)
            names.append(level.name.replace('_', '\n'))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        ax.bar(levels, counts, color=[colors[i % len(colors)] for i in levels], alpha=0.7)

        ax.set_xlabel('Circuit Level')
        ax.set_ylabel('Number of Neurons')
        ax.set_title('Neuron Count by Level', fontweight='bold')
        ax.set_xticks(levels)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

    def _plot_connection_strengths(self, ax):
        """Plot connection strengths between levels."""
        if self._complete_pathway is None or len(self._complete_pathway) == 0:
            ax.text(0.5, 0.5, 'No connection data', ha='center', va='center')
            return

        conn_summary = self._complete_pathway.groupby('source_level')['syn_count'].agg(['mean', 'std', 'median'])

        x = conn_summary.index
        means = conn_summary['mean']
        stds = conn_summary['std']

        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xlabel('Source Level')
        ax.set_ylabel('Mean Synapses per Connection')
        ax.set_title('Connection Strength by Level', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    def _plot_category_distribution(self, ax):
        """Plot target category distribution."""
        if self._complete_pathway is None or len(self._complete_pathway) == 0:
            ax.text(0.5, 0.5, 'No category data', ha='center', va='center')
            return

        category_counts = self._complete_pathway.groupby('target_category')['post_root_id'].nunique()
        category_counts = category_counts.sort_values(ascending=False).head(8)

        category_counts.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Number of Neurons')
        ax.set_ylabel('Cell Category')
        ax.set_title('Target Category Distribution', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    def _plot_convergence(self, ax):
        """Plot convergence/divergence ratios."""
        levels = []
        ratios = []

        for level_num in sorted(self._circuit_levels.keys())[:-1]:
            if level_num in self._circuit_levels and level_num + 1 in self._circuit_levels:
                source = self._circuit_levels[level_num].neuron_count
                target = self._circuit_levels[level_num + 1].neuron_count

                if source > 0:
                    ratio = target / source
                    levels.append(f"{level_num}→{level_num+1}")
                    ratios.append(ratio)

        colors = ['green' if r > 1 else 'red' for r in ratios]
        ax.barh(levels, ratios, color=colors, alpha=0.6)
        ax.axvline(1, color='black', linestyle='--', linewidth=2, label='1:1')
        ax.set_xlabel('Target/Source Ratio')
        ax.set_ylabel('Level Transition')
        ax.set_title('Convergence (red) vs Divergence (green)', fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

    def _plot_synapse_distribution(self, ax):
        """Plot synapse count distribution across all levels."""
        if self._complete_pathway is None or len(self._complete_pathway) == 0:
            ax.text(0.5, 0.5, 'No synapse data', ha='center', va='center')
            return

        for level_num in sorted(self._complete_pathway['source_level'].unique()):
            level_data = self._complete_pathway[self._complete_pathway['source_level'] == level_num]
            level_name = CIRCUIT_LEVELS.get(level_num, f'Level {level_num}')

            ax.hist(level_data['syn_count'], bins=30, alpha=0.5, label=level_name, histtype='step', linewidth=2)

        ax.set_xlabel('Synapses per Connection')
        ax.set_ylabel('Count')
        ax.set_title('Synapse Distribution by Level', fontweight='bold')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(alpha=0.3)

    def run_complete_analysis(self):
        """Execute complete pathway analysis pipeline."""
        logger.info("\n" + "="*80)
        logger.info("COMPLETE OR7a PATHWAY ANALYSIS")
        logger.info("="*80)

        # 1. Trace pathway
        self.trace_complete_pathway()

        # 2. Build complete pathway dataframe
        logger.info("\nBuilding complete pathway dataframe...")
        pathway_df = self.build_complete_pathway_dataframe()

        if len(pathway_df) > 0:
            pathway_path = self.output_dir / 'or7a_complete_pathway.csv'
            pathway_df.to_csv(pathway_path, index=False)
            logger.info(f"Saved complete pathway to {pathway_path}")

        # 3. Generate summaries
        logger.info("\nGenerating pathway summaries...")
        summaries = self.generate_pathway_summary()

        for name, summary_df in summaries.items():
            summary_path = self.output_dir / f'pathway_summary_{name}.csv'
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"  - {name}: {summary_path}")

        # 4. Identify critical targets
        logger.info("\nIdentifying critical targets...")
        targets_df = self.identify_critical_targets()

        if len(targets_df) > 0:
            targets_path = self.output_dir / 'target_priorities.csv'
            targets_df.to_csv(targets_path, index=False)
            logger.info(f"Saved target priorities to {targets_path}")

            logger.info("\n  Top 10 critical targets:")
            for idx, row in targets_df.head(10).iterrows():
                logger.info(f"    {row['cell_type']:30s} ({row['category']:10s}): "
                          f"score={row['importance_score']:.0f}")

        # 5. Create visualizations
        logger.info("\nCreating pathway visualizations...")
        self.visualize_complete_pathway()

        # 6. Print final summary
        logger.info("\n" + "="*80)
        logger.info("PATHWAY SUMMARY")
        logger.info("="*80)

        for level_num in sorted(self._circuit_levels.keys()):
            level = self._circuit_levels[level_num]
            logger.info(f"\nLevel {level_num} - {level.name}:")
            logger.info(f"  Neurons: {level.neuron_count}")
            if level.convergence_ratio is not None:
                logger.info(f"  Convergence: {level.convergence_ratio:.3f}")
            if level.mean_synapses is not None:
                logger.info(f"  Mean synapses: {level.mean_synapses:.1f}")

        logger.info("\n" + "="*80)
        logger.info(f"Analysis complete! All outputs in: {self.output_dir}")
        logger.info("="*80)

        return {
            'pathway': pathway_df,
            'summaries': summaries,
            'targets': targets_df
        }


def main():
    """Command-line interface for complete pathway mapping."""
    parser = argparse.ArgumentParser(
        description="Map complete OR7a circuit pathway from ORNs to behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data-source',
        choices=['local', 'api'],
        default='local',
        help='Data source for FlyWire connections'
    )

    parser.add_argument(
        '--or7a-data',
        type=Path,
        default=DEFAULT_OR7A_DATA,
        help='Path to OR7a neurons CSV'
    )

    parser.add_argument(
        '--connections',
        type=Path,
        default=DEFAULT_CONNECTIONS_DATA,
        help='Path to connections CSV'
    )

    parser.add_argument(
        '--cell-types',
        type=Path,
        default=DEFAULT_CELL_TYPES_DATA,
        help='Path to cell types CSV'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results'
    )

    parser.add_argument(
        '--min-synapses',
        type=int,
        default=3,
        help='Minimum synapse threshold'
    )

    parser.add_argument(
        '--max-levels',
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help='Maximum circuit levels to trace'
    )

    args = parser.parse_args()

    # Create mapper
    mapper = CompletePathwayMapper(
        or7a_data_path=args.or7a_data,
        data_source=args.data_source,
        connections_path=args.connections,
        cell_types_path=args.cell_types,
        output_dir=args.output_dir,
        min_synapses=args.min_synapses,
        max_levels=args.max_levels
    )

    # Run analysis
    results = mapper.run_complete_analysis()

    return results


if __name__ == "__main__":
    main()
