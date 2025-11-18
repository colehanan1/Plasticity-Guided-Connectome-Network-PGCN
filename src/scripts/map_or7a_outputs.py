"""
Map OR7a neuron output targets from FlyWire FAFB v783 connectome data.

This script analyzes the downstream connectivity of OR7a (ORN_DL5) neurons,
identifying their synaptic targets, target cell types, neuropils, and synapse counts.
It generates comprehensive CSV outputs and visualizations for understanding OR7a
output patterns across the population.

Usage:
    python scripts/map_or7a_outputs.py --data-source local
    python scripts/map_or7a_outputs.py --data-source api --top-targets 20
    python scripts/map_or7a_outputs.py --output-dir results/or7a_outputs/
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Project imports - use try/except for graceful degradation
try:
    from caveclient import CAVEclient
    CAVECLIENT_AVAILABLE = True
except ImportError:
    CAVECLIENT_AVAILABLE = False
    CAVEclient = None

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
DEFAULT_OUTPUT_DIR = Path("results/or7a_outputs")

# FlyWire configuration
FLYWIRE_DATASTACK = "flywire_fafb_production"
FLYWIRE_VERSION = 783  # FAFB v783


@dataclass
class OR7aOutputMapper:
    """
    Maps output targets of OR7a neurons using FlyWire connectivity data.

    Parameters
    ----------
    or7a_data_path : Path
        Path to OR7a neuron CSV file
    data_source : str
        Either 'local' (use local connection files) or 'api' (query FlyWire API)
    connections_path : Optional[Path]
        Path to local connections CSV (required if data_source='local')
    cell_types_path : Optional[Path]
        Path to local cell types CSV
    output_dir : Path
        Directory for output files
    min_synapses : int
        Minimum synapse threshold for including a connection
    top_n_targets : int
        Number of top targets to include per OR7a neuron
    """

    or7a_data_path: Path = DEFAULT_OR7A_DATA
    data_source: str = "local"
    connections_path: Optional[Path] = None
    cell_types_path: Optional[Path] = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_synapses: int = 3
    top_n_targets: int = 20

    # Internal state
    _or7a_df: Optional[pd.DataFrame] = None
    _connections_df: Optional[pd.DataFrame] = None
    _cell_types_df: Optional[pd.DataFrame] = None
    _output_targets_df: Optional[pd.DataFrame] = None
    _caveclient: Optional[object] = None

    def __post_init__(self):
        """Initialize paths and validate configuration."""
        self.or7a_data_path = Path(self.or7a_data_path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.data_source not in ["local", "api"]:
            raise ValueError("data_source must be 'local' or 'api'")

        if self.data_source == "api" and not CAVECLIENT_AVAILABLE:
            raise ImportError(
                "CAVEclient is required for API data source. "
                "Install with: pip install caveclient"
            )

    def load_or7a_data(self) -> pd.DataFrame:
        """Load OR7a neuron data from CSV."""
        if self._or7a_df is not None:
            return self._or7a_df

        logger.info(f"Loading OR7a data from {self.or7a_data_path}")

        if not self.or7a_data_path.exists():
            raise FileNotFoundError(f"OR7a data not found: {self.or7a_data_path}")

        df = pd.read_csv(self.or7a_data_path)

        # Validate required columns
        required = ["root_id", "name", "side", "input_synapses", "output_synapses"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Loaded {len(df)} OR7a neurons")
        logger.info(f"  Left hemisphere: {(df['side'] == 'left').sum()}")
        logger.info(f"  Right hemisphere: {(df['side'] == 'right').sum()}")

        self._or7a_df = df
        return df

    def load_local_connections(self) -> pd.DataFrame:
        """Load connection data from local CSV file."""
        if self._connections_df is not None:
            return self._connections_df

        conn_path = self.connections_path or DEFAULT_CONNECTIONS_DATA
        logger.info(f"Loading connections from {conn_path}")

        if not conn_path.exists():
            raise FileNotFoundError(
                f"Connections file not found: {conn_path}\n"
                "Please provide connections data or use data_source='api'"
            )

        # Load connections efficiently - check which columns are available
        # Princeton format may use different column names
        try:
            df = pd.read_csv(
                conn_path,
                usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"],
                dtype={"pre_root_id": "int64", "post_root_id": "int64", "syn_count": "int32"},
                nrows=0  # Just check columns first
            )
            df = pd.read_csv(
                conn_path,
                usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"],
                dtype={"pre_root_id": "int64", "post_root_id": "int64", "syn_count": "int32"}
            )
        except ValueError:
            # Try alternative column names
            logger.info("Trying alternative column names for connections file...")
            df = pd.read_csv(conn_path, nrows=5)
            logger.info(f"Available columns: {list(df.columns)}")

            # Read full file with available columns
            df = pd.read_csv(conn_path)

            # Standardize column names if needed
            rename_map = {}
            if 'pre_pt_root_id' in df.columns:
                rename_map['pre_pt_root_id'] = 'pre_root_id'
            if 'post_pt_root_id' in df.columns:
                rename_map['post_pt_root_id'] = 'post_root_id'
            if 'size' in df.columns and 'syn_count' not in df.columns:
                rename_map['size'] = 'syn_count'

            if rename_map:
                logger.info(f"Renaming columns: {rename_map}")
                df = df.rename(columns=rename_map)

            # Keep only required columns
            keep_cols = ['pre_root_id', 'post_root_id', 'syn_count']
            if 'neuropil' in df.columns:
                keep_cols.append('neuropil')
            df = df[keep_cols]

        logger.info(f"Loaded {len(df):,} connections")
        self._connections_df = df
        return df

    def load_local_cell_types(self) -> pd.DataFrame:
        """Load cell type annotations from local CSV file."""
        if self._cell_types_df is not None:
            return self._cell_types_df

        cell_path = self.cell_types_path or DEFAULT_CELL_TYPES_DATA

        if not cell_path.exists():
            logger.warning(f"Cell types file not found: {cell_path}")
            logger.warning("Cell type information will be limited")
            return pd.DataFrame(columns=["root_id", "cell_type", "super_class", "sub_class"])

        logger.info(f"Loading cell types from {cell_path}")

        # Try to load with standard column names
        try:
            df = pd.read_csv(cell_path, usecols=["root_id", "cell_type", "super_class", "sub_class"])
        except ValueError:
            # Try alternative column names (consolidated format uses 'primary_type')
            logger.info("Using consolidated cell types format...")
            df = pd.read_csv(cell_path)

            # Rename to standard column names
            if 'primary_type' in df.columns and 'cell_type' not in df.columns:
                df = df.rename(columns={'primary_type': 'cell_type'})

            # Keep only needed columns
            keep_cols = ['root_id', 'cell_type']
            if 'super_class' in df.columns:
                keep_cols.append('super_class')
            if 'sub_class' in df.columns:
                keep_cols.append('sub_class')

            df = df[keep_cols]

        logger.info(f"Loaded cell type info for {len(df):,} neurons")
        self._cell_types_df = df
        return df

    def initialize_caveclient(self) -> object:
        """Initialize FlyWire CAVEclient for API queries."""
        if self._caveclient is not None:
            return self._caveclient

        if not CAVECLIENT_AVAILABLE:
            raise ImportError("CAVEclient not available")

        logger.info(f"Initializing CAVEclient for {FLYWIRE_DATASTACK}")
        try:
            client = CAVEclient(FLYWIRE_DATASTACK)
            client.materialize.version = FLYWIRE_VERSION
            self._caveclient = client
            logger.info(f"CAVEclient initialized (version {FLYWIRE_VERSION})")
            return client
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CAVEclient: {e}")

    def query_outputs_local(self, or7a_root_ids: List[int]) -> pd.DataFrame:
        """Query output targets using local connection data."""
        connections = self.load_local_connections()
        cell_types = self.load_local_cell_types()

        # Filter to OR7a outputs
        or7a_outputs = connections[
            connections["pre_root_id"].isin(or7a_root_ids)
        ].copy()

        # Filter by minimum synapses
        or7a_outputs = or7a_outputs[or7a_outputs["syn_count"] >= self.min_synapses]

        logger.info(f"Found {len(or7a_outputs):,} OR7a output connections")

        # Merge with cell type information
        or7a_outputs = or7a_outputs.merge(
            cell_types,
            left_on="post_root_id",
            right_on="root_id",
            how="left",
            suffixes=("", "_target")
        )

        return or7a_outputs

    def query_outputs_api(self, or7a_root_ids: List[int]) -> pd.DataFrame:
        """Query output targets using FlyWire API."""
        client = self.initialize_caveclient()

        all_outputs = []

        logger.info("Querying FlyWire API for OR7a outputs...")
        for root_id in tqdm(or7a_root_ids, desc="Querying outputs"):
            try:
                # Query synapses from this OR7a neuron
                synapses = client.materialize.synapse_query(
                    pre_ids=root_id,
                    materialization_version=FLYWIRE_VERSION
                )

                if len(synapses) == 0:
                    continue

                # Aggregate by postsynaptic partner
                synapse_df = pd.DataFrame(synapses)
                grouped = synapse_df.groupby('post_pt_root_id').agg({
                    'id': 'count',  # synapse count
                    'neuropil': lambda x: x.mode()[0] if len(x) > 0 else None
                }).reset_index()

                grouped.columns = ['post_root_id', 'syn_count', 'neuropil']
                grouped['pre_root_id'] = root_id

                all_outputs.append(grouped)

            except Exception as e:
                logger.warning(f"Failed to query outputs for {root_id}: {e}")
                continue

        if not all_outputs:
            raise RuntimeError("No output data retrieved from API")

        outputs_df = pd.concat(all_outputs, ignore_index=True)

        # Filter by minimum synapses
        outputs_df = outputs_df[outputs_df["syn_count"] >= self.min_synapses]

        logger.info(f"Retrieved {len(outputs_df):,} OR7a output connections from API")

        # Query cell type information for targets
        unique_targets = outputs_df['post_root_id'].unique()
        logger.info(f"Querying cell types for {len(unique_targets)} unique targets...")

        try:
            cell_info = client.materialize.query_table(
                'neuron_information_v2',
                filter_in_dict={'pt_root_id': unique_targets.tolist()}
            )
            cell_df = pd.DataFrame(cell_info)

            outputs_df = outputs_df.merge(
                cell_df[['pt_root_id', 'cell_type', 'super_class', 'sub_class']],
                left_on='post_root_id',
                right_on='pt_root_id',
                how='left'
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve cell types: {e}")

        return outputs_df

    def map_all_outputs(self) -> pd.DataFrame:
        """Map outputs for all OR7a neurons."""
        or7a_df = self.load_or7a_data()
        or7a_root_ids = or7a_df['root_id'].tolist()

        logger.info(f"Mapping outputs for {len(or7a_root_ids)} OR7a neurons")

        # Query outputs based on data source
        if self.data_source == "local":
            outputs_df = self.query_outputs_local(or7a_root_ids)
        else:
            outputs_df = self.query_outputs_api(or7a_root_ids)

        # Merge with OR7a metadata
        outputs_df = outputs_df.merge(
            or7a_df[['root_id', 'name', 'side', 'output_synapses']],
            left_on='pre_root_id',
            right_on='root_id',
            how='left',
            suffixes=('_target', '_or7a')
        )

        # Rename for clarity
        outputs_df = outputs_df.rename(columns={
            'pre_root_id': 'or7a_root_id',
            'name': 'or7a_name',
            'side': 'or7a_side',
            'output_synapses': 'or7a_total_outputs',
            'post_root_id': 'target_root_id',
            'syn_count': 'synapse_count',
            'neuropil': 'target_neuropil',
            'cell_type': 'target_cell_type',
            'super_class': 'target_super_class',
            'sub_class': 'target_sub_class'
        })

        # Sort by OR7a neuron and synapse count
        outputs_df = outputs_df.sort_values(
            ['or7a_root_id', 'synapse_count'],
            ascending=[True, False]
        ).reset_index(drop=True)

        self._output_targets_df = outputs_df
        logger.info(f"Mapped {len(outputs_df)} output connections")

        return outputs_df

    def create_wide_format_csv(self) -> pd.DataFrame:
        """
        Create wide-format CSV with top N targets per OR7a neuron.

        Format:
        or7a_root_id, or7a_name, or7a_side, or7a_total_outputs,
        target_1_root_id, target_1_cell_type, target_1_neuropil, target_1_synapse_count,
        target_2_root_id, target_2_cell_type, target_2_neuropil, target_2_synapse_count,
        ...
        total_targets_found, primary_target_type, main_output_neuropil
        """
        if self._output_targets_df is None:
            self.map_all_outputs()

        outputs = self._output_targets_df
        or7a_df = self.load_or7a_data()

        wide_rows = []

        logger.info(f"Creating wide-format CSV with top {self.top_n_targets} targets per neuron")

        for or7a_id in tqdm(or7a_df['root_id'], desc="Formatting output"):
            neuron_outputs = outputs[outputs['or7a_root_id'] == or7a_id].copy()
            neuron_outputs = neuron_outputs.sort_values('synapse_count', ascending=False)

            # Base information
            or7a_info = or7a_df[or7a_df['root_id'] == or7a_id].iloc[0]
            row = {
                'or7a_root_id': or7a_id,
                'or7a_name': or7a_info['name'],
                'or7a_side': or7a_info['side'],
                'or7a_total_outputs': or7a_info['output_synapses']
            }

            # Add top N targets
            for i, (_, target) in enumerate(neuron_outputs.head(self.top_n_targets).iterrows(), 1):
                row[f'target_{i}_root_id'] = target['target_root_id']
                row[f'target_{i}_cell_type'] = target.get('target_cell_type', 'Unknown')
                row[f'target_{i}_neuropil'] = target.get('target_neuropil', 'Unknown')
                row[f'target_{i}_synapse_count'] = target['synapse_count']

            # Summary statistics
            row['total_targets_found'] = len(neuron_outputs)

            if len(neuron_outputs) > 0:
                # Primary target type (most common)
                type_counts = neuron_outputs['target_cell_type'].value_counts()
                row['primary_target_type'] = type_counts.index[0] if len(type_counts) > 0 else 'Unknown'

                # Main output neuropil
                neuropil_counts = neuron_outputs['target_neuropil'].value_counts()
                row['main_output_neuropil'] = neuropil_counts.index[0] if len(neuropil_counts) > 0 else 'Unknown'
            else:
                row['primary_target_type'] = 'None'
                row['main_output_neuropil'] = 'None'

            wide_rows.append(row)

        wide_df = pd.DataFrame(wide_rows)
        return wide_df

    def generate_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive summary statistics."""
        if self._output_targets_df is None:
            self.map_all_outputs()

        outputs = self._output_targets_df
        summaries = {}

        # 1. Overall connectivity summary
        overall = pd.DataFrame([{
            'total_or7a_neurons': outputs['or7a_root_id'].nunique(),
            'total_target_neurons': outputs['target_root_id'].nunique(),
            'total_connections': len(outputs),
            'mean_targets_per_or7a': len(outputs) / outputs['or7a_root_id'].nunique(),
            'median_synapses_per_connection': outputs['synapse_count'].median(),
            'mean_synapses_per_connection': outputs['synapse_count'].mean()
        }])
        summaries['overall'] = overall

        # 2. Target cell type distribution
        type_summary = outputs.groupby('target_cell_type').agg({
            'target_root_id': 'nunique',
            'synapse_count': ['sum', 'mean', 'median']
        }).reset_index()
        type_summary.columns = ['cell_type', 'num_neurons', 'total_synapses', 'mean_synapses', 'median_synapses']
        type_summary = type_summary.sort_values('total_synapses', ascending=False)
        summaries['target_cell_types'] = type_summary

        # 3. Neuropil distribution
        neuropil_summary = outputs.groupby('target_neuropil').agg({
            'target_root_id': 'count',
            'synapse_count': ['sum', 'mean']
        }).reset_index()
        neuropil_summary.columns = ['neuropil', 'num_connections', 'total_synapses', 'mean_synapses']
        neuropil_summary = neuropil_summary.sort_values('total_synapses', ascending=False)
        summaries['target_neuropils'] = neuropil_summary

        # 4. Hemispheric comparison
        hemispheric = outputs.groupby('or7a_side').agg({
            'target_root_id': ['nunique', 'count'],
            'synapse_count': ['sum', 'mean', 'median']
        }).reset_index()
        hemispheric.columns = ['side', 'unique_targets', 'total_connections', 'total_synapses', 'mean_synapses', 'median_synapses']
        summaries['hemispheric'] = hemispheric

        # 5. Per-neuron statistics
        per_neuron = outputs.groupby(['or7a_root_id', 'or7a_name', 'or7a_side']).agg({
            'target_root_id': 'count',
            'synapse_count': ['sum', 'mean', 'max']
        }).reset_index()
        per_neuron.columns = ['or7a_root_id', 'or7a_name', 'or7a_side', 'num_targets', 'total_output_synapses', 'mean_synapses_per_target', 'max_synapses_to_target']
        per_neuron = per_neuron.sort_values('total_output_synapses', ascending=False)
        summaries['per_neuron'] = per_neuron

        return summaries

    def visualize_output_patterns(self):
        """Create comprehensive visualizations of OR7a output patterns."""
        if self._output_targets_df is None:
            self.map_all_outputs()

        outputs = self._output_targets_df

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Target cell type distribution
        ax1 = fig.add_subplot(gs[0, :2])
        type_counts = outputs.groupby('target_cell_type')['synapse_count'].sum().sort_values(ascending=False).head(15)
        type_counts.plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_title('Top 15 Target Cell Types by Total Synapses', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Total Synapses')
        ax1.set_ylabel('Cell Type')

        # 2. Neuropil distribution
        ax2 = fig.add_subplot(gs[0, 2])
        neuropil_counts = outputs.groupby('target_neuropil')['synapse_count'].sum().sort_values(ascending=False).head(10)
        ax2.pie(neuropil_counts.values, labels=neuropil_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Output Neuropil Distribution', fontsize=12, fontweight='bold')

        # 3. Targets per OR7a neuron
        ax3 = fig.add_subplot(gs[1, 0])
        targets_per_or7a = outputs.groupby('or7a_root_id')['target_root_id'].count()
        ax3.hist(targets_per_or7a, bins=20, color='coral', edgecolor='black')
        ax3.set_title('Number of Targets per OR7a Neuron', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Number of Downstream Targets')
        ax3.set_ylabel('Number of OR7a Neurons')
        ax3.axvline(targets_per_or7a.mean(), color='red', linestyle='--', label=f'Mean: {targets_per_or7a.mean():.1f}')
        ax3.legend()

        # 4. Synapse count distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(outputs['synapse_count'], bins=30, color='mediumpurple', edgecolor='black')
        ax4.set_title('Synapse Count Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Synapses per Connection')
        ax4.set_ylabel('Count')
        ax4.set_yscale('log')

        # 5. Hemispheric comparison
        ax5 = fig.add_subplot(gs[1, 2])
        hemis_data = outputs.groupby(['or7a_side', 'or7a_root_id'])['target_root_id'].count().reset_index()
        hemis_data.columns = ['side', 'or7a_root_id', 'num_targets']
        sns.boxplot(
            data=hemis_data,
            x='side',
            y='num_targets',
            hue='side',
            palette='Set2',
            dodge=False,
            ax=ax5,
        )
        if ax5.legend_ is not None:
            ax5.legend_.remove()
        sns.swarmplot(data=hemis_data, x='side', y='num_targets', ax=ax5, color='black', alpha=0.5, size=4)
        ax5.set_title('Hemispheric Comparison', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Hemisphere')
        ax5.set_ylabel('Number of Targets per Neuron')

        # 6. Top individual targets
        ax6 = fig.add_subplot(gs[2, :])
        top_targets = outputs.groupby(['target_root_id', 'target_cell_type'])['synapse_count'].sum().sort_values(ascending=False).head(20)
        labels = [f"{cell_type}\n({root_id})" for root_id, cell_type in top_targets.index]
        ax6.barh(range(len(top_targets)), top_targets.values, color='teal')
        ax6.set_yticks(range(len(top_targets)))
        ax6.set_yticklabels(labels, fontsize=8)
        ax6.set_title('Top 20 Individual Target Neurons', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Total Synapses from All OR7a Neurons')
        ax6.invert_yaxis()

        plt.suptitle('OR7a Output Target Analysis', fontsize=16, fontweight='bold', y=0.995)

        # Save figure
        output_path = self.output_dir / 'or7a_output_analysis.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
        plt.close()

        return fig

    def run_complete_analysis(self):
        """Execute complete analysis pipeline."""
        logger.info("=" * 80)
        logger.info("OR7a Output Target Mapping Analysis")
        logger.info("=" * 80)

        # 1. Map outputs
        logger.info("\n[1/5] Mapping output targets...")
        outputs_df = self.map_all_outputs()

        # 2. Save long-format CSV
        logger.info("\n[2/5] Saving long-format output data...")
        long_path = self.output_dir / 'or7a_output_targets_long.csv'
        outputs_df.to_csv(long_path, index=False)
        logger.info(f"Saved to {long_path}")

        # 3. Create and save wide-format CSV
        logger.info("\n[3/5] Creating wide-format output data...")
        wide_df = self.create_wide_format_csv()
        wide_path = self.output_dir / 'or7a_output_targets_wide.csv'
        wide_df.to_csv(wide_path, index=False)
        logger.info(f"Saved to {wide_path}")

        # 4. Generate summary statistics
        logger.info("\n[4/5] Generating summary statistics...")
        summaries = self.generate_summary_statistics()

        for name, summary_df in summaries.items():
            summary_path = self.output_dir / f'summary_{name}.csv'
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"  - {name}: {summary_path}")

            # Print key findings
            if name == 'overall':
                logger.info("\n--- Overall Statistics ---")
                for col in summary_df.columns:
                    logger.info(f"  {col}: {summary_df[col].iloc[0]:.2f}")

        # 5. Create visualizations
        logger.info("\n[5/5] Creating visualizations...")
        self.visualize_output_patterns()

        logger.info("\n" + "=" * 80)
        logger.info("Analysis complete!")
        logger.info(f"All outputs saved to: {self.output_dir}")
        logger.info("=" * 80)

        return {
            'outputs_long': outputs_df,
            'outputs_wide': wide_df,
            'summaries': summaries
        }


def main():
    """Command-line interface for OR7a output mapping."""
    parser = argparse.ArgumentParser(
        description="Map OR7a neuron output targets from FlyWire data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local connection data
  python scripts/map_or7a_outputs.py --data-source local

  # Query FlyWire API
  python scripts/map_or7a_outputs.py --data-source api

  # Custom output directory and top targets
  python scripts/map_or7a_outputs.py --output-dir results/or7a_custom/ --top-targets 15

  # Specify custom data paths
  python scripts/map_or7a_outputs.py --connections data/my_connections.csv.gz
        """
    )

    parser.add_argument(
        '--data-source',
        choices=['local', 'api'],
        default='local',
        help='Data source: "local" (use local CSV files) or "api" (query FlyWire)'
    )

    parser.add_argument(
        '--or7a-data',
        type=Path,
        default=DEFAULT_OR7A_DATA,
        help='Path to OR7a neurons CSV file'
    )

    parser.add_argument(
        '--connections',
        type=Path,
        default=DEFAULT_CONNECTIONS_DATA,
        help='Path to connections CSV file (for local data source)'
    )

    parser.add_argument(
        '--cell-types',
        type=Path,
        default=DEFAULT_CELL_TYPES_DATA,
        help='Path to cell types CSV file'
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
        help='Minimum synapse threshold for connections'
    )

    parser.add_argument(
        '--top-targets',
        type=int,
        default=20,
        help='Number of top targets to include per neuron in wide format'
    )

    args = parser.parse_args()

    # Create mapper and run analysis
    mapper = OR7aOutputMapper(
        or7a_data_path=args.or7a_data,
        data_source=args.data_source,
        connections_path=args.connections if args.data_source == 'local' else None,
        cell_types_path=args.cell_types,
        output_dir=args.output_dir,
        min_synapses=args.min_synapses,
        top_n_targets=args.top_targets
    )

    results = mapper.run_complete_analysis()

    return results


if __name__ == "__main__":
    main()
