"""
Analyze local neuron (LN) and projection neuron (PN) connectivity patterns.

This script analyzes cross-glomerular connectivity mediated by local neurons
and maps projection neuron downstream targeting patterns to understand how
LNs mediate lateral inhibition between glomeruli and which PNs project to
which downstream targets (KCs, MBONs).

Usage:
    python scripts/analyze_ln_pn_connectivity.py --data-dir data/flywire --output-dir results/ln_pn_analysis
    python scripts/analyze_ln_pn_connectivity.py --min-synapses 1 --top-glomeruli 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import existing neuron classification functions
from data_loaders.neuron_classification import (
    get_kc_neurons,
    get_pn_neurons,
    get_mbon_neurons,
    get_local_interneurons,
    infer_pn_glomerulus_labels,
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Visualization settings
sns.set_theme(context="talk", style="whitegrid", palette="deep")
plt.rcParams['figure.max_open_warning'] = 50

# Default paths
DEFAULT_DATA_DIR = Path("data/flywire")
DEFAULT_OUTPUT_DIR = Path("results/ln_pn_analysis")

# Key glomeruli of interest (can be expanded)
GLOMERULI_OF_INTEREST = ["DL5", "DM1", "DM2", "DM3", "DM4", "DA1", "DL1", "DL3", "DC2", "VA1v", "VA1d"]

# Olfactory receptor to glomerulus mapping
OR_TO_GLOMERULUS = {
    'or7a': 'DL5',
    'or13a': 'DA2',
    'or42b': 'DM2',
    'or47b': 'VA1v',
    'or59b': 'DM4',
    'or67d': 'DA1',
    'or82a': 'VA3',
    'or88a': 'VA1d',
    'or92a': 'VA2',
    # Add more as needed
}


@dataclass
class LNPNConnectivityAnalyzer:
    """
    Analyzes LN and PN connectivity patterns from FlyWire connectome data.

    Parameters
    ----------
    data_dir : Path
        Directory containing FlyWire CSV files
    output_dir : Path
        Directory for output files
    min_synapses : int
        Minimum synapse threshold for including a connection
    top_n_glomeruli : Optional[int]
        If set, limit visualizations to top N most-connected glomeruli
    """

    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_synapses: int = 1
    top_n_glomeruli: Optional[int] = None

    # Internal state
    _classification_df: Optional[pd.DataFrame] = None
    _labels_df: Optional[pd.DataFrame] = None
    _connections_df: Optional[pd.DataFrame] = None
    _neurons_df: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Initialize paths and create output directory."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_classification(self) -> pd.DataFrame:
        """Load neuron classification data."""
        if self._classification_df is not None:
            return self._classification_df

        class_path = self.data_dir / "classification.csv.gz"
        logger.info(f"Loading classification data from {class_path}")

        if not class_path.exists():
            # Try alternative naming
            class_path = self.data_dir / "classification.csv"
            if not class_path.exists():
                raise FileNotFoundError(f"Classification file not found: {class_path}")

        df = pd.read_csv(class_path, compression='infer')

        # Standardize column names
        if 'rootid' in df.columns and 'root_id' not in df.columns:
            df = df.rename(columns={'rootid': 'root_id'})

        logger.info(f"Loaded classification for {len(df):,} neurons")
        self._classification_df = df
        return df

    def load_cell_types(self) -> pd.DataFrame:
        """Load consolidated cell types data."""
        cell_types_path = self.data_dir / "consolidated_cell_types.csv.gz"

        if not cell_types_path.exists():
            cell_types_path = self.data_dir / "consolidatedcelltypes.csv.gz"
            if not cell_types_path.exists():
                logger.warning(f"Cell types file not found: {cell_types_path}")
                # Return empty dataframe with root_id column
                return pd.DataFrame(columns=['root_id'])

        logger.info(f"Loading cell types from {cell_types_path}")
        df = pd.read_csv(cell_types_path, compression='gzip')

        # Standardize column names
        if 'rootid' in df.columns and 'root_id' not in df.columns:
            df = df.rename(columns={'rootid': 'root_id'})

        logger.info(f"Loaded cell types for {len(df):,} neurons")
        return df

    def load_labels(self) -> pd.DataFrame:
        """Load processed labels with glomerulus annotations."""
        if self._labels_df is not None:
            return self._labels_df

        labels_path = self.data_dir / "processed_labels.csv.gz"
        logger.info(f"Loading glomerulus labels from {labels_path}")

        if not labels_path.exists():
            # Try alternative naming
            labels_path = self.data_dir / "processedlabels.csv.gz"
            if not labels_path.exists():
                labels_path = self.data_dir / "processed_labels.csv"
                if not labels_path.exists():
                    raise FileNotFoundError(f"Labels file not found: {labels_path}")

        df = pd.read_csv(labels_path, compression='infer')

        # Standardize column names
        if 'rootid' in df.columns and 'root_id' not in df.columns:
            df = df.rename(columns={'rootid': 'root_id'})

        # FlyWire uses 'processed_labels' not 'label'
        if 'processed_labels' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'processed_labels': 'label'})

        # Keep only rows with labels
        if 'label' in df.columns:
            df = df[df['label'].notna()].copy()

        logger.info(f"Loaded {len(df):,} label annotations")
        self._labels_df = df
        return df

    def load_connections(self) -> pd.DataFrame:
        """Load synaptic connections data."""
        if self._connections_df is not None:
            return self._connections_df

        conn_path = self.data_dir / "connections_princeton.csv.gz"
        logger.info(f"Loading connections from {conn_path}")

        if not conn_path.exists():
            conn_path = self.data_dir / "connectionsprinceton.csv.gz"
            if not conn_path.exists():
                raise FileNotFoundError(f"Connections file not found: {conn_path}")

        # Load with efficient dtypes
        logger.info("Reading connections file (this may take a minute)...")
        df = pd.read_csv(
            conn_path,
            compression='gzip',
            dtype={'pre_pt_root_id': 'int64', 'post_pt_root_id': 'int64'}
        )

        # Standardize column names
        rename_map = {}
        if 'pre_pt_root_id' in df.columns:
            rename_map['pre_pt_root_id'] = 'pre_root_id'
        if 'post_pt_root_id' in df.columns:
            rename_map['post_pt_root_id'] = 'post_root_id'
        if 'size' in df.columns and 'syn_count' not in df.columns:
            rename_map['size'] = 'syn_count'

        if rename_map:
            df = df.rename(columns=rename_map)

        # Filter by minimum synapses
        if self.min_synapses > 0:
            df = df[df['syn_count'] >= self.min_synapses].copy()

        logger.info(f"Loaded {len(df):,} connections (min {self.min_synapses} synapses)")
        self._connections_df = df
        return df

    def load_neurons(self) -> pd.DataFrame:
        """Load neuron metadata."""
        if self._neurons_df is not None:
            return self._neurons_df

        neurons_path = self.data_dir / "neurons.csv.gz"

        if not neurons_path.exists():
            logger.warning(f"Neurons file not found: {neurons_path}")
            logger.warning("Neuron metadata will be limited")
            return pd.DataFrame(columns=['root_id', 'type'])

        logger.info(f"Loading neuron metadata from {neurons_path}")
        df = pd.read_csv(neurons_path, compression='gzip')

        # Standardize column names
        if 'rootid' in df.columns and 'root_id' not in df.columns:
            df = df.rename(columns={'rootid': 'root_id'})

        logger.info(f"Loaded metadata for {len(df):,} neurons")
        self._neurons_df = df
        return df

    def identify_neuron_types(self) -> pd.DataFrame:
        """
        Identify and classify neurons by type using existing classification functions.

        Returns
        -------
        pd.DataFrame
            Dataframe with columns: root_id, neuron_type, class, glomerulus, etc.
        """
        logger.info("Identifying neuron types using existing classification functions...")

        # Load all required data
        classification_df = self.load_classification()
        cell_types_df = self.load_cell_types()
        labels_df = self.load_labels()
        neurons_df = self.load_neurons()

        # Use existing classification functions
        logger.info("Extracting Kenyon Cells...")
        kcs = get_kc_neurons(
            cell_types_df,
            classification_df,
            processed_labels_df=labels_df
        )
        kcs['neuron_type'] = 'KC'

        logger.info("Extracting Projection Neurons...")
        pns = get_pn_neurons(
            cell_types_df,
            classification_df,
            neurons_df=neurons_df,
            processed_labels_df=labels_df
        )
        pns['neuron_type'] = 'PN'

        # Infer glomerulus for PNs
        pns['glomerulus'] = infer_pn_glomerulus_labels(pns, processed_labels_df=labels_df)

        logger.info("Extracting MBONs...")
        mbons = get_mbon_neurons(cell_types_df, classification_df)
        mbons['neuron_type'] = 'MBON'

        logger.info("Extracting Local Interneurons...")
        lns = get_local_interneurons(
            cell_types_df,
            classification_df,
            neurons_df=neurons_df
        )
        lns['neuron_type'] = 'LN'
        # Don't try to infer glomeruli here - LNs need connectivity-based inference

        # Combine all classified neurons
        classified = pd.concat([kcs, pns, mbons, lns], ignore_index=True)

        # Add ORNs by searching processed_labels
        orn_mask = classification_df['class'].astype(str).str.contains('ORN', case=False, na=False)
        orns = classification_df[orn_mask].copy()
        orns['neuron_type'] = 'ORN'

        # Infer glomerulus for ORNs
        if not orns.empty:
            orns['glomerulus'] = infer_pn_glomerulus_labels(orns, processed_labels_df=labels_df)
            classified = pd.concat([classified, orns], ignore_index=True)

        # Add all other neurons as "Other"
        all_classified_ids = set(classified['root_id'])
        other_mask = ~classification_df['root_id'].isin(all_classified_ids)
        others = classification_df[other_mask].copy()
        others['neuron_type'] = 'Other'
        others['glomerulus'] = pd.NA

        # Combine everything
        neurons = pd.concat([classified, others], ignore_index=True)
        neurons = neurons.drop_duplicates(subset=['root_id'])

        # Log statistics
        type_counts = neurons['neuron_type'].value_counts()
        logger.info("Neuron type counts:")
        for ntype, count in type_counts.items():
            logger.info(f"  {ntype}: {count:,}")

        # Log glomerulus coverage
        pn_with_glom = neurons[(neurons['neuron_type'] == 'PN') & (neurons['glomerulus'].notna())]
        logger.info(f"PNs with glomerulus labels: {len(pn_with_glom):,} / {type_counts.get('PN', 0):,}")
        logger.info(f"(LN glomeruli will be inferred from connectivity in next step)")

        return neurons

    def infer_ln_glomeruli(self, neurons: pd.DataFrame) -> pd.DataFrame:
        """
        Infer glomerulus associations for LNs based on connectivity with PNs.

        Parameters
        ----------
        neurons : pd.DataFrame
            Classified neurons

        Returns
        -------
        pd.DataFrame
            Neurons with LN glomeruli assigned
        """
        logger.info("\nInferring LN glomeruli from connectivity...")

        connections = self.load_connections()

        # Get LNs and PNs
        lns = neurons[neurons['neuron_type'] == 'LN'].copy()
        pns = neurons[
            (neurons['neuron_type'] == 'PN') &
            (neurons['glomerulus'].notna())
        ].copy()

        if len(pns) == 0:
            logger.warning("No PNs with glomerulus labels - cannot infer LN glomeruli")
            return neurons

        # Create PN lookup
        pn_glom_lookup = pns.set_index('root_id')['glomerulus'].to_dict()

        # Find PN→LN connections
        pn_to_ln = connections[
            connections['pre_root_id'].isin(pns['root_id']) &
            connections['post_root_id'].isin(lns['root_id'])
        ].copy()
        pn_to_ln['source_glom'] = pn_to_ln['pre_root_id'].map(pn_glom_lookup)
        pn_to_ln = pn_to_ln.dropna(subset=['source_glom'])

        # Find LN→PN connections
        ln_to_pn = connections[
            connections['pre_root_id'].isin(lns['root_id']) &
            connections['post_root_id'].isin(pns['root_id'])
        ].copy()
        ln_to_pn['target_glom'] = ln_to_pn['post_root_id'].map(pn_glom_lookup)
        ln_to_pn = ln_to_pn.dropna(subset=['target_glom'])

        # Aggregate all glomeruli per LN (both input and output)
        ln_input_gloms = pn_to_ln.groupby(['post_root_id', 'source_glom'])['syn_count'].sum().reset_index()
        ln_input_gloms = ln_input_gloms.rename(columns={'post_root_id': 'ln_id', 'source_glom': 'glomerulus'})

        ln_output_gloms = ln_to_pn.groupby(['pre_root_id', 'target_glom'])['syn_count'].sum().reset_index()
        ln_output_gloms = ln_output_gloms.rename(columns={'pre_root_id': 'ln_id', 'target_glom': 'glomerulus'})

        # Combine and get all unique LN-glomerulus pairs
        all_ln_gloms = pd.concat([
            ln_input_gloms[['ln_id', 'glomerulus']],
            ln_output_gloms[['ln_id', 'glomerulus']]
        ]).drop_duplicates()

        # Create a mapping of LN to list of glomeruli
        ln_to_gloms = all_ln_gloms.groupby('ln_id')['glomerulus'].apply(list).to_dict()

        # Update neurons dataframe with LN glomeruli
        # Store as comma-separated list for LNs with multiple glomeruli
        def get_ln_glomeruli(row):
            if row['neuron_type'] == 'LN' and row['root_id'] in ln_to_gloms:
                gloms = ln_to_gloms[row['root_id']]
                return ','.join(sorted(gloms)) if len(gloms) > 1 else gloms[0]
            return row.get('glomerulus', None)

        neurons['glomerulus'] = neurons.apply(get_ln_glomeruli, axis=1)

        # Log statistics
        lns_with_glom = neurons[(neurons['neuron_type'] == 'LN') & (neurons['glomerulus'].notna())]
        logger.info(f"Assigned glomeruli to {len(lns_with_glom):,} / {len(lns):,} LNs via connectivity")

        return neurons

    def analyze_ln_cross_glomerular_connections(self, neurons: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze LN-mediated cross-glomerular connections by inferring glomeruli from connectivity.

        Since LNs typically don't have glomerulus labels in their metadata, we infer them from
        their connectivity patterns with PNs:
        - Source glomerulus: Which PN glomerulus provides input to the LN
        - Target glomerulus: Which PN glomerulus receives output from the LN

        Parameters
        ----------
        neurons : pd.DataFrame
            Classified neurons with glomerulus labels

        Returns
        -------
        pd.DataFrame
            Cross-glomerular LN connections with columns:
            source_glom, target_glom, ln_count, total_synapses, mean_weight, std_weight
        """
        logger.info("\n[1/3] Analyzing LN cross-glomerular connections...")

        connections = self.load_connections()

        # Get all LNs (don't require glomerulus labels)
        lns = neurons[neurons['neuron_type'] == 'LN'].copy()
        logger.info(f"Found {len(lns):,} LNs total")

        # Get PNs with glomerulus labels
        pns = neurons[
            (neurons['neuron_type'] == 'PN') &
            (neurons['glomerulus'].notna())
        ].copy()
        logger.info(f"Found {len(pns):,} PNs with glomerulus labels")

        if len(pns) == 0:
            logger.warning("No PNs with glomerulus labels - cannot infer LN connectivity!")
            return pd.DataFrame(columns=['source_glom', 'target_glom', 'ln_count', 'total_synapses', 'mean_weight', 'std_weight'])

        # Create PN lookup: root_id -> glomerulus
        pn_glom_lookup = pns.set_index('root_id')['glomerulus'].to_dict()

        # Step 1: Find PN→LN connections (inputs to LNs)
        pn_to_ln = connections[
            connections['pre_root_id'].isin(pns['root_id']) &
            connections['post_root_id'].isin(lns['root_id'])
        ].copy()
        logger.info(f"Found {len(pn_to_ln):,} PN→LN connections")

        # Add source glomerulus from PN
        pn_to_ln['source_glom'] = pn_to_ln['pre_root_id'].map(pn_glom_lookup)
        pn_to_ln = pn_to_ln.dropna(subset=['source_glom'])

        # Step 2: Find LN→PN connections (outputs from LNs)
        ln_to_pn = connections[
            connections['pre_root_id'].isin(lns['root_id']) &
            connections['post_root_id'].isin(pns['root_id'])
        ].copy()
        logger.info(f"Found {len(ln_to_pn):,} LN→PN connections")

        # Add target glomerulus from PN
        ln_to_pn['target_glom'] = ln_to_pn['post_root_id'].map(pn_glom_lookup)
        ln_to_pn = ln_to_pn.dropna(subset=['target_glom'])

        # Step 3: For each LN, determine its primary input and output glomeruli
        # Use the glomerulus with the most synaptic weight
        ln_inputs = pn_to_ln.groupby(['post_root_id', 'source_glom'])['syn_count'].sum().reset_index()
        ln_inputs = ln_inputs.sort_values('syn_count', ascending=False).drop_duplicates('post_root_id')
        ln_inputs = ln_inputs.rename(columns={'post_root_id': 'ln_id', 'source_glom': 'input_glom'})

        ln_outputs = ln_to_pn.groupby(['pre_root_id', 'target_glom'])['syn_count'].sum().reset_index()
        ln_outputs = ln_outputs.sort_values('syn_count', ascending=False).drop_duplicates('pre_root_id')
        ln_outputs = ln_outputs.rename(columns={'pre_root_id': 'ln_id', 'target_glom': 'output_glom'})

        # Merge to get LNs with both input and output glomeruli
        ln_glomeruli = ln_inputs.merge(ln_outputs, on='ln_id', how='inner')
        logger.info(f"Assigned glomeruli to {len(ln_glomeruli):,} LNs based on connectivity")

        # Step 4: Find cross-glomerular LNs (input_glom != output_glom)
        cross_glom_lns = ln_glomeruli[ln_glomeruli['input_glom'] != ln_glomeruli['output_glom']].copy()
        logger.info(f"Found {len(cross_glom_lns):,} cross-glomerular LNs")

        if len(cross_glom_lns) == 0:
            logger.warning("No cross-glomerular LNs found!")
            return pd.DataFrame(columns=['source_glom', 'target_glom', 'ln_count', 'total_synapses', 'mean_weight', 'std_weight'])

        # Step 5: Get all LN→PN connections for these cross-glomerular LNs
        cross_ln_ids = set(cross_glom_lns['ln_id'])
        cross_ln_conns = ln_to_pn[ln_to_pn['pre_root_id'].isin(cross_ln_ids)].copy()

        # Add input glomerulus for each LN
        ln_input_lookup = ln_glomeruli.set_index('ln_id')['input_glom'].to_dict()
        cross_ln_conns['source_glom'] = cross_ln_conns['pre_root_id'].map(ln_input_lookup)
        cross_ln_conns = cross_ln_conns.dropna(subset=['source_glom'])

        # Filter for connections where target_glom != source_glom
        cross_ln_conns = cross_ln_conns[cross_ln_conns['target_glom'] != cross_ln_conns['source_glom']]
        logger.info(f"Found {len(cross_ln_conns):,} cross-glomerular connections from LNs to PNs")

        # Step 6: Aggregate by source-target glomerulus pairs
        summary = cross_ln_conns.groupby(['source_glom', 'target_glom']).agg({
            'pre_root_id': 'nunique',  # Number of unique LNs
            'syn_count': ['sum', 'mean', 'std']
        }).reset_index()

        summary.columns = ['source_glom', 'target_glom', 'ln_count', 'total_synapses', 'mean_weight', 'std_weight']
        summary = summary.sort_values('total_synapses', ascending=False)

        # Fill NaN std with 0 (happens when only 1 connection)
        summary['std_weight'] = summary['std_weight'].fillna(0)

        logger.info(f"Identified {len(summary)} unique glomerular pairs with LN-mediated connections")

        # Report top connections
        logger.info("\nTop 10 LN-mediated cross-glomerular connections:")
        for idx, row in summary.head(10).iterrows():
            logger.info(f"  {row['source_glom']} → {row['target_glom']}: "
                       f"{row['ln_count']} LNs, {row['total_synapses']:.0f} synapses "
                       f"(mean={row['mean_weight']:.1f})")

        return summary

    def analyze_pn_downstream_targets(self, neurons: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze PN downstream targeting patterns.

        Parameters
        ----------
        neurons : pd.DataFrame
            Classified neurons with glomerulus labels

        Returns
        -------
        pd.DataFrame
            PN downstream targets with columns:
            glomerulus, pn_root_id, pn_count, target_type, target_root_id, synapses
        """
        logger.info("\n[2/3] Analyzing PN downstream targets...")

        connections = self.load_connections()

        # Get PNs with glomerulus labels
        pns = neurons[
            (neurons['neuron_type'] == 'PN') &
            (neurons['glomerulus'].notna())
        ].copy()

        if len(pns) == 0:
            logger.warning("No PNs with glomerulus labels found!")
            return pd.DataFrame(columns=['glomerulus', 'pn_root_id', 'pn_count', 'target_type', 'target_root_id', 'synapses'])

        logger.info(f"Found {len(pns):,} PNs with glomerulus labels")

        # Get glomerulus counts
        glom_counts = pns['glomerulus'].value_counts()
        logger.info(f"PNs span {len(glom_counts)} glomeruli")
        logger.info(f"Top glomeruli: {dict(glom_counts.head(10))}")

        # Get PN output connections
        pn_connections = connections[
            connections['pre_root_id'].isin(pns['root_id'])
        ].copy()

        logger.info(f"Found {len(pn_connections):,} PN output connections")

        # Merge with source PN info
        pn_connections = pn_connections.merge(
            pns[['root_id', 'glomerulus']],
            left_on='pre_root_id',
            right_on='root_id',
            how='inner'
        )

        # Merge with target neuron type info
        pn_connections = pn_connections.merge(
            neurons[['root_id', 'neuron_type']],
            left_on='post_root_id',
            right_on='root_id',
            how='left',
            suffixes=('_pn', '_target')
        )

        # Fill missing target types
        pn_connections['neuron_type'] = pn_connections['neuron_type'].fillna('Unknown')

        # Create detailed output
        output = pn_connections[[
            'glomerulus', 'pre_root_id', 'neuron_type', 'post_root_id', 'syn_count'
        ]].copy()

        output.columns = ['glomerulus', 'pn_root_id', 'target_type', 'target_root_id', 'synapses']

        # Add PN count per glomerulus
        pn_counts_per_glom = pns.groupby('glomerulus').size().to_dict()
        output['pn_count'] = output['glomerulus'].map(pn_counts_per_glom)

        # Reorder columns
        output = output[['glomerulus', 'pn_root_id', 'pn_count', 'target_type', 'target_root_id', 'synapses']]

        # Sort by glomerulus and synapses
        output = output.sort_values(['glomerulus', 'synapses'], ascending=[True, False])

        # Log statistics by target type
        logger.info("\nPN downstream target statistics:")
        target_stats = pn_connections.groupby('neuron_type')['syn_count'].agg(['count', 'sum', 'mean'])
        for target_type, stats in target_stats.iterrows():
            logger.info(f"  {target_type}: {stats['count']:,} connections, "
                       f"{stats['sum']:,} total synapses (mean={stats['mean']:.1f})")

        return output

    def calculate_pn_convergence(self, neurons: pd.DataFrame, pn_targets: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate PN convergence ratios by glomerulus.

        Parameters
        ----------
        neurons : pd.DataFrame
            Classified neurons with glomerulus labels
        pn_targets : pd.DataFrame
            PN downstream targets

        Returns
        -------
        pd.DataFrame
            Convergence metrics with columns:
            glomerulus, orn_count, pn_count, kc_targets, mbon_targets,
            orn_to_pn_ratio, pn_to_kc_ratio, total_output_synapses
        """
        logger.info("\nCalculating PN convergence ratios...")

        glomeruli = pn_targets['glomerulus'].unique()
        results = []

        for glom in glomeruli:
            # Count ORNs in this glomerulus
            orn_count = len(neurons[
                (neurons['neuron_type'] == 'ORN') &
                (neurons['glomerulus'] == glom)
            ])

            # Count PNs in this glomerulus
            pn_count = len(neurons[
                (neurons['neuron_type'] == 'PN') &
                (neurons['glomerulus'] == glom)
            ])

            # Get targets for this glomerulus
            glom_targets = pn_targets[pn_targets['glomerulus'] == glom]

            # Count unique KC and MBON targets
            kc_targets = glom_targets[glom_targets['target_type'] == 'KC']['target_root_id'].nunique()
            mbon_targets = glom_targets[glom_targets['target_type'] == 'MBON']['target_root_id'].nunique()

            # Total output synapses
            total_synapses = glom_targets['synapses'].sum()

            # Calculate ratios
            orn_to_pn_ratio = orn_count / pn_count if pn_count > 0 else 0
            pn_to_kc_ratio = pn_count / kc_targets if kc_targets > 0 else 0

            results.append({
                'glomerulus': glom,
                'orn_count': orn_count,
                'pn_count': pn_count,
                'kc_targets': kc_targets,
                'mbon_targets': mbon_targets,
                'orn_to_pn_ratio': orn_to_pn_ratio,
                'pn_to_kc_ratio': pn_to_kc_ratio,
                'total_output_synapses': total_synapses
            })

        convergence_df = pd.DataFrame(results)

        # Only sort if we have data
        if not convergence_df.empty and 'total_output_synapses' in convergence_df.columns:
            convergence_df = convergence_df.sort_values('total_output_synapses', ascending=False)

        return convergence_df

    def build_glomerular_interaction_matrix(self, ln_connections: pd.DataFrame) -> pd.DataFrame:
        """
        Build cross-glomerular interaction matrix.

        Parameters
        ----------
        ln_connections : pd.DataFrame
            LN cross-glomerular connections

        Returns
        -------
        pd.DataFrame
            Pivot table with source glomeruli as rows, target glomeruli as columns
        """
        logger.info("\n[3/3] Building glomerular interaction matrix...")

        # Create pivot table
        matrix = ln_connections.pivot_table(
            index='source_glom',
            columns='target_glom',
            values='total_synapses',
            fill_value=0,
            aggfunc='sum'
        )

        logger.info(f"Interaction matrix shape: {matrix.shape[0]} sources × {matrix.shape[1]} targets")

        # Calculate asymmetry scores
        asymmetries = []
        for source in matrix.index:
            for target in matrix.columns:
                forward = matrix.loc[source, target]
                if source in matrix.columns and target in matrix.index:
                    backward = matrix.loc[target, source]
                    if forward > 0 or backward > 0:
                        asymmetry = (forward - backward) / (forward + backward) if (forward + backward) > 0 else 0
                        asymmetries.append({
                            'pair': f"{source}-{target}",
                            'forward': forward,
                            'backward': backward,
                            'asymmetry': asymmetry
                        })

        # Report top asymmetric pairs
        if asymmetries:
            asym_df = pd.DataFrame(asymmetries)
            asym_df = asym_df[asym_df['asymmetry'].abs() > 0.5]  # Strong asymmetry
            asym_df = asym_df.sort_values('asymmetry', key=abs, ascending=False)

            logger.info(f"\nTop 10 asymmetric glomerular interactions:")
            for idx, row in asym_df.head(10).iterrows():
                direction = "→" if row['asymmetry'] > 0 else "←"
                logger.info(f"  {row['pair'].replace('-', direction)}: "
                           f"forward={row['forward']:.0f}, backward={row['backward']:.0f}, "
                           f"asymmetry={row['asymmetry']:.2f}")

        return matrix

    def visualize_cross_glomerular_heatmap(self, matrix: pd.DataFrame):
        """Create heatmap of cross-glomerular LN connectivity."""
        logger.info("Creating cross-glomerular connectivity heatmap...")

        # Limit to top glomeruli if requested
        if self.top_n_glomeruli:
            # Get top glomeruli by total connectivity
            row_sums = matrix.sum(axis=1)
            col_sums = matrix.sum(axis=0)
            total_activity = row_sums.add(col_sums, fill_value=0).sort_values(ascending=False)
            top_gloms = total_activity.head(self.top_n_glomeruli).index
            matrix = matrix.loc[matrix.index.isin(top_gloms), matrix.columns.isin(top_gloms)]

        fig, ax = plt.subplots(figsize=(16, 14))

        # Create heatmap
        sns.heatmap(
            matrix,
            cmap='YlOrRd',
            annot=False,
            fmt='.0f',
            cbar_kws={'label': 'Total Synapses'},
            square=True,
            linewidths=0.5,
            ax=ax
        )

        ax.set_title('LN-Mediated Cross-Glomerular Connectivity', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Target Glomerulus', fontsize=12, fontweight='bold')
        ax.set_ylabel('Source Glomerulus', fontsize=12, fontweight='bold')

        plt.tight_layout()

        output_path = self.output_dir / 'cross_glomerular_heatmap.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {output_path}")
        plt.close()

    def visualize_glomerular_network(self, ln_connections: pd.DataFrame):
        """Create network graph of glomerular interactions."""
        logger.info("Creating glomerular network graph...")

        # Limit to top connections if there are too many
        if self.top_n_glomeruli:
            # Get top glomeruli by total connectivity
            all_gloms = set(ln_connections['source_glom']) | set(ln_connections['target_glom'])
            glom_activity = {}
            for glom in all_gloms:
                outgoing = ln_connections[ln_connections['source_glom'] == glom]['total_synapses'].sum()
                incoming = ln_connections[ln_connections['target_glom'] == glom]['total_synapses'].sum()
                glom_activity[glom] = outgoing + incoming

            top_gloms = sorted(glom_activity.keys(), key=lambda x: glom_activity[x], reverse=True)[:self.top_n_glomeruli]
            ln_connections = ln_connections[
                ln_connections['source_glom'].isin(top_gloms) &
                ln_connections['target_glom'].isin(top_gloms)
            ].copy()

        # Create directed graph
        G = nx.DiGraph()

        for _, row in ln_connections.iterrows():
            G.add_edge(
                row['source_glom'],
                row['target_glom'],
                weight=row['total_synapses'],
                ln_count=row['ln_count']
            )

        if len(G.nodes()) == 0:
            logger.warning("No nodes in network graph, skipping visualization")
            return

        fig, ax = plt.subplots(figsize=(18, 16))

        # Calculate node sizes based on degree
        node_degrees = dict(G.degree())
        node_sizes = [300 + node_degrees[node] * 50 for node in G.nodes()]

        # Calculate edge widths based on synapse count
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [1 + (w / max_weight) * 5 for w in edge_weights]

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw network
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color='lightblue',
            edgecolors='darkblue',
            linewidths=2,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.6,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )

        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        ax.set_title('Glomerular Interaction Network (LN-mediated)', fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        output_path = self.output_dir / 'glomerular_network.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved network graph to {output_path}")
        plt.close()

    def visualize_pn_convergence(self, convergence: pd.DataFrame):
        """Create bar chart of PN→KC convergence by glomerulus."""
        logger.info("Creating PN convergence visualization...")

        # Limit to top glomeruli if requested
        if self.top_n_glomeruli:
            convergence = convergence.head(self.top_n_glomeruli)

        fig, axes = plt.subplots(2, 2, figsize=(20, 14))

        # 1. PN counts by glomerulus
        ax1 = axes[0, 0]
        convergence_sorted = convergence.sort_values('pn_count', ascending=False).head(20)
        ax1.barh(range(len(convergence_sorted)), convergence_sorted['pn_count'], color='steelblue')
        ax1.set_yticks(range(len(convergence_sorted)))
        ax1.set_yticklabels(convergence_sorted['glomerulus'])
        ax1.set_xlabel('Number of PNs', fontweight='bold')
        ax1.set_title('PN Count by Glomerulus (Top 20)', fontweight='bold')
        ax1.invert_yaxis()

        # 2. KC targets by glomerulus
        ax2 = axes[0, 1]
        convergence_sorted = convergence.sort_values('kc_targets', ascending=False).head(20)
        ax2.barh(range(len(convergence_sorted)), convergence_sorted['kc_targets'], color='coral')
        ax2.set_yticks(range(len(convergence_sorted)))
        ax2.set_yticklabels(convergence_sorted['glomerulus'])
        ax2.set_xlabel('Number of KC Targets', fontweight='bold')
        ax2.set_title('KC Targets by Glomerulus (Top 20)', fontweight='bold')
        ax2.invert_yaxis()

        # 3. ORN→PN ratio
        ax3 = axes[1, 0]
        conv_with_orns = convergence[convergence['orn_count'] > 0].sort_values('orn_to_pn_ratio', ascending=False).head(20)
        ax3.barh(range(len(conv_with_orns)), conv_with_orns['orn_to_pn_ratio'], color='mediumseagreen')
        ax3.set_yticks(range(len(conv_with_orns)))
        ax3.set_yticklabels(conv_with_orns['glomerulus'])
        ax3.set_xlabel('ORN:PN Ratio', fontweight='bold')
        ax3.set_title('ORN→PN Convergence (Top 20)', fontweight='bold')
        ax3.invert_yaxis()
        ax3.axvline(1, color='red', linestyle='--', alpha=0.5, label='1:1 ratio')
        ax3.legend()

        # 4. Total output synapses
        ax4 = axes[1, 1]
        convergence_sorted = convergence.sort_values('total_output_synapses', ascending=False).head(20)
        ax4.barh(range(len(convergence_sorted)), convergence_sorted['total_output_synapses'], color='mediumpurple')
        ax4.set_yticks(range(len(convergence_sorted)))
        ax4.set_yticklabels(convergence_sorted['glomerulus'])
        ax4.set_xlabel('Total Output Synapses', fontweight='bold')
        ax4.set_title('PN Output Strength by Glomerulus (Top 20)', fontweight='bold')
        ax4.invert_yaxis()

        plt.suptitle('PN Convergence Analysis by Glomerulus', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_path = self.output_dir / 'pn_convergence.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved convergence plot to {output_path}")
        plt.close()

    def run_complete_analysis(self):
        """Execute complete LN/PN connectivity analysis pipeline."""
        logger.info("=" * 80)
        logger.info("LN and PN Connectivity Analysis")
        logger.info("=" * 80)

        # Step 1: Identify neuron types
        logger.info("\nStep 1: Loading and classifying neurons...")
        neurons = self.identify_neuron_types()

        # Step 1b: Infer LN glomeruli from connectivity
        neurons = self.infer_ln_glomeruli(neurons)

        # Step 2: Analyze LN cross-glomerular connections
        ln_connections = self.analyze_ln_cross_glomerular_connections(neurons)

        # Save LN connections CSV
        ln_output_path = self.output_dir / 'ln_cross_glomerular_connections.csv'
        ln_connections.to_csv(ln_output_path, index=False)
        logger.info(f"\nSaved LN connections to {ln_output_path}")

        # Step 3: Analyze PN downstream targets
        pn_targets = self.analyze_pn_downstream_targets(neurons)

        # Save PN targets CSV
        pn_output_path = self.output_dir / 'pn_downstream_targets.csv'
        pn_targets.to_csv(pn_output_path, index=False)
        logger.info(f"Saved PN targets to {pn_output_path}")

        # Step 4: Calculate PN convergence
        convergence = self.calculate_pn_convergence(neurons, pn_targets)

        # Save convergence CSV
        conv_output_path = self.output_dir / 'pn_convergence_ratios.csv'
        convergence.to_csv(conv_output_path, index=False)
        logger.info(f"Saved convergence metrics to {conv_output_path}")

        # Step 5: Build interaction matrix
        matrix = self.build_glomerular_interaction_matrix(ln_connections)

        # Save matrix CSV
        matrix_output_path = self.output_dir / 'glomerular_interaction_matrix.csv'
        matrix.to_csv(matrix_output_path)
        logger.info(f"Saved interaction matrix to {matrix_output_path}")

        # Step 6: Create visualizations
        logger.info("\nGenerating visualizations...")

        if len(ln_connections) > 0:
            self.visualize_cross_glomerular_heatmap(matrix)
            self.visualize_glomerular_network(ln_connections)
        else:
            logger.warning("Skipping LN visualizations (no data)")

        if len(convergence) > 0:
            self.visualize_pn_convergence(convergence)
        else:
            logger.warning("Skipping PN convergence visualization (no data)")

        # Summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total neurons analyzed: {len(neurons):,}")
        logger.info(f"  - LNs: {(neurons['neuron_type'] == 'LN').sum():,}")
        logger.info(f"  - PNs: {(neurons['neuron_type'] == 'PN').sum():,}")
        logger.info(f"  - KCs: {(neurons['neuron_type'] == 'KC').sum():,}")
        logger.info(f"  - MBONs: {(neurons['neuron_type'] == 'MBON').sum():,}")
        logger.info(f"Cross-glomerular LN connections: {len(ln_connections):,} unique pairs")
        logger.info(f"PN downstream connections: {len(pn_targets):,}")
        logger.info(f"Glomeruli analyzed: {len(convergence)}")
        logger.info(f"\nAll outputs saved to: {self.output_dir}")
        logger.info("=" * 80)

        return {
            'neurons': neurons,
            'ln_connections': ln_connections,
            'pn_targets': pn_targets,
            'convergence': convergence,
            'interaction_matrix': matrix
        }


def main():
    """Command-line interface for LN/PN connectivity analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze LN and PN connectivity patterns from FlyWire data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default settings
  python scripts/analyze_ln_pn_connectivity.py

  # Specify custom data directory
  python scripts/analyze_ln_pn_connectivity.py --data-dir data/flywire

  # Custom output directory and minimum synapses
  python scripts/analyze_ln_pn_connectivity.py --output-dir results/my_analysis --min-synapses 1

  # Limit visualizations to top 20 glomeruli
  python scripts/analyze_ln_pn_connectivity.py --top-glomeruli 20
        """
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DEFAULT_DATA_DIR,
        help='Directory containing FlyWire CSV files (default: data/flywire)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results (default: results/ln_pn_analysis)'
    )

    parser.add_argument(
        '--min-synapses',
        type=int,
        default=1,
        help='Minimum synapse threshold for connections (default: 1)'
    )

    parser.add_argument(
        '--top-glomeruli',
        type=int,
        default=None,
        help='Limit visualizations to top N glomeruli (default: all)'
    )

    args = parser.parse_args()

    # Create analyzer and run
    analyzer = LNPNConnectivityAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_synapses=args.min_synapses,
        top_n_glomeruli=args.top_glomeruli
    )

    results = analyzer.run_complete_analysis()

    return results


if __name__ == "__main__":
    main()
