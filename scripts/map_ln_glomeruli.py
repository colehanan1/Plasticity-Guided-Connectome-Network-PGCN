"""
Map Local Neurons (LNs) to Glomeruli via Connectivity Inference

This script identifies which glomeruli each Local Neuron is associated with by analyzing
their synaptic connections to Projection Neurons with known glomerulus labels.

Unlike PNs, LNs typically don't have glomerulus labels in their cell type annotations.
We infer associations from connectivity patterns:
- Source glomerulus: Which PNs provide INPUT to the LN
- Target glomerulus: Which PNs receive OUTPUT from the LN

Usage:
    python scripts/map_ln_glomeruli.py --data-dir data/flywire --output-dir results/ln_mapping
    python scripts/map_ln_glomeruli.py --min-synapses 5 --neuropil AL
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import existing neuron classification functions
from data_loaders.neuron_classification import (
    get_pn_neurons,
    get_local_interneurons,
    infer_pn_glomerulus_labels,
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("data/flywire")
DEFAULT_OUTPUT_DIR = Path("results/ln_mapping")


def categorize_ln_by_breadth(num_glomeruli: int) -> str:
    """
    Categorize LN by glomerular breadth.

    Parameters
    ----------
    num_glomeruli : int
        Number of glomeruli the LN connects to

    Returns
    -------
    str
        Category: 'uniglomerular', 'oligoglomerular', 'multiglomerular', or 'broad'
    """
    if num_glomeruli == 1:
        return 'uniglomerular'
    elif num_glomeruli <= 3:
        return 'oligoglomerular'
    elif num_glomeruli <= 10:
        return 'multiglomerular'
    else:
        return 'broad'


class LNGlomerulusMapper:
    """
    Maps Local Neurons to glomeruli via connectivity inference.

    Parameters
    ----------
    data_dir : Path
        Directory containing FlyWire CSV files
    output_dir : Path
        Directory for output files
    min_synapses : int
        Minimum synapse threshold for including a connection
    neuropil : Optional[str]
        Neuropil to focus on (e.g., 'AL' for antennal lobe)
    """

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        min_synapses: int = 3,
        neuropil: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.min_synapses = min_synapses
        self.neuropil = neuropil

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache loaded data
        self._connections = None
        self._cell_types = None
        self._classification = None
        self._labels = None
        self._neurons = None

    def load_connections(self) -> pd.DataFrame:
        """Load synaptic connections."""
        if self._connections is not None:
            return self._connections

        conn_path = self.data_dir / "connections_princeton.csv.gz"
        logger.info(f"Loading connections from {conn_path}")

        if not conn_path.exists():
            raise FileNotFoundError(f"Connections file not found: {conn_path}")

        df = pd.read_csv(conn_path, compression='gzip')

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

        # Filter by neuropil if specified
        if self.neuropil and 'neuropil' in df.columns:
            df = df[df['neuropil'].str.contains(self.neuropil, case=False, na=False)].copy()
            logger.info(f"Filtered to {self.neuropil} neuropil")

        logger.info(f"Loaded {len(df):,} connections (min {self.min_synapses} synapses)")
        self._connections = df
        return df

    def load_cell_types(self) -> pd.DataFrame:
        """Load consolidated cell types."""
        if self._cell_types is not None:
            return self._cell_types

        ct_path = self.data_dir / "consolidated_cell_types.csv.gz"
        logger.info(f"Loading cell types from {ct_path}")

        if not ct_path.exists():
            raise FileNotFoundError(f"Cell types file not found: {ct_path}")

        df = pd.read_csv(ct_path, compression='gzip')
        logger.info(f"Loaded cell types for {len(df):,} neurons")

        self._cell_types = df
        return df

    def load_classification(self) -> pd.DataFrame:
        """Load classification data."""
        if self._classification is not None:
            return self._classification

        class_path = self.data_dir / "classification.csv.gz"
        logger.info(f"Loading classification from {class_path}")

        if not class_path.exists():
            logger.warning(f"Classification file not found: {class_path}")
            return pd.DataFrame()

        df = pd.read_csv(class_path, compression='gzip')
        logger.info(f"Loaded classification for {len(df):,} neurons")

        self._classification = df
        return df

    def load_labels(self) -> pd.DataFrame:
        """Load processed labels."""
        if self._labels is not None:
            return self._labels

        labels_path = self.data_dir / "processed_labels.csv.gz"
        logger.info(f"Loading labels from {labels_path}")

        if not labels_path.exists():
            logger.warning(f"Labels file not found: {labels_path}")
            return pd.DataFrame()

        df = pd.read_csv(labels_path, compression='gzip')

        # Rename processed_labels to label if needed
        if 'processed_labels' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'processed_labels': 'label'})

        logger.info(f"Loaded {len(df):,} label annotations")

        self._labels = df
        return df

    def load_neurons(self) -> pd.DataFrame:
        """Load neuron metadata."""
        if self._neurons is not None:
            return self._neurons

        neurons_path = self.data_dir / "neurons.csv.gz"

        if not neurons_path.exists():
            logger.warning(f"Neurons file not found: {neurons_path}")
            return pd.DataFrame()

        logger.info(f"Loading neuron metadata from {neurons_path}")
        df = pd.read_csv(neurons_path, compression='gzip')

        # Standardize column names
        if 'rootid' in df.columns and 'root_id' not in df.columns:
            df = df.rename(columns={'rootid': 'root_id'})

        logger.info(f"Loaded metadata for {len(df):,} neurons")
        self._neurons = df
        return df

    def identify_local_neurons(self) -> pd.DataFrame:
        """
        Identify Local Neurons using existing classification functions.

        Returns
        -------
        pd.DataFrame
            Local neurons with root_id
        """
        logger.info("\nIdentifying Local Neurons...")

        # Load required data
        cell_types = self.load_cell_types()
        classification = self.load_classification()
        neurons = self.load_neurons()

        # Use existing get_local_interneurons function (AL-specific)
        lns = get_local_interneurons(
            cell_types,
            classification,
            neurons_df=neurons
        )

        logger.info(f"Found {len(lns):,} Local Neurons (antennal lobe)")

        return lns

    def identify_projection_neurons(self) -> pd.DataFrame:
        """
        Identify Projection Neurons with glomerulus labels using existing classification functions.

        Returns
        -------
        pd.DataFrame
            Projection neurons with root_id and glomerulus
        """
        logger.info("\nIdentifying Projection Neurons...")

        # Load required data
        cell_types = self.load_cell_types()
        classification = self.load_classification()
        labels = self.load_labels()
        neurons = self.load_neurons()

        # Use existing get_pn_neurons function
        pns = get_pn_neurons(
            cell_types,
            classification,
            neurons_df=neurons,
            processed_labels_df=labels
        )

        logger.info(f"Found {len(pns):,} Projection Neurons")

        # Infer glomerulus labels using existing function
        pns['glomerulus'] = infer_pn_glomerulus_labels(pns, processed_labels_df=labels)

        # Keep only PNs with glomerulus labels
        pns = pns[pns['glomerulus'].notna()].copy()

        logger.info(f"Found {len(pns):,} Projection Neurons with glomerulus labels")
        logger.info(f"Unique glomeruli: {pns['glomerulus'].nunique()}")

        # Show top glomeruli
        if len(pns) > 0:
            top_gloms = pns['glomerulus'].value_counts().head(10)
            logger.info(f"Top glomeruli: {dict(top_gloms)}")

        return pns

    def map_ln_to_glomeruli(
        self,
        lns: pd.DataFrame,
        pns: pd.DataFrame,
        connections: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Map LNs to glomeruli based on connectivity with PNs.

        Parameters
        ----------
        lns : pd.DataFrame
            Local neurons
        pns : pd.DataFrame
            Projection neurons with glomerulus labels
        connections : pd.DataFrame
            Synaptic connections

        Returns
        -------
        pd.DataFrame
            LN-glomerulus associations
        """
        logger.info("\nMapping LN-glomerulus associations via connectivity...")

        ln_ids = set(lns['root_id'])
        pn_ids = set(pns['root_id'])

        # Create PN -> glomerulus lookup
        pn_to_glom = dict(zip(pns['root_id'], pns['glomerulus']))

        # Find PN → LN connections (inputs to LNs)
        logger.info("Finding PN → LN connections...")
        pn_to_ln = connections[
            connections['pre_root_id'].isin(pn_ids) &
            connections['post_root_id'].isin(ln_ids)
        ].copy()
        pn_to_ln['source_glomerulus'] = pn_to_ln['pre_root_id'].map(pn_to_glom)
        pn_to_ln = pn_to_ln.dropna(subset=['source_glomerulus'])
        logger.info(f"Found {len(pn_to_ln):,} PN → LN connections")

        # Find LN → PN connections (outputs from LNs)
        logger.info("Finding LN → PN connections...")
        ln_to_pn = connections[
            connections['pre_root_id'].isin(ln_ids) &
            connections['post_root_id'].isin(pn_ids)
        ].copy()
        ln_to_pn['target_glomerulus'] = ln_to_pn['post_root_id'].map(pn_to_glom)
        ln_to_pn = ln_to_pn.dropna(subset=['target_glomerulus'])
        logger.info(f"Found {len(ln_to_pn):,} LN → PN connections")

        # Aggregate input glomeruli for each LN
        logger.info("Aggregating LN input glomeruli...")
        ln_input_glom = pn_to_ln.groupby(['post_root_id', 'source_glomerulus']).agg({
            'syn_count': 'sum',
            'pre_root_id': 'nunique'
        }).rename(columns={
            'syn_count': 'input_synapses',
            'pre_root_id': 'num_input_pns'
        }).reset_index()
        ln_input_glom.rename(columns={
            'post_root_id': 'ln_id',
            'source_glomerulus': 'glomerulus'
        }, inplace=True)

        # Aggregate output glomeruli for each LN
        logger.info("Aggregating LN output glomeruli...")
        ln_output_glom = ln_to_pn.groupby(['pre_root_id', 'target_glomerulus']).agg({
            'syn_count': 'sum',
            'post_root_id': 'nunique'
        }).rename(columns={
            'syn_count': 'output_synapses',
            'post_root_id': 'num_output_pns'
        }).reset_index()
        ln_output_glom.rename(columns={
            'pre_root_id': 'ln_id',
            'target_glomerulus': 'glomerulus'
        }, inplace=True)

        # Merge input and output associations
        logger.info("Merging input and output associations...")
        ln_glom_full = ln_input_glom.merge(
            ln_output_glom,
            on=['ln_id', 'glomerulus'],
            how='outer'
        ).fillna(0)

        # Calculate total synapse strength
        ln_glom_full['total_synapses'] = (
            ln_glom_full['input_synapses'] + ln_glom_full['output_synapses']
        )

        # Determine connection direction
        ln_glom_full['connection_direction'] = ln_glom_full.apply(
            lambda x: 'bidirectional' if x['input_synapses'] > 0 and x['output_synapses'] > 0
                      else ('input_only' if x['input_synapses'] > 0 else 'output_only'),
            axis=1
        )

        # Convert numeric columns to int
        for col in ['input_synapses', 'output_synapses', 'num_input_pns', 'num_output_pns']:
            ln_glom_full[col] = ln_glom_full[col].astype(int)

        logger.info(f"Total LN-glomerulus associations: {len(ln_glom_full):,}")
        logger.info(f"LNs with glomerulus labels: {ln_glom_full['ln_id'].nunique():,}")

        return ln_glom_full

    def categorize_lns(self, ln_glom: pd.DataFrame, lns: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize LNs by glomerular breadth.

        Parameters
        ----------
        ln_glom : pd.DataFrame
            LN-glomerulus associations
        lns : pd.DataFrame
            Local neurons with cell type info

        Returns
        -------
        pd.DataFrame
            LN-glomerulus associations with categories
        """
        logger.info("\nCategorizing LNs by glomerular breadth...")

        # Count glomeruli per LN
        ln_glom_counts = ln_glom.groupby('ln_id')['glomerulus'].nunique().reset_index()
        ln_glom_counts.rename(columns={'glomerulus': 'num_glomeruli'}, inplace=True)

        # Categorize
        ln_glom_counts['ln_category'] = ln_glom_counts['num_glomeruli'].apply(
            categorize_ln_by_breadth
        )

        # Merge back with main data
        ln_glom = ln_glom.merge(ln_glom_counts, on='ln_id')

        # Add cell type information
        if 'cell_type' in lns.columns:
            ln_glom = ln_glom.merge(
                lns[['root_id', 'cell_type']].rename(columns={'root_id': 'ln_id'}),
                on='ln_id',
                how='left'
            )

        # Log category distribution
        logger.info("\nLN Category Distribution:")
        category_counts = ln_glom_counts['ln_category'].value_counts()
        for category, count in category_counts.items():
            pct = 100 * count / len(ln_glom_counts)
            logger.info(f"  {category}: {count:,} ({pct:.1f}%)")

        return ln_glom

    def validate_results(self, ln_glom: pd.DataFrame, lns: pd.DataFrame, pns: pd.DataFrame):
        """
        Run validation checks on the results.

        Parameters
        ----------
        ln_glom : pd.DataFrame
            LN-glomerulus associations
        lns : pd.DataFrame
            Local neurons
        pns : pd.DataFrame
            Projection neurons
        """
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION CHECKS")
        logger.info("=" * 80)

        # Check 1: LN detection
        num_lns = len(lns)
        logger.info(f"\n✓ Check 1: LN Detection")
        logger.info(f"  Found {num_lns:,} LNs")
        if num_lns < 800:
            logger.warning(f"  ⚠️  Expected 800+ LNs, found {num_lns}")
        else:
            logger.info(f"  ✅ Within expected range")

        # Check 2: Glomerulus coverage
        num_glom = ln_glom['glomerulus'].nunique()
        logger.info(f"\n✓ Check 2: Glomerulus Coverage")
        logger.info(f"  Mapped {num_glom} glomeruli")
        if not (30 <= num_glom <= 60):
            logger.warning(f"  ⚠️  Expected 30-60 glomeruli, found {num_glom}")
        else:
            logger.info(f"  ✅ Within expected range")

        # Check 3: Association quality
        ln_glom_stats = ln_glom.groupby('ln_id')['glomerulus'].nunique().describe()
        logger.info(f"\n✓ Check 3: LN Glomerular Breadth")
        logger.info(f"  Median glomeruli per LN: {ln_glom_stats['50%']:.1f}")
        logger.info(f"  Mean glomeruli per LN: {ln_glom_stats['mean']:.1f}")
        if ln_glom_stats['50%'] < 3:
            logger.warning(f"  ⚠️  LNs have fewer glomeruli than expected")
        else:
            logger.info(f"  ✅ Plausible glomerular breadth")

        # Check 4: Connection direction balance
        direction_counts = ln_glom['connection_direction'].value_counts()
        bidirectional_pct = 100 * direction_counts.get('bidirectional', 0) / len(ln_glom)
        logger.info(f"\n✓ Check 4: Connection Direction Balance")
        logger.info(f"  Bidirectional: {bidirectional_pct:.1f}%")
        logger.info(f"  Input only: {100 * direction_counts.get('input_only', 0) / len(ln_glom):.1f}%")
        logger.info(f"  Output only: {100 * direction_counts.get('output_only', 0) / len(ln_glom):.1f}%")
        if bidirectional_pct < 40:
            logger.warning(f"  ⚠️  Expected >40% bidirectional connections")
        else:
            logger.info(f"  ✅ Good bidirectional balance")

        # Check 5: Coverage
        lns_mapped = ln_glom['ln_id'].nunique()
        coverage_pct = 100 * lns_mapped / len(lns)
        logger.info(f"\n✓ Check 5: LN Mapping Coverage")
        logger.info(f"  LNs mapped: {lns_mapped:,} / {len(lns):,} ({coverage_pct:.1f}%)")
        if coverage_pct < 90:
            logger.warning(f"  ⚠️  Expected >90% coverage")
        else:
            logger.info(f"  ✅ Excellent coverage")

        logger.info("\n" + "=" * 80)

    def run(self) -> dict:
        """
        Run complete LN-glomerulus mapping analysis.

        Returns
        -------
        dict
            Results dictionary with all dataframes
        """
        logger.info("=" * 80)
        logger.info("LN-GLOMERULUS MAPPING ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Minimum synapses: {self.min_synapses}")
        if self.neuropil:
            logger.info(f"Neuropil filter: {self.neuropil}")

        # Load data
        connections = self.load_connections()

        # Identify neurons
        lns = self.identify_local_neurons()
        pns = self.identify_projection_neurons()

        if len(pns) == 0:
            logger.error("No PNs with glomerulus labels found - cannot proceed!")
            return {}

        # Map LN-glomerulus associations
        ln_glom = self.map_ln_to_glomeruli(lns, pns, connections)

        if len(ln_glom) == 0:
            logger.error("No LN-glomerulus associations found!")
            return {}

        # Categorize LNs
        ln_glom = self.categorize_lns(ln_glom, lns)

        # Validate results
        self.validate_results(ln_glom, lns, pns)

        # Export results
        self.export_results(ln_glom, lns)

        return {
            'ln_glom_associations': ln_glom,
            'lns': lns,
            'pns': pns
        }

    def export_results(self, ln_glom: pd.DataFrame, lns: pd.DataFrame):
        """
        Export results to CSV files.

        Parameters
        ----------
        ln_glom : pd.DataFrame
            LN-glomerulus associations
        lns : pd.DataFrame
            Local neurons
        """
        logger.info("\n" + "=" * 80)
        logger.info("EXPORTING RESULTS")
        logger.info("=" * 80)

        # Sort by total synapses
        ln_glom = ln_glom.sort_values('total_synapses', ascending=False)

        # File 1: Complete associations
        output_cols = [
            'ln_id', 'cell_type', 'glomerulus', 'ln_category', 'num_glomeruli',
            'input_synapses', 'output_synapses', 'total_synapses',
            'num_input_pns', 'num_output_pns', 'connection_direction'
        ]
        # Only include columns that exist
        output_cols = [col for col in output_cols if col in ln_glom.columns]

        assoc_path = self.output_dir / 'ln_glomerulus_associations.csv'
        ln_glom[output_cols].to_csv(assoc_path, index=False)
        logger.info(f"\n✅ Saved {len(ln_glom):,} LN-glomerulus associations")
        logger.info(f"   {assoc_path}")

        # File 2: Primary glomerulus per LN
        ln_primary = ln_glom.loc[ln_glom.groupby('ln_id')['total_synapses'].idxmax()]
        primary_path = self.output_dir / 'ln_primary_glomerulus.csv'
        ln_primary[output_cols].to_csv(primary_path, index=False)
        logger.info(f"\n✅ Saved {len(ln_primary):,} LNs with primary glomerulus")
        logger.info(f"   {primary_path}")

        # File 3: Glomerulus summary
        glom_summary = ln_glom.groupby('glomerulus').agg({
            'ln_id': 'nunique',
            'total_synapses': 'sum',
            'input_synapses': 'sum',
            'output_synapses': 'sum'
        }).rename(columns={'ln_id': 'num_lns'}).reset_index()
        glom_summary = glom_summary.sort_values('num_lns', ascending=False)

        summary_path = self.output_dir / 'glomerulus_ln_summary.csv'
        glom_summary.to_csv(summary_path, index=False)
        logger.info(f"\n✅ Saved summary for {len(glom_summary)} glomeruli")
        logger.info(f"   {summary_path}")

        # File 4: LN category summary
        ln_categories = ln_glom.groupby('ln_id').first()[['ln_category', 'num_glomeruli']]
        if 'cell_type' in ln_glom.columns:
            ln_categories['cell_type'] = ln_glom.groupby('ln_id')['cell_type'].first()

        category_path = self.output_dir / 'ln_categories.csv'
        ln_categories.reset_index().to_csv(category_path, index=False)
        logger.info(f"\n✅ Saved categories for {len(ln_categories)} LNs")
        logger.info(f"   {category_path}")

        logger.info("\n" + "=" * 80)
        logger.info(f"All outputs saved to: {self.output_dir}")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Map Local Neurons to glomeruli via connectivity inference"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DEFAULT_DATA_DIR,
        help='Directory containing FlyWire CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory for output files'
    )
    parser.add_argument(
        '--min-synapses',
        type=int,
        default=3,
        help='Minimum synapse threshold for connections'
    )
    parser.add_argument(
        '--neuropil',
        type=str,
        default=None,
        help='Neuropil to focus on (e.g., "AL" for antennal lobe)'
    )

    args = parser.parse_args()

    # Run analysis
    mapper = LNGlomerulusMapper(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_synapses=args.min_synapses,
        neuropil=args.neuropil
    )

    results = mapper.run()

    if results:
        logger.info("\n✅ Analysis complete!")
    else:
        logger.error("\n❌ Analysis failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
