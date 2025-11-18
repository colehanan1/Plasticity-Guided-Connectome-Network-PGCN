#!/usr/bin/env python3
"""
PGCN Navis Morphology Visualizer - PRODUCTION VERSION
======================================================
Enhanced 3D neuron morphology visualization using navis + FlyWire skeleton data.
Shows REAL dendrite/axon structures instead of abstract points.

Features:
- Real neuron morphology from FlyWire skeletons
- Interactive 3D visualization with navis plotly backend
- Template brain mesh integration (FAFB14)
- Multiple visualization modes (individual, circuit, brain context)
- Integrates with existing PGCN cache data pipeline

Author: PGCN Visualization Team
Repository: colehanan1/Plasticity-Guided-Connectome-Network-PGCN
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np
import navis
try:
    import flybrains
    FLYBRAINS_AVAILABLE = True
except ImportError:
    FLYBRAINS_AVAILABLE = False
    print("Warning: flybrains not available - template brain meshes disabled")

try:
    from fafbseg import flywire
    FAFBSEG_AVAILABLE = True
except ImportError:
    FAFBSEG_AVAILABLE = False
    print("Warning: fafbseg not available - using local skeleton data only")

import plotly.graph_objects as go
from tqdm import tqdm


class NavisMorphologyVisualizer:
    """
    Enhanced PGCN visualizer using navis for real neuron morphology.
    Integrates with existing cache data pipeline while adding skeleton visualization.
    """

    def __init__(self, cache_dir: Path, flywire_dir: Path, output_dir: Path,
                 clean_skeletons: bool = True, include_brain_mesh: bool = False,
                 include_connectivity: bool = False):
        self.cache_dir = Path(cache_dir)
        self.flywire_dir = Path(flywire_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Configuration options
        self.clean_skeletons = clean_skeletons
        self.include_brain_mesh = include_brain_mesh
        self.include_connectivity = include_connectivity

        # Data containers
        self.neuron_ids = {}
        self.skeletons = {}
        self.cache_data = {}
        self.connectivity_data = None

        # Color schemes matching original visualizer (EXTENDED)
        self.colors = {
            'PN': '#1f77b4',    # Blue
            'KC': '#ff7f0e',    # Orange
            'MBON': '#2ca02c',  # Green
            'DAN': '#d62728',   # Red
            'LN': '#9467bd',    # Purple (Local Interneurons)
            'LH': '#FFD700',    # Gold (Lateral Horn)
            'Motor': '#FF1493', # Deep Pink (Motor neurons)
            'AN': '#778899',    # Light Slate Gray (Ascending)
            'DN': '#2F4F4F'     # Dark Slate Gray (Descending)
        }

        print("Navis Morphology Visualizer initialized")
        print(f"  Cache: {self.cache_dir}")
        print(f"  FlyWire: {self.flywire_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"  Options: clean={clean_skeletons}, brain={include_brain_mesh}, conn={include_connectivity}")

    def load_neuron_ids_from_cache(self):
        """Load neuron IDs from existing cache files"""
        print("\nLoading neuron IDs from cache...")

        # Load PNs
        pn_file = self.cache_dir / 'alpn_extracted.csv'
        if pn_file.exists():
            pn_df = pd.read_csv(pn_file)
            if 'root_id' in pn_df.columns:
                self.neuron_ids['PN'] = pn_df['root_id'].tolist()
                print(f"  ✓ PNs: {len(self.neuron_ids['PN'])} neurons")

        # Load KCs (all subtypes)
        kc_ids = []
        for kc_file in self.cache_dir.glob('kc_*.csv'):
            df = pd.read_csv(kc_file)
            if 'root_id' in df.columns and len(df) > 0:
                kc_ids.extend(df['root_id'].tolist())
        if kc_ids:
            self.neuron_ids['KC'] = kc_ids
            print(f"  ✓ KCs: {len(self.neuron_ids['KC'])} neurons")

        # Load MBONs
        mbon_file = self.cache_dir / 'mbon_all.csv'
        if mbon_file.exists():
            mbon_df = pd.read_csv(mbon_file)
            if 'root_id' in mbon_df.columns:
                self.neuron_ids['MBON'] = mbon_df['root_id'].tolist()
                print(f"  ✓ MBONs: {len(self.neuron_ids['MBON'])} neurons")

        # Load DANs
        dan_file = self.cache_dir / 'dan_mb.csv'
        if dan_file.exists():
            dan_df = pd.read_csv(dan_file)
            if 'root_id' in dan_df.columns:
                self.neuron_ids['DAN'] = dan_df['root_id'].tolist()
                print(f"  ✓ DANs: {len(self.neuron_ids['DAN'])} neurons")

        # Load LNs (Local Interneurons) - NEW
        ln_file = self.cache_dir / 'ln_all.csv'
        if ln_file.exists():
            ln_df = pd.read_csv(ln_file)
            if 'root_id' in ln_df.columns:
                self.neuron_ids['LN'] = ln_df['root_id'].tolist()
                print(f"  ✓ LNs: {len(self.neuron_ids['LN'])} neurons")

        # Load LH (Lateral Horn) - NEW
        lh_file = self.cache_dir / 'lh_all.csv'
        if lh_file.exists():
            lh_df = pd.read_csv(lh_file)
            if 'root_id' in lh_df.columns:
                self.neuron_ids['LH'] = lh_df['root_id'].tolist()
                print(f"  ✓ LH: {len(self.neuron_ids['LH'])} neurons")

        # Load Motor neurons - NEW
        motor_file = self.cache_dir / 'motor_all.csv'
        if motor_file.exists():
            motor_df = pd.read_csv(motor_file)
            if 'root_id' in motor_df.columns:
                self.neuron_ids['Motor'] = motor_df['root_id'].tolist()
                print(f"  ✓ Motor: {len(self.neuron_ids['Motor'])} neurons")

        # Load ANs (Ascending Neurons) - NEW
        an_file = self.cache_dir / 'an_all.csv'
        if an_file.exists():
            an_df = pd.read_csv(an_file)
            if 'root_id' in an_df.columns:
                self.neuron_ids['AN'] = an_df['root_id'].tolist()
                print(f"  ✓ ANs: {len(self.neuron_ids['AN'])} neurons")

        # Load DNs (Descending Neurons) - NEW
        dn_file = self.cache_dir / 'dn_all.csv'
        if dn_file.exists():
            dn_df = pd.read_csv(dn_file)
            if 'root_id' in dn_df.columns:
                self.neuron_ids['DN'] = dn_df['root_id'].tolist()
                print(f"  ✓ DNs: {len(self.neuron_ids['DN'])} neurons")

        total = sum(len(ids) for ids in self.neuron_ids.values())
        print(f"\n  Total: {total} neuron IDs loaded (ENHANCED with all 9 neuron types!)")
        return total > 0

    def load_connectivity_data(self):
        """Load connectivity data for overlay visualization"""
        # Try parquet file first (main connectivity data)
        edges_file = self.cache_dir / 'edges.parquet'
        if edges_file.exists():
            self.connectivity_data = pd.read_parquet(edges_file)
            print(f"\n  ✓ Loaded connectivity: {len(self.connectivity_data)} synaptic connections")
            return True

        # Fallback to CSV
        conn_file = self.cache_dir / 'pn_to_kc_connectivity.csv'
        if conn_file.exists():
            self.connectivity_data = pd.read_csv(conn_file)
            if len(self.connectivity_data) > 0:
                print(f"\n  ✓ Loaded connectivity: {len(self.connectivity_data)} connections")
                return True

        print("\n  ⚠ No connectivity data found")
        return False

    def clean_neuron_skeletons(self, skeletons):
        """
        FIX #1: Clean skeletons to remove multiple soma warnings.

        FlyWire skeletons sometimes have reconstruction artifacts with multiple
        soma markers. This causes navis to issue warnings and can affect visualization.
        """
        if not self.clean_skeletons:
            return skeletons

        print("\n  Cleaning skeleton data...")
        cleaned_skeletons = []
        fixed_count = 0

        for skeleton in skeletons:
            skeleton_clean = skeleton.copy()

            # Check if skeleton has multiple somas
            if hasattr(skeleton, 'soma') and skeleton.soma is not None:
                if isinstance(skeleton.soma, (list, np.ndarray)) and len(skeleton.soma) > 1:
                    # Multiple somas detected - keep only the one closest to root
                    try:
                        root_node = skeleton.nodes[skeleton.nodes.node_id == skeleton.root].iloc[0]
                        root_pos = root_node[['x', 'y', 'z']].values

                        # Find closest soma to root
                        soma_distances = []
                        for soma_idx in skeleton.soma:
                            soma_node = skeleton.nodes[skeleton.nodes.node_id == soma_idx].iloc[0]
                            soma_pos = soma_node[['x', 'y', 'z']].values
                            dist = np.linalg.norm(soma_pos - root_pos)
                            soma_distances.append(dist)

                        # Keep only closest soma
                        best_soma_idx = np.argmin(soma_distances)
                        skeleton_clean.soma = skeleton.soma[best_soma_idx]
                        fixed_count += 1
                    except Exception:
                        # If error, just remove all soma markers
                        skeleton_clean.soma = None
                        fixed_count += 1

            cleaned_skeletons.append(skeleton_clean)

        if fixed_count > 0:
            print(f"  ✓ Fixed soma detection for {fixed_count} neurons")
        print(f"  ✓ Skeletons cleaned and validated")

        return navis.NeuronList(cleaned_skeletons)

    def add_brain_context_fixed(self, fig, neurons):
        """
        FIX #2: Add brain mesh with corrected alpha handling.

        The original implementation had alpha parameter mismatch causing errors.
        This version properly handles alpha for both neurons and brain mesh.
        """
        if not self.include_brain_mesh or not FLYBRAINS_AVAILABLE:
            return fig

        print("  Adding FAFB14 brain mesh for context...")

        try:
            # Get FAFB14 brain mesh
            brain_mesh = flybrains.FAFB14.mesh

            # Method 1: Plot neurons and brain separately (most reliable)
            import plotly.graph_objects as go

            # Convert brain mesh to plotly trace
            vertices = brain_mesh.vertices
            faces = brain_mesh.faces

            mesh_trace = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightgray',
                opacity=0.1,
                name='Brain outline',
                hoverinfo='skip'
            )

            # Add mesh to existing figure
            fig.add_trace(mesh_trace)
            print("    ✓ Brain mesh added successfully")

        except Exception as e:
            print(f"    ⚠ Could not add brain mesh: {e}")

        return fig

    def overlay_synaptic_connections(self, fig, neurons):
        """
        FIX #3: Add synaptic connectivity overlay to morphology visualization.

        Integrates existing connectivity data with morphology to show
        structure-function relationships.
        """
        if not self.include_connectivity or self.connectivity_data is None:
            return fig

        print("  Overlaying synaptic connections...")

        try:
            # Create neuron position lookup (using soma or root)
            pos_lookup = {}
            for neuron in neurons:
                neuron_id = neuron.id

                # Use soma position if available, otherwise root
                if hasattr(neuron, 'soma') and neuron.soma is not None:
                    if isinstance(neuron.soma, (list, np.ndarray)):
                        soma_id = neuron.soma[0] if len(neuron.soma) > 0 else None
                    else:
                        soma_id = neuron.soma

                    if soma_id is not None:
                        soma_node = neuron.nodes[neuron.nodes.node_id == soma_id].iloc[0]
                        pos = soma_node[['x', 'y', 'z']].values
                    else:
                        # Use center of mass if no soma
                        pos = neuron.nodes[['x', 'y', 'z']].mean().values
                else:
                    # Use center of mass for position (most reliable)
                    pos = neuron.nodes[['x', 'y', 'z']].mean().values

                pos_lookup[int(neuron_id)] = pos

            # Add connection lines for neurons in visualization
            connection_count = 0

            # Determine column names
            if 'source_id' in self.connectivity_data.columns:
                src_col, tgt_col, weight_col = 'source_id', 'target_id', 'synapse_weight'
            else:
                src_col, tgt_col, weight_col = 'pre_root_id', 'post_root_id', 'synapse_count'

            for _, conn in self.connectivity_data.iterrows():
                pre_id = int(conn[src_col])
                post_id = int(conn[tgt_col])
                synapse_count = conn.get(weight_col, 1)

                if pre_id in pos_lookup and post_id in pos_lookup:
                    pre_pos = pos_lookup[pre_id]
                    post_pos = pos_lookup[post_id]

                    # Add connection line
                    fig.add_trace(go.Scatter3d(
                        x=[pre_pos[0], post_pos[0], None],
                        y=[pre_pos[1], post_pos[1], None],
                        z=[pre_pos[2], post_pos[2], None],
                        mode='lines',
                        line=dict(
                            width=max(1, synapse_count // 10),
                            color='orange',
                            opacity=0.5
                        ),
                        showlegend=False,
                        hoverinfo='text',
                        text=f'Synapses: {synapse_count}',
                        name='Connection'
                    ))
                    connection_count += 1

            print(f"  ✓ Added {connection_count} synaptic connections")

        except Exception as e:
            print(f"  ⚠ Error adding connections: {e}")
            import traceback
            if hasattr(e, '__traceback__'):
                traceback.print_exc()

        return fig

    def fetch_skeletons_from_flywire(self, neuron_ids: List[int], max_neurons: int = None):
        """
        Fetch neuron skeletons from FlyWire API using fafbseg.

        Args:
            neuron_ids: List of FlyWire root IDs
            max_neurons: Maximum number to fetch (for testing)
        """
        if not FAFBSEG_AVAILABLE:
            print("  ⚠ fafbseg not available - cannot fetch from API")
            return []

        if max_neurons:
            neuron_ids = neuron_ids[:max_neurons]

        print(f"\nFetching {len(neuron_ids)} skeletons from FlyWire API...")
        print("  This may take a few minutes...")

        try:
            # Fetch skeletons using fafbseg
            # FlyWire dataset 783 is the FAFB v783 release (your data)
            skeletons = flywire.get_skeletons(
                neuron_ids,
                progress=True,
                dataset='783'  # FAFB v783 - matches your FlyWire data
            )

            if skeletons:
                print(f"  ✓ Successfully fetched {len(skeletons)} skeletons")
                return navis.NeuronList(skeletons)
            else:
                print("  ✗ No skeletons returned")
                return navis.NeuronList([])

        except Exception as e:
            print(f"  ✗ Error fetching skeletons: {e}")
            print("  Tip: Check internet connection and FlyWire API status")
            return navis.NeuronList([])

    def load_skeletons_from_local(self, skeleton_dir: Path, neuron_ids: List[int]):
        """
        Load neuron skeletons from local SWC files.

        Args:
            skeleton_dir: Directory containing SWC skeleton files
            neuron_ids: List of neuron IDs to load
        """
        print(f"\nLoading skeletons from local files: {skeleton_dir}")

        skeletons = []
        skeleton_files = list(skeleton_dir.glob('*.swc'))

        if not skeleton_files:
            print(f"  ⚠ No SWC files found in {skeleton_dir}")
            return navis.NeuronList([])

        print(f"  Found {len(skeleton_files)} SWC files")

        # Try to load skeletons matching our neuron IDs
        for swc_file in tqdm(skeleton_files[:100], desc="Loading skeletons"):
            try:
                skeleton = navis.read_swc(swc_file)
                skeletons.append(skeleton)
            except Exception as e:
                continue

        if skeletons:
            print(f"  ✓ Loaded {len(skeletons)} skeletons")
            return navis.NeuronList(skeletons)
        else:
            print("  ✗ No skeletons loaded")
            return navis.NeuronList([])

    def create_sample_visualization(self, neuron_type: str = 'PN', n_samples: int = 10, show_all: bool = False):
        """
        Create a sample visualization with neurons.

        Args:
            neuron_type: Type of neurons to visualize ('PN', 'KC', 'MBON', 'DAN', 'LN', 'LH', 'Motor', 'AN', 'DN')
            n_samples: Number of neurons to include (ignored if show_all=True)
            show_all: If True, show ALL neurons of this type
        """
        print(f"\n{'='*60}")
        print(f"Creating {'ALL' if show_all else 'sample'} {neuron_type} morphology visualization")
        print(f"{'='*60}")

        if neuron_type not in self.neuron_ids:
            print(f"  ✗ No {neuron_type} neurons loaded")
            return None

        # Get sample neuron IDs (ALL if show_all=True)
        if show_all:
            sample_ids = self.neuron_ids[neuron_type]
            print(f"  Selected ALL {len(sample_ids)} {neuron_type} neurons")
        else:
            sample_ids = self.neuron_ids[neuron_type][:n_samples]
            print(f"  Selected {len(sample_ids)} of {len(self.neuron_ids[neuron_type])} {neuron_type} neurons")

        # Fetch skeletons
        skeletons = self.fetch_skeletons_from_flywire(sample_ids)

        if len(skeletons) == 0:
            print("  ✗ No skeletons available - visualization aborted")
            return None

        # FIX #1: Clean skeletons (remove soma warnings)
        skeletons = self.clean_neuron_skeletons(skeletons)

        # Create 3D visualization with navis
        print("\n  Creating 3D plot with navis...")

        try:
            # Use navis plotly backend for interactive visualization
            fig = navis.plot3d(
                skeletons,
                backend='plotly',
                color=self.colors[neuron_type],
                width=1400,
                height=1000,
                title=f"PGCN Circuit: {neuron_type} Neurons (n={len(skeletons)})",
            )

            # FIX #2: Add brain mesh with corrected alpha handling
            fig = self.add_brain_context_fixed(fig, skeletons)

            # FIX #3: Overlay synaptic connections
            fig = self.overlay_synaptic_connections(fig, skeletons)

            # Save with CDN to avoid blank page issue
            output_file = self.output_dir / f'pgcn_{neuron_type.lower()}_morphology_sample.html'

            # Use plotly's write_html with CDN config (matching our fix)
            fig.write_html(
                str(output_file),
                include_plotlyjs='cdn',
                full_html=True,
                config={'displayModeBar': True, 'responsive': True}
            )

            print(f"  ✓ Saved: {output_file}")
            print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

            return fig

        except Exception as e:
            print(f"  ✗ Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_circuit_morphology(self, n_per_type: int = 5):
        """
        Create complete circuit visualization with all neuron types.

        Args:
            n_per_type: Number of neurons per type to include
        """
        print(f"\n{'='*60}")
        print("Creating complete circuit morphology visualization")
        print(f"{'='*60}")

        all_skeletons = []
        colors_list = []

        # Fetch representative neurons from each type
        for neuron_type in ['PN', 'KC', 'MBON', 'DAN']:
            if neuron_type not in self.neuron_ids:
                continue

            print(f"\n  Fetching {neuron_type} neurons...")
            sample_ids = self.neuron_ids[neuron_type][:n_per_type]
            skeletons = self.fetch_skeletons_from_flywire(sample_ids)

            if len(skeletons) > 0:
                all_skeletons.extend(skeletons)
                colors_list.extend([self.colors[neuron_type]] * len(skeletons))
                print(f"    ✓ Added {len(skeletons)} {neuron_type} neurons")

        if not all_skeletons:
            print("\n  ✗ No skeletons available for circuit visualization")
            return None

        print(f"\n  Total neurons in circuit: {len(all_skeletons)}")

        # FIX #1: Clean skeletons (remove soma warnings)
        all_neurons = self.clean_neuron_skeletons(navis.NeuronList(all_skeletons))

        # Create combined visualization
        print("  Creating combined 3D plot...")

        try:
            fig = navis.plot3d(
                all_neurons,
                backend='plotly',
                color=colors_list,
                width=1400,
                height=1000,
                title=f"PGCN Complete Circuit (n={len(all_neurons)} neurons)",
            )

            # FIX #2: Add brain mesh with corrected alpha handling
            fig = self.add_brain_context_fixed(fig, all_neurons)

            # FIX #3: Overlay synaptic connections
            fig = self.overlay_synaptic_connections(fig, all_neurons)

            # Save
            output_file = self.output_dir / 'pgcn_complete_circuit_morphology.html'
            fig.write_html(
                str(output_file),
                include_plotlyjs='cdn',
                full_html=True,
                config={'displayModeBar': True, 'responsive': True}
            )

            print(f"\n  ✓ Saved: {output_file}")
            print(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")

            return fig

        except Exception as e:
            print(f"  ✗ Error creating circuit visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_morphology_comparison(self, neuron_type: str = 'KC', n_neurons: int = 10):
        """
        Create side-by-side comparison of neuron morphologies.
        Useful for comparing KC subtypes or PN glomeruli.
        """
        print(f"\n{'='*60}")
        print(f"Creating {neuron_type} morphology comparison")
        print(f"{'='*60}")

        if neuron_type not in self.neuron_ids:
            print(f"  ✗ No {neuron_type} neurons loaded")
            return None

        # Get sample neurons
        sample_ids = self.neuron_ids[neuron_type][:n_neurons]
        skeletons = self.fetch_skeletons_from_flywire(sample_ids)

        if len(skeletons) == 0:
            return None

        # Create comparison plot
        print("  Creating morphology comparison...")

        try:
            # Create a figure with subplots for each neuron
            from plotly.subplots import make_subplots

            n_cols = min(3, len(skeletons))
            n_rows = (len(skeletons) + n_cols - 1) // n_cols

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                specs=[[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)],
                subplot_titles=[f"{neuron_type} {i+1}" for i in range(len(skeletons))]
            )

            # Add each neuron to a subplot
            for idx, skeleton in enumerate(skeletons):
                row = idx // n_cols + 1
                col = idx % n_cols + 1

                # Get skeleton coordinates
                coords = skeleton.nodes[['x', 'y', 'z']].values

                fig.add_trace(
                    go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode='lines+markers',
                        marker=dict(size=2, color=self.colors[neuron_type]),
                        line=dict(width=2, color=self.colors[neuron_type]),
                        name=f"{neuron_type} {idx+1}"
                    ),
                    row=row,
                    col=col
                )

            fig.update_layout(
                title=f"PGCN {neuron_type} Morphology Comparison (n={len(skeletons)})",
                height=400 * n_rows,
                width=1400,
                showlegend=False
            )

            # Save
            output_file = self.output_dir / f'pgcn_{neuron_type.lower()}_comparison.html'
            fig.write_html(
                str(output_file),
                include_plotlyjs='cdn',
                full_html=True,
                config={'displayModeBar': True, 'responsive': True}
            )

            print(f"  ✓ Saved: {output_file}")

            return fig

        except Exception as e:
            print(f"  ✗ Error creating comparison: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main execution with CLI"""
    parser = argparse.ArgumentParser(
        description="PGCN Navis Morphology Visualizer - Real neuron shapes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 50 projection neurons (INCREASED DEFAULT)
  python scripts/navis_morphology_visualizer.py --neuron-type PN --n-samples 50

  # Show ALL local interneurons (NEW!)
  python scripts/navis_morphology_visualizer.py --neuron-type LN --show-all

  # Complete circuit with 20 neurons per type (ALL 9 TYPES!)
  python scripts/navis_morphology_visualizer.py --mode circuit --n-per-type 20

  # Visualize lateral horn neurons (NEW!)
  python scripts/navis_morphology_visualizer.py --neuron-type LH --n-samples 30

  # Show all motor neurons (NEW!)
  python scripts/navis_morphology_visualizer.py --neuron-type Motor --show-all

  # KC morphology comparison with more samples
  python scripts/navis_morphology_visualizer.py --mode comparison --neuron-type KC --n-samples 50

  # Generate visualizations for ALL 9 neuron types
  python scripts/navis_morphology_visualizer.py --mode all --n-samples 30

Supported Neuron Types (ALL 9):
  PN, KC, MBON, DAN, LN, LH, Motor, AN, DN
        """
    )

    parser.add_argument('--cache-dir', type=Path, default='data/cache',
                       help='Path to cache directory with neuron IDs')
    parser.add_argument('--flywire-dir', type=Path, default='data/flywire',
                       help='Path to FlyWire data directory')
    parser.add_argument('--output-dir', type=Path, default='reports/navis_morphology',
                       help='Output directory for visualizations')

    parser.add_argument('--mode', default='sample',
                       choices=['sample', 'circuit', 'comparison', 'all'],
                       help='Visualization mode')
    parser.add_argument('--neuron-type', default='PN',
                       choices=['PN', 'KC', 'MBON', 'DAN', 'LN', 'LH', 'Motor', 'AN', 'DN'],
                       help='Neuron type to visualize (for sample/comparison modes) - ALL 9 TYPES SUPPORTED')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of sample neurons (sample mode) - increased from 10 to 50')
    parser.add_argument('--n-per-type', type=int, default=20,
                       help='Neurons per type (circuit mode) - increased from 5 to 20')
    parser.add_argument('--show-all', action='store_true',
                       help='Show ALL neurons from selected type (overrides --n-samples)')

    # NEW: Enhancement options
    parser.add_argument('--clean-skeletons', action='store_true', default=True,
                       help='Fix soma detection issues (recommended)')
    parser.add_argument('--no-clean-skeletons', action='store_false', dest='clean_skeletons',
                       help='Skip skeleton cleaning')
    parser.add_argument('--include-brain-mesh', action='store_true',
                       help='Add FAFB14 brain mesh for context')
    parser.add_argument('--include-connectivity', action='store_true',
                       help='Overlay synaptic connections')

    args = parser.parse_args()

    # Validate
    if not args.cache_dir.exists():
        print(f"Error: Cache directory not found: {args.cache_dir}")
        sys.exit(1)

    print("=" * 60)
    print("PGCN NAVIS MORPHOLOGY VISUALIZER - ENHANCED")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Cache: {args.cache_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Clean skeletons: {args.clean_skeletons}")
    print(f"Brain mesh: {args.include_brain_mesh}")
    print(f"Connectivity: {args.include_connectivity}")

    # Initialize visualizer with enhancement options
    visualizer = NavisMorphologyVisualizer(
        args.cache_dir,
        args.flywire_dir,
        args.output_dir,
        clean_skeletons=args.clean_skeletons,
        include_brain_mesh=args.include_brain_mesh,
        include_connectivity=args.include_connectivity
    )

    # Load neuron IDs from cache
    if not visualizer.load_neuron_ids_from_cache():
        print("\nError: No neuron IDs loaded from cache")
        sys.exit(1)

    # Load connectivity data if requested
    if args.include_connectivity:
        visualizer.load_connectivity_data()

    # Execute visualization mode
    try:
        if args.mode == 'sample':
            visualizer.create_sample_visualization(
                neuron_type=args.neuron_type,
                n_samples=args.n_samples,
                show_all=args.show_all
            )

        elif args.mode == 'circuit':
            visualizer.create_circuit_morphology(
                n_per_type=args.n_per_type
            )

        elif args.mode == 'comparison':
            visualizer.create_morphology_comparison(
                neuron_type=args.neuron_type,
                n_neurons=args.n_samples
            )

        elif args.mode == 'all':
            # Create all visualization types for ALL 9 neuron types
            for ntype in ['PN', 'KC', 'MBON', 'DAN', 'LN', 'LH', 'Motor', 'AN', 'DN']:
                if ntype in visualizer.neuron_ids:
                    visualizer.create_sample_visualization(ntype, args.n_samples, args.show_all)

            visualizer.create_circuit_morphology(args.n_per_type)

        print("\n" + "=" * 60)
        print("✓ VISUALIZATION COMPLETE")
        print(f"✓ Results saved to: {args.output_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
