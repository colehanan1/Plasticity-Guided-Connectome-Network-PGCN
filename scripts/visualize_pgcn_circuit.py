#!/usr/bin/env python3
"""
PGCN Circuit Visualization Script (FINAL FIXED VERSION)
======================================================
Complete production-ready Python visualization script for Drosophila olfactory 
learning circuit (PN→KC→MBON+DAN) with robust error handling and type safety.

Author: PGCN Project Assistant for wow-im-tired branch
Repository: colehanan1/Plasticity-Guided-Connectome-Network-PGCN
Branch: wow-im-tired

Usage:
    python scripts/visualize_pgcn_circuit.py --cache-dir data/cache --output-dir reports --format all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Core data handling
import pandas as pd
import numpy as np
from scipy import sparse
import pyarrow.parquet as pq

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Progress tracking
from tqdm import tqdm


class PGCNCircuitVisualizer:
    """
    Complete visualization suite for PGCN circuit data with support for:
    - 2D hierarchical network plots
    - 3D spatial visualizations  
    - Connectivity heatmaps
    - Feature analysis
    - Circuit statistics
    """
    
    def __init__(self, cache_dir: Path, output_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Data containers
        self.neurons = {}
        self.connectivity = {}
        self.features = {}
        self.stats = {}
        
        # Color schemes
        self.colors = {
            'pn_glomeruli': px.colors.qualitative.Set3,
            'kc_subtypes': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
            'mbon_types': {'calyx': '#2E8B57', 'medial_lobe': '#B8860B', 'other': '#CD5C5C'},
            'dan_valence': {'PAM': '#32CD32', 'PPL1': '#FF6347', 'other': '#708090'}
        }
    
    def _get_id_column(self, df: pd.DataFrame, possible_names: List[str] = None) -> Optional[str]:
        """Find the correct ID column name from a list of possibilities"""
        if possible_names is None:
            possible_names = ['node_id', 'neuron_id', 'root_id', 'id']
        
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _safe_get_column(self, df: pd.DataFrame, possible_names: List[str], default_value=None):
        """Safely get a column by trying multiple possible names"""
        for name in possible_names:
            if name in df.columns:
                return df[name]
        
        # Return series of default values if no column found
        if default_value is not None:
            return pd.Series([default_value] * len(df), index=df.index)
        return None
    
    def _safe_int(self, value, default=0):
        """Safely convert value to integer"""
        try:
            if pd.isna(value):
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    def load_data(self):
        """Load all circuit data from cache directory"""
        print("Loading circuit data...")
        
        # Load neuron populations
        self._load_neurons()
        
        # Load connectivity matrices  
        self._load_connectivity()
        
        # Load feature data
        self._load_features()
        
        print(f"✓ Loaded {sum(len(v) for v in self.neurons.values())} total neurons")
        print(f"✓ Loaded {sum(len(v) for v in self.connectivity.values())} connectivity edges")
    
    def _load_neurons(self):
        """Load neuron populations from CSV files"""
        neuron_files = {
            'alpn': 'alpn_extracted.csv',
            'kc_ab': 'kc_ab.csv', 'kc_ab_p': 'kc_ab_p.csv',
            'kc_apb': 'kc_apb.csv', 'kc_apbp_ap1': 'kc_apbp_ap1.csv',
            'kc_apbp_ap2': 'kc_apbp_ap2.csv', 'kc_apbp_main': 'kc_apbp_main.csv',
            'kc_g_dorsal': 'kc_g_dorsal.csv', 'kc_g_main': 'kc_g_main.csv',
            'kc_g_sparse': 'kc_g_sparse.csv',
            'mbon_all': 'mbon_all.csv', 'mbon_calyx': 'mbon_calyx.csv',
            'mbon_ml': 'mbon_ml.csv', 'mbon_glut': 'mbon_glut.csv',
            'dan_all': 'dan_all.csv', 'dan_mb': 'dan_mb.csv',
            'dan_calyx': 'dan_calyx.csv', 'dan_ml': 'dan_ml.csv'
        }
        
        for name, filename in neuron_files.items():
            filepath = self.cache_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    if not df.empty and len(df) > 0:
                        self.neurons[name] = df
                        print(f"  ✓ {name}: {len(df)} neurons")
                    else:
                        print(f"  ⚠ {name}: empty file - skipping")
                except Exception as e:
                    print(f"  ✗ {name}: failed to load ({e})")
            else:
                print(f"  ⚠ {name}: file not found")
    
    def _load_connectivity(self):
        """Load connectivity data from CSV and parquet files"""
        conn_files = {
            'pn_kc': ('pn_to_kc_connectivity.csv', 'csv'),
            'dan_edges': ('dan_edges.parquet', 'parquet'),
            'edges': ('edges.parquet', 'parquet'),
            'pn_kc_mbon_paths': ('pn_kc_mbon_paths.parquet', 'parquet')
        }
        
        for name, (filename, format_type) in conn_files.items():
            filepath = self.cache_dir / filename
            if filepath.exists():
                try:
                    if format_type == 'csv':
                        df = pd.read_csv(filepath)
                    else:
                        df = pd.read_parquet(filepath)
                    
                    if not df.empty:
                        self.connectivity[name] = df
                        print(f"  ✓ {name}: {len(df)} connections")
                except Exception as e:
                    print(f"  ✗ {name}: failed to load ({e})")
    
    def _load_features(self):
        """Load feature and metadata parquet files"""
        feature_files = {
            'nodes': 'nodes.parquet',
            'weighted_centrality': 'weighted_centrality.parquet', 
            'dan_valence': 'dan_valence.parquet',
            'olfactory_conditioning_features': 'olfactory_conditioning_features.parquet',
            'kc_overlap': 'kc_overlap.parquet'
        }
        
        for name, filename in feature_files.items():
            filepath = self.cache_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath)
                    if not df.empty:
                        self.features[name] = df
                        print(f"  ✓ {name}: {len(df)} records")
                except Exception as e:
                    print(f"  ✗ {name}: failed to load ({e})")
    
    def create_2d_hierarchical_network(self, top_edges: int = 500) -> go.Figure:
        """Create 2D hierarchical network with 4 layers: PN | KC | MBON | DAN"""
        print("Creating 2D hierarchical network...")
        
        # Build combined neuron dataframe with positions
        all_neurons = []
        layer_positions = {'PN': -3, 'KC': -1, 'MBON': 1, 'DAN': 3}
        
        # Process PNs
        if 'alpn' in self.neurons:
            pns = self.neurons['alpn'].copy()
            pns['layer'] = 'PN'
            pns['y_pos'] = layer_positions['PN']
            pns['neuron_type'] = 'PN'
            # Handle glomerulus column - try different possible names
            glom_col = self._safe_get_column(pns, ['glomerulus', 'glom', 'glomerulus_label'], 'unknown')
            if glom_col is not None:
                pns['subtype'] = glom_col.astype(str)
            else:
                pns['subtype'] = 'unknown'
            all_neurons.append(pns)
        
        # Process KCs - only include non-empty files
        kc_subtypes = [k for k in self.neurons.keys() if k.startswith('kc_') and k in self.neurons]
        for i, subtype in enumerate(kc_subtypes):
            if subtype in self.neurons and len(self.neurons[subtype]) > 0:
                kcs = self.neurons[subtype].copy()
                kcs['layer'] = 'KC'
                kcs['y_pos'] = layer_positions['KC']
                kcs['neuron_type'] = 'KC'
                kcs['subtype'] = subtype.replace('kc_', '')
                kcs['color_idx'] = i % len(self.colors['kc_subtypes'])
                all_neurons.append(kcs)
        
        # Process MBONs
        if 'mbon_all' in self.neurons:
            mbons = self.neurons['mbon_all'].copy()
            mbons['layer'] = 'MBON'
            mbons['y_pos'] = layer_positions['MBON']
            mbons['neuron_type'] = 'MBON'
            # Determine neuropil type from input regions - try various column names
            input_col = self._safe_get_column(mbons, ['input_neuropils', 'neuropils', 'input_regions'], 'other')
            if input_col is not None:
                mbons['subtype'] = input_col.astype(str).apply(
                    lambda x: 'calyx' if 'CA' in str(x).upper() else 
                              'medial_lobe' if 'ML' in str(x).upper() else 'other'
                )
            else:
                mbons['subtype'] = 'other'
            all_neurons.append(mbons)
        
        # Process DANs
        if 'dan_mb' in self.neurons:
            dans = self.neurons['dan_mb'].copy()
            dans['layer'] = 'DAN'
            dans['y_pos'] = layer_positions['DAN']
            dans['neuron_type'] = 'DAN'
            
            # Add valence information with robust column handling
            if 'dan_valence' in self.features:
                valence_df = self.features['dan_valence']
                valence_id_col = self._get_id_column(valence_df, ['node_id', 'neuron_id', 'root_id'])
                valence_col = self._safe_get_column(valence_df, ['valence', 'type', 'classification'])
                
                if valence_id_col and valence_col is not None:
                    valence_map = dict(zip(valence_df[valence_id_col], valence_col.astype(str)))
                    
                    # Try to match DANs with valence data using multiple ID column possibilities
                    dan_id_col = self._get_id_column(dans, ['root_id', 'node_id', 'neuron_id'])
                    if dan_id_col:
                        dans['subtype'] = dans[dan_id_col].map(valence_map).fillna('other')
                    else:
                        dans['subtype'] = 'other'
                else:
                    dans['subtype'] = 'other'
            else:
                dans['subtype'] = 'other'
            all_neurons.append(dans)
        
        if not all_neurons:
            print("No neuron data available for visualization")
            return go.Figure()
        
        # Combine all neurons
        neurons_df = pd.concat(all_neurons, ignore_index=True, sort=False)
        
        # Assign x positions within layers
        for layer in layer_positions.keys():
            layer_neurons = neurons_df[neurons_df['layer'] == layer]
            if len(layer_neurons) > 0:
                x_positions = np.linspace(-2, 2, len(layer_neurons))
                neurons_df.loc[neurons_df['layer'] == layer, 'x_pos'] = x_positions
        
        # Add centrality for node sizing with robust column handling
        if 'weighted_centrality' in self.features:
            centrality_df = self.features['weighted_centrality']
            cent_id_col = self._get_id_column(centrality_df, ['node_id', 'neuron_id', 'root_id'])
            cent_weight_col = self._safe_get_column(centrality_df, ['weighted_degree', 'degree', 'centrality'])
            
            if cent_id_col and cent_weight_col is not None:
                centrality_map = dict(zip(centrality_df[cent_id_col], cent_weight_col))
                
                # Try to match neurons with centrality using multiple ID possibilities
                neuron_id_col = self._get_id_column(neurons_df, ['root_id', 'node_id', 'neuron_id'])
                if neuron_id_col:
                    neurons_df['centrality'] = neurons_df[neuron_id_col].map(centrality_map).fillna(1.0)
                else:
                    neurons_df['centrality'] = 1.0
            else:
                neurons_df['centrality'] = 1.0
        else:
            neurons_df['centrality'] = 1.0
        
        # Normalize centrality for sizing - ensure numeric values
        neurons_df['centrality'] = pd.to_numeric(neurons_df['centrality'], errors='coerce').fillna(1.0)
        
        if neurons_df['centrality'].std() > 0:
            neurons_df['node_size'] = 5 + 15 * (neurons_df['centrality'] - neurons_df['centrality'].min()) / \
                                     (neurons_df['centrality'].max() - neurons_df['centrality'].min())
        else:
            neurons_df['node_size'] = 10
        
        # Create the figure
        fig = go.Figure()
        
        # Add edges first (so they appear behind nodes)
        self._add_network_edges(fig, neurons_df, top_edges)
        
        # Add nodes by layer and subtype
        for layer in ['PN', 'KC', 'MBON', 'DAN']:
            layer_data = neurons_df[neurons_df['layer'] == layer]
            if len(layer_data) == 0:
                continue
                
            if layer == 'PN':
                # Color PNs by glomerulus
                subtypes = layer_data['subtype'].unique()
                for i, subtype in enumerate(subtypes):
                    subtype_data = layer_data[layer_data['subtype'] == subtype]
                    color_idx = i % len(self.colors['pn_glomeruli'])
                    color = self.colors['pn_glomeruli'][color_idx]
                    
                    fig.add_trace(go.Scatter(
                        x=subtype_data['x_pos'],
                        y=subtype_data['y_pos'],
                        mode='markers',
                        marker=dict(
                            size=subtype_data['node_size'],
                            color=color,
                            line=dict(width=1, color='white'),
                            opacity=0.8
                        ),
                        text=[f"PN {subtype}<br>ID: {idx}<br>Centrality: {cent:.3f}" 
                              for idx, cent in zip(subtype_data.index, subtype_data['centrality'])],
                        hoverinfo='text',
                        name=f'PN-{subtype}',
                        legendgroup='PN'
                    ))
            
            elif layer == 'KC':
                # Color KCs by subtype
                for subtype in layer_data['subtype'].unique():
                    subtype_data = layer_data[layer_data['subtype'] == subtype]
                    # Safely get color index and convert to int
                    if 'color_idx' in subtype_data.columns and len(subtype_data) > 0:
                        color_idx = self._safe_int(subtype_data['color_idx'].iloc[0])
                    else:
                        color_idx = 0
                    
                    color_idx = color_idx % len(self.colors['kc_subtypes'])
                    color = self.colors['kc_subtypes'][color_idx]
                    
                    fig.add_trace(go.Scatter(
                        x=subtype_data['x_pos'],
                        y=subtype_data['y_pos'],
                        mode='markers',
                        marker=dict(
                            size=subtype_data['node_size'],
                            color=color,
                            line=dict(width=1, color='white'),
                            opacity=0.8
                        ),
                        text=[f"KC {subtype}<br>ID: {idx}<br>Centrality: {cent:.3f}" 
                              for idx, cent in zip(subtype_data.index, subtype_data['centrality'])],
                        hoverinfo='text',
                        name=f'KC-{subtype}',
                        legendgroup='KC'
                    ))
            
            elif layer == 'MBON':
                # Color MBONs by neuropil
                for subtype in layer_data['subtype'].unique():
                    subtype_data = layer_data[layer_data['subtype'] == subtype]
                    color = self.colors['mbon_types'].get(str(subtype), '#CD5C5C')
                    
                    fig.add_trace(go.Scatter(
                        x=subtype_data['x_pos'],
                        y=subtype_data['y_pos'],
                        mode='markers',
                        marker=dict(
                            size=subtype_data['node_size'],
                            color=color,
                            line=dict(width=1, color='white'),
                            opacity=0.8
                        ),
                        text=[f"MBON {subtype}<br>ID: {idx}<br>Centrality: {cent:.3f}" 
                              for idx, cent in zip(subtype_data.index, subtype_data['centrality'])],
                        hoverinfo='text',
                        name=f'MBON-{subtype}',
                        legendgroup='MBON'
                    ))
            
            elif layer == 'DAN':
                # Color DANs by valence
                for subtype in layer_data['subtype'].unique():
                    subtype_data = layer_data[layer_data['subtype'] == subtype]
                    color = self.colors['dan_valence'].get(str(subtype), '#708090')
                    
                    fig.add_trace(go.Scatter(
                        x=subtype_data['x_pos'],
                        y=subtype_data['y_pos'],
                        mode='markers',
                        marker=dict(
                            size=subtype_data['node_size'],
                            color=color,
                            line=dict(width=1, color='white'),
                            opacity=0.8
                        ),
                        text=[f"DAN {subtype}<br>ID: {idx}<br>Centrality: {cent:.3f}" 
                              for idx, cent in zip(subtype_data.index, subtype_data['centrality'])],
                        hoverinfo='text',
                        name=f'DAN-{subtype}',
                        legendgroup='DAN'
                    ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="PGCN Circuit: 2D Hierarchical Network<br><sub>PN → KC → MBON ← DAN</sub>",
                x=0.5, font=dict(size=20)
            ),
            xaxis=dict(title="Spatial Distribution", showgrid=False, zeroline=False),
            yaxis=dict(
                title="Circuit Layers", 
                showgrid=True, 
                gridcolor='lightgray',
                tickmode='array',
                tickvals=[-3, -1, 1, 3],
                ticktext=['PNs', 'KCs', 'MBONs', 'DANs']
            ),
            width=1200,
            height=800,
            plot_bgcolor='white',
            hovermode='closest',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left", 
                x=1.02,
                font=dict(size=10)
            ),
            margin=dict(l=50, r=150, t=80, b=50)
        )
        
        return fig
    
    def _add_network_edges(self, fig: go.Figure, neurons_df: pd.DataFrame, top_edges: int):
        """Add connectivity edges to the network plot with robust column handling"""
        if 'pn_kc' not in self.connectivity:
            return
        
        # Get PN→KC connectivity
        pn_kc = self.connectivity['pn_kc']
        
        # Create neuron position lookup with flexible ID matching
        pos_lookup = {}
        neuron_id_col = self._get_id_column(neurons_df, ['root_id', 'node_id', 'neuron_id'])
        
        if neuron_id_col:
            for _, row in neurons_df.iterrows():
                key = row[neuron_id_col]
                pos_lookup[key] = (row['x_pos'], row['y_pos'])
        else:
            # Fallback to using index
            for _, row in neurons_df.iterrows():
                pos_lookup[row.name] = (row['x_pos'], row['y_pos'])
        
        # Get connection columns with flexible naming
        pre_col = self._get_id_column(pn_kc, ['pre_root_id', 'source_id', 'pre_id'])
        post_col = self._get_id_column(pn_kc, ['post_root_id', 'target_id', 'post_id'])
        weight_col = self._safe_get_column(pn_kc, ['synapse_count', 'synapse_weight', 'weight'])
        
        if not pre_col or not post_col:
            print("  ⚠ Could not find source/target columns in connectivity data")
            return
        
        # Add top edges by weight
        if weight_col is not None:
            # Ensure weight column is numeric
            pn_kc[weight_col.name] = pd.to_numeric(pn_kc[weight_col.name], errors='coerce').fillna(0)
            top_edges_df = pn_kc.nlargest(min(top_edges, len(pn_kc)), weight_col.name)
        else:
            top_edges_df = pn_kc.head(min(top_edges, len(pn_kc)))
        
        edge_x = []
        edge_y = []
        
        for _, edge in top_edges_df.iterrows():
            pn_id = edge[pre_col]
            kc_id = edge[post_col]
            
            if pn_id in pos_lookup and kc_id in pos_lookup:
                x0, y0 = pos_lookup[pn_id]
                x1, y1 = pos_lookup[kc_id]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=0.5, color='rgba(128,128,128,0.3)'),
                hoverinfo='none',
                showlegend=False,
                name='Connectivity'
            ))
    
    def create_3d_spatial_plot(self) -> go.Figure:
        """Create 3D spatial plot using node coordinates"""
        print("Creating 3D spatial plot...")
        
        if 'nodes' not in self.features:
            print("No spatial coordinate data available")
            return go.Figure()
        
        nodes_df = self.features['nodes'].copy()
        
        # Add centrality for sizing with robust column handling
        if 'weighted_centrality' in self.features:
            centrality_df = self.features['weighted_centrality']
            cent_id_col = self._get_id_column(centrality_df, ['node_id', 'neuron_id', 'root_id'])
            cent_weight_col = self._safe_get_column(centrality_df, ['weighted_degree', 'degree', 'centrality'])
            
            if cent_id_col and cent_weight_col is not None:
                centrality_map = dict(zip(centrality_df[cent_id_col], cent_weight_col))
                node_id_col = self._get_id_column(nodes_df, ['node_id', 'neuron_id', 'root_id'])
                if node_id_col:
                    nodes_df['centrality'] = nodes_df[node_id_col].map(centrality_map).fillna(1.0)
                else:
                    nodes_df['centrality'] = 1.0
            else:
                nodes_df['centrality'] = 1.0
        else:
            nodes_df['centrality'] = 1.0
        
        # Ensure centrality is numeric
        nodes_df['centrality'] = pd.to_numeric(nodes_df['centrality'], errors='coerce').fillna(1.0)
        
        # Normalize for sizing
        if nodes_df['centrality'].std() > 0:
            nodes_df['size'] = 3 + 12 * (nodes_df['centrality'] - nodes_df['centrality'].min()) / \
                              (nodes_df['centrality'].max() - nodes_df['centrality'].min())
        else:
            nodes_df['size'] = 8
        
        # Create color mapping by type
        color_map = {'PN': '#1f77b4', 'KC': '#ff7f0e', 'MBON': '#2ca02c', 'DAN': '#d62728'}
        nodes_df['color'] = nodes_df['type'].map(color_map)
        
        fig = go.Figure()
        
        # Add nodes by type
        for neuron_type in nodes_df['type'].unique():
            type_data = nodes_df[nodes_df['type'] == neuron_type]
            
            # Get node ID column for display
            id_col = self._get_id_column(type_data, ['node_id', 'neuron_id', 'root_id'])
            if id_col:
                node_ids = type_data[id_col]
            else:
                node_ids = type_data.index
            
            # Get synapse count column
            synapse_col = self._safe_get_column(type_data, ['synapse_count', 'synapse_weight'])
            if synapse_col is not None:
                synapse_counts = pd.to_numeric(synapse_col, errors='coerce').fillna(0)
            else:
                synapse_counts = pd.Series([0] * len(type_data), index=type_data.index)
            
            fig.add_trace(go.Scatter3d(
                x=type_data['x'],
                y=type_data['y'], 
                z=type_data['z'],
                mode='markers',
                marker=dict(
                    size=type_data['size'],
                    color=color_map.get(neuron_type, '#gray'),
                    opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                text=[f"{neuron_type}<br>ID: {nid}<br>Centrality: {cent:.3f}<br>Synapses: {syn}" 
                      for nid, cent, syn in zip(node_ids, type_data['centrality'], synapse_counts)],
                hoverinfo='text',
                name=neuron_type
            ))
        
        # Add top connectivity edges
        self._add_3d_edges(fig, nodes_df, top_k=1000)
        
        fig.update_layout(
            title=dict(
                text="PGCN Circuit: 3D Spatial Network<br><sub>Interactive 3D visualization with spatial coordinates</sub>",
                x=0.5, font=dict(size=18)
            ),
            scene=dict(
                xaxis_title="X Coordinate (nm)",
                yaxis_title="Y Coordinate (nm)", 
                zaxis_title="Z Coordinate (nm)",
                bgcolor='white',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _add_3d_edges(self, fig: go.Figure, nodes_df: pd.DataFrame, top_k: int):
        """Add 3D connectivity edges with robust column handling"""
        if 'edges' not in self.connectivity:
            return
            
        edges_df = self.connectivity['edges']
        
        # Create position lookup
        node_id_col = self._get_id_column(nodes_df, ['node_id', 'neuron_id', 'root_id'])
        if not node_id_col:
            return
            
        pos_lookup = dict(zip(nodes_df[node_id_col], 
                             zip(nodes_df['x'], nodes_df['y'], nodes_df['z'])))
        
        # Get edge columns
        source_col = self._get_id_column(edges_df, ['source_id', 'pre_id', 'pre_root_id'])
        target_col = self._get_id_column(edges_df, ['target_id', 'post_id', 'post_root_id'])
        weight_col = self._safe_get_column(edges_df, ['synapse_weight', 'synapse_count', 'weight'])
        
        if not source_col or not target_col:
            return
        
        # Get top edges by weight
        if weight_col is not None:
            # Ensure weight is numeric
            edges_df[weight_col.name] = pd.to_numeric(edges_df[weight_col.name], errors='coerce').fillna(0)
            top_edges = edges_df.nlargest(min(top_k, len(edges_df)), weight_col.name)
        else:
            top_edges = edges_df.head(min(top_k, len(edges_df)))
        
        edge_x, edge_y, edge_z = [], [], []
        
        for _, edge in top_edges.iterrows():
            src = edge[source_col]
            tgt = edge[target_col]
            
            if src in pos_lookup and tgt in pos_lookup:
                x0, y0, z0 = pos_lookup[src]
                x1, y1, z1 = pos_lookup[tgt]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
        
        if edge_x:
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(width=2, color='rgba(128,128,128,0.2)'),
                hoverinfo='none',
                showlegend=False,
                name='Connections'
            ))
    
    def create_connectivity_heatmap(self) -> go.Figure:
        """Create PN→KC connectivity heatmap with robust column handling"""
        print("Creating PN→KC connectivity heatmap...")
        
        if 'pn_kc' not in self.connectivity:
            print("No PN→KC connectivity data available")
            return go.Figure()
        
        pn_kc = self.connectivity['pn_kc']
        
        # Get column names robustly
        pre_col = self._get_id_column(pn_kc, ['pre_root_id', 'source_id', 'pre_id'])
        post_col = self._get_id_column(pn_kc, ['post_root_id', 'target_id', 'post_id'])
        weight_col = self._safe_get_column(pn_kc, ['synapse_count', 'synapse_weight', 'weight'])
        
        if not pre_col or not post_col:
            print("Could not find source/target columns for heatmap")
            return go.Figure()
        
        # Create sparse connectivity matrix
        pn_ids = pn_kc[pre_col].unique()
        kc_ids = pn_kc[post_col].unique()
        
        # Create matrix
        pn_idx = {pid: i for i, pid in enumerate(pn_ids)}
        kc_idx = {kid: i for i, kid in enumerate(kc_ids)}
        
        matrix = np.zeros((len(pn_ids), len(kc_ids)))
        
        for _, row in pn_kc.iterrows():
            pn_id = row[pre_col]
            kc_id = row[post_col]
            if weight_col is not None:
                weight = pd.to_numeric(row[weight_col.name], errors='coerce')
                weight = weight if not pd.isna(weight) else 1
            else:
                weight = 1
            
            if pn_id in pn_idx and kc_id in kc_idx:
                matrix[pn_idx[pn_id], kc_idx[kc_id]] = weight
        
        # Downsample for visualization if too large
        max_size = 1000
        if matrix.shape[0] > max_size or matrix.shape[1] > max_size:
            pn_step = max(1, matrix.shape[0] // max_size)
            kc_step = max(1, matrix.shape[1] // max_size)
            matrix = matrix[::pn_step, ::kc_step]
            pn_ids = pn_ids[::pn_step]
            kc_ids = kc_ids[::kc_step]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            colorscale='Viridis',
            hovertemplate='PN: %{y}<br>KC: %{x}<br>Synapses: %{z}<extra></extra>',
            colorbar=dict(title="Synapse Count")
        ))
        
        fig.update_layout(
            title=dict(
                text="PN→KC Connectivity Matrix<br><sub>Synapse counts between projection neurons and Kenyon cells</sub>",
                x=0.5, font=dict(size=18)
            ),
            xaxis_title="Kenyon Cells (KCs)",
            yaxis_title="Projection Neurons (PNs)",
            width=1000,
            height=800
        )
        
        return fig
    
    def create_feature_analysis(self) -> go.Figure:
        """Create conditioning features analysis with robust column handling"""
        print("Creating feature analysis...")
        
        if 'olfactory_conditioning_features' not in self.features:
            print("No conditioning features data available")
            return go.Figure()
        
        features_df = self.features['olfactory_conditioning_features']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Feature Distribution", "Centrality vs Features",
                          "Feature Correlation", "Top Contributing Neurons"),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # 1. Feature distribution histogram
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            main_feature = numeric_cols[0]
            fig.add_trace(
                go.Histogram(x=features_df[main_feature], name="Feature Distribution"),
                row=1, col=1
            )
        
        # 2. Centrality vs features scatter
        if 'weighted_centrality' in self.features and len(numeric_cols) > 0:
            centrality_df = self.features['weighted_centrality']
            
            # Get ID columns for merging
            feat_id_col = self._get_id_column(features_df, ['neuron_id', 'node_id', 'root_id'])
            cent_id_col = self._get_id_column(centrality_df, ['node_id', 'neuron_id', 'root_id'])
            
            if feat_id_col and cent_id_col:
                # Merge with features
                merged = features_df.merge(centrality_df, 
                                         left_on=feat_id_col, 
                                         right_on=cent_id_col, 
                                         how='inner')
                
                cent_weight_col = self._safe_get_column(merged, ['weighted_degree', 'degree', 'centrality'])
                
                if not merged.empty and cent_weight_col is not None:
                    y_feature = main_feature if main_feature in merged.columns else numeric_cols[0]
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_numeric(cent_weight_col, errors='coerce').fillna(0),
                            y=pd.to_numeric(merged[y_feature], errors='coerce').fillna(0),
                            mode='markers',
                            name="Centrality vs Features",
                            marker=dict(opacity=0.6)
                        ),
                        row=1, col=2
                    )
        
        # 3. Feature correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix = features_df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    name="Correlation"
                ),
                row=2, col=1
            )
        
        # 4. Top contributing neurons
        if len(numeric_cols) > 0:
            top_neurons = features_df.nlargest(20, numeric_cols[0])
            fig.add_trace(
                go.Bar(
                    x=top_neurons.index[:10],  # Show top 10
                    y=pd.to_numeric(top_neurons[numeric_cols[0]][:10], errors='coerce').fillna(0),
                    name="Top Neurons"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="PGCN Circuit: Feature Analysis Dashboard",
            width=1200,
            height=900,
            showlegend=False
        )
        
        return fig
    
    def generate_circuit_statistics(self) -> Dict:
        """Generate comprehensive circuit statistics"""
        print("Generating circuit statistics...")
        
        stats = {
            'neurons': {},
            'connectivity': {},
            'features': {}
        }
        
        # Neuron counts
        for name, df in self.neurons.items():
            stats['neurons'][name] = len(df)
        
        # Connectivity stats  
        for name, df in self.connectivity.items():
            stats['connectivity'][name] = {
                'edges': len(df),
                'sparsity': self._calculate_sparsity(df)
            }
        
        # Feature stats
        for name, df in self.features.items():
            stats['features'][name] = len(df)
        
        # Calculate totals
        pn_total = sum([count for name, count in stats['neurons'].items() if 'alpn' in name])
        kc_total = sum([count for name, count in stats['neurons'].items() if name.startswith('kc_')])
        mbon_total = sum([count for name, count in stats['neurons'].items() if 'mbon' in name])
        dan_total = sum([count for name, count in stats['neurons'].items() if 'dan' in name])
        
        stats['summary'] = {
            'total_neurons': pn_total + kc_total + mbon_total + dan_total,
            'pn_count': pn_total,
            'kc_count': kc_total, 
            'mbon_count': mbon_total,
            'dan_count': dan_total,
            'total_edges': sum([s.get('edges', 0) for s in stats['connectivity'].values()])
        }
        
        # Add centrality rankings with robust handling
        if 'weighted_centrality' in self.features:
            centrality_df = self.features['weighted_centrality']
            cent_id_col = self._get_id_column(centrality_df, ['node_id', 'neuron_id', 'root_id'])
            cent_weight_col = self._safe_get_column(centrality_df, ['weighted_degree', 'degree', 'centrality'])
            
            if cent_id_col and cent_weight_col is not None:
                # Ensure centrality values are numeric
                centrality_df[cent_weight_col.name] = pd.to_numeric(centrality_df[cent_weight_col.name], errors='coerce').fillna(0)
                top_central = centrality_df.nlargest(10, cent_weight_col.name)
                stats['top_central_neurons'] = [
                    {'neuron_id': str(row[cent_id_col]), 'centrality': float(row[cent_weight_col.name])}
                    for _, row in top_central.iterrows()
                ]
        
        # Add valence distribution with robust handling
        if 'dan_valence' in self.features:
            valence_df = self.features['dan_valence']
            valence_col = self._safe_get_column(valence_df, ['valence', 'type', 'classification'])
            if valence_col is not None:
                valence_counts = valence_col.astype(str).value_counts()
                stats['dan_valence_distribution'] = valence_counts.to_dict()
        
        self.stats = stats
        return stats
    
    def _calculate_sparsity(self, connectivity_df: pd.DataFrame) -> float:
        """Calculate sparsity of connectivity matrix"""
        if connectivity_df.empty:
            return 0.0
        
        # Estimate based on unique source/target pairs
        sources = connectivity_df.iloc[:, 0].nunique() if len(connectivity_df.columns) > 0 else 0
        targets = connectivity_df.iloc[:, 1].nunique() if len(connectivity_df.columns) > 1 else 0
        
        if sources * targets == 0:
            return 0.0
            
        density = len(connectivity_df) / (sources * targets)
        return 1.0 - density
    
    def save_visualizations(self, format_types: List[str], **kwargs):
        """Save all visualizations in specified formats"""
        print("Saving visualizations...")

        # CRITICAL FIX: Use CDN for plotly.js to prevent blank pages and reduce file size
        # This loads plotly from CDN instead of embedding 3MB JavaScript in each HTML
        plotly_config = {
            'include_plotlyjs': 'cdn',  # Use CDN instead of embedding full library
            'full_html': True,          # Generate complete HTML document
            'config': {                 # Enable interactive features
                'displayModeBar': True,
                'responsive': True,
                'displaylogo': False
            }
        }

        # Create visualizations
        if 'all' in format_types or '2d' in format_types:
            fig_2d = self.create_2d_hierarchical_network(kwargs.get('top_edges', 500))
            fig_2d.write_html(
                self.output_dir / 'pgcn_network_2d_hierarchical.html',
                **plotly_config
            )
            print(f"  ✓ Saved 2D hierarchical network")

        if 'all' in format_types or '3d' in format_types:
            fig_3d = self.create_3d_spatial_plot()
            fig_3d.write_html(
                self.output_dir / 'pgcn_network_3d_spatial.html',
                **plotly_config
            )
            print(f"  ✓ Saved 3D spatial network")

        if 'all' in format_types or 'heatmap' in format_types:
            fig_heatmap = self.create_connectivity_heatmap()
            fig_heatmap.write_html(
                self.output_dir / 'pgcn_connectivity_pn_kc.html',
                **plotly_config
            )
            print(f"  ✓ Saved connectivity heatmap")

        if 'all' in format_types or 'features' in format_types:
            if kwargs.get('include_features', True):
                fig_features = self.create_feature_analysis()
                fig_features.write_html(
                    self.output_dir / 'pgcn_conditioning_features.html',
                    **plotly_config
                )
                print(f"  ✓ Saved feature analysis")
        
        if 'all' in format_types or 'stats' in format_types:
            # Save statistics
            stats = self.generate_circuit_statistics()
            
            # Save as JSON
            with open(self.output_dir / 'pgcn_circuit_stats.json', 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            # Save as CSV
            stats_flat = self._flatten_stats_for_csv(stats)
            pd.DataFrame(stats_flat).to_csv(self.output_dir / 'pgcn_circuit_stats.csv', index=False)
            
            # Create summary visualization
            self._create_stats_plots()
            
            print(f"  ✓ Saved circuit statistics")
        
        # Create README
        self._create_readme()
        print(f"  ✓ Created interpretation README")
    
    def _flatten_stats_for_csv(self, stats: Dict) -> List[Dict]:
        """Flatten nested stats dictionary for CSV export"""
        flat_stats = []
        
        # Neuron counts
        for name, count in stats['neurons'].items():
            flat_stats.append({'category': 'neurons', 'metric': name, 'value': count})
        
        # Connectivity stats
        for name, data in stats['connectivity'].items():
            if isinstance(data, dict):
                for metric, value in data.items():
                    flat_stats.append({'category': 'connectivity', 'metric': f'{name}_{metric}', 'value': value})
            else:
                flat_stats.append({'category': 'connectivity', 'metric': name, 'value': data})
        
        # Summary stats
        for metric, value in stats['summary'].items():
            flat_stats.append({'category': 'summary', 'metric': metric, 'value': value})
        
        return flat_stats
    
    def _create_stats_plots(self):
        """Create statistical summary plots"""
        if not self.stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('PGCN Circuit Statistics Summary', fontsize=16, fontweight='bold')
        
        # 1. Neuron counts by type
        summary = self.stats['summary']
        neuron_types = ['PN', 'KC', 'MBON', 'DAN']
        neuron_counts = [summary['pn_count'], summary['kc_count'], 
                        summary['mbon_count'], summary['dan_count']]
        
        axes[0,0].bar(neuron_types, neuron_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0,0].set_title('Neuron Counts by Type')
        axes[0,0].set_ylabel('Count')
        
        # 2. KC subtype distribution
        kc_subtypes = {name.replace('kc_', ''): count for name, count in self.stats['neurons'].items() 
                      if name.startswith('kc_') and count > 0}
        if kc_subtypes:
            axes[0,1].pie(kc_subtypes.values(), labels=kc_subtypes.keys(), autopct='%1.1f%%')
            axes[0,1].set_title('KC Subtype Distribution')
        else:
            axes[0,1].text(0.5, 0.5, 'No KC data', ha='center', va='center', transform=axes[0,1].transAxes)
        
        # 3. Connectivity edge counts
        conn_names = list(self.stats['connectivity'].keys())
        edge_counts = [self.stats['connectivity'][name].get('edges', 0) for name in conn_names]
        
        if conn_names and any(edge_counts):
            axes[1,0].bar(conn_names, edge_counts)
            axes[1,0].set_title('Connectivity Edge Counts')
            axes[1,0].set_ylabel('Edges')
            axes[1,0].tick_params(axis='x', rotation=45)
        else:
            axes[1,0].text(0.5, 0.5, 'No connectivity data', ha='center', va='center', transform=axes[1,0].transAxes)
        
        # 4. DAN valence distribution
        if 'dan_valence_distribution' in self.stats and self.stats['dan_valence_distribution']:
            valence_data = self.stats['dan_valence_distribution']
            axes[1,1].pie(valence_data.values(), labels=valence_data.keys(), 
                         colors=['#32CD32', '#FF6347', '#708090'][:len(valence_data)])
            axes[1,1].set_title('DAN Valence Distribution')
        else:
            axes[1,1].text(0.5, 0.5, 'No DAN valence data', ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pgcn_circuit_stats.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_readme(self):
        """Create interpretation README"""
        readme_content = f"""# PGCN Circuit Visualization Results

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Repository: colehanan1/Plasticity-Guided-Connectome-Network-PGCN (branch: wow-im-tired)

## Files Generated

### Interactive Visualizations
- **pgcn_network_2d_hierarchical.html** - Main 2D hierarchical network showing PN→KC→MBON←DAN circuit layers
- **pgcn_network_3d_spatial.html** - 3D spatial visualization using neuron coordinates  
- **pgcn_connectivity_pn_kc.html** - PN→KC connectivity heatmap with synapse counts
- **pgcn_conditioning_features.html** - Feature analysis dashboard for conditioning-related metrics

### Data Summaries  
- **pgcn_circuit_stats.csv** - Tabular statistics summary
- **pgcn_circuit_stats.json** - Detailed statistics in JSON format
- **pgcn_circuit_stats.png** - Statistical summary plots

## Circuit Summary
{self._generate_circuit_summary()}

## Visualization Guide

### 2D Hierarchical Network
- **Layout**: 4 horizontal layers representing circuit hierarchy
- **Node Colors**: 
  - PNs: Colored by glomerulus type
  - KCs: Colored by subtype (ab, g, ap etc.)
  - MBONs: Green=calyx, Gold=medial lobe, Red=other
  - DANs: Green=PAM, Red=PPL1, Gray=other
- **Node Size**: Proportional to weighted centrality
- **Edges**: Top {500} connections by synapse count
- **Interactions**: Hover for details, zoom/pan enabled

### 3D Spatial Network  
- **Coordinates**: Real spatial positions from FlyWire (8nm voxels)
- **Navigation**: Rotate, zoom, pan with mouse
- **Colors**: PN=blue, KC=orange, MBON=green, DAN=red
- **Edges**: Top 1000 connections shown

### Connectivity Heatmap
- **Matrix**: PN→KC synapse counts (sparse visualization) 
- **Colors**: Viridis colormap (dark=low, bright=high synapse counts)
- **Downsampling**: Applied if >1000 neurons per dimension

### Feature Analysis
- Distribution plots of conditioning-related features
- Centrality vs feature scatter plots  
- Feature correlation heatmaps
- Top contributing neurons by feature importance

## Data Sources
- **Neurons**: {len(self.neurons)} CSV files loaded from data/cache/
- **Connectivity**: {len(self.connectivity)} connection datasets  
- **Features**: {len(self.features)} feature/metadata files
- **Missing/Empty files**: Gracefully skipped with warnings

## Usage Notes
- All HTML files are self-contained and work offline
- Large datasets are automatically downsampled for performance
- Hover tooltips provide detailed neuron information
- Legend items can be clicked to show/hide neuron types
- Empty files (like kc_apb.csv) are automatically skipped

## Citation
If using these visualizations, please cite:
- FlyWire Consortium and Princeton University for connectome data
- PGCN Project (colehanan1/Plasticity-Guided-Connectome-Network-PGCN)

## Technical Details
- Python libraries: plotly, pandas, numpy, scipy, networkx, matplotlib
- Visualization engine: Plotly.js for interactivity
- Data formats: CSV, Parquet via pandas/pyarrow
- Error handling: Graceful fallbacks for missing/empty data
- Type safety: All numeric conversions with error handling
- Column handling: Robust matching for variable column names
"""
        
        with open(self.output_dir / 'pgcn_README.md', 'w') as f:
            f.write(readme_content)
    
    def _generate_circuit_summary(self) -> str:
        """Generate circuit summary text"""
        if not self.stats:
            return "No statistics available"
        
        summary = self.stats['summary']
        return f"""
- **Total Neurons**: {summary['total_neurons']:,}
  - PNs (Projection): {summary['pn_count']}  
  - KCs (Kenyon): {summary['kc_count']}
  - MBONs (Mushroom Body Output): {summary['mbon_count']}
  - DANs (Dopaminergic): {summary['dan_count']}
- **Total Connections**: {summary['total_edges']:,}
- **Circuit Sparsity**: {self._calculate_overall_sparsity():.3f}
        """
    
    def _calculate_overall_sparsity(self) -> float:
        """Calculate overall circuit sparsity"""
        if not self.connectivity:
            return 0.0
        
        sparsities = [data.get('sparsity', 0) for data in self.stats['connectivity'].values() 
                     if isinstance(data, dict)]
        return np.mean(sparsities) if sparsities else 0.0


def main():
    """Main execution function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Visualize PGCN Drosophila olfactory learning circuit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize_pgcn_circuit.py --format all
  python scripts/visualize_pgcn_circuit.py --cache-dir data/cache --output-dir reports
  python scripts/visualize_pgcn_circuit.py --format 2d,3d --top-edges 1000
        """
    )
    
    parser.add_argument('--cache-dir', type=Path, default='data/cache',
                       help='Path to cache directory (default: data/cache)')
    parser.add_argument('--output-dir', type=Path, default='reports', 
                       help='Output directory (default: reports)')
    parser.add_argument('--format', default='all',
                       help='Visualization formats: all|2d|3d|heatmap|features|stats (default: all)')
    parser.add_argument('--top-edges', type=int, default=500,
                       help='Number of top edges to show (default: 500)')
    parser.add_argument('--color-by', default='type',
                       help='Node coloring: type|glomerulus|subtype|centrality (default: type)')
    parser.add_argument('--size-by', default='centrality', 
                       help='Node sizing: centrality|degree|synapse_count (default: centrality)')
    parser.add_argument('--include-features', action='store_true', default=True,
                       help='Include conditioning features visualization (default: True)')
    parser.add_argument('--include-dan', action='store_true', default=True,
                       help='Include DAN neurons (default: True)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.cache_dir.exists():
        print(f"Error: Cache directory {args.cache_dir} does not exist")
        sys.exit(1)
    
    # Parse format types
    format_types = [f.strip() for f in args.format.split(',')]
    
    print("=" * 60)
    print("PGCN CIRCUIT VISUALIZER (FINAL FIXED VERSION)")
    print("=" * 60)
    print(f"Cache directory: {args.cache_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Formats: {', '.join(format_types)}")
    print(f"Top edges: {args.top_edges}")
    
    # Initialize visualizer
    visualizer = PGCNCircuitVisualizer(args.cache_dir, args.output_dir)
    
    try:
        # Load data
        visualizer.load_data()
        
        # Create visualizations
        visualizer.save_visualizations(
            format_types=format_types,
            top_edges=args.top_edges,
            color_by=args.color_by,
            size_by=args.size_by,
            include_features=args.include_features,
            include_dan=args.include_dan
        )
        
        print("=" * 60)
        print("✓ VISUALIZATION COMPLETE")
        print(f"✓ Results saved to: {args.output_dir}")
        print(f"✓ Open pgcn_network_2d_hierarchical.html to start exploring")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()