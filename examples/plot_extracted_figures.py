#!/usr/bin/env python3
"""
Example Plotting Script for Extracted Figure Data
==================================================

This script demonstrates how to create publication-ready figures using
the data extracted by extract_figure_data.py.

Usage:
    # First, extract the data
    python extract_figure_data.py --task all

    # Then create the figures
    python examples/plot_extracted_figures.py --figure all
    python examples/plot_extracted_figures.py --figure behavioral
    python examples/plot_extracted_figures.py --figure schematic
    python examples/plot_extracted_figures.py --figure synapse_map
    python examples/plot_extracted_figures.py --figure ml_comparison

Output:
    Figures saved to figures/publication/
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import seaborn as sns
import yaml
import pickle
from typing import Dict, List, Any

# Set publication-quality defaults
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['figure.dpi'] = 300
sns.set_style("ticks")

# Color schemes (colorblind-safe)
COLORS = {
    'wildtype': '#0173B2',      # Blue
    'or7a_mutant': '#DE8F05',   # Orange
    'control': '#029E73',       # Green
    'protected': '#CC78BC',     # Purple
    'unprotected': '#ECE133'    # Yellow
}

MODEL_COLORS = {
    'MBON_veto': '#0173B2',     # Blue (best)
    'GEM': '#029E73',           # Green
    'EWC': '#56B4E9',           # Light blue
    'SI': '#CC79A7',            # Pink
    'LwF': '#E69F00',           # Orange
    'Dense_ANN': '#D55E00'      # Red-orange (worst)
}


# ==============================================================================
# Figure 1: Behavioral Prediction (Memory Scores Across Phases)
# ==============================================================================

def plot_behavioral_prediction(
    data_file: str = "data/extracted_figures/behavioral_data.csv",
    output_dir: str = "figures/publication"
):
    """
    Create Figure 1: Behavioral prediction showing memory retention across
    training phases for different genotypes.

    Creates a line plot with error bars (if multiple trials available).
    """
    print("\n" + "="*70)
    print("Creating Figure 1: Behavioral Prediction")
    print("="*70)

    # Load data
    df = pd.read_csv(data_file)
    print(f"✓ Loaded data from: {data_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Phase labels (x-axis)
    phases = df['phase'].values
    x_positions = np.arange(len(phases))

    # Plot each group
    for group in ['wildtype', 'or7a_mutant', 'control']:
        if group in df.columns:
            scores = df[group].values
            ax.plot(x_positions, scores,
                   marker='o', markersize=10, linewidth=2.5,
                   label=group.replace('_', ' ').title(),
                   color=COLORS[group])

            # Add value labels
            for x, y in zip(x_positions, scores):
                ax.text(x, y + 0.03, f'{y:.2f}',
                       ha='center', va='bottom', fontsize=9,
                       color=COLORS[group])

    # Formatting
    ax.set_xlabel('Training Phase', fontsize=13, fontweight='bold')
    ax.set_ylabel('Memory Score', fontsize=13, fontweight='bold')
    ax.set_title('Behavioral Prediction: Memory Retention Across Training',
                fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([p.replace('_', '\n') for p in phases], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add horizontal line at 0.5 (chance level)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
              label='Chance level')

    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True,
             shadow=True, fontsize=11)

    # Add annotation for forgetting
    ax.annotate('Catastrophic\nForgetting',
               xy=(2, df['or7a_mutant'].values[2]),
               xytext=(2.3, 0.35),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='red', alpha=0.8))

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "figure1_behavioral_prediction.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to: {output_file}")

    # Also save as PDF for publication
    output_pdf = output_path / "figure1_behavioral_prediction.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_pdf}")

    plt.show()


# ==============================================================================
# Figure 2: Model Schematic (Architecture Diagram)
# ==============================================================================

def plot_model_schematic(
    data_file: str = "data/extracted_figures/model_schematic_info.yaml",
    output_dir: str = "figures/publication"
):
    """
    Create Figure 2: Model schematic showing neural network architecture
    with neuron counts and synapse statistics.
    """
    print("\n" + "="*70)
    print("Creating Figure 2: Model Schematic")
    print("="*70)

    # Load data
    with open(data_file, 'r') as f:
        schematic = yaml.safe_load(f)
    print(f"✓ Loaded schematic info from: {data_file}")

    # Extract counts
    n_pn = schematic.get('n_pn', 50)
    n_kc = schematic.get('n_kc', 2000)
    n_mbon = schematic.get('n_mbon', 44)
    n_synapses = schematic.get('n_synapses', n_kc * n_mbon)
    n_protected = schematic.get('n_protected', 0)
    prot_pct = schematic.get('protection_percentage', 0.0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'PGCN Model Architecture', fontsize=18,
           fontweight='bold', ha='center')

    # Layer positions
    pn_x, kc_x, mbon_x = 1.5, 5, 8.5
    layer_y = 5

    # Draw layers as rectangles
    layers = [
        {'name': 'PN\nProjection\nNeurons', 'x': pn_x, 'count': n_pn, 'color': '#E1F5FE'},
        {'name': 'KC\nKenyon\nCells', 'x': kc_x, 'count': n_kc, 'color': '#FFF9C4'},
        {'name': 'MBON\nMushroom\nBody Output', 'x': mbon_x, 'count': n_mbon, 'color': '#F8BBD0'}
    ]

    box_width, box_height = 1.5, 3

    for layer in layers:
        # Draw box
        box = FancyBboxPatch(
            (layer['x'] - box_width/2, layer_y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            edgecolor='black', facecolor=layer['color'],
            linewidth=2.5, alpha=0.8
        )
        ax.add_patch(box)

        # Layer name
        ax.text(layer['x'], layer_y + 0.8, layer['name'],
               ha='center', va='center', fontsize=11,
               fontweight='bold', multialignment='center')

        # Neuron count
        ax.text(layer['x'], layer_y - 0.5, f"n = {layer['count']:,}",
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='black', linewidth=1))

    # Draw connections
    # PN → KC
    arrow1 = FancyArrowPatch(
        (pn_x + box_width/2 + 0.1, layer_y),
        (kc_x - box_width/2 - 0.1, layer_y),
        arrowstyle='->', mutation_scale=30, linewidth=3,
        color='#666666', alpha=0.7
    )
    ax.add_patch(arrow1)
    ax.text((pn_x + kc_x)/2, layer_y + 0.5, 'Sparse\nConnectivity',
           ha='center', va='bottom', fontsize=9, style='italic')

    # KC → MBON (with veto gate)
    arrow2 = FancyArrowPatch(
        (kc_x + box_width/2 + 0.1, layer_y),
        (mbon_x - box_width/2 - 0.1, layer_y),
        arrowstyle='->', mutation_scale=30, linewidth=3,
        color='#CC78BC', alpha=0.8
    )
    ax.add_patch(arrow2)

    # Synapse info
    synapse_text = (f"KC→MBON Synapses\n"
                   f"Total: {n_synapses:,}\n"
                   f"Protected: {n_protected:,} ({prot_pct:.1f}%)")
    ax.text((kc_x + mbon_x)/2, layer_y - 1.5, synapse_text,
           ha='center', va='top', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0',
                    edgecolor='#CC78BC', linewidth=2))

    # Veto gate annotation
    veto_box = FancyBboxPatch(
        ((kc_x + mbon_x)/2 - 0.8, layer_y + 1.2),
        1.6, 0.6,
        boxstyle="round,pad=0.05",
        edgecolor='#CC78BC', facecolor='#F3E5F5',
        linewidth=2, linestyle='--'
    )
    ax.add_patch(veto_box)
    ax.text((kc_x + mbon_x)/2, layer_y + 1.5, 'Veto Gate',
           ha='center', va='center', fontsize=10,
           fontweight='bold', color='#7B1FA2')

    # Add plasticity rule
    plasticity_text = ("Three-Factor Hebbian Learning:\n"
                      "ΔW = α × KC × MBON × Dopamine")
    ax.text(5, 1.5, plasticity_text,
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFDE7',
                    edgecolor='black', linewidth=1.5),
           family='monospace')

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "figure2_model_schematic.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to: {output_file}")

    output_pdf = output_path / "figure2_model_schematic.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_pdf}")

    plt.show()


# ==============================================================================
# Figure 3: Critical Synapse Map (Veto Gate Protection Heatmap)
# ==============================================================================

def plot_synapse_map(
    data_file: str = "data/extracted_figures/veto_mask.npy",
    summary_file: str = "data/extracted_figures/synapse_map_summary.csv",
    output_dir: str = "figures/publication"
):
    """
    Create Figure 3: Heatmap showing which KC→MBON synapses are protected
    by the veto gate mechanism.
    """
    print("\n" + "="*70)
    print("Creating Figure 3: Critical Synapse Map")
    print("="*70)

    # Load veto mask
    veto_mask = np.load(data_file)
    print(f"✓ Loaded veto mask from: {data_file}")
    print(f"  Shape: {veto_mask.shape}")

    # Load summary
    df_summary = pd.read_csv(summary_file)
    n_protected = int(df_summary.iloc[0]['n_protected'])
    prot_pct = float(df_summary.iloc[0]['protection_pct'])

    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

    # Left panel: Full heatmap
    ax1 = fig.add_subplot(gs[0])

    # For large matrices, downsample for visualization
    if veto_mask.shape[0] > 500:
        # Bin the matrix for display (e.g., 100×100)
        from scipy.ndimage import zoom
        scale = min(100 / veto_mask.shape[0], 100 / veto_mask.shape[1])
        veto_display = zoom(veto_mask.astype(float), scale, order=0)
        print(f"  Downsampled to {veto_display.shape} for visualization")
    else:
        veto_display = veto_mask

    im = ax1.imshow(veto_display, cmap='RdPu', aspect='auto',
                   interpolation='nearest', vmin=0, vmax=1)

    ax1.set_xlabel('MBON Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Kenyon Cell Index', fontsize=12, fontweight='bold')
    ax1.set_title(f'Veto Gate Protection Mask\n'
                 f'{n_protected:,} protected synapses ({prot_pct:.1f}%)',
                 fontsize=13, fontweight='bold', pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Protected', fontsize=11, fontweight='bold')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['No', 'Yes'])

    # Right panel: Statistics
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    # Calculate statistics
    n_kc, n_mbon = veto_mask.shape
    total_synapses = veto_mask.size
    n_unprotected = total_synapses - n_protected

    # Bar chart of protected vs unprotected
    ax2_inner = fig.add_axes([0.72, 0.3, 0.2, 0.4])

    categories = ['Protected', 'Unprotected']
    counts = [n_protected, n_unprotected]
    colors_bar = ['#CC78BC', '#E0E0E0']

    bars = ax2_inner.bar(categories, counts, color=colors_bar,
                        edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2_inner.text(bar.get_x() + bar.get_width()/2, height,
                     f'{count:,}\n({count/total_synapses*100:.1f}%)',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2_inner.set_ylabel('Number of Synapses', fontsize=10, fontweight='bold')
    ax2_inner.set_title('Protection Statistics', fontsize=11, fontweight='bold')
    ax2_inner.spines['top'].set_visible(False)
    ax2_inner.spines['right'].set_visible(False)
    ax2_inner.tick_params(axis='x', labelsize=9)
    ax2_inner.grid(axis='y', alpha=0.3, linestyle='--')

    # Add text summary
    summary_text = (
        f"Architecture:\n"
        f"  KCs: {n_kc:,}\n"
        f"  MBONs: {n_mbon:,}\n"
        f"  Total synapses: {total_synapses:,}\n\n"
        f"Protection:\n"
        f"  Protected: {n_protected:,}\n"
        f"  Coverage: {prot_pct:.1f}%\n\n"
        f"Mechanism:\n"
        f"  Or7a-inspired\n"
        f"  veto gate"
    )
    ax2.text(0.5, 0.85, summary_text, fontsize=9,
            ha='center', va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5',
                     edgecolor='black', linewidth=1.5))

    plt.suptitle('Figure 3: Critical Synapse Protection Map',
                fontsize=15, fontweight='bold', y=0.98)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "figure3_synapse_map.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to: {output_file}")

    output_pdf = output_path / "figure3_synapse_map.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_pdf}")

    plt.show()


# ==============================================================================
# Figure 4: ML Model Comparison (Forgetting Scores)
# ==============================================================================

def plot_ml_comparison(
    data_file: str = "data/extracted_figures/ml_comparison_data.csv",
    output_dir: str = "figures/publication"
):
    """
    Create Figure 4: Bar chart comparing forgetting scores across different
    continual learning methods.
    """
    print("\n" + "="*70)
    print("Creating Figure 4: ML Model Comparison")
    print("="*70)

    # Load data
    df = pd.read_csv(data_file)
    print(f"✓ Loaded ML comparison from: {data_file}")
    print(f"  Models: {df['model_type'].tolist()}")

    # Sort by forgetting score (best = lowest)
    df = df.sort_values('forgetting_score')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get colors for each model
    colors = [MODEL_COLORS.get(model, '#888888') for model in df['model_type']]

    # Create bar chart
    bars = ax.barh(df['model_type'], df['forgetting_score'],
                  color=colors, edgecolor='black', linewidth=1.5,
                  alpha=0.85)

    # Add value labels
    for i, (model, score) in enumerate(zip(df['model_type'], df['forgetting_score'])):
        ax.text(score + 0.02, i, f'{score:.3f}',
               va='center', ha='left', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_xlabel('Forgetting Score (lower is better)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model Type', fontsize=13, fontweight='bold')
    ax.set_title('Continual Learning Performance Comparison',
                fontsize=14, fontweight='bold', pad=20)

    ax.set_xlim(0, max(df['forgetting_score']) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation for best model
    best_model = df.iloc[0]['model_type']
    best_score = df.iloc[0]['forgetting_score']
    ax.annotate('Best\nPerformance',
               xy=(best_score, 0),
               xytext=(best_score + 0.15, 0.8),
               arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
               fontsize=11, color='green', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='green', linewidth=2))

    # Add legend explaining methods
    legend_text = (
        "Methods:\n"
        "• MBON_veto: Or7a-inspired veto gate (this work)\n"
        "• GEM: Gradient Episodic Memory\n"
        "• EWC: Elastic Weight Consolidation\n"
        "• SI: Synaptic Intelligence\n"
        "• LwF: Learning without Forgetting\n"
        "• Dense_ANN: Standard neural network (baseline)"
    )
    ax.text(0.98, 0.97, legend_text,
           transform=ax.transAxes,
           fontsize=8, ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9F9F9',
                    edgecolor='gray', linewidth=1, alpha=0.9),
           family='monospace')

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "figure4_ml_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to: {output_file}")

    output_pdf = output_path / "figure4_ml_comparison.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_pdf}")

    plt.show()


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main execution: create selected figures."""
    parser = argparse.ArgumentParser(
        description="Create publication figures from extracted data"
    )
    parser.add_argument(
        '--figure',
        type=str,
        default='all',
        choices=['all', 'behavioral', 'schematic', 'synapse_map', 'ml_comparison'],
        help='Which figure to generate (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/extracted_figures',
        help='Directory with extracted data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures/publication',
        help='Output directory for figures'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("PGCN Publication Figure Generator")
    print("="*70)
    print(f"Figure: {args.figure}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Generate selected figures
    if args.figure in ['all', 'behavioral']:
        plot_behavioral_prediction(
            data_file=f"{args.data_dir}/behavioral_data.csv",
            output_dir=args.output_dir
        )

    if args.figure in ['all', 'schematic']:
        plot_model_schematic(
            data_file=f"{args.data_dir}/model_schematic_info.yaml",
            output_dir=args.output_dir
        )

    if args.figure in ['all', 'synapse_map']:
        plot_synapse_map(
            data_file=f"{args.data_dir}/veto_mask.npy",
            summary_file=f"{args.data_dir}/synapse_map_summary.csv",
            output_dir=args.output_dir
        )

    if args.figure in ['all', 'ml_comparison']:
        plot_ml_comparison(
            data_file=f"{args.data_dir}/ml_comparison_data.csv",
            output_dir=args.output_dir
        )

    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)
    print(f"✓ All figures saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  • figure1_behavioral_prediction.png/pdf")
    print("  • figure2_model_schematic.png/pdf")
    print("  • figure3_synapse_map.png/pdf")
    print("  • figure4_ml_comparison.png/pdf")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
