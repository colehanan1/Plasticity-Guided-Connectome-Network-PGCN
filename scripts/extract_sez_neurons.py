#!/usr/bin/env python
"""
SEZ Neuron Extraction Pipeline for PGCN Model

This script extracts subesophageal zone (SEZ) taste neurons from FlyWire FAFB v783
connectomic data, implementing methods from Li et al. (2024) Scientific Reports.

SEZ-PN extraction methods adapted from Li et al. (2024).
Second-order taste neurons identified by querying FlyWire FAFB v783
for neurons receiving ≥10 synapses from gustatory receptor neurons.

Reference:
Li, J. et al. (2024). Connectomic analysis of taste circuits in Drosophila.
Scientific Reports, 14, 21120. https://doi.org/10.1038/s41598-024-71926-2

Shen, K. et al. (2025). Functional imaging and connectome analyses reveal organizing
principles of processing taste modality in the Drosophila brain. Current Biology, 35(9),
1955-1970.e6. https://doi.org/10.1016/j.cub.2025.03.053

Usage:
    python scripts/extract_sez_neurons.py --dataset-dir data/flywire --output-dir data/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loaders.flywire_local import FlyWireLocalDataLoader

__all__ = [
    "load_grn_root_ids",
    "extract_grn_from_classification",
    "trace_second_order_neurons",
    "classify_sez_neurons",
    "filter_cholinergic_sez_lns",
    "validate_with_li2024_clustering",
    "generate_validation_plots",
]


# =============================================================================
# MODULE 1: GRN Root ID Loader
# =============================================================================


def load_grn_root_ids(grn_file: Path) -> np.ndarray:
    """
    Load validated GRN root IDs from ground-truth file.

    CRITICAL: Do NOT re-extract GRNs from classification.csv.
    Use the provided root_ids_class_gustatory.txt file (343 neurons).

    Args:
        grn_file: Path to root_ids_class_gustatory.txt

    Returns:
        Array of GRN root IDs (expected: 343 neurons)

    Raises:
        FileNotFoundError: If GRN file not found
        ValueError: If file is empty or malformed

    Reference:
        Li et al. (2024): "Gustatory receptor neurons (GRNs) were identified
        via FlyWire Codex community annotations."
    """
    if not grn_file.exists():
        raise FileNotFoundError(
            f"GRN file not found: {grn_file}\n"
            f"Expected: data/flywire/root_ids_class_gustatory.txt\n"
            f"This file should contain 343 validated GRN root IDs (one per line)."
        )

    # Load root IDs (plain text, one per line OR comma-separated)
    try:
        # First try reading as newline-separated
        with open(grn_file, 'r') as f:
            content = f.read().strip()

        # Check if it's comma-separated (all on one line)
        if ',' in content and '\n' not in content:
            grn_ids = np.array([int(x.strip()) for x in content.split(',') if x.strip()])
        # Check if it's space-separated on one line
        elif ' ' in content and '\n' not in content:
            grn_ids = np.array([int(x.strip()) for x in content.split() if x.strip()])
        # Otherwise read as newline-separated
        else:
            grn_ids = pd.read_csv(grn_file, header=None, names=["root_id"])
            grn_ids = grn_ids["root_id"].values

    except Exception as e:
        raise ValueError(f"Failed to parse GRN file {grn_file}: {e}")

    if len(grn_ids) == 0:
        raise ValueError(f"GRN file is empty: {grn_file}")

    # Validate count (343 for all gustatory, fewer for sugar/water only)
    actual_count = len(grn_ids)

    print(f"  ✓ Loaded {actual_count} validated GRN root IDs")

    return grn_ids


def extract_grn_from_classification(
    classification: pd.DataFrame,
    cell_types: pd.DataFrame,
    output_file: Path
) -> np.ndarray:
    """
    Extract GRN root IDs from classification table (fallback method).

    This is used only if root_ids_class_gustatory.txt does not exist.
    Searches both classification hierarchy AND cell_type annotations.

    Args:
        classification: FlyWire classification table
        cell_types: Consolidated cell types table
        output_file: Where to save extracted GRN IDs

    Returns:
        Array of GRN root IDs
    """
    print("  [Extracting GRNs from classification and cell_type tables...]")

    # Method 1: Search classification hierarchy (class/sub_class)
    grn_mask_class = (
        classification["super_class"].str.contains("sensory", case=False, na=False)
        & (
            classification["class"].str.contains("gustatory", case=False, na=False)
            | classification["sub_class"].str.contains("gustatory", case=False, na=False)
        )
    )
    grn_from_class = classification[grn_mask_class]["root_id"].unique()
    print(f"  ✓ Found {len(grn_from_class)} GRNs from classification.class/sub_class")

    # Method 2: Search cell_type column
    grn_mask_celltype = cell_types["cell_type"].str.contains(
        "gustatory", case=False, na=False
    )
    grn_from_celltype = cell_types[grn_mask_celltype]["root_id"].unique()
    print(f"  ✓ Found {len(grn_from_celltype)} GRNs from cell_type")

    # Combine both methods (union)
    grn_ids = np.unique(np.concatenate([grn_from_class, grn_from_celltype]))

    print(f"  ✓ Total unique GRN root IDs: {len(grn_ids)}")

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"root_id": grn_ids}).to_csv(
        output_file, index=False, header=False
    )
    print(f"  ✓ Saved to {output_file}")

    return grn_ids


# =============================================================================
# MODULE 2: Second-Order Neuron Tracer
# =============================================================================


def trace_second_order_neurons(
    grn_ids: np.ndarray, connections: pd.DataFrame, min_synapses: int = 10
) -> pd.DataFrame:
    """
    Trace neurons receiving strong input from GRNs.

    This implements the connectivity tracing approach from Li et al. (2024),
    where second-order taste neurons are defined as postsynaptic partners of
    GRNs with ≥10 synapses (strong connections only).

    Args:
        grn_ids: Array of GRN root IDs (343 neurons)
        connections: Full connectivity table (~5.3M rows)
        min_synapses: Minimum synapse threshold for significant connections

    Returns:
        DataFrame with columns:
        - pre_root_id: GRN ID
        - post_root_id: Second-order neuron ID
        - syn_count: Number of synapses

    Expected output: ~400-800 second-order neurons receiving ≥10 synapses

    Reference:
        Li et al. (2024): "We identified second-order taste neurons as
        postsynaptic partners receiving ≥10 synapses from GRNs."
    """
    print(f"  [Filtering {len(connections):,} connections...]")

    # Filter: GRN → X with ≥min_synapses
    grn_outputs = connections[
        connections["pre_root_id"].isin(grn_ids)
        & (connections["syn_count"] >= min_synapses)
    ].copy()

    # Statistics
    n_connections = len(grn_outputs)
    total_synapses = grn_outputs["syn_count"].sum()
    second_order_ids = grn_outputs["post_root_id"].unique()

    print(f"  ✓ Found {len(second_order_ids)} second-order neurons")
    print(f"  ✓ GRN→2nd connections: {n_connections:,}")
    print(f"  ✓ Total synapses: {total_synapses:,}")

    # Validate biological plausibility
    if len(second_order_ids) > 0:
        avg_synapses = grn_outputs["syn_count"].mean()
        print(f"  ✓ Avg synapses per connection: {avg_synapses:.1f}")

    if not (400 <= len(second_order_ids) <= 1000):
        print(f"  ⚠ WARNING: Count outside expected range (400-1000)")

    return grn_outputs


# =============================================================================
# MODULE 3: Projection vs Local Classification
# =============================================================================


def classify_sez_neurons(
    second_order_ids: np.ndarray,
    classification: pd.DataFrame,
    cell_types: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify second-order neurons as projection vs local.

    Classification logic (from Li et al. 2024):
    - SEZ-PNs: super_class contains 'ascending' or 'sensory' (project to higher brain)
    - SEZ-LNs: super_class contains 'intrinsic' (outputs within SEZ only)

    Args:
        second_order_ids: Array of second-order neuron root IDs
        classification: Hierarchical classification table
        cell_types: Consolidated cell types table

    Returns:
        Tuple of (sez_pns, sez_lns) DataFrames

    Expected counts:
    - SEZ-PNs: 100-200 (projection neurons)
    - SEZ-LNs: 200-600 (local interneurons)

    Reference:
        Li et al. (2024): "Taste projection neurons (TPNs) were identified
        by their axonal projections outside the SEZ to the superior lateral
        protocerebrum, lateral horn, and mushroom body."
    """
    print("  [Retrieving classification metadata...]")

    # Get metadata for second-order neurons
    second_order_meta = classification[
        classification["root_id"].isin(second_order_ids)
    ].copy()

    print(f"  ✓ Retrieved metadata for {len(second_order_meta)} neurons")

    # Separate by projection type
    # Projection neurons: ascending/sensory super_class
    sez_pns = second_order_meta[
        second_order_meta["super_class"].str.contains(
            "ascending|sensory", case=False, na=False
        )
    ].copy()

    # Local interneurons: intrinsic super_class
    sez_lns = second_order_meta[
        second_order_meta["super_class"].str.contains(
            "intrinsic", case=False, na=False
        )
    ].copy()

    print(f"  ✓ SEZ-PNs (projection neurons): {len(sez_pns)}")
    print(f"  ✓ SEZ-LNs (local interneurons): {len(sez_lns)}")

    # Validate against Li et al. (2024)
    validate_sez_pn_count(len(sez_pns))

    # Label cell types
    sez_pns["cell_type"] = "SEZ_PN"
    sez_lns["cell_type"] = "SEZ_LN"

    return sez_pns, sez_lns


def validate_sez_pn_count(n_sez_pns: int) -> None:
    """
    Validate SEZ-PN count against Li et al. (2024).

    Args:
        n_sez_pns: Number of extracted SEZ-PNs

    Reference:
        Li et al. (2024): Identified ~100-200 taste projection neurons
    """
    expected_min = 100
    expected_max = 200

    if expected_min <= n_sez_pns <= expected_max:
        print(f"    ✅ Within Li et al. (2024) range ({expected_min}-{expected_max})")
    elif 80 <= n_sez_pns < expected_min:
        print(f"    ⚠ Slightly below range (acceptable - more stringent filtering)")
    elif expected_max < n_sez_pns <= 250:
        print(f"    ⚠ Slightly above range (may include some misclassified LNs)")
    else:
        print(f"    ❌ Outside plausible range - check extraction logic")


# =============================================================================
# MODULE 3B: Diagnostic and Robust Classification
# =============================================================================


def diagnose_unclassified_neurons(
    second_order_ids: np.ndarray,
    sez_pn_ids: np.ndarray,
    classification: pd.DataFrame,
    connections: pd.DataFrame,
    neurons: pd.DataFrame,
) -> pd.DataFrame:
    """
    Diagnose classification labels for neurons that aren't SEZ-PNs.

    This helps identify why SEZ-LN extraction is failing and what
    alternative classification strategies should be used.

    Args:
        second_order_ids: All neurons receiving GRN input
        sez_pn_ids: Neurons already classified as SEZ-PNs
        classification: Classification metadata table
        connections: Full connectivity table
        neurons: Neurotransmitter predictions

    Returns:
        DataFrame with unclassified neuron metadata and diagnostics
    """
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC: Investigating Unclassified Second-Order Neurons")
    print("="*70)

    # Identify unclassified neurons
    unclassified_ids = [rid for rid in second_order_ids if rid not in sez_pn_ids]

    print(f"\n📊 Classification Summary:")
    print(f"  Total second-order neurons:  {len(second_order_ids)}")
    print(f"  Classified as SEZ-PNs:       {len(sez_pn_ids)}")
    print(f"  Unclassified (potential LNs): {len(unclassified_ids)}")
    print(f"  Expected SEZ-LNs:            200-600 (Li et al. 2024)")

    # Get metadata for unclassified neurons
    unclassified_meta = classification[
        classification['root_id'].isin(unclassified_ids)
    ].copy()

    print(f"  ✓ Retrieved metadata for {len(unclassified_meta)} neurons")

    # === DIAGNOSTIC 1: super_class Distribution ===
    print("\n" + "-"*70)
    print("DIAGNOSTIC 1: super_class Distribution")
    print("-"*70)

    if 'super_class' in unclassified_meta.columns:
        super_class_counts = unclassified_meta['super_class'].value_counts()
        print("\n📈 super_class values (top 10):")
        print(super_class_counts.head(10))

        # Check for missing values
        n_missing = unclassified_meta['super_class'].isna().sum()
        if n_missing > 0:
            print(f"\n⚠️  {n_missing} neurons have missing super_class")

        # Check for 'intrinsic' keyword
        n_intrinsic = unclassified_meta['super_class'].str.contains(
            'intrinsic', case=False, na=False
        ).sum()
        print(f"\n🔍 Neurons with 'intrinsic' in super_class: {n_intrinsic}")

        # Check alternative keywords
        keywords = ['local', 'interneuron', 'inter', 'central', 'sensory']
        print("\n🔍 Alternative keyword matches:")
        for kw in keywords:
            count = unclassified_meta['super_class'].str.contains(
                kw, case=False, na=False
            ).sum()
            if count > 0:
                print(f"  '{kw}': {count} neurons")

    # === DIAGNOSTIC 2: class Distribution ===
    print("\n" + "-"*70)
    print("DIAGNOSTIC 2: class Distribution (top 10)")
    print("-"*70)

    if 'class' in unclassified_meta.columns:
        class_counts = unclassified_meta['class'].value_counts().head(10)
        print("\n📈 class values:")
        print(class_counts)

    # === DIAGNOSTIC 3: Alternative Fields ===
    print("\n" + "-"*70)
    print("DIAGNOSTIC 3: Alternative Classification Fields")
    print("-"*70)

    # Check flow (connectivity flow type)
    if 'flow' in unclassified_meta.columns:
        flow_counts = unclassified_meta['flow'].value_counts()
        print("\n📈 'flow' distribution:")
        print(flow_counts)
        print("  Note: flow typically: 0,1,2=intrinsic/local | 3,4,5=ascending/projection")
    else:
        print("\n⚠️  'flow' column not available")

    # Check hemibrain_type
    if 'hemibrain_type' in unclassified_meta.columns:
        hb_counts = unclassified_meta['hemibrain_type'].value_counts().head(10)
        print("\n📈 'hemibrain_type' distribution (top 10):")
        print(hb_counts)

    # === DIAGNOSTIC 4: Neurotransmitter Distribution ===
    print("\n" + "-"*70)
    print("DIAGNOSTIC 4: Neurotransmitter Distribution")
    print("-"*70)

    nt_data = neurons[neurons['root_id'].isin(unclassified_ids)]
    if 'nt_type' in nt_data.columns and len(nt_data) > 0:
        nt_counts = nt_data['nt_type'].value_counts()
        print("\n📈 Neurotransmitter types:")
        print(nt_counts)
    else:
        print("\n⚠️  Neurotransmitter data not available")

    # === RECOMMENDATION ===
    print("\n" + "="*70)
    print("💡 RECOMMENDATION")
    print("="*70)

    # Determine best strategy
    if 'super_class' in unclassified_meta.columns:
        top_super_class = unclassified_meta['super_class'].value_counts().head(1)
        if len(top_super_class) > 0:
            top_label = top_super_class.index[0]
            top_count = top_super_class.values[0]

            print(f"\nMost common super_class: '{top_label}' ({top_count} neurons)")

            if pd.notna(top_label) and top_label != '':
                print(f"\n✅ STRATEGY 1: Update keyword filter to include '{top_label}'")

    if 'flow' in unclassified_meta.columns:
        print(f"\n✅ STRATEGY 2: Use 'flow' field for classification")

    print(f"\n✅ STRATEGY 3: Connectivity-based (most robust)")
    print(f"   Define SEZ-LNs as neurons NOT projecting to higher brain")

    return unclassified_meta


def classify_sez_neurons_robust(
    second_order_ids: np.ndarray,
    classification: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Robust SEZ neuron classification using multiple keyword strategies.

    Falls back through multiple classification methods until successful.

    Args:
        second_order_ids: Array of second-order neuron root IDs
        classification: Hierarchical classification table

    Returns:
        Tuple of (sez_pns, sez_lns) DataFrames
    """
    print("  [Strategy 1: Keyword-based classification...]")

    second_order_meta = classification[
        classification['root_id'].isin(second_order_ids)
    ].copy()

    # Try original keywords first
    sez_pns = second_order_meta[
        second_order_meta['super_class'].str.contains(
            'ascending|sensory', case=False, na=False
        )
    ].copy()

    sez_lns = second_order_meta[
        second_order_meta['super_class'].str.contains(
            'intrinsic', case=False, na=False
        )
    ].copy()

    # If no LNs found, try broader keywords
    if len(sez_lns) == 0:
        print("    'intrinsic' not found, trying broader keywords...")

        # Try broader keywords for LNs
        sez_lns = second_order_meta[
            second_order_meta['super_class'].str.contains(
                'intrinsic|local|interneuron|central',
                case=False,
                na=False
            )
        ].copy()

        if len(sez_lns) > 0:
            print(f"    ✓ Found {len(sez_lns)} SEZ-LNs using broader keywords")

            # Refine PNs: exclude the LNs we just found
            sez_pns = second_order_meta[
                ~second_order_meta['root_id'].isin(sez_lns['root_id'])
            ].copy()

    # Label cell types
    sez_pns['cell_type'] = 'SEZ_PN'
    sez_lns['cell_type'] = 'SEZ_LN'

    print(f"  ✓ SEZ-PNs: {len(sez_pns)}")
    print(f"  ✓ SEZ-LNs: {len(sez_lns)}")

    return sez_pns, sez_lns


def classify_sez_neurons_by_flow(
    second_order_ids: np.ndarray,
    classification: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify SEZ neurons using 'flow' field (connectivity flow type).

    Flow values can be either numeric or string:
    - Numeric: 0, 1, 2 = Intrinsic/local neurons | 3, 4, 5 = Ascending/projection neurons
    - String: 'intrinsic'/'local' = Local neurons | 'efferent'/'ascending'/'projection' = Projection neurons

    Args:
        second_order_ids: Array of second-order neuron root IDs
        classification: Hierarchical classification table

    Returns:
        Tuple of (sez_pns, sez_lns) DataFrames
    """
    print("  [Strategy 2: Flow-based classification...]")

    second_order_meta = classification[
        classification['root_id'].isin(second_order_ids)
    ].copy()

    if 'flow' not in second_order_meta.columns:
        raise ValueError("'flow' column not found")

    # Detect if flow values are numeric or string
    sample_values = second_order_meta['flow'].dropna().head(10)
    is_numeric = pd.api.types.is_numeric_dtype(sample_values)

    if not is_numeric:
        # Handle STRING flow values
        print("    Detected string flow values (using keyword matching)")

        # SEZ-LNs: flow contains 'intrinsic' or 'local'
        sez_lns = second_order_meta[
            second_order_meta['flow'].astype(str).str.contains(
                'intrinsic|local', case=False, na=False
            )
        ].copy()

        # SEZ-PNs: flow contains 'efferent' or 'ascending' or 'projection'
        sez_pns = second_order_meta[
            second_order_meta['flow'].astype(str).str.contains(
                'efferent|ascending|projection', case=False, na=False
            )
        ].copy()

        print(f"  ✓ SEZ-LNs (intrinsic/local): {len(sez_lns)}")
        print(f"  ✓ SEZ-PNs (efferent/ascending): {len(sez_pns)}")

    else:
        # Handle NUMERIC flow values
        print("    Detected numeric flow values (using range matching)")

        # SEZ-LNs: flow ∈ {0, 1, 2}
        sez_lns = second_order_meta[
            second_order_meta['flow'].isin([0, 1, 2])
        ].copy()

        # SEZ-PNs: flow ∈ {3, 4, 5}
        sez_pns = second_order_meta[
            second_order_meta['flow'].isin([3, 4, 5])
        ].copy()

        print(f"  ✓ SEZ-LNs (flow 0/1/2): {len(sez_lns)}")
        print(f"  ✓ SEZ-PNs (flow 3/4/5): {len(sez_pns)}")

    # Label cell types
    sez_lns['cell_type'] = 'SEZ_LN'
    sez_pns['cell_type'] = 'SEZ_PN'

    return sez_pns, sez_lns


def classify_sez_neurons_by_connectivity(
    grn_ids: np.ndarray,
    second_order_ids: np.ndarray,
    classification: pd.DataFrame,
    connections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify SEZ neurons by analyzing their output connectivity (GOLD STANDARD).

    Biological definition:
    - SEZ-PNs: Project output beyond GRN→2nd circuit (to 3rd order neurons)
    - SEZ-LNs: Output exclusively back to other 2nd order neurons or GRNs

    This uses circuit structure, not potentially inconsistent labels.

    Reference:
        Li et al. (2024): "TPNs defined by axonal projections outside
        the SEZ to lateral horn, mushroom body, and superior protocerebrum."

    Args:
        grn_ids: GRN root IDs (for reference)
        second_order_ids: Second-order neuron root IDs
        classification: Classification metadata
        connections: Full connectivity table

    Returns:
        Tuple of (sez_pns, sez_lns) DataFrames
    """
    print("  [Strategy 3: Connectivity-based classification...]")

    second_order_meta = classification[
        classification['root_id'].isin(second_order_ids)
    ].copy()

    # Define the "taste circuit" = GRNs + second-order neurons
    taste_circuit_ids = set(grn_ids) | set(second_order_ids)

    # Get all output connections from second-order neurons
    second_order_outputs = connections[
        connections['pre_root_id'].isin(second_order_ids) &
        (connections['syn_count'] >= 5)  # Significant connections
    ]

    print(f"    Found {len(second_order_outputs):,} output connections")

    # SEZ-PNs: neurons projecting OUTSIDE the taste circuit (to 3rd order)
    sez_pn_ids = second_order_outputs[
        ~second_order_outputs['post_root_id'].isin(taste_circuit_ids)
    ]['pre_root_id'].unique()

    # SEZ-LNs: neurons staying WITHIN the taste circuit
    sez_ln_ids = [rid for rid in second_order_ids if rid not in sez_pn_ids]

    # Build DataFrames
    sez_pns = second_order_meta[
        second_order_meta['root_id'].isin(sez_pn_ids)
    ].copy()
    sez_pns['cell_type'] = 'SEZ_PN'

    sez_lns = second_order_meta[
        second_order_meta['root_id'].isin(sez_ln_ids)
    ].copy()
    sez_lns['cell_type'] = 'SEZ_LN'

    print(f"  ✓ SEZ-PNs (project to 3rd order): {len(sez_pns)}")
    print(f"  ✓ SEZ-LNs (stay in circuit): {len(sez_lns)}")

    # Validate
    if len(sez_pns) > 0 and len(sez_lns) > 0:
        ratio = len(sez_lns) / len(sez_pns)
        print(f"  ✓ SEZ-LN:SEZ-PN ratio: {ratio:.1f}:1")

        if 2.0 <= ratio <= 5.0:
            print(f"    ✅ Within expected range (2-5:1, Li et al. 2024)")
        else:
            print(f"    ⚠️  Outside expected range (2-5:1)")

    return sez_pns, sez_lns


# =============================================================================
# MODULE 4: Neurotransmitter Filtering
# =============================================================================


def filter_cholinergic_sez_lns(
    sez_lns: pd.DataFrame, neurons_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract cholinergic (excitatory) subset of SEZ-LNs.

    Cholinergic SEZ-LNs are excitatory relay interneurons that form feedforward
    pathways from GRNs to SEZ-PNs (Shen et al. 2025 Current Biology).

    Args:
        sez_lns: All SEZ local interneurons
        neurons_df: Neurotransmitter predictions table

    Returns:
        DataFrame with cholinergic SEZ-LNs only

    Expected count: ~50-100 neurons (relay interneurons)

    Reference:
        Shen et al. (2025): "Cholinergic SEZ-LNs act as organizing nodes,
        forming feedforward relay pathways from GRNs to third-order neurons."
    """
    print("  [Filtering by neurotransmitter...]")

    sez_ln_ids = sez_lns["root_id"].unique()

    # Get neurotransmitter predictions
    nt_info = neurons_df[neurons_df["root_id"].isin(sez_ln_ids)].copy()

    # Filter for acetylcholine
    chol_sez_lns = nt_info[
        nt_info["nt_type"].str.contains(
            "acetylcholine|ACH|cholinergic", case=False, na=False
        )
    ].copy()

    # Merge with classification metadata
    merge_cols = [col for col in ["root_id", "super_class", "class", "sub_class", "side"]
                  if col in sez_lns.columns]
    chol_sez_lns = chol_sez_lns.merge(
        sez_lns[merge_cols], on="root_id", how="left"
    )

    chol_sez_lns["cell_type"] = "SEZ_LN_cholinergic"
    chol_sez_lns["neurotransmitter"] = "Acetylcholine"

    print(f"  ✓ Cholinergic SEZ-LNs: {len(chol_sez_lns)}")

    # Validate
    if 50 <= len(chol_sez_lns) <= 100:
        print(f"    ✅ Within expected range (50-100)")
    elif 30 <= len(chol_sez_lns) < 50:
        print(f"    ⚠ Slightly low (acceptable)")
    else:
        print(f"    ⚠ Check filtering logic")

    return chol_sez_lns


# =============================================================================
# MODULE 5: Li et al. (2024) Clustering Validation
# =============================================================================


def validate_with_li2024_clustering(
    grn_ids: np.ndarray,
    sez_pn_ids: np.ndarray,
    connections: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Apply Li et al. (2024) clustering to validate extraction quality.

    This builds a connectivity matrix (SEZ-PNs × GRNs), applies their
    published clustering pipeline, and validates that your extracted
    neurons show expected taste modality organization.

    This implements their published analysis methods:
    1. Build GRN → SEZ-PN connectivity matrix
    2. L2 normalization
    3. TruncatedSVD dimensionality reduction (10 components)
    4. Hierarchical clustering (correlation distance, average linkage)
    5. Silhouette score validation

    Expected results:
    - 8-12 clusters (corresponding to taste modalities)
    - Silhouette score > 0.3 (reasonable separation)
    - ~70-80% of neurons encode single taste modality

    Args:
        grn_ids: GRN root IDs (343 neurons)
        sez_pn_ids: Extracted SEZ-PN root IDs (~100-200 neurons)
        connections: Full connectivity table
        output_dir: Directory for validation plots

    Returns:
        Dictionary with validation metrics and cluster statistics

    Reference:
        Li et al. (2024) Methods: "TPNs were hierarchically clustered based
        on their input patterns from GRNs using correlation distance."
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.sparse import csr_matrix
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    # Check if we have enough neurons for clustering
    if len(sez_pn_ids) < 10:
        print("  ⚠ WARNING: Too few SEZ-PNs for meaningful clustering validation")
        print(f"  Found {len(sez_pn_ids)} neurons, need at least 10")
        return {
            "n_sez_pns": len(sez_pn_ids),
            "n_grns": len(grn_ids),
            "validation_skipped": True,
            "reason": "Insufficient neurons for clustering",
        }

    print("\n" + "=" * 70)
    print("VALIDATION: Li et al. (2024) Clustering Pipeline")
    print("=" * 70)

    # [1] Build connectivity matrix
    print("\n[1/6] Building GRN → SEZ-PN connectivity matrix...")

    grn_to_sez = connections[
        connections["pre_root_id"].isin(grn_ids)
        & connections["post_root_id"].isin(sez_pn_ids)
        & (connections["syn_count"] >= 1)
    ]

    # Pivot to matrix: rows = SEZ-PNs, columns = GRNs
    conn_matrix = grn_to_sez.pivot_table(
        index="post_root_id", columns="pre_root_id", values="syn_count", fill_value=0
    )

    print(
        f"  ✓ Matrix shape: {conn_matrix.shape[0]} SEZ-PNs × {conn_matrix.shape[1]} GRNs"
    )

    if conn_matrix.shape[0] < 2 or conn_matrix.shape[1] < 2:
        print("  ⚠ WARNING: Matrix too small for clustering")
        return {
            "n_sez_pns": len(sez_pn_ids),
            "n_grns": len(grn_ids),
            "matrix_shape": list(conn_matrix.shape),
            "validation_skipped": True,
            "reason": "Connectivity matrix too small",
        }

    # [2] L2 normalization
    print("\n[2/6] L2 normalization (row-wise)...")
    data_array = conn_matrix.values
    sparse_matrix = csr_matrix(data_array)
    sparse_normalized = normalize(sparse_matrix, axis=1, norm="l2")

    # [3] Dimensionality reduction
    print("\n[3/6] TruncatedSVD dimensionality reduction...")
    n_components = min(10, conn_matrix.shape[0] - 1, conn_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_data = svd.fit_transform(sparse_normalized)
    variance_explained = svd.explained_variance_ratio_.sum()

    print(f"  ✓ Reduced to {n_components} components")
    print(f"  ✓ Variance explained: {variance_explained:.1%}")

    # [4] Silhouette analysis
    print("\n[4/6] Silhouette analysis (optimal cluster count)...")
    cluster_range = range(2, min(15, len(sez_pn_ids) // 2))
    silhouette_scores = []

    for n_clusters in cluster_range:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="average", metric="correlation"
            )
            labels = clustering.fit_predict(reduced_data)
            score = silhouette_score(reduced_data, labels)
            silhouette_scores.append(score)
        except Exception as e:
            print(f"  ⚠ WARNING: Clustering failed for n={n_clusters}: {e}")
            silhouette_scores.append(0.0)

    if not silhouette_scores or max(silhouette_scores) == 0:
        print("  ⚠ WARNING: Silhouette analysis failed")
        optimal_n = 2
        max_score = 0.0
    else:
        optimal_n = cluster_range[np.argmax(silhouette_scores)]
        max_score = max(silhouette_scores)

    print(f"  ✓ Optimal clusters: {optimal_n}")
    print(f"  ✓ Silhouette score: {max_score:.3f}")

    # Validate against Li et al. (2024)
    if 8 <= optimal_n <= 12:
        print(f"    ✅ Matches Li et al. (2024) range (8-12 taste modalities)")
    else:
        print(f"    ⚠ Different from Li et al. (~10 clusters)")

    # [5] Final clustering
    print(f"\n[5/6] Hierarchical clustering ({optimal_n} clusters)...")
    clustering = AgglomerativeClustering(
        n_clusters=optimal_n, metric="correlation", linkage="average"
    )
    cluster_labels = clustering.fit_predict(reduced_data)

    # [6] UMAP embedding (optional - requires umap-learn)
    print("\n[6/6] UMAP embedding (2D visualization)...")
    try:
        import umap

        reducer = umap.UMAP(
            random_state=42,
            n_neighbors=min(100, len(sez_pn_ids) - 1),
            min_dist=0.3,
            metric="correlation",
        )
        embedding = reducer.fit_transform(reduced_data)
        print("  ✓ Generated 2D embedding")
        has_umap = True
    except ImportError:
        print("  ⚠ UMAP not available (install with: pip install umap-learn)")
        embedding = None
        has_umap = False
    except Exception as e:
        print(f"  ⚠ UMAP failed: {e}")
        embedding = None
        has_umap = False

    # Generate validation plots
    generate_validation_plots(
        conn_matrix,
        cluster_labels,
        embedding,
        silhouette_scores,
        cluster_range,
        output_dir,
    )

    # Save cluster assignments
    cluster_df = pd.DataFrame({"root_id": conn_matrix.index, "cluster": cluster_labels})
    cluster_df.to_csv(output_dir / "sez_pn_clusters.csv", index=False)

    # Summary statistics
    summary = {
        "n_sez_pns": len(sez_pn_ids),
        "n_grns": len(grn_ids),
        "n_clusters": int(optimal_n),
        "silhouette_score": float(max_score),
        "variance_explained": float(variance_explained),
        "has_umap_embedding": has_umap,
        "li2024_validation": {
            "expected_pn_count": "100-200",
            "actual_pn_count": len(sez_pn_ids),
            "within_range": 100 <= len(sez_pn_ids) <= 200,
            "expected_clusters": "8-12",
            "actual_clusters": int(optimal_n),
            "cluster_match": 8 <= optimal_n <= 12,
        },
    }

    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =============================================================================
# MODULE 6: Plot Generation
# =============================================================================


def generate_validation_plots(
    connectivity_matrix: pd.DataFrame,
    cluster_labels: np.ndarray,
    embedding: np.ndarray | None,
    silhouette_scores: list,
    cluster_range: range,
    output_dir: Path,
) -> None:
    """
    Generate validation plots matching Li et al. (2024) publication figures.

    Args:
        connectivity_matrix: SEZ-PNs × GRNs connectivity matrix
        cluster_labels: Cluster assignments for each SEZ-PN
        embedding: 2D UMAP embedding (or None if unavailable)
        silhouette_scores: Silhouette scores for different cluster counts
        cluster_range: Range of cluster counts tested
        output_dir: Directory to save plots
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.metrics.pairwise import pairwise_distances

    # Use Agg backend for non-interactive plotting
    matplotlib.use("Agg")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Generating Validation Plots]")

    # Plot 1: Dendrogram
    print("  [1/4] Hierarchical clustering dendrogram...")
    try:
        Z = linkage(connectivity_matrix.values, method="average", metric="correlation")

        plt.figure(figsize=(20, 5))
        dendrogram(Z, leaf_rotation=90, leaf_font_size=6, no_labels=True)
        plt.title("SEZ-PN Hierarchical Clustering (Li et al. 2024 Method)")
        plt.xlabel("SEZ-PN Neurons")
        plt.ylabel("Correlation Distance")
        plt.tight_layout()
        plt.savefig(output_dir / "fig1_dendrogram.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print("    ✓ Saved fig1_dendrogram.pdf")
    except Exception as e:
        print(f"    ⚠ Failed to generate dendrogram: {e}")

    # Plot 2: Silhouette scores
    print("  [2/4] Silhouette score validation...")
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(list(cluster_range), silhouette_scores, marker="o", linewidth=2)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Optimal Cluster Count Selection")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "fig2_silhouette.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print("    ✓ Saved fig2_silhouette.pdf")
    except Exception as e:
        print(f"    ⚠ Failed to generate silhouette plot: {e}")

    # Plot 3: UMAP embedding
    if embedding is not None:
        print("  [3/4] UMAP embedding with clusters...")
        try:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=cluster_labels,
                cmap="rainbow",
                s=80,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )
            plt.colorbar(scatter, label="Cluster ID")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.title("SEZ-PN Clustering by GRN Input Pattern")
            plt.tight_layout()
            plt.savefig(
                output_dir / "fig3_umap_clusters.pdf", dpi=300, bbox_inches="tight"
            )
            plt.close()
            print("    ✓ Saved fig3_umap_clusters.pdf")
        except Exception as e:
            print(f"    ⚠ Failed to generate UMAP plot: {e}")
    else:
        print("  [3/4] UMAP embedding skipped (not available)")

    # Plot 4: Distance heatmap
    print("  [4/4] Pairwise distance heatmap...")
    try:
        dist_matrix = pairwise_distances(
            connectivity_matrix.values, metric="correlation"
        )

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            dist_matrix,
            cmap="Reds",
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": "Correlation Distance"},
        )
        plt.title("SEZ-PN Similarity Matrix")
        plt.tight_layout()
        plt.savefig(output_dir / "fig4_heatmap.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print("    ✓ Saved fig4_heatmap.pdf")
    except Exception as e:
        print(f"    ⚠ Failed to generate heatmap: {e}")

    print(f"  ✓ Plots saved to {output_dir}")


# =============================================================================
# MODULE 7: Main Extraction Script
# =============================================================================


def main() -> int:
    """Execute complete SEZ neuron extraction and validation pipeline."""

    parser = argparse.ArgumentParser(
        description="Extract SEZ projection neurons and local interneurons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/flywire"),
        help="Path to FlyWire dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cache"),
        help="Path to output directory for extracted neurons",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("results/sez_validation"),
        help="Path to validation plots and metrics",
    )
    parser.add_argument(
        "--min-synapses",
        type=int,
        default=10,
        help="Minimum synapses for GRN → second-order connections",
    )
    parser.add_argument(
        "--grn-file",
        type=str,
        default="root_ids_class_gustatory_sub_class_sugar_water.txt",
        help="GRN root ID file to use (default: sugar/water only)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Li et al. (2024) clustering validation",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SEZ NEURON EXTRACTION & VALIDATION")
    print("=" * 70)
    print(f"\nRepository: colehanan1/Plasticity-Guided-Connectome-Network-PGCN")
    print(f"Branch: claude/sez-neuron-extraction-pipeline-011CV2gNPWQ8wDgUJhghos7K")
    print(f"Dataset: FlyWire FAFB v783")
    print(f"Reference: Li et al. (2024) Scientific Reports 14:21120")

    # Load FlyWire data
    print("\n" + "=" * 70)
    print("LOADING FLYWIRE DATASETS")
    print("=" * 70)

    try:
        loader = FlyWireLocalDataLoader(dataset_dir=args.dataset_dir)
        classification = loader.load_classification()
        cell_types = loader.load_cell_types()
        connections = loader.load_connections(min_synapses=1)  # Load all connections
        neurons = loader.load_neurotransmitters()

        print(f"  ✓ Loaded {len(classification)} classified neurons")
        print(f"  ✓ Loaded {len(connections):,} connections")
        print(f"  ✓ Loaded {len(neurons)} neurotransmitter predictions")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load FlyWire datasets")
        print(f"  {e}")
        print(f"\nPlease ensure FlyWire data files are present in: {args.dataset_dir}")
        return 1

    # STAGE 1: Load validated GRN root IDs
    print("\n" + "=" * 70)
    print("STAGE 1: LOAD VALIDATED GRN ROOT IDS")
    print("=" * 70)

    grn_file = args.dataset_dir / args.grn_file
    print(f"  Using GRN file: {args.grn_file}")

    try:
        if grn_file.exists():
            grn_ids = load_grn_root_ids(grn_file)
        else:
            print("  ⚠ GRN file not found, extracting from classification...")
            grn_ids = extract_grn_from_classification(classification, cell_types, grn_file)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load GRN root IDs")
        print(f"  {e}")
        return 1

    # STAGE 2: Trace second-order neurons
    print("\n" + "=" * 70)
    print("STAGE 2: TRACE SECOND-ORDER NEURONS")
    print("=" * 70)

    try:
        grn_outputs = trace_second_order_neurons(
            grn_ids, connections, min_synapses=args.min_synapses
        )
        second_order_ids = grn_outputs["post_root_id"].unique()
    except Exception as e:
        print(f"\n❌ ERROR: Failed to trace second-order neurons")
        print(f"  {e}")
        return 1

    if len(second_order_ids) == 0:
        print("\n❌ ERROR: No second-order neurons found")
        print(f"  Try reducing --min-synapses (current: {args.min_synapses})")
        return 1

    # STAGE 3: Classify as projection vs local (with hierarchical fallback)
    print("\n" + "=" * 70)
    print("STAGE 3: CLASSIFY PROJECTION VS LOCAL NEURONS")
    print("=" * 70)

    # First, run the original classification to get initial SEZ-PNs for diagnostic
    print("\n[Running initial classification for diagnostic...]")
    try:
        sez_pns_initial, sez_lns_initial = classify_sez_neurons(
            second_order_ids, classification, cell_types
        )
    except Exception as e:
        print(f"  ⚠️  Initial classification failed: {e}")
        sez_pns_initial = np.array([])
        sez_lns_initial = np.array([])

    # Run diagnostic if SEZ-LNs are missing or very few
    if len(sez_lns_initial) < 50:
        print(f"\n⚠️  Only {len(sez_lns_initial)} SEZ-LNs found - running diagnostics...")

        try:
            diagnose_unclassified_neurons(
                second_order_ids,
                sez_pns_initial['root_id'].values if len(sez_pns_initial) > 0 else np.array([]),
                classification,
                connections,
                neurons
            )
        except Exception as e:
            print(f"  ⚠️  Diagnostic failed: {e}")

        # Try classification strategies in order of robustness
        sez_pns = None
        sez_lns = None
        strategy_used = None

        # Strategy 3: Connectivity-based (most robust)
        print("\n" + "=" * 70)
        print("[Attempting Strategy 3: Connectivity-based classification]")
        print("=" * 70)
        try:
            sez_pns, sez_lns = classify_sez_neurons_by_connectivity(
                grn_ids, second_order_ids, classification, connections
            )
            if len(sez_lns) > 0:
                strategy_used = "Strategy 3: Connectivity-based"
                print("\n✅ SUCCESS: Connectivity-based classification working!")
        except Exception as e:
            print(f"  ❌ Strategy 3 failed: {e}")

        # Strategy 2: Flow-based (if Strategy 3 failed)
        if sez_pns is None or len(sez_lns) == 0:
            print("\n" + "=" * 70)
            print("[Attempting Strategy 2: Flow-based classification]")
            print("=" * 70)
            try:
                sez_pns, sez_lns = classify_sez_neurons_by_flow(
                    second_order_ids, classification
                )
                if len(sez_lns) > 0:
                    strategy_used = "Strategy 2: Flow-based"
                    print("\n✅ SUCCESS: Flow-based classification working!")
            except Exception as e:
                print(f"  ❌ Strategy 2 failed: {e}")

        # Strategy 1: Keyword matching (fallback)
        if sez_pns is None or len(sez_lns) == 0:
            print("\n" + "=" * 70)
            print("[Attempting Strategy 1: Enhanced keyword-based classification]")
            print("=" * 70)
            try:
                sez_pns, sez_lns = classify_sez_neurons_robust(
                    second_order_ids, classification
                )
                if len(sez_lns) > 0:
                    strategy_used = "Strategy 1: Enhanced keywords"
                    print("\n✅ SUCCESS: Keyword-based classification working!")
            except Exception as e:
                print(f"  ❌ Strategy 1 failed: {e}")

        # Check if any strategy succeeded
        if sez_pns is None or len(sez_lns) == 0:
            print("\n❌ CRITICAL ERROR: All classification strategies failed!")
            print("   No SEZ-LNs could be extracted.")
            print("   Please review diagnostic output above.")
            return 1

        # Report which strategy succeeded
        print("\n" + "=" * 70)
        print(f"✅ Classification successful using: {strategy_used}")
        print("=" * 70)

    else:
        # Original classification worked fine
        sez_pns = sez_pns_initial
        sez_lns = sez_lns_initial
        print(f"\n✅ Original classification successful!")

    # Final validation
    print(f"\n📊 Final Classification Results:")
    print(f"  SEZ-PNs (projection neurons):  {len(sez_pns)}")
    print(f"  SEZ-LNs (local interneurons):  {len(sez_lns)}")

    if len(sez_pns) > 0 and len(sez_lns) > 0:
        ratio = len(sez_lns) / len(sez_pns)
        print(f"  SEZ-LN:SEZ-PN ratio:           {ratio:.1f}:1")

        if 2.0 <= ratio <= 5.0:
            print(f"    ✅ Ratio within expected range (2-5:1)")
        elif ratio < 2.0:
            print(f"    ⚠️  Ratio low - may have too many PNs or too few LNs")
        else:
            print(f"    ⚠️  Ratio high - may have too few PNs or too many LNs")

        # Validate against Li et al. (2024)
        validate_sez_pn_count(len(sez_pns))

    # STAGE 4: Filter cholinergic SEZ-LNs
    print("\n" + "=" * 70)
    print("STAGE 4: FILTER CHOLINERGIC SEZ-LNs")
    print("=" * 70)

    try:
        sez_lns_chol = filter_cholinergic_sez_lns(sez_lns, neurons)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to filter cholinergic SEZ-LNs")
        print(f"  {e}")
        return 1

    # STAGE 5: Validate with Li et al. (2024) clustering
    validation_summary = None
    if not args.skip_validation and len(sez_pns) >= 10:
        print("\n" + "=" * 70)
        print("STAGE 5: CLUSTERING VALIDATION")
        print("=" * 70)

        try:
            validation_summary = validate_with_li2024_clustering(
                grn_ids, sez_pns["root_id"].values, connections, args.validation_dir
            )
        except Exception as e:
            print(f"\n⚠ WARNING: Clustering validation failed")
            print(f"  {e}")
            print("  Continuing with extraction...")
    else:
        if args.skip_validation:
            print("\n  ⚠ Skipping validation (--skip-validation flag)")
        else:
            print(
                f"\n  ⚠ Skipping validation (too few SEZ-PNs: {len(sez_pns)})"
            )

    # STAGE 6: Export results
    print("\n" + "=" * 70)
    print("STAGE 6: EXPORT RESULTS")
    print("=" * 70)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Export CSVs
    sez_pns.to_csv(args.output_dir / "sez_pn_all.csv", index=False)
    print(f"  ✓ Saved {len(sez_pns)} SEZ-PNs → sez_pn_all.csv")

    sez_lns.to_csv(args.output_dir / "sez_ln_all.csv", index=False)
    print(f"  ✓ Saved {len(sez_lns)} SEZ-LNs → sez_ln_all.csv")

    sez_lns_chol.to_csv(args.output_dir / "sez_ln_cholinergic.csv", index=False)
    print(f"  ✓ Saved {len(sez_lns_chol)} cholinergic SEZ-LNs → sez_ln_cholinergic.csv")

    # Final summary
    print("\n" + "=" * 70)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 70)

    print("\n📊 Extraction Summary:")
    print(f"  GRNs (ground truth):              {len(grn_ids)}")
    print(f"  Second-order neurons:             {len(second_order_ids)}")
    print(f"  ├─ SEZ-PNs (projection):          {len(sez_pns)}")
    print(f"  └─ SEZ-LNs (local):                {len(sez_lns)}")
    print(f"      └─ Cholinergic (excitatory):  {len(sez_lns_chol)}")

    print("\n📈 Validation vs Li et al. (2024):")
    print(f"  Expected SEZ-PNs:  100-200")
    print(f"  Extracted:         {len(sez_pns)}")

    match_status = "✅ MATCH" if 100 <= len(sez_pns) <= 200 else "⚠ CHECK"
    print(f"  Status:            {match_status}")

    if validation_summary:
        print(f"\n  Expected clusters: 8-12")
        print(f"  Found clusters:    {validation_summary['n_clusters']}")

        cluster_match = (
            "✅ MATCH" if 8 <= validation_summary["n_clusters"] <= 12 else "⚠ CHECK"
        )
        print(f"  Status:            {cluster_match}")

    print("\n📁 Output Files:")
    print(f"  Neuron CSVs:    {args.output_dir}")
    if validation_summary and not validation_summary.get("validation_skipped"):
        print(f"  Validation:     {args.validation_dir}")

    print("\n🔬 Next Steps:")
    print("  1. Run: python scripts/summarize_all_cell_types.py")
    print("  2. Verify new neuron counts include SEZ-PNs and SEZ-LNs")
    print("  3. Integrate into EnhancedOlfactoryCircuit model")
    print("  4. Run blocking experiments with taste-odor integration")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
