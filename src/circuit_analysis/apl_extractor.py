#!/usr/bin/env python3
"""Extract and analyze APL (anterior paired lateral) inhibition neurons.

APL neurons provide global gain control via GABAergic inhibition to PNs,
enabling sparse KC coding and introducing trial-to-trial variability.

Functions
---------
extract_apl_neurons : Extract APL neurons from FlyWire classification
analyze_apl_inhibition : Quantify APL→PN and APL→KC inhibitory connections
compute_apl_statistics : Calculate APL inhibition strength and coverage

References
----------
.. [1] Caron et al. (2013). "Random convergence of olfactory inputs in the
       Drosophila mushroom body." Nature 497, 113-117.
.. [2] Lin et al. (2014). "Neural correlates of water reward in thirsty
       Drosophila." Nature Neuroscience 17, 1536-1542.
.. [3] Schlegel et al. (2023). "Whole-brain annotation and multi-connectome
       cell typing of Drosophila." Nature.

Author: Claude Code (Anthropic)
Date: 2025-12-01
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_apl_neurons(
    classification: pd.DataFrame,
    connections: pd.DataFrame,
    target_pns: pd.DataFrame,
    min_inhibitory_synapses: int = 5,
) -> pd.DataFrame:
    """Extract APL neurons using multiple identification strategies.

    APL (anterior paired lateral) neurons are large GABAergic neurons that
    provide global inhibition to PNs and KCs. Expected: 1-2 per hemisphere.

    Strategy
    --------
    1. Search for 'APL' in classification labels (cell_type, cell_class)
    2. Search for neurons with broad inhibitory output to PNs
    3. Filter by neurotransmitter (GABA) and connectivity patterns

    Parameters
    ----------
    classification : pd.DataFrame
        FlyWire neuron classification with columns: root_id, cell_type,
        cell_class, hemibrain_type, nt_type
    connections : pd.DataFrame
        FlyWire connections with columns: pre_root_id, post_root_id, syn_count
    target_pns : pd.DataFrame
        PN neurons to check APL connectivity against
    min_inhibitory_synapses : int, default=5
        Minimum synapses for APL→PN connection

    Returns
    -------
    apl_neurons : pd.DataFrame
        APL neurons with columns: root_id, cell_type, hemisphere, nt_type,
        n_pn_targets, total_pn_synapses, n_kc_targets

    Raises
    ------
    ValueError
        If no PNs provided or connections table is empty

    Examples
    --------
    >>> apl = extract_apl_neurons(classification, connections, pns)
    >>> print(f"Found {len(apl)} APL neurons")
    Found 2 APL neurons
    """
    if target_pns.empty:
        raise ValueError("No target PNs provided")
    if connections.empty:
        raise ValueError("Connections table is empty")

    logger.info("Extracting APL neurons...")

    # Strategy 1: Direct label search
    logger.info("  Strategy 1: Searching for 'APL' in classification labels...")
    apl_candidates = []

    # Check standard columns
    for col in ["cell_type", "cell_class", "hemibrain_type"]:
        if col in classification.columns:
            matches = classification[
                classification[col].str.contains("APL", case=False, na=False)
            ]
            if not matches.empty:
                logger.info(f"    Found {len(matches)} candidates in '{col}' column")
                apl_candidates.append(matches)

    # Also check processed_labels if available
    if "processed_labels" in classification.columns and not apl_candidates:
        logger.info("    Searching in 'processed_labels' column...")
        matches = classification[
            classification["processed_labels"].astype(str).str.contains(
                "APL", case=False, na=False
            )
        ]
        if not matches.empty:
            logger.info(f"    Found {len(matches)} candidates in 'processed_labels'")
            apl_candidates.append(matches)

    # Strategy 2: Search for broad inhibitory neurons
    logger.info("  Strategy 2: Identifying broad inhibitory neurons...")

    # Find neurons with broad PN connectivity
    pn_root_ids = set(target_pns["root_id"].values)

    # Count PN targets for each presynaptic neuron
    pn_connections = connections[connections["post_root_id"].isin(pn_root_ids)]

    pn_fanout = (
        pn_connections.groupby("pre_root_id")
        .agg({"post_root_id": "nunique", "syn_count": "sum"})
        .rename(columns={"post_root_id": "n_pn_targets", "syn_count": "total_pn_synapses"})
    )

    # APL should contact many PNs (>10% of total PNs)
    min_pn_targets = max(5, int(0.1 * len(target_pns)))
    broad_inhibitors = pn_fanout[
        (pn_fanout["n_pn_targets"] >= min_pn_targets)
        & (pn_fanout["total_pn_synapses"] >= min_inhibitory_synapses)
    ]

    logger.info(
        f"    Found {len(broad_inhibitors)} neurons with broad PN connectivity "
        f"(>={min_pn_targets} PNs)"
    )

    # Filter for GABAergic neurons using nt_type from connections table
    if "nt_type" in connections.columns:
        # Get neurons that make GABA connections
        gaba_connections = connections[
            connections["nt_type"].str.contains("gaba", case=False, na=False)
        ]
        gaba_neuron_ids = set(gaba_connections["pre_root_id"].unique())

        # Find GABAergic neurons with broad inhibitory output
        gaba_broad_ids = broad_inhibitors.index[broad_inhibitors.index.isin(gaba_neuron_ids)]

        if len(gaba_broad_ids) > 0:
            gaba_broad = classification[classification["root_id"].isin(gaba_broad_ids)]
            logger.info(f"    {len(gaba_broad)} GABAergic neurons with broad PN output")
            apl_candidates.append(gaba_broad)
    else:
        logger.warning("    'nt_type' column not found in connections, cannot filter by GABA")

    # Combine all strategies
    if not apl_candidates:
        logger.warning("⚠️  No APL neurons found using any strategy")
        return pd.DataFrame(
            columns=[
                "root_id",
                "cell_type",
                "hemisphere",
                "nt_type",
                "n_pn_targets",
                "total_pn_synapses",
                "n_kc_targets",
            ]
        )

    apl_neurons = pd.concat(apl_candidates, ignore_index=True)
    apl_neurons = apl_neurons.drop_duplicates(subset=["root_id"])

    # Add connectivity statistics
    apl_root_ids = apl_neurons["root_id"].values

    # PN connectivity
    apl_pn_stats = pn_fanout.loc[pn_fanout.index.isin(apl_root_ids)]
    apl_neurons = apl_neurons.merge(
        apl_pn_stats, left_on="root_id", right_index=True, how="left"
    )

    # KC connectivity
    kc_connections = connections[
        (connections["pre_root_id"].isin(apl_root_ids))
        & (connections["post_root_id"].isin(classification["root_id"]))
    ]

    # Filter for KCs (search in available label columns)
    if "cell_type" in classification.columns:
        kc_candidates = classification[
            classification["cell_type"].str.contains("KC", case=False, na=False)
        ]
    elif "processed_labels" in classification.columns:
        kc_candidates = classification[
            classification["processed_labels"].astype(str).str.contains(
                "KC", case=False, na=False
            )
        ]
    else:
        # Fallback: use all classified neurons (won't filter for KCs specifically)
        kc_candidates = classification

    kc_root_ids = set(kc_candidates["root_id"].values)

    apl_kc_stats = (
        kc_connections[kc_connections["post_root_id"].isin(kc_root_ids)]
        .groupby("pre_root_id")["post_root_id"]
        .nunique()
        .rename("n_kc_targets")
    )

    apl_neurons = apl_neurons.merge(
        apl_kc_stats, left_on="root_id", right_index=True, how="left"
    )

    # Fill NaN with 0
    apl_neurons["n_pn_targets"] = apl_neurons["n_pn_targets"].fillna(0).astype(int)
    apl_neurons["total_pn_synapses"] = apl_neurons["total_pn_synapses"].fillna(0).astype(int)
    apl_neurons["n_kc_targets"] = apl_neurons["n_kc_targets"].fillna(0).astype(int)

    # Sort by PN connectivity
    apl_neurons = apl_neurons.sort_values("total_pn_synapses", ascending=False)

    logger.info(f"✅ Extracted {len(apl_neurons)} APL neurons")
    logger.info(f"   Total PN targets: {apl_neurons['n_pn_targets'].sum()}")
    logger.info(f"   Total PN synapses: {apl_neurons['total_pn_synapses'].sum()}")

    return apl_neurons


def analyze_apl_inhibition(
    apl_neurons: pd.DataFrame,
    connections: pd.DataFrame,
    target_pns: pd.DataFrame,
    target_glomeruli: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Quantify APL→PN inhibitory connections.

    Computes synapse counts, coverage, and statistics for APL inhibition
    of PNs, grouped by glomerulus.

    Parameters
    ----------
    apl_neurons : pd.DataFrame
        APL neurons from extract_apl_neurons()
    connections : pd.DataFrame
        FlyWire connections table
    target_pns : pd.DataFrame
        PN neurons with 'glomerulus' column
    target_glomeruli : list of str, optional
        Glomeruli to analyze (default: all in target_pns)

    Returns
    -------
    inhibition_stats : pd.DataFrame
        Columns: glomerulus, n_pns, n_pns_inhibited, coverage_pct,
        mean_synapses, std_synapses, total_synapses

    Examples
    --------
    >>> stats = analyze_apl_inhibition(apl, connections, pns, ['DM3', 'VM2', 'DM1'])
    >>> print(stats)
      glomerulus  n_pns  n_pns_inhibited  coverage_pct  mean_synapses
    0       DM3     48               45          93.8          12.3
    """
    if apl_neurons.empty:
        logger.warning("No APL neurons provided")
        return pd.DataFrame(
            columns=[
                "glomerulus",
                "n_pns",
                "n_pns_inhibited",
                "coverage_pct",
                "mean_synapses",
                "std_synapses",
                "total_synapses",
            ]
        )

    logger.info("Analyzing APL→PN inhibition...")

    apl_root_ids = apl_neurons["root_id"].values
    pn_root_ids = set(target_pns["root_id"].values)

    # Get APL→PN connections
    apl_pn_connections = connections[
        (connections["pre_root_id"].isin(apl_root_ids))
        & (connections["post_root_id"].isin(pn_root_ids))
    ].copy()

    # Merge with PN glomeruli
    apl_pn_connections = apl_pn_connections.merge(
        target_pns[["root_id", "glomerulus"]], left_on="post_root_id", right_on="root_id", how="left"
    )

    # Filter by target glomeruli
    if target_glomeruli is not None:
        apl_pn_connections = apl_pn_connections[
            apl_pn_connections["glomerulus"].isin(target_glomeruli)
        ]
        pns_to_analyze = target_pns[target_pns["glomerulus"].isin(target_glomeruli)]
    else:
        pns_to_analyze = target_pns

    # Compute statistics by glomerulus
    stats_list = []

    for glom in pns_to_analyze["glomerulus"].unique():
        glom_pns = pns_to_analyze[pns_to_analyze["glomerulus"] == glom]
        glom_connections = apl_pn_connections[apl_pn_connections["glomerulus"] == glom]

        n_pns = len(glom_pns)
        n_pns_inhibited = glom_connections["post_root_id"].nunique()
        coverage_pct = 100 * n_pns_inhibited / n_pns if n_pns > 0 else 0

        if not glom_connections.empty:
            mean_synapses = glom_connections["syn_count"].mean()
            std_synapses = glom_connections["syn_count"].std()
            total_synapses = glom_connections["syn_count"].sum()
        else:
            mean_synapses = 0
            std_synapses = 0
            total_synapses = 0

        stats_list.append(
            {
                "glomerulus": glom,
                "n_pns": n_pns,
                "n_pns_inhibited": n_pns_inhibited,
                "coverage_pct": coverage_pct,
                "mean_synapses": mean_synapses,
                "std_synapses": std_synapses,
                "total_synapses": total_synapses,
            }
        )

    inhibition_stats = pd.DataFrame(stats_list)

    logger.info(f"  Analyzed {len(inhibition_stats)} glomeruli")
    if not inhibition_stats.empty:
        logger.info(f"  Mean coverage: {inhibition_stats['coverage_pct'].mean():.1f}%")
        logger.info(f"  Total APL→PN synapses: {inhibition_stats['total_synapses'].sum()}")
    else:
        logger.warning("  No glomeruli found matching target criteria")

    return inhibition_stats


def compute_apl_statistics(
    apl_neurons: pd.DataFrame,
    connections: pd.DataFrame,
    target_pns: pd.DataFrame,
) -> Dict[str, float]:
    """Compute global APL inhibition statistics.

    Parameters
    ----------
    apl_neurons : pd.DataFrame
        APL neurons
    connections : pd.DataFrame
        FlyWire connections
    target_pns : pd.DataFrame
        Target PN neurons

    Returns
    -------
    stats : dict
        Keys: n_apl, mean_pn_targets, total_pn_synapses, mean_syn_per_pn,
        pn_coverage_pct, inhibition_strength

    Examples
    --------
    >>> stats = compute_apl_statistics(apl, connections, pns)
    >>> print(f"APL coverage: {stats['pn_coverage_pct']:.1f}%")
    """
    if apl_neurons.empty:
        return {
            "n_apl": 0,
            "mean_pn_targets": 0,
            "total_pn_synapses": 0,
            "mean_syn_per_pn": 0,
            "pn_coverage_pct": 0,
            "inhibition_strength": 0,
        }

    apl_root_ids = apl_neurons["root_id"].values
    pn_root_ids = set(target_pns["root_id"].values)

    apl_pn_connections = connections[
        (connections["pre_root_id"].isin(apl_root_ids))
        & (connections["post_root_id"].isin(pn_root_ids))
    ]

    n_apl = len(apl_neurons)
    n_pn_targets = apl_pn_connections["post_root_id"].nunique()
    total_pn_synapses = apl_pn_connections["syn_count"].sum()
    mean_syn_per_pn = total_pn_synapses / n_pn_targets if n_pn_targets > 0 else 0
    pn_coverage_pct = 100 * n_pn_targets / len(target_pns) if len(target_pns) > 0 else 0

    # Inhibition strength: normalized by PN count
    inhibition_strength = total_pn_synapses / len(target_pns) if len(target_pns) > 0 else 0

    stats = {
        "n_apl": n_apl,
        "mean_pn_targets": n_pn_targets / n_apl if n_apl > 0 else 0,
        "total_pn_synapses": total_pn_synapses,
        "mean_syn_per_pn": mean_syn_per_pn,
        "pn_coverage_pct": pn_coverage_pct,
        "inhibition_strength": inhibition_strength,
    }

    return stats
