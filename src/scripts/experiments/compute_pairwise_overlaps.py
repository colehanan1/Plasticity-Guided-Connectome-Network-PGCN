#!/usr/bin/env python3
"""
Compute Pairwise KC Overlaps for Odor Pair Selection

This script computes KC overlap for all pairwise combinations of 10 curated
odors to identify representative pairs spanning 0-40% overlap for the KC
overlap vs protection experiment.

Usage:
    python scripts/experiments/compute_pairwise_overlaps.py \
        --kc-sparsity 0.05 \
        --output reports/experiments/overlap/pairwise_kc_overlaps.csv

Author: PGCN Project
Date: 2025-11-19
"""

import argparse
import sys
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from door_integration.pgcn_door import PGCNDoorIntegration
from pgcn.analysis.odor_similarity import measure_kc_overlap, compute_chemical_similarity


def main():
    """Compute all pairwise KC overlaps for 10 curated odors."""
    parser = argparse.ArgumentParser(
        description="Compute pairwise KC overlaps for odor selection"
    )
    parser.add_argument(
        '--kc-sparsity',
        type=float,
        default=0.05,
        help='KC sparsity (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/experiments/overlap/pairwise_kc_overlaps.csv',
        help='Output CSV path'
    )

    args = parser.parse_args()

    # 10 curated odors (excluding citral due to missing Or83c mapping)
    odors = [
        '1-hexanol',
        'ethyl butyrate',
        'linalool',
        '3-octanol',
        'benzaldehyde',
        '2-heptanone',
        '1-octanol',
        'ethyl acetate',
        'geranyl acetate',
        'methyl salicylate',
    ]

    print("Loading circuit and DoOR database...")
    loader = CircuitLoader(cache_dir='data/cache')
    connectivity = loader.load_connectivity_matrix(
        normalize_weights='row',
        include_dan=False,
        include_extended=False,
    )

    circuit = OlfactoryCircuit(connectivity=connectivity, kc_sparsity_target=args.kc_sparsity)
    door = PGCNDoorIntegration()

    print(f"✅ Circuit: {connectivity.n_pn} PNs, {connectivity.n_kc} KCs, {connectivity.n_mbon} MBONs")
    print(f"   KC sparsity: {args.kc_sparsity:.1%}")
    print(f"\n📊 Computing pairwise overlaps for {len(odors)} odors ({len(list(combinations(odors, 2)))} pairs)...\n")

    results = []

    for i, (odor_a, odor_b) in enumerate(combinations(odors, 2), 1):
        print(f"[{i:2d}/{len(list(combinations(odors, 2)))}] {odor_a:20s} vs {odor_b:20s}", end=" ")

        try:
            # Measure KC overlap
            overlap = measure_kc_overlap(circuit, odor_a, odor_b, door)

            # Compute chemical similarity
            chem_sim = compute_chemical_similarity(odor_a, odor_b, door)

            result = {
                'odor_a': odor_a,
                'odor_b': odor_b,
                'kc_overlap_pct': overlap['kc_overlap_pct'],
                'kc_overlap_count': overlap['kc_overlap_count'],
                'jaccard_index': overlap['jaccard_index'],
                'chemical_similarity': chem_sim['cosine_similarity'],
                'chemical_correlation': chem_sim['correlation'],
                'n_kc_a': overlap['active_kcs_A'],
                'n_kc_b': overlap['active_kcs_B'],
                'dice_coefficient': overlap['dice_coefficient'],
            }

            results.append(result)

            print(f"→ KC overlap: {overlap['kc_overlap_pct']:5.1f}%, chem_sim: {chem_sim['cosine_similarity']:.3f}")

        except Exception as e:
            print(f"→ ERROR: {e}")
            continue

    # Convert to DataFrame and sort by KC overlap
    df = pd.DataFrame(results)
    df = df.sort_values('kc_overlap_pct', ascending=True)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Results saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total pairs analyzed: {len(df)}")
    print(f"KC overlap range: {df['kc_overlap_pct'].min():.1f}% - {df['kc_overlap_pct'].max():.1f}%")
    print(f"Chemical similarity range: {df['chemical_similarity'].min():.3f} - {df['chemical_similarity'].max():.3f}")

    # Categorize by overlap bins
    bins = [
        (0, 5, "Very Low (0-5%)"),
        (5, 15, "Low (5-15%)"),
        (15, 25, "Medium (15-25%)"),
        (25, 40, "High (25-40%)"),
        (40, 100, "Very High (>40%)"),
    ]

    print("\n" + "="*80)
    print("OVERLAP DISTRIBUTION")
    print("="*80)

    for min_val, max_val, label in bins:
        mask = (df['kc_overlap_pct'] >= min_val) & (df['kc_overlap_pct'] < max_val)
        count = mask.sum()
        if count > 0:
            print(f"\n{label}: {count} pairs")
            subset = df[mask].head(3)
            for _, row in subset.iterrows():
                print(f"  • {row['odor_a']:20s} vs {row['odor_b']:20s} → {row['kc_overlap_pct']:5.1f}% overlap")

    # Suggest representative pairs for experiment
    print("\n" + "="*80)
    print("SUGGESTED ODOR PAIRS FOR EXPERIMENT")
    print("="*80)

    # Select representative pairs from each bin
    selected_pairs = []

    for min_val, max_val, label in bins[:4]:  # Exclude very high
        mask = (df['kc_overlap_pct'] >= min_val) & (df['kc_overlap_pct'] < max_val)
        if mask.sum() > 0:
            # Select 2 pairs from each bin (one low chem sim, one high chem sim)
            subset = df[mask].sort_values('chemical_similarity')

            if len(subset) >= 2:
                # Low chem sim
                selected_pairs.append(subset.iloc[0])
                # High chem sim
                selected_pairs.append(subset.iloc[-1])
            elif len(subset) == 1:
                selected_pairs.append(subset.iloc[0])

    print(f"\nSelected {len(selected_pairs)} representative pairs:\n")
    for i, row in enumerate(selected_pairs, 1):
        print(f"{i}. {row['odor_a']:20s} vs {row['odor_b']:20s}")
        print(f"   KC overlap: {row['kc_overlap_pct']:5.1f}%, Chemical similarity: {row['chemical_similarity']:.3f}")
        print()


if __name__ == '__main__':
    main()
