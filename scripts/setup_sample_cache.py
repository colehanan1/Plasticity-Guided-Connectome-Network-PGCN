#!/usr/bin/env python3
"""
Generate minimal sample cache for testing realistic training protocol.

This creates a small, synthetic connectome cache that allows testing
the realistic_behavioral_training.py script without requiring FlyWire data.

Author: PGCN Enhancement
Date: 2025-11-11
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path


def create_sample_cache(cache_dir: Path, n_pn: int = 150, n_kc: int = 2000,
                       n_mbon: int = 44, n_dan: int = 100):
    """
    Create minimal sample cache for testing.

    Args:
        cache_dir: Directory to save cache files
        n_pn: Number of projection neurons
        n_kc: Number of Kenyon cells
        n_mbon: Number of MBONs
        n_dan: Number of DANs
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING SAMPLE CONNECTOME CACHE")
    print("=" * 70)
    print(f"Cache directory: {cache_dir}")
    print(f"Creating synthetic connectome:")
    print(f"  PNs:   {n_pn}")
    print(f"  KCs:   {n_kc}")
    print(f"  MBONs: {n_mbon}")
    print(f"  DANs:  {n_dan}")
    print()

    # Generate neuron IDs
    pn_ids = np.arange(1000000, 1000000 + n_pn)
    kc_ids = np.arange(2000000, 2000000 + n_kc)
    mbon_ids = np.arange(3000000, 3000000 + n_mbon)
    dan_ids = np.arange(4000000, 4000000 + n_dan)

    # Define glomeruli for PNs
    glomeruli = ['DA1', 'DL1', 'DL3', 'DL5', 'DM1', 'DM2', 'DM4', 'DM5',
                'VA1d', 'VA1v', 'VA2', 'VA6', 'VC1', 'DC3', 'DA2', 'DA4m']

    # Create nodes DataFrame
    print("📊 Creating nodes...")
    nodes_data = []

    # PNs
    for i, pn_id in enumerate(pn_ids):
        glom = glomeruli[i % len(glomeruli)]
        nodes_data.append({
            'node_id': pn_id,
            'type': 'PN',
            'glomerulus': glom,
            'hemisphere': 'L' if i % 2 == 0 else 'R',
            'x': np.random.randn() * 10 + 100,
            'y': np.random.randn() * 10 + 100,
            'z': np.random.randn() * 10 + 100
        })

    # KCs
    kc_types = ['KCab', 'KCg', 'KCapbp']
    for i, kc_id in enumerate(kc_ids):
        nodes_data.append({
            'node_id': kc_id,
            'type': kc_types[i % len(kc_types)],
            'glomerulus': None,
            'hemisphere': 'L' if i % 2 == 0 else 'R',
            'x': np.random.randn() * 10 + 200,
            'y': np.random.randn() * 10 + 200,
            'z': np.random.randn() * 10 + 200
        })

    # MBONs
    for i, mbon_id in enumerate(mbon_ids):
        nodes_data.append({
            'node_id': mbon_id,
            'type': 'MBON',
            'glomerulus': None,
            'hemisphere': 'L' if i % 2 == 0 else 'R',
            'x': np.random.randn() * 10 + 300,
            'y': np.random.randn() * 10 + 300,
            'z': np.random.randn() * 10 + 300
        })

    # DANs
    for i, dan_id in enumerate(dan_ids):
        nodes_data.append({
            'node_id': dan_id,
            'type': 'DAN',
            'glomerulus': None,
            'hemisphere': 'L' if i % 2 == 0 else 'R',
            'x': np.random.randn() * 10 + 400,
            'y': np.random.randn() * 10 + 400,
            'z': np.random.randn() * 10 + 400
        })

    nodes_df = pd.DataFrame(nodes_data)
    nodes_df.to_parquet(cache_dir / 'nodes.parquet', index=False)
    print(f"  ✓ Saved {len(nodes_df)} nodes to nodes.parquet")

    # Create edges DataFrame (PN→KC and KC→MBON)
    print("\n🔗 Creating edges...")
    edges_data = []

    # PN→KC connections (sparse: each KC receives from ~6 PNs)
    n_connections_per_kc = 6
    for kc_id in kc_ids:
        # Select random PNs to connect to this KC
        source_pns = np.random.choice(pn_ids, size=n_connections_per_kc, replace=False)
        for pn_id in source_pns:
            weight = np.random.exponential(scale=10) + 1  # Synapse count
            edges_data.append({
                'source': pn_id,
                'target': kc_id,
                'weight': weight,
                'edge_type': 'PN_KC'
            })

    # KC→MBON connections (dense: each MBON receives from many KCs)
    n_connections_per_mbon = int(n_kc * 0.3)  # 30% of KCs connect to each MBON
    for mbon_id in mbon_ids:
        source_kcs = np.random.choice(kc_ids, size=n_connections_per_mbon, replace=False)
        for kc_id in source_kcs:
            weight = np.random.exponential(scale=5) + 1
            edges_data.append({
                'source': kc_id,
                'target': mbon_id,
                'weight': weight,
                'edge_type': 'KC_MBON'
            })

    edges_df = pd.DataFrame(edges_data)
    edges_df.to_parquet(cache_dir / 'edges.parquet', index=False)
    print(f"  ✓ Saved {len(edges_df)} edges to edges.parquet")
    print(f"    - PN→KC: {(edges_df['edge_type'] == 'PN_KC').sum()}")
    print(f"    - KC→MBON: {(edges_df['edge_type'] == 'KC_MBON').sum()}")

    # Create DAN edges DataFrame
    print("\n💊 Creating DAN edges...")
    dan_edges_data = []

    # DAN→KC connections (modulatory)
    for dan_id in dan_ids:
        # Each DAN projects to a subset of KCs
        target_kcs = np.random.choice(kc_ids, size=int(n_kc * 0.1), replace=False)
        for kc_id in target_kcs:
            weight = np.random.exponential(scale=3) + 1
            dan_edges_data.append({
                'source': dan_id,
                'target': kc_id,
                'weight': weight,
                'edge_type': 'DAN_KC'
            })

    # DAN→MBON connections
    for dan_id in dan_ids:
        target_mbons = np.random.choice(mbon_ids, size=int(n_mbon * 0.3), replace=False)
        for mbon_id in target_mbons:
            weight = np.random.exponential(scale=3) + 1
            dan_edges_data.append({
                'source': dan_id,
                'target': mbon_id,
                'weight': weight,
                'edge_type': 'DAN_MBON'
            })

    dan_edges_df = pd.DataFrame(dan_edges_data)
    dan_edges_df.to_parquet(cache_dir / 'dan_edges.parquet', index=False)
    print(f"  ✓ Saved {len(dan_edges_df)} DAN edges to dan_edges.parquet")
    print(f"    - DAN→KC: {(dan_edges_df['edge_type'] == 'DAN_KC').sum()}")
    print(f"    - DAN→MBON: {(dan_edges_df['edge_type'] == 'DAN_MBON').sum()}")

    # Create metadata
    print("\n📝 Creating metadata...")
    meta = {
        'datastack': 'sample_synthetic',
        'materialization_version': 0,
        'created_by': 'setup_sample_cache.py',
        'counts': {
            'n_pn': int(n_pn),
            'n_kc': int(n_kc),
            'n_mbon': int(n_mbon),
            'n_dan': int(n_dan),
            'pn_kc_edges': int((edges_df['edge_type'] == 'PN_KC').sum()),
            'kc_mbon_edges': int((edges_df['edge_type'] == 'KC_MBON').sum()),
            'dan_kc_edges': int((dan_edges_df['edge_type'] == 'DAN_KC').sum()),
            'dan_mbon_edges': int((dan_edges_df['edge_type'] == 'DAN_MBON').sum())
        }
    }

    import json
    with open(cache_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Saved metadata to meta.json")

    print("\n" + "=" * 70)
    print("✅ SAMPLE CACHE GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nCache files created in: {cache_dir}")
    print("  • nodes.parquet")
    print("  • edges.parquet")
    print("  • dan_edges.parquet")
    print("  • meta.json")
    print()
    print("You can now run:")
    print("  python scripts/realistic_behavioral_training.py")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate sample connectome cache')
    parser.add_argument('--cache-dir', type=Path, default=Path('data/cache'),
                       help='Cache directory (default: data/cache)')
    parser.add_argument('--n-pn', type=int, default=150, help='Number of PNs')
    parser.add_argument('--n-kc', type=int, default=2000, help='Number of KCs')
    parser.add_argument('--n-mbon', type=int, default=44, help='Number of MBONs')
    parser.add_argument('--n-dan', type=int, default=100, help='Number of DANs')

    args = parser.parse_args()

    create_sample_cache(
        cache_dir=args.cache_dir,
        n_pn=args.n_pn,
        n_kc=args.n_kc,
        n_mbon=args.n_mbon,
        n_dan=args.n_dan
    )
