#!/usr/bin/env python3
"""Debug script to check which Or47b PNs are being filtered out."""
import pandas as pd
from pathlib import Path

# All 15 expected PNs
expected_pns = [
    720575940629733626, 720575940623739076, 720575940620199962, 720575940621696747,
    720575940628283560, 720575940630989354, 720575940625014928, 720575940633165025,
    720575940629097922, 720575940632720026, 720575940631742156, 720575940634229615,
    720575940621103743, 720575940615189465, 720575940638288447
]

# Load data
data_dir = Path("data/flywire")
conns = pd.read_csv(data_dir / "connections_princeton.csv.gz")
cell_types = pd.read_csv(data_dir / "consolidated_cell_types.csv.gz")
orn_df = pd.read_csv(data_dir / "search_results_Or47b.csv")

# Get outputs from Or47b ORNs
outputs = conns[conns['pre_root_id'].isin(orn_df['root_id'])]
outputs = outputs[outputs['syn_count'] >= 3]

# Merge with cell types
outputs = outputs.merge(
    cell_types,
    left_on='post_root_id',
    right_on='root_id',
    how='left'
)

print("="*70)
print("Checking which Or47b PNs are in the data and how they're categorized")
print("="*70)

# Check each expected PN
for pn_id in expected_pns:
    # Check if it exists in connections
    in_conns = pn_id in outputs['post_root_id'].values

    if not in_conns:
        print(f"\n✗ {pn_id}: NOT FOUND in connections")
        continue

    # Get cell type info
    pn_info = outputs[outputs['post_root_id'] == pn_id].iloc[0]
    cell_type = pn_info['primary_type']

    # Check if it would be identified as PN by pattern matching
    pn_patterns = ["adpn", "lpn", "_pn"]
    is_pn_by_pattern = any(pat in str(cell_type).lower() for pat in pn_patterns)

    # Check if it matches VA1v pattern
    matches_va1v = "va1v" in str(cell_type).lower()

    status = "✓" if is_pn_by_pattern else "✗"
    print(f"\n{status} {pn_id}:")
    print(f"   Cell type: {cell_type}")
    print(f"   Matches PN pattern: {is_pn_by_pattern}")
    print(f"   Matches VA1v pattern: {matches_va1v}")

    # Get synapse count
    syn_count = outputs[outputs['post_root_id'] == pn_id]['syn_count'].sum()
    print(f"   Total synapses: {syn_count}")

print("\n" + "="*70)
print("Summary")
print("="*70)

found_pns = []
missing_pns = []

for pn_id in expected_pns:
    in_conns = pn_id in outputs['post_root_id'].values
    if not in_conns:
        missing_pns.append(pn_id)
        continue

    pn_info = outputs[outputs['post_root_id'] == pn_id].iloc[0]
    cell_type = str(pn_info['primary_type']).lower()
    pn_patterns = ["adpn", "lpn", "_pn"]
    is_pn_by_pattern = any(pat in cell_type for pat in pn_patterns)

    if is_pn_by_pattern:
        found_pns.append((pn_id, pn_info['primary_type']))
    else:
        missing_pns.append((pn_id, pn_info['primary_type']))

print(f"\nFound {len(found_pns)} PNs that pass pattern matching:")
for pn_id, cell_type in found_pns:
    print(f"  {pn_id}: {cell_type}")

print(f"\nMissing {len(missing_pns)} PNs (don't pass pattern matching):")
for item in missing_pns:
    if isinstance(item, tuple):
        pn_id, cell_type = item
        print(f"  {pn_id}: {cell_type}")
    else:
        print(f"  {item}: NOT IN CONNECTIONS")
