#!/usr/bin/env python3
"""
Test ORN→PN mapping to debug zero PN activity issue.

This script specifically checks if:
1. DoOR has the expected ORN column names
2. PN→glomerulus mapping is correct
3. ORN responses are being mapped to PNs correctly
"""

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

print("="*70)
print("ORN→PN Mapping Diagnostic")
print("="*70)

sys.path.insert(0, 'src')

# Import
try:
    from pgcn.data.door_integration import DoORIntegration, GLOMERULUS_TO_ORN_MAPPING
    import numpy as np
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Initialize
cache_dir = Path("data/cache")
if not cache_dir.exists():
    # Try alternative locations
    alt_caches = [
        Path("data/door_cache"),
        Path.home() / "Documents/cole/VSCode/Plasticity-Guided-Connectome-Network-PGCN-/data/cache",
    ]
    for alt in alt_caches:
        if alt.exists():
            cache_dir = alt
            break

print(f"\nUsing cache directory: {cache_dir}")

try:
    door = DoORIntegration(cache_dir=cache_dir)
except Exception as e:
    print(f"✗ Failed to initialize DoORIntegration: {e}")
    sys.exit(1)

print(f"\n[1/4] DoOR Data Structure")
print(f"  Odorants (rows): {len(door.door_data)}")
print(f"  ORN types (columns): {len(door.door_data.columns)}")
print(f"\n  DoOR column names (first 20 ORN types):")
for i, col in enumerate(door.door_data.columns[:20]):
    print(f"    {i+1:2d}. {col}")

print(f"\n[2/4] PN→Glomerulus Mapping")
print(f"  Total PNs mapped: {len(door.pn_glomeruli)}")
if len(door.pn_glomeruli) == 0:
    print("  ✗ NO PNs MAPPED TO GLOMERULI!")
    print("  This is the problem - nodes.parquet may be missing or malformed")
    sys.exit(1)
else:
    print(f"  Glomeruli represented: {sorted(set(door.pn_glomeruli.values()))[:10]}...")

print(f"\n[3/4] Glomerulus→ORN Mapping")
print(f"  Total mappings in GLOMERULUS_TO_ORN_MAPPING: {len(GLOMERULUS_TO_ORN_MAPPING)}")
print(f"  Sample mappings:")
for i, (glom, orn) in enumerate(list(GLOMERULUS_TO_ORN_MAPPING.items())[:10]):
    in_door = "✓" if orn in door.door_data.columns else "✗"
    print(f"    {glom} → {orn} {in_door}")

# Check how many ORN types in GLOMERULUS_TO_ORN_MAPPING are actually in DoOR columns
orn_types_in_mapping = set(GLOMERULUS_TO_ORN_MAPPING.values())
orn_types_in_door = set(door.door_data.columns)
matched_orns = orn_types_in_mapping & orn_types_in_door
print(f"\n  ORN types in mapping: {len(orn_types_in_mapping)}")
print(f"  ORN types in DoOR data: {len(orn_types_in_door)}")
print(f"  Matched: {len(matched_orns)}")

if len(matched_orns) == 0:
    print("  ✗ NO ORN TYPES MATCH!")
    print("  This means DoOR column names don't match expected ORN names")
    print("\n  Expected ORN names:", sorted(list(orn_types_in_mapping))[:10])
    print("  Actual DoOR columns:", sorted(list(orn_types_in_door))[:10])
    print("\n  Possible issue: DoOR data format mismatch")

print(f"\n[4/4] Testing Odor→PN Pathway")
test_odor = 'benzaldehyde'  # Should be exact match
print(f"  Testing with odor: '{test_odor}'")

# Step 1: Resolve name
resolved = door._resolve_odor_name(test_odor)
print(f"  1. Name resolution: '{test_odor}' → '{resolved}'")

if resolved is None:
    print("     ✗ Name not resolved!")
else:
    # Step 2: Get ORN responses
    try:
        orn_responses = door.door_data.loc[resolved]
        n_responding_orns = sum(orn_responses > 0.1)
        print(f"  2. ORN responses: {n_responding_orns} ORNs respond > 0.1")

        # Show top responding ORNs
        top_orns = orn_responses.nlargest(5)
        print(f"     Top 5 responding ORNs:")
        for orn, resp in top_orns.items():
            print(f"       {orn}: {resp:.3f}")

    except KeyError:
        print(f"     ✗ KeyError for '{resolved}'")

    # Step 3: Map to PNs
    pn_activity = door.odor_to_pn_activity(test_odor, n_pn=150)
    n_active_pns = sum(pn_activity > 0.1)
    print(f"  3. PN activity: {n_active_pns} PNs active > 0.1")

    if n_active_pns == 0:
        print("     ✗ ZERO PNs ACTIVE!")
        print("\n     Debugging ORN→PN mapping...")

        # Check how many PNs have glomeruli that map to ORNs in DoOR
        pns_with_valid_mapping = 0
        for pn_idx, glomerulus in door.pn_glomeruli.items():
            orn_type = GLOMERULUS_TO_ORN_MAPPING.get(glomerulus)
            if orn_type and orn_type in door.door_data.columns:
                pns_with_valid_mapping += 1

        print(f"     PNs with valid ORN mapping: {pns_with_valid_mapping}/{len(door.pn_glomeruli)}")

        if pns_with_valid_mapping == 0:
            print("     → Issue: No PNs have glomeruli that map to ORNs in DoOR")
            print("     → Likely cause: DoOR column names don't match expected ORN names")
        else:
            print("     → Some PNs have valid mappings but still zero activity")
            print("     → Check if odor actually activates those specific ORNs")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("\nIf you see '0 ORN types matched', the issue is:")
print("  DoOR data column names don't match GLOMERULUS_TO_ORN_MAPPING")
print("  Solution: Update GLOMERULUS_TO_ORN_MAPPING to use actual DoOR column names")
print("\nIf you see '0 PNs mapped to glomeruli', the issue is:")
print("  nodes.parquet is missing or doesn't contain PN glomerulus assignments")
print("  Solution: Check that nodes.parquet exists and has 'glomerulus' column")
