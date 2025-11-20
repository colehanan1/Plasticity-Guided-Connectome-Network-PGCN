#!/usr/bin/env python3
"""Diagnose why citral returns 0 active PNs.

This script investigates the complete pathway:
1. DoOR data: Which ORNs respond to citral?
2. FlyWire data: Which glomeruli do our PNs map to?
3. Mapping: Which ORN types do our glomeruli map to?
4. Overlap: Where is the mismatch?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.pgcn.data.door_integration import DoORIntegration, GLOMERULUS_TO_ORN_MAPPING


def diagnose_citral():
    """Diagnose citral zero PN activity issue."""
    print("=" * 70)
    print("Citral Zero PN Activity Diagnostic")
    print("=" * 70)
    print()

    # Initialize DoOR integration
    cache_dir = Path("data/cache")
    door = DoORIntegration(cache_dir)

    # Step 1: Check if citral is in DoOR
    print("[1/5] Checking if citral is in DoOR database...")
    if 'citral' in door.door_data.index:
        print("  ✓ citral found in DoOR index")
    else:
        print("  ✗ citral NOT found in DoOR index")
        print(f"  Available odors (sample): {list(door.door_data.index[:10])}")
        return
    print()

    # Step 2: Get citral ORN responses
    print("[2/5] Getting ORN responses for citral...")
    citral_responses = door.door_data.loc['citral']
    active_orns = citral_responses[citral_responses > 0.1]  # Threshold at 0.1
    print(f"  Total ORN types in DoOR: {len(citral_responses)}")
    print(f"  ORN types responding to citral (>0.1): {len(active_orns)}")
    print()

    if len(active_orns) > 0:
        print("  Top 10 responding ORN types:")
        for orn, response in active_orns.sort_values(ascending=False).head(10).items():
            print(f"    {orn:15s}: {response:.3f}")
    else:
        print("  ⚠️  NO ORN types respond to citral in DoOR!")
        print("     This is the root cause - citral has no response data!")
        return
    print()

    # Step 3: Check PN→glomerulus mapping
    print("[3/5] Checking PN→glomerulus mapping from FlyWire...")
    pn_glomeruli = door.pn_glomeruli
    print(f"  Total PNs mapped: {len(pn_glomeruli)}")

    glom_counts = {}
    for glom in pn_glomeruli.values():
        glom_counts[glom] = glom_counts.get(glom, 0) + 1

    print(f"  Unique glomeruli: {len(glom_counts)}")
    print(f"  Sample glomeruli: {list(glom_counts.keys())[:20]}")
    print()

    # Step 4: Check glomerulus→ORN mapping coverage
    print("[4/5] Checking glomerulus→ORN mapping coverage...")
    print(f"  Glomeruli in GLOMERULUS_TO_ORN_MAPPING: {len(GLOMERULUS_TO_ORN_MAPPING)}")
    print()

    # Find which ORNs we can map to
    mapped_orns = set()
    for glom in glom_counts.keys():
        orn = GLOMERULUS_TO_ORN_MAPPING.get(glom)
        if orn:
            mapped_orns.add(orn)

    print(f"  ORN types we can map to (from our PNs): {len(mapped_orns)}")
    print(f"  Sample ORN types: {list(mapped_orns)[:20]}")
    print()

    # Step 5: Find the overlap
    print("[5/5] Finding overlap between citral-responsive ORNs and mappable ORNs...")
    print()

    active_orn_set = set(active_orns.index)
    overlap = active_orn_set & mapped_orns

    print(f"  Citral-responsive ORNs (DoOR): {len(active_orn_set)}")
    print(f"  Mappable ORNs (from PNs):      {len(mapped_orns)}")
    print(f"  Overlap (should activate PNs): {len(overlap)}")
    print()

    if len(overlap) == 0:
        print("  ❌ NO OVERLAP - This is the problem!")
        print()
        print("  Citral activates these ORNs in DoOR:")
        for orn in sorted(active_orn_set):
            response = citral_responses[orn]
            in_mapping = "✓" if orn in GLOMERULUS_TO_ORN_MAPPING.values() else "✗"
            print(f"    {in_mapping} {orn:15s}: {response:.3f}")
        print()

        print("  But we can only map to these ORNs (from FlyWire PNs):")
        for orn in sorted(mapped_orns)[:30]:
            in_citral = "✓" if orn in active_orn_set else "✗"
            print(f"    {in_citral} {orn:15s}")

        print()
        print("  DIAGNOSIS:")
        print("  ----------")
        print("  The FlyWire PNs in your data map to glomeruli that don't")
        print("  correspond to the ORN types that respond to citral in DoOR.")
        print()
        print("  SOLUTION OPTIONS:")
        print("  1. Check if glomerulus names in FlyWire data are correct")
        print("  2. Verify GLOMERULUS_TO_ORN_MAPPING has correct mappings")
        print("  3. Check if DoOR uses different ORN naming conventions")
        print("  4. Examine the PN→glomerulus assignments in nodes.parquet")

    else:
        print("  ✓ Found overlap! These ORNs should activate PNs:")
        for orn in sorted(overlap):
            response = citral_responses[orn]
            # Find glomeruli for this ORN
            gloms = [g for g, o in GLOMERULUS_TO_ORN_MAPPING.items() if o == orn]
            # Count PNs
            pn_count = sum(1 for g in pn_glomeruli.values() if g in gloms)
            print(f"    {orn:15s}: response={response:.3f}, glomeruli={gloms}, PNs={pn_count}")

        print()
        print("  Expected active PNs:", sum(
            sum(1 for g in pn_glomeruli.values() if g in [gl for gl, o in GLOMERULUS_TO_ORN_MAPPING.items() if o == orn])
            for orn in overlap
        ))

    print()
    print("=" * 70)

    # Additional diagnostic: Check DoOR column names
    print()
    print("BONUS: DoOR column analysis")
    print("-" * 70)
    print(f"DoOR columns (ORN types): {len(door.door_data.columns)}")
    print()
    print("Sample DoOR columns (ORN types):")
    for col in list(door.door_data.columns[:30]):
        print(f"  {col}")
    print()

    # Check what ORN types are in GLOMERULUS_TO_ORN_MAPPING
    print("ORN types in GLOMERULUS_TO_ORN_MAPPING:")
    unique_orns = sorted(set(GLOMERULUS_TO_ORN_MAPPING.values()))
    for orn in unique_orns[:30]:
        in_door = "✓" if orn in door.door_data.columns else "✗"
        print(f"  {in_door} {orn}")


if __name__ == '__main__':
    diagnose_citral()
