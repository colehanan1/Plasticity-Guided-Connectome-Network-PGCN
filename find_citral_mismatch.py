#!/usr/bin/env python3
"""Find the exact mismatch between DoOR citral data and PN glomerulus mapping.

This script identifies why citral returns 0 PNs by comparing:
- What ORN types DoOR says respond to citral
- What ORN types our glomerulus mapping expects
- What glomeruli our PNs actually map to
"""

import pandas as pd
from pathlib import Path
from src.pgcn.data.door_integration import DoORIntegration, GLOMERULUS_TO_ORN_MAPPING


def find_mismatch():
    """Find the exact mismatch."""
    print("=" * 80)
    print("Citral Mismatch Analysis")
    print("=" * 80)
    print()

    # Load DoOR data
    cache_dir = Path("data/cache")
    door = DoORIntegration(cache_dir)

    # Get citral responses
    if 'citral' not in door.door_data.index:
        print("❌ ERROR: citral not in DoOR index!")
        return

    citral_responses = door.door_data.loc['citral']
    active_orns = citral_responses[citral_responses > 0.05].sort_values(ascending=False)

    print(f"PART 1: What DoOR says about citral")
    print("-" * 80)
    print(f"Total ORN types in DoOR: {len(citral_responses)}")
    print(f"ORN types responding to citral (>0.05): {len(active_orns)}")
    print()
    print("Top 20 citral-responsive ORN types from DoOR:")
    for i, (orn, response) in enumerate(active_orns.head(20).items(), 1):
        print(f"  {i:2d}. {orn:20s}: {response:.4f}")
    print()

    print(f"PART 2: What glomerulus mapping expects")
    print("-" * 80)
    print(f"Glomeruli in GLOMERULUS_TO_ORN_MAPPING: {len(GLOMERULUS_TO_ORN_MAPPING)}")
    print()

    # Get unique ORN types from mapping
    mapped_orn_types = sorted(set(GLOMERULUS_TO_ORN_MAPPING.values()))
    print(f"Unique ORN types in mapping: {len(mapped_orn_types)}")
    print()
    print("All ORN types we can map to:")
    for i, orn in enumerate(mapped_orn_types, 1):
        in_door = "✓" if orn in door.door_data.columns else "✗"
        responds = "✓" if orn in active_orns.index else " "
        print(f"  {i:2d}. {in_door} {responds} {orn:20s}")
    print()
    print("Legend: First ✓/✗ = in DoOR columns, Second ✓ = responds to citral")
    print()

    print(f"PART 3: What PNs actually map to")
    print("-" * 80)
    pn_glomeruli = door.pn_glomeruli
    print(f"Total PNs: {len(pn_glomeruli)}")

    # Count glomeruli
    glom_counts = {}
    for glom in pn_glomeruli.values():
        glom_counts[glom] = glom_counts.get(glom, 0) + 1

    print(f"Unique glomeruli: {len(glom_counts)}")
    print()
    print("Glomeruli with PN assignments:")
    for glom, count in sorted(glom_counts.items(), key=lambda x: x[1], reverse=True)[:30]:
        orn = GLOMERULUS_TO_ORN_MAPPING.get(glom, None)
        has_mapping = "✓" if orn else "✗"
        in_door = "✓" if orn and orn in door.door_data.columns else "✗"
        responds = "✓" if orn and orn in active_orns.index else " "

        status = "MATCH!" if orn and orn in active_orns.index else ""
        print(f"  {glom:15s}: {count:3d} PNs | Map:{has_mapping} Door:{in_door} Citral:{responds} | {orn if orn else 'NO MAPPING':20s} {status}")
    print()

    print(f"PART 4: THE CRITICAL OVERLAP")
    print("-" * 80)

    # Find overlap
    citral_orn_set = set(active_orns.index)
    available_orn_set = set(mapped_orn_types)
    pn_orn_set = set()

    for glom in glom_counts.keys():
        orn = GLOMERULUS_TO_ORN_MAPPING.get(glom)
        if orn and orn in door.door_data.columns:
            pn_orn_set.add(orn)

    overlap = citral_orn_set & pn_orn_set

    print(f"ORN types that respond to citral (DoOR):     {len(citral_orn_set)}")
    print(f"ORN types in glomerulus mapping:             {len(available_orn_set)}")
    print(f"ORN types from PNs (glomeruli→ORN→DoOR):     {len(pn_orn_set)}")
    print(f"OVERLAP (citral-responsive AND mapped PNs):  {len(overlap)}")
    print()

    if len(overlap) == 0:
        print("❌ ZERO OVERLAP - This is why citral returns 0 active PNs!")
        print()
        print("Missing links analysis:")
        print()

        print("1. Citral-responsive ORNs NOT in our PN mapping:")
        missing = citral_orn_set - pn_orn_set
        for orn in sorted(missing)[:15]:
            response = citral_responses[orn]
            # Check if it's in the mapping at all
            in_mapping = orn in available_orn_set
            has_pn = any(GLOMERULUS_TO_ORN_MAPPING.get(g) == orn for g in glom_counts.keys())
            print(f"   {orn:20s}: response={response:.4f}, in_mapping={in_mapping}, has_PN={has_pn}")

        print()
        print("2. Checking if DoOR uses different ORN naming...")
        print("   DoOR column names (sample):")
        for col in list(door.door_data.columns[:20]):
            print(f"     {col}")

        print()
        print("3. Recommended fixes:")
        print("   a. Check if DoOR column names match expected ORN names (Or22a vs or22a)")
        print("   b. Verify PN→glomerulus mapping in nodes.parquet is correct")
        print("   c. Add missing glomeruli to GLOMERULUS_TO_ORN_MAPPING")
        print("   d. Check if DoOR uses different receptor naming conventions")

    else:
        print("✓ Found overlap! These ORNs should work:")
        for orn in sorted(overlap):
            response = citral_responses[orn]
            gloms = [g for g, o in GLOMERULUS_TO_ORN_MAPPING.items() if o == orn]
            pn_count = sum(1 for g in pn_glomeruli.values() if g in gloms)
            print(f"  {orn:20s}: response={response:.4f}, glomeruli={gloms}, PNs={pn_count}")

    print()
    print("=" * 80)


if __name__ == '__main__':
    find_mismatch()
