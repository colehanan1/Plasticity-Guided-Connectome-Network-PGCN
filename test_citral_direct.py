#!/usr/bin/env python3
"""Direct test of citral PN activity generation."""

from src.pgcn.data.door_integration import DoORIntegration
import numpy as np
from pathlib import Path


def test_citral():
    """Test citral PN activity generation with detailed logging."""
    print("=" * 70)
    print("Direct Citral PN Activity Test")
    print("=" * 70)
    print()

    # Initialize DoOR integration
    cache_dir = Path("data/cache")
    door = DoORIntegration(cache_dir)

    # Test citral
    print("Testing: citral")
    print("-" * 70)

    # Check if citral is in DoOR
    print(f"1. Is 'citral' in DoOR index? {('citral' in door.door_data.index)}")

    if 'citral' in door.door_data.index:
        citral_row = door.door_data.loc['citral']
        print(f"2. DoOR row for citral: {len(citral_row)} ORN types")
        print(f"3. Non-zero responses: {(citral_row > 0).sum()}")
        print(f"4. Max response: {citral_row.max():.3f}")

        # Show top responses
        top_responses = citral_row.sort_values(ascending=False).head(10)
        print(f"\n5. Top 10 ORN responses:")
        for orn, resp in top_responses.items():
            print(f"   {orn:20s}: {resp:.4f}")

    print()

    # Generate PN activity
    print("6. Generating PN activity pattern...")
    pn_activity = door.odor_to_pn_activity('citral', n_pn=150, intensity=1.0)

    print(f"   PN activity shape: {pn_activity.shape}")
    print(f"   Active PNs (>0.1): {np.sum(pn_activity > 0.1)}")
    print(f"   Max PN activity: {pn_activity.max():.3f}")
    print(f"   Mean PN activity: {pn_activity.mean():.3f}")

    if np.sum(pn_activity > 0.1) > 0:
        print(f"\n7. Active PN indices:")
        active_pns = np.where(pn_activity > 0.1)[0]
        for pn_idx in active_pns[:10]:
            print(f"   PN {pn_idx:3d}: activity={pn_activity[pn_idx]:.3f}, glomerulus={door.pn_glomeruli.get(pn_idx, 'UNKNOWN')}")
    else:
        print(f"\n7. ❌ NO ACTIVE PNs - Debugging...")

        # Debug: Check PN glomeruli
        print(f"\n   PNs mapped to glomeruli: {len(door.pn_glomeruli)}")
        print(f"   Sample PN→glomerulus mappings:")
        for pn_idx, glom in list(door.pn_glomeruli.items())[:10]:
            from src.pgcn.data.door_integration import GLOMERULUS_TO_ORN_MAPPING
            orn = GLOMERULUS_TO_ORN_MAPPING.get(glom, None)
            print(f"     PN {pn_idx:3d} → {glom:10s} → {orn if orn else 'NO MAPPING'}")

    print()
    print("=" * 70)


if __name__ == '__main__':
    test_citral()
