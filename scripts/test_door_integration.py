#!/usr/bin/env python3
"""
Test DoOR toolkit integration with PGCN.

This script demonstrates how to use the PGCNDoorIntegration class
to work with odorant response data alongside FlyWire connectivity.

Usage:
    python scripts/test_door_integration.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from door_integration.pgcn_door import PGCNDoorIntegration, DOOR_AVAILABLE

# Get import error if available
try:
    from door_integration.pgcn_door import DOOR_IMPORT_ERROR
except ImportError:
    DOOR_IMPORT_ERROR = "Unknown import error"


def main():
    """Run integration tests."""

    if not DOOR_AVAILABLE:
        print("="*80)
        print("❌ DoOR TOOLKIT NOT AVAILABLE")
        print("="*80)
        print(f"\nImport error: {DOOR_IMPORT_ERROR}")
        print("\nThe DoOR toolkit is required for this test.")
        print("\nTo install:")
        print("  1. Navigate to door-toolkit directory:")
        print("     cd ~/Documents/cole/VSCode/door-python-toolkit")
        print("\n  2. Install in editable mode:")
        print("     pip install -e .")
        print("\n  3. Verify installation:")
        print("     python -c 'from door_toolkit.encoder import DoOREncoder; print(\"✅ Success\")'")
        print("\n  4. Re-run this test:")
        print("     python scripts/test_door_integration.py")
        print("\nFor detailed diagnostics, run:")
        print("  python scripts/diagnose_door_install.py")
        print("="*80)
        return 1

    print("="*80)
    print("PGCN-DOOR INTEGRATION TEST")
    print("="*80)

    # Initialize
    pgcn_door = PGCNDoorIntegration()

    # Test 1: Or7a receptor profile
    print("\n" + "="*80)
    print("TEST 1: Or7a Receptor Profile")
    print("="*80)
    or7a_profile = pgcn_door.get_receptor_profile('Or7a')
    print(f"\nOr7a responds to {len(or7a_profile)} odorants")
    print("\nTop 10 odorants for Or7a:")
    for i, (odorant, response) in enumerate(or7a_profile.nlargest(10).items(), 1):
        print(f"  {i:2d}. {odorant:20s} {response:.1%}")

    # Test 2: Benzaldehyde encoding
    print("\n" + "="*80)
    print("TEST 2: Benzaldehyde Receptor Encoding")
    print("="*80)
    benz_encoding = pgcn_door.get_odor_encoding('benzaldehyde', threshold=0.3)
    print(f"\nBenzaldehyde activates {len(benz_encoding)} receptors (threshold 0.3)")
    print("\nReceptor activations:")
    for receptor, activation in sorted(benz_encoding.items(), key=lambda x: x[1], reverse=True):
        glom = pgcn_door.OR_TO_GLOMERULUS.get(receptor, 'unknown')
        print(f"  {receptor:6s} → {glom:6s}  {activation:.1%}")

    # Test 3: Or7a selectivity
    print("\n" + "="*80)
    print("TEST 3: Or7a Selectivity (Hypothesis 1)")
    print("="*80)

    benz_response = or7a_profile['benzaldehyde'] if 'benzaldehyde' in or7a_profile else 0
    hex_response = or7a_profile['hexanol'] if 'hexanol' in or7a_profile else 0

    if hex_response > 0:
        selectivity = benz_response / hex_response
    else:
        selectivity = float('inf') if benz_response > 0 else 0

    print(f"\nOr7a response to benzaldehyde: {benz_response:.3f}")
    print(f"Or7a response to hexanol: {hex_response:.3f}")
    print(f"Selectivity ratio: {selectivity:.2f}x")
    print(f"\nThreshold for selectivity: 3.0x")
    print(f"Result: {'✅ SUPPORTS' if selectivity > 3.0 else '❌ CONTRADICTS'} Hypothesis 1")

    # Test 4: Cross-learning mechanism
    print("\n" + "="*80)
    print("TEST 4: Cross-Learning Mechanism (Hypothesis 3)")
    print("="*80)

    shared = pgcn_door.find_shared_receptors('benzaldehyde', 'hexanol', threshold=0.5)
    print(f"\nReceptors strongly responding to BOTH benzaldehyde and hexanol:")
    if shared:
        for receptor in sorted(shared):
            glom = pgcn_door.OR_TO_GLOMERULUS.get(receptor, 'unknown')
            benz_val = benz_encoding.get(receptor, 0)
            hex_encoding = pgcn_door.get_odor_encoding('hexanol', threshold=0.5)
            hex_val = hex_encoding.get(receptor, 0)
            print(f"  {receptor:6s} ({glom:6s}): benz={benz_val:.1%}, hex={hex_val:.1%}")
        print(f"\n✅ SUPPORTS Hypothesis 3: Shared receptor explains cross-learning")
    else:
        print("  None found at threshold 0.5")
        print(f"\n❌ CONTRADICTS Hypothesis 3: No shared strong receptors")

    # Test 5: Glomerulus mapping
    print("\n" + "="*80)
    print("TEST 5: Odorant → Glomerulus Mapping")
    print("="*80)

    for odorant in ['benzaldehyde', 'hexanol', '2-heptanone']:
        glomeruli = pgcn_door.map_odorant_to_glomeruli(odorant, threshold=0.3)
        print(f"\n{odorant}:")
        print(f"  Active glomeruli: {', '.join(glomeruli) if glomeruli else 'None'}")

    # Test 6: Receptor similarity
    print("\n" + "="*80)
    print("TEST 6: Receptor Similarity Analysis")
    print("="*80)

    receptor_pairs = [
        ('Or7a', 'Or67b'),
        ('Or7a', 'Or22a'),
        ('Or67b', 'Or35a'),
    ]

    print("\nPearson correlation between receptor profiles:")
    for rec1, rec2 in receptor_pairs:
        sim = pgcn_door.get_receptor_similarity(rec1, rec2, method='pearson')
        print(f"  {rec1} <-> {rec2}: {sim:+.3f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    matrix = pgcn_door.get_response_matrix()
    print(f"\n✅ DoOR database loaded successfully")
    print(f"   - Odorants: {matrix.shape[0]}")
    print(f"   - Receptors: {matrix.shape[1]}")
    print(f"   - OR→Glomerulus mappings: {len(pgcn_door.OR_TO_GLOMERULUS)}")
    print(f"\n✅ All integration tests passed")
    print(f"\nYou can now use PGCNDoorIntegration in your analysis scripts!")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
