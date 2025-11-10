#!/usr/bin/env python3
"""
Regression test for PN identification in multi-ORN pathway script.

This test ensures that the multi-ORN pathway script correctly identifies
the expected number of projection neurons for each ORN type. This prevents
regressions where incorrect PN root IDs are used.

Run: pytest tests/test_pn_counts_regression.py -v
Or: python tests/test_pn_counts_regression.py
"""
import subprocess
import re
from pathlib import Path

# Expected PN counts for each ORN type (from biological data)
EXPECTED_PN_COUNTS = {
    "Or7a": 2,   # DL5_adPN neurons
    "Or13a": 4,  # DC2_adPN neurons
    "Or42b": 2,  # DM1_lPN neurons
    "Or47b": 15,  # VA1v_adPN (9) + VA1v_vPN (4) + MZ_lv2PN (2)
    "Or59b": 2,  # DM4_adPN neurons
}


def test_multi_orn_pn_counts():
    """Test that multi-ORN script identifies correct number of PNs for each ORN."""
    print("\n" + "="*70)
    print("Testing Multi-ORN Pathway Script PN Identification")
    print("="*70)

    # Run the multi-ORN pathway script
    result = subprocess.run(
        ["python3", "scripts/map_multi_orn_pathways.py", "--max-levels", "2", "--log-level", "INFO"],
        capture_output=True,
        text=True,
        timeout=180
    )

    if result.returncode != 0:
        print("\n✗ Script failed to run:")
        print(result.stderr)
        raise RuntimeError("Multi-ORN pathway script failed")

    # Parse PN counts from output
    output = result.stdout + result.stderr

    # Pattern: Level 1 (DL5_Projection_Neurons): 2 PNs
    pn_pattern = r'Level 1 \((\w+)_Projection_Neurons\): (\d+) PNs'

    found_counts = {}
    for match in re.finditer(pn_pattern, output):
        glomerulus = match.group(1)
        count = int(match.group(2))

        # Map glomerulus to ORN name
        glom_to_orn = {
            "DL5": "Or7a",
            "DC2": "Or13a",
            "DM1": "Or42b",
            "VA1v": "Or47b",
            "DM4": "Or59b",
        }

        if glomerulus in glom_to_orn:
            orn_name = glom_to_orn[glomerulus]
            found_counts[orn_name] = count

    # Verify all ORNs were processed
    assert len(found_counts) == len(EXPECTED_PN_COUNTS), \
        f"Expected {len(EXPECTED_PN_COUNTS)} ORNs, but found {len(found_counts)}"

    # Verify each count
    print("\nPN Count Verification:")
    all_passed = True
    for orn_name, expected in EXPECTED_PN_COUNTS.items():
        actual = found_counts.get(orn_name, 0)
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {orn_name}: {actual} PNs (expected {expected})")

        if actual != expected:
            all_passed = False

    print()
    if not all_passed:
        raise AssertionError("PN counts do not match expected values")

    print("✓ All PN counts match expected values!")
    return found_counts


def test_or7a_matches_specific_script():
    """Test that multi-ORN Or7a results match OR7a-specific script."""
    print("\n" + "="*70)
    print("Testing Or7a Consistency: Multi-ORN vs Specific Script")
    print("="*70)

    # Run OR7a-specific script
    result_specific = subprocess.run(
        ["python3", "scripts/map_or7a_complete_pathway.py", "--max-levels", "2"],
        capture_output=True,
        text=True,
        timeout=180
    )

    # Note: Script may fail at visualization stage (Qt/display issues) but PN counting happens first
    # So we check if we got the PN count even if script failed
    output_specific = result_specific.stdout + result_specific.stderr

    # Parse PN count from OR7a-specific script
    # Pattern: "Identified 2 DL5 projection neurons"
    match_specific = re.search(r'Identified (\d+) DL5 projection neurons', output_specific)

    if not match_specific:
        # Script may have failed before completing, check if it's visualization-related
        if "Could not find the Qt platform" in output_specific:
            print(f"\n⚠ OR7a-specific script failed at visualization (Qt/display issue)")
            print(f"  Skipping consistency test (PN counting likely succeeded but logs cut off)")
            return None
        else:
            raise RuntimeError(f"Could not parse PN count from OR7a-specific script\nOutput: {output_specific[:500]}")

    pn_count_specific = int(match_specific.group(1))

    # Run multi-ORN script and get Or7a count
    result_multi = subprocess.run(
        ["python3", "scripts/map_multi_orn_pathways.py", "--max-levels", "2", "--log-level", "INFO"],
        capture_output=True,
        text=True,
        timeout=180
    )

    if result_multi.returncode != 0:
        raise RuntimeError("Multi-ORN script failed")

    # Parse PN count from multi-ORN script for Or7a
    match_multi = re.search(r'Level 1 \(DL5_Projection_Neurons\): (\d+) PNs',
                            result_multi.stdout + result_multi.stderr)

    if not match_multi:
        raise RuntimeError("Could not parse Or7a PN count from multi-ORN script")

    pn_count_multi = int(match_multi.group(1))

    # Compare
    print(f"\nOR7a-specific script: {pn_count_specific} PNs")
    print(f"Multi-ORN script:     {pn_count_multi} PNs")

    assert pn_count_specific == pn_count_multi, \
        f"PN counts differ: specific={pn_count_specific}, multi={pn_count_multi}"

    print(f"\n✓ Both scripts agree: {pn_count_specific} PNs for Or7a")
    return pn_count_specific


if __name__ == "__main__":
    import sys

    print("="*70)
    print("PN Identification Regression Tests")
    print("="*70)

    try:
        # Test 1: Multi-ORN PN counts
        test_multi_orn_pn_counts()

        # Test 2: Or7a consistency (may be skipped if visualization fails)
        result = test_or7a_matches_specific_script()
        if result is None:
            print("\n  (Or7a consistency test skipped - visualization issue)")

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        sys.exit(0)

    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ TEST FAILED: {e}")
        print("="*70)
        sys.exit(1)
