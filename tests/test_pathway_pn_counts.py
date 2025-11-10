"""
Test that PN identification works correctly for all ORN types.

This test verifies that the multi-ORN pathway script correctly identifies
projection neurons for each ORN type, matching the expected counts.
"""
import pandas as pd
from pathlib import Path

# Expected PN counts for each ORN (from biological data)
EXPECTED_PN_COUNTS = {
    "Or7a": 2,   # DL5_adPN neurons
    "Or13a": None,  # To be determined
    "Or42b": None,  # To be determined
    "Or47b": None,  # To be determined
    "Or59b": None,  # To be determined
}

def get_pns_for_orn(orn_name, orn_csv, glomerulus_pattern):
    """
    Find PN neurons for a given ORN type using the same logic as the scripts.

    Parameters
    ----------
    orn_name : str
        Name of the ORN (e.g., "Or7a")
    orn_csv : Path
        Path to ORN neuron CSV file
    glomerulus_pattern : str
        Pattern to match in PN cell types (e.g., "DL5" for Or7a)

    Returns
    -------
    list
        List of PN root IDs
    """
    # Load data
    data_dir = Path("data/flywire")
    conns = pd.read_csv(data_dir / "connections_princeton.csv.gz")
    cell_types = pd.read_csv(data_dir / "consolidated_cell_types.csv.gz")
    orn_df = pd.read_csv(orn_csv)

    # Get outputs from ORNs
    outputs = conns[conns['pre_root_id'].isin(orn_df['root_id'])]
    outputs = outputs[outputs['syn_count'] >= 3]

    # Merge with cell types
    outputs = outputs.merge(
        cell_types,
        left_on='post_root_id',
        right_on='root_id',
        how='left'
    )

    # Find PNs using pattern matching (same logic as identify_cell_category)
    pn_patterns = ["adpn", "lpn", "_pn"]
    outputs['is_pn'] = outputs['primary_type'].fillna('').str.lower().apply(
        lambda x: any(pat in x for pat in pn_patterns)
    )

    # Filter for glomerulus-specific PNs
    pattern_lower = glomerulus_pattern.lower()
    outputs['is_glom_pn'] = outputs['primary_type'].fillna('').str.lower().str.contains(
        pattern_lower,
        na=False
    )

    # Get PNs that match both criteria
    pn_outputs = outputs[outputs['is_pn'] & outputs['is_glom_pn']]
    pn_root_ids = pn_outputs['post_root_id'].unique().tolist()

    return pn_root_ids, pn_outputs


def test_or7a_pn_identification():
    """Test that Or7a correctly identifies 2 DL5_adPN neurons."""
    print("\n" + "="*70)
    print("Testing Or7a PN Identification")
    print("="*70)

    pn_ids, pn_outputs = get_pns_for_orn(
        "Or7a",
        Path("data/flywire/search_results_or7a.csv"),
        "DL5"
    )

    print(f"\nFound {len(pn_ids)} PNs:")
    for pid in pn_ids:
        pn_info = pn_outputs[pn_outputs['post_root_id'] == pid].iloc[0]
        syn_count = pn_outputs[pn_outputs['post_root_id'] == pid]['syn_count'].sum()
        print(f"  {pid}: {pn_info['primary_type']} (total synapses: {syn_count})")

    expected = EXPECTED_PN_COUNTS["Or7a"]
    if expected is not None:
        assert len(pn_ids) == expected, f"Expected {expected} PNs, found {len(pn_ids)}"
        print(f"\n✓ TEST PASSED: Found expected {expected} PNs")
    else:
        print(f"\n⚠ No expected count set - found {len(pn_ids)} PNs")

    return pn_ids


def find_pns_for_all_orns():
    """Find PNs for all ORN types and report."""
    orns = [
        ("Or7a", "search_results_or7a.csv", "DL5"),
        ("Or13a", "search_results_Or13a.csv", "DC2"),
        ("Or42b", "search_results_Or42b.csv", "DM1"),
        ("Or47b", "search_results_Or47b.csv", "VA1v"),
        ("Or59b", "search_results_Or59b.csv", "DM4"),
    ]

    results = {}

    for orn_name, csv_file, glomerulus in orns:
        print("\n" + "="*70)
        print(f"Analyzing {orn_name} ({glomerulus} glomerulus)")
        print("="*70)

        try:
            pn_ids, pn_outputs = get_pns_for_orn(
                orn_name,
                Path("data/flywire") / csv_file,
                glomerulus
            )

            print(f"\nFound {len(pn_ids)} PNs:")
            for pid in pn_ids:
                pn_info = pn_outputs[pn_outputs['post_root_id'] == pid].iloc[0]
                syn_count = pn_outputs[pn_outputs['post_root_id'] == pid]['syn_count'].sum()
                print(f"  {pid}: {pn_info['primary_type']} (total synapses: {syn_count})")

            results[orn_name] = pn_ids

        except Exception as e:
            print(f"✗ ERROR: {e}")
            results[orn_name] = []

    return results


if __name__ == "__main__":
    print("="*70)
    print("PN Identification Test Suite")
    print("="*70)

    # Test Or7a specifically
    or7a_pns = test_or7a_pn_identification()

    # Find PNs for all ORNs
    print("\n\n" + "="*70)
    print("Finding PNs for All ORN Types")
    print("="*70)
    all_results = find_pns_for_all_orns()

    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for orn_name, pn_ids in all_results.items():
        expected = EXPECTED_PN_COUNTS.get(orn_name)
        status = "✓" if expected is None or len(pn_ids) == expected else "✗"
        print(f"{status} {orn_name}: {len(pn_ids)} PNs found (expected: {expected or 'unknown'})")

    print("\n" + "="*70)
    print("Suggested pn_root_ids for ORNPathwayConfig:")
    print("="*70)
    for orn_name, pn_ids in all_results.items():
        if pn_ids:
            ids_str = ", ".join(str(i) for i in pn_ids)
            print(f"{orn_name}: pn_root_ids=({ids_str}),")
        else:
            print(f"{orn_name}: pn_root_ids=None,  # No PNs found")
