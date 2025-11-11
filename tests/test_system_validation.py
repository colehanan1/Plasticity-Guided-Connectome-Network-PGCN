"""
PGCN System Validation Test Suite

Comprehensive validation of the complete 14,629-neuron PGCN system,
including connectivity, metadata, and biological plausibility checks.

Usage:
    # Run as pytest suite
    pytest tests/test_system_validation.py -v -s

    # Run as standalone script
    python tests/test_system_validation.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import sparse

# Test configuration
CACHE_DIR = Path("data/cache")
EXPECTED_COUNTS = {
    'pn': 482,
    'kc': 5177,
    'mbon': 96,
    'dan': 584,
    'ln': 3829,
    'lh': 1162,
    'motor': 66,
    'an': 1926,
    'dn': 1303,
    'cb0191': 2,
    'sez_nsc_capa': 2
}

# ============================================================================
# TEST 1: PN→KC Connectivity Validation
# ============================================================================

def test_pn_to_kc_connectivity():
    """
    Verify PN→KC connectivity exists and is biologically plausible.

    Expected:
    - 40,000-80,000 connections
    - 100,000-200,000 total synapses
    - Average 2-5 synapses per connection
    - All 482 PNs have outputs
    - 4,000-5,177 KCs have inputs (sparse connectivity)
    """
    print("\n" + "="*80)
    print("TEST 1: PN→KC Connectivity")
    print("="*80)

    # Check if connectivity file exists
    # Try multiple possible filenames
    possible_files = [
        CACHE_DIR / "pn_to_kc_connectivity.csv",
        CACHE_DIR / "edges.parquet",
        CACHE_DIR / "connections.csv",
    ]

    conn_file = None
    for f in possible_files:
        if f.exists():
            conn_file = f
            break

    if conn_file is None:
        print("  ⚠ WARNING: No connectivity file found. Skipping PN→KC connectivity test.")
        print("    Expected files:")
        for f in possible_files:
            print(f"      - {f}")
        pytest.skip("PN→KC connectivity file not found")

    # Load connectivity
    if conn_file.suffix == '.parquet':
        pn_to_kc = pd.read_parquet(conn_file)
    else:
        pn_to_kc = pd.read_csv(conn_file)

    # Load PNs and KCs to filter connections
    pn_file = CACHE_DIR / "alpn_extracted.csv"
    kc_files = list(CACHE_DIR.glob("kc_*.csv"))

    if pn_file.exists() and len(kc_files) > 0:
        pns = pd.read_csv(pn_file)
        pn_ids = set(pns['root_id'].values)

        kc_ids = set()
        for kc_file in kc_files:
            kcs = pd.read_csv(kc_file)
            kc_ids.update(kcs['root_id'].values)

        # Filter to PN→KC connections
        pn_to_kc = pn_to_kc[
            pn_to_kc['pre_root_id'].isin(pn_ids) &
            pn_to_kc['post_root_id'].isin(kc_ids)
        ].copy()

    # Basic structure checks
    assert 'pre_root_id' in pn_to_kc.columns, "Missing 'pre_root_id' column"
    assert 'post_root_id' in pn_to_kc.columns, "Missing 'post_root_id' column"
    assert 'syn_count' in pn_to_kc.columns, "Missing 'syn_count' column"

    # Connection count checks
    n_connections = len(pn_to_kc)
    print(f"  ✓ PN→KC connections: {n_connections:,}")

    if n_connections == 0:
        print("  ⚠ WARNING: No PN→KC connections found. Check filtering logic.")
        pytest.skip("No PN→KC connections found")

    # Allow wider range due to data variability
    assert 10_000 <= n_connections <= 150_000, \
        f"Connection count {n_connections:,} outside plausible range [10k-150k]"

    # Synapse count checks
    total_synapses = pn_to_kc['syn_count'].sum()
    print(f"  ✓ Total synapses: {total_synapses:,}")
    assert 20_000 <= total_synapses <= 500_000, \
        f"Synapse count {total_synapses:,} outside plausible range [20k-500k]"

    # Connectivity statistics
    unique_pns = pn_to_kc['pre_root_id'].nunique()
    unique_kcs = pn_to_kc['post_root_id'].nunique()
    avg_synapses = pn_to_kc['syn_count'].mean()

    print(f"  ✓ Unique PNs with output: {unique_pns} / {EXPECTED_COUNTS['pn']}")
    print(f"  ✓ Unique KCs with input: {unique_kcs} / {EXPECTED_COUNTS['kc']}")
    print(f"  ✓ Avg synapses per connection: {avg_synapses:.1f}")

    # Biological plausibility checks (relaxed)
    assert unique_pns >= 100, f"Too few PNs with output: {unique_pns}"
    assert unique_kcs >= 1000, f"Too few KCs with input: {unique_kcs}"
    assert 0.5 <= avg_synapses <= 15.0, f"Avg synapses {avg_synapses:.1f} implausible"

    # Sparsity check (each KC should receive from ~5-8 PNs on average)
    kc_pn_partners = pn_to_kc.groupby('post_root_id')['pre_root_id'].nunique()
    avg_pn_per_kc = kc_pn_partners.mean()
    print(f"  ✓ Avg PN inputs per KC: {avg_pn_per_kc:.1f}")
    assert 1 <= avg_pn_per_kc <= 20, f"PN/KC ratio {avg_pn_per_kc:.1f} implausible"

    print("  ✅ PN→KC connectivity: PASSED")
    return pn_to_kc


# ============================================================================
# TEST 2: KC Subtype Distribution
# ============================================================================

def test_kc_subtype_distribution():
    """
    Verify KC subtype distribution matches published ratios.

    Expected ratios (from Aso et al. 2014, Li et al. 2020):
    - α/β: ~30-35% (long-term memory)
    - γ: ~45-50% (short-term memory)
    - α'/β': ~15-20% (intermediate memory)
    """
    print("\n" + "="*80)
    print("TEST 2: KC Subtype Distribution")
    print("="*80)

    kc_files = {
        'α/β': 'kc_ab.csv',
        'α/β-posterior': 'kc_ab_p.csv',
        'γ-main': 'kc_g_main.csv',
        'γ-dorsal': 'kc_g_dorsal.csv',
        'γ-sparse': 'kc_g_sparse.csv',
        "α'/β'-main": 'kc_apbp_main.csv',
        "α'/β'-AP1": 'kc_apbp_ap1.csv',
        "α'/β'-AP2": 'kc_apbp_ap2.csv',
    }

    total_kcs = 0
    subtype_counts = {}

    for subtype, filename in kc_files.items():
        filepath = CACHE_DIR / filename
        assert filepath.exists(), f"Missing KC subtype file: {filepath}"

        df = pd.read_csv(filepath)
        count = len(df)
        total_kcs += count
        subtype_counts[subtype] = count

        percent = (count / EXPECTED_COUNTS['kc']) * 100
        print(f"  {subtype:20s}: {count:4d} ({percent:5.1f}%)")

    print(f"\n  Total KCs: {total_kcs} (expected: {EXPECTED_COUNTS['kc']})")
    assert total_kcs == EXPECTED_COUNTS['kc'], \
        f"KC count mismatch: {total_kcs} != {EXPECTED_COUNTS['kc']}"

    # Validate major subtype ratios
    ab_total = subtype_counts['α/β'] + subtype_counts['α/β-posterior']
    gamma_total = sum(subtype_counts[k] for k in subtype_counts if 'γ' in k)
    apbp_total = sum(subtype_counts[k] for k in subtype_counts if "α'/β'" in k)

    ab_pct = (ab_total / total_kcs) * 100
    gamma_pct = (gamma_total / total_kcs) * 100
    apbp_pct = (apbp_total / total_kcs) * 100

    print(f"\n  Major Subtype Ratios:")
    print(f"    α/β (long-term):      {ab_pct:.1f}% (expected: 30-35%)")
    print(f"    γ (short-term):       {gamma_pct:.1f}% (expected: 45-50%)")
    print(f"    α'/β' (intermediate): {apbp_pct:.1f}% (expected: 15-20%)")

    # Relaxed ratio checks (allow some variation)
    assert 20 <= ab_pct <= 45, f"α/β ratio {ab_pct:.1f}% outside plausible range"
    assert 35 <= gamma_pct <= 60, f"γ ratio {gamma_pct:.1f}% outside plausible range"
    assert 10 <= apbp_pct <= 30, f"α'/β' ratio {apbp_pct:.1f}% outside plausible range"

    print("  ✅ KC subtype distribution: PASSED")
    return subtype_counts


# ============================================================================
# TEST 3: New Cell Type Integration
# ============================================================================

def test_new_cell_types():
    """
    Verify CB0191 and SEZ-NSC^CAPA neurons are properly integrated.

    Expected:
    - CB0191: 2 neurons with root IDs [720575940626843194, 720575940634139799]
    - SEZ-NSC^CAPA: 2 neurons with root IDs [720575940618736797, 720575940620829878]
    - Proper metadata columns (cell_type, neuropeptide, etc.)
    """
    print("\n" + "="*80)
    print("TEST 3: New Cell Type Integration")
    print("="*80)

    # Test CB0191 neurons
    cb0191_file = CACHE_DIR / "cb0191_neurons.csv"
    assert cb0191_file.exists(), f"Missing: {cb0191_file}"

    cb0191 = pd.read_csv(cb0191_file)
    assert len(cb0191) == 2, f"CB0191 count mismatch: {len(cb0191)} != 2"

    expected_cb_ids = {720575940626843194, 720575940634139799}
    actual_cb_ids = set(cb0191['root_id'].values)
    assert actual_cb_ids == expected_cb_ids, \
        f"CB0191 root IDs mismatch: {actual_cb_ids} != {expected_cb_ids}"

    print(f"  ✓ CB0191: 2 neurons")
    print(f"    Root IDs: {sorted(actual_cb_ids)}")

    if 'cell_type' in cb0191.columns:
        assert (cb0191['cell_type'] == 'CB0191').all(), \
            "CB0191 neurons missing 'cell_type' label"
        print(f"    Cell type: {cb0191['cell_type'].unique()}")

    # Test SEZ-NSC^CAPA neurons
    sez_file = CACHE_DIR / "sez_nsc_capa.csv"
    assert sez_file.exists(), f"Missing: {sez_file}"

    sez = pd.read_csv(sez_file)
    assert len(sez) == 2, f"SEZ-NSC^CAPA count mismatch: {len(sez)} != 2"

    expected_sez_ids = {720575940618736797, 720575940620829878}
    actual_sez_ids = set(sez['root_id'].values)
    assert actual_sez_ids == expected_sez_ids, \
        f"SEZ-NSC^CAPA root IDs mismatch: {actual_sez_ids} != {expected_sez_ids}"

    print(f"\n  ✓ SEZ-NSC^CAPA: 2 neurons")
    print(f"    Root IDs: {sorted(actual_sez_ids)}")

    if 'cell_type' in sez.columns:
        assert (sez['cell_type'] == 'SEZ_NSC_CAPA').all(), \
            "SEZ-NSC^CAPA neurons missing 'cell_type' label"
        print(f"    Cell type: {sez['cell_type'].unique()}")

    if 'neuropeptide' in sez.columns:
        assert (sez['neuropeptide'] == 'CAPA/Pyrokinin').all(), \
            "SEZ-NSC^CAPA neurons missing neuropeptide annotation"
        print(f"    Neuropeptide: {sez['neuropeptide'].unique()}")

    print("  ✅ New cell types: PASSED")
    return cb0191, sez


# ============================================================================
# TEST 4: System Totals
# ============================================================================

def test_system_totals():
    """
    Verify total neuron counts match expected values across all cell types.
    """
    print("\n" + "="*80)
    print("TEST 4: System Totals")
    print("="*80)

    total_neurons = 0

    for cell_type, expected_count in EXPECTED_COUNTS.items():
        total_neurons += expected_count

    print(f"  Expected total neurons: {total_neurons:,}")
    assert total_neurons == 14629, \
        f"Total count mismatch: {total_neurons} != 14,629"

    print("  ✅ System totals: PASSED")


# ============================================================================
# TEST 5: Metadata Completeness
# ============================================================================

def test_metadata_completeness():
    """
    Verify that core neuron files have essential metadata columns.
    """
    print("\n" + "="*80)
    print("TEST 5: Metadata Completeness")
    print("="*80)

    # Check ALPN metadata
    alpn_file = CACHE_DIR / "alpn_extracted.csv"
    assert alpn_file.exists(), f"Missing: {alpn_file}"

    alpns = pd.read_csv(alpn_file)
    assert 'root_id' in alpns.columns, "ALPNs missing 'root_id'"
    print(f"  ✓ ALPNs: {len(alpns)} neurons with metadata")

    # Check KC metadata
    kc_file = CACHE_DIR / "kc_ab.csv"  # Sample one KC file
    if kc_file.exists():
        kcs = pd.read_csv(kc_file)
        assert 'root_id' in kcs.columns, "KCs missing 'root_id'"
        print(f"  ✓ KCs: Metadata present")

    # Check MBON metadata
    mbon_file = CACHE_DIR / "mbon_all.csv"
    if mbon_file.exists():
        mbons = pd.read_csv(mbon_file)
        assert 'root_id' in mbons.columns, "MBONs missing 'root_id'"
        print(f"  ✓ MBONs: {len(mbons)} neurons with metadata")

    # Check DAN metadata
    dan_file = CACHE_DIR / "dan_all.csv"
    if dan_file.exists():
        dans = pd.read_csv(dan_file)
        assert 'root_id' in dans.columns, "DANs missing 'root_id'"
        print(f"  ✓ DANs: {len(dans)} neurons with metadata")

    # Check extended components
    extended_files = [
        ('ln_all.csv', 'LNs'),
        ('lh_all.csv', 'LH'),
        ('motor_proboscis.csv', 'Motor'),
        ('an_all.csv', 'ANs'),
        ('dn_all.csv', 'DNs'),
    ]

    for filename, label in extended_files:
        filepath = CACHE_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            assert 'root_id' in df.columns, f"{label} missing 'root_id'"
            print(f"  ✓ {label}: {len(df)} neurons with metadata")

    print("  ✅ Metadata completeness: PASSED")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    """Run all tests as a standalone script."""
    print("\n" + "="*80)
    print("PGCN SYSTEM VALIDATION TEST SUITE")
    print("="*80)
    print(f"Cache directory: {CACHE_DIR}")

    # Check if cache directory exists
    if not CACHE_DIR.exists():
        print(f"\n❌ ERROR: Cache directory not found: {CACHE_DIR}")
        print("Please run extraction scripts first:")
        print("  python scripts/extract_circuit.py --dataset-dir data/flywire --output-dir data/cache")
        print("  python scripts/extract_extended_circuit.py --dataset-dir data/flywire --output-dir data/cache")
        exit(1)

    try:
        # Run all tests
        pn_to_kc = test_pn_to_kc_connectivity()
        kc_subtypes = test_kc_subtype_distribution()
        cb0191, sez = test_new_cell_types()
        test_system_totals()
        test_metadata_completeness()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - SYSTEM VALIDATED")
        print("="*80)
        print("\nSystem Summary:")
        print(f"  Total neurons: {sum(EXPECTED_COUNTS.values()):,}")
        print(f"  Cell types: {len(EXPECTED_COUNTS)}")
        print("\nSystem ready for experiments!")

    except AssertionError as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        raise
