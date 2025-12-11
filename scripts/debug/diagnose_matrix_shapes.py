#!/usr/bin/env python3
"""
Diagnose connectivity matrix shape errors.

This script validates that connectivity matrices have the correct orientation
for matrix multiplication operations in the stochastic PER model.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import json

def diagnose_matrices():
    """Run comprehensive matrix shape diagnostics."""
    cache_dir = Path('data/cache/ethyl_butyrate_circuit/connectivity_matrices')

    print("=" * 80)
    print("CONNECTIVITY MATRIX SHAPE DIAGNOSTICS")
    print("=" * 80)

    # Load matrices
    matrices = {
        'orn_to_pn': sp.load_npz(cache_dir / 'orn_to_pn.npz'),
        'pn_to_kc': sp.load_npz(cache_dir / 'pn_to_kc.npz'),
        'kc_to_mbon': sp.load_npz(cache_dir / 'kc_to_mbon.npz')
    }

    # Load index files
    with open(cache_dir / 'orn_to_pn_indices.json') as f:
        orn_pn_idx = json.load(f)
    with open(cache_dir / 'pn_to_kc_indices.json') as f:
        pn_kc_idx = json.load(f)
    with open(cache_dir / 'kc_to_mbon_indices.json') as f:
        kc_mbon_idx = json.load(f)

    print("\n1. ACTUAL SHAPES (as saved):")
    for name, matrix in matrices.items():
        print(f"   {name:15s}: {matrix.shape} (nnz={matrix.nnz:,})")

    print("\n2. EXPECTED SHAPES (for matrix multiplication):")
    print("   For W @ activity_vector:")
    print(f"   orn_to_pn:      (n_pn, n_orn)   = ({len(orn_pn_idx['post_ids'])}, {len(orn_pn_idx['pre_ids'])})")
    print(f"   pn_to_kc:       (n_kc, n_pn)    = ({len(pn_kc_idx['post_ids'])}, {len(pn_kc_idx['pre_ids'])})")
    print(f"   kc_to_mbon:     (n_mbon, n_kc)  = ({len(kc_mbon_idx['post_ids'])}, {len(kc_mbon_idx['pre_ids'])})")

    print("\n3. SHAPE VALIDATION:")
    issues = []

    # Test 1: ORN→PN
    expected_orn_pn = (len(orn_pn_idx['pre_ids']), len(orn_pn_idx['post_ids']))
    actual_orn_pn = matrices['orn_to_pn'].shape
    if actual_orn_pn == expected_orn_pn:
        print(f"   ✅ orn_to_pn: Correct orientation (pre, post)")
    elif actual_orn_pn == expected_orn_pn[::-1]:
        print(f"   ⚠️  orn_to_pn: Matrix is (post, pre) - common format, may need transpose")
        issues.append(('orn_to_pn', 'transposed'))
    else:
        print(f"   ❌ orn_to_pn: Shape mismatch! Expected {expected_orn_pn}, got {actual_orn_pn}")
        issues.append(('orn_to_pn', 'shape_error'))

    # Test 2: PN→KC
    expected_pn_kc = (len(pn_kc_idx['pre_ids']), len(pn_kc_idx['post_ids']))
    actual_pn_kc = matrices['pn_to_kc'].shape
    if actual_pn_kc == expected_pn_kc:
        print(f"   ✅ pn_to_kc: Correct orientation (pre, post)")
    elif actual_pn_kc == expected_pn_kc[::-1]:
        print(f"   ⚠️  pn_to_kc: Matrix is (post, pre) - may need transpose")
        issues.append(('pn_to_kc', 'transposed'))
    else:
        print(f"   ❌ pn_to_kc: Shape mismatch! Expected {expected_pn_kc}, got {actual_pn_kc}")
        issues.append(('pn_to_kc', 'shape_error'))

    # Test 3: KC→MBON (THE CRITICAL ONE)
    expected_kc_mbon = (len(kc_mbon_idx['pre_ids']), len(kc_mbon_idx['post_ids']))
    actual_kc_mbon = matrices['kc_to_mbon'].shape
    if actual_kc_mbon == expected_kc_mbon:
        print(f"   ✅ kc_to_mbon: Correct orientation (pre, post)")
    elif actual_kc_mbon == expected_kc_mbon[::-1]:
        print(f"   ⚠️  kc_to_mbon: Matrix is (post, pre) - NEEDS transpose for matmul!")
        print(f"      Expected (KC, MBON) = {expected_kc_mbon}")
        print(f"      Got (MBON, KC) = {actual_kc_mbon}")
        issues.append(('kc_to_mbon', 'transposed'))
    else:
        print(f"   ❌ kc_to_mbon: Shape mismatch! Expected {expected_kc_mbon}, got {actual_kc_mbon}")
        issues.append(('kc_to_mbon', 'shape_error'))

    print("\n4. MATRIX MULTIPLICATION TESTS:")
    print("   Testing the operation: mbon_activity = W_kc_mbon @ kc_activity")

    # Create dummy KC activity vector
    n_kcs = len(pn_kc_idx['post_ids'])
    kc_activity = np.random.rand(n_kcs)
    print(f"   KC activity vector shape: ({n_kcs},)")

    # Test with actual saved matrix
    print(f"\n   Test 1: Using matrix as saved {actual_kc_mbon}")
    try:
        result = matrices['kc_to_mbon'] @ kc_activity
        print(f"   ✅ Multiplication succeeded! Result shape: {result.shape}")
        if result.shape[0] == len(kc_mbon_idx['post_ids']):
            print(f"      ✅ Result has correct length ({len(kc_mbon_idx['post_ids'])} MBONs)")
        else:
            print(f"      ❌ Result length wrong! Expected {len(kc_mbon_idx['post_ids'])}, got {result.shape[0]}")
    except ValueError as e:
        print(f"   ❌ Multiplication FAILED:")
        print(f"      Error: {str(e)}")
        print(f"      → Matrix needs to be transposed!")

    # Test with transposed matrix
    print(f"\n   Test 2: Using transposed matrix {actual_kc_mbon[::-1]}")
    try:
        result = matrices['kc_to_mbon'].T @ kc_activity
        print(f"   ✅ Multiplication succeeded! Result shape: {result.shape}")
        if result.shape[0] == len(kc_mbon_idx['post_ids']):
            print(f"      ✅ Result has correct length ({len(kc_mbon_idx['post_ids'])} MBONs)")
        else:
            print(f"      ❌ Result length wrong!")
    except ValueError as e:
        print(f"   ❌ Still fails: {e}")

    print("\n5. ROOT CAUSE ANALYSIS:")
    if issues:
        print("   Issues found:")
        for matrix_name, issue_type in issues:
            if issue_type == 'transposed':
                print(f"   • {matrix_name}: Saved with (pre, post) orientation")
                print(f"     → This is standard sparse matrix format")
                print(f"     → Model must transpose for correct matmul direction")
    else:
        print("   ✅ All matrices have expected shapes!")

    print("\n6. RECOMMENDED FIX:")
    print("   The stochastic model auto-transposes matrices if needed:")
    print("   ```python")
    print("   # In StochasticPERModel.__init__():")
    print("   self.W_kc_mbon = W_kc_mbon.T if W_kc_mbon.shape[0] < W_kc_mbon.shape[1] else W_kc_mbon")
    print("   ```")
    print("   This converts (KC, MBON) → (MBON, KC) for matmul compatibility")

    print("\n7. KC MAPPING DIAGNOSTIC:")
    print(f"   PNs in PN→KC: {len(pn_kc_idx['pre_ids'])}")
    print(f"   KCs in PN→KC: {len(pn_kc_idx['post_ids'])}")
    print(f"   KCs in KC→MBON: {len(kc_mbon_idx['pre_ids'])}")
    print(f"   MBONs in KC→MBON: {len(kc_mbon_idx['post_ids'])}")

    kc_pn_set = set(pn_kc_idx['post_ids'])
    kc_mbon_set = set(kc_mbon_idx['pre_ids'])
    overlap = len(kc_pn_set & kc_mbon_set)

    print(f"\n   KC overlap between layers:")
    print(f"   - KCs receiving PN input: {len(kc_pn_set)}")
    print(f"   - KCs projecting to MBON: {len(kc_mbon_set)}")
    print(f"   - Overlap: {overlap}")
    print(f"   - Only in PN→KC: {len(kc_pn_set - kc_mbon_set)}")
    print(f"   - Only in KC→MBON: {len(kc_mbon_set - kc_pn_set)}")

    if len(kc_pn_set) != len(kc_mbon_set):
        print(f"\n   ⚠️  KC spaces don't match! Need kc_mapping to align them")
        print(f"   → load_connectivity_matrices() creates this mapping")

    print("=" * 80)
    return issues

if __name__ == '__main__':
    issues = diagnose_matrices()
    sys.exit(0 if not issues else 1)
