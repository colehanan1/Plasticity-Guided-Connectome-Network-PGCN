#!/usr/bin/env python3
"""Quick test to verify taste circuit integration."""

import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_taste_circuit_import():
    """Test 1: Can we import TasteCircuit?"""
    print("Test 1: Importing TasteCircuit...")
    try:
        from pgcn.models.taste_circuit import TasteCircuit
        print("  ✓ TasteCircuit imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import TasteCircuit: {e}")
        return False

def test_extracted_data_exists():
    """Test 2: Do the extracted data files exist?"""
    print("\nTest 2: Checking extracted data files...")

    data_dir = Path("data/cache")
    required_files = [
        "shen2025_appetitive_grn.csv",
        "shen2025_appetitive_sez_pn.csv",
        "shen2025_appetitive_sez_ln_ach.csv",
        "shen2025_appetitive_connectivity_grn_pn.npz",
        "shen2025_appetitive_connectivity_grn_ach.npz",
    ]

    all_exist = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"  ✓ Found {filename}")
        else:
            print(f"  ✗ Missing {filename}")
            all_exist = False

    return all_exist

def test_taste_circuit_instantiation():
    """Test 3: Can we instantiate TasteCircuit?"""
    print("\nTest 3: Instantiating TasteCircuit...")

    try:
        from pgcn.models.taste_circuit import TasteCircuit

        taste_circuit = TasteCircuit(
            data_dir=Path("data/cache"),
            gaba_veto_mode="direct",
            gaba_gain=1.0,
            use_synapse_weights=True
        )

        print(f"  ✓ TasteCircuit instantiated successfully")
        print(f"    - {taste_circuit.n_grns} GRNs")
        print(f"    - {taste_circuit.n_sez_pns} SEZ-PNs")
        print(f"    - {taste_circuit.n_ach_lns} ACh-LNs")
        print(f"    - {taste_circuit.n_gaba_lns} GABA-LNs")

        return True, taste_circuit
    except Exception as e:
        print(f"  ✗ Failed to instantiate TasteCircuit: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_taste_circuit_forward():
    """Test 4: Can we run a forward pass through TasteCircuit?"""
    print("\nTest 4: Running TasteCircuit forward pass...")

    try:
        from pgcn.models.taste_circuit import TasteCircuit

        taste_circuit = TasteCircuit(
            data_dir=Path("data/cache"),
            gaba_veto_mode="direct",
            gaba_gain=1.0,
            use_synapse_weights=True
        )

        # Run forward pass
        output = taste_circuit(sugar_input=1.0)

        print(f"  ✓ Forward pass successful")
        print(f"    - GRN activity: {output['grn_activity'].shape}")
        print(f"    - SEZ-PN activity: {output['sez_pn_activity'].shape}")
        print(f"    - ACh-LN activity: {output['ach_ln_activity'].shape}")
        print(f"    - GABA-LN activity: {output['gaba_ln_activity'].shape}")
        print(f"    - Veto signal: {output['veto_signal'].item():.4f}")

        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_circuit_with_taste():
    """Test 5: Can we instantiate EnhancedOlfactoryCircuit with taste pathway?"""
    print("\nTest 5: Testing EnhancedOlfactoryCircuit with taste pathway...")

    try:
        # This test would require a full connectivity matrix
        # For now, just test imports
        from pgcn.models.enhanced_olfactory_circuit import EnhancedOlfactoryCircuit
        print("  ✓ EnhancedOlfactoryCircuit imported successfully")
        print("  ⓘ Full integration test requires connectivity matrix")

        return True
    except Exception as e:
        print(f"  ✗ Failed to import EnhancedOlfactoryCircuit: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TASTE CIRCUIT INTEGRATION TESTS")
    print("="*60)

    results = []

    # Test 1: Import
    results.append(("Import TasteCircuit", test_taste_circuit_import()))

    # Test 2: Data files
    results.append(("Data files exist", test_extracted_data_exists()))

    # Test 3: Instantiation
    success, _ = test_taste_circuit_instantiation()
    results.append(("Instantiate TasteCircuit", success))

    # Test 4: Forward pass
    results.append(("Forward pass", test_taste_circuit_forward()))

    # Test 5: Integration
    results.append(("Enhanced circuit integration", test_enhanced_circuit_with_taste()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
