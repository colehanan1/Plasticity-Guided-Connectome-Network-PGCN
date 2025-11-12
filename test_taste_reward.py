#!/usr/bin/env python3
"""Test the simplified TasteRewardCircuit module.

This tests the basic taste reward functionality without GABA veto gates.
"""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_import():
    """Test 1: Can we import TasteRewardCircuit?"""
    print("Test 1: Importing TasteRewardCircuit...")
    try:
        from pgcn.models.taste_reward import TasteRewardCircuit
        print("  ✓ TasteRewardCircuit imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import: {e}")
        return False


def test_instantiation():
    """Test 2: Can we instantiate the circuit?"""
    print("\nTest 2: Instantiating TasteRewardCircuit...")
    try:
        from pgcn.models.taste_reward import TasteRewardCircuit

        taste = TasteRewardCircuit(
            data_dir=Path("data/cache"),
            use_synapse_weights=True
        )

        print(f"  ✓ Circuit instantiated successfully")
        print(f"    - {taste.n_grns} GRNs")
        print(f"    - {taste.n_ach_lns} ACh-LNs")
        print(f"    - {taste.n_sez_pns} SEZ-PNs")

        return True, taste
    except Exception as e:
        print(f"  ✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_scalar():
    """Test 3: Forward pass with scalar input."""
    print("\nTest 3: Forward pass with scalar sugar input...")
    try:
        from pgcn.models.taste_reward import TasteRewardCircuit

        taste = TasteRewardCircuit(data_dir=Path("data/cache"))

        # Test with full sugar
        reward_full = taste(sugar_input=1.0)
        print(f"  Full sugar (1.0): reward = {reward_full.item():.4f}")

        # Test with half sugar
        reward_half = taste(sugar_input=0.5)
        print(f"  Half sugar (0.5): reward = {reward_half.item():.4f}")

        # Test with no sugar
        reward_none = taste(sugar_input=0.0)
        print(f"  No sugar (0.0): reward = {reward_none.item():.4f}")

        # Verify ordering
        if reward_full > reward_half > reward_none:
            print(f"  ✓ Reward scales correctly with sugar")
            return True
        else:
            print(f"  ✗ Reward scaling unexpected")
            return False

    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_tensor():
    """Test 4: Forward pass with tensor input."""
    print("\nTest 4: Forward pass with tensor input...")
    try:
        from pgcn.models.taste_reward import TasteRewardCircuit

        taste = TasteRewardCircuit(data_dir=Path("data/cache"))

        # Batch of 3 samples
        batch_size = 3
        sugar_batch = torch.tensor([1.0, 0.5, 0.0]).unsqueeze(1).expand(batch_size, taste.n_grns)

        rewards = taste(sugar_batch)
        print(f"  Batch rewards shape: {rewards.shape}")
        print(f"  Rewards: {rewards.detach().numpy()}")

        if rewards.shape == (3,):
            print(f"  ✓ Batched forward pass works")
            return True
        else:
            print(f"  ✗ Unexpected output shape")
            return False

    except Exception as e:
        print(f"  ✗ Tensor forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dopamine_computation():
    """Test 5: Dopamine (RPE) computation."""
    print("\nTest 5: Computing dopamine (RPE)...")
    try:
        from pgcn.models.taste_reward import TasteRewardCircuit

        taste = TasteRewardCircuit(data_dir=Path("data/cache"))

        # Scenario 1: Better than expected (positive RPE)
        predicted = torch.tensor([0.2])
        dopamine_pos = taste.compute_dopamine(sugar_input=1.0, predicted_reward=predicted)
        print(f"  Better than expected: dopamine = {dopamine_pos.item():.4f} (should be positive)")

        # Scenario 2: Worse than expected (negative RPE)
        predicted = torch.tensor([0.8])
        dopamine_neg = taste.compute_dopamine(sugar_input=0.0, predicted_reward=predicted)
        print(f"  Worse than expected: dopamine = {dopamine_neg.item():.4f} (should be negative)")

        # Scenario 3: As expected (small RPE)
        actual_reward = taste(sugar_input=0.5)
        dopamine_zero = taste.compute_dopamine(sugar_input=0.5, predicted_reward=actual_reward)
        print(f"  As expected: dopamine = {dopamine_zero.item():.4f} (should be ~0)")

        if dopamine_pos > 0 and dopamine_neg < 0 and abs(dopamine_zero.item()) < 0.1:
            print(f"  ✓ Dopamine computation correct")
            return True
        else:
            print(f"  ✗ Dopamine values unexpected")
            return False

    except Exception as e:
        print(f"  ✗ Dopamine computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_return_details():
    """Test 6: Detailed return mode."""
    print("\nTest 6: Returning detailed activations...")
    try:
        from pgcn.models.taste_reward import TasteRewardCircuit

        taste = TasteRewardCircuit(data_dir=Path("data/cache"))

        output = taste(sugar_input=1.0, return_details=True)

        print(f"  Output keys: {list(output.keys())}")
        print(f"  Reward signal: {output['reward_signal'].item():.4f}")
        print(f"  GRN activity shape: {output['grn_activity'].shape}")
        print(f"  ACh-LN activity shape: {output['ach_ln_activity'].shape}")
        print(f"  SEZ-PN activity shape: {output['sez_pn_activity'].shape}")

        expected_keys = ['reward_signal', 'grn_activity', 'ach_ln_activity', 'sez_pn_activity']
        if all(k in output for k in expected_keys):
            print(f"  ✓ Detailed output works")
            return True
        else:
            print(f"  ✗ Missing expected keys")
            return False

    except Exception as e:
        print(f"  ✗ Detailed return failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics():
    """Test 7: Get circuit statistics."""
    print("\nTest 7: Getting circuit statistics...")
    try:
        from pgcn.models.taste_reward import TasteRewardCircuit

        taste = TasteRewardCircuit(data_dir=Path("data/cache"))

        stats = taste.get_statistics()

        print(f"  Statistics:")
        for key, value in stats.items():
            print(f"    {key}: {value}")

        if 'n_grns' in stats and 'grn_to_ach_connections' in stats:
            print(f"  ✓ Statistics retrieved")
            return True
        else:
            print(f"  ✗ Incomplete statistics")
            return False

    except Exception as e:
        print(f"  ✗ Statistics failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("TASTE REWARD CIRCUIT TESTS (Simplified Version)")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Import module", test_import()))

    success, _ = test_instantiation()
    results.append(("Instantiate circuit", success))

    results.append(("Forward (scalar)", test_forward_scalar()))
    results.append(("Forward (tensor)", test_forward_tensor()))
    results.append(("Dopamine computation", test_dopamine_computation()))
    results.append(("Detailed output", test_return_details()))
    results.append(("Statistics", test_statistics()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe simplified taste reward circuit is ready for integration!")
        print("\nNext steps:")
        print("  1. Integrate into EnhancedOlfactoryCircuit")
        print("  2. Add to training loop for OR7a blocking experiments")
        print("  3. Test Experiments 1, 2, 3 with taste reward")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix failing tests before integration.")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
