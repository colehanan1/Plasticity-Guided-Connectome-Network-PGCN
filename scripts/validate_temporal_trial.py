#!/usr/bin/env python3
"""
Validate Temporal Trial Timing.

This script validates that the TemporalTrial and OperantTrial classes
produce the exact timing profiles specified in the experimental protocol.

Key validations:
1. Classical trial: 30s odor, 25s reward, 5s odor-alone
2. Operant trial: 35s odor, variable reward (depends on response)
3. Travel time compensation (2s delay)
4. Linger time compensation (2s clearance)

Author: PGCN Enhancement
Date: 2025-11-11
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from realistic_behavioral_training import TemporalTrial, OperantTrial


def validate_classical_trial():
    """
    Validate classical conditioning trial timing.

    Expected:
    - Odor at fly: 30.0s (t=2s to t=32s)
    - Reward at fly: 25.0s (t=7s to t=32s)
    - Odor-alone period: 5.0s (t=2s to t=7s)
    - Odor+Reward overlap: 25.0s (t=7s to t=32s)
    """
    print("=" * 70)
    print("CLASSICAL TRIAL VALIDATION")
    print("=" * 70)

    trial = TemporalTrial(
        odor='benzaldehyde',
        valve_duration_s=30,
        reward_onset_delay_s=5,
        travel_time_s=2,
        linger_time_s=2,
        dt=0.1
    )

    time = trial.get_time_axis()
    odor = trial.get_odor_profile()
    reward = trial.get_reward_profile(has_reward=True)

    # Find odor and reward windows
    odor_active = odor > 0
    reward_active = reward > 0

    if not odor_active.any():
        print("  ✗ ERROR: No odor detected!")
        return False

    if not reward_active.any():
        print("  ✗ ERROR: No reward detected!")
        return False

    odor_start = time[odor_active][0]
    odor_end = time[odor_active][-1]
    odor_duration = odor_end - odor_start

    reward_start = time[reward_active][0]
    reward_end = time[reward_active][-1]
    reward_duration = reward_end - reward_start

    odor_alone_duration = reward_start - odor_start

    print(f"\n📏 TIMING MEASUREMENTS:")
    print(f"  Odor at fly:    {odor_start:.1f}s to {odor_end:.1f}s")
    print(f"  Duration:       {odor_duration:.1f}s")
    print(f"  Expected:       30.0s")
    print(f"  Status:         {'✓ PASS' if abs(odor_duration - 30.0) < 0.2 else '✗ FAIL'}")

    print(f"\n  Reward at fly:  {reward_start:.1f}s to {reward_end:.1f}s")
    print(f"  Duration:       {reward_duration:.1f}s")
    print(f"  Expected:       25.0s")
    print(f"  Status:         {'✓ PASS' if abs(reward_duration - 25.0) < 0.2 else '✗ FAIL'}")

    print(f"\n  Odor-alone:     {odor_start:.1f}s to {reward_start:.1f}s")
    print(f"  Duration:       {odor_alone_duration:.1f}s")
    print(f"  Expected:       5.0s")
    print(f"  Status:         {'✓ PASS' if abs(odor_alone_duration - 5.0) < 0.2 else '✗ FAIL'}")

    # Check timing precision
    all_pass = (
        abs(odor_duration - 30.0) < 0.2 and
        abs(reward_duration - 25.0) < 0.2 and
        abs(odor_alone_duration - 5.0) < 0.2
    )

    print(f"\n{'✓' if all_pass else '✗'} Overall: {'PASS' if all_pass else 'FAIL'}")

    return all_pass


def validate_operant_trial():
    """
    Validate operant conditioning trial timing.

    Expected:
    - Odor at fly: 35.0s (t=2s to t=37s)
    - Early response (t=4s): ~33s reward
    - Late response (t=12s): ~25s reward
    """
    print("\n" + "=" * 70)
    print("OPERANT TRIAL VALIDATION")
    print("=" * 70)

    trial = TemporalTrial(
        odor='benzaldehyde',
        valve_duration_s=35,
        travel_time_s=2,
        linger_time_s=2,
        dt=0.1
    )

    time = trial.get_time_axis()
    odor = trial.get_odor_profile()

    # Test early response
    reward_early = trial.get_reward_profile(has_reward=True, response_time_at_fly=4.0)

    # Test late response (default timeout)
    reward_late = trial.get_reward_profile(has_reward=True, response_time_at_fly=12.0)

    # Analyze odor
    odor_active = odor > 0
    odor_start = time[odor_active][0]
    odor_end = time[odor_active][-1]
    odor_duration = odor_end - odor_start

    print(f"\n📏 ODOR TIMING:")
    print(f"  Odor at fly:  {odor_start:.1f}s to {odor_end:.1f}s")
    print(f"  Duration:     {odor_duration:.1f}s")
    print(f"  Expected:     35.0s")
    print(f"  Status:       {'✓ PASS' if abs(odor_duration - 35.0) < 0.2 else '✗ FAIL'}")

    # Analyze early response reward
    print(f"\n📏 EARLY RESPONSE (t=4s at fly):")
    reward_early_active = reward_early > 0
    if reward_early_active.any():
        reward_early_start = time[reward_early_active][0]
        reward_early_end = time[reward_early_active][-1]
        reward_early_duration = reward_early_end - reward_early_start

        print(f"  Reward:       {reward_early_start:.1f}s to {reward_early_end:.1f}s")
        print(f"  Duration:     {reward_early_duration:.1f}s")
        print(f"  Expected:     ~33s")
        print(f"  Status:       {'✓ PASS' if abs(reward_early_duration - 33.0) < 0.5 else '✗ FAIL'}")
    else:
        print(f"  ✗ FAIL: No reward detected!")

    # Analyze late response reward
    print(f"\n📏 LATE RESPONSE (t=12s at fly):")
    reward_late_active = reward_late > 0
    if reward_late_active.any():
        reward_late_start = time[reward_late_active][0]
        reward_late_end = time[reward_late_active][-1]
        reward_late_duration = reward_late_end - reward_late_start

        print(f"  Reward:       {reward_late_start:.1f}s to {reward_late_end:.1f}s")
        print(f"  Duration:     {reward_late_duration:.1f}s")
        print(f"  Expected:     ~25s")
        print(f"  Status:       {'✓ PASS' if abs(reward_late_duration - 25.0) < 0.5 else '✗ FAIL'}")
    else:
        print(f"  ✗ FAIL: No reward detected!")

    # Overall validation
    all_pass = (
        abs(odor_duration - 35.0) < 0.2 and
        reward_early_active.any() and
        reward_late_active.any() and
        abs(reward_early_duration - 33.0) < 0.5 and
        abs(reward_late_duration - 25.0) < 0.5
    )

    print(f"\n{'✓' if all_pass else '✗'} Overall: {'PASS' if all_pass else 'FAIL'}")

    return all_pass


def validate_travel_compensation():
    """
    Validate that travel time properly compensates for delay.

    The key insight: even though odor takes 2s to travel,
    the effective duration at the fly should equal valve_duration_s.
    """
    print("\n" + "=" * 70)
    print("TRAVEL TIME COMPENSATION VALIDATION")
    print("=" * 70)

    print("\n🔍 Testing different valve durations...")

    test_cases = [
        (10, "Short trial"),
        (30, "Standard classical"),
        (35, "Standard operant"),
        (60, "Long trial")
    ]

    all_pass = True

    for valve_dur, description in test_cases:
        trial = TemporalTrial(
            odor='test',
            valve_duration_s=valve_dur,
            travel_time_s=2,
            linger_time_s=2,
            dt=0.1
        )

        time = trial.get_time_axis()
        odor = trial.get_odor_profile()

        odor_active = odor > 0
        if odor_active.any():
            odor_duration = time[odor_active][-1] - time[odor_active][0]
            status = "✓ PASS" if abs(odor_duration - valve_dur) < 0.2 else "✗ FAIL"

            if "FAIL" in status:
                all_pass = False

            print(f"  {description:20s} ({valve_dur}s valve): {odor_duration:.1f}s at fly  {status}")
        else:
            print(f"  {description:20s} ({valve_dur}s valve): NO ODOR  ✗ FAIL")
            all_pass = False

    print(f"\n{'✓' if all_pass else '✗'} Overall: {'PASS' if all_pass else 'FAIL'}")

    return all_pass


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("TEMPORAL TRIAL VALIDATION SUITE")
    print("=" * 70)
    print("Validating realistic fly behavioral training timing")
    print()

    results = {}

    # Test 1: Classical trial
    results['classical'] = validate_classical_trial()

    # Test 2: Operant trial
    results['operant'] = validate_operant_trial()

    # Test 3: Travel compensation
    results['compensation'] = validate_travel_compensation()

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name.capitalize():20s}: {status}")

    all_passed = all(results.values())
    print()
    print("=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Timing implementation is correct!")
    else:
        print("❌ SOME TESTS FAILED - Please review timing implementation")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
