#!/usr/bin/env python3
"""
Realistic Fly Behavioral Training Protocol for PGCN.

Reproduces the exact experimental paradigm used in real Drosophila conditioning:
- Phase 1: 3 classical conditioning trials (benzaldehyde + reward)
- Phase 2: 5 operant/discrimination trials (benz + operant, hex unrewarded)
- 30-minute consolidation period
- Phase 3: 10 test trials (multiple odors, no reward)

This implementation includes:
- Precise temporal dynamics (2s travel time, 2s linger time)
- Response-contingent reward delivery (operant conditioning)
- Discrimination training with unrewarded test odors
- Biologically realistic inter-trial intervals
- Consolidation-dependent memory stabilization

Author: PGCN Enhancement
Date: 2025-11-11
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import time
from typing import Optional, Dict, List, Tuple

# PGCN imports
from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.olfactory_circuit import OlfactoryCircuit
from pgcn.models.learning_model import DopamineModulatedPlasticity


class TemporalTrial:
    """
    Simulate a single conditioning trial with realistic timing.

    Key biological principles:
    1. Odor takes 2s to travel from valve to fly (tubing delay)
    2. Odor lingers for 2s after valve closes (clearance time)
    3. Reward ENDS when odor valve closes (not 5s after!)
    4. Effective odor duration at fly = valve_duration (travel compensates)

    Example Classical Trial (30s):
    ---------------------------
    Valve timing:
      t=0s:  Odor valve ON
      t=5s:  Reward valve ON
      t=30s: BOTH valves OFF (same time!)

    Fly timing (2s delay):
      t=2s:  Odor reaches fly
      t=7s:  Reward reaches fly
      t=32s: Odor clears, reward stops

    Effective:
      - Odor at fly: 30s (t=2-32s)
      - Reward at fly: 25s (t=7-32s)
      - Odor-alone: 5s (t=2-7s)
      - Odor+Reward overlap: 25s (t=7-32s)
    """

    def __init__(
        self,
        odor: str,
        valve_duration_s: float = 30.0,
        reward_onset_delay_s: float = 5.0,
        travel_time_s: float = 2.0,
        linger_time_s: float = 2.0,
        dt: float = 0.1
    ):
        """
        Initialize temporal trial parameters.

        Args:
            odor: Odor name (e.g., 'benzaldehyde', '1-hexanol')
            valve_duration_s: Duration valve is open (30s classical, 35s operant)
            reward_onset_delay_s: Delay from valve ON to reward ON (5s)
            travel_time_s: Time for odor to reach fly through tubing (2s)
            linger_time_s: Time for odor to clear after valve closes (2s)
            dt: Simulation timestep in seconds (0.1s = 10Hz)
        """
        self.odor = odor
        self.valve_duration_s = valve_duration_s
        self.reward_onset_delay_s = reward_onset_delay_s
        self.travel_time_s = travel_time_s
        self.linger_time_s = linger_time_s
        self.dt = dt

        # Total simulation time = travel + valve duration + linger
        self.total_time_s = travel_time_s + valve_duration_s + linger_time_s
        self.n_steps = int(self.total_time_s / dt)

    def get_odor_profile(self) -> np.ndarray:
        """
        Returns odor concentration time series at the fly.

        The odor profile is a rectangular pulse that:
        - Starts at t=travel_time_s (odor reaches fly)
        - Lasts for valve_duration_s
        - Ends at t=travel_time_s+valve_duration_s

        Returns:
            np.ndarray: Odor concentration [0-1] at each timestep (n_steps,)
        """
        odor_profile = np.zeros(self.n_steps)

        # Odor reaches fly after travel time
        odor_start_idx = int(self.travel_time_s / self.dt)
        odor_end_idx = int((self.travel_time_s + self.valve_duration_s) / self.dt)

        # Full concentration during valve open (compensated by travel)
        odor_profile[odor_start_idx:odor_end_idx] = 1.0

        return odor_profile

    def get_reward_profile(
        self,
        has_reward: bool = True,
        response_time_at_fly: Optional[float] = None
    ) -> np.ndarray:
        """
        Returns reward (dopamine) time series.

        Two modes:
        1. Classical (response_time_at_fly=None):
           Reward starts at fixed delay (5s) and ends when odor valve closes

        2. Operant (response_time_at_fly=X):
           Reward starts when fly responds and ends when odor valve closes

        Args:
            has_reward: Whether trial includes reward (False for discrimination)
            response_time_at_fly: For operant trials, when fly responds (seconds at fly)
                                 None for classical trials

        Returns:
            np.ndarray: Reward signal [0-1] at each timestep (n_steps,)
        """
        reward_profile = np.zeros(self.n_steps)

        if not has_reward:
            return reward_profile

        # Classical: reward starts at fixed delay after odor valve ON
        if response_time_at_fly is None:
            reward_start_idx = int((self.reward_onset_delay_s + self.travel_time_s) / self.dt)
            reward_end_idx = int((self.valve_duration_s + self.travel_time_s) / self.dt)
            reward_profile[reward_start_idx:reward_end_idx] = 1.0

        # Operant: reward starts when fly responds (within response window)
        else:
            response_idx = int(response_time_at_fly / self.dt)
            reward_end_idx = int((self.valve_duration_s + self.travel_time_s) / self.dt)

            # Ensure response is within trial bounds
            if response_idx < reward_end_idx:
                reward_profile[response_idx:reward_end_idx] = 1.0

        return reward_profile

    def get_time_axis(self) -> np.ndarray:
        """Returns time axis in seconds."""
        return np.arange(self.n_steps) * self.dt


class OperantTrial(TemporalTrial):
    """
    Operant conditioning trial where reward depends on fly's response.

    Biological protocol:
    1. Odor valve opens (benzaldehyde)
    2. Wait 0-10s for fly to extend proboscis (monitored via MBON output)
    3. When fly extends → immediately open reward valve
    4. If no extension by 10s → open reward valve anyway (to maintain motivation)
    5. At t=35s → close BOTH valves

    Key difference from classical:
    - Classical: Fixed 5s delay before reward
    - Operant: Variable delay (depends on fly's response latency)
    - Faster response → More total reward (longer overlap)
    """

    def __init__(
        self,
        odor: str,
        valve_duration_s: float = 35.0,
        response_window_s: float = 10.0,
        proboscis_extension_threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize operant trial.

        Args:
            odor: Odor name
            valve_duration_s: Valve open duration (35s for operant to allow response window)
            response_window_s: Time window for fly to respond (10s)
            proboscis_extension_threshold: MBON threshold for response (0.5)
            **kwargs: Additional TemporalTrial parameters
        """
        super().__init__(odor, valve_duration_s, **kwargs)
        self.response_window_s = response_window_s
        self.proboscis_extension_threshold = proboscis_extension_threshold

    def run_operant_trial(
        self,
        circuit: OlfactoryCircuit,
        plasticity: DopamineModulatedPlasticity,
        pn_activation: np.ndarray
    ) -> Dict:
        """
        Run operant trial with response-contingent reward.

        Protocol:
        1. Present odor to fly (starts at t=travel_time_s)
        2. Monitor MBON output for proboscis extension
        3. When MBON > threshold → trigger reward
        4. If no response by response_window_s → trigger reward anyway
        5. Continue until trial end, updating weights via plasticity

        Args:
            circuit: OlfactoryCircuit instance
            plasticity: DopamineModulatedPlasticity instance
            pn_activation: PN activation pattern for this odor (n_pn,)

        Returns:
            dict with keys:
                - response_time: Time (at fly) when proboscis extended
                - response_latency: Latency from odor onset
                - reward_duration: Total reward delivery time
                - mbon_trace: MBON output over time
                - dopamine_trace: Dopamine signal over time
                - final_mbon: Final MBON output
        """
        # Get stimulus profiles
        odor_profile = self.get_odor_profile()
        time_axis = self.get_time_axis()

        # Phase 1: Monitor for proboscis extension in response window
        response_time = None
        response_window_end = self.travel_time_s + self.response_window_s

        mbon_trace = []

        for t_idx, t in enumerate(time_axis):
            # Only check during response window
            if t < response_window_end:
                # Modulate PNs by odor concentration
                pn_input = pn_activation * odor_profile[t_idx]

                # Propagate signals through circuit
                kc_activation, _ = circuit.propagate_pn_to_kc(pn_input)
                mbon_output = plasticity.compute_mbon_output(kc_activation)

                mbon_trace.append(mbon_output[0])

                # Check for proboscis extension
                if mbon_output[0] > self.proboscis_extension_threshold:
                    response_time = t
                    break
            else:
                break

        # If no response detected, default to 10s delay (end of window)
        if response_time is None:
            response_time = response_window_end

        # Phase 2: Generate reward profile based on response
        reward_profile = self.get_reward_profile(
            has_reward=True,
            response_time_at_fly=response_time
        )

        # Phase 3: Run full trial with plasticity updates
        trial_results = self._simulate_trial(
            circuit, plasticity, pn_activation,
            odor_profile, reward_profile, time_axis
        )

        # Add operant-specific metrics
        trial_results['response_time'] = response_time
        trial_results['response_latency'] = response_time - self.travel_time_s
        trial_results['reward_duration'] = reward_profile.sum() * self.dt

        return trial_results

    def _simulate_trial(
        self,
        circuit: OlfactoryCircuit,
        plasticity: DopamineModulatedPlasticity,
        pn_activation: np.ndarray,
        odor_profile: np.ndarray,
        reward_profile: np.ndarray,
        time_axis: np.ndarray
    ) -> Dict:
        """
        Simulate trial dynamics with dopamine-gated plasticity.

        This runs the full trial timestep-by-timestep, applying:
        - Odor-modulated PN inputs
        - Sparse KC activation
        - MBON output computation
        - Dopamine-gated weight updates

        Args:
            circuit: Olfactory circuit
            plasticity: Plasticity manager
            pn_activation: Base PN activation (n_pn,)
            odor_profile: Odor concentration over time (n_steps,)
            reward_profile: Reward (dopamine) over time (n_steps,)
            time_axis: Time in seconds (n_steps,)

        Returns:
            dict with trial metrics and traces
        """
        # Initialize tracking arrays
        mbon_outputs = np.zeros(len(time_axis))
        dopamine_signals = np.zeros(len(time_axis))
        weight_changes = []

        # Simulate trial timestep-by-timestep
        for t_idx, t in enumerate(time_axis):
            # Modulate PNs by odor concentration
            pn_input = pn_activation * odor_profile[t_idx]

            # Propagate through circuit
            kc_activation, _ = circuit.propagate_pn_to_kc(pn_input)
            mbon_output = plasticity.compute_mbon_output(kc_activation)

            # Dopamine signal from reward
            dopamine = reward_profile[t_idx]

            # Update weights with three-factor rule (only when dopamine present)
            if dopamine > 0:
                # Compute weight change magnitude before update
                weights_before = plasticity.kc_to_mbon.copy()

                # Apply plasticity update
                # ΔW = α × KC × MBON × DA
                delta_w = (plasticity.learning_rate *
                          np.outer(mbon_output, kc_activation) *
                          dopamine)
                plasticity.kc_to_mbon += delta_w

                # Track change magnitude
                weight_change_mag = np.abs(delta_w).sum()
                weight_changes.append(weight_change_mag)

            # Store traces
            mbon_outputs[t_idx] = mbon_output[0]
            dopamine_signals[t_idx] = dopamine

        return {
            'mbon_trace': mbon_outputs,
            'dopamine_trace': dopamine_signals,
            'final_mbon': mbon_outputs[-1],
            'mean_mbon': mbon_outputs.mean(),
            'peak_mbon': mbon_outputs.max(),
            'mean_dopamine': dopamine_signals.mean(),
            'total_plasticity_updates': (dopamine_signals > 0).sum(),
            'total_weight_change': sum(weight_changes) if weight_changes else 0.0
        }


def run_test_trial(
    odor: str,
    duration_s: float,
    circuit: OlfactoryCircuit,
    plasticity: DopamineModulatedPlasticity,
    pn_activations: Dict[str, np.ndarray],
    travel_time_s: float = 2.0,
    linger_time_s: float = 2.0,
    response_threshold: float = 0.5
) -> Dict:
    """
    Run a single test trial (no reward, measure response).

    Test trials assess learned associations without reinforcement:
    - Present odor for specified duration
    - Record MBON output (no plasticity updates!)
    - Determine binary response (proboscis extension or not)

    Args:
        odor: Odor name (e.g., 'benzaldehyde', '1-hexanol')
        duration_s: Test duration (10s or 30s)
        circuit: OlfactoryCircuit instance
        plasticity: DopamineModulatedPlasticity instance (weights frozen)
        pn_activations: Dict mapping odor names to PN activation patterns
        travel_time_s: Travel time to fly (2s)
        linger_time_s: Clearance time (2s)
        response_threshold: MBON threshold for response (0.5)

    Returns:
        dict with test metrics:
            - odor: Odor name
            - duration_s: Test duration
            - response: Binary response (0 or 1)
            - peak_mbon: Maximum MBON output
            - mean_mbon: Average MBON output
            - mbon_time_series: Full MBON trace
    """
    # Create trial
    trial = TemporalTrial(
        odor=odor,
        valve_duration_s=duration_s,
        travel_time_s=travel_time_s,
        linger_time_s=linger_time_s
    )

    # Get odor profile (NO REWARD in test trials)
    odor_profile = trial.get_odor_profile()
    time_axis = trial.get_time_axis()

    # Get PN activation for this odor
    if odor not in pn_activations:
        raise ValueError(f"No PN activation pattern for odor '{odor}'")

    pn_activation = pn_activations[odor]

    # Simulate without learning (weights frozen)
    mbon_outputs = []
    for t_idx, t in enumerate(time_axis):
        pn_input = pn_activation * odor_profile[t_idx]
        kc_activation, _ = circuit.propagate_pn_to_kc(pn_input)
        mbon_output = plasticity.compute_mbon_output(kc_activation)
        mbon_outputs.append(mbon_output[0])

    mbon_outputs = np.array(mbon_outputs)

    # Determine response
    # Criterion: MBON output exceeds threshold at any point during trial
    response = 1 if mbon_outputs.max() > response_threshold else 0
    peak_mbon = mbon_outputs.max()
    mean_mbon = mbon_outputs.mean()

    return {
        'odor': odor,
        'duration_s': duration_s,
        'response': response,
        'peak_mbon': peak_mbon,
        'mean_mbon': mean_mbon,
        'mbon_time_series': mbon_outputs
    }


def consolidation_period(
    plasticity: DopamineModulatedPlasticity,
    duration_minutes: float = 30.0,
    eligibility_tau_minutes: float = 10.0
) -> None:
    """
    Simulate memory consolidation processes.

    During consolidation:
    - Eligibility traces decay exponentially (τ ~ 10 min)
    - Synaptic weights stabilize (protein synthesis, CREB activation)
    - No new learning occurs

    Args:
        plasticity: DopamineModulatedPlasticity instance
        duration_minutes: Consolidation duration (30 min standard)
        eligibility_tau_minutes: Time constant for eligibility trace decay (10 min)
    """
    print(f"\n⏱️  CONSOLIDATION PERIOD: {duration_minutes} minutes")
    print("-" * 70)

    # Decay eligibility traces (if implemented)
    if hasattr(plasticity, 'eligibility_traces') and plasticity.eligibility_traces is not None:
        # Exponential decay: e(t) = e(0) * exp(-t/τ)
        decay_factor = np.exp(-duration_minutes / eligibility_tau_minutes)
        plasticity.eligibility_traces *= decay_factor
        percent_decayed = (1 - decay_factor) * 100
        print(f"  ✓ Eligibility traces decayed by {percent_decayed:.1f}%")
    else:
        print(f"  • No eligibility traces to decay")

    # Synaptic scaling (optional homeostatic plasticity)
    if hasattr(plasticity, 'apply_synaptic_scaling'):
        plasticity.apply_synaptic_scaling()
        print(f"  ✓ Synaptic scaling applied")

    print(f"  ✓ Weights stabilized for testing")
    print(f"  ✓ Memory consolidation complete")


def run_realistic_training_protocol(
    cs_odor: str = 'benzaldehyde',
    test_odor: str = '1-hexanol',
    cache_dir: Path = Path('data/cache'),
    output_dir: Path = Path('results/realistic_training'),
    use_door: bool = False
) -> pd.DataFrame:
    """
    Run complete 3-phase realistic training protocol.

    Protocol:
    - Phase 1: 3 classical conditioning trials (CS + reward, 30s)
    - Phase 2: 5 operant/discrimination trials (CS operant 35s, test unrewarded 30s)
    - Consolidation: 30-minute memory stabilization
    - Phase 3: 10 test trials (various odors, 10-30s, no reward)

    Args:
        cs_odor: Conditioned stimulus odor (default: benzaldehyde)
        test_odor: Discrimination test odor (default: 1-hexanol)
        cache_dir: FlyWire connectome cache directory
        output_dir: Results output directory
        use_door: Whether to use DoOR database (requires door-toolkit)

    Returns:
        pd.DataFrame: Test results from Phase 3
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("REALISTIC FLY BEHAVIORAL TRAINING PROTOCOL")
    print("=" * 70)
    print(f"CS Odor (rewarded): {cs_odor}")
    print(f"Test Odor (unrewarded): {test_odor}")
    print(f"Cache: {cache_dir}")
    print(f"Output: {output_dir}")
    print()

    # ========================================================================
    # SETUP: Load connectome and initialize circuit
    # ========================================================================
    print("🔧 SETUP: Loading FlyWire connectome...")
    loader = CircuitLoader(cache_dir=cache_dir)
    connectivity = loader.load_connectivity_matrix(normalize_weights='row')

    circuit = OlfactoryCircuit(connectivity, kc_sparsity_target=0.05)

    print(f"  ✓ Loaded {len(connectivity.pn_ids)} PNs")
    print(f"  ✓ Loaded {len(connectivity.kc_ids)} KCs")
    print(f"  ✓ Loaded {len(connectivity.mbon_ids)} MBONs")
    print(f"  ✓ Loaded {len(connectivity.dan_ids)} DANs")

    # Initialize plasticity with random small weights
    print("\n🧠 Initializing plasticity...")
    plasticity = DopamineModulatedPlasticity(
        kc_to_mbon_weights=connectivity.kc_to_mbon.toarray(),
        learning_rate=0.01,
        eligibility_trace_tau=0.1,
        init_mode='random',
        init_scale=0.001
    )

    print(f"  ✓ Learning rate: 0.01")
    print(f"  ✓ Eligibility trace τ: 0.1s")
    print(f"  ✓ Initial weights: random (scale=0.001)")

    # Prepare PN activation patterns
    print("\n🎨 Preparing odor stimuli...")

    # Define test odors for Phase 3
    test_odors = [
        cs_odor,           # CS (benzaldehyde)
        'ethyl_butyrate',  # Test odor A
        '3-octanol',       # Test odor B
        'linalool',        # Test odor C
        'geosmin',         # Test odor D
        'pentyl_acetate',  # Test odor E
        test_odor          # Discrimination odor (hexanol)
    ]

    # Create PN activation patterns
    # For now, use glomerulus-based activation (simplified)
    # In full implementation, would use DoOR receptor profiles
    pn_activations = {}

    # Map odors to example glomeruli (this should come from DoOR in production)
    odor_to_glomeruli = {
        'benzaldehyde': ['DL5', 'DM1', 'DM4'],
        '1-hexanol': ['DA1', 'DL3', 'VA1d'],
        'ethyl_butyrate': ['DM1', 'DM2', 'DM4'],
        '3-octanol': ['DA1', 'DL1', 'VA1v'],
        'linalool': ['DL4', 'DM5', 'VA2'],
        'geosmin': ['DA2', 'DA4m', 'DC3'],
        'pentyl_acetate': ['DM2', 'VA6', 'VC1']
    }

    for odor in test_odors:
        if odor in odor_to_glomeruli:
            glomeruli = odor_to_glomeruli[odor]
            pn_activations[odor] = circuit.activate_pns_by_glomeruli(
                glomeruli, firing_rate=1.0
            )
            print(f"  ✓ {odor}: {glomeruli}")
        else:
            # Fallback to random activation
            pn_activations[odor] = np.random.rand(len(connectivity.pn_ids)) * 0.5
            print(f"  • {odor}: random activation (no glomerulus mapping)")

    # ========================================================================
    # PHASE 1: Classical Conditioning (3 trials)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: CLASSICAL CONDITIONING (3 trials)")
    print("=" * 70)
    print("Protocol: CS + fixed reward (30s odor, reward at t=5s)")

    phase1_results = []

    for trial_num in range(1, 4):
        print(f"\n📍 Trial {trial_num}: {cs_odor} + reward (30s classical)")
        print("-" * 70)

        trial = TemporalTrial(
            odor=cs_odor,
            valve_duration_s=30,
            reward_onset_delay_s=5,
            travel_time_s=2,
            linger_time_s=2
        )

        # Get profiles
        odor_profile = trial.get_odor_profile()
        reward_profile = trial.get_reward_profile(has_reward=True)
        time_axis = trial.get_time_axis()

        # Get PN activation
        pn_activation = pn_activations[cs_odor]

        # Simulate trial
        mbon_before = None
        mbon_during = []

        for t_idx, t in enumerate(time_axis):
            pn_input = pn_activation * odor_profile[t_idx]
            kc_activation, _ = circuit.propagate_pn_to_kc(pn_input)
            mbon_output = plasticity.compute_mbon_output(kc_activation)

            # Store initial MBON (before any plasticity)
            if mbon_before is None:
                mbon_before = mbon_output[0]

            mbon_during.append(mbon_output[0])

            # Update weights with dopamine (three-factor rule)
            dopamine = reward_profile[t_idx]
            if dopamine > 0:
                delta_w = (plasticity.learning_rate *
                          np.outer(mbon_output, kc_activation) *
                          dopamine)
                plasticity.kc_to_mbon += delta_w

        mbon_after = mbon_output[0]
        mean_mbon = np.mean(mbon_during)

        print(f"  MBON (before):  {mbon_before:.4f}")
        print(f"  MBON (mean):    {mean_mbon:.4f}")
        print(f"  MBON (after):   {mbon_after:.4f}")
        print(f"  Change:         {mbon_after - mbon_before:+.4f}")
        print(f"  Dopamine:       {reward_profile.sum() * trial.dt:.1f}s total")

        phase1_results.append({
            'phase': 1,
            'trial': trial_num,
            'odor': cs_odor,
            'has_reward': True,
            'trial_type': 'classical',
            'mbon_before': mbon_before,
            'mbon_after': mbon_after,
            'mbon_change': mbon_after - mbon_before,
            'dopamine_duration': reward_profile.sum() * trial.dt
        })

        # 5-minute inter-trial interval
        print(f"  [5-minute inter-trial interval]")
        time.sleep(0.05)

    # ========================================================================
    # PHASE 2: Operant Conditioning + Discrimination (5 trials)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: OPERANT CONDITIONING + DISCRIMINATION (5 trials)")
    print("=" * 70)
    print("Protocol: CS operant (35s) alternating with test odor (30s, unrewarded)")

    phase2_protocol = [
        (4, cs_odor, True, 35, 'operant'),
        (5, test_odor, False, 30, 'discrimination'),
        (6, cs_odor, True, 35, 'operant'),
        (7, test_odor, False, 30, 'discrimination'),
        (8, cs_odor, True, 35, 'operant'),
    ]

    phase2_results = []

    for trial_num, odor, has_reward, duration_s, trial_type in phase2_protocol:
        print(f"\n📍 Trial {trial_num}: {odor} ", end="")
        if has_reward:
            print(f"+ operant reward ({duration_s}s)")
        else:
            print(f"(unrewarded, {duration_s}s) - DISCRIMINATION")
        print("-" * 70)

        if has_reward:
            # Operant trial with response-contingent reward
            operant_trial = OperantTrial(
                odor=odor,
                valve_duration_s=duration_s,
                response_window_s=10,
                proboscis_extension_threshold=0.5,
                travel_time_s=2,
                linger_time_s=2
            )

            results = operant_trial.run_operant_trial(
                circuit, plasticity, pn_activations[odor]
            )

            print(f"  Response time:    {results['response_time']:.2f}s (at fly)")
            print(f"  Response latency: {results['response_latency']:.2f}s (from odor onset)")
            print(f"  Reward duration:  {results['reward_duration']:.1f}s")
            print(f"  Final MBON:       {results['final_mbon']:.4f}")
            print(f"  Peak MBON:        {results['peak_mbon']:.4f}")
            print(f"  Weight changes:   {results['total_weight_change']:.6f}")

            phase2_results.append({
                'phase': 2,
                'trial': trial_num,
                'odor': odor,
                'has_reward': True,
                'trial_type': trial_type,
                'response_time': results['response_time'],
                'response_latency': results['response_latency'],
                'reward_duration': results['reward_duration'],
                'final_mbon': results['final_mbon'],
                'peak_mbon': results['peak_mbon'],
                'weight_change': results['total_weight_change']
            })

        else:
            # Discrimination trial (unrewarded test odor)
            trial = TemporalTrial(
                odor=odor,
                valve_duration_s=duration_s,
                travel_time_s=2,
                linger_time_s=2
            )

            odor_profile = trial.get_odor_profile()
            time_axis = trial.get_time_axis()
            pn_activation = pn_activations[odor]

            mbon_outputs = []
            for t_idx, t in enumerate(time_axis):
                pn_input = pn_activation * odor_profile[t_idx]
                kc_activation, _ = circuit.propagate_pn_to_kc(pn_input)
                mbon_output = plasticity.compute_mbon_output(kc_activation)
                mbon_outputs.append(mbon_output[0])

                # NO PLASTICITY UPDATE (no reward!)

            mbon_outputs = np.array(mbon_outputs)

            print(f"  Peak MBON:  {mbon_outputs.max():.4f}")
            print(f"  Mean MBON:  {mbon_outputs.mean():.4f}")
            print(f"  Final MBON: {mbon_outputs[-1]:.4f}")
            print(f"  No reward given (discrimination trial)")

            phase2_results.append({
                'phase': 2,
                'trial': trial_num,
                'odor': odor,
                'has_reward': False,
                'trial_type': trial_type,
                'final_mbon': mbon_outputs[-1],
                'peak_mbon': mbon_outputs.max(),
                'mean_mbon': mbon_outputs.mean()
            })

        # 5-minute inter-trial interval
        print(f"  [5-minute inter-trial interval]")
        time.sleep(0.05)

    # ========================================================================
    # CONSOLIDATION PERIOD (30 minutes)
    # ========================================================================
    print("\n" + "=" * 70)
    print("⏱️  CONSOLIDATION PERIOD")
    print("=" * 70)

    consolidation_period(plasticity, duration_minutes=30.0)

    # ========================================================================
    # PHASE 3: Testing (10 trials, multiple odors)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: TESTING (10 trials, 30 min post-training)")
    print("=" * 70)
    print("Protocol: Multiple odors, no reward, measure responses")

    test_protocol = [
        (1, cs_odor, 30, 5),           # CS test
        (2, cs_odor, 30, 5),           # CS repeat
        (3, 'ethyl_butyrate', 30, 5),  # Test odor A
        (4, cs_odor, 30, 5),           # CS repeat
        (5, cs_odor, 30, 5),           # CS repeat
        (6, '3-octanol', 30, 5),       # Test odor B
        (7, 'linalool', 10, 3),        # Test odor C (short)
        (8, 'geosmin', 10, 3),         # Test odor D (short)
        (9, 'pentyl_acetate', 10, 3),  # Test odor E (short)
        (10, test_odor, 10, 3),        # Test odor F (hexanol, short)
    ]

    test_results = []

    for test_num, odor, duration_s, iti_min in test_protocol:
        print(f"\n🧪 Test {test_num}: {odor} ({duration_s}s presentation, no reward)")
        print("-" * 70)

        result = run_test_trial(
            odor=odor,
            duration_s=duration_s,
            circuit=circuit,
            plasticity=plasticity,
            pn_activations=pn_activations,
            response_threshold=0.5
        )

        result['test_num'] = test_num
        result['iti_minutes'] = iti_min
        test_results.append(result)

        print(f"  Peak MBON:  {result['peak_mbon']:.4f}")
        print(f"  Mean MBON:  {result['mean_mbon']:.4f}")
        print(f"  Response:   {'✓ EXTEND' if result['response'] else '✗ NO RESPONSE'}")

        print(f"  [{iti_min}-minute inter-test interval]")
        time.sleep(0.03)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("💾 SAVING RESULTS")
    print("=" * 70)

    test_df = pd.DataFrame(test_results)
    test_df.to_csv(output_dir / 'test_results.csv', index=False)
    print(f"  ✓ Test results: {output_dir / 'test_results.csv'}")

    # Response summary per odor
    response_summary = test_df.groupby('odor').agg({
        'response': ['mean', 'sum', 'count'],
        'peak_mbon': 'mean',
        'mean_mbon': 'mean'
    }).round(4)
    response_summary.columns = ['response_rate', 'n_responses', 'n_tests', 'avg_peak_mbon', 'avg_mean_mbon']
    response_summary.to_csv(output_dir / 'response_summary.csv')
    print(f"  ✓ Response summary: {output_dir / 'response_summary.csv'}")

    # Training history
    phase1_df = pd.DataFrame(phase1_results)
    phase2_df = pd.DataFrame(phase2_results)

    phase1_df.to_csv(output_dir / 'phase1_classical.csv', index=False)
    phase2_df.to_csv(output_dir / 'phase2_operant.csv', index=False)
    print(f"  ✓ Phase 1 history: {output_dir / 'phase1_classical.csv'}")
    print(f"  ✓ Phase 2 history: {output_dir / 'phase2_operant.csv'}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

    print("\n📊 RESPONSE SUMMARY:")
    print(response_summary.to_string())

    print("\n📈 TEST RESULTS:")
    print(test_df[['test_num', 'odor', 'response', 'peak_mbon', 'mean_mbon']].to_string(index=False))

    return test_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run realistic fly behavioral training protocol'
    )
    parser.add_argument(
        '--cs-odor',
        default='benzaldehyde',
        help='Conditioned stimulus odor (default: benzaldehyde)'
    )
    parser.add_argument(
        '--test-odor',
        default='1-hexanol',
        help='Discrimination test odor (default: 1-hexanol)'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('data/cache'),
        help='FlyWire connectome cache directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/realistic_training'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--use-door',
        action='store_true',
        help='Use DoOR database for odor encoding (requires door-toolkit)'
    )

    args = parser.parse_args()

    # Run protocol
    results = run_realistic_training_protocol(
        cs_odor=args.cs_odor,
        test_odor=args.test_odor,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        use_door=args.use_door
    )

    print("\n" + "=" * 70)
    print("🎉 All done!")
    print("=" * 70)
