#!/usr/bin/env python
"""
Experiment 7: GABA Veto Gate Explains Odor-Specific Learning Failure

This experiment tests the hypothesis that benzaldehyde training failure
(despite sugar reward) is caused by stronger GABA inhibitory pathway activation
compared to OR7a odor.

Hypothesis:
    Benzaldehyde activates GABA inhibitory pathway more than OR7a,
    suppressing sugar reward signal and preventing learning.

Test Design:
    1. OR7a + sugar (low GABA) → Learning succeeds
    2. Benzaldehyde + sugar (high GABA) → Learning fails
    3. Benzaldehyde + sugar + GABA ablation → Learning recovers

Expected Results:
    - OR7a: veto_signal < 0.3 → RPE strong → Learning ✅
    - Benzaldehyde: veto_signal > 0.7 → RPE weak → No learning ❌
    - Benzaldehyde + GABA ablation: veto_signal = 0 → Learning restored ✅

Reference:
    Based on experimental data showing OR7a learns but benzaldehyde doesn't,
    despite both receiving sugar reward.

Usage:
    python scripts/experiments/experiment_7_gaba_veto_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pgcn.models.taste_circuit import TasteCircuit


def simulate_odor_dependent_gaba_activation(
    odor_type: str, sugar_input: torch.Tensor, gaba_modulation: float = 1.0
) -> torch.Tensor:
    """Simulate how different odors modulate GABA pathway activation.

    Parameters
    ----------
    odor_type : str
        'OR7a' or 'benzaldehyde'
    sugar_input : torch.Tensor
        Base sugar GRN activation, shape (batch, n_grns)
    gaba_modulation : float
        GABA pathway modulation factor (1.0 = normal, 2.0 = doubled, 0.0 = ablated)

    Returns
    -------
    torch.Tensor
        Modulated sugar input that affects GABA pathway differently

    Notes
    -----
    This simulates the hypothesis that benzaldehyde somehow enhances
    GABA pathway responsivity, either through:
    - Cross-modal interactions (odor → taste)
    - State-dependent gating
    - Neuromodulatory effects
    """
    if odor_type == "OR7a":
        # OR7a: Normal GABA activation
        gaba_enhanced_input = sugar_input.clone()
    elif odor_type == "benzaldehyde":
        # Benzaldehyde: Enhanced GABA activation (hypothesis)
        # Add a bias to GABA-preferring GRNs (first 1/3 of GRNs)
        gaba_enhanced_input = sugar_input.clone()
        n_grns = sugar_input.shape[1]
        gaba_grns = n_grns // 3  # First 1/3 preferentially drive GABA
        gaba_enhanced_input[:, :gaba_grns] *= 1.5 * gaba_modulation
    else:
        raise ValueError(f"Unknown odor type: {odor_type}")

    return gaba_enhanced_input


def compute_reward_prediction_error(
    sez_pn_activity: torch.Tensor,
    actual_reward: float,
    predicted_reward: float,
    veto_signal: torch.Tensor,
) -> torch.Tensor:
    """Compute reward prediction error gated by GABA veto signal.

    Parameters
    ----------
    sez_pn_activity : torch.Tensor
        SEZ-PN projection neuron activity, shape (batch, n_sez_pns)
    actual_reward : float
        Actual reward magnitude (0-1)
    predicted_reward : float
        Predicted reward from previous learning
    veto_signal : torch.Tensor
        GABA veto signal strength, shape (batch,)

    Returns
    -------
    torch.Tensor
        Gated reward prediction error, shape (batch,)

    Notes
    -----
    The veto gate suppresses learning by reducing the effective RPE.
    High veto (>0.7) → weak RPE → no learning
    Low veto (<0.3) → strong RPE → learning occurs
    """
    # Raw RPE (actual - predicted)
    rpe_raw = torch.tensor([actual_reward - predicted_reward])
    rpe_raw = rpe_raw.expand(veto_signal.shape[0])

    # Gate RPE by veto signal (sigmoid suppression)
    # High veto → sigmoid approaches 0 → RPE suppressed
    veto_gate = torch.sigmoid(-veto_signal)  # High veto → low gate
    rpe_gated = rpe_raw * veto_gate

    return rpe_gated


def run_condition(
    taste_circuit: TasteCircuit,
    condition_name: str,
    odor_type: str,
    gaba_gain: float,
    sugar_input: torch.Tensor,
    n_trials: int = 100,
) -> dict:
    """Run one experimental condition.

    Parameters
    ----------
    taste_circuit : TasteCircuit
        Taste circuit model
    condition_name : str
        Name of condition for logging
    odor_type : str
        'OR7a' or 'benzaldehyde'
    gaba_gain : float
        GABA inhibition strength (0.0 = ablated, 1.0 = normal, 2.0 = enhanced)
    sugar_input : torch.Tensor
        Base sugar GRN activation
    n_trials : int
        Number of training trials

    Returns
    -------
    dict
        Results containing metrics across trials
    """
    print(f"\n[Running: {condition_name}]")
    print(f"  Odor: {odor_type}, GABA gain: {gaba_gain:.2f}")

    # Set GABA gain
    taste_circuit.gaba_gain.data = torch.tensor(gaba_gain)

    # Simulate odor-dependent modulation
    modulated_sugar = simulate_odor_dependent_gaba_activation(odor_type, sugar_input, gaba_gain)

    # Track metrics across trials
    veto_signals = []
    rpes = []
    learning_rates = []
    predicted_reward = 0.0  # Initial prediction

    for trial in range(n_trials):
        # Forward pass through taste circuit
        with torch.no_grad():
            taste_output = taste_circuit(modulated_sugar)
            sez_pn = taste_output['sez_pn_activity']
            ach_ln = taste_output['ach_ln_activity']
            gaba_ln = taste_output['gaba_ln_activity']
            veto_signal = taste_output['veto_signal']

        # Compute RPE (gated by veto)
        actual_reward = 1.0  # Sugar is present
        rpe = compute_reward_prediction_error(sez_pn, actual_reward, predicted_reward, veto_signal)

        # Update prediction (simple delta rule)
        learning_rate_trial = 0.1 * abs(rpe.item())
        predicted_reward += learning_rate_trial * rpe.item()
        predicted_reward = np.clip(predicted_reward, 0.0, 1.0)

        # Store metrics
        veto_signals.append(veto_signal.item())
        rpes.append(rpe.item())
        learning_rates.append(learning_rate_trial)

    # Compute summary statistics
    results = {
        "condition": condition_name,
        "odor_type": odor_type,
        "gaba_gain": gaba_gain,
        "mean_veto": float(np.mean(veto_signals)),
        "std_veto": float(np.std(veto_signals)),
        "mean_rpe": float(np.mean(rpes)),
        "std_rpe": float(np.std(rpes)),
        "total_learning": float(np.sum(learning_rates)),
        "final_prediction": float(predicted_reward),
        "veto_signals": veto_signals,
        "rpes": rpes,
        "learning_rates": learning_rates,
    }

    print(f"  Mean veto signal: {results['mean_veto']:.3f} ± {results['std_veto']:.3f}")
    print(f"  Mean RPE: {results['mean_rpe']:.3f} ± {results['std_rpe']:.3f}")
    print(f"  Total learning: {results['total_learning']:.2f}")
    print(f"  Final prediction: {results['final_prediction']:.3f}")

    # Determine if learning succeeded (arbitrary threshold)
    learning_threshold = 5.0  # Cumulative learning required for "success"
    results["learning_success"] = results["total_learning"] > learning_threshold

    if results["learning_success"]:
        print(f"  ✅ LEARNING SUCCESS (total > {learning_threshold})")
    else:
        print(f"  ❌ LEARNING FAILURE (total < {learning_threshold})")

    return results


def plot_results(all_results: list[dict], output_dir: Path) -> None:
    """Generate plots comparing all conditions.

    Parameters
    ----------
    all_results : list[dict]
        Results from all conditions
    output_dir : Path
        Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Experiment 7: GABA Veto Gate Hypothesis Test", fontsize=16, fontweight="bold")

    conditions = [r["condition"] for r in all_results]
    colors = ["green", "red", "blue"]

    # Plot 1: Mean veto signal
    ax = axes[0, 0]
    veto_means = [r["mean_veto"] for r in all_results]
    veto_stds = [r["std_veto"] for r in all_results]
    bars = ax.bar(range(len(conditions)), veto_means, yerr=veto_stds, color=colors, alpha=0.7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.replace(" + Sugar", "") for c in conditions], rotation=15, ha="right")
    ax.set_ylabel("Mean GABA Veto Signal")
    ax.set_title("GABA Inhibition Strength")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Mean RPE
    ax = axes[0, 1]
    rpe_means = [r["mean_rpe"] for r in all_results]
    rpe_stds = [r["std_rpe"] for r in all_results]
    ax.bar(range(len(conditions)), rpe_means, yerr=rpe_stds, color=colors, alpha=0.7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.replace(" + Sugar", "") for c in conditions], rotation=15, ha="right")
    ax.set_ylabel("Mean Reward Prediction Error")
    ax.set_title("Learning Signal Strength")
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Total learning
    ax = axes[0, 2]
    learning_totals = [r["total_learning"] for r in all_results]
    bars = ax.bar(range(len(conditions)), learning_totals, color=colors, alpha=0.7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.replace(" + Sugar", "") for c in conditions], rotation=15, ha="right")
    ax.set_ylabel("Cumulative Learning")
    ax.set_title("Learning Outcome")
    ax.axhline(y=5.0, color="gray", linestyle="--", linewidth=1, label="Success threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add success/failure labels
    for i, (bar, result) in enumerate(zip(bars, all_results)):
        label = "✅" if result["learning_success"] else "❌"
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            label,
            ha="center",
            va="bottom",
            fontsize=16,
        )

    # Plot 4: Veto signal over trials
    ax = axes[1, 0]
    for i, result in enumerate(all_results):
        ax.plot(result["veto_signals"], label=conditions[i], color=colors[i], alpha=0.7)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Veto Signal")
    ax.set_title("Veto Signal Time Course")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 5: RPE over trials
    ax = axes[1, 1]
    for i, result in enumerate(all_results):
        ax.plot(result["rpes"], label=conditions[i], color=colors[i], alpha=0.7)
    ax.set_xlabel("Trial")
    ax.set_ylabel("RPE")
    ax.set_title("RPE Time Course")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 6: Cumulative learning over trials
    ax = axes[1, 2]
    for i, result in enumerate(all_results):
        cumulative = np.cumsum(result["learning_rates"])
        ax.plot(cumulative, label=conditions[i], color=colors[i], alpha=0.7)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative Learning")
    ax.set_title("Learning Progress")
    ax.axhline(y=5.0, color="gray", linestyle="--", linewidth=1, label="Threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save plots
    plot_file = output_dir / "experiment_7_gaba_veto_gate.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved plot: {plot_file}")

    plot_file_png = output_dir / "experiment_7_gaba_veto_gate.png"
    plt.savefig(plot_file_png, dpi=150, bbox_inches="tight")
    print(f"✓ Saved plot: {plot_file_png}")

    plt.close()


def main():
    """Run Experiment 7: GABA veto gate hypothesis test."""
    print("=" * 70)
    print("EXPERIMENT 7: GABA VETO GATE HYPOTHESIS")
    print("=" * 70)
    print("\nHypothesis:")
    print("  Benzaldehyde activates GABA pathway more than OR7a,")
    print("  suppressing sugar reward signal and preventing learning.")
    print("\nTest design:")
    print("  1. OR7a + sugar → low GABA → learning succeeds")
    print("  2. Benzaldehyde + sugar → high GABA → learning fails")
    print("  3. Benzaldehyde + GABA ablation → learning recovers")
    print("=" * 70)

    # Initialize taste circuit
    try:
        taste_circuit = TasteCircuit(
            data_dir=Path("data/cache"),
            gaba_veto_mode="direct",  # GABA-LN → SEZ-PN inhibition
            gaba_gain=1.0,  # Will be modified per condition
            use_synapse_weights=True,
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease run extraction first:")
        print("  python scripts/extract_from_paper_data.py --mode appetitive")
        return 1

    # Print circuit statistics
    stats = taste_circuit.get_synapse_statistics()
    print(f"\nTaste Circuit Statistics:")
    print(f"  GRN→PN connections: {stats['grn_to_pn_connections']}")
    print(f"  GRN→ACh connections: {stats['grn_to_ach_connections']}")
    print(f"  GRN→GABA connections: {stats['grn_to_gaba_connections']}")
    print(f"  Mean GRN→PN weight: {stats['grn_to_pn_mean_weight']:.4f}")
    print(f"  GABA veto mode: {stats['gaba_veto_mode']}")

    # Define base sugar input (moderate activation)
    batch_size = 1
    sugar_reward = torch.ones(batch_size, taste_circuit.n_grns) * 0.5

    # Run 3 experimental conditions
    all_results = []

    # Condition 1: OR7a + Sugar (normal GABA)
    results_or7a = run_condition(
        taste_circuit=taste_circuit,
        condition_name="OR7a + Sugar",
        odor_type="OR7a",
        gaba_gain=1.0,
        sugar_input=sugar_reward,
        n_trials=100,
    )
    all_results.append(results_or7a)

    # Condition 2: Benzaldehyde + Sugar (enhanced GABA)
    results_benz = run_condition(
        taste_circuit=taste_circuit,
        condition_name="Benzaldehyde + Sugar",
        odor_type="benzaldehyde",
        gaba_gain=2.0,  # HYPOTHESIS: Benzaldehyde enhances GABA
        sugar_input=sugar_reward,
        n_trials=100,
    )
    all_results.append(results_benz)

    # Condition 3: Benzaldehyde + Sugar + GABA ablation
    results_ablated = run_condition(
        taste_circuit=taste_circuit,
        condition_name="Benzaldehyde + GABA ablated",
        odor_type="benzaldehyde",
        gaba_gain=0.0,  # Ablate GABA pathway
        sugar_input=sugar_reward,
        n_trials=100,
    )
    all_results.append(results_ablated)

    # Save results to CSV
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame(
        [
            {
                "condition": r["condition"],
                "odor_type": r["odor_type"],
                "gaba_gain": r["gaba_gain"],
                "mean_veto": r["mean_veto"],
                "mean_rpe": r["mean_rpe"],
                "total_learning": r["total_learning"],
                "final_prediction": r["final_prediction"],
                "learning_success": r["learning_success"],
            }
            for r in all_results
        ]
    )

    results_file = output_dir / "experiment_7_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Saved results: {results_file}")

    # Generate plots
    plot_results(all_results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 7 SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    # Check hypothesis
    or7a_success = results_or7a["learning_success"]
    benz_failure = not results_benz["learning_success"]
    ablated_recovery = results_ablated["learning_success"]

    print("\nHypothesis Validation:")
    print(f"  1. OR7a learns: {or7a_success} {'✅' if or7a_success else '❌'}")
    print(f"  2. Benzaldehyde fails: {benz_failure} {'✅' if benz_failure else '❌'}")
    print(
        f"  3. GABA ablation recovers: {ablated_recovery} {'✅' if ablated_recovery else '❌'}"
    )

    if or7a_success and benz_failure and ablated_recovery:
        print("\n✅ HYPOTHESIS CONFIRMED: GABA veto gate explains benzaldehyde failure!")
    else:
        print("\n❌ HYPOTHESIS REJECTED: GABA veto gate does not fully explain pattern")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
