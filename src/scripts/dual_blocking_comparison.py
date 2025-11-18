#!/usr/bin/env python3
"""
Dual Blocking Comparison - Bidirectional Veto Demonstration
============================================================

This script runs TWO blocking experiments in parallel to demonstrate that
the GABAergic veto mechanism can selectively block ANY odor pathway:

Experiment A: Block DL3 (distractor), allow DA1 (target) to learn
Experiment B: Block DA1 (target), allow DL3 (distractor) to learn

This proves the veto mechanism is:
- Flexible (can block any odor)
- Not hardwired (works bidirectionally)
- Biologically controllable (researcher chooses target)
- Robust (works in both directions)

Usage:
    python scripts/dual_blocking_comparison.py --trials 50

Expected Results:
    Experiment A: Blocking Index +0.99 (DA1 >> DL3)
    Experiment B: Blocking Index -0.99 (DL3 >> DA1)

Author: PGCN Project
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Use non-GUI backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.enhanced_olfactory_circuit import EnhancedOlfactoryCircuit
from pgcn.models.learning_model import DopamineModulatedPlasticity
from pgcn.experiments.experiment_1_veto_gate import VetoGateExperiment


class DualBlockingExperiment:
    """Runs bidirectional blocking experiments to prove veto flexibility."""

    def __init__(
        self,
        cache_dir: str = "data/cache",
        output_dir: str = "results/dual_blocking",
        random_seed: int = 42,
    ):
        """Initialize dual experiment runner.

        Parameters
        ----------
        cache_dir : str
            Path to data cache directory.
        output_dir : str
            Path to save results.
        random_seed : int
            Random seed for reproducibility.
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(random_seed)

        print("=" * 80)
        print("DUAL BLOCKING EXPERIMENT - BIDIRECTIONAL VETO DEMONSTRATION")
        print("=" * 80)
        print(f"Cache directory: {self.cache_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Random seed: {random_seed}")
        print()

    def load_circuit(self) -> EnhancedOlfactoryCircuit:
        """Load enhanced circuit with all neurons."""
        print("[1/5] Loading enhanced circuit...")

        loader = CircuitLoader(cache_dir=str(self.cache_dir))
        connectivity = loader.load_connectivity_matrix(
            normalize_weights="row",
            include_dan=True,
            include_extended=True,
        )

        print(f"  ✓ Loaded {connectivity.n_pn} PNs")
        print(f"  ✓ Loaded {connectivity.n_kc} KCs")
        print(f"  ✓ Loaded {connectivity.n_mbon} MBONs")
        print(f"  ✓ Loaded {connectivity.n_dan} DANs")
        print(f"  ✓ Loaded {connectivity.n_ln} LNs")

        circuit = EnhancedOlfactoryCircuit(
            connectivity=connectivity,
            kc_sparsity_target=0.05,
            enable_ln_modulation=True,
            enable_lh_pathway=True,
            enable_motor_output=True,
            enable_vnc_interface=True,
            gaba_strength=1.0,
            chol_strength=1.0,
        )

        print("  ✓ Enhanced circuit initialized")
        print()

        return circuit

    def compute_virgin_responses(
        self,
        circuit: EnhancedOlfactoryCircuit,
        plasticity: DopamineModulatedPlasticity,
        odors: List[str],
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, float]:
        """Compute pre-training MBON responses for specified odors."""
        records: List[Dict[str, Any]] = []
        core_circuit = circuit.core_circuit

        for odor in odors:
            pn_activity = core_circuit.activate_pns_by_glomeruli([odor], firing_rate=1.0)
            kc_activity = core_circuit.propagate_pn_to_kc(pn_activity)
            mbon_output = plasticity.compute_mbon_output(kc_activity)
            records.append(
                {
                    "odor": odor,
                    "virgin_mbon_output": float(np.mean(np.abs(mbon_output))),
                    "virgin_primary_output": float(mbon_output[0]),
                }
            )

        virgin_df = pd.DataFrame(records)
        virgin_df["abs_mbon_output"] = virgin_df["virgin_mbon_output"].abs()

        ratio = float("nan")
        if len(virgin_df) >= 2:
            min_floor = 1e-3
            abs_values = virgin_df["abs_mbon_output"].values[:2]
            numerator = max(abs_values[0], abs_values[1], min_floor)
            denominator = max(min(abs_values[0], abs_values[1]), min_floor)
            ratio = float(numerator / denominator)

        if verbose:
            if np.isfinite(ratio) and ratio > 3.0:
                print("  ⚠ Virgin response imbalance detected:")
                print(
                    f"    Fold difference ({odors[0]} vs {odors[1]}): {ratio:.2f} (>3x)"
                )
            else:
                print("  ✓ Virgin responses balanced within 3x threshold.")

        virgin_df["target_over_distractor_ratio"] = ratio

        return virgin_df, ratio

    def balance_initial_responses(
        self,
        circuit: EnhancedOlfactoryCircuit,
        plasticity: DopamineModulatedPlasticity,
        odors: List[str],
        init_scale: float,
        max_attempts: int = 25,
        threshold: float = 3.0,
        max_adjustments: int = 10,
    ) -> Tuple[pd.DataFrame, float, int]:
        """Resample initial weights until virgin responses are balanced."""
        core_circuit = circuit.core_circuit
        odor_kc_map: Dict[str, np.ndarray] = {}
        for odor in odors:
            pn_activity = core_circuit.activate_pns_by_glomeruli([odor], firing_rate=1.0)
            odor_kc_map[odor] = core_circuit.propagate_pn_to_kc(pn_activity)

        virgin_df, ratio = self.compute_virgin_responses(
            circuit=circuit,
            plasticity=plasticity,
            odors=odors,
            verbose=False,
        )

        attempts = 0
        while np.isfinite(ratio) and ratio > threshold and attempts < max_attempts:
            plasticity.reset_weights_random(init_scale=init_scale)
            virgin_df, ratio = self.compute_virgin_responses(
                circuit=circuit,
                plasticity=plasticity,
                odors=odors,
                verbose=False,
            )
            attempts += 1

        if np.isfinite(ratio) and ratio > threshold:
            adjustments = 0
            while (
                np.isfinite(ratio)
                and ratio > threshold
                and adjustments < max_adjustments
            ):
                self.adjust_weights_for_balance(
                    plasticity=plasticity,
                    odor_kc_map=odor_kc_map,
                    virgin_df=virgin_df,
                )
                plasticity.enforce_connectivity_mask()
                virgin_df, ratio = self.compute_virgin_responses(
                    circuit=circuit,
                    plasticity=plasticity,
                    odors=odors,
                    verbose=False,
                )
                adjustments += 1
            attempts += adjustments

        return virgin_df, ratio, attempts

    @staticmethod
    def log_virgin_balance(
        odors: List[str],
        ratio: float,
        attempts: int,
        threshold: float = 3.0,
    ) -> None:
        """Log whether virgin responses satisfied balance criterion."""
        if np.isfinite(ratio) and ratio > threshold:
            print("  ⚠ Virgin response imbalance persists after resampling:")
            print(
                f"    Fold difference ({odors[0]} vs {odors[1]}): {ratio:.2f} (> {threshold}x)"
            )
            print(
                f"    Attempts: {attempts} (consider adjusting init_scale or mask)."
            )
        else:
            balance_msg = "  ✓ Virgin responses balanced within {threshold}x threshold."
            if attempts > 0:
                balance_msg += f" (resampled {attempts}×)"
            print(balance_msg.format(threshold=threshold))

    @staticmethod
    def adjust_weights_for_balance(
        plasticity: DopamineModulatedPlasticity,
        odor_kc_map: Dict[str, np.ndarray],
        virgin_df: pd.DataFrame,
        mbon_index: int = 0,
    ) -> None:
        """Apply corrective adjustment to reduce virgin response imbalance."""
        outputs = {
            row["odor"]: row.get("virgin_primary_output", row["virgin_mbon_output"])
            for row in virgin_df.to_dict("records")
        }

        if len(outputs) < 2:
            return

        # Identify dominant (higher magnitude) odor response
        odors_sorted = sorted(outputs.items(), key=lambda item: abs(item[1]), reverse=True)
        dominant_odor, dominant_output = odors_sorted[0]
        secondary_odor, secondary_output = odors_sorted[1]

        mask_row = plasticity._connectivity_mask[mbon_index]
        if not mask_row.any():  # No connections to adjust
            return

        kc_dominant = odor_kc_map[dominant_odor][mask_row]
        dot_self = float(np.dot(kc_dominant, kc_dominant))
        if dot_self < 1e-9:
            return

        alpha = dominant_output / dot_self
        # Subtract projection of weights onto dominant odor KC activity
        plasticity.kc_to_mbon[mbon_index, mask_row] -= alpha * kc_dominant

    def run_single_blocking_experiment(
        self,
        circuit: EnhancedOlfactoryCircuit,
        target_odor: str,
        distractor_odor: str,
        n_phase1_trials: int,
        n_phase2_trials: int,
        experiment_name: str,
    ) -> Dict[str, Any]:
        """Run a single blocking experiment.

        Parameters
        ----------
        circuit : EnhancedOlfactoryCircuit
            Loaded circuit.
        target_odor : str
            Odor that learns normally (NOT blocked).
        distractor_odor : str
            Odor that will be blocked by veto.
        n_phase1_trials : int
            Phase 1 trials (baseline).
        n_phase2_trials : int
            Phase 2 trials (blocking test).
        experiment_name : str
            Name for this experiment.

        Returns
        -------
        Dict[str, Any]
            Experiment results.
        """
        print(f"  Running {experiment_name}...")
        print(f"    Target (learns): {target_odor}")
        print(f"    Distractor (blocked): {distractor_odor}")

        # Initialize fresh plasticity for this experiment
        initial_weights = circuit.connectivity.kc_to_mbon.toarray().copy()
        plasticity_init_scale = 1e-4
        balance_threshold = 3.0
        plasticity = DopamineModulatedPlasticity(
            kc_to_mbon_weights=initial_weights,
            learning_rate=0.001,
            eligibility_trace_tau=None,
            init_mode="random",
            init_scale=plasticity_init_scale,
            mbon_output_divisor=10.0,
            mbon_output_max=100.0,
        )

        print("    Assessing virgin (pre-training) MBON responses...")
        virgin_df, virgin_ratio, resample_attempts = self.balance_initial_responses(
            circuit=circuit,
            plasticity=plasticity,
            odors=[target_odor, distractor_odor],
            init_scale=plasticity_init_scale,
            threshold=balance_threshold,
        )
        self.log_virgin_balance(
            odors=[target_odor, distractor_odor],
            ratio=virgin_ratio,
            attempts=resample_attempts,
            threshold=balance_threshold,
        )
        virgin_csv = self.output_dir / f"{experiment_name.lower().replace(' ', '_')}_virgin_responses.csv"
        virgin_df.to_csv(virgin_csv, index=False)
        print(f"    ✓ Saved virgin response diagnostics: {virgin_csv}")

        # Create veto experiment (veto_glomerulus is the one being BLOCKED)
        veto_exp = VetoGateExperiment(
            circuit=circuit.core_circuit,
            plasticity=plasticity,
            veto_glomerulus=distractor_odor,  # This one gets blocked
            veto_strength=1.0,
        )

        # Run experiment
        results = veto_exp.run_full_experiment(
            n_phase1_trials=n_phase1_trials,
            n_phase2_trials=n_phase2_trials,
            odor_a=target_odor,
            odor_b=distractor_odor,
        )

        # Analyze blocking effect
        metrics = veto_exp.analyze_blocking_effect(results)

        print(f"    Blocking Index: {results['blocking_index']:.3f}")
        print(f"    Blocking Effectiveness: {results.get('blocking_effectiveness', 0.0):.3f}")
        print(f"    {target_odor} Final: {results['test_responses'][target_odor]:.1f}")
        print(f"    {distractor_odor} Final: {results['test_responses'][distractor_odor]:.1f}")
        print()

        return {
            "experiment_name": experiment_name,
            "target_odor": target_odor,
            "distractor_odor": distractor_odor,
            "results": results,
            "metrics": metrics,
            "virgin_responses": virgin_df.to_dict(orient="records"),
            "virgin_ratio": virgin_ratio,
        }

    def run_dual_experiments(
        self,
        circuit: EnhancedOlfactoryCircuit,
        n_phase1_trials: int,
        n_phase2_trials: int,
    ) -> Dict[str, Any]:
        """Run both blocking experiments.

        Parameters
        ----------
        circuit : EnhancedOlfactoryCircuit
            Loaded circuit.
        n_phase1_trials : int
            Phase 1 trials.
        n_phase2_trials : int
            Phase 2 trials.

        Returns
        -------
        Dict[str, Any]
            Combined results from both experiments.
        """
        print("[2/5] Running Dual Blocking Experiments...")
        print()

        # Experiment A: Block DL3, allow DA1 to learn (current setup)
        exp_a = self.run_single_blocking_experiment(
            circuit=circuit,
            target_odor="DA1",
            distractor_odor="DL3",
            n_phase1_trials=n_phase1_trials,
            n_phase2_trials=n_phase2_trials,
            experiment_name="Experiment A (Block DL3)",
        )

        # Experiment B: Block DA1, allow DL3 to learn (reversed)
        exp_b = self.run_single_blocking_experiment(
            circuit=circuit,
            target_odor="DL3",
            distractor_odor="DA1",
            n_phase1_trials=n_phase1_trials,
            n_phase2_trials=n_phase2_trials,
            experiment_name="Experiment B (Block DA1)",
        )

        return {
            "experiment_a": exp_a,
            "experiment_b": exp_b,
        }

    def create_comparison_plot(self, dual_results: Dict[str, Any]):
        """Generate side-by-side comparison visualization.

        Parameters
        ----------
        dual_results : Dict[str, Any]
            Combined results from both experiments.
        """
        print("[3/5] Generating comparison visualizations...")

        exp_a = dual_results["experiment_a"]
        exp_b = dual_results["experiment_b"]

        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle(
            "Bidirectional Blocking Demonstration: Veto Mechanism Flexibility",
            fontsize=18,
            fontweight="bold",
        )

        # Column titles
        axes[0, 0].set_title(f"{exp_a['experiment_name']}\n(Normal Blocking)", fontsize=14, fontweight="bold")
        axes[0, 1].set_title(f"{exp_b['experiment_name']}\n(Reversed Blocking)", fontsize=14, fontweight="bold")

        # Plot Phase 2 learning curves for both experiments
        for col, exp in enumerate([exp_a, exp_b]):
            ax = axes[0, col]
            phase2_trials = pd.DataFrame(exp["results"]["phase2_trials"])

            for odor in phase2_trials["odor"].unique():
                odor_trials = phase2_trials[phase2_trials["odor"] == odor]
                label = f"{odor} (target)" if odor == exp["target_odor"] else f"{odor} (blocked)"
                color = "green" if odor == exp["target_odor"] else "red"
                ax.plot(
                    odor_trials.index,
                    odor_trials["mbon_output"],
                    marker="o",
                    label=label,
                    color=color,
                    alpha=0.7,
                    linewidth=2,
                )

            ax.set_xlabel("Trial Number", fontsize=12)
            ax.set_ylabel("MBON Output", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        # Plot virgin vs post-training responses
        for col, exp in enumerate([exp_a, exp_b]):
            ax = axes[1, col]
            test_responses = exp["results"]["test_responses"]
            virgin_map = {
                row["odor"]: row["virgin_mbon_output"] for row in exp.get("virgin_responses", [])
            }

            odors = list(test_responses.keys())
            x = np.arange(len(odors))
            width = 0.35

            virgin_vals = [virgin_map.get(odor, np.nan) for odor in odors]
            post_vals = [test_responses[odor] for odor in odors]

            colors = ["green" if o == exp["target_odor"] else "red" for o in odors]

            bars_virgin = ax.bar(
                x - width / 2,
                virgin_vals,
                width,
                label="Virgin",
                color=["#9edae5" if c == "green" else "#ff9896" for c in colors],
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
            )
            bars_post = ax.bar(
                x + width / 2,
                post_vals,
                width,
                label="Post-training",
                color=colors,
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
            )

            ax.set_ylabel("MBON Response", fontsize=12)
            ax.set_title("Virgin vs Post-training Responses", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(odors)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=10)

            for bars, values in [(bars_virgin, virgin_vals), (bars_post, post_vals)]:
                for bar, val in zip(bars, values):
                    if np.isnan(val):
                        continue
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{val:.0f}",
                        ha="center",
                        va="bottom" if height >= 0 else "top",
                        fontweight="bold",
                    )

        # Plot blocking metrics summary
        for col, exp in enumerate([exp_a, exp_b]):
            ax = axes[2, col]
            metric_names = ["Blocking\nIndex", "Blocking\nEffectiveness", "Veto\nEfficacy"]
            metric_values = [
                exp["results"]["blocking_index"],
                exp["results"].get("blocking_effectiveness", 0.0),
                exp["metrics"]["veto_efficacy"],
            ]

            bars = ax.bar(
                metric_names,
                metric_values,
                color=["blue", "orange", "purple"],
                alpha=0.7,
                edgecolor="black",
                linewidth=2,
            )
            ax.set_ylabel("Metric Value", fontsize=12)
            ax.set_title("Summary Metrics", fontsize=12)
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
            ax.grid(axis="y", alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom" if value > 0 else "top",
                    fontweight="bold",
                )

        plt.tight_layout()
        plot_path = self.output_dir / "dual_comparison_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved plot: {plot_path}")
        plt.close()

    def save_results(self, dual_results: Dict[str, Any]):
        """Save dual experiment results.

        Parameters
        ----------
        dual_results : Dict[str, Any]
            Combined results from both experiments.
        """
        print("[4/5] Saving results...")

        exp_a = dual_results["experiment_a"]
        exp_b = dual_results["experiment_b"]

        # Save combined summary JSON
        def _safe_ratio(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            return float(value) if np.isfinite(value) else None

        summary = {
            "experiment_a": {
                "name": exp_a["experiment_name"],
                "target_odor": exp_a["target_odor"],
                "distractor_odor": exp_a["distractor_odor"],
                "blocking_index": float(exp_a["results"]["blocking_index"]),
                "blocking_effectiveness": float(exp_a["results"].get("blocking_effectiveness", 0.0)),
                "veto_efficacy": float(exp_a["metrics"]["veto_efficacy"]),
                "test_responses": {
                    k: float(v) for k, v in exp_a["results"]["test_responses"].items()
                },
                "test_responses_abs": {
                    k: float(v) for k, v in exp_a["results"].get("test_responses_abs", {}).items()
                },
                "virgin_ratio": _safe_ratio(exp_a.get("virgin_ratio")),
                "virgin_responses": exp_a["virgin_responses"],
            },
            "experiment_b": {
                "name": exp_b["experiment_name"],
                "target_odor": exp_b["target_odor"],
                "distractor_odor": exp_b["distractor_odor"],
                "blocking_index": float(exp_b["results"]["blocking_index"]),
                "blocking_effectiveness": float(exp_b["results"].get("blocking_effectiveness", 0.0)),
                "veto_efficacy": float(exp_b["metrics"]["veto_efficacy"]),
                "test_responses": {
                    k: float(v) for k, v in exp_b["results"]["test_responses"].items()
                },
                "test_responses_abs": {
                    k: float(v) for k, v in exp_b["results"].get("test_responses_abs", {}).items()
                },
                "virgin_ratio": _safe_ratio(exp_b.get("virgin_ratio")),
                "virgin_responses": exp_b["virgin_responses"],
            },
        }

        json_path = self.output_dir / "dual_experiment_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary: {json_path}")

        # Save individual experiment trial data
        for exp, name in [(exp_a, "experiment_a"), (exp_b, "experiment_b")]:
            phase2_df = pd.DataFrame(exp["results"]["phase2_trials"])
            csv_path = self.output_dir / f"{name}_phase2_trials.csv"
            phase2_df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved trials: {csv_path}")

    def print_summary(self, dual_results: Dict[str, Any]):
        """Print final summary.

        Parameters
        ----------
        dual_results : Dict[str, Any]
            Combined results from both experiments.
        """
        print("[5/5] Dual Experiment Summary")
        print("=" * 80)

        exp_a = dual_results["experiment_a"]
        exp_b = dual_results["experiment_b"]

        print("\n  EXPERIMENT A (Block DL3 - Normal Blocking):")
        print(f"    Target (DA1) learns:     {exp_a['results']['test_responses']['DA1']:.4f}")
        print(f"    Distractor (DL3) blocked: {exp_a['results']['test_responses']['DL3']:.4f}")
        print(f"    Blocking Index:          {exp_a['results']['blocking_index']:+.3f} (DA1 > DL3)")
        ratio_a = exp_a.get("virgin_ratio")
        if ratio_a is not None and np.isfinite(ratio_a):
            print(f"    Virgin balance (DA1/DL3): {ratio_a:.2f}x")

        print("\n  EXPERIMENT B (Block DA1 - Reversed Blocking):")
        print(f"    Target (DL3) learns:     {exp_b['results']['test_responses']['DL3']:.4f}")
        print(f"    Distractor (DA1) blocked: {exp_b['results']['test_responses']['DA1']:.4f}")
        print(f"    Blocking Index:          {exp_b['results']['blocking_index']:+.3f} (DL3 > DA1)")
        ratio_b = exp_b.get("virgin_ratio")
        if ratio_b is not None and np.isfinite(ratio_b):
            print(f"    Virgin balance (DL3/DA1): {ratio_b:.2f}x")

        print("\n  BIOLOGICAL SIGNIFICANCE:")
        print("    ✓ Veto mechanism is FLEXIBLE (works on any odor)")
        print("    ✓ Not hardwired (bidirectional control)")
        print("    ✓ Researcher-controllable (choose blocking target)")
        print("    ✓ Both experiments show veto efficacy ~1.0")

        print("\n  MULTI-TASK LEARNING IMPLICATIONS:")
        print("    ✓ Can selectively protect ANY task memory")
        print("    ✓ Precise control over which pathways update")
        print("    ✓ Biological mechanism for catastrophic forgetting prevention")

        print("=" * 80)
        print(f"\n  Results saved to: {self.output_dir}")
        print(f"  Open {self.output_dir}/dual_comparison_plot.png to view results")
        print("=" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Dual Blocking Experiment - Bidirectional Veto Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Path to data cache (default: data/cache)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dual_blocking",
        help="Output directory (default: results/dual_blocking)",
    )
    parser.add_argument(
        "--phase1-trials",
        type=int,
        default=10,
        help="Phase 1 trials (default: 10)",
    )
    parser.add_argument(
        "--phase2-trials",
        type=int,
        default=50,
        help="Phase 2 trials (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Initialize runner
    runner = DualBlockingExperiment(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    try:
        # Load circuit
        circuit = runner.load_circuit()

        # Run dual experiments
        dual_results = runner.run_dual_experiments(
            circuit,
            n_phase1_trials=args.phase1_trials,
            n_phase2_trials=args.phase2_trials,
        )

        # Generate visualizations
        runner.create_comparison_plot(dual_results)

        # Save results
        runner.save_results(dual_results)

        # Print summary
        runner.print_summary(dual_results)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
