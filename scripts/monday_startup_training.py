#!/usr/bin/env python3
"""
Monday Startup Training Script - One-Command Experiment Execution
=================================================================

This script provides a one-command startup for Monday's blocking experiments.
It loads the full enhanced circuit with 13K+ neurons and runs Experiment 1
(GABAergic veto gate) to test pathway-specific plasticity blocking.

Usage:
    python scripts/monday_startup_training.py --experiment veto --trials 30

Expected Monday Outcome:
    Quantitative evidence for PN→KC pathway blocking hypothesis using
    real FlyWire neural populations with GABAergic veto mechanism.

Author: PGCN Project
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

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


class MondayExperimentRunner:
    """Orchestrates Monday blocking experiments with full reporting."""

    def __init__(
        self,
        cache_dir: str = "data/cache",
        output_dir: str = "results/monday",
        random_seed: int = 42,
    ):
        """Initialize experiment runner.

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

        print("=" * 70)
        print("MONDAY PGCN BLOCKING EXPERIMENT RUNNER")
        print("=" * 70)
        print(f"Cache directory: {self.cache_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Random seed: {random_seed}")
        print()

    def load_enhanced_circuit(self) -> EnhancedOlfactoryCircuit:
        """Load enhanced circuit with all 13K+ neurons."""
        print("[1/4] Loading enhanced circuit...")

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
        print(f"  ✓ Loaded {connectivity.n_ln} LNs (LOCAL INTERNEURONS)")
        print(f"  ✓ Loaded {connectivity.n_lh} LH neurons")
        print(f"  ✓ Loaded {connectivity.n_motor} Motor neurons")
        print(f"  ✓ Loaded {connectivity.n_an} ANs")
        print(f"  ✓ Loaded {connectivity.n_dn} DNs")

        total_neurons = (
            connectivity.n_pn
            + connectivity.n_kc
            + connectivity.n_mbon
            + connectivity.n_dan
            + connectivity.n_ln
            + connectivity.n_lh
            + connectivity.n_motor
            + connectivity.n_an
            + connectivity.n_dn
        )
        print(f"\n  TOTAL: {total_neurons:,} neurons loaded!")

        circuit = EnhancedOlfactoryCircuit(
            connectivity=connectivity,
            kc_sparsity_target=0.05,
            enable_ln_modulation=True,
            enable_lh_pathway=True,
            enable_motor_output=True,
            enable_vnc_interface=True,
            gaba_strength=1.0,  # Normal GABAergic strength
            chol_strength=1.0,  # Normal cholinergic strength
        )

        print("  ✓ Enhanced circuit initialized")
        print()

        return circuit

    def run_experiment_1_veto_gate(
        self,
        circuit: EnhancedOlfactoryCircuit,
        n_phase1_trials: int = 10,
        n_phase2_trials: int = 30,
        target_glomerulus: str = "DA1",
        distractor_glomerulus: str = "DL3",
    ) -> Dict[str, Any]:
        """Run Experiment 1: GABAergic veto gate blocking.

        Parameters
        ----------
        circuit : EnhancedOlfactoryCircuit
            Loaded enhanced circuit.
        n_phase1_trials : int
            Phase 1 trials (baseline learning).
        n_phase2_trials : int
            Phase 2 trials (blocking test).
        target_glomerulus : str
            Target odor that learns normally (NOT blocked).
        distractor_glomerulus : str
            Distractor odor that will be blocked by veto.

        Returns
        -------
        Dict[str, Any]
            Experiment results.
        """
        print(f"[2/4] Running Experiment 1: Veto Gate Blocking...")
        print(f"  Target glomerulus (learns): {target_glomerulus}")
        print(f"  Distractor glomerulus (blocked): {distractor_glomerulus}")
        print(f"  Phase 1 trials: {n_phase1_trials}")
        print(f"  Phase 2 trials: {n_phase2_trials}")

        # Initialize plasticity with KC→MBON weights
        initial_weights = circuit.connectivity.kc_to_mbon.toarray().copy()
        plasticity = DopamineModulatedPlasticity(
            kc_to_mbon_weights=initial_weights,
            learning_rate=0.01,
            eligibility_trace_tau=None,
        )

        # Create veto experiment (veto_glomerulus is the one being BLOCKED)
        veto_exp = VetoGateExperiment(
            circuit=circuit.core_circuit,  # Use core circuit for veto experiment
            plasticity=plasticity,
            veto_glomerulus=distractor_glomerulus,  # DL3 will be blocked
            veto_strength=1.0,  # Maximum veto strength
        )

        # Run full experiment
        # odor_a = target (DA1, learns normally)
        # odor_b = distractor (DL3, blocked by veto)
        print("  Running training phases...")
        results = veto_exp.run_full_experiment(
            n_phase1_trials=n_phase1_trials,
            n_phase2_trials=n_phase2_trials,
            odor_a=target_glomerulus,
            odor_b=distractor_glomerulus,
        )

        # Analyze blocking effect
        metrics = veto_exp.analyze_blocking_effect(results)

        print(f"\n  RESULTS:")
        print(f"    Blocking Index (final responses): {results['blocking_index']:.3f}")
        print(f"    Blocking Effectiveness (Phase 2 learning): {results.get('blocking_effectiveness', 0.0):.3f}")
        print(f"    Veto Efficacy: {metrics['veto_efficacy']:.3f}")
        print(f"    Mean Gating Suppression: {metrics['mean_gating_suppression']:.3f}")
        print(f"    Target ({target_glomerulus}) Phase 2 Change: {results.get('odor_a_phase2_change', 0.0):.6f}")
        print(f"    Distractor ({distractor_glomerulus}) Phase 2 Change: {results.get('odor_b_phase2_change', 0.0):.6f}")
        print(f"    Target ({target_glomerulus}) Final Response: {results['test_responses'][target_glomerulus]:.3f}")
        print(f"    Distractor ({distractor_glomerulus}) Final Response: {results['test_responses'][distractor_glomerulus]:.3f}")
        print()

        return {"results": results, "metrics": metrics, "experiment": "veto_gate"}

    def generate_visualizations(self, experiment_data: Dict[str, Any]):
        """Generate comprehensive visualization plots.

        Parameters
        ----------
        experiment_data : Dict[str, Any]
            Experiment results from run_experiment_1_veto_gate.
        """
        print("[3/4] Generating visualizations...")

        results = experiment_data["results"]
        metrics = experiment_data["metrics"]

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Experiment 1: GABAergic Veto Gate Blocking Results",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: Phase 2 RPE over trials
        phase2_trials = pd.DataFrame(results["phase2_trials"])
        ax = axes[0, 0]
        for odor in phase2_trials["odor"].unique():
            odor_trials = phase2_trials[phase2_trials["odor"] == odor]
            ax.plot(
                odor_trials.index,
                odor_trials["rpe"].abs(),
                marker="o",
                label=f"{odor}",
                alpha=0.7,
            )
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("|Reward Prediction Error|")
        ax.set_title("Phase 2: Learning Curves")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: Veto gating effect
        ax = axes[0, 1]
        odor_a_trials = phase2_trials[phase2_trials["odor"] == list(results["test_responses"].keys())[0]]
        ax.plot(
            odor_a_trials.index,
            odor_a_trials["gating_factor"],
            marker="s",
            color="red",
            label="Gating Factor (1 - βv)",
            linewidth=2,
        )
        ax.axhline(y=1.0, color="gray", linestyle="--", label="No Veto (=1.0)")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Gating Factor")
        ax.set_title("Veto Gate Modulation Over Time")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 3: Final test responses
        ax = axes[1, 0]
        test_responses = results["test_responses"]
        odors = list(test_responses.keys())
        responses = list(test_responses.values())
        colors = ["red" if "DA1" in o else "blue" for o in odors]
        ax.bar(odors, responses, color=colors, alpha=0.7, edgecolor="black")
        ax.set_ylabel("Final MBON Response")
        ax.set_title("Test Responses (Post-Training)")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

        # Plot 4: Blocking metrics summary
        ax = axes[1, 1]
        metric_names = ["Blocking\nIndex", "Veto\nEfficacy", "Gating\nSuppression"]
        metric_values = [
            results["blocking_index"],
            metrics["veto_efficacy"],
            metrics["mean_gating_suppression"],
        ]
        bars = ax.bar(metric_names, metric_values, color=["green", "orange", "purple"], alpha=0.7, edgecolor="black")
        ax.set_ylabel("Metric Value")
        ax.set_title("Summary Metrics")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plot_path = self.output_dir / "experiment_1_veto_gate_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved plot: {plot_path}")
        plt.close()

    def save_results(self, experiment_data: Dict[str, Any]):
        """Save results to JSON and CSV.

        Parameters
        ----------
        experiment_data : Dict[str, Any]
            Experiment results.
        """
        print("[4/4] Saving results...")

        # Save summary JSON
        summary = {
            "experiment": experiment_data["experiment"],
            "blocking_index": float(experiment_data["results"]["blocking_index"]),
            "metrics": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in experiment_data["metrics"].items()
            },
            "test_responses": {
                k: float(v) for k, v in experiment_data["results"]["test_responses"].items()
            },
        }

        json_path = self.output_dir / "experiment_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary: {json_path}")

        # Save trial data
        phase2_df = pd.DataFrame(experiment_data["results"]["phase2_trials"])
        csv_path = self.output_dir / "phase2_trials.csv"
        phase2_df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved trials: {csv_path}")

        # Create README
        readme_path = self.output_dir / "README.md"
        readme_content = f"""# Monday Experiment Results

## Experiment: {experiment_data['experiment']}

### Key Results

- **Blocking Index**: {summary['blocking_index']:.3f}
  - > 0: Blocking successful (OdorB learned more than OdorA)
  - ≈ 0: No blocking (equal learning)
  - < 0: Blocking failed (OdorA learned more)

- **Veto Efficacy**: {summary['metrics']['veto_efficacy']:.3f}
  - Measures how strongly veto pathway activates (0-1 range)

- **Mean Gating Suppression**: {summary['metrics']['mean_gating_suppression']:.3f}
  - Fraction of plasticity blocked by veto (0-1 range)

### Test Responses

"""
        for odor, response in summary["test_responses"].items():
            readme_content += f"- **{odor}**: {response:.3f}\n"

        readme_content += f"""

### Files Generated

1. **experiment_summary.json** - JSON summary of all metrics
2. **phase2_trials.csv** - Trial-by-trial data for Phase 2
3. **experiment_1_veto_gate_results.png** - Visualization plots
4. **README.md** - This file

### Interpretation

"""
        if summary["blocking_index"] > 0.2:
            readme_content += "✓ **Blocking effect detected!** The GABAergic veto successfully suppressed learning in the blocked pathway.\n"
        else:
            readme_content += "⚠ **Weak/no blocking detected.** May need to adjust veto strength or training parameters.\n"

        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"  ✓ Saved README: {readme_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Monday Startup Training - One-Command Blocking Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="veto",
        choices=["veto"],
        help="Experiment to run (default: veto)",
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
        default="results/monday",
        help="Output directory (default: results/monday)",
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
        default=30,
        help="Phase 2 trials (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Initialize runner
    runner = MondayExperimentRunner(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    try:
        # Load circuit
        circuit = runner.load_enhanced_circuit()

        # Run experiment
        if args.experiment == "veto":
            results = runner.run_experiment_1_veto_gate(
                circuit,
                n_phase1_trials=args.phase1_trials,
                n_phase2_trials=args.phase2_trials,
            )

        # Generate visualizations
        runner.generate_visualizations(results)

        # Save results
        runner.save_results(results)

        print("=" * 70)
        print("✓ MONDAY EXPERIMENT COMPLETE!")
        print(f"✓ Results saved to: {runner.output_dir}")
        print(f"✓ Open {runner.output_dir}/experiment_1_veto_gate_results.png to view plots")
        print(f"✓ Blocking Index: {results['results']['blocking_index']:.3f}")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
