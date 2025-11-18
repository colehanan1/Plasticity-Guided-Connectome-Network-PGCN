#!/usr/bin/env python3
"""
Comprehensive Veto Mechanism Debugging & Validation
====================================================

This script performs rigorous validation of the PGCN veto blocking mechanism
to distinguish between:
- REAL biological effects
- MATHEMATICAL artifacts (scaling issues, initialization bias)
- UNREALISTIC parameters (machine epsilon values, explosion effects)

Key Questions:
1. Is the blocking mechanism real or just initialization bias?
2. Are response magnitudes biologically realistic?
3. Does veto work gradually or as mathematical cutoff?
4. Do learning rates match real fly experiments?
5. What parameters need adjustment for biological realism?

Usage:
    python scripts/debug_veto_mechanism.py --output docs/debug_analysis/

Author: PGCN Project
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

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


class VetoMechanismDebugger:
    """Comprehensive debugging suite for veto mechanism validation."""

    def __init__(self, cache_dir: str = "data/cache", output_dir: str = "docs/debug_analysis"):
        """Initialize debugger.

        Parameters
        ----------
        cache_dir : str
            Path to data cache.
        output_dir : str
            Path to save debug results.
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}

        print("=" * 80)
        print("VETO MECHANISM COMPREHENSIVE DEBUGGING")
        print("=" * 80)
        print(f"Cache directory: {self.cache_dir}")
        print(f"Output directory: {self.output_dir}")
        print()

    def load_circuit(self) -> EnhancedOlfactoryCircuit:
        """Load circuit for testing."""
        print("[1/8] Loading circuit...")

        loader = CircuitLoader(cache_dir=str(self.cache_dir))
        connectivity = loader.load_connectivity_matrix(
            normalize_weights="row",
            include_dan=True,
            include_extended=True,
        )

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

        print(f"  ✓ Loaded circuit: {connectivity.n_pn} PNs, {connectivity.n_kc} KCs, {connectivity.n_mbon} MBONs")
        print()

        return circuit

    def test_virgin_responses(self, circuit: EnhancedOlfactoryCircuit) -> Dict[str, Any]:
        """Test virgin network responses (NO training).

        This is CRITICAL: if DA1 >> DL3 before any training,
        then the "blocking effect" might just be initialization bias.

        Parameters
        ----------
        circuit : EnhancedOlfactoryCircuit
            Loaded circuit.

        Returns
        -------
        Dict[str, Any]
            Virgin response analysis.
        """
        print("[2/8] Testing virgin network responses (no training)...")

        # Initialize FRESH plasticity (random small weights)
        np.random.seed(42)
        initial_weights = circuit.connectivity.kc_to_mbon.toarray().copy()

        # Test with random initialization (typical range)
        random_init = np.random.uniform(-0.01, 0.01, initial_weights.shape)

        plasticity_random = DopamineModulatedPlasticity(
            kc_to_mbon_weights=random_init,
            learning_rate=0.01,
            eligibility_trace_tau=None,
        )

        # Test with existing weights
        plasticity_existing = DopamineModulatedPlasticity(
            kc_to_mbon_weights=initial_weights,
            learning_rate=0.01,
            eligibility_trace_tau=None,
        )

        results = {}

        for name, plasticity in [("random_init", plasticity_random), ("flywire_init", plasticity_existing)]:
            # Get virgin responses for DA1 and DL3
            da1_pn = circuit.core_circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)
            da1_kc = circuit.core_circuit.propagate_pn_to_kc(da1_pn)
            da1_mbon = float((plasticity.kc_to_mbon @ da1_kc)[0])

            dl3_pn = circuit.core_circuit.activate_pns_by_glomeruli(["DL3"], firing_rate=1.0)
            dl3_kc = circuit.core_circuit.propagate_pn_to_kc(dl3_pn)
            dl3_mbon = float((plasticity.kc_to_mbon @ dl3_kc)[0])

            # Compute connectivity strengths
            da1_pn_count = np.sum(da1_pn > 0)
            dl3_pn_count = np.sum(dl3_pn > 0)
            da1_kc_count = np.sum(da1_kc > 0)
            dl3_kc_count = np.sum(dl3_kc > 0)

            results[name] = {
                "da1_virgin_response": float(da1_mbon),
                "dl3_virgin_response": float(dl3_mbon),
                "response_ratio_da1_dl3": float(da1_mbon / (dl3_mbon + 1e-9)),
                "da1_pn_count": int(da1_pn_count),
                "dl3_pn_count": int(dl3_pn_count),
                "da1_kc_count": int(da1_kc_count),
                "dl3_kc_count": int(dl3_kc_count),
                "connectivity_ratio": float(da1_pn_count / (dl3_pn_count + 1e-9)),
            }

            print(f"\n  {name.upper()}:")
            print(f"    DA1 virgin MBON output: {da1_mbon:.4f}")
            print(f"    DL3 virgin MBON output: {dl3_mbon:.4f}")
            print(f"    Response ratio (DA1/DL3): {results[name]['response_ratio_da1_dl3']:.2f}x")
            print(f"    DA1 active PNs: {da1_pn_count}, active KCs: {da1_kc_count}")
            print(f"    DL3 active PNs: {dl3_pn_count}, active KCs: {dl3_kc_count}")
            print(f"    Connectivity ratio: {results[name]['connectivity_ratio']:.2f}x")

            if results[name]['response_ratio_da1_dl3'] > 10:
                print(f"    🚨 WARNING: Massive initial bias detected! ({results[name]['response_ratio_da1_dl3']:.0f}x)")
                print(f"    This suggests FlyWire connectivity differences, not learning effects")

        self.results["virgin_responses"] = results
        print()
        return results

    def test_veto_dose_response(self, circuit: EnhancedOlfactoryCircuit) -> Dict[str, Any]:
        """Test veto mechanism with different strengths.

        A real biological mechanism should show GRADUAL reduction.
        A mathematical cutoff would show sudden drop at high values.

        Parameters
        ----------
        circuit : EnhancedOlfactoryCircuit
            Loaded circuit.

        Returns
        -------
        Dict[str, Any]
            Veto dose-response analysis.
        """
        print("[3/8] Testing veto dose-response curve...")

        # Test range of veto strengths
        veto_strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]

        results = []

        initial_weights = circuit.connectivity.kc_to_mbon.toarray().copy()

        for veto_strength in veto_strengths:
            plasticity = DopamineModulatedPlasticity(
                kc_to_mbon_weights=initial_weights.copy(),
                learning_rate=0.01,
                eligibility_trace_tau=None,
            )

            veto_exp = VetoGateExperiment(
                circuit=circuit.core_circuit,
                plasticity=plasticity,
                veto_glomerulus="DL3",
                veto_strength=veto_strength,
            )

            # Run single trial with veto active (unless veto_strength=0)
            trial_data = veto_exp.run_trial_with_veto(
                odor="DL3",
                reward=1.0,
                veto_active=(veto_strength > 0.0),  # Only activate if strength > 0
            )

            results.append({
                "veto_strength": veto_strength,
                "mbon_response": trial_data["mbon_output"],
                "veto_value": trial_data["veto_value"],
                "gating_factor": trial_data["gating_factor"],
                "weight_change": trial_data["weight_change_magnitude"],
            })

            print(f"  Veto={veto_strength:.3f}: Response={trial_data['mbon_output']:.4f}, "
                  f"Gating={trial_data['gating_factor']:.6f}, "
                  f"Weight_change={trial_data['weight_change_magnitude']:.6f}")

        self.results["veto_dose_response"] = results

        # Check for sudden cutoff vs gradual reduction
        response_range = max(r["mbon_response"] for r in results) - min(r["mbon_response"] for r in results)
        midpoint_idx = len(results) // 2
        if results[midpoint_idx]["gating_factor"] > 0.3:
            print("  ✓ Gradual veto effect observed (biological)")
        else:
            print("  ⚠️  Sudden cutoff detected (may be mathematical artifact)")

        print()
        return results

    def test_learning_dynamics(self, circuit: EnhancedOlfactoryCircuit) -> Dict[str, Any]:
        """Test learning dynamics over multiple trials.

        Realistic learning should show:
        - Small, gradual weight changes (0.01-1.0 per trial)
        - S-curve response growth (not exponential)
        - Saturation after ~20-50 trials

        Parameters
        ----------
        circuit : EnhancedOlfactoryCircuit
            Loaded circuit.

        Returns
        -------
        Dict[str, Any]
            Learning dynamics analysis.
        """
        print("[4/8] Testing learning dynamics over time...")

        n_trials = 30
        learning_rates = [0.001, 0.01, 0.1]

        all_results = {}

        for lr in learning_rates:
            print(f"\n  Testing learning rate = {lr}")

            initial_weights = circuit.connectivity.kc_to_mbon.toarray().copy()
            plasticity = DopamineModulatedPlasticity(
                kc_to_mbon_weights=initial_weights.copy(),
                learning_rate=lr,
                eligibility_trace_tau=None,
            )

            veto_exp = VetoGateExperiment(
                circuit=circuit.core_circuit,
                plasticity=plasticity,
                veto_glomerulus="DL3",
                veto_strength=0.0,  # No veto for baseline learning
            )

            trials = []
            for trial_idx in range(n_trials):
                before_weights = plasticity.kc_to_mbon.copy()

                trial_data = veto_exp.run_trial_with_veto(
                    odor="DA1",
                    reward=1.0,
                    veto_active=False,
                )

                after_weights = plasticity.kc_to_mbon

                weight_change_magnitude = np.linalg.norm(after_weights - before_weights)

                trials.append({
                    "trial": trial_idx,
                    "mbon_response": trial_data["mbon_output"],
                    "weight_change_magnitude": float(weight_change_magnitude),
                    "rpe": trial_data["rpe"],
                })

            all_results[f"lr_{lr}"] = trials

            # Analyze learning curve
            responses = [t["mbon_response"] for t in trials]
            weight_changes = [t["weight_change_magnitude"] for t in trials]

            print(f"    Initial response: {responses[0]:.4f}")
            print(f"    Final response: {responses[-1]:.4f}")
            print(f"    Mean weight change: {np.mean(weight_changes):.6f}")
            print(f"    Max weight change: {np.max(weight_changes):.6f}")

            # Check for unrealistic values
            if np.max(weight_changes) > 10.0:
                print(f"    🚨 WARNING: Unrealistically large weight changes (>{np.max(weight_changes):.1f})")
            if responses[-1] > 1000:
                print(f"    🚨 WARNING: Unrealistic response magnitude ({responses[-1]:.0f})")

        self.results["learning_dynamics"] = all_results
        print()
        return all_results

    def test_blocking_with_controls(self, circuit: EnhancedOlfactoryCircuit) -> Dict[str, Any]:
        """Test blocking effect with proper controls.

        Compare:
        - No veto (baseline learning)
        - Veto on DL3 (should suppress learning)
        - Different Phase 1 durations (check for order effects)

        Parameters
        ----------
        circuit : EnhancedOlfactoryCircuit
            Loaded circuit.

        Returns
        -------
        Dict[str, Any]
            Blocking analysis with controls.
        """
        print("[5/8] Testing blocking effect with controls...")

        conditions = [
            {"name": "no_veto", "veto_strength": 0.0, "n_phase1": 10, "n_phase2": 30},
            {"name": "veto_weak", "veto_strength": 0.5, "n_phase1": 10, "n_phase2": 30},
            {"name": "veto_strong", "veto_strength": 1.0, "n_phase1": 10, "n_phase2": 30},
            {"name": "short_phase1", "veto_strength": 1.0, "n_phase1": 2, "n_phase2": 30},
            {"name": "long_phase1", "veto_strength": 1.0, "n_phase1": 20, "n_phase2": 30},
        ]

        results = []

        for cond in conditions:
            print(f"\n  Condition: {cond['name']}")

            initial_weights = circuit.connectivity.kc_to_mbon.toarray().copy()
            plasticity = DopamineModulatedPlasticity(
                kc_to_mbon_weights=initial_weights.copy(),
                learning_rate=0.01,
                eligibility_trace_tau=None,
            )

            veto_exp = VetoGateExperiment(
                circuit=circuit.core_circuit,
                plasticity=plasticity,
                veto_glomerulus="DL3",
                veto_strength=cond["veto_strength"],
            )

            exp_results = veto_exp.run_full_experiment(
                n_phase1_trials=cond["n_phase1"],
                n_phase2_trials=cond["n_phase2"],
                odor_a="DA1",
                odor_b="DL3",
            )

            metrics = veto_exp.analyze_blocking_effect(exp_results)

            result = {
                "condition": cond["name"],
                "veto_strength": cond["veto_strength"],
                "blocking_index": exp_results["blocking_index"],
                "da1_final": exp_results["test_responses"]["DA1"],
                "dl3_final": exp_results["test_responses"]["DL3"],
                "veto_efficacy": metrics["veto_efficacy"],
            }

            results.append(result)

            print(f"    Blocking Index: {result['blocking_index']:.3f}")
            print(f"    DA1 final: {result['da1_final']:.2f}")
            print(f"    DL3 final: {result['dl3_final']:.2f}")
            print(f"    Veto efficacy: {result['veto_efficacy']:.3f}")

        self.results["blocking_with_controls"] = results
        print()
        return results

    def analyze_weight_distributions(self, circuit: EnhancedOlfactoryCircuit) -> Dict[str, Any]:
        """Analyze weight matrix distributions.

        Check if pathway differences come from:
        - FlyWire connectivity structure
        - Learning-induced changes
        - Random initialization

        Parameters
        ----------
        circuit : EnhancedOlfactoryCircuit
            Loaded circuit.

        Returns
        -------
        Dict[str, Any]
            Weight distribution analysis.
        """
        print("[6/8] Analyzing weight matrix distributions...")

        initial_weights = circuit.connectivity.kc_to_mbon.toarray()

        # Get pathway-specific weights
        da1_pn = circuit.core_circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)
        da1_kc = circuit.core_circuit.propagate_pn_to_kc(da1_pn)
        da1_kc_indices = np.where(da1_kc > 0)[0]

        dl3_pn = circuit.core_circuit.activate_pns_by_glomeruli(["DL3"], firing_rate=1.0)
        dl3_kc = circuit.core_circuit.propagate_pn_to_kc(dl3_pn)
        dl3_kc_indices = np.where(dl3_kc > 0)[0]

        # Get weights for DA1 and DL3 pathways
        da1_weights = initial_weights[0, da1_kc_indices]
        dl3_weights = initial_weights[0, dl3_kc_indices]

        results = {
            "da1_pathway": {
                "n_synapses": len(da1_kc_indices),
                "mean_weight": float(np.mean(da1_weights)),
                "std_weight": float(np.std(da1_weights)),
                "min_weight": float(np.min(da1_weights)),
                "max_weight": float(np.max(da1_weights)),
            },
            "dl3_pathway": {
                "n_synapses": len(dl3_kc_indices),
                "mean_weight": float(np.mean(dl3_weights)),
                "std_weight": float(np.std(dl3_weights)),
                "min_weight": float(np.min(dl3_weights)),
                "max_weight": float(np.max(dl3_weights)),
            },
            "connectivity_ratio": len(da1_kc_indices) / (len(dl3_kc_indices) + 1e-9),
        }

        print(f"  DA1 pathway: {results['da1_pathway']['n_synapses']} synapses, "
              f"mean weight = {results['da1_pathway']['mean_weight']:.6f}")
        print(f"  DL3 pathway: {results['dl3_pathway']['n_synapses']} synapses, "
              f"mean weight = {results['dl3_pathway']['mean_weight']:.6f}")
        print(f"  Connectivity ratio (DA1/DL3): {results['connectivity_ratio']:.2f}x")

        if results['connectivity_ratio'] > 5:
            print(f"  🚨 WARNING: DA1 has {results['connectivity_ratio']:.0f}x more synapses than DL3")
            print(f"  This natural bias may dominate any veto effects")

        self.results["weight_distributions"] = results
        print()
        return results

    def generate_visualizations(self):
        """Generate comprehensive debug visualization plots."""
        print("[7/8] Generating debug visualizations...")

        fig = plt.figure(figsize=(20, 12))

        # Plot 1: Virgin responses comparison
        if "virgin_responses" in self.results:
            ax1 = plt.subplot(3, 3, 1)
            data = self.results["virgin_responses"]

            for init_type in ["random_init", "flywire_init"]:
                da1 = data[init_type]["da1_virgin_response"]
                dl3 = data[init_type]["dl3_virgin_response"]
                ax1.bar([f"{init_type}\nDA1", f"{init_type}\nDL3"],
                       [da1, dl3], alpha=0.7)

            ax1.set_ylabel('Virgin MBON Output', fontsize=12, fontweight='bold')
            ax1.set_title('Virgin Network Responses\n(Before Training)', fontsize=12, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Veto dose-response
        if "veto_dose_response" in self.results:
            ax2 = plt.subplot(3, 3, 2)
            data = pd.DataFrame(self.results["veto_dose_response"])

            ax2.plot(data["veto_strength"], data["gating_factor"],
                    marker='o', linewidth=2, markersize=6)
            ax2.set_xlabel('Veto Strength', fontsize=12)
            ax2.set_ylabel('Gating Factor\n(1.0 = no block, 0.0 = full block)', fontsize=12)
            ax2.set_title('Veto Dose-Response Curve', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% blocking')
            ax2.legend()

        # Plot 3: Learning dynamics
        if "learning_dynamics" in self.results:
            ax3 = plt.subplot(3, 3, 3)

            for lr_key, trials in self.results["learning_dynamics"].items():
                df = pd.DataFrame(trials)
                ax3.plot(df["trial"], df["mbon_response"],
                        marker='o', label=lr_key, linewidth=2, markersize=4)

            ax3.set_xlabel('Trial Number', fontsize=12)
            ax3.set_ylabel('MBON Response', fontsize=12)
            ax3.set_title('Learning Dynamics Over Time', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)

        # Plot 4: Weight changes over time
        if "learning_dynamics" in self.results:
            ax4 = plt.subplot(3, 3, 4)

            for lr_key, trials in self.results["learning_dynamics"].items():
                df = pd.DataFrame(trials)
                ax4.plot(df["trial"], df["weight_change_magnitude"],
                        marker='o', label=lr_key, linewidth=2, markersize=4)

            ax4.set_xlabel('Trial Number', fontsize=12)
            ax4.set_ylabel('Weight Change Magnitude', fontsize=12)
            ax4.set_title('Weight Updates Over Time', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)
            ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Realistic threshold')

        # Plot 5: Blocking comparison
        if "blocking_with_controls" in self.results:
            ax5 = plt.subplot(3, 3, 5)
            data = pd.DataFrame(self.results["blocking_with_controls"])

            x = np.arange(len(data))
            ax5.bar(x, data["blocking_index"], alpha=0.7)
            ax5.set_xticks(x)
            ax5.set_xticklabels(data["condition"], rotation=45, ha='right')
            ax5.set_ylabel('Blocking Index', fontsize=12)
            ax5.set_title('Blocking Effect Across Conditions', fontsize=12, fontweight='bold')
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax5.grid(axis='y', alpha=0.3)

        # Plot 6: Final responses comparison
        if "blocking_with_controls" in self.results:
            ax6 = plt.subplot(3, 3, 6)
            data = pd.DataFrame(self.results["blocking_with_controls"])

            x = np.arange(len(data))
            width = 0.35
            ax6.bar(x - width/2, data["da1_final"], width, label='DA1', alpha=0.7)
            ax6.bar(x + width/2, data["dl3_final"], width, label='DL3', alpha=0.7)

            ax6.set_xticks(x)
            ax6.set_xticklabels(data["condition"], rotation=45, ha='right')
            ax6.set_ylabel('Final MBON Response', fontsize=12)
            ax6.set_title('Final Responses by Condition', fontsize=12, fontweight='bold')
            ax6.legend()
            ax6.grid(axis='y', alpha=0.3)

        # Plot 7: Weight distributions
        if "weight_distributions" in self.results:
            ax7 = plt.subplot(3, 3, 7)
            data = self.results["weight_distributions"]

            categories = ['N Synapses', 'Mean Weight']
            da1_vals = [data["da1_pathway"]["n_synapses"], data["da1_pathway"]["mean_weight"] * 1000]
            dl3_vals = [data["dl3_pathway"]["n_synapses"], data["dl3_pathway"]["mean_weight"] * 1000]

            x = np.arange(len(categories))
            width = 0.35
            ax7.bar(x - width/2, da1_vals, width, label='DA1', alpha=0.7)
            ax7.bar(x + width/2, dl3_vals, width, label='DL3', alpha=0.7)

            ax7.set_xticks(x)
            ax7.set_xticklabels(categories)
            ax7.set_ylabel('Value', fontsize=12)
            ax7.set_title('Pathway Connectivity Comparison\n(Weight × 1000)', fontsize=12, fontweight='bold')
            ax7.legend()
            ax7.grid(axis='y', alpha=0.3)

        # Plot 8: Veto efficacy vs blocking
        if "blocking_with_controls" in self.results:
            ax8 = plt.subplot(3, 3, 8)
            data = pd.DataFrame(self.results["blocking_with_controls"])

            ax8.scatter(data["veto_efficacy"], data["blocking_index"],
                       s=100, alpha=0.7)

            for i, row in data.iterrows():
                ax8.annotate(row["condition"], (row["veto_efficacy"], row["blocking_index"]),
                           fontsize=8, ha='right')

            ax8.set_xlabel('Veto Efficacy', fontsize=12)
            ax8.set_ylabel('Blocking Index', fontsize=12)
            ax8.set_title('Veto Efficacy vs Blocking', fontsize=12, fontweight='bold')
            ax8.grid(alpha=0.3)

        # Plot 9: Connectivity ratio impact
        if "virgin_responses" in self.results:
            ax9 = plt.subplot(3, 3, 9)
            data = self.results["virgin_responses"]

            ratios = [data[k]["response_ratio_da1_dl3"] for k in ["random_init", "flywire_init"]]
            conn_ratios = [data[k]["connectivity_ratio"] for k in ["random_init", "flywire_init"]]

            ax9.scatter(conn_ratios, ratios, s=150, alpha=0.7)
            ax9.plot([0, max(conn_ratios)*1.1], [0, max(conn_ratios)*1.1],
                    'r--', alpha=0.5, label='y=x (perfect correlation)')

            for i, label in enumerate(["Random Init", "FlyWire Init"]):
                ax9.annotate(label, (conn_ratios[i], ratios[i]), fontsize=10, ha='left')

            ax9.set_xlabel('Connectivity Ratio (DA1/DL3)', fontsize=12)
            ax9.set_ylabel('Response Ratio (DA1/DL3)', fontsize=12)
            ax9.set_title('Connectivity vs Response Bias', fontsize=12, fontweight='bold')
            ax9.legend()
            ax9.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / "comprehensive_debug_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved visualization: {plot_path}")
        plt.close()

    def generate_report(self):
        """Generate comprehensive debug report."""
        print("[8/8] Generating debug report...")

        # Save all results
        json_path = self.output_dir / "debug_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"  ✓ Saved results: {json_path}")

        # Generate summary report
        report_path = self.output_dir / "DEBUG_SUMMARY.txt"
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("VETO MECHANISM DEBUG SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Virgin responses
            if "virgin_responses" in self.results:
                f.write("1. VIRGIN NETWORK RESPONSES (Before Training)\n")
                f.write("-" * 80 + "\n")
                for init_type, data in self.results["virgin_responses"].items():
                    f.write(f"\n{init_type.upper()}:\n")
                    f.write(f"  DA1 virgin response: {data['da1_virgin_response']:.4f}\n")
                    f.write(f"  DL3 virgin response: {data['dl3_virgin_response']:.4f}\n")
                    f.write(f"  Response ratio: {data['response_ratio_da1_dl3']:.2f}x\n")
                    f.write(f"  Connectivity ratio: {data['connectivity_ratio']:.2f}x\n")

                    if data['response_ratio_da1_dl3'] > 10:
                        f.write(f"  🚨 WARNING: Massive initial bias ({data['response_ratio_da1_dl3']:.0f}x)\n")
                f.write("\n")

            # Weight distributions
            if "weight_distributions" in self.results:
                f.write("2. WEIGHT DISTRIBUTION ANALYSIS\n")
                f.write("-" * 80 + "\n")
                data = self.results["weight_distributions"]
                f.write(f"  DA1 pathway: {data['da1_pathway']['n_synapses']} synapses\n")
                f.write(f"  DL3 pathway: {data['dl3_pathway']['n_synapses']} synapses\n")
                f.write(f"  Connectivity ratio: {data['connectivity_ratio']:.2f}x\n\n")

            # Veto dose-response
            if "veto_dose_response" in self.results:
                f.write("3. VETO DOSE-RESPONSE\n")
                f.write("-" * 80 + "\n")
                data = pd.DataFrame(self.results["veto_dose_response"])
                f.write(f"  Veto Strength | Gating Factor | Weight Change\n")
                f.write(f"  {'-'*14}|{'-'*15}|{'-'*15}\n")
                for _, row in data.iterrows():
                    f.write(f"  {row['veto_strength']:13.3f} | {row['gating_factor']:13.6f} | {row['weight_change']:13.6f}\n")
                f.write("\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("=" * 80 + "\n\n")

            # Analyze findings
            if "virgin_responses" in self.results:
                flywire_data = self.results["virgin_responses"]["flywire_init"]
                if flywire_data['response_ratio_da1_dl3'] > 10:
                    f.write("❌ FINDING 1: Massive initialization bias detected\n")
                    f.write(f"   DA1/DL3 ratio = {flywire_data['response_ratio_da1_dl3']:.0f}x BEFORE any training\n")
                    f.write("   This suggests the 'blocking effect' may be initialization artifact,\n")
                    f.write("   not a learned veto mechanism.\n\n")
                else:
                    f.write("✅ FINDING 1: Relatively balanced initialization\n")
                    f.write(f"   DA1/DL3 ratio = {flywire_data['response_ratio_da1_dl3']:.1f}x\n\n")

            if "veto_dose_response" in self.results:
                data = pd.DataFrame(self.results["veto_dose_response"])
                midpoint = data[data["veto_strength"] == 0.5]
                if len(midpoint) > 0 and midpoint.iloc[0]["gating_factor"] < 0.3:
                    f.write("❌ FINDING 2: Sudden veto cutoff detected\n")
                    f.write("   Gating drops to near-zero at moderate veto strengths.\n")
                    f.write("   This suggests mathematical artifact, not biological mechanism.\n\n")
                else:
                    f.write("✅ FINDING 2: Gradual veto effect observed\n")
                    f.write("   Gating factor decreases smoothly with veto strength.\n\n")

        print(f"  ✓ Saved report: {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Veto Mechanism Debugging & Validation",
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
        default="docs/debug_analysis",
        help="Output directory (default: docs/debug_analysis)",
    )

    args = parser.parse_args()

    # Initialize debugger
    debugger = VetoMechanismDebugger(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )

    try:
        # Load circuit
        circuit = debugger.load_circuit()

        # Run all debug tests
        debugger.test_virgin_responses(circuit)
        debugger.test_veto_dose_response(circuit)
        debugger.test_learning_dynamics(circuit)
        debugger.test_blocking_with_controls(circuit)
        debugger.analyze_weight_distributions(circuit)

        # Generate outputs
        debugger.generate_visualizations()
        debugger.generate_report()

        print("\n" + "=" * 80)
        print("DEBUG ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {debugger.output_dir}")
        print(f"  - comprehensive_debug_analysis.png (visualizations)")
        print(f"  - debug_results.json (raw data)")
        print(f"  - DEBUG_SUMMARY.txt (findings summary)")
        print("\nReview these files to determine if veto mechanism is real or artifact.")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
