#!/usr/bin/env python3
"""
Generate Real Data for Publication Figures

This template script shows how to extract data from your PGCN model
for the four publication figures. Customize the sections marked with
# TODO: to match your specific model and experiments.

Usage:
    python scripts/generate_figure_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def save_behavioral_data_simple(results_dict, output_dir="results/behavioral_sim"):
    """
    Save behavioral results in format expected by extraction script.

    Parameters
    ----------
    results_dict : dict
        Format: {
            'wildtype': [score_phase1, score_phase2, score_phase3],
            'or7a_mutant': [score_phase1, score_phase2, score_phase3],
            'control': [score_phase1, score_phase2, score_phase3]
        }
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    phases = ['after_A_train', 'after_B_train', 'A_test']

    for group, scores in results_dict.items():
        data = []
        for phase, score in zip(phases, scores):
            data.append({
                'phase': phase,
                'memory_score': score
            })

        df = pd.DataFrame(data)
        output_file = Path(output_dir) / f"{group}_behavioral.csv"
        df.to_csv(output_file, index=False)
        print(f"✓ Saved {group}: {output_file}")


def save_veto_mask_simple(mask, output_dir="results"):
    """
    Save veto gate protection mask.

    Parameters
    ----------
    mask : np.ndarray
        2D binary array (n_KC, n_MBON) where 1 = protected, 0 = unprotected
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file = Path(output_dir) / "veto_mask.npy"
    np.save(output_file, mask)

    n_protected = int(np.sum(mask))
    total = mask.size
    pct = (n_protected / total) * 100

    print(f"✓ Saved veto mask: {output_file}")
    print(f"  Shape: {mask.shape}")
    print(f"  Protected: {n_protected:,} / {total:,} ({pct:.1f}%)")


def save_ml_comparison_simple(scores_dict, output_dir="results"):
    """
    Save ML model comparison results.

    Parameters
    ----------
    scores_dict : dict
        Format: {
            'MBON_veto': forgetting_score,
            'Dense_ANN': forgetting_score,
            'EWC': forgetting_score,
            ...
        }
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([
        {'model_type': model, 'forgetting_score': score}
        for model, score in scores_dict.items()
    ])

    output_file = Path(output_dir) / "forgetting_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved ML comparison: {output_file}")


# ==============================================================================
# MAIN: Generate Your Real Data
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("Generating Real Data for Publication Figures")
    print("="*70)

    # ==========================================================================
    # TODO: Replace this section with your actual PGCN experiments
    # ==========================================================================

    # Example: Load your circuit and run experiments
    # ------------------------------------------------
    # Uncomment and modify these lines:

    # from data_loaders.circuit_loader import CircuitLoader
    # from pgcn.models.olfactory_circuit import OlfactoryCircuit
    # from pgcn.models.learning_model import LearningExperiment, DopamineModulatedPlasticity
    # from pgcn.models.veto_gate import SelectiveVetoGate

    # # Load circuit
    # loader = CircuitLoader(cache_dir="data/cache")
    # conn_matrix = loader.load_connectivity_matrix(normalize_weights="row")

    # # Create circuit and plasticity
    # circuit = OlfactoryCircuit(conn_matrix)
    # plasticity = DopamineModulatedPlasticity(
    #     kc_to_mbon_weights=conn_matrix.kc_to_mbon.toarray(),
    #     learning_rate=0.01
    # )

    # # Run experiments
    # experiment = LearningExperiment(circuit, plasticity)

    # # Task A training
    # task_A = experiment.run_experiment(
    #     odor_sequence=["DA1"] * 100,
    #     reward_sequence=[1] * 100
    # )
    # after_A_train = task_A['mbon_valence'].iloc[-10:].mean()

    # # Task B training
    # task_B = experiment.run_experiment(
    #     odor_sequence=["DL3"] * 100,
    #     reward_sequence=[1] * 100
    # )
    # after_B_train = task_B['mbon_valence'].iloc[-10:].mean()

    # # Test Task A
    # task_A_test = experiment.run_experiment(
    #     odor_sequence=["DA1"] * 20,
    #     reward_sequence=[0] * 20
    # )
    # A_test = task_A_test['mbon_valence'].iloc[-10:].mean()

    # ==========================================================================
    # Placeholder data (replace with your real results from above)
    # ==========================================================================

    print("\n⚠️  Using placeholder data (replace with real experiments)")
    print("   Edit scripts/generate_figure_data.py and uncomment TODO sections\n")

    # Figure 1: Behavioral Data
    # -------------------------
    behavioral_results = {
        'wildtype': [0.85, 0.72, 0.45],      # [after_A, after_B, test_A]
        'or7a_mutant': [0.84, 0.71, 0.68],   # With veto gate protection
        'control': [0.50, 0.50, 0.50]        # Baseline
    }

    save_behavioral_data_simple(behavioral_results)

    # Figure 2 & 3: Veto Mask
    # -----------------------
    # TODO: Replace with your actual veto gate protection mask
    #
    # Example:
    # veto_gate = SelectiveVetoGate(...)
    # veto_gate.identify_critical_pathways(...)
    # veto_mask = veto_gate.protection_mask

    n_kc, n_mbon = 2000, 44
    veto_mask = np.random.rand(n_kc, n_mbon) < 0.08  # 8% protected (placeholder)

    save_veto_mask_simple(veto_mask)

    # Figure 4: ML Comparison
    # -----------------------
    # TODO: Replace with your actual continual learning benchmarks
    #
    # Example:
    # def compute_forgetting(perf_before, perf_after):
    #     return (perf_before - perf_after) / perf_before
    #
    # ml_scores = {
    #     'MBON_veto': compute_forgetting(after_A_train, A_test),
    #     'Dense_ANN': compute_forgetting(...),
    #     ...
    # }

    ml_scores = {
        'MBON_veto': 0.12,      # Your method (lowest forgetting)
        'GEM': 0.35,
        'EWC': 0.42,
        'SI': 0.51,
        'LwF': 0.58,
        'Dense_ANN': 0.79       # Baseline (highest forgetting)
    }

    save_ml_comparison_simple(ml_scores)

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE")
    print("="*70)
    print("✓ Behavioral data: results/behavioral_sim/")
    print("✓ Veto mask: results/veto_mask.npy")
    print("✓ ML comparison: results/forgetting_summary.csv")
    print("\nNext steps:")
    print("  1. python extract_figure_data.py --task all")
    print("  2. python examples/plot_extracted_figures.py --figure all")
    print("  3. View figures in figures/publication/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
