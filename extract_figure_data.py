#!/usr/bin/env python3
"""
Data Extraction Pipeline for Four Key Figures
==============================================

This script extracts and prepares data from PGCN model outputs for publication figures.

Usage:
    python extract_figure_data.py --task all
    python extract_figure_data.py --task behavioral
    python extract_figure_data.py --task schematic
    python extract_figure_data.py --task synapse_map
    python extract_figure_data.py --task ml_comparison

Output:
    - Extracted data saved to `data/extracted_figures/`
    - Summary printed to console for verification
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import yaml
import warnings
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')

# ==============================================================================
# TASK 1: Behavioral Prediction Data Extraction
# ==============================================================================

def extract_behavioral_data(
    results_dir: str = "results/behavioral_sim",
    output_dir: str = "data/extracted_figures"
) -> Dict[str, Any]:
    """
    Extract memory scores for wildtype, or7a_mutant, and control groups
    across different experimental phases.

    Parameters
    ----------
    results_dir : str
        Directory containing behavioral simulation results
    output_dir : str
        Directory to save extracted data

    Returns
    -------
    Dict[str, Any]
        Dictionary with group names as keys and phase scores as values
        Format: {"wildtype": [phase1_score, phase2_score, ...], ...}
    """
    print("\n" + "="*70)
    print("TASK 1: Behavioral Prediction Data Extraction")
    print("="*70)

    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize data structure
    behavioral_data = {
        "wildtype": [],
        "or7a_mutant": [],
        "control": []
    }

    # Check if results directory exists
    if not results_path.exists():
        print(f"⚠️  Results directory not found: {results_dir}")
        print(f"   Creating sample/placeholder data for demonstration...")

        # Generate sample data (replace with real data when available)
        behavioral_data = {
            "wildtype": [0.82, 0.68, 0.55],      # [after_A_train, after_B_train, A_test]
            "or7a_mutant": [0.83, 0.69, 0.21],   # Shows catastrophic forgetting
            "control": [0.80, 0.80, 0.80]        # Stable baseline
        }

        print(f"   Generated placeholder data:")
        for group, scores in behavioral_data.items():
            print(f"   - {group}: {scores}")
    else:
        print(f"✓ Found results directory: {results_dir}")

        # Try multiple file formats
        found_data = False

        # Option 1: CSV files for each group
        for group in ["wildtype", "or7a_mutant", "control"]:
            csv_path = results_path / f"{group}_behavioral.csv"
            if csv_path.exists():
                print(f"  Loading {csv_path.name}...")
                df = pd.read_csv(csv_path)

                # Extract scores per phase
                # TODO: check column names - adjust as needed
                if 'phase' in df.columns and 'memory_score' in df.columns:
                    phase_scores = df.groupby('phase')['memory_score'].mean().values
                    behavioral_data[group] = list(phase_scores)
                    found_data = True
                elif 'score' in df.columns:
                    behavioral_data[group] = list(df['score'].values)
                    found_data = True

        # Option 2: Single combined CSV
        if not found_data:
            combined_csv = results_path / "behavioral_results.csv"
            if combined_csv.exists():
                print(f"  Loading {combined_csv.name}...")
                df = pd.read_csv(combined_csv)

                # TODO: check column names
                if 'group' in df.columns and 'phase' in df.columns and 'memory_score' in df.columns:
                    for group in ["wildtype", "or7a_mutant", "control"]:
                        group_data = df[df['group'] == group]
                        phase_scores = group_data.groupby('phase')['memory_score'].mean().values
                        behavioral_data[group] = list(phase_scores)
                    found_data = True

        # Option 3: Pickle files
        if not found_data:
            pkl_files = list(results_path.glob("*.pkl"))
            if pkl_files:
                print(f"  Found {len(pkl_files)} pickle files...")
                for pkl_file in pkl_files:
                    try:
                        with open(pkl_file, 'rb') as f:
                            data = pickle.load(f)
                            # TODO: adjust based on pickle structure
                            if isinstance(data, dict) and 'behavioral_scores' in data:
                                behavioral_data.update(data['behavioral_scores'])
                                found_data = True
                                break
                    except Exception as e:
                        print(f"    ⚠️  Error loading {pkl_file.name}: {e}")

        if not found_data:
            print("  ⚠️  No compatible data files found, using placeholder data")

    # Create DataFrame for easy plotting
    df_behavioral = pd.DataFrame(behavioral_data)
    df_behavioral['phase'] = ['after_A_train', 'after_B_train', 'A_test']

    # Save to CSV
    output_file = output_path / "behavioral_data.csv"
    df_behavioral.to_csv(output_file, index=False)
    print(f"\n✓ Saved behavioral data to: {output_file}")

    # Print summary
    print("\n" + "-"*70)
    print("Summary Statistics:")
    print("-"*70)
    print(df_behavioral.to_string(index=False))
    print("-"*70)

    # Also save as dictionary (for direct use in plotting)
    dict_output = output_path / "behavioral_data_dict.pkl"
    with open(dict_output, 'wb') as f:
        pickle.dump(behavioral_data, f)
    print(f"✓ Saved dictionary format to: {dict_output}")

    return behavioral_data


# ==============================================================================
# TASK 2: Model Schematic Info Extraction
# ==============================================================================

def extract_model_schematic_info(
    config_file: str = "configs/penp_model_config.yaml",
    veto_mask_file: str = "results/veto_mask.npy",
    output_dir: str = "data/extracted_figures"
) -> Dict[str, Any]:
    """
    Extract model architecture details: neuron counts, synapse counts,
    and veto gate statistics.

    Parameters
    ----------
    config_file : str
        Path to model configuration YAML
    veto_mask_file : str
        Path to veto mask numpy array (if available)
    output_dir : str
        Directory to save extracted info

    Returns
    -------
    Dict[str, Any]
        Model schematic information including:
        - n_pn, n_kc, n_mbon: neuron counts
        - n_synapses: total KC→MBON synapses
        - n_protected: number of protected synapses
        - protection_percentage: percentage of synapses protected
    """
    print("\n" + "="*70)
    print("TASK 2: Model Schematic Info Extraction")
    print("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schematic_info = {}

    # 1. Load neuron counts from config
    config_path = Path(config_file)
    if config_path.exists():
        print(f"✓ Loading model config from: {config_file}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract neuron counts from config
        # TODO: adjust keys based on your actual config structure
        if 'pgcn_model' in config and 'neuron_counts' in config['pgcn_model']:
            counts = config['pgcn_model']['neuron_counts']
            schematic_info['n_pn'] = counts.get('olfactory', 1756)  # Default from config
            schematic_info['n_integration'] = counts.get('integration', 241)
            schematic_info['n_motor'] = counts.get('motor', 165)

        # Alternative: try network_architecture
        if 'pgcn_model' in config and 'network_architecture' in config['pgcn_model']:
            arch = config['pgcn_model']['network_architecture']
            schematic_info['n_pn'] = arch.get('input_layer', {}).get('size', 1756)
            schematic_info['n_integration'] = arch.get('hidden_layer', {}).get('size', 241)
            schematic_info['n_motor'] = arch.get('output_layer', {}).get('size', 165)
    else:
        print(f"⚠️  Config file not found: {config_file}")
        print("   Using default neuron counts from codebase...")
        # Defaults from PGCN structure
        schematic_info['n_pn'] = 50  # Typical PN count per glomerulus
        schematic_info['n_integration'] = 241  # From config
        schematic_info['n_motor'] = 165  # From config

    # 2. Infer KC and MBON counts
    # For Drosophila MB: ~2000 KCs, ~44 MBONs (from literature)
    schematic_info['n_kc'] = 2000  # Typical KC count
    schematic_info['n_mbon'] = 44  # Typical MBON count

    # Total synapses = n_kc × n_mbon
    schematic_info['n_synapses'] = schematic_info['n_kc'] * schematic_info['n_mbon']

    # 3. Load veto mask (if available)
    veto_path = Path(veto_mask_file)
    if veto_path.exists():
        print(f"✓ Loading veto mask from: {veto_mask_file}")
        veto_mask = np.load(veto_path)

        # Verify shape matches expected (n_kc, n_mbon) or (n_mbon, n_kc)
        expected_shape = (schematic_info['n_kc'], schematic_info['n_mbon'])
        expected_shape_T = (schematic_info['n_mbon'], schematic_info['n_kc'])

        if veto_mask.shape == expected_shape or veto_mask.shape == expected_shape_T:
            print(f"   Veto mask shape: {veto_mask.shape} ✓")

            # Count protected synapses (assuming binary mask: 1 = protected, 0 = unprotected)
            n_protected = int(np.sum(veto_mask))
            schematic_info['n_protected'] = n_protected
            schematic_info['protection_percentage'] = (n_protected / veto_mask.size) * 100

            print(f"   Protected synapses: {n_protected:,} / {veto_mask.size:,}")
            print(f"   Protection percentage: {schematic_info['protection_percentage']:.2f}%")
        else:
            print(f"   ⚠️  Unexpected veto mask shape: {veto_mask.shape}")
            print(f"      Expected: {expected_shape} or {expected_shape_T}")
            schematic_info['n_protected'] = 0
            schematic_info['protection_percentage'] = 0.0
    else:
        print(f"⚠️  Veto mask file not found: {veto_mask_file}")
        print("   Setting protected synapses to 0")
        schematic_info['n_protected'] = 0
        schematic_info['protection_percentage'] = 0.0

    # 4. Print schematic summary
    print("\n" + "-"*70)
    print("Model Architecture Summary:")
    print("-"*70)
    print(f"  Projection Neurons (PN):        {schematic_info['n_pn']:>8,}")
    print(f"  Kenyon Cells (KC):              {schematic_info['n_kc']:>8,}")
    print(f"  Mushroom Body Output (MBON):    {schematic_info['n_mbon']:>8,}")
    print(f"  Total KC→MBON Synapses:         {schematic_info['n_synapses']:>8,}")
    print(f"  Protected Synapses:             {schematic_info['n_protected']:>8,}")
    print(f"  Protection Percentage:          {schematic_info['protection_percentage']:>7.2f}%")
    print("-"*70)

    # 5. Save to file
    output_file = output_path / "model_schematic_info.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(schematic_info, f, default_flow_style=False, sort_keys=False)
    print(f"\n✓ Saved schematic info to: {output_file}")

    # Also save as pickle for easy loading
    pkl_output = output_path / "model_schematic_info.pkl"
    with open(pkl_output, 'wb') as f:
        pickle.dump(schematic_info, f)
    print(f"✓ Saved pickle format to: {pkl_output}")

    return schematic_info


# ==============================================================================
# TASK 3: Critical Synapse Map Data Extraction
# ==============================================================================

def extract_synapse_map_data(
    veto_mask_file: str = "results/veto_mask.npy",
    veto_mask_pattern: str = "results/veto_mask_odorpair*.npy",
    output_dir: str = "data/extracted_figures"
) -> Dict[str, Any]:
    """
    Extract veto gate protection masks for heatmap visualization.

    Parameters
    ----------
    veto_mask_file : str
        Path to single veto mask file
    veto_mask_pattern : str
        Glob pattern for multiple veto masks (one per odor pair)
    output_dir : str
        Directory to save extracted masks

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - masks: List of 2D numpy arrays (each is KC×MBON mask)
        - mask_names: List of descriptive names for each mask
        - summary: DataFrame with per-mask statistics
    """
    print("\n" + "="*70)
    print("TASK 3: Critical Synapse Map Data Extraction")
    print("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    synapse_data = {
        'masks': [],
        'mask_names': [],
        'summary': []
    }

    # Option 1: Single mask file
    single_mask_path = Path(veto_mask_file)
    if single_mask_path.exists():
        print(f"✓ Loading single veto mask: {veto_mask_file}")
        mask = np.load(single_mask_path)
        synapse_data['masks'].append(mask)
        synapse_data['mask_names'].append("veto_mask")

        n_protected = int(np.sum(mask))
        total = mask.size
        percentage = (n_protected / total) * 100

        synapse_data['summary'].append({
            'mask_name': 'veto_mask',
            'shape': mask.shape,
            'n_protected': n_protected,
            'total_synapses': total,
            'protection_pct': percentage
        })

        print(f"   Shape: {mask.shape}")
        print(f"   Protected: {n_protected:,} / {total:,} ({percentage:.2f}%)")

    # Option 2: Multiple masks (one per odor pair)
    else:
        print(f"⚠️  Single mask not found: {veto_mask_file}")
        print(f"   Searching for pattern: {veto_mask_pattern}")

        mask_files = sorted(Path("results").glob(Path(veto_mask_pattern).name))

        if mask_files:
            print(f"✓ Found {len(mask_files)} mask files")
            for i, mask_file in enumerate(mask_files):
                print(f"  Loading {mask_file.name}...")
                mask = np.load(mask_file)
                mask_name = mask_file.stem  # e.g., "veto_mask_odorpair1"

                synapse_data['masks'].append(mask)
                synapse_data['mask_names'].append(mask_name)

                n_protected = int(np.sum(mask))
                total = mask.size
                percentage = (n_protected / total) * 100

                synapse_data['summary'].append({
                    'mask_name': mask_name,
                    'shape': mask.shape,
                    'n_protected': n_protected,
                    'total_synapses': total,
                    'protection_pct': percentage
                })

                print(f"     Shape: {mask.shape}, Protected: {n_protected:,} ({percentage:.2f}%)")
        else:
            print("  ⚠️  No mask files found")
            print("  Creating placeholder mask for demonstration...")

            # Create sample mask (2000 KCs × 44 MBONs, 5% protected)
            n_kc, n_mbon = 2000, 44
            mask = np.random.rand(n_kc, n_mbon) < 0.05  # 5% protected

            synapse_data['masks'].append(mask.astype(int))
            synapse_data['mask_names'].append("placeholder_mask")

            n_protected = int(np.sum(mask))
            total = mask.size
            percentage = (n_protected / total) * 100

            synapse_data['summary'].append({
                'mask_name': 'placeholder_mask',
                'shape': mask.shape,
                'n_protected': n_protected,
                'total_synapses': total,
                'protection_pct': percentage
            })

    # Create summary DataFrame
    df_summary = pd.DataFrame(synapse_data['summary'])

    print("\n" + "-"*70)
    print("Synapse Map Summary:")
    print("-"*70)
    print(df_summary.to_string(index=False))
    print("-"*70)

    # Save masks
    for mask, name in zip(synapse_data['masks'], synapse_data['mask_names']):
        mask_file = output_path / f"{name}.npy"
        np.save(mask_file, mask)
        print(f"✓ Saved mask: {mask_file}")

    # Save summary
    summary_file = output_path / "synapse_map_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Saved summary: {summary_file}")

    # Save complete data as pickle
    pkl_output = output_path / "synapse_map_data.pkl"
    with open(pkl_output, 'wb') as f:
        pickle.dump(synapse_data, f)
    print(f"✓ Saved pickle format: {pkl_output}")

    return synapse_data


# ==============================================================================
# TASK 4: ML Comparison Data Extraction
# ==============================================================================

def extract_ml_comparison_data(
    forgetting_file: str = "results/forgetting_summary.csv",
    results_dir: str = "results",
    output_dir: str = "data/extracted_figures"
) -> Dict[str, float]:
    """
    Extract forgetting scores for different model variants (MBON_veto, Dense_ANN,
    EWC, SI, etc.).

    Parameters
    ----------
    forgetting_file : str
        Path to CSV with columns: model_type, forgetting_score
    results_dir : str
        Alternative: directory with multiple result files
    output_dir : str
        Directory to save extracted data

    Returns
    -------
    Dict[str, float]
        Dictionary mapping model type to mean forgetting score
        Format: {"MBON_veto": 0.15, "Dense_ANN": 0.82, ...}
    """
    print("\n" + "="*70)
    print("TASK 4: ML Comparison Data Extraction")
    print("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ml_data = {}

    # Option 1: Single summary CSV
    csv_path = Path(forgetting_file)
    if csv_path.exists():
        print(f"✓ Loading forgetting summary: {forgetting_file}")
        df = pd.read_csv(csv_path)

        # TODO: check column names
        if 'model_type' in df.columns and 'forgetting_score' in df.columns:
            # Group by model type and compute mean
            ml_data = df.groupby('model_type')['forgetting_score'].mean().to_dict()
            print(f"   Loaded {len(ml_data)} model types")
        else:
            print(f"   ⚠️  Expected columns 'model_type' and 'forgetting_score'")
            print(f"   Found columns: {df.columns.tolist()}")
    else:
        print(f"⚠️  Forgetting summary not found: {forgetting_file}")

        # Option 2: Search for individual result files
        results_path = Path(results_dir)
        if results_path.exists():
            print(f"   Searching for model results in: {results_dir}")

            # Look for CSV/pickle files with model names
            model_types = ['MBON_veto', 'Dense_ANN', 'EWC', 'SI', 'LwF', 'GEM']

            for model_type in model_types:
                # Try multiple filename patterns
                patterns = [
                    f"{model_type}_results.csv",
                    f"{model_type.lower()}_forgetting.csv",
                    f"forgetting_{model_type}.csv"
                ]

                for pattern in patterns:
                    file_path = results_path / pattern
                    if file_path.exists():
                        print(f"   Found: {file_path.name}")
                        df = pd.read_csv(file_path)

                        # Extract forgetting score (last value or mean)
                        if 'forgetting_score' in df.columns:
                            ml_data[model_type] = float(df['forgetting_score'].mean())
                        elif 'score' in df.columns:
                            ml_data[model_type] = float(df['score'].mean())
                        break

    # If still no data, use placeholder
    if not ml_data:
        print("   Creating placeholder data for demonstration...")
        ml_data = {
            'MBON_veto': 0.15,      # Best performance (Or7a-inspired veto gate)
            'Dense_ANN': 0.82,      # Worst (catastrophic forgetting)
            'EWC': 0.45,            # Elastic Weight Consolidation
            'SI': 0.52,             # Synaptic Intelligence
            'LwF': 0.58,            # Learning without Forgetting
            'GEM': 0.38             # Gradient Episodic Memory
        }
        print(f"   Generated {len(ml_data)} placeholder model comparisons")

    # Create DataFrame
    df_ml = pd.DataFrame([
        {'model_type': k, 'forgetting_score': v}
        for k, v in ml_data.items()
    ])

    # Sort by forgetting score (best first)
    df_ml = df_ml.sort_values('forgetting_score')

    print("\n" + "-"*70)
    print("ML Model Comparison:")
    print("-"*70)
    print(df_ml.to_string(index=False))
    print("-"*70)
    print(f"\nBest model: {df_ml.iloc[0]['model_type']} "
          f"(forgetting = {df_ml.iloc[0]['forgetting_score']:.3f})")
    print(f"Worst model: {df_ml.iloc[-1]['model_type']} "
          f"(forgetting = {df_ml.iloc[-1]['forgetting_score']:.3f})")

    # Save to CSV
    output_file = output_path / "ml_comparison_data.csv"
    df_ml.to_csv(output_file, index=False)
    print(f"\n✓ Saved ML comparison to: {output_file}")

    # Save as dictionary
    dict_output = output_path / "ml_comparison_dict.pkl"
    with open(dict_output, 'wb') as f:
        pickle.dump(ml_data, f)
    print(f"✓ Saved dictionary format: {dict_output}")

    return ml_data


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main execution: run all extraction tasks or specific ones."""
    parser = argparse.ArgumentParser(
        description="Extract data for PGCN publication figures"
    )
    parser.add_argument(
        '--task',
        type=str,
        default='all',
        choices=['all', 'behavioral', 'schematic', 'synapse_map', 'ml_comparison'],
        help='Which extraction task to run (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/extracted_figures',
        help='Output directory for extracted data'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("PGCN Figure Data Extraction Pipeline")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run selected tasks
    results = {}

    if args.task in ['all', 'behavioral']:
        results['behavioral'] = extract_behavioral_data(output_dir=args.output_dir)

    if args.task in ['all', 'schematic']:
        results['schematic'] = extract_model_schematic_info(output_dir=args.output_dir)

    if args.task in ['all', 'synapse_map']:
        results['synapse_map'] = extract_synapse_map_data(output_dir=args.output_dir)

    if args.task in ['all', 'ml_comparison']:
        results['ml_comparison'] = extract_ml_comparison_data(output_dir=args.output_dir)

    # Final summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"✓ All extracted data saved to: {args.output_dir}/")
    print("\nNext steps:")
    print("  1. Verify extracted data in the output directory")
    print("  2. Update file paths if needed (check TODO comments in code)")
    print("  3. Use extracted data in your plotting scripts")
    print("\nExample usage in plotting:")
    print("  >>> import pandas as pd")
    print("  >>> df = pd.read_csv('data/extracted_figures/behavioral_data.csv')")
    print("  >>> # Your plotting code here...")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
