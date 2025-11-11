#!/usr/bin/env python3
"""
Or7a Extended Analyses - Follow-up analyses for dual veto mechanism

This module provides 3 additional analyses based on initial findings:
5. Serotonergic pathway characterization
6. Synapse-weighted KC overlap refinement
7. DP1m hub detailed analysis

Plus supplementary figure generation.

To integrate into analyze_or7a_dual_veto.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analysis_5_serotonin_pathways(neurons, connections, ln_ids, output_dir):
    """
    Analysis 5: Serotonergic LN pathway characterization

    Question: Do serotonergic LNs project to mushroom body (central modulation)
              or stay within antennal lobe (peripheral modulation)?

    Args:
        neurons: DataFrame with neuron metadata including nt_type
        connections: DataFrame with synaptic connections
        ln_ids: List of cross-glomerular LN root_ids
        output_dir: Path to save results

    Returns:
        DataFrame with NT-specific connectivity statistics
    """
    print("\n" + "="*80)
    print("ANALYSIS 5: SEROTONERGIC PATHWAY CHARACTERIZATION")
    print("="*80)

    if 'nt_type' not in neurons.columns:
        print("⚠️  Skipping - nt_type column not available")
        return None

    # Separate LNs by neurotransmitter
    ln_neurons = neurons[neurons['root_id'].isin(ln_ids)].copy()

    # Normalize NT type names
    ln_neurons['nt_type'] = ln_neurons['nt_type'].str.upper()

    # Map common variations
    nt_mapping = {
        'GABA': 'GABA',
        'SER': 'SER',
        'SEROTONIN': 'SER',
        'ACH': 'ACH',
        'ACETYLCHOLINE': 'ACH',
        'GLUT': 'GLUT',
        'GLUTAMATE': 'GLUT'
    }
    ln_neurons['nt_type'] = ln_neurons['nt_type'].map(lambda x: nt_mapping.get(x, x))

    ser_lns = ln_neurons[ln_neurons['nt_type'] == 'SER']['root_id'].tolist()
    gaba_lns = ln_neurons[ln_neurons['nt_type'] == 'GABA']['root_id'].tolist()
    ach_lns = ln_neurons[ln_neurons['nt_type'] == 'ACH']['root_id'].tolist()

    print(f"\nAnalyzing pathway targets by neurotransmitter:")
    print(f"  SER LNs: {len(ser_lns)}")
    print(f"  GABA LNs: {len(gaba_lns)}")
    print(f"  ACH LNs: {len(ach_lns)}")

    # Get syn_count column name
    syn_col = 'size' if 'size' in connections.columns else 'syn_count'

    # Get downstream connectivity for each NT type
    results = {}
    for nt_name, ln_list in [('SER', ser_lns), ('GABA', gaba_lns), ('ACH', ach_lns)]:
        if len(ln_list) == 0:
            continue

        # Find all connections from these LNs
        downstream = connections[connections['pre_root_id'].isin(ln_list)].copy()

        # Categorize by target neuron type
        mb_synapses = 0
        al_synapses = 0
        lh_synapses = 0
        other_synapses = 0

        if 'post_class' in downstream.columns:
            # Use post_class to identify targets
            for _, row in downstream.iterrows():
                post_class = str(row.get('post_class', '')).upper()
                synapses = row[syn_col]

                if any(x in post_class for x in ['KC', 'MBON', 'DAN']):
                    mb_synapses += synapses
                elif any(x in post_class for x in ['PN', 'LN']):
                    al_synapses += synapses
                elif 'LH' in post_class:
                    lh_synapses += synapses
                else:
                    other_synapses += synapses

        total_synapses = downstream[syn_col].sum()

        results[nt_name] = {
            'ln_count': len(ln_list),
            'total_synapses': int(total_synapses),
            'mb_synapses': int(mb_synapses),
            'al_synapses': int(al_synapses),
            'lh_synapses': int(lh_synapses),
            'other_synapses': int(other_synapses),
            'mb_ratio': mb_synapses / total_synapses if total_synapses > 0 else 0,
            'al_ratio': al_synapses / total_synapses if total_synapses > 0 else 0,
            'mean_syn_per_ln': total_synapses / len(ln_list) if len(ln_list) > 0 else 0
        }

        print(f"\n{nt_name} LNs ({len(ln_list)}):")
        print(f"  Total output: {total_synapses:,} synapses")
        print(f"  → Mushroom Body: {mb_synapses:,} ({results[nt_name]['mb_ratio']:.1%})")
        print(f"  → Antennal Lobe: {al_synapses:,} ({results[nt_name]['al_ratio']:.1%})")
        print(f"  → Lateral Horn: {lh_synapses:,}")
        print(f"  → Other: {other_synapses:,}")
        print(f"  Mean syn/LN: {results[nt_name]['mean_syn_per_ln']:.1f}")

    # Statistical comparison
    if 'SER' in results and 'GABA' in results:
        ser_mb_ratio = results['SER']['mb_ratio']
        gaba_mb_ratio = results['GABA']['mb_ratio']

        if gaba_mb_ratio > 0:
            enrichment = ser_mb_ratio / gaba_mb_ratio

            if ser_mb_ratio > gaba_mb_ratio * 1.5:
                print(f"\n✅ SEROTONERGIC LNs show {enrichment:.1f}x enrichment for MB projections")
                print("   Interpretation: Serotonin preferentially modulates central learning circuits")
            elif ser_mb_ratio < gaba_mb_ratio * 0.67:
                print(f"\n⚠️  SEROTONERGIC LNs show {1/enrichment:.1f}x DEPLETION for MB projections")
                print("   Interpretation: Serotonin preferentially modulates local AL circuits")
            else:
                print(f"\n⚪ No significant enrichment (SER:{ser_mb_ratio:.1%} vs GABA:{gaba_mb_ratio:.1%})")

    # Save results
    results_df = pd.DataFrame(results).T
    output_path = Path(output_dir) / 'analysis5_serotonin_pathways.csv'
    results_df.to_csv(output_path)
    print(f"\n✅ Saved: {output_path.name}")

    return results_df


def analysis_6_kc_overlap_weighted(connections, labels, dl5_pns, dm_pns, output_dir):
    """
    Analysis 6: Synapse-weighted KC overlap

    Question: Does filtering by synapse strength reduce overlap to ~25%?

    Args:
        connections: DataFrame with synaptic connections
        labels: DataFrame with glomerulus labels
        dl5_pns: Array of DL5 PN root_ids
        dm_pns: Array of DM PN root_ids
        output_dir: Path to save results

    Returns:
        DataFrame with overlap statistics at different thresholds
    """
    print("\n" + "="*80)
    print("ANALYSIS 6: SYNAPSE-WEIGHTED KC OVERLAP")
    print("="*80)

    # Get syn_count column name
    syn_col = 'size' if 'size' in connections.columns else 'syn_count'

    # Filter for PN→KC connections
    pn_kc = connections[
        (connections['pre_root_id'].isin(np.concatenate([dl5_pns, dm_pns])))
    ].copy()

    # Identify KCs by post class
    if 'post_class' in pn_kc.columns:
        pn_kc = pn_kc[pn_kc['post_class'].str.contains('KC', na=False, case=False)]

    # Merge with glomerulus labels
    if labels is not None:
        pn_kc = pn_kc.merge(labels[['root_id', 'glomerulus']],
                            left_on='pre_root_id', right_on='root_id',
                            how='left', suffixes=('', '_label'))

    print(f"\nAnalyzing {len(pn_kc):,} PN→KC connections")

    # Test multiple synapse thresholds
    thresholds = [1, 2, 3, 5, 7, 10, 15, 20]
    results = []

    for threshold in thresholds:
        pn_kc_filtered = pn_kc[pn_kc[syn_col] >= threshold]

        dl5_kcs = set(pn_kc_filtered[
            pn_kc_filtered['glomerulus'] == 'DL5'
        ]['post_root_id'])

        dm_kcs = set(pn_kc_filtered[
            pn_kc_filtered['glomerulus'].isin(['DM1','DM2','DM3','DM4'])
        ]['post_root_id'])

        if len(dl5_kcs) == 0:
            continue

        shared = dl5_kcs & dm_kcs
        overlap_pct = len(shared) / len(dl5_kcs) * 100

        # Calculate mean synapse strength for shared KCs
        if len(shared) > 0:
            shared_conns = pn_kc_filtered[
                pn_kc_filtered['post_root_id'].isin(shared)
            ]
            mean_strength = shared_conns[syn_col].mean()
        else:
            mean_strength = 0

        results.append({
            'threshold': threshold,
            'dl5_kcs': len(dl5_kcs),
            'dm_kcs': len(dm_kcs),
            'shared_kcs': len(shared),
            'overlap_pct': overlap_pct,
            'mean_synapse_strength': mean_strength,
            'distance_from_25pct': abs(overlap_pct - 25.0)
        })

        marker = "🎯" if abs(overlap_pct - 25.0) < 3.0 else "  "
        print(f"{marker} Threshold ≥{threshold:2d} synapses: "
              f"DL5={len(dl5_kcs):4d}, Shared={len(shared):3d} ({overlap_pct:5.1f}%)")

    results_df = pd.DataFrame(results)

    # Find threshold closest to 25%
    if len(results_df) > 0:
        closest_idx = results_df['distance_from_25pct'].idxmin()
        optimal = results_df.loc[closest_idx]

        print(f"\n🎯 OPTIMAL THRESHOLD: ≥{optimal['threshold']} synapses")
        print(f"   Overlap: {optimal['overlap_pct']:.1f}% (target: 25%)")
        print(f"   DL5 KCs: {optimal['dl5_kcs']}")
        print(f"   Shared KCs: {optimal['shared_kcs']}")

        if abs(optimal['overlap_pct'] - 25.0) < 5.0:
            print(f"   ✅ Closely matches behavioral cross-learning effect!")
        else:
            print(f"   ⚠️  Deviation: {optimal['overlap_pct'] - 25.0:+.1f} percentage points")

        # Interpretation
        print("\n📊 Interpretation:")
        if optimal['threshold'] > 1:
            print(f"   Anatomical overlap (all connections): {results_df.iloc[0]['overlap_pct']:.1f}%")
            print(f"   Functional overlap (≥{optimal['threshold']} syn): {optimal['overlap_pct']:.1f}%")
            print(f"   → Weak connections inflate anatomical estimate")
            print(f"   → Strong connections drive behavioral effect")

    # Save results
    output_path = Path(output_dir) / 'analysis6_kc_overlap_weighted.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path.name}")

    return results_df


def analysis_7_dp1m_hub(ln_cross, output_dir):
    """
    Analysis 7: DP1m as aversive relay hub

    Question: Does DP1m receive input from multiple aversive glomeruli?

    Args:
        ln_cross: DataFrame with cross-glomerular LN connectivity
        output_dir: Path to save results

    Returns:
        Dict with DP1m inputs and outputs DataFrames
    """
    print("\n" + "="*80)
    print("ANALYSIS 7: DP1M HUB CHARACTERIZATION")
    print("="*80)

    # Known glomerular valences (from literature)
    aversive_gloms = ['DL5', 'DA1', 'VA1v', 'DL3', 'DC3', 'VA7m', 'DL2d', 'DL2v']
    appetitive_gloms = ['DM1', 'DM2', 'DM3', 'DM4', 'DM5', 'DM6', 'VA2', 'VC3', 'VC2']

    # DP1m inputs
    dp1m_inputs = ln_cross[ln_cross['target_glom'] == 'DP1m'].sort_values(
        'total_synapses', ascending=False
    ).copy()

    if len(dp1m_inputs) == 0:
        print("⚠️  No DP1m connections found in LN cross-glomerular data")
        return None

    print(f"\nDP1m receives input from {len(dp1m_inputs)} glomeruli")
    print(f"Total input: {dp1m_inputs['total_synapses'].sum():,} synapses\n")

    print("Top 15 sources:")
    for idx, row in dp1m_inputs.head(15).iterrows():
        valence = "🔴 AVERSIVE " if row['source_glom'] in aversive_gloms else \
                  "🟢 APPETITIVE" if row['source_glom'] in appetitive_gloms else \
                  "⚪ UNKNOWN   "
        print(f"  {row['source_glom']:6} → DP1m: {row['total_synapses']:5,.0f} syn  "
              f"({row['ln_count']:2.0f} LNs)  {valence}")

    # Calculate valence balance
    aversive_input = dp1m_inputs[
        dp1m_inputs['source_glom'].isin(aversive_gloms)
    ]['total_synapses'].sum()

    appetitive_input = dp1m_inputs[
        dp1m_inputs['source_glom'].isin(appetitive_gloms)
    ]['total_synapses'].sum()

    total_classified = aversive_input + appetitive_input

    if total_classified > 0:
        aversive_ratio = aversive_input / total_classified

        print(f"\nValence Balance (classified inputs only):")
        print(f"  Aversive input:   {aversive_input:,} syn ({aversive_ratio:.1%})")
        print(f"  Appetitive input: {appetitive_input:,} syn ({1-aversive_ratio:.1%})")

        if aversive_ratio > 0.6:
            print(f"\n✅ DP1m is AVERSIVE-DOMINATED hub ({aversive_ratio:.0%})")
            print("   Interpretation: DP1m amplifies aversive signals to DM glomeruli")
        elif aversive_ratio < 0.4:
            print(f"\n⚠️  DP1m is APPETITIVE-DOMINATED hub ({1-aversive_ratio:.0%})")
        else:
            print(f"\n⚪ DP1m receives balanced aversive/appetitive input")

    # DP1m outputs
    dp1m_outputs = ln_cross[ln_cross['source_glom'] == 'DP1m'].sort_values(
        'total_synapses', ascending=False
    ).copy()

    print(f"\nDP1m projects to {len(dp1m_outputs)} glomeruli")
    print(f"Total output: {dp1m_outputs['total_synapses'].sum():,} synapses\n")

    print("Top 15 targets:")
    for idx, row in dp1m_outputs.head(15).iterrows():
        valence = "🔴 AVERSIVE " if row['target_glom'] in aversive_gloms else \
                  "🟢 APPETITIVE" if row['target_glom'] in appetitive_gloms else \
                  "⚪ UNKNOWN   "
        print(f"  DP1m → {row['target_glom']:6}: {row['total_synapses']:5,.0f} syn  "
              f"({row['ln_count']:2.0f} LNs)  {valence}")

    # Analyze output valence
    aversive_output = dp1m_outputs[
        dp1m_outputs['target_glom'].isin(aversive_gloms)
    ]['total_synapses'].sum()

    appetitive_output = dp1m_outputs[
        dp1m_outputs['target_glom'].isin(appetitive_gloms)
    ]['total_synapses'].sum()

    total_output_classified = aversive_output + appetitive_output

    if total_output_classified > 0:
        output_appetitive_ratio = appetitive_output / total_output_classified

        print(f"\nOutput Valence Balance:")
        print(f"  → Aversive glomeruli:   {aversive_output:,} syn ({1-output_appetitive_ratio:.1%})")
        print(f"  → Appetitive glomeruli: {appetitive_output:,} syn ({output_appetitive_ratio:.1%})")

        if output_appetitive_ratio > 0.5:
            print(f"\n🎯 DP1m primarily INHIBITS APPETITIVE glomeruli ({output_appetitive_ratio:.0%})")
            print("   Interpretation: Aversive input → DP1m → inhibit appetitive responses")
            print("   This explains the DL5→DP1m→DM pathway for veto mechanism!")

    # Save
    output_path_inputs = Path(output_dir) / 'analysis7_dp1m_inputs.csv'
    output_path_outputs = Path(output_dir) / 'analysis7_dp1m_outputs.csv'

    dp1m_inputs.to_csv(output_path_inputs, index=False)
    dp1m_outputs.to_csv(output_path_outputs, index=False)

    print(f"\n✅ Saved: {output_path_inputs.name}")
    print(f"✅ Saved: {output_path_outputs.name}")

    return {
        'inputs': dp1m_inputs,
        'outputs': dp1m_outputs,
        'aversive_ratio_input': aversive_ratio if total_classified > 0 else None,
        'appetitive_ratio_output': output_appetitive_ratio if total_output_classified > 0 else None
    }


def generate_supplementary_figures(results, output_dir):
    """
    Generate supplementary figures for analyses 5, 6, 7

    Args:
        results: Dict containing all analysis results
        output_dir: Path to save figures
    """
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY FIGURES")
    print("="*80)

    # Supplementary Figure 1: NT pathway comparison
    if results.get('analysis5') is not None:
        generate_suppfig1_nt_pathways(results['analysis5'], output_dir)

    # Supplementary Figure 2: KC overlap vs threshold
    if results.get('analysis6') is not None:
        generate_suppfig2_kc_threshold(results['analysis6'], output_dir)

    # Supplementary Figure 3: DP1m hub network
    if results.get('analysis7') is not None:
        generate_suppfig3_dp1m_network(results['analysis7'], output_dir)


def generate_suppfig1_nt_pathways(analysis5_results, output_dir):
    """Supplementary Figure 1: Neurotransmitter pathway targeting"""
    print("\nGenerating Supplementary Figure 1: NT Pathway Targeting...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Target distribution by NT
    ax = axes[0]

    nt_types = analysis5_results.index.tolist()
    targets = ['MB', 'AL', 'LH', 'Other']

    data = {
        'MB': analysis5_results['mb_synapses'].values,
        'AL': analysis5_results['al_synapses'].values,
        'LH': analysis5_results['lh_synapses'].values,
        'Other': analysis5_results['other_synapses'].values
    }

    x = np.arange(len(nt_types))
    width = 0.2

    colors = ['#CC79A7', '#009E73', '#E69F00', '#CCCCCC']

    for i, (target, color) in enumerate(zip(targets, colors)):
        ax.bar(x + i*width, data[target], width, label=target, color=color, alpha=0.8)

    ax.set_xlabel('Neurotransmitter Type', fontsize=11)
    ax.set_ylabel('Synapses', fontsize=11)
    ax.set_title('A. Target Distribution by NT', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(nt_types)
    ax.legend()
    ax.set_yscale('log')

    # Panel B: MB enrichment ratios
    ax = axes[1]

    mb_ratios = analysis5_results['mb_ratio'].values * 100
    colors_nt = ['#D55E00' if nt == 'GABA' else '#E69F00' if nt == 'SER' else '#0173B2'
                 for nt in nt_types]

    bars = ax.bar(nt_types, mb_ratios, color=colors_nt, alpha=0.7, edgecolor='black')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_ylabel('MB Projection (%)', fontsize=11)
    ax.set_xlabel('Neurotransmitter Type', fontsize=11)
    ax.set_title('B. MB Enrichment', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()

    # Add values on bars
    for bar, val in zip(bars, mb_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Panel C: Synapses per LN
    ax = axes[2]

    mean_syn = analysis5_results['mean_syn_per_ln'].values

    bars = ax.bar(nt_types, mean_syn, color=colors_nt, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Mean Synapses per LN', fontsize=11)
    ax.set_xlabel('Neurotransmitter Type', fontsize=11)
    ax.set_title('C. Output Strength', fontsize=12, fontweight='bold')

    # Add values on bars
    for bar, val in zip(bars, mean_syn):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
               f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = Path(output_dir) / 'suppfig1_nt_pathway_targeting.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path.name}")


def generate_suppfig2_kc_threshold(analysis6_results, output_dir):
    """Supplementary Figure 2: KC overlap vs synapse threshold"""
    print("\nGenerating Supplementary Figure 2: KC Overlap Threshold Analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Overlap vs threshold
    ax = axes[0]

    thresholds = analysis6_results['threshold'].values
    overlap_pct = analysis6_results['overlap_pct'].values

    ax.plot(thresholds, overlap_pct, 'o-', linewidth=2, markersize=8,
           color='#0173B2', label='Observed overlap')
    ax.axhline(25, color='red', linestyle='--', linewidth=2,
              alpha=0.7, label='Behavioral target (25%)')

    # Highlight optimal threshold
    closest_idx = analysis6_results['distance_from_25pct'].idxmin()
    optimal = analysis6_results.loc[closest_idx]

    ax.plot(optimal['threshold'], optimal['overlap_pct'], 'r*',
           markersize=20, label=f'Optimal (≥{optimal["threshold"]} syn)')

    ax.set_xlabel('Minimum Synapse Threshold', fontsize=11)
    ax.set_ylabel('KC Overlap (%)', fontsize=11)
    ax.set_title('A. Overlap vs Synapse Strength', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(overlap_pct) * 1.1)

    # Panel B: KC counts vs threshold
    ax = axes[1]

    ax.plot(thresholds, analysis6_results['dl5_kcs'], 'o-', linewidth=2,
           markersize=6, color='#CC3311', label='DL5 KCs', alpha=0.7)
    ax.plot(thresholds, analysis6_results['shared_kcs'], 's-', linewidth=2,
           markersize=6, color='#882255', label='Shared KCs', alpha=0.7)

    ax.set_xlabel('Minimum Synapse Threshold', fontsize=11)
    ax.set_ylabel('KC Count', fontsize=11)
    ax.set_title('B. KC Counts vs Threshold', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Add annotation for optimal point
    ax.axvline(optimal['threshold'], color='red', linestyle=':', alpha=0.5)
    ax.text(optimal['threshold'], ax.get_ylim()[1] * 0.5,
           f"Optimal\n≥{optimal['threshold']} syn",
           ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_path = Path(output_dir) / 'suppfig2_kc_overlap_threshold.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path.name}")


def generate_suppfig3_dp1m_network(analysis7_results, output_dir):
    """Supplementary Figure 3: DP1m hub network diagram"""
    print("\nGenerating Supplementary Figure 3: DP1m Hub Network...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    dp1m_inputs = analysis7_results['inputs']
    dp1m_outputs = analysis7_results['outputs']

    # Known valences
    aversive_gloms = ['DL5', 'DA1', 'VA1v', 'DL3', 'DC3', 'VA7m']
    appetitive_gloms = ['DM1', 'DM2', 'DM3', 'DM4', 'DM5', 'DM6', 'VA2']

    # Panel A: Top inputs
    ax = axes[0]

    top_inputs = dp1m_inputs.head(10)
    y_pos = np.arange(len(top_inputs))

    colors = ['#D55E00' if glom in aversive_gloms else
             '#009E73' if glom in appetitive_gloms else
             '#CCCCCC'
             for glom in top_inputs['source_glom']]

    ax.barh(y_pos, top_inputs['total_synapses'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_inputs['source_glom'], fontsize=10)
    ax.set_xlabel('Synapses to DP1m', fontsize=11)
    ax.set_title('A. Top DP1m Inputs', fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D55E00', alpha=0.7, label='Aversive'),
        Patch(facecolor='#009E73', alpha=0.7, label='Appetitive'),
        Patch(facecolor='#CCCCCC', alpha=0.7, label='Unknown')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Panel B: Top outputs
    ax = axes[1]

    top_outputs = dp1m_outputs.head(10)
    y_pos = np.arange(len(top_outputs))

    colors = ['#D55E00' if glom in aversive_gloms else
             '#009E73' if glom in appetitive_gloms else
             '#CCCCCC'
             for glom in top_outputs['target_glom']]

    ax.barh(y_pos, top_outputs['total_synapses'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_outputs['target_glom'], fontsize=10)
    ax.set_xlabel('Synapses from DP1m', fontsize=11)
    ax.set_title('B. Top DP1m Outputs', fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    # Add legend
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()

    output_path = Path(output_dir) / 'suppfig3_dp1m_hub_network.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path.name}")
