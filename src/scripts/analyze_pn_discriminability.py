#!/usr/bin/env python3
"""
Analyze discriminability of DoOR-based PN patterns.

This script quantifies whether DoOR olfactory receptor responses,
when mapped to FlyWire projection neurons, provide sufficient
discriminability for the CCBPN model to learn odor associations.

Run from repository root:
    python src/scripts/analyze_pn_discriminability.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import logging
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

# Import from PGCN codebase
from src.pgcn.data.door_integration import DoORIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run complete PN discriminability analysis."""

    # Configuration
    cache_dir = Path('data/cache')
    output_dir = Path('results/pn_discriminability_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Experimental odors (from dataset mapping)
    odors = [
        'hexanol',
        'benzaldehyde',
        'ethyl_butyrate',
        '3-octanol',
        'citral',
        'linalool',
        'apple_cider_vinegar',
        'air'  # Control
    ]

    # Initialize DoOR integration
    logger.info("Initializing DoOR integration...")
    door = DoORIntegration(cache_dir)

    # Get PN patterns (no noise)
    logger.info("Generating canonical PN patterns...")
    n_pn = 150  # From FlyWire connectivity
    patterns = {}
    for odor in odors:
        pattern = door.odor_to_pn_activity(odor_name=odor, n_pn=n_pn)
        patterns[odor] = pattern
        n_active = np.sum(pattern > 0.1)
        logger.info(f"  {odor:25s}: {n_active:3d} active PNs (mean={pattern.mean():.3f})")

    # Analysis 1: Pairwise Correlations (No Noise)
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 1: Pairwise Correlations (Clean Patterns)")
    logger.info("="*60)

    correlation_matrix = compute_correlation_matrix(patterns, odors)
    plot_correlation_heatmap(correlation_matrix, odors, output_dir / 'pn_correlation_heatmap.png')

    # Find most/least similar pairs
    most_similar, least_similar = find_extreme_pairs(correlation_matrix, odors)
    logger.info(f"\nMost similar pair: {most_similar[0]} vs {most_similar[1]} (r={most_similar[2]:.3f})")
    logger.info(f"Least similar pair: {least_similar[0]} vs {least_similar[1]} (r={least_similar[2]:.3f})")

    # Analysis 2: Signal-to-Noise Ratio with 8% Noise
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 2: Signal-to-Noise Ratio (8% Additive Noise)")
    logger.info("="*60)

    snr_results = compute_snr_with_noise(patterns, odors, noise_std=0.08, n_trials=100)
    plot_snr_results(snr_results, odors, output_dir / 'snr_analysis.png')

    # Analysis 3: PCA Visualization
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 3: PCA Visualization of PN Space")
    logger.info("="*60)

    pca_results = perform_pca_analysis(patterns, odors, noise_std=0.08, n_trials_per_odor=50)
    plot_pca(pca_results, odors, output_dir / 'pn_pca.png')

    # Analysis 4: KC-Level Discriminability
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 4: KC-Level Discriminability (After Sparse Coding)")
    logger.info("="*60)

    kc_results = analyze_kc_discriminability(patterns, odors, cache_dir, noise_std=0.08)
    plot_kc_distances(kc_results, odors, output_dir / 'kc_distances.png')

    # Generate Report
    logger.info("\n" + "="*60)
    logger.info("Generating comprehensive report...")
    logger.info("="*60)

    generate_report(
        output_dir / 'PN_DISCRIMINABILITY_REPORT.md',
        correlation_matrix,
        snr_results,
        pca_results,
        kc_results,
        odors
    )

    logger.info(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    logger.info(f"📄 See report: {output_dir / 'PN_DISCRIMINABILITY_REPORT.md'}")


def compute_correlation_matrix(patterns: dict, odors: list) -> np.ndarray:
    """Compute pairwise correlations between odor patterns."""
    n_odors = len(odors)
    corr_matrix = np.zeros((n_odors, n_odors))

    for i, odor1 in enumerate(odors):
        for j, odor2 in enumerate(odors):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                r, _ = pearsonr(patterns[odor1], patterns[odor2])
                corr_matrix[i, j] = r

    return corr_matrix


def find_extreme_pairs(corr_matrix: np.ndarray, odors: list) -> tuple:
    """Find most and least similar odor pairs."""
    n = len(odors)
    most_similar = (None, None, -1.0)
    least_similar = (None, None, 2.0)

    for i in range(n):
        for j in range(i+1, n):
            corr = corr_matrix[i, j]
            if corr > most_similar[2]:
                most_similar = (odors[i], odors[j], corr)
            if corr < least_similar[2]:
                least_similar = (odors[i], odors[j], corr)

    return most_similar, least_similar


def compute_snr_with_noise(patterns: dict, odors: list, noise_std: float, n_trials: int) -> dict:
    """Compute signal-to-noise ratio for each odor with additive Gaussian noise."""
    snr_results = {}

    for odor in odors:
        clean_pattern = patterns[odor]

        # Generate noisy versions
        noisy_patterns = []
        for _ in range(n_trials):
            noise = np.random.randn(len(clean_pattern)) * noise_std
            noisy_pattern = np.clip(clean_pattern + noise, 0, 1)
            noisy_patterns.append(noisy_pattern)

        noisy_patterns = np.array(noisy_patterns)  # (n_trials, n_pn)

        # Compute signal and noise magnitudes
        signal = np.mean(noisy_patterns, axis=0)  # Mean across trials
        noise = np.std(noisy_patterns, axis=0)    # Std across trials

        # SNR = mean / std for each PN
        snr_per_pn = np.divide(signal, noise, where=noise > 0, out=np.zeros_like(signal))

        # Average SNR across active PNs
        active_mask = clean_pattern > 0.1
        mean_snr = np.mean(snr_per_pn[active_mask]) if np.any(active_mask) else 0.0

        # Correlation between clean and noisy mean
        mean_corr = pearsonr(clean_pattern, signal)[0]

        snr_results[odor] = {
            'mean_snr': mean_snr,
            'correlation_with_clean': mean_corr,
            'n_active_pns': np.sum(active_mask)
        }

        logger.info(f"  {odor:25s}: SNR={mean_snr:.2f}, corr={mean_corr:.3f}")

    return snr_results


def perform_pca_analysis(patterns: dict, odors: list, noise_std: float, n_trials_per_odor: int) -> dict:
    """Perform PCA on PN patterns with noise."""
    all_patterns = []
    all_labels = []

    for odor in odors:
        clean_pattern = patterns[odor]

        for _ in range(n_trials_per_odor):
            noise = np.random.randn(len(clean_pattern)) * noise_std
            noisy_pattern = np.clip(clean_pattern + noise, 0, 1)
            all_patterns.append(noisy_pattern)
            all_labels.append(odor)

    all_patterns = np.array(all_patterns)  # (n_trials * n_odors, n_pn)

    # Perform PCA
    pca = PCA(n_components=3)
    pca_coords = pca.fit_transform(all_patterns)

    explained_var = pca.explained_variance_ratio_
    logger.info(f"  PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%}, PC3: {explained_var[2]:.1%}")
    logger.info(f"  Total variance explained (PC1+PC2+PC3): {explained_var[:3].sum():.1%}")

    return {
        'pca_coords': pca_coords,
        'labels': all_labels,
        'explained_variance': explained_var
    }


def analyze_kc_discriminability(patterns: dict, odors: list, cache_dir: Path, noise_std: float) -> dict:
    """Analyze discriminability after KC sparse coding."""
    from src.pgcn.models.ccbpn import ConnectomeConstrainedBehavioralPredictor
    import torch

    # Load model to get KC encoder
    logger.info("  Loading CCBPN model for KC encoding...")

    model = ConnectomeConstrainedBehavioralPredictor(
        cache_dir=cache_dir,
        kc_sparsity_target=0.10
    )
    model.eval()

    # Encode patterns to KC space
    kc_patterns = {}
    for odor in odors:
        pn_pattern = patterns[odor]

        # Add noise
        noise = np.random.randn(len(pn_pattern)) * noise_std
        pn_noisy = np.clip(pn_pattern + noise, 0, 1)

        # Encode to KC using forward pass
        # Use 100 timesteps to allow recurrent dynamics to settle (tau_kc=20ms)
        # Shape: (batch=1, time=100, n_pn=150)
        n_timesteps = 100
        pn_tensor = torch.tensor(pn_noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pn_tensor = pn_tensor.repeat(1, n_timesteps, 1)  # Repeat across time
        # Dummy dopamine signal (not used for encoding, just required for forward)
        dopamine_signal = torch.zeros(1, n_timesteps)

        with torch.no_grad():
            outputs = model.forward(pn_tensor, dopamine_signal, return_intermediates=True)
            # Extract KC activity at last timestep after dynamics settle: (batch, time, n_kc) -> (n_kc,)
            kc_activity = outputs['kc_activity'][0, -1, :].numpy()

        kc_patterns[odor] = kc_activity
        n_active_kc = np.sum(kc_activity > 0.01)
        logger.info(f"  {odor:25s}: {n_active_kc} active KCs")

    # Compute KC-level distances
    distance_matrix = compute_distance_matrix(kc_patterns, odors)

    return {
        'kc_patterns': kc_patterns,
        'distance_matrix': distance_matrix
    }


def compute_distance_matrix(patterns: dict, odors: list) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    n = len(odors)
    dist_matrix = np.zeros((n, n))

    for i, odor1 in enumerate(odors):
        for j, odor2 in enumerate(odors):
            dist = np.linalg.norm(patterns[odor1] - patterns[odor2])
            dist_matrix[i, j] = dist

    return dist_matrix


def plot_correlation_heatmap(corr_matrix: np.ndarray, odors: list, output_path: Path):
    """Plot PN pattern correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        xticklabels=odors,
        yticklabels=odors,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        center=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Pearson Correlation'}
    )
    plt.title('Pairwise Correlations Between PN Patterns\n(DoOR-Based, No Noise)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_snr_results(snr_results: dict, odors: list, output_path: Path):
    """Plot SNR analysis results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # SNR bar plot
    snrs = [snr_results[o]['mean_snr'] for o in odors]
    ax1.bar(range(len(odors)), snrs, color='steelblue')
    ax1.set_xticks(range(len(odors)))
    ax1.set_xticklabels(odors, rotation=45, ha='right')
    ax1.set_ylabel('Mean SNR (Signal / Noise)', fontsize=11)
    ax1.set_title('Signal-to-Noise Ratio with 8% Additive Noise', fontsize=12, fontweight='bold')
    ax1.axhline(y=10, color='red', linestyle='--', label='SNR=10 (good)')
    ax1.axhline(y=5, color='orange', linestyle='--', label='SNR=5 (marginal)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Correlation with clean pattern
    corrs = [snr_results[o]['correlation_with_clean'] for o in odors]
    ax2.bar(range(len(odors)), corrs, color='coral')
    ax2.set_xticks(range(len(odors)))
    ax2.set_xticklabels(odors, rotation=45, ha='right')
    ax2.set_ylabel('Correlation (Noisy vs Clean)', fontsize=11)
    ax2.set_title('Pattern Preservation After Noise', fontsize=12, fontweight='bold')
    ax2.axhline(y=0.95, color='green', linestyle='--', label='r=0.95 (excellent)')
    ax2.axhline(y=0.85, color='orange', linestyle='--', label='r=0.85 (marginal)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0.7, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_pca(pca_results: dict, odors: list, output_path: Path):
    """Plot PCA visualization of PN space."""
    pca_coords = pca_results['pca_coords']
    labels = pca_results['labels']
    explained_var = pca_results['explained_variance']

    fig = plt.figure(figsize=(12, 5))

    # PC1 vs PC2
    ax1 = fig.add_subplot(121)
    for odor in odors:
        mask = np.array(labels) == odor
        ax1.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            label=odor,
            alpha=0.6,
            s=30
        )
    ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=11)
    ax1.set_title('PN Patterns in PC Space (8% Noise)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(alpha=0.3)

    # PC2 vs PC3
    ax2 = fig.add_subplot(122)
    for odor in odors:
        mask = np.array(labels) == odor
        ax2.scatter(
            pca_coords[mask, 1],
            pca_coords[mask, 2],
            label=odor,
            alpha=0.6,
            s=30
        )
    ax2.set_xlabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=11)
    ax2.set_ylabel(f'PC3 ({explained_var[2]:.1%} variance)', fontsize=11)
    ax2.set_title('PN Patterns in PC Space (8% Noise)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"  Saved: {output_path}")


def plot_kc_distances(kc_results: dict, odors: list, output_path: Path):
    """Plot KC-level distance matrix."""
    dist_matrix = kc_results['distance_matrix']

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        dist_matrix,
        xticklabels=odors,
        yticklabels=odors,
        annot=True,
        fmt='.1f',
        cmap='viridis',
        cbar_kws={'label': 'Euclidean Distance'}
    )
    plt.title('Pairwise Distances in KC Space\n(After Sparse Coding, 8% Noise)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"  Saved: {output_path}")


def generate_report(
    output_path: Path,
    correlation_matrix: np.ndarray,
    snr_results: dict,
    pca_results: dict,
    kc_results: dict,
    odors: list
):
    """Generate comprehensive discriminability report."""

    # Compute summary statistics
    corr_upper_tri = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    mean_corr = corr_upper_tri.mean()
    max_corr = corr_upper_tri.max()

    mean_snr = np.mean([snr_results[o]['mean_snr'] for o in odors if o != 'air'])
    min_snr = np.min([snr_results[o]['mean_snr'] for o in odors if o != 'air'])

    total_var_explained = pca_results['explained_variance'][:3].sum()

    with open(output_path, 'w') as f:
        f.write("# PN Pattern Discriminability Analysis Report\n\n")
        f.write("## Executive Summary\n\n")

        # Determine if patterns are discriminative
        if max_corr > 0.85:
            verdict = "❌ **POOR DISCRIMINABILITY**"
            explanation = "Odor patterns are too similar (max correlation >0.85). Model cannot reliably distinguish odors."
        elif max_corr > 0.75:
            verdict = "⚠️  **MARGINAL DISCRIMINABILITY**"
            explanation = "Odor patterns are moderately similar (max correlation 0.75-0.85). May cause learning difficulties."
        else:
            verdict = "✅ **GOOD DISCRIMINABILITY**"
            explanation = "Odor patterns are sufficiently distinct (max correlation <0.75). Discriminability is adequate."

        f.write(f"**{verdict}**\n\n")
        f.write(f"{explanation}\n\n")

        f.write("### Key Metrics\n\n")
        f.write(f"- **Mean pairwise correlation**: {mean_corr:.3f}\n")
        f.write(f"- **Max pairwise correlation**: {max_corr:.3f}\n")
        f.write(f"- **Mean SNR (8% noise)**: {mean_snr:.2f}\n")
        f.write(f"- **Min SNR (8% noise)**: {min_snr:.2f}\n")
        f.write(f"- **PCA variance (PC1-3)**: {total_var_explained:.1%}\n\n")

        f.write("---\n\n")
        f.write("## Analysis 1: Pairwise Correlations (No Noise)\n\n")
        f.write("### Correlation Matrix\n\n")
        f.write("| Odor | " + " | ".join(odors) + " |\n")
        f.write("|" + "|".join(["---"] * (len(odors) + 1)) + "|\n")
        for i, odor1 in enumerate(odors):
            row = f"| {odor1} |"
            for j, odor2 in enumerate(odors):
                corr = correlation_matrix[i, j]
                if i == j:
                    row += " 1.00 |"
                elif corr > 0.85:
                    row += f" **{corr:.2f}** |"  # Highlight high correlations
                else:
                    row += f" {corr:.2f} |"
            f.write(row + "\n")

        f.write("\n### Interpretation\n\n")
        high_corr_pairs = []
        for i in range(len(odors)):
            for j in range(i+1, len(odors)):
                if correlation_matrix[i, j] > 0.75:
                    high_corr_pairs.append((odors[i], odors[j], correlation_matrix[i, j]))

        if high_corr_pairs:
            f.write("⚠️  **High-correlation pairs** (r > 0.75, difficult to distinguish):\n\n")
            for o1, o2, r in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
                f.write(f"- **{o1}** vs **{o2}**: r={r:.3f}\n")
        else:
            f.write("✅ No high-correlation pairs (r < 0.75). All odors are well-separated.\n")

        f.write("\n---\n\n")
        f.write("## Analysis 2: Signal-to-Noise Ratio (8% Noise)\n\n")
        f.write("| Odor | Mean SNR | Corr (Noisy vs Clean) | Active PNs |\n")
        f.write("|------|----------|----------------------|------------|\n")
        for odor in odors:
            snr = snr_results[odor]['mean_snr']
            corr = snr_results[odor]['correlation_with_clean']
            n_pn = snr_results[odor]['n_active_pns']

            if snr < 5:
                snr_str = f"**{snr:.2f}** ⚠️ "
            else:
                snr_str = f"{snr:.2f}"

            if corr < 0.90:
                corr_str = f"**{corr:.3f}** ⚠️ "
            else:
                corr_str = f"{corr:.3f}"

            f.write(f"| {odor} | {snr_str} | {corr_str} | {n_pn} |\n")

        f.write("\n### Interpretation\n\n")
        f.write("**SNR Guidelines**:\n")
        f.write("- SNR > 10: Excellent (pattern robust to noise)\n")
        f.write("- SNR 5-10: Good (pattern mostly preserved)\n")
        f.write("- SNR < 5: Poor (pattern degraded by noise)\n\n")

        low_snr_odors = [o for o in odors if snr_results[o]['mean_snr'] < 5]
        if low_snr_odors:
            f.write(f"⚠️  **Low SNR odors**: {', '.join(low_snr_odors)}\n")
            f.write("These odors are heavily degraded by 8% noise and may be difficult to learn.\n")
        else:
            f.write("✅ All odors have SNR ≥ 5. Noise impact is manageable.\n")

        f.write("\n---\n\n")
        f.write("## Analysis 3: PCA Visualization\n\n")
        f.write(f"**Variance explained by first 3 PCs**: {total_var_explained:.1%}\n\n")
        f.write("See `pn_pca.png` for visualization.\n\n")

        if total_var_explained < 0.70:
            f.write("⚠️  Low variance explained (<70%). PN patterns are high-dimensional and complex.\n")
        else:
            f.write("✅ High variance explained (≥70%). PN patterns can be well-represented in low dimensions.\n")

        f.write("\n---\n\n")
        f.write("## Analysis 4: KC-Level Discriminability\n\n")
        f.write("After sparse coding (10% sparsity), odor distances in KC space:\n\n")

        dist_matrix = kc_results['distance_matrix']
        f.write("| Odor | " + " | ".join(odors) + " |\n")
        f.write("|" + "|".join(["---"] * (len(odors) + 1)) + "|\n")
        for i, odor1 in enumerate(odors):
            row = f"| {odor1} |"
            for j, odor2 in enumerate(odors):
                dist = dist_matrix[i, j]
                if i == j:
                    row += " 0.0 |"
                else:
                    row += f" {dist:.1f} |"
            f.write(row + "\n")

        f.write("\n### Interpretation\n\n")
        min_nonzero_dist = np.min(dist_matrix[dist_matrix > 0])
        f.write(f"**Minimum inter-odor distance**: {min_nonzero_dist:.2f}\n\n")

        if min_nonzero_dist < 2.0:
            f.write("⚠️  Some odors are very close in KC space (<2.0). May be difficult to separate.\n")
        else:
            f.write("✅ All odors are well-separated in KC space (≥2.0).\n")

        f.write("\n---\n\n")
        f.write("## Conclusions\n\n")

        if max_corr > 0.85 or mean_snr < 5:
            f.write("### ❌ DoOR Patterns Have Poor Discriminability\n\n")
            f.write("**Primary Issues**:\n")
            if max_corr > 0.85:
                f.write(f"- High inter-odor correlations (max={max_corr:.3f} > 0.85)\n")
            if mean_snr < 5:
                f.write(f"- Low signal-to-noise ratio (mean={mean_snr:.2f} < 5)\n")
            f.write("\n**Root Cause**: DoOR receptor responses, when mapped to FlyWire PNs, ")
            f.write("do not provide sufficient discriminability for the model to learn odor-specific associations.\n\n")

            f.write("### Recommended Next Steps\n\n")
            f.write("1. **Try alternative odor encodings**:\n")
            f.write("   - Learned embeddings (trainable PN→KC weights)\n")
            f.write("   - Random projections (initialize PN→KC randomly)\n")
            f.write("   - Amplified DoOR responses (scale up differences)\n\n")
            f.write("2. **Reduce noise** to 4% (at cost of biological realism)\n\n")
            f.write("3. **Simplify task** to fewer odors (e.g., 4 instead of 8)\n\n")

        elif max_corr > 0.75 or mean_snr < 8:
            f.write("### ⚠️  DoOR Patterns Have Marginal Discriminability\n\n")
            f.write("Patterns are moderately discriminative but may struggle with noise.\n\n")
            f.write("### Recommended Next Steps\n\n")
            f.write("1. Increase KC sparsity to 15-20% (if not already done in Prompt 1)\n")
            f.write("2. Reduce noise to 6% (slight reduction while maintaining realism)\n")
            f.write("3. Add learned gain parameters to amplify DoOR responses\n\n")

        else:
            f.write("### ✅ DoOR Patterns Have Good Discriminability\n\n")
            f.write("PN patterns are sufficiently distinct for learning. ")
            f.write("The bottleneck is likely elsewhere (capacity, task complexity, optimization).\n\n")
            f.write("### Recommended Next Steps\n\n")
            f.write("1. Proceed to **Prompt 3** (single-dataset training) to test task complexity hypothesis\n")
            f.write("2. Consider architectural enhancements (recurrent connections, context embeddings)\n")
            f.write("3. Verify dopamine assignment and training protocol are correct\n\n")

    logger.info(f"  Saved report: {output_path}")


if __name__ == '__main__':
    main()
