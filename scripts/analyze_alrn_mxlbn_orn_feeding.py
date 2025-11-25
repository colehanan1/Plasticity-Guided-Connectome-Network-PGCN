#!/usr/bin/env python
"""Map ALRN+MxLbN neurons to ORN/IR identities and test a feeding-odor hypothesis.

This script:
1) Loads the 159 ALRN+MxLbN neurons (dual olfactory–gustatory pathway)
2) Maps them to ORN/IR labels from FlyWire processed labels
3) Counts neurons per ORN/IR type
4) Integrates DoOR responses to quantify tuning breadth and feeding-odor responses
5) Tests whether ORNs with more ALRN+MxLbN neurons are broadly tuned and food-biased

Outputs
-------
- data/analysis/orn_alrn_mxlbn_counts.csv
- data/analysis/alrn_mxlbn_orn_analysis.csv
- results/alrn_mxlbn_hypothesis_test.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


DEFAULT_ROOT_ID_PATH = Path("data/flywire/root_ids_ALRN_nerve_MxLbN.txt")
DEFAULT_LABELS_PATH = Path("data/flywire/processed_labels.csv.gz")
DEFAULT_OUTPUT_DIR = Path("data/analysis")
DEFAULT_RESULTS_DIR = Path("results")

# Candidate DoOR response matrices (odorants × receptors)
DOOR_MATRIX_CANDIDATES = [
    Path("data/cache/door_response_matrix.csv"),
    Path("data/door_cache/door_response_matrix.csv"),
    Path("door_cache/door_response_matrix.csv"),
]

FEEDING_ODORS = [
    "acetic acid",
    "ethanol",
    "acetoin",
    "ethyl acetate",
    "methyl acetate",
    "2-methylpyrazine",
    "benzaldehyde",
    "1-hexanol",
    "phenylacetaldehyde",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map ALRN+MxLbN neurons to ORNs and test feeding-odor hypothesis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root-ids",
        type=Path,
        default=DEFAULT_ROOT_ID_PATH,
        help="Path to comma-separated ALRN+MxLbN root IDs.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="FlyWire processed labels CSV (root_id, processed_labels).",
    )
    parser.add_argument(
        "--door-path",
        type=Path,
        default=None,
        help="Optional direct path to DoOR response matrix (odorants × receptors).",
    )
    parser.add_argument(
        "--response-threshold",
        type=float,
        default=0.3,
        help="Response threshold (>0-1) to count an odor as eliciting a response.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for plot outputs.",
    )
    return parser.parse_args()


def load_root_ids(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(f"Root ID file not found: {path}")
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"No root IDs found in {path}")
    root_ids: List[int] = [int(rid.strip()) for rid in text.split(",") if rid.strip()]
    return root_ids


def load_processed_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed labels file not found: {path}")
    labels = pd.read_csv(path)
    expected_cols = {"root_id", "processed_labels"}
    if not expected_cols.issubset(labels.columns):
        raise ValueError(f"Expected columns {expected_cols}, found {labels.columns.tolist()}")
    return labels[["root_id", "processed_labels"]]


def extract_orn_type(label_text: str) -> Optional[str]:
    """Extract ORN/IR receptor name from a processed label string."""
    if pd.isna(label_text):
        return None
    # Normalize brackets/quotes; regex works on the raw string representation
    orn_match = re.search(r"\b(Or\d+[a-zA-Z]*)\b", label_text, flags=re.IGNORECASE)
    if orn_match:
        return orn_match.group(1).capitalize()
    ir_match = re.search(r"\b(Ir\d+[a-zA-Z]*)\b", label_text, flags=re.IGNORECASE)
    if ir_match:
        return ir_match.group(1).capitalize()
    return None


def count_neurons_per_orn(alrn_neurons: pd.DataFrame) -> pd.DataFrame:
    return (
        alrn_neurons.dropna(subset=["orn_type"])["orn_type"]
        .value_counts()
        .rename_axis("orn_type")
        .reset_index(name="alrn_mxlbn_count")
    )


def load_door_matrix(user_path: Optional[Path]) -> Optional[pd.DataFrame]:
    candidates: Iterable[Path] = []
    if user_path is not None:
        candidates = [user_path, *DOOR_MATRIX_CANDIDATES]
    else:
        candidates = DOOR_MATRIX_CANDIDATES

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            door = pd.read_csv(path, index_col=0)
            print(f"Loaded DoOR data from {path} ({door.shape[0]} odors × {door.shape[1]} receptors)")
            return door
    print("WARNING: DoOR response matrix not found. Odor analyses will be skipped.")
    return None


def compute_tuning_stats(
    orn_counts: pd.DataFrame, door: pd.DataFrame, response_threshold: float
) -> pd.DataFrame:
    records = []
    col_lookup = {col.lower(): col for col in door.columns}
    for orn_type in orn_counts["orn_type"]:
        col_name = col_lookup.get(orn_type.lower())
        if col_name is None:
            records.append(
                {
                    "orn_type": orn_type,
                    "n_odors_responsive": np.nan,
                    "pct_odors_responsive": np.nan,
                    "mean_response_strength": np.nan,
                }
            )
            continue
        responses = door[col_name].dropna()
        total_odors = len(responses)
        if total_odors == 0:
            records.append(
                {
                    "orn_type": orn_type,
                    "n_odors_responsive": 0,
                    "pct_odors_responsive": np.nan,
                    "mean_response_strength": np.nan,
                }
            )
            continue

        responsive = responses > response_threshold
        n_responsive = int(responsive.sum())
        pct_responsive = 100 * n_responsive / total_odors
        mean_response = responses[responsive].mean() if n_responsive > 0 else np.nan

        records.append(
            {
                "orn_type": orn_type,
                "n_odors_responsive": n_responsive,
                "pct_odors_responsive": pct_responsive,
                "mean_response_strength": mean_response,
            }
        )
    return pd.DataFrame.from_records(records)


def compute_feeding_stats(
    orn_counts: pd.DataFrame,
    door: pd.DataFrame,
    feeding_odors: List[str],
    response_threshold: float,
) -> Tuple[pd.DataFrame, List[str]]:
    records = []
    col_lookup = {col.lower(): col for col in door.columns}
    index_lookup = {idx.lower(): idx for idx in door.index}
    available_feeding = [index_lookup[o.lower()] for o in feeding_odors if o.lower() in index_lookup]

    if not available_feeding:
        print("WARNING: No feeding odors present in DoOR dataset; skipping feeding analysis.")
        return pd.DataFrame(
            {
                "orn_type": orn_counts["orn_type"],
                "mean_feeding_response": np.nan,
                "n_feeding_odors": np.nan,
            }
        ), []

    for orn_type in orn_counts["orn_type"]:
        col_name = col_lookup.get(orn_type.lower())
        if col_name is None:
            records.append(
                {
                    "orn_type": orn_type,
                    "mean_feeding_response": np.nan,
                    "n_feeding_odors": np.nan,
                }
            )
            continue

        feeding_responses = door.loc[available_feeding, col_name].dropna()
        if feeding_responses.empty:
            records.append(
                {
                    "orn_type": orn_type,
                    "mean_feeding_response": np.nan,
                    "n_feeding_odors": 0,
                }
            )
            continue

        mean_feeding = feeding_responses[feeding_responses > 0.1].mean()
        n_feeding_responsive = int((feeding_responses > response_threshold).sum())

        records.append(
            {
                "orn_type": orn_type,
                "mean_feeding_response": mean_feeding,
                "n_feeding_odors": n_feeding_responsive,
            }
        )
    return pd.DataFrame.from_records(records), available_feeding


def compute_correlation(
    df: pd.DataFrame, x_col: str, y_col: str
) -> Optional[Tuple[float, float]]:
    valid = df.dropna(subset=[x_col, y_col])
    if len(valid) < 3:
        return None
    r, p = pearsonr(valid[x_col], valid[y_col])
    return r, p


def plot_hypothesis_scatter(
    analysis: pd.DataFrame,
    results_dir: Path,
    tuning_corr: Optional[Tuple[float, float]],
    feeding_corr: Optional[Tuple[float, float]],
) -> Path:
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(
        analysis["alrn_mxlbn_count"],
        analysis["pct_odors_responsive"],
        s=90,
        alpha=0.7,
        color="#1f77b4",
    )
    ax1.set_xlabel("ALRN+MxLbN neuron count")
    ax1.set_ylabel("Tuning breadth (% odors responsive)")
    ax1.set_title("More ALRN+MxLbN → Broader tuning?")
    if tuning_corr:
        r, p = tuning_corr
        ax1.text(0.05, 0.95, f"r = {r:.3f}, p = {p:.3f}", transform=ax1.transAxes, va="top")

    ax2 = axes[1]
    ax2.scatter(
        analysis["alrn_mxlbn_count"],
        analysis["mean_feeding_response"],
        s=90,
        alpha=0.7,
        color="#ff7f0e",
    )
    ax2.set_xlabel("ALRN+MxLbN neuron count")
    ax2.set_ylabel("Mean response to feeding odors")
    ax2.set_title("More ALRN+MxLbN → Stronger feeding responses?")
    if feeding_corr:
        r, p = feeding_corr
        ax2.text(0.05, 0.95, f"r = {r:.3f}, p = {p:.3f}", transform=ax2.transAxes, va="top")

    plt.tight_layout()
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "alrn_mxlbn_hypothesis_test.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


def print_summary(
    analysis: pd.DataFrame,
    tuning_corr: Optional[Tuple[float, float]],
    feeding_corr: Optional[Tuple[float, float]],
) -> None:
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST RESULTS")
    print("=" * 60)

    print("\nTop 5 ORNs with most ALRN+MxLbN neurons:")
    top5 = analysis.nlargest(5, "alrn_mxlbn_count")
    print(top5[["orn_type", "alrn_mxlbn_count", "pct_odors_responsive", "mean_feeding_response"]])

    print("\nBottom 5 ORNs with fewest ALRN+MxLbN neurons:")
    bottom5 = analysis.nsmallest(5, "alrn_mxlbn_count")
    print(bottom5[["orn_type", "alrn_mxlbn_count", "pct_odors_responsive", "mean_feeding_response"]])

    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    if tuning_corr and tuning_corr[0] > 0.5 and tuning_corr[1] < 0.05:
        print("✓ Supported: ORNs with more ALRN+MxLbN neurons are more broadly tuned.")
    elif tuning_corr and tuning_corr[0] < 0:
        print("✗ Rejected: Negative correlation between ALRN+MxLbN count and tuning breadth.")
    else:
        print("? Inconclusive: Correlation between ALRN+MxLbN count and tuning breadth is weak or non-significant.")

    if feeding_corr and feeding_corr[0] > 0.5 and feeding_corr[1] < 0.05:
        print("✓ Feeding: More ALRN+MxLbN neurons → stronger feeding-odor responses.")
    elif feeding_corr and feeding_corr[0] < 0:
        print("✗ Feeding: Negative correlation between ALRN+MxLbN count and feeding-odor responses.")
    else:
        print("? Feeding: Correlation between ALRN+MxLbN count and feeding responses is weak or non-significant.")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    root_ids = load_root_ids(args.root_ids)
    print(f"Loaded {len(root_ids)} ALRN+MxLbN neurons to analyze")

    labels = load_processed_labels(args.labels)
    alrn_neurons = labels[labels["root_id"].isin(root_ids)].copy()
    missing = set(root_ids) - set(alrn_neurons["root_id"])
    if missing:
        print(f"WARNING: {len(missing)} root IDs missing from labels (examples: {list(missing)[:3]})")

    alrn_neurons["orn_type"] = alrn_neurons["processed_labels"].apply(extract_orn_type)
    mapped = alrn_neurons["orn_type"].notna().sum()
    print(f"Mapped {mapped} / {len(alrn_neurons)} neurons to ORN/IR identities")
    print("\nORN types found:")
    print(alrn_neurons["orn_type"].value_counts(dropna=True))

    orn_counts = count_neurons_per_orn(alrn_neurons)
    print("\nORNs ranked by ALRN+MxLbN neuron count:")
    print(orn_counts.head(20))

    orn_counts_path = args.output_dir / "orn_alrn_mxlbn_counts.csv"
    orn_counts.to_csv(orn_counts_path, index=False)

    door = load_door_matrix(args.door_path)
    analysis = orn_counts.copy()
    tuning_corr: Optional[Tuple[float, float]] = None
    feeding_corr: Optional[Tuple[float, float]] = None

    if door is not None:
        tuning_df = compute_tuning_stats(orn_counts, door, args.response_threshold)
        analysis = analysis.merge(tuning_df, on="orn_type", how="left")

        feeding_df, available_feeding = compute_feeding_stats(
            orn_counts, door, FEEDING_ODORS, args.response_threshold
        )
        analysis = analysis.merge(feeding_df, on="orn_type", how="left")
        print(f"\nFeeding odors available in DoOR: {available_feeding}")

        tuning_corr = compute_correlation(analysis, "alrn_mxlbn_count", "pct_odors_responsive")
        feeding_corr = compute_correlation(analysis, "alrn_mxlbn_count", "mean_feeding_response")

        plot_path = plot_hypothesis_scatter(analysis, args.results_dir, tuning_corr, feeding_corr)
        print(f"\nSaved plot: {plot_path}")
    else:
        for col in [
            "n_odors_responsive",
            "pct_odors_responsive",
            "mean_response_strength",
            "mean_feeding_response",
            "n_feeding_odors",
        ]:
            analysis[col] = np.nan
        print("\nSkipping tuning/feeding analyses (DoOR matrix unavailable).")

    analysis_path = args.output_dir / "alrn_mxlbn_orn_analysis.csv"
    analysis.to_csv(analysis_path, index=False)

    print_summary(analysis, tuning_corr, feeding_corr)


if __name__ == "__main__":
    main()
