"""Group-aware cross-validation for the chemically informed model."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ImportError("PyTorch is required to run the cross-validation pipeline.") from exc

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ImportError("scikit-learn must be installed to compute AUROC scores.") from exc

from pgcn.chemical.features import get_chemical_features
from pgcn.chemical.mappings import COMPLETE_ODOR_MAPPINGS
from pgcn.data.behavioral_data import load_behavioral_dataframe, make_group_kfold
from pgcn.models import ChemicalSTDP, ChemicallyInformedDrosophilaModel

try:
    from statistical_tests import run_all_statistical_tests
    STATISTICAL_TESTS_AVAILABLE = True
except ImportError as e:
    STATISTICAL_TESTS_AVAILABLE = False

TRAINED_TRIAL_LABELS = {"testing_2", "testing_4", "testing_5"}
CONTROL_DATASET = "hex_control"
CHANCE_LEVEL = 0.5
PERFORMANCE_TARGETS = {
    "overall_accuracy": 0.70,
    "trained_odor_accuracy": 0.80,
    "control_separation": 0.90,
}


@dataclass(frozen=True)
class FoldMetrics:
    """Primary performance indicators recorded per fold."""

    fold_index: int
    overall_accuracy: float
    trained_odor_accuracy: float | None
    control_separation: float | None
    auroc: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-aware behavioural cross-validation")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional path to behavioural CSV (defaults to packaged dataset)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of GroupKFold splits (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/cross_validation"),
        help="Directory for per-fold outputs and aggregate reports",
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="week4",
        help="Prefix for aggregate report filenames",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Base learning rate for ChemicalSTDP updates",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Decision boundary for accuracy calculations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (default: auto-detect)",
    )
    parser.add_argument(
        "--statistical-tests",
        action="store_true",
        default=True,
        help="Run statistical tests after cross-validation (default: True)",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        default=False,
        help="Skip statistical tests (for quick runs)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
        help="Number of permutation resamples for statistical tests (default: 5000)",
    )
    parser.add_argument(
        "--n-bootstrap-samples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for confidence intervals (default: 5000)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip data validation (use when data has duplicates or other validation issues)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific dataset(s) (e.g., opto_EB opto_hex). If not specified, uses all datasets.",
    )
    parser.add_argument(
        "--per-dataset",
        action="store_true",
        default=False,
        help="Run separate cross-validation and statistical tests for each dataset independently",
    )
    return parser.parse_args()


def determine_device(override: str | None) -> torch.device:
    if override is not None:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_training_odor(dataset: str) -> str:
    mapping = COMPLETE_ODOR_MAPPINGS.get(dataset)
    if not mapping:
        raise KeyError(f"Dataset '{dataset}' missing from COMPLETE_ODOR_MAPPINGS.")
    candidates = {odor for label, odor in mapping.items() if label in TRAINED_TRIAL_LABELS}
    if not candidates:
        raise ValueError(f"Unable to infer training odor for dataset '{dataset}'.")
    if len(candidates) > 1:
        raise ValueError(
            "Training odor inference yielded multiple candidates for "
            f"dataset '{dataset}': {sorted(candidates)}"
        )
    return next(iter(candidates))


def resolve_test_odor(dataset: str, trial_label: str) -> str:
    mapping = COMPLETE_ODOR_MAPPINGS.get(dataset)
    if mapping is None or trial_label not in mapping:
        raise KeyError(
            f"Missing odor mapping for dataset '{dataset}' and trial '{trial_label}'."
        )
    return mapping[trial_label]


def freeze_model_parameters(model: ChemicallyInformedDrosophilaModel) -> None:
    for param in model.parameters():
        param.requires_grad = False


def compute_model_response(
    model: ChemicallyInformedDrosophilaModel,
    training_odor: str,
    test_odor: str,
    *,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """Run the model and return probability plus KC activity."""

    with torch.no_grad():
        dtype = next(model.parameters()).dtype
        train_features = get_chemical_features(training_odor, as_tensor=True).to(device=device, dtype=dtype)
        test_features = get_chemical_features(test_odor, as_tensor=True).to(device=device, dtype=dtype)
        train_features = train_features.unsqueeze(0)
        test_features = test_features.unsqueeze(0)

        train_repr = model.chemical_encoder(train_features)
        test_repr = model.chemical_encoder(test_features)
        condition_index = torch.tensor(
            [model.condition_lookup.get(training_odor.lower(), 0)],
            device=device,
            dtype=torch.long,
        )
        condition_embedding = model.training_encoder(condition_index)
        condition_projection = model.condition_projector(condition_embedding)

        interaction = model.interaction_layer(train_repr, test_repr) + condition_projection
        pn_drive = model.interaction_projector(interaction)
        reservoir = model.reservoir
        pn_activity = pn_drive
        if pn_activity.dim() == 1:
            pn_activity = pn_activity.unsqueeze(0)
        kc_activity = torch.relu(reservoir.pn_to_kc(pn_activity))
        kc_activity = reservoir._enforce_sparsity(kc_activity)
        mbon_drive = reservoir.kc_to_mbon(kc_activity)
        mbon_activity = torch.relu(mbon_drive)
        logits = model.classifier(mbon_activity)
        probability = torch.sigmoid(logits.squeeze(-1))
        return float(probability.item()), kc_activity.squeeze(0)


def train_fold(
    model: ChemicallyInformedDrosophilaModel,
    stdp: ChemicalSTDP,
    train_frame: pd.DataFrame,
    *,
    device: torch.device,
) -> None:
    model.eval()
    stdp.eval()
    for row in train_frame.itertuples(index=False):
        dataset = str(row.dataset)
        training_odor = resolve_training_odor(dataset)
        test_odor = resolve_test_odor(dataset, str(row.trial_label))
        reward = float(row.prediction)
        _, kc_activity = compute_model_response(model, training_odor, test_odor, device=device)
        delta = stdp.update_plasticity(training_odor, test_odor, reward=reward, kc_activity=kc_activity)
        weight = model.reservoir.kc_to_mbon.weight
        with torch.no_grad():
            weight.add_(delta.to(weight.device).T)


def evaluate_fold(
    model: ChemicallyInformedDrosophilaModel,
    test_frame: pd.DataFrame,
    *,
    device: torch.device,
) -> pd.DataFrame:
    records: List[MutableMapping[str, object]] = []
    for row in test_frame.itertuples(index=False):
        dataset = str(row.dataset)
        training_odor = resolve_training_odor(dataset)
        test_odor = resolve_test_odor(dataset, str(row.trial_label))
        probability, _ = compute_model_response(model, training_odor, test_odor, device=device)
        records.append(
            {
                "dataset": dataset,
                "fly": str(row.fly),
                "trial_label": str(row.trial_label),
                "training_odor": training_odor,
                "test_odor": test_odor,
                "probability": probability,
                "label": float(row.prediction),
            }
        )
    return pd.DataFrame.from_records(records)


def summarise_metrics(
    fold_df: pd.DataFrame,
    *,
    threshold: float,
    fold_index: int,
) -> FoldMetrics:
    if fold_df.empty:
        raise ValueError("Fold evaluation dataframe is empty.")
    labels = fold_df["label"].astype(int)
    predictions = (fold_df["probability"] >= threshold).astype(int)
    overall_accuracy = float((predictions == labels).mean())

    trained_mask = fold_df["test_odor"] == fold_df["training_odor"]
    if trained_mask.any():
        trained_accuracy = float((predictions[trained_mask] == labels[trained_mask]).mean())
    else:
        trained_accuracy = None

    unique_labels = labels.nunique()
    if unique_labels > 1:
        auroc = float(roc_auc_score(labels, fold_df["probability"]))
    else:
        auroc = None

    control_mask = fold_df["dataset"].str.lower() == CONTROL_DATASET
    if control_mask.any() and (~control_mask).any():
        control_mean = float(fold_df.loc[control_mask, "probability"].mean())
        trained_mean = float(fold_df.loc[~control_mask, "probability"].mean())
        eps = 1e-6
        control_separation = float(np.clip(1.0 - control_mean / max(trained_mean, eps), 0.0, 1.0))
    else:
        control_separation = None

    return FoldMetrics(
        fold_index=fold_index,
        overall_accuracy=overall_accuracy,
        trained_odor_accuracy=trained_accuracy,
        control_separation=control_separation,
        auroc=auroc,
    )


def summarise_generalisation(
    fold_df: pd.DataFrame,
    *,
    threshold: float,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for (dataset, test_odor), group in fold_df.groupby(["dataset", "test_odor"]):
        accuracy = float(
            ((group["probability"] >= threshold).astype(int) == group["label"].astype(int)).mean()
        )
        records.append(
            {
                "dataset": dataset,
                "test_odor": test_odor,
                "mean_probability": float(group["probability"].mean()),
                "response_rate": float(group["label"].mean()),
                "accuracy": accuracy,
                "count": int(len(group)),
            }
        )
    return records


def _nan_safe_mean(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return float(np.mean(numeric))


def _nan_safe_std(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if len(numeric) <= 1:
        return None
    return float(np.std(numeric, ddof=1))


def metrics_to_dict(metrics: FoldMetrics) -> Dict[str, object]:
    return {
        "fold": metrics.fold_index,
        "overall_accuracy": metrics.overall_accuracy,
        "trained_odor_accuracy": metrics.trained_odor_accuracy,
        "control_separation": metrics.control_separation,
        "auroc": metrics.auroc,
    }


def write_fold_summary(
    output_dir: Path,
    *,
    fold_index: int,
    metrics: FoldMetrics,
    generalisation: List[Dict[str, object]],
    train_flies: List[str],
    test_flies: List[str],
) -> None:
    def _serialisable(value: float | None) -> float | None:
        if value is None:
            return None
        return float(value)

    summary = {
        "fold": fold_index,
        "train_flies": train_flies,
        "test_flies": test_flies,
        "metrics": {
            "overall_accuracy": metrics.overall_accuracy,
            "trained_odor_accuracy": _serialisable(metrics.trained_odor_accuracy),
            "control_separation": _serialisable(metrics.control_separation),
            "auroc": _serialisable(metrics.auroc),
        },
        "generalisation": generalisation,
    }
    output_path = output_dir / f"fold_{fold_index:02d}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    generalisation_path = output_dir / f"fold_{fold_index:02d}_generalisation.csv"
    with generalisation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "test_odor",
                "mean_probability",
                "response_rate",
                "accuracy",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(generalisation)


def write_aggregate_reports(
    output_dir: Path,
    *,
    prefix: str,
    fold_metrics: List[Dict[str, object]],
) -> None:
    metrics_df = pd.DataFrame(fold_metrics)
    overall_values = metrics_df["overall_accuracy"].tolist()
    trained_values = metrics_df["trained_odor_accuracy"].tolist()
    control_values = metrics_df["control_separation"].tolist()
    auroc_values = metrics_df["auroc"].tolist()

    aggregate = {
        "overall_accuracy": {
            "mean": float(np.mean(overall_values)),
            "std": _nan_safe_std(overall_values),
        },
        "trained_odor_accuracy": {
            "mean": _nan_safe_mean(trained_values),
            "std": _nan_safe_std(trained_values),
        },
        "control_separation": {
            "mean": _nan_safe_mean(control_values),
            "std": _nan_safe_std(control_values),
        },
        "auroc": {
            "mean": _nan_safe_mean(auroc_values),
            "std": _nan_safe_std(auroc_values),
        },
    }
    json_report = {
        "folds": fold_metrics,
        "aggregate": aggregate,
    }
    json_path = output_dir / f"{prefix}_report.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_report, handle, indent=2)

    csv_path = output_dir / f"{prefix}_report.csv"
    csv_df = metrics_df.copy()
    if "fold" not in csv_df.columns:
        csv_df.insert(0, "fold", range(1, len(csv_df) + 1))
    else:
        # Ensure fold ordering is stable before appending aggregate statistics.
        csv_df = csv_df.sort_values("fold").reset_index(drop=True)
    aggregate_mean = {
        "fold": "mean",
        "overall_accuracy": float(np.mean(overall_values)),
        "trained_odor_accuracy": _nan_safe_mean(trained_values),
        "control_separation": _nan_safe_mean(control_values),
        "auroc": _nan_safe_mean(auroc_values),
    }
    aggregate_std = {
        "fold": "std",
        "overall_accuracy": _nan_safe_std(overall_values),
        "trained_odor_accuracy": _nan_safe_std(trained_values),
        "control_separation": _nan_safe_std(control_values),
        "auroc": _nan_safe_std(auroc_values),
    }
    csv_df = pd.concat([csv_df, pd.DataFrame([aggregate_mean, aggregate_std])], ignore_index=True)
    csv_df.to_csv(csv_path, index=False)
    _emit_console_summary(metrics_df, aggregate)


def _emit_console_summary(metrics_df: pd.DataFrame, aggregate: Dict[str, Dict[str, object]]) -> None:
    fold_count = len(metrics_df)
    print("\n=== Cross-validation aggregate summary ===")
    print(f"Folds evaluated: {fold_count}")

    def _format_stat(value: float | None) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value:.3f}"

    for metric in ["overall_accuracy", "trained_odor_accuracy", "control_separation", "auroc"]:
        mean_value = aggregate[metric]["mean"]
        std_value = aggregate[metric]["std"]
        defined_folds = int(metrics_df[metric].notna().sum())
        coverage_note = f"{defined_folds}/{fold_count} folds"
        stat_repr = f"mean={_format_stat(mean_value)}"
        if std_value is not None:
            stat_repr += f" ±{_format_stat(std_value)}"
        else:
            stat_repr += " ±N/A"

        messages = [f"{metric}: {stat_repr} ({coverage_note})"]

        target = PERFORMANCE_TARGETS.get(metric)
        if isinstance(mean_value, (float, int)) and not np.isnan(mean_value):
            if metric == "overall_accuracy":
                delta = float(mean_value) - CHANCE_LEVEL
                messages.append(f"  ↳ vs. chance ({CHANCE_LEVEL:.3f}): {delta:+.3f}")
            if target is not None:
                if float(mean_value) >= target:
                    messages.append(f"  ↳ meets target ≥{target:.3f}")
                else:
                    messages.append(f"  ↳ below target ≥{target:.3f}")
        else:
            if defined_folds == 0:
                messages.append("  ↳ insufficient data to compute this metric")

        for line in messages:
            print(line)


def print_statistical_summary(report: Dict[str, object]) -> None:
    """Print concise summary of statistical test results."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    vs_chance = report.get("permutation_tests", {}).get("vs_chance", {})
    for metric_name, results in vs_chance.items():
        p_val = results.get("p_value_one_tailed", 1.0)
        sig = "✓✓" if p_val < 0.01 else ("✓" if p_val < 0.05 else "✗")

        print(f"\n{metric_name}:")
        print(f"  Observed: {results['observed_mean']:.4f} vs chance {results['chance_level']:.2f}")
        print(f"  p-value: {p_val:.4f} {sig}")

        if metric_name in report.get("effect_sizes", {}).get("vs_chance", {}):
            effect = report["effect_sizes"]["vs_chance"][metric_name]
            print(f"  Cohen's d: {effect['cohens_d']:.3f} ({effect['interpretation']})")

    # Confidence intervals
    print(f"\nConfidence Intervals (95%):")
    for metric_name, ci_data in report.get("confidence_intervals", {}).items():
        ci_95 = ci_data.get("ci_95", [0, 0])
        print(f"  {metric_name}: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

    print("\n" + "=" * 80)
    print("Legend: ✓ = p < 0.05, ✓✓ = p < 0.01, ✗ = not significant")
    print("=" * 80 + "\n")


def run_cross_validation_for_dataset(
    args: argparse.Namespace,
    data_frame: pd.DataFrame,
    dataset_name: str,
    device: torch.device,
    output_dir: Path
) -> List[Dict[str, object]]:
    """Run cross-validation for a single dataset."""
    from sklearn.model_selection import GroupKFold

    print(f"\n{'=' * 80}")
    print(f"Running cross-validation for dataset: {dataset_name}")
    print(f"{'=' * 80}\n")

    # Create unique fly identifier for grouping
    # If fly_number exists, combine fly and fly_number to create unique identifier
    if "fly_number" in data_frame.columns:
        groups = (data_frame["fly"] + "_" + data_frame["fly_number"].astype(str)).to_numpy()
    else:
        groups = data_frame["fly"].to_numpy()

    # Determine the actual number of folds based on unique groups
    n_unique_groups = len(np.unique(groups))
    actual_folds = min(args.folds, n_unique_groups)

    if actual_folds < args.folds:
        print(f"Warning: Dataset has only {n_unique_groups} unique flies/groups.")
        print(f"Reducing folds from {args.folds} to {actual_folds} for this dataset.\n")

    if actual_folds < 2:
        print(f"Error: Cannot perform cross-validation with only {n_unique_groups} unique group(s).")
        print(f"Skipping dataset: {dataset_name}\n")
        return []

    # Create GroupKFold splitter directly on the filtered data
    feature_index = np.arange(len(data_frame))
    splitter = GroupKFold(n_splits=actual_folds)
    fold_iterator = splitter.split(feature_index, groups=groups)

    fold_metrics_records: List[Dict[str, object]] = []

    for fold_index, (train_idx, test_idx) in enumerate(fold_iterator, start=1):
        model = ChemicallyInformedDrosophilaModel()
        model.to(device)
        freeze_model_parameters(model)
        stdp = ChemicalSTDP(
            model.reservoir.n_kc,
            model.reservoir.n_mbon,
            base_lr=args.learning_rate,
        )
        stdp.to(device)

        train_frame = data_frame.iloc[train_idx].reset_index(drop=True)
        test_frame = data_frame.iloc[test_idx].reset_index(drop=True)
        train_fold(model, stdp, train_frame, device=device)

        fold_df = evaluate_fold(model, test_frame, device=device)
        metrics = summarise_metrics(
            fold_df,
            threshold=args.decision_threshold,
            fold_index=fold_index,
        )
        generalisation = summarise_generalisation(fold_df, threshold=args.decision_threshold)

        train_flies = sorted(train_frame["fly"].unique())
        test_flies = sorted(test_frame["fly"].unique())
        write_fold_summary(
            output_dir,
            fold_index=fold_index,
            metrics=metrics,
            generalisation=generalisation,
            train_flies=train_flies,
            test_flies=test_flies,
        )
        fold_metrics_records.append(metrics_to_dict(metrics))

    write_aggregate_reports(output_dir, prefix=args.report_prefix, fold_metrics=fold_metrics_records)

    # Run statistical tests if enabled
    run_stats = args.statistical_tests and not args.skip_stats and STATISTICAL_TESTS_AVAILABLE
    if run_stats:
        print("\nRunning statistical tests...")

        # Load fold results from saved JSON files
        fold_results = []
        for fold_index in range(1, args.folds + 1):
            fold_path = output_dir / f"fold_{fold_index:02d}.json"
            if fold_path.exists():
                with fold_path.open("r", encoding="utf-8") as f:
                    fold_results.append(json.load(f))

        if fold_results:
            try:
                # Run statistical tests
                statistical_report = run_all_statistical_tests(
                    fold_results=fold_results,
                    chance_level=CHANCE_LEVEL,
                    n_permutations=args.n_permutations,
                    n_bootstrap_samples=args.n_bootstrap_samples,
                    chemical_similarity_data=None,  # TODO: Add chemical similarity data extraction
                    random_seed=None
                )

                # Save statistical report (convert numpy types to native Python types)
                def convert_numpy_types(obj):
                    """Convert numpy types to native Python types for JSON serialization."""
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj

                stat_report_path = output_dir / f"{args.report_prefix}_statistical_report.json"
                with stat_report_path.open("w", encoding="utf-8") as f:
                    json.dump(convert_numpy_types(statistical_report), f, indent=2)

                print(f"Statistical report saved to: {stat_report_path}")

                # Print summary
                print_statistical_summary(statistical_report)

            except Exception as e:
                print(f"Warning: Statistical tests failed: {e}")
                print("Cross-validation results are still valid.")
        else:
            print("Warning: No fold results found for statistical analysis.")
    elif not STATISTICAL_TESTS_AVAILABLE:
        print("\nNote: Statistical tests skipped (statistical_tests module not available)")
    elif args.skip_stats:
        print("\nNote: Statistical tests skipped (--skip-stats flag)")

    return fold_metrics_records


def main() -> None:
    args = parse_args()
    device = determine_device(args.device)

    # Skip validation if --skip-validation flag is set, otherwise validate default data
    should_validate = (args.data is None) and (not args.skip_validation)
    data_frame = load_behavioral_dataframe(args.data, validate=should_validate)

    # Filter by dataset if specified
    if args.dataset is not None:
        available_datasets = data_frame["dataset"].unique()
        invalid_datasets = set(args.dataset) - set(available_datasets)
        if invalid_datasets:
            print(f"Warning: Invalid dataset(s) specified: {invalid_datasets}")
            print(f"Available datasets: {list(available_datasets)}")
            return

        data_frame = data_frame[data_frame["dataset"].isin(args.dataset)].reset_index(drop=True)
        print(f"Filtered to dataset(s): {', '.join(args.dataset)}")
        print(f"Total samples: {len(data_frame)}\n")

    # Determine if we need to run per-dataset or combined
    if args.per_dataset:
        # Run separate CV for each dataset
        datasets = sorted(data_frame["dataset"].unique())
        print(f"\nRunning per-dataset analysis for {len(datasets)} datasets: {', '.join(datasets)}\n")

        for dataset in datasets:
            dataset_df = data_frame[data_frame["dataset"] == dataset].reset_index(drop=True)

            # Create dataset-specific output directory
            dataset_output_dir = args.output_dir / dataset
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

            # Run CV for this dataset
            run_cross_validation_for_dataset(
                args=args,
                data_frame=dataset_df,
                dataset_name=dataset,
                device=device,
                output_dir=dataset_output_dir
            )

    else:
        # Run combined CV for all datasets
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_names = sorted(data_frame["dataset"].unique())
        if len(dataset_names) > 1:
            dataset_label = f"Combined ({', '.join(dataset_names)})"
        else:
            dataset_label = dataset_names[0]

        run_cross_validation_for_dataset(
            args=args,
            data_frame=data_frame,
            dataset_name=dataset_label,
            device=device,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()
