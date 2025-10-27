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

TRAINED_TRIAL_LABELS = {"testing_2", "testing_4", "testing_5"}
CONTROL_DATASET = "hex_control"


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
    csv_df.insert(0, "fold", range(1, len(csv_df) + 1))
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


def main() -> None:
    args = parse_args()
    device = determine_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_frame = load_behavioral_dataframe(args.data, validate=True)
    fold_iterator = make_group_kfold(
        args.data,
        n_splits=args.folds,
        groups=data_frame["fly"].to_numpy(),
        validate=False,
    )

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


if __name__ == "__main__":
    main()
