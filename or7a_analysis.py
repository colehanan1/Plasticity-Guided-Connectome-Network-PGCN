"""
Utilities for processing and analysing FlyWire OR7a (ORN_DL5) neurons.

This module provides a validated data-loading workflow, connectivity analysis
helpers, ranking utilities for experimental targeting, and publication-grade
visualisation routines tailored to optogenetic suppression experiments.

Example
-------
>>> from pathlib import Path
>>> processor = OR7aDataProcessor(Path("data/flywire/search_results_or7a.csv"))
>>> data = processor.load_data()
>>> analysis = OR7aAnalysis(data)
>>> summary = analysis.synaptic_summary()
>>> ranking = analysis.rank_neurons(top_n=10)
>>> analysis.export_target_list(Path("artifacts/or7a_targets.csv"), top_n=15)
>>> fig, ax = analysis.plot_synapse_distribution(metric="input_synapses")
>>> fig.savefig("artifacts/or7a_input_distribution.png", dpi=300)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:  # SciPy is part of the project requirements but keep a graceful fallback.
    from scipy import stats
except ImportError:  # pragma: no cover - only executed if SciPy missing.
    stats = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# Use a consistent visual aesthetic across all plots.
sns.set_theme(context="talk", style="whitegrid", palette="deep")


REQUIRED_COLUMNS: Tuple[str, ...] = (
    "root_id",
    "label",
    "name",
    "nt_type",
    "flow",
    "super_class",
    "class",
    "cell_type",
    "nerve",
    "connectivity_tag",
    "side",
    "input_synapses",
    "output_synapses",
)

LIST_LIKE_COLUMNS: Tuple[str, ...] = ("label", "cell_type", "connectivity_tag")
# Attach metadata to each neuron for downstream reporting.
TARGET_METADATA_COLUMNS: Tuple[str, ...] = (
    "root_id",
    "name",
    "side",
    "input_synapses",
    "output_synapses",
    "connectivity_score",
    "connectivity_rank",
)


class OR7aDataError(RuntimeError):
    """Raised when OR7a neuron data fails validation."""


@dataclass
class OR7aDataProcessor:
    """Loader and validator for FlyWire OR7a neuron exports.

    Parameters
    ----------
    csv_path:
        Path to the FlyWire search export containing OR7a neurons.
    expected_count:
        Optional row-count check. Defaults to 41 OR7a neurons.
    strict:
        When True, raise :class:`OR7aDataError` on any validation failure.
        When False, emit warnings and attempt to coerce the data.
    """

    csv_path: Path
    expected_count: Optional[int] = 41
    strict: bool = True

    _data: Optional[pd.DataFrame] = None

    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """Read, clean, and validate the OR7a dataset."""

        if self._data is not None and not force_reload:
            return self._data.copy()

        if not self.csv_path.exists():
            raise OR7aDataError(f"File not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        df = self._normalise_list_columns(df)
        df = self._coerce_types(df)
        self._validate(df)
        self._data = df
        return df.copy()

    def _normalise_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse FlyWire's semicolon-separated list representation."""

        cleaned = df.copy()
        for column in LIST_LIKE_COLUMNS:
            if column not in cleaned.columns:
                continue
            cleaned[column] = cleaned[column].apply(self._parse_semicolon_list)
        return cleaned

    @staticmethod
    def _parse_semicolon_list(value: object) -> List[str]:
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text or text == "[]":
            return []
        text = text.strip("[]")
        if not text:
            return []
        items = [
            token.strip().strip(" '\"")
            for token in text.split(";")
            if token.strip()
        ]
        return [item for item in items if item]

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure numeric columns use proper numeric dtypes."""

        coerced = df.copy()
        for column in ("input_synapses", "output_synapses", "root_id"):
            if column in coerced.columns:
                coerced[column] = pd.to_numeric(coerced[column], errors="raise")
        coerced["side"] = coerced["side"].str.lower().str.strip()
        return coerced

    def _validate(self, df: pd.DataFrame) -> None:
        """Enforce schema and sanity checks on the dataset."""

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            self._handle_error(f"Dataset missing required columns: {missing}")

        if self.expected_count is not None and len(df) != self.expected_count:
            self._handle_error(
                f"Expected {self.expected_count} neurons but found {len(df)}"
            )

        duplicate_ids = df["root_id"].duplicated()
        if duplicate_ids.any():
            dupe_values = df.loc[duplicate_ids, "root_id"].tolist()
            self._handle_error(f"Duplicate root_id entries detected: {dupe_values}")

        if not set(df["side"].unique()).issubset({"left", "right"}):
            self._handle_error("`side` column must only contain 'left' or 'right'.")

        for column in ("input_synapses", "output_synapses"):
            if (df[column] <= 0).any():
                self._handle_error(f"Column `{column}` contains non-positive values.")

    def _handle_error(self, message: str) -> None:
        if self.strict:
            raise OR7aDataError(message)
        logger.warning(message)

    @property
    def data(self) -> pd.DataFrame:
        """Return a defensive copy of the cached data."""

        if self._data is None:
            raise OR7aDataError("Data has not been loaded yet. Call `load_data()` first.")
        return self._data.copy()


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for independent samples."""

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")

    mean_diff = float(np.mean(x) - np.mean(y))
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (
        (nx + ny) - 2
    )
    if pooled_var <= 0:
        return float("nan")
    return mean_diff / math.sqrt(pooled_var)


def _welch_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Welch's t-test with fallback when SciPy is unavailable."""

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan"), float("nan"), float("nan")

    var_x, var_y = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    mean_diff = float(np.mean(x) - np.mean(y))
    se = math.sqrt(var_x / nx + var_y / ny)
    if se == 0:
        return float("nan"), float("nan"), float("nan")

    t_stat = mean_diff / se
    df_num = (var_x / nx + var_y / ny) ** 2
    df_den = (var_x ** 2) / (nx**2 * (nx - 1)) + (var_y ** 2) / (ny**2 * (ny - 1))
    df = df_num / df_den if df_den != 0 else float("nan")

    if stats is not None and not math.isnan(df):
        p_value = 2 * stats.t.sf(abs(t_stat), df)
    else:
        # Two-sided p-value using normal approximation when SciPy missing or df invalid.
        p_value = 2 * (1 - stats_norm_cdf(abs(t_stat)))
    return t_stat, p_value, df


def stats_norm_cdf(z: float) -> float:
    """Cumulative distribution for standard normal (SciPy fallback)."""

    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class OR7aAnalysis:
    """Connectivity and ranking utilities for OR7a neuron populations."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        missing_metrics = {
            column for column in ("input_synapses", "output_synapses") if column not in self.data
        }
        if missing_metrics:
            raise OR7aDataError(f"Data missing required metrics: {missing_metrics}")

    def synaptic_summary(self) -> pd.DataFrame:
        """Return descriptive statistics for synaptic counts."""

        stats_frames = []
        for metric in ("input_synapses", "output_synapses"):
            series = self.data[metric]
            stats_frames.append(
                pd.Series(
                    {
                        "metric": metric,
                        "count": series.count(),
                        "mean": series.mean(),
                        "std": series.std(ddof=1),
                        "median": series.median(),
                        "iqr": series.quantile(0.75) - series.quantile(0.25),
                        "min": series.min(),
                        "max": series.max(),
                    }
                )
            )
        return pd.DataFrame(stats_frames).set_index("metric")

    def hemispheric_summary(self) -> pd.DataFrame:
        """Summaries by hemisphere with Welch's t-tests and effect sizes."""

        results: List[Dict[str, float]] = []
        for metric in ("input_synapses", "output_synapses"):
            subset = self.data[["side", metric]].dropna()
            groups = {side: group.values for side, group in subset.groupby("side")[metric]}
            left = groups.get("left", np.array([]))
            right = groups.get("right", np.array([]))
            if len(left) == 0 or len(right) == 0:
                logger.warning("Insufficient data for hemispheric comparison on %s", metric)
                continue

            t_stat, p_value, dof = _welch_ttest(left, right)
            effect = _cohens_d(left, right)
            mean_diff = float(np.mean(left) - np.mean(right))
            se_diff = math.sqrt(
                np.var(left, ddof=1) / len(left) + np.var(right, ddof=1) / len(right)
            )
            if stats is not None and not math.isnan(dof):
                ci_multiplier = stats.t.ppf(0.975, dof)
            else:
                ci_multiplier = 1.96
            if math.isnan(se_diff):
                ci_low = float("nan")
                ci_high = float("nan")
            else:
                ci_low = mean_diff - ci_multiplier * se_diff
                ci_high = mean_diff + ci_multiplier * se_diff

            results.append(
                {
                    "metric": metric,
                    "left_mean": float(np.mean(left)),
                    "right_mean": float(np.mean(right)),
                    "mean_difference": mean_diff,
                    "ci95_lower": ci_low,
                    "ci95_upper": ci_high,
                    "t_statistic": t_stat,
                    "degrees_of_freedom": dof,
                    "p_value": p_value,
                    "cohens_d": effect,
                    "n_left": float(len(left)),
                    "n_right": float(len(right)),
                }
            )
        return pd.DataFrame(results).set_index("metric")

    def rank_neurons(
        self,
        top_n: Optional[int] = None,
        *,
        input_weight: float = 0.4,
        output_weight: float = 0.6,
        minimum_inputs: Optional[int] = None,
        minimum_outputs: Optional[int] = None,
    ) -> pd.DataFrame:
        """Rank neurons by a weighted connectivity score."""

        if not math.isclose(input_weight + output_weight, 1.0, rel_tol=1e-3):
            raise ValueError("Weights must sum to 1.0")

        ranked = self.data.copy()
        if minimum_inputs is not None:
            ranked = ranked[ranked["input_synapses"] >= minimum_inputs]
        if minimum_outputs is not None:
            ranked = ranked[ranked["output_synapses"] >= minimum_outputs]
        if ranked.empty:
            raise OR7aDataError("No neurons remain after applying filters.")

        for metric, weight in (("input_synapses", input_weight), ("output_synapses", output_weight)):
            std = ranked[metric].std(ddof=1)
            if std == 0:
                ranked[f"z_{metric}"] = 0.0
            else:
                ranked[f"z_{metric}"] = (ranked[metric] - ranked[metric].mean()) / std
            ranked[f"weighted_{metric}"] = weight * ranked[f"z_{metric}"]

        ranked["connectivity_score"] = ranked[["weighted_input_synapses", "weighted_output_synapses"]].sum(axis=1)
        ranked = ranked.sort_values("connectivity_score", ascending=False).reset_index(drop=True)
        ranked["connectivity_rank"] = np.arange(1, len(ranked) + 1)
        if top_n is not None:
            ranked = ranked.head(top_n)
        return ranked

    def export_target_list(
        self,
        output_path: Path,
        *,
        top_n: Optional[int] = None,
        input_weight: float = 0.4,
        output_weight: float = 0.6,
        minimum_inputs: Optional[int] = None,
        minimum_outputs: Optional[int] = None,
        include_metadata_columns: Optional[Sequence[str]] = None,
    ) -> Path:
        """Write ranked targets to CSV for optogenetic experiments."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ranked = self.rank_neurons(
            top_n=top_n,
            input_weight=input_weight,
            output_weight=output_weight,
            minimum_inputs=minimum_inputs,
            minimum_outputs=minimum_outputs,
        )

        metadata_cols = set(TARGET_METADATA_COLUMNS)
        if include_metadata_columns:
            metadata_cols |= {col for col in include_metadata_columns if col in ranked.columns}

        export_columns = [col for col in ranked.columns if col in metadata_cols]
        if not export_columns:
            raise OR7aDataError("No columns selected for export.")

        ranked.to_csv(output_path, index=False, columns=export_columns)
        logger.info("Exported OR7a target list to %s", output_path)
        return output_path

    def plot_synapse_distribution(
        self,
        *,
        metric: str,
        bins: int = 12,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Histogram + KDE of synaptic counts across the population."""

        if metric not in {"input_synapses", "output_synapses"}:
            raise ValueError("Metric must be 'input_synapses' or 'output_synapses'.")

        fig, ax = self._prepare_axes(ax)
        sns.histplot(
            data=self.data,
            x=metric,
            bins=bins,
            kde=True,
            ax=ax,
            color="#1f77b4" if metric == "input_synapses" else "#ff7f0e",
        )
        ax.set_title(f"OR7a {metric.replace('_', ' ').title()} Distribution")
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("Neuron count")

        if save_path is not None:
            self._save_figure(fig, save_path)
        return fig, ax

    def plot_hemispheric_comparison(
        self,
        *,
        metric: str,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Boxplot + swarm overlay comparing hemispheres."""

        if metric not in {"input_synapses", "output_synapses"}:
            raise ValueError("Metric must be 'input_synapses' or 'output_synapses'.")

        fig, ax = self._prepare_axes(ax)
        sns.boxplot(data=self.data, x="side", y=metric, ax=ax, whis=1.5, palette="deep")
        sns.swarmplot(data=self.data, x="side", y=metric, ax=ax, color="black", size=5, alpha=0.7)
        ax.set_title(f"OR7a {metric.replace('_', ' ').title()} by Hemisphere")
        ax.set_xlabel("Hemisphere")
        ax.set_ylabel(metric.replace("_", " ").title())

        if save_path is not None:
            self._save_figure(fig, save_path)
        return fig, ax

    def plot_input_output_relationship(
        self,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Scatter plot relating input and output synapses per neuron."""

        fig, ax = self._prepare_axes(ax)
        sns.scatterplot(
            data=self.data,
            x="input_synapses",
            y="output_synapses",
            hue="side",
            style="side",
            ax=ax,
            s=80,
        )
        ax.set_title("OR7a Input vs Output Synapses")
        ax.set_xlabel("Input synapses (afferent)")
        ax.set_ylabel("Output synapses (efferent)")
        ax.legend(title="Hemisphere", loc="best")

        if save_path is not None:
            self._save_figure(fig, save_path)
        return fig, ax

    @staticmethod
    def _prepare_axes(ax: Optional[plt.Axes]) -> Tuple[plt.Figure, plt.Axes]:
        if ax is not None:
            return ax.figure, ax
        fig, ax = plt.subplots(figsize=(8, 6))
        return fig, ax

    @staticmethod
    def _save_figure(fig: plt.Figure, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        logger.info("Saved figure to %s", path)


def example_usage() -> None:
    """Minimal end-to-end demonstration intended for documentation."""

    csv_path = Path("data/flywire/search_results_or7a.csv")
    processor = OR7aDataProcessor(csv_path)
    data = processor.load_data()

    analysis = OR7aAnalysis(data)
    summary = analysis.synaptic_summary()
    hemispheres = analysis.hemispheric_summary()
    ranking = analysis.rank_neurons(top_n=10)

    artifacts_dir = Path("artifacts/or7a_analysis")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(artifacts_dir / "synaptic_summary.csv")
    hemispheres.to_csv(artifacts_dir / "hemisphere_summary.csv")
    ranking.to_csv(artifacts_dir / "top10_targets.csv", index=False)

    analysis.plot_synapse_distribution(metric="input_synapses", save_path=artifacts_dir / "inputs_hist.png")
    analysis.plot_synapse_distribution(metric="output_synapses", save_path=artifacts_dir / "outputs_hist.png")
    analysis.plot_hemispheric_comparison(metric="input_synapses", save_path=artifacts_dir / "hemisphere_inputs.png")
    analysis.plot_input_output_relationship(save_path=artifacts_dir / "input_output_scatter.png")


if __name__ == "__main__":  # pragma: no cover - manual execution entry point.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    example_usage()
