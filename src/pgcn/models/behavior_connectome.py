"""Behavior-connectome analysis utilities linking structure and performance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:  # pragma: no cover - optional dependency resolved at runtime
    import torch
except ImportError:  # pragma: no cover - handled for doc builds / CI without torch
    torch = None  # type: ignore[assignment]

from .reservoir import DrosophilaReservoir

__all__ = ["BehaviorConnectomeAnalyzer", "GlomerulusStructuralSummary"]


@dataclass(frozen=True)
class GlomerulusStructuralSummary:
    """Aggregated structural properties for a single glomerulus."""

    glomerulus: str
    pn_indices: tuple[int, ...]
    pn_count: int
    kc_coverage: int
    total_edges: int
    mean_edges_per_pn: float


class BehaviorConnectomeAnalyzer:
    """Analyse how connectome structure aligns with behavioural outcomes."""

    def __init__(self, cache_dir: Path | str, behavior_data: pd.DataFrame) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to instantiate BehaviorConnectomeAnalyzer.")
        if not {"dataset", "trial_label", "prediction"}.issubset(behavior_data.columns):
            raise ValueError(
                "Behavior dataframe must include 'dataset', 'trial_label', and 'prediction' columns."
            )
        self.behavior = behavior_data.copy()
        self.behavior["prediction"] = self.behavior["prediction"].astype(float)
        self.reservoir = DrosophilaReservoir(cache_dir=cache_dir)
        if not hasattr(self.reservoir, "pn_kc_mask"):
            raise AttributeError("Reservoir instance must expose a pn_kc_mask buffer.")
        mask = torch.as_tensor(self.reservoir.pn_kc_mask, dtype=torch.float32)
        self._pn_kc_mask = mask.cpu().numpy()

    @property
    def pn_count(self) -> int:
        return self.reservoir.n_pn

    @property
    def kc_count(self) -> int:
        return self.reservoir.n_kc

    def structural_summary(self, glomerulus_assignments: pd.DataFrame) -> dict[str, GlomerulusStructuralSummary]:
        if not {"pn_index", "glomerulus"}.issubset(glomerulus_assignments.columns):
            raise ValueError("Glomerulus assignments require 'pn_index' and 'glomerulus' columns.")
        grouped = glomerulus_assignments.groupby("glomerulus")
        pn_degree = self._pn_kc_mask.sum(axis=0)
        summary: dict[str, GlomerulusStructuralSummary] = {}
        for glomerulus, frame in grouped:
            pn_indices = tuple(int(idx) for idx in frame["pn_index"].unique())
            total_edges = int(np.sum(pn_degree[np.array(pn_indices)]))
            kc_targets = self._pn_kc_mask[:, pn_indices].sum(axis=1) > 0
            kc_coverage = int(np.count_nonzero(kc_targets))
            mean_edges = float(total_edges / len(pn_indices)) if pn_indices else 0.0
            summary[glomerulus] = GlomerulusStructuralSummary(
                glomerulus=glomerulus,
                pn_indices=pn_indices,
                pn_count=len(pn_indices),
                kc_coverage=kc_coverage,
                total_edges=total_edges,
                mean_edges_per_pn=mean_edges,
            )
        return summary

    def _behaviour_summary(self) -> pd.DataFrame:
        return (
            self.behavior.groupby(["dataset", "trial_label"])["prediction"]
            .agg(success_rate="mean", n_trials="count")
            .reset_index()
        )

    def analyze_glomerulus_enrichment(
        self,
        glomerulus_assignments: pd.DataFrame,
        trial_to_glomerulus: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        """Return enrichment statistics linking behaviour with structural motifs."""

        structural = self.structural_summary(glomerulus_assignments)
        behaviour = self._behaviour_summary()
        if trial_to_glomerulus is None and "trial_label" in glomerulus_assignments.columns:
            candidate = glomerulus_assignments.dropna(subset=["trial_label"])
            if not candidate.empty:
                trial_to_glomerulus = {
                    str(row.trial_label): str(row.glomerulus)
                    for row in candidate.itertuples(index=False)
                }
        cleaned_map: dict[str, str] = {}
        if trial_to_glomerulus is not None:
            for trial, glomerulus in trial_to_glomerulus.items():
                if glomerulus is None:
                    continue
                value = str(glomerulus).strip()
                if not value:
                    continue
                if value.lower() in {"unknown", "unknown_glomerulus", "todo", "tbd"}:
                    continue
                cleaned_map[str(trial)] = value
        reverse_map = cleaned_map
        if not reverse_map:
            raise ValueError(
                "No trial→glomerulus mapping available. Update configs/trial_to_glomerulus.yaml "
                "with real glomerulus labels or supply a populated mapping file via "
                "--trial-to-glomerulus."
            )
        enrichment_rows: list[dict[str, object]] = []
        for dataset, dataset_frame in behaviour.groupby("dataset"):
            for glomerulus, summary in structural.items():
                trial_mask = dataset_frame["trial_label"].map(reverse_map)
                matched = dataset_frame.loc[trial_mask == glomerulus]
                if matched.empty:
                    continue
                enrichment_rows.append(
                    {
                        "dataset": dataset,
                        "glomerulus": glomerulus,
                        "success_rate": matched["success_rate"].mean(),
                        "n_trials": int(matched["n_trials"].sum()),
                        "pn_count": summary.pn_count,
                        "kc_coverage": summary.kc_coverage,
                        "total_edges": summary.total_edges,
                        "mean_edges_per_pn": summary.mean_edges_per_pn,
                    }
                )
        if not enrichment_rows:
            raise ValueError(
                "No behavioural trials matched the provided glomerulus mapping. "
                "Verify that trial labels correspond to the mapping entries."
            )
        return pd.DataFrame(enrichment_rows)

    def structural_performance_correlation(
        self,
        glomerulus_assignments: pd.DataFrame,
        trial_to_glomerulus: Optional[Mapping[str, str]] = None,
        *,
        structural_metric: str = "total_edges",
    ) -> pd.DataFrame:
        """Compute correlations between structural statistics and behaviour."""

        enrichment = self.analyze_glomerulus_enrichment(glomerulus_assignments, trial_to_glomerulus)
        results: list[dict[str, object]] = []
        for dataset, frame in enrichment.groupby("dataset"):
            metric_values = frame[structural_metric].to_numpy(dtype=float)
            behaviour_values = frame["success_rate"].to_numpy(dtype=float)
            if len(metric_values) < 2 or np.all(metric_values == metric_values[0]):
                corr = np.nan
                p_value = np.nan
            else:
                corr, p_value = stats.pearsonr(metric_values, behaviour_values)
            results.append(
                {
                    "dataset": dataset,
                    "metric": structural_metric,
                    "correlation": corr,
                    "p_value": p_value,
                    "n": len(frame),
                }
            )
        return pd.DataFrame(results)
