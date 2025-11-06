"""
Generalised multi-ORN pathway tracing: ORNs → PNs → KCs → MBONs → behavior.

This script extends the validated OR7a pathway workflow to Or7a, Or13a, Or42b,
Or47b, and Or59b. For each receptor class it reproduces the complete circuit
analysis (pathway CSV, level summaries, target priorities, publication-quality
figures) and then aggregates comparative statistics and visualisations across
all receptors.
"""
from __future__ import annotations

import argparse
import json
import logging
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy import stats
except ImportError:  # pragma: no cover - SciPy optional
    stats = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
sns.set_theme(context="talk", style="whitegrid", palette="deep")

CELL_TYPE_MAPPING: Dict[str, str] = {
    "VA1v_adPN": "VA1v antenno-deuterocerebral projection neuron",
    "VA1v_vPN": "VA1v ventral projection neuron",
    "VA1d_adPN": "VA1d antenno-deuterocerebral projection neuron",
    "VA1d_vPN": "VA1d ventral projection neuron",
    "VA6_adPN": "VA6 antenno-deuterocerebral projection neuron",
    "DA4l_adPN": "DA4l antenno-deuterocerebral projection neuron",
    "DA4m_adPN": "DA4m antenno-deuterocerebral projection neuron",
    "DC2_adPN": "DC2 antenno-deuterocerebral projection neuron",
    "DL1_adPN": "DL1 antenno-deuterocerebral projection neuron",
    "DL5_adPN": "DL5 antenno-deuterocerebral projection neuron",
    "DM1_lPN": "DM1 lateral projection neuron",
    "DM4_adPN": "DM4 antenno-deuterocerebral projection neuron",
    "DP1l_adPN": "DP1l antenno-deuterocerebral projection neuron",
    "MZ_lv2PN": "Medial zone lateral ventral 2 projection neuron",
    "M_l2PNl20": "Multiglomerular lateral 2 projection neuron l20",
    "M_l2PNl21": "Multiglomerular lateral 2 projection neuron l21",
    "M_lPNm11C": "Multiglomerular lateral projection neuron m11C",
    "M_lvPNm25": "Multiglomerular lateral ventral projection neuron m25",
    "M_lvPNm27": "Multiglomerular lateral ventral projection neuron m27",
    "M_lvPNm35": "Multiglomerular lateral ventral projection neuron m35",
    "M_vPNml54": "Multiglomerular ventral projection neuron ml54",
    "M_vPNml63": "Multiglomerular ventral projection neuron ml63",
    "M_vPNml64": "Multiglomerular ventral projection neuron ml64",
    "M_vPNml65": "Multiglomerular ventral projection neuron ml65",
    "M_vPNml68": "Multiglomerular ventral projection neuron ml68",
    "M_vPNml72": "Multiglomerular ventral projection neuron ml72",
    "lLN1_bc": "lateral local neuron 1bc",
    "lLN2F_a": "lateral local neuron 2Fa",
    "lLN2F_b": "lateral local neuron 2Fb",
    "lLN2P_a": "lateral local neuron 2Pa",
    "lLN2P_b": "lateral local neuron 2Pb",
    "lLN2P_c": "lateral local neuron 2Pc",
    "lLN2T_b": "lateral local neuron 2Tb",
    "lLN2T_c": "lateral local neuron 2Tc",
    "lLN2T_d": "lateral local neuron 2Td",
    "lLN2T_e": "lateral local neuron 2Te",
    "lLN2X02": "lateral local neuron 2X02",
    "lLN2X03": "lateral local neuron 2X03",
    "lLN2X04": "lateral local neuron 2X04",
    "lLN2X05": "lateral local neuron 2X05",
    "lLN2X06": "lateral local neuron 2X06",
    "lLN2X07": "lateral local neuron 2X07",
    "lLN2X09": "lateral local neuron 2X09",
    "lLN2X10": "lateral local neuron 2X10",
    "lLN2X11": "lateral local neuron 2X11",
    "lLN2X12": "lateral local neuron 2X12",
    "lLN8": "lateral local neuron 8",
    "lLN9": "lateral local neuron 9",
    "il3LN6": "inner lateral 3 local neuron 6",
    "v2LN30": "ventral 2 local neuron 30",
    "v2LN31": "ventral 2 local neuron 31",
    "v2LN36": "ventral 2 local neuron 36",
    "v2LN3A1_b": "ventral 2 local neuron 3A1b",
    "vLN25": "ventral local neuron 25",
    "ALBN1": "antennal lobe broad neuron 1",
    "AL-AST1": "antennal lobe AST1 neuron",
    "ALIN1": "antennal lobe interneuron 1",
    "ALIN7": "antennal lobe interneuron 7",
    "ALON3": "antennal lobe output neuron 3",
    "CSD": "centrifugal sensory dendrite neuron",
    "LN60a": "local neuron 60a",
    "LN60b": "local neuron 60b",
    "CB0683": "callback neuron 0683",
    "CB1266": "callback neuron 1266",
    "CB1676": "callback neuron 1676",
    "CB1988": "callback neuron 1988",
    "CB2161": "callback neuron 2161",
    "ORN_DC2": "olfactory receptor neuron DC2 type",
    "ORN_DC3": "olfactory receptor neuron DC3 type",
    "ORN_DL5": "olfactory receptor neuron DL5 type",
    "ORN_DM1": "olfactory receptor neuron DM1 type",
    "ORN_DP1l": "olfactory receptor neuron DP1l type",
    "ORN_VA1v": "olfactory receptor neuron VA1v type",
}


def expand_cell_type_name(abbreviation: Optional[str]) -> Optional[str]:
    if abbreviation is None:
        return None
    if not isinstance(abbreviation, str):
        return abbreviation
    return CELL_TYPE_MAPPING.get(abbreviation, abbreviation)


def expand_cell_type_field(value: Union[str, Sequence[str], float, None]) -> Union[str, Sequence[str], float, None]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(";")]
        expanded = [expand_cell_type_name(part) for part in parts]
        return "; ".join(expanded)
    if isinstance(value, (list, tuple, set)):
        expanded = [expand_cell_type_name(str(item)) for item in value]
        return "; ".join(expanded)
    return value


def wrap_label(label: str, width: int = 32) -> str:
    if not isinstance(label, str):
        return label
    if len(label) <= width:
        return label
    return "\n".join(textwrap.wrap(label, width=width))


# --------------------------------------------------------------------------- #
# Dataclasses and configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ORNPathwayConfig:
    """Metadata describing a single ORN population."""

    name: str
    orn_csv: Path
    glomerulus: str
    neurotransmitter_profile: str
    pn_patterns: Tuple[str, ...]
    expected_orn_count: Optional[int] = None
    pn_root_ids: Optional[Tuple[int, ...]] = None
    color: str = "#1f77b4"


@dataclass
class CircuitLevel:
    """Summary of a circuit level."""

    level: int
    name: str
    root_ids: List[int]
    neuron_count: int
    convergence_ratio: Optional[float]
    mean_synapses: Optional[float]
    connections_to_next: Optional[pd.DataFrame]


@dataclass
class ORNPathwayResult:
    """Container for per-ORN outputs."""

    config: ORNPathwayConfig
    pathway_df: pd.DataFrame
    summaries: Dict[str, pd.DataFrame]
    target_priorities: pd.DataFrame
    levels: Dict[int, CircuitLevel]
    circuit_levels: Dict[int, str]


# --------------------------------------------------------------------------- #
# Baseline classification helpers
# --------------------------------------------------------------------------- #


BASE_CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "PN": ["adpn", "lpn", "_pn"],
    "KC": ["kc", "kenyon"],
    "MBON": ["mbon"],
    "Motor": ["dnp", "dna", "dnb", "mdn", "motor"],
    "Descending": ["dn"],
    "Central_Complex": [" fb", " eb", " pb", " no", "fb_", "eb_", "pb_", "no_"],
    "Other": [],
    "Unknown": [],
}

BEHAVIOR_CATEGORIES = ("Motor", "Descending", "Central_Complex")


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #


def gini(values: Iterable[float]) -> float:
    """Compute the Gini coefficient."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(arr)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def normalise_patterns(patterns: Iterable[str]) -> List[str]:
    """Lower-case and trim pattern list."""
    cleaned: List[str] = []
    for pattern in patterns:
        text = pattern.strip().lower()
        if text:
            cleaned.append(text)
    return cleaned


def load_connections(path: Path) -> pd.DataFrame:
    """Read FlyWire connections table."""
    logger.info("Loading connections from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Connections file not found: {path}")

    base_cols = ["pre_root_id", "post_root_id", "syn_count", "neuropil", "nt_type"]
    try:
        df = pd.read_csv(path, usecols=[col for col in base_cols if col in pd.read_csv(path, nrows=0).columns])
    except ValueError:
        df = pd.read_csv(path)

    rename_map = {}
    if "pre_pt_root_id" in df.columns:
        rename_map["pre_pt_root_id"] = "pre_root_id"
    if "post_pt_root_id" in df.columns:
        rename_map["post_pt_root_id"] = "post_root_id"
    if "size" in df.columns and "syn_count" not in df.columns:
        rename_map["size"] = "syn_count"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"pre_root_id", "post_root_id", "syn_count"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Connections file missing required columns: {sorted(missing)}")

    keep_order = ["pre_root_id", "post_root_id", "syn_count"]
    for optional in ("neuropil", "nt_type"):
        if optional in df.columns:
            keep_order.append(optional)
    df = df[keep_order]

    df["pre_root_id"] = pd.to_numeric(df["pre_root_id"], errors="raise", downcast="integer")
    df["post_root_id"] = pd.to_numeric(df["post_root_id"], errors="raise", downcast="integer")
    df["syn_count"] = pd.to_numeric(df["syn_count"], errors="raise", downcast="integer")

    logger.info("Loaded %d connections", len(df))
    return df


def load_cell_types(path: Path) -> pd.DataFrame:
    """Read FlyWire cell type annotations."""
    logger.info("Loading cell type annotations from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Cell types file not found: {path}")

    try:
        df = pd.read_csv(path, usecols=["root_id", "cell_type"])
    except ValueError:
        df = pd.read_csv(path)
        if "primary_type" in df.columns and "cell_type" not in df.columns:
            df = df.rename(columns={"primary_type": "cell_type"})
        df = df[["root_id", "cell_type"]]

    df["root_id"] = pd.to_numeric(df["root_id"], errors="coerce").astype("Int64")
    df["cell_type"] = df["cell_type"].fillna("unknown")
    logger.info("Loaded cell type annotations for %d neurons", len(df))
    return df


# --------------------------------------------------------------------------- #
# ORN pathway tracer
# --------------------------------------------------------------------------- #


class ORNPathwayTracer:
    """Trace ORN pathways through PN, KC, MBON, and behavioral levels."""

    def __init__(
        self,
        config: ORNPathwayConfig,
        connections: pd.DataFrame,
        cell_types: pd.DataFrame,
        min_synapses: int = 3,
        max_levels: int = 5,
    ) -> None:
        self.config = config
        self.connections = connections
        self.cell_types = cell_types
        self.min_synapses = int(min_synapses)
        self.max_levels = max(1, min(max_levels, 5))

        self.circuit_levels = {
            0: f"{config.name}_ORN",
            1: f"{config.glomerulus}_PN",
            2: "Kenyon_Cell",
            3: "MBON",
            4: "Behavioral_Output",
        }
        self.level_titles = {
            0: f"{config.name}_ORN",
            1: f"{config.glomerulus}_Projection_Neurons",
            2: "Kenyon_Cells",
            3: "MBONs",
            4: "Behavioral_Outputs",
        }

        self.category_patterns = copy.deepcopy(BASE_CATEGORY_PATTERNS)
        pn_patterns = normalise_patterns(self.category_patterns.get("PN", []))
        pn_patterns.extend(normalise_patterns(config.pn_patterns))
        self.category_patterns["PN"] = list(dict.fromkeys(pn_patterns))
        for key, values in list(self.category_patterns.items()):
            self.category_patterns[key] = normalise_patterns(values)

        self.levels: Dict[int, CircuitLevel] = {}
        self.complete_pathway: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Core helpers
    # ------------------------------------------------------------------ #
    def identify_cell_category(self, cell_type: str) -> str:
        """Assign a broad category to a cell type."""
        if pd.isna(cell_type):
            return "Unknown"
        lowered = str(cell_type).lower()
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if pattern and pattern in lowered:
                    return category
        return "Other"

    def _load_orn_root_ids(self) -> List[int]:
        df = pd.read_csv(self.config.orn_csv)
        if "root_id" not in df.columns:
            raise ValueError(f"{self.config.name}: missing `root_id` column in {self.config.orn_csv}")
        if self.config.expected_orn_count is not None and len(df) != self.config.expected_orn_count:
            logger.warning(
                "%s: expected %d ORNs but found %d",
                self.config.name,
                self.config.expected_orn_count,
                len(df),
            )
        return pd.to_numeric(df["root_id"], errors="raise").astype("int64").tolist()

    def _get_outputs(self, source_ids: Sequence[int]) -> pd.DataFrame:
        if not source_ids:
            return pd.DataFrame(columns=["pre_root_id", "post_root_id", "syn_count"])
        outputs = self.connections[self.connections["pre_root_id"].isin(source_ids)].copy()
        outputs = outputs[outputs["syn_count"] >= self.min_synapses]
        if outputs.empty:
            return outputs
        outputs = outputs.merge(
            self.cell_types,
            left_on="post_root_id",
            right_on="root_id",
            how="left",
        )
        outputs = outputs.drop(columns=["root_id"])
        outputs["target_category"] = outputs["cell_type"].apply(self.identify_cell_category)
        outputs["cell_type"] = outputs["cell_type"].apply(expand_cell_type_field)
        return outputs

    @staticmethod
    def _filter_targets(
        outputs: pd.DataFrame,
        allowed_categories: Optional[Tuple[str, ...]],
    ) -> Tuple[List[int], pd.DataFrame]:
        if outputs.empty:
            return [], outputs
        if allowed_categories is None:
            targets = outputs["post_root_id"].unique().tolist()
            return targets, outputs
        mask = outputs["target_category"].isin(allowed_categories)
        filtered = outputs[mask].copy()
        targets = filtered["post_root_id"].unique().tolist()
        return targets, filtered

    def _summarise_level(
        self,
        level_num: int,
        level_name: str,
        source_ids: List[int],
        allowed_categories: Optional[Tuple[str, ...]],
    ) -> CircuitLevel:
        outputs = self._get_outputs(source_ids)
        targets, filtered_outputs = self._filter_targets(outputs, allowed_categories)

        convergence = len(source_ids) / len(targets) if targets else None
        mean_syn = filtered_outputs["syn_count"].mean() if not filtered_outputs.empty else None

        circuit_level = CircuitLevel(
            level=level_num,
            name=level_name,
            root_ids=targets,
            neuron_count=len(targets),
            convergence_ratio=convergence,
            mean_synapses=mean_syn,
            connections_to_next=outputs if not outputs.empty else None,
        )
        return circuit_level

    # ------------------------------------------------------------------ #
    # Public workflow
    # ------------------------------------------------------------------ #
    def trace(self) -> None:
        """Trace all circuit levels for this ORN population."""
        logger.info("\n%s: tracing pathway", self.config.name)

        orn_root_ids = self._load_orn_root_ids()
        self.levels[0] = CircuitLevel(
            level=0,
            name=self.level_titles[0],
            root_ids=orn_root_ids,
            neuron_count=len(orn_root_ids),
            convergence_ratio=None,
            mean_synapses=None,
            connections_to_next=None,
        )
        logger.info("  Level 0 (%s): %d ORNs", self.level_titles[0], len(orn_root_ids))

        source_ids = orn_root_ids
        pn_root_ids: List[int] = []
        kc_root_ids: List[int] = []
        mbon_root_ids: List[int] = []

        if self.max_levels >= 2:
            level_1 = self._summarise_level(1, self.level_titles[1], source_ids, ("PN",))
            self.levels[1] = level_1
            pn_root_ids = level_1.root_ids
            if self.config.pn_root_ids:
                whitelist = set(self.config.pn_root_ids)
                filtered_ids = [rid for rid in pn_root_ids if rid in whitelist]
                if filtered_ids:
                    pn_root_ids = filtered_ids
                    if level_1.connections_to_next is not None:
                        mask = level_1.connections_to_next["post_root_id"].isin(whitelist)
                        level_1.connections_to_next = level_1.connections_to_next[mask].copy()
                else:
                    logger.warning(
                        "%s: PN whitelist provided but no matches found; using detected PN set",
                        self.config.name,
                    )
            if level_1.connections_to_next is not None:
                level_1.root_ids = pn_root_ids
                level_1.neuron_count = len(pn_root_ids)
            logger.info("  Level 1 (%s): %d PNs", self.level_titles[1], len(pn_root_ids))

        if self.max_levels >= 3 and pn_root_ids:
            level_2 = self._summarise_level(2, self.level_titles[2], pn_root_ids, ("KC",))
            self.levels[2] = level_2
            kc_root_ids = level_2.root_ids
            logger.info("  Level 2 (%s): %d KCs", self.level_titles[2], len(kc_root_ids))

        if self.max_levels >= 4 and kc_root_ids:
            level_3 = self._summarise_level(3, self.level_titles[3], kc_root_ids, ("MBON",))
            self.levels[3] = level_3
            mbon_root_ids = level_3.root_ids
            logger.info("  Level 3 (%s): %d MBONs", self.level_titles[3], len(mbon_root_ids))

        if self.max_levels >= 5 and mbon_root_ids:
            level_4 = self._summarise_level(4, self.level_titles[4], mbon_root_ids, BEHAVIOR_CATEGORIES)
            self.levels[4] = level_4
            logger.info("  Level 4 (%s): %d behavioral targets", self.level_titles[4], level_4.neuron_count)

    def build_complete_pathway_dataframe(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for level_num, level in sorted(self.levels.items()):
            if level.connections_to_next is None:
                continue
            df = level.connections_to_next.copy()
            df["source_level"] = level_num
            df["source_level_name"] = level.name
            df["target_level"] = level_num + 1
            df["target_level_name"] = self.circuit_levels.get(level_num + 1, f"Level_{level_num+1}")
            frames.append(df)

        if frames:
            self.complete_pathway = pd.concat(frames, ignore_index=True)
        else:
            self.complete_pathway = pd.DataFrame(
                columns=[
                    "pre_root_id",
                    "post_root_id",
                    "syn_count",
                    "root_id",
                    "cell_type",
                    "target_category",
                    "source_level",
                    "source_level_name",
                    "target_level",
                    "target_level_name",
                ]
            )
        return self.complete_pathway

    def generate_pathway_summary(self) -> Dict[str, pd.DataFrame]:
        summaries: Dict[str, pd.DataFrame] = {}

        level_summary = []
        for level_num, level in sorted(self.levels.items()):
            level_summary.append(
                {
                    "level": level_num,
                    "level_name": level.name,
                    "neuron_count": level.neuron_count,
                    "convergence_ratio": level.convergence_ratio or 0,
                    "mean_synapses_per_connection": level.mean_synapses or 0,
                }
            )
        summaries["by_level"] = pd.DataFrame(level_summary)

        if self.complete_pathway is not None and not self.complete_pathway.empty:
            conn_stats = (
                self.complete_pathway.groupby(["source_level", "target_level"])
                .agg(
                    pre_root_id=("pre_root_id", "nunique"),
                    post_root_id=("post_root_id", "nunique"),
                    total_synapses=("syn_count", "sum"),
                    mean_synapses=("syn_count", "mean"),
                    median_synapses=("syn_count", "median"),
                    std_synapses=("syn_count", "std"),
                )
                .reset_index()
            )
            conn_stats = conn_stats.rename(
                columns={
                    "pre_root_id": "source_neurons",
                    "post_root_id": "target_neurons",
                }
            )
            summaries["connections"] = conn_stats

            category_dist = (
                self.complete_pathway.groupby(["source_level", "target_category"])
                .agg(
                    num_neurons=("post_root_id", "nunique"),
                    total_synapses=("syn_count", "sum"),
                    mean_synapses=("syn_count", "mean"),
                )
                .reset_index()
                .rename(columns={"target_category": "category"})
            )
            summaries["categories"] = category_dist

        bottlenecks = []
        for i in range(len(self.circuit_levels) - 1):
            if i in self.levels and i + 1 in self.levels:
                source_count = self.levels[i].neuron_count
                target_count = self.levels[i + 1].neuron_count
                if source_count > 0:
                    expansion_ratio = target_count / source_count
                    severity = (
                        "High"
                        if expansion_ratio < 0.5
                        else ("Medium" if expansion_ratio < 1.0 else "Low")
                    )
                    bottlenecks.append(
                        {
                            "transition": f"{self.circuit_levels[i]} → {self.circuit_levels[i + 1]}",
                            "source_neurons": source_count,
                            "target_neurons": target_count,
                            "expansion_ratio": expansion_ratio,
                            "bottleneck_severity": severity,
                        }
                    )
        summaries["bottlenecks"] = pd.DataFrame(bottlenecks)
        return summaries

    def identify_critical_targets(self) -> pd.DataFrame:
        if self.complete_pathway is None or self.complete_pathway.empty:
            return pd.DataFrame(
                columns=[
                    "root_id",
                    "cell_type",
                    "category",
                    "total_synapses",
                    "num_connections",
                    "num_levels",
                    "importance_score",
                ]
            )

        neuron_importance: Dict[int, Dict[str, object]] = defaultdict(
            lambda: {"total_synapses": 0, "connections": 0, "levels": set()}
        )

        for _, conn in self.complete_pathway.iterrows():
            post_id = int(conn["post_root_id"])
            neuron_importance[post_id]["total_synapses"] = (
                neuron_importance[post_id]["total_synapses"] + int(conn["syn_count"])
            )
            neuron_importance[post_id]["connections"] = (
                neuron_importance[post_id]["connections"] + 1
            )
            neuron_importance[post_id]["levels"].add(int(conn["target_level"]))

        records = []
        for root_id, stats in neuron_importance.items():
            cell_info = self.cell_types[self.cell_types["root_id"] == root_id]
            raw_cell_type = cell_info["cell_type"].iloc[0] if not cell_info.empty else "Unknown"
            expanded_label = expand_cell_type_field(raw_cell_type)
            if self.complete_pathway is not None:
                pathway_labels = self.complete_pathway[
                    self.complete_pathway["post_root_id"] == root_id
                ]["cell_type"].dropna()
                if not pathway_labels.empty:
                    expanded_label = pathway_labels.iloc[0]
            records.append(
                {
                    "root_id": root_id,
                    "cell_type": expanded_label,
                    "category": self.identify_cell_category(raw_cell_type),
                    "total_synapses": stats["total_synapses"],
                    "num_connections": stats["connections"],
                    "num_levels": len(stats["levels"]),
                    "importance_score": stats["total_synapses"] * stats["connections"],  # type: ignore[arg-type]
                }
            )
        targets_df = pd.DataFrame(records).sort_values("importance_score", ascending=False)
        return targets_df

    # ------------------------------------------------------------------ #
    # Visualisation helpers (mirroring OR7a workflow)
    # ------------------------------------------------------------------ #
    def visualize_complete_pathway(self) -> None:
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        self._plot_pathway_diagram(ax1)

        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_neuron_counts(ax2)

        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_connection_strengths(ax3)

        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_category_distribution(ax4)

        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_convergence(ax5)

        ax6 = fig.add_subplot(gs[2, 1:])
        self._plot_synapse_distribution(ax6)

        fig.suptitle(
            f"Complete {self.config.name} Pathway Analysis: ORN → PN → KC → MBON → Behavior",
            fontsize=16,
            fontweight="bold",
        )
        output_path = self.output_dir / f"{self.config.name.lower()}_complete_pathway_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved pathway visualisation to %s", output_path)

    def _plot_pathway_diagram(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

        level_positions = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9}
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]

        for level_num, level in self.levels.items():
            x = level_positions.get(level_num, level_num * 2)
            y = 3
            box = FancyBboxPatch(
                (x - 0.4, y - 0.5),
                0.8,
                1,
                boxstyle="round,pad=0.1",
                facecolor=colors[level_num % len(colors)],
                edgecolor="black",
                linewidth=2,
                alpha=0.7,
            )
            ax.add_patch(box)
            ax.text(
                x,
                y + 0.8,
                level.name.replace("_", "\n"),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                x,
                y,
                f"n={level.neuron_count}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

            if level_num < len(self.levels) - 1 and level_num + 1 in level_positions:
                x_next = level_positions[level_num + 1]
                ax.annotate(
                    "",
                    xy=(x_next - 0.4, y),
                    xytext=(x + 0.4, y),
                    arrowprops=dict(arrowstyle="->", lw=2, color="gray"),
                )
                if level.mean_synapses is not None:
                    mid_x = (x + x_next) / 2
                    ax.text(
                        mid_x,
                        y - 0.3,
                        f"~{level.mean_synapses:.0f} syn",
                        ha="center",
                        va="top",
                        fontsize=8,
                        style="italic",
                    )

        ax.set_title(f"{self.config.name} Circuit Architecture", fontsize=14, fontweight="bold", pad=20)

    def _plot_neuron_counts(self, ax: plt.Axes) -> None:
        levels = []
        counts = []
        names = []
        for level_num in sorted(self.levels.keys()):
            level = self.levels[level_num]
            levels.append(level_num)
            counts.append(max(level.neuron_count, 1))
            names.append(level.name.replace("_", "\n"))

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
        ax.bar(levels, counts, color=[colors[i % len(colors)] for i in levels], alpha=0.7)
        ax.set_xlabel("Circuit Level")
        ax.set_ylabel("Number of Neurons")
        ax.set_title("Neuron Count by Level", fontweight="bold")
        ax.set_xticks(levels)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    def _plot_connection_strengths(self, ax: plt.Axes) -> None:
        if self.complete_pathway is None or self.complete_pathway.empty:
            ax.text(0.5, 0.5, "No connection data", ha="center", va="center")
            return
        conn_summary = self.complete_pathway.groupby("source_level")["syn_count"].agg(["mean", "std"])
        x = conn_summary.index
        ax.bar(x, conn_summary["mean"], yerr=conn_summary["std"], capsize=5, alpha=0.7, color="steelblue")
        ax.set_xlabel("Source Level")
        ax.set_ylabel("Mean Synapses per Connection")
        ax.set_title("Connection Strength by Level", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    def _plot_category_distribution(self, ax: plt.Axes) -> None:
        if self.complete_pathway is None or self.complete_pathway.empty:
            ax.text(0.5, 0.5, "No category data", ha="center", va="center")
            return
        category_counts = (
            self.complete_pathway.groupby("target_category")["post_root_id"].nunique().sort_values(ascending=False).head(8)
        )
        category_counts.plot(kind="barh", ax=ax, color="coral")
        ax.set_xlabel("Number of Neurons")
        ax.set_ylabel("Cell Category")
        ax.set_title("Target Category Distribution", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    def _plot_convergence(self, ax: plt.Axes) -> None:
        levels = []
        ratios = []
        for level_num in sorted(self.levels.keys())[:-1]:
            if level_num in self.levels and level_num + 1 in self.levels:
                source = self.levels[level_num].neuron_count
                target = self.levels[level_num + 1].neuron_count
                if source > 0:
                    ratios.append(target / source)
                    levels.append(f"{level_num}→{level_num+1}")
        colors = ["green" if r > 1 else "red" for r in ratios]
        ax.barh(levels, ratios, color=colors, alpha=0.6)
        ax.axvline(1, color="black", linestyle="--", linewidth=2, label="1:1")
        ax.set_xlabel("Target/Source Ratio")
        ax.set_ylabel("Level Transition")
        ax.set_title("Convergence (red) vs Divergence (green)", fontweight="bold")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

    def _plot_synapse_distribution(self, ax: plt.Axes) -> None:
        if self.complete_pathway is None or self.complete_pathway.empty:
            ax.text(0.5, 0.5, "No synapse data", ha="center", va="center")
            return
        for level_num in sorted(self.complete_pathway["source_level"].unique()):
            level_data = self.complete_pathway[self.complete_pathway["source_level"] == level_num]
            level_name = self.circuit_levels.get(level_num, f"Level {level_num}")
            ax.hist(
                level_data["syn_count"],
                bins=30,
                alpha=0.5,
                label=level_name,
                histtype="step",
                linewidth=2,
            )
        ax.set_xlabel("Synapses per Connection")
        ax.set_ylabel("Count")
        ax.set_title("Synapse Distribution by Level", fontweight="bold")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)

    # ------------------------------------------------------------------ #
    # Export orchestration
    # ------------------------------------------------------------------ #
    def export_results(self, output_dir: Path) -> ORNPathwayResult:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pathway_df = self.build_complete_pathway_dataframe()
        pathway_path = self.output_dir / f"{self.config.name.lower()}_complete_pathway.csv"
        pathway_df.to_csv(pathway_path, index=False)
        logger.info("  Wrote %s", pathway_path)

        summaries = self.generate_pathway_summary()
        for name, df in summaries.items():
            summary_path = self.output_dir / f"pathway_summary_{name}.csv"
            df.to_csv(summary_path, index=False)
            logger.info("  Wrote %s", summary_path)

        targets_df = self.identify_critical_targets()
        targets_path = self.output_dir / "target_priorities.csv"
        targets_df.to_csv(targets_path, index=False)
        logger.info("  Wrote %s", targets_path)

        self.visualize_complete_pathway()

        return ORNPathwayResult(
            config=self.config,
            pathway_df=pathway_df,
            summaries=summaries,
            target_priorities=targets_df,
            levels=self.levels,
            circuit_levels=self.circuit_levels,
        )


# --------------------------------------------------------------------------- #
# Comparative analysis helpers
# --------------------------------------------------------------------------- #


def build_comparative_tables(results: List[ORNPathwayResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    convergence_rows = []
    bottleneck_frames = []
    behavior_rows = []
    kc_rows = []

    for result in results:
        orn = result.config.name
        summaries = result.summaries

        connections = summaries.get("connections", pd.DataFrame()).copy()
        if not connections.empty:
            connections["orn_type"] = orn
            connections["divergence_ratio"] = connections["target_neurons"] / connections["source_neurons"]
            convergence_rows.append(connections)

        bottlenecks = summaries.get("bottlenecks", pd.DataFrame()).copy()
        if not bottlenecks.empty:
            bottlenecks["orn_type"] = orn
            bottleneck_frames.append(bottlenecks)

        by_level = summaries.get("by_level", pd.DataFrame())
        kc_level = by_level[by_level["level_name"] == "Kenyon_Cells"]
        if not kc_level.empty:
            kc_row = kc_level.iloc[0]
            kc_rows.append(
                {
                    "orn_type": orn,
                    "kc_neurons": kc_row["neuron_count"],
                    "kc_convergence_ratio": kc_row["convergence_ratio"],
                    "kc_mean_synapses": kc_row["mean_synapses_per_connection"],
                }
            )

        categories = summaries.get("categories", pd.DataFrame())
        if not categories.empty:
            mbon_categories = categories[categories["source_level"] == 3]
            total = mbon_categories["num_neurons"].sum()
            row = {
                "orn_type": orn,
                "mbon_neurons": result.levels.get(3, CircuitLevel(3, "", [], 0, None, None, None)).neuron_count,
            }
            for _, cat_row in mbon_categories.iterrows():
                share = (cat_row["num_neurons"] / total) if total else 0
                row[f"mbon_share_{str(cat_row['category']).lower()}"] = share
            behavior_rows.append(row)

    if convergence_rows:
        pd.concat(convergence_rows, ignore_index=True).to_csv(
            output_dir / "multi_orn_convergence_comparison.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(output_dir / "multi_orn_convergence_comparison.csv", index=False)

    if bottleneck_frames:
        pd.concat(bottleneck_frames, ignore_index=True).to_csv(
            output_dir / "receptor_specific_bottlenecks.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(output_dir / "receptor_specific_bottlenecks.csv", index=False)

    pd.DataFrame(kc_rows).to_csv(output_dir / "cross_orn_kc_recruitment.csv", index=False)
    pd.DataFrame(behavior_rows).to_csv(output_dir / "behavioral_output_specialization.csv", index=False)


def build_significance_tests(results: List[ORNPathwayResult], output_dir: Path) -> None:
    if stats is None:
        logger.warning("SciPy not installed; skipping significance tests.")
        return

    pathway_levels: Dict[int, Dict[str, np.ndarray]] = {}
    for result in results:
        pathway = result.pathway_df
        if pathway.empty:
            continue
        for level_num, group in pathway.groupby("source_level"):
            pathway_levels.setdefault(level_num, {})[result.config.name] = group["syn_count"].values

    tests: Dict[str, Dict[str, float]] = {}
    for level_num, data in pathway_levels.items():
        samples = [values for values in data.values() if len(values) > 0]
        if len(samples) < 2:
            continue
        stat, pval = stats.kruskal(*samples)
        tests[f"level_{level_num}_synapses"] = {"H": float(stat), "p_value": float(pval)}

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pathway_significance_tests.json", "w", encoding="utf-8") as fh:
        json.dump(tests, fh, indent=2)


def plot_multi_panel(results: List[ORNPathwayResult], output_path: Path) -> None:
    cols = 3
    rows = ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows), sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    for idx, result in enumerate(results):
        ax = axes[idx // cols, idx % cols]
        summary = result.summaries.get("by_level", pd.DataFrame())
        x = summary["level"]
        y = summary["neuron_count"].replace({0: np.nan})
        ax.bar(x, y, color=result.config.color, alpha=0.8)
        ax.set_title(f"{result.config.name} ({result.config.glomerulus})", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["level_name"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Neurons (log scale)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    total_axes = rows * cols
    for idx in range(len(results), total_axes):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle("Multi-ORN pathway comparison", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved multi-panel pathway overview to %s", output_path)


# --------------------------------------------------------------------------- #
# Command-line entry point
# --------------------------------------------------------------------------- #


DEFAULT_DATA_DIR = Path("data/flywire")
DEFAULT_CONNECTIONS = DEFAULT_DATA_DIR / "connections_princeton.csv.gz"
DEFAULT_CELL_TYPES = DEFAULT_DATA_DIR / "consolidated_cell_types.csv.gz"

ORN_CONFIGS: Tuple[ORNPathwayConfig, ...] = (
    ORNPathwayConfig(
        name="Or7a",
        orn_csv=DEFAULT_DATA_DIR / "search_results_or7a.csv",
        glomerulus="DL5",
        neurotransmitter_profile="ACH",
        pn_patterns=("dl5_adpn", "dl5_lpn", "dl5"),
        expected_orn_count=41,
        pn_root_ids=(720575940639080700, 720575940617207200),
        color="#9467bd",
    ),
    ORNPathwayConfig(
        name="Or13a",
        orn_csv=DEFAULT_DATA_DIR / "search_results_Or13a.csv",
        glomerulus="DC2",
        neurotransmitter_profile="ACH",
        pn_patterns=("dc2_adpn", "dc2_lpn", "dc2"),
        expected_orn_count=20,
        color="#1f77b4",
    ),
    ORNPathwayConfig(
        name="Or42b",
        orn_csv=DEFAULT_DATA_DIR / "search_results_Or42b.csv",
        glomerulus="DM1",
        neurotransmitter_profile="mixed ACH/SER",
        pn_patterns=("dm1_adpn", "dm1_lpn", "dm1"),
        expected_orn_count=71,
        color="#ff7f0e",
    ),
    ORNPathwayConfig(
        name="Or47b",
        orn_csv=DEFAULT_DATA_DIR / "search_results_Or47b.csv",
        glomerulus="VA1v",
        neurotransmitter_profile="mixed ACH/SER",
        pn_patterns=("va1v_adpn", "va1v_lpn", "va1v"),
        expected_orn_count=98,
        color="#2ca02c",
    ),
    ORNPathwayConfig(
        name="Or59b",
        orn_csv=DEFAULT_DATA_DIR / "search_results_Or59b.csv",
        glomerulus="DM4",
        neurotransmitter_profile="ACH",
        pn_patterns=("dm4_adpn", "dm4_lpn", "dm4"),
        expected_orn_count=40,
        color="#d62728",
    ),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace ORN→PN→KC→MBON→behavior pathways for multiple receptor types."
    )
    parser.add_argument(
        "--connections",
        type=Path,
        default=DEFAULT_CONNECTIONS,
        help="Path to FlyWire connections CSV (default: data/flywire/connections_princeton.csv.gz)",
    )
    parser.add_argument(
        "--cell-types",
        type=Path,
        default=DEFAULT_CELL_TYPES,
        help="Path to cell type annotations CSV (default: data/flywire/consolidated_cell_types.csv.gz)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Root directory for outputs (default: results/)",
    )
    parser.add_argument(
        "--min-synapses",
        type=int,
        default=3,
        help="Minimum synapse count threshold per connection (default: 3)",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Maximum circuit levels to trace (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    connections = load_connections(args.connections)
    cell_types = load_cell_types(args.cell_types)

    results: List[ORNPathwayResult] = []
    for config in ORN_CONFIGS:
        tracer = ORNPathwayTracer(
            config=config,
            connections=connections,
            cell_types=cell_types,
            min_synapses=args.min_synapses,
            max_levels=args.max_levels,
        )
        tracer.trace()
        output_dir = args.output_root / f"{config.name.lower()}_complete_pathway"
        result = tracer.export_results(output_dir)
        results.append(result)

    comparative_dir = args.output_root / "multi_orn_pathways" / "comparative_analysis"
    build_comparative_tables(results, comparative_dir)
    build_significance_tests(results, comparative_dir)
    plot_multi_panel(results, comparative_dir / "multi_orn_pathway_overview.png")
    logger.info("All comparative outputs saved to %s", comparative_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
