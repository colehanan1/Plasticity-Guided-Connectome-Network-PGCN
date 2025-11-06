"""
Multi-ORN output target mapping and comparative analysis workflow.

This script processes FlyWire FAFB v783 exports for multiple olfactory receptor
neuron (ORN) populations (Or13a, Or42b, Or47b, Or59b). It generates consistent
output-target tables, per-population summary statistics, cross-ORN comparative
analyses, statistical significance tests, and publication-quality visualisations.

Outputs per ORN type:
    - {orn}_output_targets_long.csv  : All ORN→target connections
    - {orn}_output_targets_wide.csv  : Top-N targets per neuron (wide format)
    - {orn}_summary_statistics.csv   : Population-level metrics

Cross-ORN outputs:
    - comparative_orn_analysis.csv           : Combined summary metrics
    - comparative_orn_pairwise_tests.csv     : Pairwise statistical comparisons
    - comparative_orn_significance.json      : Global test statistics
    - orn_synapse_distribution.png           : Synapse distribution visualisation
    - orn_target_celltype_heatmap.png        : Target cell-type heatmap
"""
from __future__ import annotations

import argparse
import json
import logging
import textwrap
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:  # SciPy is part of the project requirements but keep a graceful fallback.
    from scipy import stats
except ImportError:  # pragma: no cover - executed only if SciPy missing.
    stats = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

# Use a consistent visual aesthetic across plots.
sns.set_theme(context="talk", style="whitegrid", palette="deep")

from map_or7a_outputs import OR7aOutputMapper

CELL_TYPE_MAPPING: Dict[str, str] = {
    # Projection neurons - uniglomerular
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
    # Multiglomerular projection neurons
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
    # Local neurons - Type 2 (lateral)
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
    # Local neurons - Type 3 (inner lateral and ventral)
    "il3LN6": "inner lateral 3 local neuron 6",
    "v2LN30": "ventral 2 local neuron 30",
    "v2LN31": "ventral 2 local neuron 31",
    "v2LN36": "ventral 2 local neuron 36",
    "v2LN3A1_b": "ventral 2 local neuron 3A1b",
    "vLN25": "ventral local neuron 25",
    # Other specialized neuron types
    "ALBN1": "antennal lobe broad neuron 1",
    "AL-AST1": "antennal lobe AST1 neuron",
    "ALIN1": "antennal lobe interneuron 1",
    "ALIN7": "antennal lobe interneuron 7",
    "ALON3": "antennal lobe output neuron 3",
    "CSD": "centrifugal sensory dendrite neuron",
    "LN60a": "local neuron 60a",
    "LN60b": "local neuron 60b",
    # Callback neurons
    "CB0683": "callback neuron 0683",
    "CB1266": "callback neuron 1266",
    "CB1676": "callback neuron 1676",
    "CB1988": "callback neuron 1988",
    "CB2161": "callback neuron 2161",
    # ORN types
    "ORN_DC2": "olfactory receptor neuron DC2 type",
    "ORN_DC3": "olfactory receptor neuron DC3 type",
    "ORN_DL5": "olfactory receptor neuron DL5 type",
    "ORN_DM1": "olfactory receptor neuron DM1 type",
    "ORN_DP1l": "olfactory receptor neuron DP1l type",
    "ORN_VA1v": "olfactory receptor neuron VA1v type",
}


def expand_cell_type_name(abbreviation: Optional[str]) -> Optional[str]:
    """
    Convert a cell type abbreviation to its descriptive name.

    Parameters
    ----------
    abbreviation: Optional[str]
        Abbreviated cell type name.

    Returns
    -------
    Optional[str]
        Expanded descriptive name if available; otherwise the original value.
    """
    if abbreviation is None:
        return None
    if not isinstance(abbreviation, str):
        return abbreviation
    return CELL_TYPE_MAPPING.get(abbreviation, abbreviation)


def expand_cell_type_field(value: Union[str, Sequence[str], float, None]) -> Union[str, Sequence[str], float, None]:
    """
    Expand abbreviations within a field that may contain delimited or iterable labels.

    Parameters
    ----------
    value:
        Raw value potentially containing abbreviated cell type names.

    Returns
    -------
    Union[str, Sequence[str], float, None]
        Value with any recognised abbreviations replaced by descriptive names.
    """
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
    """
    Wrap a label onto multiple lines to improve readability in plots.

    Parameters
    ----------
    label: str
        Label to wrap.
    width: int
        Maximum characters per line before wrapping.

    Returns
    -------
    str
        Wrapped label.
    """
    if not isinstance(label, str):
        return label
    if len(label) <= width:
        return label
    return "\n".join(textwrap.wrap(label, width=width))


# Default locations
DEFAULT_DATA_DIR = Path("data/flywire")
DEFAULT_OUTPUT_DIR = Path("results/multi_orn_outputs")
DEFAULT_CONNECTIONS = DEFAULT_DATA_DIR / "connections_princeton.csv.gz"
DEFAULT_CELL_TYPES = DEFAULT_DATA_DIR / "consolidated_cell_types.csv.gz"


@dataclass(frozen=True)
class ORNConfig:
    """Configuration metadata for a single ORN population."""

    name: str
    csv_path: Path
    glomerulus: str
    neurotransmitter_profile: str
    expected_count: Optional[int] = None


# Canonical ORN definitions for this analysis.
ORN_DEFINITIONS: Tuple[ORNConfig, ...] = (
    ORNConfig(
        name="Or7a",
        csv_path=DEFAULT_DATA_DIR / "search_results_or7a.csv",
        glomerulus="DL5",
        neurotransmitter_profile="ACH",
        expected_count=41,
    ),
    ORNConfig(
        name="Or13a",
        csv_path=DEFAULT_DATA_DIR / "search_results_Or13a.csv",
        glomerulus="DC2",
        neurotransmitter_profile="ACH",
        expected_count=20,
    ),
    ORNConfig(
        name="Or42b",
        csv_path=DEFAULT_DATA_DIR / "search_results_Or42b.csv",
        glomerulus="DM1",
        neurotransmitter_profile="mixed ACH/SER",
        expected_count=71,
    ),
    ORNConfig(
        name="Or47b",
        csv_path=DEFAULT_DATA_DIR / "search_results_Or47b.csv",
        glomerulus="VA1v",
        neurotransmitter_profile="mixed ACH/SER",
        expected_count=98,
    ),
    ORNConfig(
        name="Or59b",
        csv_path=DEFAULT_DATA_DIR / "search_results_Or59b.csv",
        glomerulus="DM4",
        neurotransmitter_profile="ACH",
        expected_count=40,
    ),
)


class ORNProcessingError(RuntimeError):
    """Raised when a specific ORN dataset fails processing."""


@dataclass
class ORNProcessingResult:
    """Container for outputs associated with a single ORN population."""

    long_df: pd.DataFrame
    wide_df: pd.DataFrame
    summary_df: pd.DataFrame
    original_long_df: Optional[pd.DataFrame] = None
    original_wide_df: Optional[pd.DataFrame] = None


class MultiORNOutputMapper:
    """End-to-end pipeline for ORN output mapping and comparative analysis."""

    def __init__(
        self,
        orn_configs: Sequence[ORNConfig] = ORN_DEFINITIONS,
        connections_path: Path = DEFAULT_CONNECTIONS,
        cell_types_path: Path = DEFAULT_CELL_TYPES,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        min_synapses: int = 3,
        top_targets: int = 20,
        chunk_size: int = 500_000,
    ) -> None:
        self.orn_configs = tuple(orn_configs)
        self.connections_path = Path(connections_path)
        self.cell_types_path = Path(cell_types_path)
        self.output_dir = Path(output_dir)
        self.min_synapses = int(min_synapses)
        self.top_targets = int(top_targets)
        self.chunk_size = int(chunk_size)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._cell_types_df: Optional[pd.DataFrame] = None
        self._connection_columns: Optional[List[str]] = None
        self._results: Dict[str, ORNProcessingResult] = {}

    # --------------------------------------------------------------------- #
    # Data loading helpers
    # --------------------------------------------------------------------- #
    def _load_cell_types(self) -> pd.DataFrame:
        if self._cell_types_df is not None:
            return self._cell_types_df

        if not self.cell_types_path.exists():
            raise FileNotFoundError(
                f"Cell type annotations not found at {self.cell_types_path}"
            )

        df = pd.read_csv(self.cell_types_path)
        if "root_id" not in df.columns:
            raise ValueError("`consolidated_cell_types` missing `root_id` column")

        df["root_id"] = pd.to_numeric(df["root_id"], errors="coerce").astype("Int64")
        df["primary_type"] = df["primary_type"].fillna("unknown").apply(expand_cell_type_name)
        df["additional_type(s)"] = df["additional_type(s)"].fillna("")

        self._cell_types_df = df
        logger.info("Loaded %s cell type annotations", f"{len(df):,}")
        return df

    def _detect_connection_columns(self) -> List[str]:
        if self._connection_columns is not None:
            return self._connection_columns

        if not self.connections_path.exists():
            raise FileNotFoundError(
                f"Connections table not found at {self.connections_path}"
            )

        header = pd.read_csv(self.connections_path, nrows=0)
        required = {"pre_root_id", "post_root_id", "syn_count"}
        missing = required.difference(header.columns)
        if missing:
            raise ValueError(
                "Connections table missing required columns: "
                f"{', '.join(sorted(missing))}"
            )

        optional = [col for col in ["neuropil", "nt_type"] if col in header.columns]
        self._connection_columns = ["pre_root_id", "post_root_id", "syn_count", *optional]
        logger.info("Using connection columns: %s", self._connection_columns)
        return self._connection_columns

    # --------------------------------------------------------------------- #
    # ORN dataset processing
    # --------------------------------------------------------------------- #
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
            token.strip().strip("'\" ")
            for token in text.split(";")
            if token.strip()
        ]
        return [item for item in items if item]

    def _load_orn_table(self, config: ORNConfig) -> pd.DataFrame:
        if not config.csv_path.exists():
            raise ORNProcessingError(
                f"{config.name}: data file not found at {config.csv_path}"
            )

        df = pd.read_csv(config.csv_path)

        required_cols = {
            "root_id",
            "name",
            "side",
            "input_synapses",
            "output_synapses",
            "nt_type",
        }
        missing = required_cols.difference(df.columns)
        if missing:
            raise ORNProcessingError(
                f"{config.name}: dataset missing required columns {sorted(missing)}"
            )

        # Normalise list-like columns for downstream parsing.
        list_cols = [
            col for col in ("label", "cell_type", "connectivity_tag")
            if col in df.columns
        ]
        for col in list_cols:
            df[col] = df[col].apply(self._parse_semicolon_list)

        df["root_id"] = pd.to_numeric(df["root_id"], errors="raise").astype("int64")
        df["input_synapses"] = pd.to_numeric(df["input_synapses"], errors="raise")
        df["output_synapses"] = pd.to_numeric(df["output_synapses"], errors="raise")
        df["side"] = df["side"].str.lower().str.strip()
        df["side"] = df["side"].fillna("unknown")
        df["nt_type"] = df["nt_type"].fillna("unknown")

        if not set(df["side"].unique()).issubset({"left", "right", "unknown"}):
            raise ORNProcessingError(
                f"{config.name}: `side` column must contain only left/right/unknown values"
            )

        duplicate_ids = df["root_id"].duplicated()
        if duplicate_ids.any():
            example = df.loc[duplicate_ids, "root_id"].tolist()
            raise ORNProcessingError(
                f"{config.name}: duplicate root ids detected {example}"
            )

        if config.expected_count is not None and len(df) != config.expected_count:
            logger.warning(
                "%s: expected %d neurons, found %d",
                config.name,
                config.expected_count,
                len(df),
            )

        return df

    def _load_relevant_connections(self, pre_root_ids: Iterable[int]) -> pd.DataFrame:
        columns = self._detect_connection_columns()
        root_ids = {int(x) for x in pre_root_ids}
        if not root_ids:
            return pd.DataFrame(columns=columns)

        logger.info(
            "Filtering connections for %d ORN root ids (chunk size=%d)",
            len(root_ids),
            self.chunk_size,
        )

        frames: List[pd.DataFrame] = []
        for chunk in pd.read_csv(
            self.connections_path,
            usecols=columns,
            dtype={
                "pre_root_id": "int64",
                "post_root_id": "int64",
                "syn_count": "int32",
            },
            chunksize=self.chunk_size,
        ):
            mask = chunk["pre_root_id"].isin(root_ids)
            if mask.any():
                frames.append(chunk.loc[mask].copy())

        if not frames:
            logger.warning("No connections found for provided ORN root ids.")
            return pd.DataFrame(columns=columns)

        filtered = pd.concat(frames, ignore_index=True)
        logger.info("Retrieved %s matching connections", f"{len(filtered):,}")
        return filtered

    # --------------------------------------------------------------------- #
    # Core processing routines
    # --------------------------------------------------------------------- #
    @staticmethod
    def _gini(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return float("nan")
        if np.all(arr == 0):
            return 0.0
        sorted_arr = np.sort(arr)
        n = sorted_arr.size
        cumulative = np.cumsum(sorted_arr)
        gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
        return float(gini)

    def _build_long_table(
        self,
        orn_df: pd.DataFrame,
        connections: pd.DataFrame,
        config: ORNConfig,
    ) -> pd.DataFrame:
        if connections.empty:
            return pd.DataFrame(
                columns=[
                    "orn_type",
                    "orn_root_id",
                    "orn_name",
                    "orn_side",
                    "orn_nt_type",
                    "orn_glomerulus",
                    "orn_neurotransmitter_profile",
                    "target_root_id",
                    "target_primary_type",
                    "target_additional_types",
                    "neuropil",
                    "syn_count",
                    "syn_fraction_of_neuron",
                    "syn_fraction_of_population",
                ]
            )

        meta = (
            orn_df[["root_id", "name", "side", "nt_type", "output_synapses"]]
            .rename(columns={
                "root_id": "orn_root_id",
                "name": "orn_name",
                "side": "orn_side",
                "nt_type": "orn_nt_type",
                "output_synapses": "orn_output_synapses_reported",
            })
        )

        long_df = connections.rename(
            columns={
                "pre_root_id": "orn_root_id",
                "post_root_id": "target_root_id",
            }
        )

        long_df = long_df.merge(meta, on="orn_root_id", how="left", validate="many_to_one")
        long_df["orn_type"] = config.name
        long_df["orn_glomerulus"] = config.glomerulus
        long_df["orn_neurotransmitter_profile"] = config.neurotransmitter_profile

        # Attach target annotations
        cell_types = self._load_cell_types()
        long_df = long_df.merge(
            cell_types,
            left_on="target_root_id",
            right_on="root_id",
            how="left",
        )
        long_df = long_df.drop(columns=["root_id"])
        long_df = long_df.rename(
            columns={
                "primary_type": "target_primary_type",
                "additional_type(s)": "target_additional_types",
            }
        )
        long_df["target_primary_type"] = (
            long_df["target_primary_type"]
            .fillna("unknown")
            .apply(expand_cell_type_field)
        )
        long_df["target_additional_types"] = (
            long_df["target_additional_types"]
            .fillna("")
            .apply(expand_cell_type_field)
        )

        # Enforce minimum synapse threshold
        long_df = long_df[long_df["syn_count"] >= self.min_synapses].copy()
        if long_df.empty:
            logger.warning(
                "%s: no connections survive min_synapses=%d threshold",
                config.name,
                self.min_synapses,
            )
            return long_df

        long_df["syn_count"] = long_df["syn_count"].astype(int)
        long_df["syn_fraction_of_neuron"] = (
            long_df["syn_count"]
            / long_df.groupby("orn_root_id")["syn_count"].transform("sum")
        )
        population_total = long_df["syn_count"].sum()
        long_df["syn_fraction_of_population"] = long_df["syn_count"] / population_total

        cols_order = [
            "orn_type",
            "orn_root_id",
            "orn_name",
            "orn_side",
            "orn_nt_type",
            "orn_glomerulus",
            "orn_neurotransmitter_profile",
            "orn_output_synapses_reported",
            "target_root_id",
            "target_primary_type",
            "target_additional_types",
            "neuropil",
            "syn_count",
            "syn_fraction_of_neuron",
            "syn_fraction_of_population",
        ]
        optional_cols = [col for col in ["nt_type"] if col in connections.columns]
        cols_order.extend(optional_cols)
        existing_cols = [col for col in cols_order if col in long_df.columns]
        remaining = [col for col in long_df.columns if col not in existing_cols]
        return long_df[existing_cols + remaining]

    def _build_wide_table(self, long_df: pd.DataFrame) -> pd.DataFrame:
        if long_df.empty:
            return pd.DataFrame()

        sorted_df = long_df.sort_values(
            ["orn_root_id", "syn_count", "target_root_id"],
            ascending=[True, False, True],
        )
        sorted_df["rank"] = sorted_df.groupby("orn_root_id").cumcount() + 1
        truncated = sorted_df[sorted_df["rank"] <= self.top_targets].copy()

        pivot_cols = [
            "target_root_id",
            "target_primary_type",
            "neuropil",
            "syn_count",
            "syn_fraction_of_neuron",
        ]
        pivot = (
            truncated
            .set_index(["orn_root_id", "rank"])[pivot_cols]
            .unstack("rank")
        )

        # Flatten MultiIndex columns
        pivot.columns = [
            f"{col}_{int(rank)}"
            for col, rank in pivot.columns.to_flat_index()
        ]

        meta_cols = [
            "orn_type",
            "orn_name",
            "orn_side",
            "orn_nt_type",
            "orn_glomerulus",
            "orn_neurotransmitter_profile",
        ]
        meta = (
            long_df.drop_duplicates("orn_root_id")
            .set_index("orn_root_id")[meta_cols]
        )

        wide_df = meta.join(pivot, how="left").reset_index()
        wide_df = wide_df.rename(columns={"index": "orn_root_id"})

        # Expand any target cell type columns to descriptive names
        for column in wide_df.columns:
            if column.startswith("target_primary_type_"):
                wide_df[column] = wide_df[column].apply(expand_cell_type_field)

        return wide_df

    def _summarise_population(
        self,
        config: ORNConfig,
        orn_df: pd.DataFrame,
        long_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if long_df.empty:
            return pd.DataFrame(
                {
                    "orn_type": [config.name],
                    "n_neurons": [len(orn_df)],
                    "total_synapses_retained": [0],
                    "mean_targets_per_neuron": [float("nan")],
                    "median_targets_per_neuron": [float("nan")],
                    "total_unique_targets": [0],
                    "dominant_target_primary_type": ["NA"],
                    "dominant_target_fraction": [float("nan")],
                    "dominant_neuropil": ["NA"],
                    "dominant_neuropil_fraction": [float("nan")],
                    "hemisphere_left": [(orn_df["side"] == "left").sum()],
                    "hemisphere_right": [(orn_df["side"] == "right").sum()],
                    "hemisphere_unknown": [(orn_df["side"] == "unknown").sum()],
                    "mean_synapses_per_connection": [float("nan")],
                    "median_synapses_per_connection": [float("nan")],
                    "synapse_gini": [float("nan")],
                    "reported_mean_output_synapses": [orn_df["output_synapses"].mean()],
                    "reported_median_output_synapses": [orn_df["output_synapses"].median()],
                }
            )

        per_neuron_targets = (
            long_df.groupby("orn_root_id")["target_root_id"].nunique()
        )
        per_neuron_synapses = long_df.groupby("orn_root_id")["syn_count"].sum()
        neuropil_totals = long_df.groupby("neuropil")["syn_count"].sum().sort_values(ascending=False)
        target_totals = long_df.groupby("target_primary_type")["syn_count"].sum().sort_values(ascending=False)

        dominant_target = target_totals.index[0] if not target_totals.empty else "unknown"
        dominant_target_fraction = (
            target_totals.iloc[0] / long_df["syn_count"].sum()
            if not target_totals.empty else float("nan")
        )
        dominant_neuropil = neuropil_totals.index[0] if not neuropil_totals.empty else "unknown"
        dominant_neuropil_fraction = (
            neuropil_totals.iloc[0] / long_df["syn_count"].sum()
            if not neuropil_totals.empty else float("nan")
        )

        left_count = (orn_df["side"] == "left").sum()
        right_count = (orn_df["side"] == "right").sum()
        unknown_count = len(orn_df) - left_count - right_count
        known_total = max(left_count + right_count, 1)

        summary = pd.DataFrame(
            {
                "orn_type": [config.name],
                "n_neurons": [len(orn_df)],
                "glomerulus": [config.glomerulus],
                "neurotransmitter_profile": [config.neurotransmitter_profile],
                "total_synapses_retained": [long_df["syn_count"].sum()],
                "mean_targets_per_neuron": [per_neuron_targets.mean()],
                "median_targets_per_neuron": [per_neuron_targets.median()],
                "total_unique_targets": [long_df["target_root_id"].nunique()],
                "dominant_target_primary_type": [expand_cell_type_name(dominant_target)],
                "dominant_target_fraction": [dominant_target_fraction],
                "dominant_neuropil": [dominant_neuropil],
                "dominant_neuropil_fraction": [dominant_neuropil_fraction],
                "hemisphere_left": [left_count],
                "hemisphere_right": [right_count],
                "hemisphere_unknown": [unknown_count],
                "hemisphere_asymmetry_index": [
                    (right_count - left_count) / known_total if known_total else float("nan")
                ],
                "mean_synapses_per_connection": [long_df["syn_count"].mean()],
                "median_synapses_per_connection": [long_df["syn_count"].median()],
                "synapse_gini": [self._gini(long_df["syn_count"])],
                "reported_mean_output_synapses": [orn_df["output_synapses"].mean()],
                "reported_median_output_synapses": [orn_df["output_synapses"].median()],
                "mean_synapses_per_neuron_filtered": [per_neuron_synapses.mean()],
                "median_synapses_per_neuron_filtered": [per_neuron_synapses.median()],
            }
        )
        return summary

    def _process_or7a_outputs(self, config: ORNConfig) -> ORNProcessingResult:
        """Use the dedicated OR7a mapper to mirror single-ORN outputs."""
        logger.info("Processing %s via OR7aOutputMapper for compatibility", config.name)
        per_orn_dir = Path("results") / f"{config.name.lower()}_outputs"
        mapper = OR7aOutputMapper(
            or7a_data_path=config.csv_path,
            data_source="local",
            connections_path=self.connections_path,
            cell_types_path=self.cell_types_path,
            output_dir=per_orn_dir,
            min_synapses=self.min_synapses,
            top_n_targets=self.top_targets,
        )

        analysis_outputs = mapper.run_complete_analysis()
        outputs_raw = analysis_outputs["outputs_long"].copy()
        wide_raw = analysis_outputs["outputs_wide"].copy()
        or7a_df = mapper.load_or7a_data()

        # Build standardised long-format table for comparative analysis
        meta = or7a_df.set_index("root_id")
        nt_map = meta["nt_type"].fillna("unknown").to_dict()
        output_syn_map = meta["output_synapses"].to_dict()

        additional_labels: List[str] = []
        for row in outputs_raw.itertuples():
            parts: List[str] = []
            if hasattr(row, "target_super_class"):
                val = getattr(row, "target_super_class")
                if isinstance(val, str) and val:
                    parts.append(val)
            if hasattr(row, "target_sub_class"):
                val = getattr(row, "target_sub_class")
                if isinstance(val, str) and val:
                    parts.append(val)
            additional_labels.append("; ".join(parts))

        long_df = pd.DataFrame(
            {
                "orn_type": config.name,
                "orn_root_id": outputs_raw["or7a_root_id"],
                "orn_name": outputs_raw["or7a_name"],
                "orn_side": outputs_raw["or7a_side"].astype(str).str.lower(),
                "orn_nt_type": outputs_raw["or7a_root_id"].map(nt_map).fillna("unknown"),
                "orn_glomerulus": config.glomerulus,
                "orn_neurotransmitter_profile": config.neurotransmitter_profile,
                "orn_output_synapses_reported": outputs_raw["or7a_root_id"].map(output_syn_map).fillna(0),
                "target_root_id": outputs_raw["target_root_id"],
                "target_primary_type": outputs_raw["target_cell_type"],
                "target_additional_types": additional_labels,
                "neuropil": outputs_raw["target_neuropil"],
                "syn_count": outputs_raw["synapse_count"],
            }
        )

        if "nt_type" in outputs_raw.columns:
            long_df["nt_type"] = outputs_raw["nt_type"]

        long_df["target_primary_type"] = long_df["target_primary_type"].apply(expand_cell_type_field)
        long_df["target_additional_types"] = long_df["target_additional_types"].apply(expand_cell_type_field)

        neuron_totals = long_df.groupby("orn_root_id")["syn_count"].transform("sum")
        neuron_totals = neuron_totals.replace(0, np.nan)
        long_df["syn_fraction_of_neuron"] = long_df["syn_count"] / neuron_totals
        long_df["syn_fraction_of_neuron"] = long_df["syn_fraction_of_neuron"].fillna(0.0)

        total_synapses = long_df["syn_count"].sum()
        if total_synapses > 0:
            long_df["syn_fraction_of_population"] = long_df["syn_count"] / total_synapses
        else:
            long_df["syn_fraction_of_population"] = 0.0

        long_df = long_df.sort_values(["orn_root_id", "syn_count"], ascending=[True, False]).reset_index(drop=True)

        summary_df = self._summarise_population(config, or7a_df, long_df)
        wide_df = self._build_wide_table(long_df)

        return ORNProcessingResult(
            long_df=long_df,
            wide_df=wide_df,
            summary_df=summary_df,
            original_long_df=outputs_raw,
            original_wide_df=wide_raw,
        )

    def _visualize_orn_outputs(self, config: ORNConfig, long_df: pd.DataFrame) -> None:
        """
        Create an OR7a-style analysis figure for a generic ORN population.

        Parameters
        ----------
        config: ORNConfig
            Metadata for the ORN population.
        long_df: pd.DataFrame
            Long-format dataframe containing connection information.
        """
        if long_df.empty:
            logger.info("%s: skipping visualization (no connections)", config.name)
            return

        per_orn_dir = Path("results") / f"{config.name.lower()}_outputs"
        per_orn_dir.mkdir(parents=True, exist_ok=True)

        outputs = long_df.copy()
        outputs["target_cell_type"] = outputs["target_primary_type"].fillna("Unknown")
        outputs["target_neuropil"] = outputs["neuropil"].fillna("Unknown")
        outputs["synapse_count"] = outputs["syn_count"]
        outputs["orn_id"] = outputs["orn_root_id"]
        outputs["orn_side"] = outputs["orn_side"].fillna("unknown")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Target cell type distribution
        ax1 = fig.add_subplot(gs[0, :2])
        type_counts = (
            outputs.groupby("target_cell_type")["synapse_count"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
        )
        type_counts.index = [wrap_label(str(idx)) for idx in type_counts.index]
        type_counts.plot(kind="barh", ax=ax1, color="steelblue")
        ax1.set_title(f"Top 15 Target Cell Types ({config.name})", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Total Synapses")
        ax1.set_ylabel("Cell Type")

        # 2. Neuropil distribution (pie)
        ax2 = fig.add_subplot(gs[0, 2])
        neuropil_counts = (
            outputs.groupby("target_neuropil")["synapse_count"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        ax2.pie(
            neuropil_counts.values,
            labels=[wrap_label(str(lbl), width=18) for lbl in neuropil_counts.index],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title("Output Neuropil Distribution", fontsize=12, fontweight="bold")

        # 3. Targets per neuron histogram
        ax3 = fig.add_subplot(gs[1, 0])
        targets_per_orn = outputs.groupby("orn_id")["target_root_id"].count()
        ax3.hist(targets_per_orn, bins=20, color="coral", edgecolor="black")
        ax3.set_title("Number of Targets per Neuron", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Number of Downstream Targets")
        ax3.set_ylabel("Neuron Count")
        if len(targets_per_orn) > 0:
            mean_targets = targets_per_orn.mean()
            ax3.axvline(mean_targets, color="red", linestyle="--", label=f"Mean: {mean_targets:.1f}")
            ax3.legend()

        # 4. Synapse count distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(outputs["synapse_count"], bins=30, color="mediumpurple", edgecolor="black")
        ax4.set_title("Synapse Count Distribution", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Synapses per Connection")
        ax4.set_ylabel("Count")
        ax4.set_yscale("log")

        # 5. Hemispheric comparison
        ax5 = fig.add_subplot(gs[1, 2])
        hemis_data = (
            outputs.groupby(["orn_side", "orn_id"])["target_root_id"]
            .count()
            .reset_index()
            .rename(columns={"target_root_id": "num_targets", "orn_side": "side"})
        )
        if not hemis_data.empty:
            sns.boxplot(data=hemis_data, x="side", y="num_targets", ax=ax5, palette="Set2")
            sns.swarmplot(data=hemis_data, x="side", y="num_targets", ax=ax5, color="black", alpha=0.5, size=4)
        ax5.set_title("Hemispheric Comparison", fontsize=12, fontweight="bold")
        ax5.set_xlabel("Hemisphere")
        ax5.set_ylabel("Number of Targets per Neuron")

        # 6. Top individual targets
        ax6 = fig.add_subplot(gs[2, :])
        top_targets = (
            outputs.groupby(["target_root_id", "target_cell_type"])["synapse_count"]
            .sum()
            .sort_values(ascending=False)
            .head(20)
        )
        labels = [wrap_label(f"{cell_type}\n({root_id})", width=28) for root_id, cell_type in top_targets.index]
        ax6.barh(range(len(top_targets)), top_targets.values, color="teal")
        ax6.set_yticks(range(len(top_targets)))
        ax6.set_yticklabels(labels, fontsize=8)
        ax6.set_title("Top 20 Individual Target Neurons", fontsize=14, fontweight="bold")
        ax6.set_xlabel(f"Total Synapses from All {config.name} Neurons")
        ax6.invert_yaxis()

        plt.suptitle(f"{config.name} Output Target Analysis", fontsize=16, fontweight="bold", y=0.995)

        output_path = per_orn_dir / f"{config.name.lower()}_output_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("%s: wrote %s", config.name, output_path)
        plt.close(fig)

    def _process_single_orn(self, config: ORNConfig) -> ORNProcessingResult:
        if config.name == "Or7a":
            try:
                return self._process_or7a_outputs(config)
            except Exception as exc:  # pragma: no cover - fallback handling
                logger.exception("Falling back to generic processing for Or7a due to error: %s", exc)
        logger.info("Processing %s (glomerulus %s)", config.name, config.glomerulus)
        orn_df = self._load_orn_table(config)
        connections = self._load_relevant_connections(orn_df["root_id"])
        long_df = self._build_long_table(orn_df, connections, config)
        wide_df = self._build_wide_table(long_df)
        summary_df = self._summarise_population(config, orn_df, long_df)
        self._visualize_orn_outputs(config, long_df)
        return ORNProcessingResult(long_df=long_df, wide_df=wide_df, summary_df=summary_df)

    # --------------------------------------------------------------------- #
    # Statistical analysis
    # --------------------------------------------------------------------- #
    def _compute_global_tests(self, combined_long: pd.DataFrame) -> Dict[str, float]:
        results: Dict[str, float] = {}
        if combined_long.empty or stats is None:
            return results

        grouped = {
            orn: df["syn_count"].values
            for orn, df in combined_long.groupby("orn_type")
        }

        if len(grouped) >= 2:
            samples = [vals for vals in grouped.values() if len(vals) > 0]
            if all(len(vals) > 0 for vals in samples):
                h_stat, p_value = stats.kruskal(*samples)
                results["kruskal_syn_count_H"] = float(h_stat)
                results["kruskal_syn_count_p"] = float(p_value)

        # Hemisphere analysis using reported neuron counts
        hemisphere_counts = combined_long[["orn_type", "orn_root_id", "orn_side"]].drop_duplicates()
        contingency = (
            hemisphere_counts
            .pivot_table(index="orn_side", columns="orn_type", aggfunc="size", fill_value=0)
        )
        if stats is not None and contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            results["chi2_hemisphere"] = float(chi2)
            results["chi2_hemisphere_p"] = float(p)

        return results

    def _compute_pairwise_tests(
        self,
        orn_summaries: pd.DataFrame,
        combined_long: pd.DataFrame,
    ) -> pd.DataFrame:
        if stats is None:
            return pd.DataFrame(
                columns=[
                    "orn_a",
                    "orn_b",
                    "metric",
                    "statistic",
                    "p_value",
                    "effect_size",
                ]
            )

        rows = []
        # Use per-neuron total synapses for robust comparison
        per_neuron_totals = combined_long.groupby(
            ["orn_type", "orn_root_id"]
        )["syn_count"].sum().reset_index()

        for orn_a, orn_b in combinations(per_neuron_totals["orn_type"].unique(), 2):
            data_a = per_neuron_totals.loc[per_neuron_totals["orn_type"] == orn_a, "syn_count"]
            data_b = per_neuron_totals.loc[per_neuron_totals["orn_type"] == orn_b, "syn_count"]
            if len(data_a) < 2 or len(data_b) < 2:
                continue
            stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative="two-sided")
            effect = self._cohens_d(data_a.values, data_b.values)
            rows.append(
                {
                    "orn_a": orn_a,
                    "orn_b": orn_b,
                    "metric": "per_neuron_synapses_filtered",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "effect_size_cohens_d": effect,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return float("nan")
        mean_diff = float(x.mean() - y.mean())
        pooled_var = (
            ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
            / (nx + ny - 2)
        )
        if pooled_var <= 0:
            return float("nan")
        return mean_diff / np.sqrt(pooled_var)

    # --------------------------------------------------------------------- #
    # Visualisation helpers
    # --------------------------------------------------------------------- #
    def _plot_synapse_distribution(self, combined_long: pd.DataFrame, output_dir: Path) -> None:
        if combined_long.empty:
            return
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(
            data=combined_long,
            x="orn_type",
            y="syn_count",
            inner="quartile",
            cut=0,
        )
        ax.set_title("Synapse distribution per ORN target connection")
        ax.set_xlabel("ORN type")
        ax.set_ylabel("Synapse count per connection")
        plt.tight_layout()
        path = output_dir / "orn_synapse_distribution.png"
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info("Saved synapse distribution plot to %s", path)

    def _plot_celltype_heatmap(self, combined_long: pd.DataFrame, output_dir: Path) -> None:
        if combined_long.empty:
            return
        top_targets = (
            combined_long.groupby(["orn_type", "target_primary_type"])["syn_count"]
            .sum()
            .reset_index()
        )
        pivot = top_targets.pivot_table(
            index="target_primary_type",
            columns="orn_type",
            values="syn_count",
            aggfunc="sum",
            fill_value=0,
        )
        # Focus on top 20 target cell types across all ORNs
        pivot["total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("total", ascending=False).head(20)
        pivot = pivot.drop(columns="total")

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            pivot,
            annot=False,
            cmap="viridis",
            linewidths=0.5,
            cbar_kws={"label": "Synapse count"},
        )
        plt.title("Top target cell types across ORN populations")
        plt.xlabel("ORN type")
        plt.ylabel("Target primary cell type")
        ax = plt.gca()
        ax.set_yticklabels([wrap_label(label) for label in pivot.index], rotation=0)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = output_dir / "orn_target_celltype_heatmap.png"
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info("Saved target cell-type heatmap to %s", path)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def run(self) -> None:
        for config in self.orn_configs:
            try:
                result = self._process_single_orn(config)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to process %s: %s", config.name, exc)
                continue

            self._results[config.name] = result
            self._persist_orn_outputs(config, result)

        if not self._results:
            logger.error("No ORN datasets were processed successfully.")
            return

        self._finalise_comparative_outputs()

    # --------------------------------------------------------------------- #
    def _persist_orn_outputs(self, config: ORNConfig, result: ORNProcessingResult) -> None:
        prefix = config.name.lower()
        long_path = self.output_dir / f"{prefix}_output_targets_long.csv"
        wide_path = self.output_dir / f"{prefix}_output_targets_wide.csv"
        summary_path = self.output_dir / f"{prefix}_summary_statistics.csv"

        long_to_write = (
            result.original_long_df
            if config.name == "Or7a" and result.original_long_df is not None
            else result.long_df
        )
        wide_to_write = (
            result.original_wide_df
            if config.name == "Or7a" and result.original_wide_df is not None
            else result.wide_df
        )

        long_to_write.to_csv(long_path, index=False)
        logger.info("%s: wrote %s", config.name, long_path)

        wide_to_write.to_csv(wide_path, index=False)
        logger.info("%s: wrote %s", config.name, wide_path)

        result.summary_df.to_csv(summary_path, index=False)
        logger.info("%s: wrote %s", config.name, summary_path)

    def _finalise_comparative_outputs(self) -> None:
        combined_long = pd.concat(
            [res.long_df for res in self._results.values()],
            ignore_index=True,
        )
        combined_summary = pd.concat(
            [res.summary_df for res in self._results.values()],
            ignore_index=True,
        ).sort_values("orn_type")

        analysis_path = self.output_dir / "comparative_orn_analysis.csv"
        combined_summary.to_csv(analysis_path, index=False)
        logger.info("Wrote comparative summary to %s", analysis_path)

        tests_df = self._compute_pairwise_tests(combined_summary, combined_long)
        pairwise_path = self.output_dir / "comparative_orn_pairwise_tests.csv"
        tests_df.to_csv(pairwise_path, index=False)
        logger.info("Wrote pairwise statistical comparisons to %s", pairwise_path)

        global_tests = self._compute_global_tests(combined_long)
        global_path = self.output_dir / "comparative_orn_significance.json"
        with open(global_path, "w", encoding="utf-8") as fh:
            json.dump(global_tests, fh, indent=2)
        logger.info("Wrote global statistical tests to %s", global_path)

        self._plot_synapse_distribution(combined_long, self.output_dir)
        self._plot_celltype_heatmap(combined_long, self.output_dir)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map FlyWire ORN outputs across multiple receptor types."
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
        help="Path to consolidated cell types CSV (default: data/flywire/consolidated_cell_types.csv.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV outputs and figures (default: results/multi_orn_outputs)",
    )
    parser.add_argument(
        "--min-synapses",
        type=int,
        default=3,
        help="Minimum synapse count threshold per connection (default: 3)",
    )
    parser.add_argument(
        "--top-targets",
        type=int,
        default=20,
        help="Number of top targets per neuron in wide tables (default: 20)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Chunk size for streaming the connections table (default: 500000)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(args)


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = parse_args(cli_args)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    mapper = MultiORNOutputMapper(
        connections_path=args.connections,
        cell_types_path=args.cell_types,
        output_dir=args.output_dir,
        min_synapses=args.min_synapses,
        top_targets=args.top_targets,
        chunk_size=args.chunk_size,
    )
    mapper.run()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
