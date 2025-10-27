"""Convert FlyWire Codex exports into a local connectome cache."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd

from .connectome_pipeline import CACHE_FILENAMES, CacheArtifacts


DEFAULT_TYPE_PATTERNS: Mapping[str, Sequence[str]] = {
    "PN": (r"\bPN\b", r"PROJECTION\s*NEURON"),
    "KC": (r"\bKC\b", r"KENYON"),
    "MBON": (r"MBON",),
    "DAN": (r"\bDAN\b", r"DOPAMINE"),
}

CANONICAL_TYPES = tuple(DEFAULT_TYPE_PATTERNS.keys())


@dataclass(slots=True)
class CodexImportConfig:
    """Configuration for interpreting Codex neuron metadata."""

    type_patterns: MutableMapping[str, List[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_TYPE_PATTERNS.items()}
    )

    def add_pattern(self, neuron_type: str, pattern: str) -> None:
        neuron_type = neuron_type.upper()
        if neuron_type not in self.type_patterns:
            raise ValueError(
                f"Unsupported neuron type '{neuron_type}'. "
                f"Expected one of {', '.join(CANONICAL_TYPES)}."
            )
        self.type_patterns[neuron_type].append(pattern)


def _read_table(path: Path) -> pd.DataFrame:
    """Load Codex exports regardless of delimiter/compression quirks."""

    lowered = path.name.lower()
    if lowered.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.zip")):
        return pd.read_csv(path)
    if lowered.endswith((".tsv", ".tsv.gz", ".tsv.bz2", ".tsv.zip")):
        return pd.read_csv(path, sep="\t")
    if lowered.endswith((".parquet", ".parquet.gz")):
        return pd.read_parquet(path)
    if lowered.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(
        f"Unsupported table format for '{path}'. Use CSV/TSV (optionally gz/bz2/zip), Parquet, or JSON."
    )


def _infer_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _normalise_types(series: pd.Series, config: CodexImportConfig) -> pd.Series:
    compiled = {
        neuron_type: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
        for neuron_type, patterns in config.type_patterns.items()
    }

    def classify(value: object) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        for neuron_type, patterns in compiled.items():
            if any(pattern.search(text) for pattern in patterns):
                return neuron_type
        return None

    return series.map(classify)


def build_codex_cache(
    neuron_table: Path,
    synapse_table: Path,
    output_dir: Path,
    config: Optional[CodexImportConfig] = None,
) -> CacheArtifacts:
    """Convert Codex exports into the on-disk cache layout used by PGCN."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or CodexImportConfig()

    neurons = _read_table(neuron_table)
    synapses = _read_table(synapse_table)

    node_id_col = _infer_column(neurons.columns, ["pt_root_id", "root_id", "id", "node_id"])
    type_col = _infer_column(
        neurons.columns,
        ["type", "cell_type", "celltype", "class", "soma_type", "group"],
    )
    name_col = _infer_column(neurons.columns, ["name", "cell_body", "pt_position", "cell_id"])
    if node_id_col is None or type_col is None:
        raise ValueError(
            "Neuron table must contain identifier and cell type columns. "
            "If your export uses different names supply Parquet/CSV with matching headings."
        )

    neurons = neurons[[node_id_col, type_col] + ([name_col] if name_col else [])].copy()
    neurons[node_id_col] = neurons[node_id_col].astype("int64", errors="ignore")
    neurons[type_col] = _normalise_types(neurons[type_col], config)
    neurons = neurons.dropna(subset=[type_col])
    neurons = neurons[neurons[type_col].isin(CANONICAL_TYPES)]

    if neurons.empty:
        raise ValueError(
            "No PN/KC/MBON/DAN neurons detected in the provided Codex export. "
            "Consider adding explicit --pn-pattern/--kc-pattern overrides."
        )

    syn_pre_col = _infer_column(
        synapses.columns,
        ["pre_pt_root_id", "pre_root_id", "pre_id", "source", "source_id"],
    )
    syn_post_col = _infer_column(
        synapses.columns,
        ["post_pt_root_id", "post_root_id", "post_id", "target", "target_id"],
    )
    weight_col = _infer_column(
        synapses.columns,
        ["weight", "synapse_weight", "synapse_count", "count", "size"],
    )
    if syn_pre_col is None or syn_post_col is None or weight_col is None:
        raise ValueError(
            "Synapse table missing source/target/weight columns. Provide Codex synapses.csv export."
        )

    neuron_lookup = neurons[[node_id_col, type_col]].drop_duplicates()
    neuron_lookup = neuron_lookup.rename(columns={node_id_col: "node_id", type_col: "type"})
    neuron_lookup["node_id"] = neuron_lookup["node_id"].astype("int64")

    synapses = synapses[[syn_pre_col, syn_post_col, weight_col]].copy()
    synapses[syn_pre_col] = synapses[syn_pre_col].astype("int64", errors="ignore")
    synapses[syn_post_col] = synapses[syn_post_col].astype("int64", errors="ignore")
    synapses = synapses.rename(
        columns={syn_pre_col: "source_id", syn_post_col: "target_id", weight_col: "synapse_weight"}
    )

    synapses = synapses.merge(
        neuron_lookup.rename(columns={"node_id": "source_id", "type": "source_type"}),
        on="source_id",
        how="inner",
    ).merge(
        neuron_lookup.rename(columns={"node_id": "target_id", "type": "target_type"}),
        on="target_id",
        how="inner",
    )

    pn_kc_mask = (synapses["source_type"] == "PN") & (synapses["target_type"] == "KC")
    kc_mbon_mask = (synapses["source_type"] == "KC") & (synapses["target_type"] == "MBON")
    dan_mask = synapses["source_type"] == "DAN"

    pn_kc_edges = synapses[pn_kc_mask]
    if pn_kc_edges.empty:
        raise ValueError(
            "Codex export does not contain PN→KC synapses. Verify your selection criteria and patterns."
        )

    edges = pd.concat([
        pn_kc_edges.assign(edge_type="PN_KC"),
        synapses[kc_mbon_mask].assign(edge_type="KC_MBON"),
    ])[["source_id", "target_id", "synapse_weight", "edge_type"]]
    dan_edges = synapses[dan_mask][["source_id", "target_id", "synapse_weight"]]

    node_columns = {node_id_col: "node_id", type_col: "type"}
    if name_col:
        node_columns[name_col] = "name"
    nodes = neurons.rename(columns=node_columns)[list(node_columns.values())].drop_duplicates()

    nodes_path = output_dir / CACHE_FILENAMES["nodes"]
    edges_path = output_dir / CACHE_FILENAMES["edges"]
    dan_path = output_dir / CACHE_FILENAMES["dan_edges"]
    meta_path = output_dir / CACHE_FILENAMES["meta"]

    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    dan_edges.to_parquet(dan_path, index=False)

    meta = {
        "source": "codex_snapshot_783",
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "pn_kc_edges": len(pn_kc_edges),
            "dan_edges": len(dan_edges),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return CacheArtifacts(nodes=nodes_path, edges=edges_path, dan_edges=dan_path, meta=meta_path)


def cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FlyWire Codex exports (Cell Types + Connections/Synapses) into a PGCN cache."
        )
    )
    parser.add_argument(
        "--neurons",
        type=Path,
        required=True,
        help=(
            "Path to the Codex 'Cell Types' export (CSV/TSV/Parquet). Include proofread names if available."
        ),
    )
    parser.add_argument(
        "--synapses",
        type=Path,
        required=True,
        help=(
            "Path to the Codex 'Connections (Filtered)' or 'Synapse Table' export (CSV/TSV/Parquet)."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("data") / "cache", help="Output directory for cache files.")
    for neuron_type in CANONICAL_TYPES:
        parser.add_argument(
            f"--{neuron_type.lower()}-pattern",
            action="append",
            dest=f"{neuron_type.lower()}_patterns",
            default=[],
            help=f"Additional regex pattern to classify {neuron_type} neurons.",
        )
    args = parser.parse_args(argv)

    config = CodexImportConfig()
    for neuron_type in CANONICAL_TYPES:
        for pattern in getattr(args, f"{neuron_type.lower()}_patterns"):
            config.add_pattern(neuron_type, pattern)

    build_codex_cache(args.neurons, args.synapses, args.out, config=config)
    return 0


__all__ = ["CodexImportConfig", "build_codex_cache", "cli"]

