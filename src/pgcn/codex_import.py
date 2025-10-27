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
    "PN": (
        r"\bPN\b",
        r"PROJECTION\s*NEURON",
        r"\bALPN\b",
        r"ANTENNAL\s*LOBE",
    ),
    "KC": (r"\bKC\b", r"KENYON", r"KENYON[_\s-]*CELL"),
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
        cand = candidate.lower()
        if cand in lowered:
            return lowered[cand]
        # Many Codex exports suffix identifiers with server prefixes (e.g. ``pre_root_id_720575940``)
        for column_lower, original in lowered.items():
            if column_lower.startswith(f"{cand}_"):
                return original
    return None


def _normalise_root_ids(
    column_name: str, series: pd.Series, reference_ids: pd.Series
) -> pd.Series:
    """Ensure synapse root IDs line up with neuron root IDs."""

    def _to_int(value: object) -> Optional[int]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if pd.isna(value):
            return None
        return int(value)

    ref_set = set(reference_ids.dropna().astype("int64").tolist())
    match = re.search(r"_(\d+)$", column_name)
    prefix = match.group(1) if match else None
    suffix_width = None
    if prefix and ref_set:
        target_width = max(len(str(value)) for value in ref_set)
        suffix_width = max(target_width - len(prefix), 0)

    def _coerce(value: object) -> object:
        numeric = _to_int(value)
        if numeric is None:
            return pd.NA
        if numeric in ref_set:
            return numeric
        if prefix is not None:
            suffix = str(numeric)
            if suffix_width:
                suffix = suffix.zfill(suffix_width)
            candidate = int(prefix + suffix)
            if candidate in ref_set:
                return candidate
        return numeric

    coerced = series.map(_coerce)
    return coerced.astype("Int64")


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
    type_candidates = [
        ("type",),
        ("cell_type", "celltype"),
        ("primary_type", "primary type"),
        ("additional_type(s)", "additional types"),
        ("class",),
        ("soma_type",),
        ("group",),
    ]
    type_columns: list[str] = []
    for candidate_group in type_candidates:
        inferred = _infer_column(neurons.columns, candidate_group)
        if inferred and inferred not in type_columns:
            type_columns.append(inferred)

    name_col = _infer_column(neurons.columns, ["name", "cell_body", "pt_position", "cell_id"])
    if node_id_col is None or not type_columns:
        raise ValueError(
            "Neuron table must contain identifier and cell type columns. "
            "If your export uses different names supply Parquet/CSV with matching headings."
        )

    type_strings = (
        neurons[type_columns]
        .fillna("")
        .astype(str)
        .apply(
            lambda row: " ".join(
                value
                for value in (str(item).strip() for item in row)
                if value and value.lower() not in {"nan", "none", "nat"}
            ),
            axis=1,
        )
    )

    selected_columns = [node_id_col] + type_columns + ([name_col] if name_col else [])

    glomerulus_col = _infer_column(neurons.columns, ["glomerulus"])
    if glomerulus_col is None:
        for column in neurons.columns:
            lower = column.lower()
            if "glom" in lower:
                glomerulus_col = column
                break
    if glomerulus_col and glomerulus_col not in selected_columns:
        selected_columns.append(glomerulus_col)

    neurons = neurons[selected_columns].copy()
    neurons[node_id_col] = pd.to_numeric(neurons[node_id_col], errors="coerce")
    neurons = neurons.dropna(subset=[node_id_col])
    neurons[node_id_col] = neurons[node_id_col].astype("int64")
    neurons["type"] = _normalise_types(type_strings.loc[neurons.index], config)
    neurons = neurons.dropna(subset=["type"])
    neurons = neurons[neurons["type"].isin(CANONICAL_TYPES)]

    base_columns = [node_id_col, "type"] + ([name_col] if name_col else [])
    if glomerulus_col:
        base_columns.append(glomerulus_col)
    neurons = neurons[base_columns]

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

    neuron_lookup = neurons[[node_id_col, "type"]].drop_duplicates()
    neuron_lookup = neuron_lookup.rename(columns={node_id_col: "node_id"})
    neuron_lookup["node_id"] = neuron_lookup["node_id"].astype("int64")

    synapses = synapses[[syn_pre_col, syn_post_col, weight_col]].copy()
    neuron_ids = neuron_lookup["node_id"]
    synapses[syn_pre_col] = _normalise_root_ids(syn_pre_col, synapses[syn_pre_col], neuron_ids)
    synapses[syn_post_col] = _normalise_root_ids(
        syn_post_col, synapses[syn_post_col], neuron_ids
    )
    synapses = synapses.rename(
        columns={syn_pre_col: "source_id", syn_post_col: "target_id", weight_col: "synapse_weight"}
    )
    synapses["source_id"] = synapses["source_id"].astype("int64", errors="ignore")
    synapses["target_id"] = synapses["target_id"].astype("int64", errors="ignore")

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

    rename_map = {node_id_col: "node_id"}
    if name_col:
        rename_map[name_col] = "name"
    if glomerulus_col:
        rename_map[glomerulus_col] = "glomerulus"

    nodes = neurons.rename(columns=rename_map)
    keep_columns = ["node_id", "type"]
    if name_col:
        keep_columns.append("name")
    if glomerulus_col:
        keep_columns.append("glomerulus")
    nodes = nodes[keep_columns].drop_duplicates()
    if "glomerulus" not in nodes.columns:
        nodes["glomerulus"] = pd.Series([pd.NA] * len(nodes), dtype="object")

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

