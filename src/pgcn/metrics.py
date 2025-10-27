"""Structural metrics for the Plasticity-Guided Connectome Network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
LOGGER = logging.getLogger(__name__)


def jaccard_kc_overlap(e_pn_kc: pd.DataFrame, pn_nodes: pd.DataFrame) -> pd.DataFrame:
    """Compute Jaccard overlap of Kenyon cell targets per glomerulus."""

    required_edge_cols = {"source_id", "target_id"}
    required_node_cols = {"node_id"}
    missing_edges = required_edge_cols - set(e_pn_kc.columns)
    missing_nodes = required_node_cols - set(pn_nodes.columns)
    if missing_edges:
        raise ValueError(f"PN→KC edge table missing columns: {sorted(missing_edges)}")
    if missing_nodes:
        raise ValueError(f"PN node table missing columns: {sorted(missing_nodes)}")

    if "glomerulus" not in pn_nodes.columns:
        LOGGER.warning(
            "PN node table missing 'glomerulus'. Emitting empty overlap table. "
            "Rebuild the cache with Codex annotations or rerun pgcn-cache once API access is available."
        )
        pn_nodes = pn_nodes.copy()
        pn_nodes["glomerulus"] = pd.NA

    pn_nodes = pn_nodes.dropna(subset=["glomerulus"]).copy()
    pn_nodes["glomerulus"] = pn_nodes["glomerulus"].astype(str)
    pn_to_glom = pn_nodes.set_index("node_id")["glomerulus"].to_dict()
    glom_to_kcs: dict[str, set[int]] = {}
    for pn_id, kc_id in e_pn_kc[["source_id", "target_id"]].itertuples(index=False):
        glom = pn_to_glom.get(int(pn_id))
        if not glom:
            continue
        glom_to_kcs.setdefault(glom, set()).add(int(kc_id))

    records: list[dict[str, object]] = []
    gloms = sorted(glom_to_kcs.keys())
    for i, g_a in enumerate(gloms):
        kc_a = glom_to_kcs[g_a]
        for g_b in gloms[i:]:
            kc_b = glom_to_kcs[g_b]
            intersection = kc_a.intersection(kc_b)
            union = kc_a.union(kc_b)
            jaccard = len(intersection) / len(union) if union else 0.0
            records.append(
                {
                    "glomerulus_a": g_a,
                    "glomerulus_b": g_b,
                    "kc_count_a": len(kc_a),
                    "kc_count_b": len(kc_b),
                    "intersection": len(intersection),
                    "union": len(union),
                    "jaccard": float(jaccard),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "glomerulus_a",
                "glomerulus_b",
                "kc_count_a",
                "kc_count_b",
                "intersection",
                "union",
                "jaccard",
            ]
        )
    return pd.DataFrame.from_records(records)


def path_lengths_pn_kc_mbon(
    edges: pd.DataFrame,
    nodes: Optional[pd.DataFrame] = None,
    pn_ids: Optional[Iterable[int]] = None,
    kc_ids: Optional[Iterable[int]] = None,
    mbon_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Summarise PN→KC→MBON two-hop paths."""

    for column in ["source_id", "target_id", "synapse_weight"]:
        if column not in edges.columns:
            raise ValueError(f"Edge table missing required column '{column}'")

    type_lookup: dict[int, str] = {}
    if nodes is not None and {"node_id", "type"}.issubset(nodes.columns):
        type_lookup = nodes.set_index("node_id")["type"].to_dict()

    def _resolve_ids(candidate_ids: Optional[Iterable[int]], type_name: str) -> set[int]:
        if candidate_ids is not None:
            return {int(i) for i in candidate_ids}
        if type_lookup:
            return {int(idx) for idx, label in type_lookup.items() if label == type_name}
        raise ValueError(
            f"Unable to infer {type_name} identifiers. Provide nodes DataFrame or explicit id list."
        )

    pn_set = _resolve_ids(pn_ids, "PN")
    kc_set = _resolve_ids(kc_ids, "KC")
    mbon_set = _resolve_ids(mbon_ids, "MBON")

    edges = edges.copy()
    edges["source_id"] = edges["source_id"].astype(np.int64)
    edges["target_id"] = edges["target_id"].astype(np.int64)

    pn_kc = edges[edges["source_id"].isin(pn_set) & edges["target_id"].isin(kc_set)]
    kc_mbon = edges[edges["source_id"].isin(kc_set) & edges["target_id"].isin(mbon_set)]

    if pn_kc.empty or kc_mbon.empty:
        return pd.DataFrame(columns=["pn_id", "mbon_id", "kc_count", "total_path_weight", "max_path_weight"])

    merged = pn_kc.merge(
        kc_mbon,
        left_on="target_id",
        right_on="source_id",
        suffixes=("_pn_kc", "_kc_mbon"),
    )
    merged["pn_id"] = merged["source_id_pn_kc"]
    merged["kc_id"] = merged["target_id_pn_kc"]
    merged["mbon_id"] = merged["target_id_kc_mbon"]
    merged["path_weight"] = merged["synapse_weight_pn_kc"] * merged["synapse_weight_kc_mbon"]

    summary = (
        merged.groupby(["pn_id", "mbon_id"])
        .agg(
            kc_count=("kc_id", "nunique"),
            total_path_weight=("path_weight", "sum"),
            max_path_weight=("path_weight", "max"),
        )
        .reset_index()
    )
    return summary


def weighted_centralities(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted in/out degree and betweenness centrality."""

    required_node_cols = {"node_id"}
    required_edge_cols = {"source_id", "target_id", "synapse_weight"}
    if missing := required_node_cols - set(nodes.columns):
        raise ValueError(f"Node table missing columns: {sorted(missing)}")
    if missing := required_edge_cols - set(edges.columns):
        raise ValueError(f"Edge table missing columns: {sorted(missing)}")

    graph = nx.DiGraph()
    for node_id in nodes["node_id"].astype(np.int64):
        graph.add_node(int(node_id))
    for source, target, weight in edges[["source_id", "target_id", "synapse_weight"]].itertuples(index=False):
        graph.add_edge(int(source), int(target), weight=float(weight))

    in_weights = edges.groupby("target_id")["synapse_weight"].sum()
    out_weights = edges.groupby("source_id")["synapse_weight"].sum()

    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)

    df = nodes[["node_id"]].copy()
    df["deg_in_w"] = df["node_id"].map(in_weights).fillna(0.0)
    df["deg_out_w"] = df["node_id"].map(out_weights).fillna(0.0)
    df["betweenness_w"] = df["node_id"].map(betweenness).fillna(0.0)
    return df


def label_dan_valence(dan_nodes: pd.DataFrame) -> pd.DataFrame:
    """Assign DAN valence labels (PAM, PPL1, other) based on text fields."""

    if "node_id" not in dan_nodes.columns:
        raise ValueError("DAN node table must include 'node_id'.")

    text_columns = [col for col in dan_nodes.columns if dan_nodes[col].dtype == object]
    records = []
    for _, row in dan_nodes.iterrows():
        node_id = int(row["node_id"])
        text = " ".join(str(row[col]).lower() for col in text_columns if isinstance(row[col], str))
        if "pam" in text:
            cluster = "PAM"
        elif "ppl1" in text:
            cluster = "PPL1"
        elif "dan" in text or "dopamine" in text:
            cluster = "DAN_other"
        else:
            cluster = "DAN_other"
        records.append({"node_id": node_id, "dan_cluster": cluster})
    return pd.DataFrame.from_records(records)


def _resolve_cache_paths(cache_dir: Path, stem: Optional[str]) -> Mapping[str, Path]:
    if stem:
        candidates = {
            "nodes": cache_dir / f"{stem}.nodes.parquet",
            "edges": cache_dir / f"{stem}.edges.parquet",
            "dan_edges": cache_dir / f"{stem}.dan_edges.parquet",
            "meta": cache_dir / f"{stem}.meta.json",
        }
        if all(path.exists() for path in candidates.values() if path.suffix != ".json"):
            return candidates
    return {
        "nodes": cache_dir / "nodes.parquet",
        "edges": cache_dir / "edges.parquet",
        "dan_edges": cache_dir / "dan_edges.parquet",
        "meta": cache_dir / "meta.json",
    }


def cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="PGCN structural metrics")
    parser.add_argument("--cache-dir", type=Path, default=Path("data") / "cache")
    parser.add_argument("--cache-stem", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cache_paths = _resolve_cache_paths(args.cache_dir, args.cache_stem)
    nodes = pd.read_parquet(cache_paths["nodes"])
    edges = pd.read_parquet(cache_paths["edges"])
    dan_edges = pd.read_parquet(cache_paths["dan_edges"])

    jaccard = jaccard_kc_overlap(
        edges[edges["source_id"].isin(nodes.loc[nodes["type"] == "PN", "node_id"])],
        pn_nodes=nodes[nodes["type"] == "PN"],
    )
    path_summary = path_lengths_pn_kc_mbon(edges, nodes=nodes)
    centrality = weighted_centralities(nodes, pd.concat([edges, dan_edges], ignore_index=True))
    dan_labels = label_dan_valence(nodes[nodes["type"] == "DAN"])

    out_dir = args.out or args.cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jaccard.to_parquet(out_dir / "kc_overlap.parquet", index=False)
    path_summary.to_parquet(out_dir / "pn_kc_mbon_paths.parquet", index=False)
    centrality.to_parquet(out_dir / "weighted_centrality.parquet", index=False)
    dan_labels.to_parquet(out_dir / "dan_valence.parquet", index=False)

    summary = {
        "jaccard_rows": len(jaccard),
        "path_rows": len(path_summary),
        "centrality_rows": len(centrality),
        "dan_rows": len(dan_labels),
    }
    (out_dir / "metrics_meta.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    LOGGER.info("Metrics written to %s", out_dir)


def main() -> None:  # pragma: no cover - CLI forwarding wrapper
    cli()


__all__ = [
    "jaccard_kc_overlap",
    "path_lengths_pn_kc_mbon",
    "weighted_centralities",
    "label_dan_valence",
    "cli",
    "main",
]
