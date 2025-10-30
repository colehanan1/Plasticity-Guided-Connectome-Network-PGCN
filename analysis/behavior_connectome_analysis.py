"""CLI for linking behavioral performance with connectome structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
import yaml

from pgcn.data import load_behavioral_dataframe
from pgcn.models import BehaviorConnectomeAnalyzer
from pgcn.models.reservoir import DrosophilaReservoir


def _load_mapping(path: Optional[Path]) -> Optional[Mapping[str, str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Mapping file '{path}' does not exist.")
    if path.suffix in {".yaml", ".yml"}:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Mapping files must be JSON or YAML.")
    if not isinstance(payload, Mapping):
        raise ValueError("Mapping payload must be a dictionary of trial_label → glomerulus.")
    return {str(key): str(value) for key, value in payload.items()}


def _resolve_cache_path(cache_dir: Path, stem: str) -> Path:
    canonical = cache_dir / f"{stem}.parquet"
    if canonical.exists():
        return canonical
    candidates = sorted(cache_dir.glob(f"{stem}*.parquet"), key=lambda path: path.stat().st_mtime)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        (
            f"Could not find '{stem}.parquet' in {cache_dir}. "
            "Run `pgcn-cache --local-data data/flywire --out data/cache/` "
            "before executing the behaviour-connectome analysis."
        )
    )


def _infer_glomerulus_from_cache(cache_dir: Path) -> pd.DataFrame:
    nodes_path = _resolve_cache_path(cache_dir, "nodes")
    nodes = pd.read_parquet(nodes_path)

    type_column = DrosophilaReservoir._infer_column(  # type: ignore[attr-defined]
        nodes.columns,
        ["type", "cell_type", "node_type"],
    )
    if type_column is None:
        raise ValueError("Connectome nodes table does not contain a cell-type column.")

    node_id_column = DrosophilaReservoir._infer_column(  # type: ignore[attr-defined]
        nodes.columns,
        ["node_id", "id"],
    )
    if node_id_column is None:
        raise ValueError("Connectome nodes table does not contain a node identifier column.")

    glomerulus_column = DrosophilaReservoir._infer_column(  # type: ignore[attr-defined]
        nodes.columns,
        [
            "glomerulus",
            "pn_glomerulus",
            "glomerulus_name",
            "glomerulus_label",
            "pn_class",
            "pn_type",
        ],
    )
    if glomerulus_column is None:
        raise FileNotFoundError(
            "Glomerulus labels were not discovered in the cache nodes table. "
            "Provide --glomerulus-assignments to supply an explicit CSV."
        )

    pn_mask = nodes[type_column].astype(str).str.upper() == "PN"
    pn_nodes = nodes.loc[pn_mask].sort_values(node_id_column)
    if pn_nodes.empty:
        raise ValueError("Connectome cache does not contain any projection neurons (PN).")

    assignments = pd.DataFrame(
        {
            "pn_index": range(len(pn_nodes)),
            "pn_id": pn_nodes[node_id_column].astype(int).to_numpy(),
            "glomerulus": pn_nodes[glomerulus_column]
            .fillna("unknown")
            .astype(str)
            .to_numpy(),
        }
    )
    return assignments


def _load_glomerulus_assignments(path_arg: Optional[str], cache_dir: Path) -> pd.DataFrame:
    if path_arg:
        candidate = Path(path_arg)
        if candidate.exists():
            return pd.read_csv(candidate)
        print(
            f"Glomerulus assignments '{candidate}' not found. "
            "Attempting to infer labels directly from the FlyWire cache..."
        )
    return _infer_glomerulus_from_cache(cache_dir)


def run_analysis(args: argparse.Namespace) -> None:
    behavior_path = Path(args.behavior_data) if args.behavior_data else None
    behavior_df = load_behavioral_dataframe(behavior_path)
    cache_dir = Path(args.cache_dir)
    glomerulus_df = _load_glomerulus_assignments(args.glomerulus_assignments, cache_dir)
    mapping = _load_mapping(Path(args.trial_to_glomerulus) if args.trial_to_glomerulus else None)

    analyzer = BehaviorConnectomeAnalyzer(cache_dir=cache_dir, behavior_data=behavior_df)
    enrichment = analyzer.analyze_glomerulus_enrichment(glomerulus_df, mapping)
    correlations = analyzer.structural_performance_correlation(glomerulus_df, mapping)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        enrichment.to_csv(output_dir / "glomerulus_enrichment.csv", index=False)
        correlations.to_csv(output_dir / "structural_behavior_correlations.csv", index=False)
    else:
        print("=== Glomerulus Enrichment ===")
        print(enrichment.to_string(index=False))
        print("\n=== Structural-Performance Correlations ===")
        print(correlations.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="FlyWire cache directory containing PN→KC matrix")
    parser.add_argument(
        "--glomerulus-assignments",
        default=None,
        help="Optional CSV with columns pn_index, glomerulus (auto-inferred when omitted)",
    )
    parser.add_argument("--behavior-data", help="Optional override for behavioral CSV")
    parser.add_argument(
        "--trial-to-glomerulus",
        help="Optional YAML/JSON mapping from trial_label to glomerulus identifier",
    )
    parser.add_argument("--output-dir", help="Directory for CSV outputs. Defaults to stdout display.")
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
