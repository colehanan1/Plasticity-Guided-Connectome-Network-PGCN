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


def run_analysis(args: argparse.Namespace) -> None:
    behavior_path = Path(args.behavior_data) if args.behavior_data else None
    behavior_df = load_behavioral_dataframe(behavior_path)
    glomerulus_df = pd.read_csv(args.glomerulus_assignments)
    mapping = _load_mapping(Path(args.trial_to_glomerulus) if args.trial_to_glomerulus else None)

    analyzer = BehaviorConnectomeAnalyzer(cache_dir=Path(args.cache_dir), behavior_data=behavior_df)
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
    parser.add_argument("--glomerulus-assignments", required=True, help="CSV with columns pn_index, glomerulus")
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
