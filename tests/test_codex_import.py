from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pgcn.codex_import import CodexImportConfig, build_codex_cache, cli


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_build_codex_cache_basic(tmp_path: Path) -> None:
    neurons_csv = tmp_path / "neurons.csv"
    synapses_csv = tmp_path / "synapses.csv"

    _write_csv(
        neurons_csv,
        [
            {"root_id": 1, "cell_type": "PN", "name": "PN-a"},
            {"root_id": 2, "cell_type": "Kenyon cell", "name": "KC-a"},
            {"root_id": 3, "cell_type": "MBON", "name": "MBON-a"},
            {"root_id": 4, "cell_type": "DAN", "name": "DAN-a"},
            {"root_id": 5, "cell_type": "Other"},
        ],
    )
    _write_csv(
        synapses_csv,
        [
            {"pre_root_id": 1, "post_root_id": 2, "weight": 5},
            {"pre_root_id": 2, "post_root_id": 3, "weight": 7},
            {"pre_root_id": 4, "post_root_id": 2, "weight": 2},
            {"pre_root_id": 5, "post_root_id": 1, "weight": 1},
        ],
    )

    artifacts = build_codex_cache(neurons_csv, synapses_csv, tmp_path)

    nodes = pd.read_parquet(artifacts.nodes)
    edges = pd.read_parquet(artifacts.edges)
    dan_edges = pd.read_parquet(artifacts.dan_edges)
    meta = json.loads(artifacts.meta.read_text())

    assert set(nodes.type.unique()) == {"PN", "KC", "MBON", "DAN"}
    assert ((edges.edge_type == "PN_KC").sum()) == 1
    assert ((edges.edge_type == "KC_MBON").sum()) == 1
    assert len(dan_edges) == 1
    assert meta["counts"]["pn_kc_edges"] == 1


def test_build_codex_cache_custom_patterns(tmp_path: Path) -> None:
    neurons_csv = tmp_path / "neurons.csv"
    synapses_csv = tmp_path / "synapses.csv"

    _write_csv(
        neurons_csv,
        [
            {"pt_root_id": 10, "group": "Projection neuron"},
            {"pt_root_id": 11, "group": "Kenyon subset"},
        ],
    )
    _write_csv(
        synapses_csv,
        [
            {"source": 10, "target": 11, "synapse_count": 3},
        ],
    )

    config = CodexImportConfig()
    config.add_pattern("PN", "Projection")
    config.add_pattern("KC", "Kenyon subset")

    artifacts = build_codex_cache(neurons_csv, synapses_csv, tmp_path / "out", config=config)
    nodes = pd.read_parquet(artifacts.nodes)
    assert set(nodes.type.unique()) == {"PN", "KC"}


def test_cli_invocation(tmp_path: Path) -> None:
    neurons_csv = tmp_path / "neurons.csv"
    synapses_csv = tmp_path / "synapses.csv"

    _write_csv(neurons_csv, [{"root_id": 1, "cell_type": "PN"}, {"root_id": 2, "cell_type": "KC"}])
    _write_csv(synapses_csv, [{"pre_root_id": 1, "post_root_id": 2, "weight": 1}])

    out_dir = tmp_path / "cache"
    exit_code = cli(
        [
            "--neurons",
            str(neurons_csv),
            "--synapses",
            str(synapses_csv),
            "--out",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    assert (out_dir / "nodes.parquet").exists()
