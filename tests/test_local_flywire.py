from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_loaders.connectivity import build_kc_pn_matrix, filter_mushroom_body_connections
from data_loaders.flywire_local import FlyWireLocalDataLoader, load_flywire_connections
from data_loaders.neuron_classification import (
    extract_neurotransmitter_info,
    get_dan_neurons,
    get_kc_neurons,
    get_mbon_neurons,
    get_pn_neurons,
    infer_pn_glomerulus_labels,
    map_brain_regions,
)
from pgcn.connectome_pipeline import ConnectomePipeline


def _write_gz_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def test_load_flywire_connections_filters(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "pre_root_id": [111111111111111111, 222222222222222222],
            "post_root_id": [333333333333333333, 444444444444444444],
            "neuropil": ["MB_CA", "AL"],
            "syn_count": [10, 3],
            "nt_type": ["Cholinergic", "GABA"],
        }
    )
    path = tmp_path / "connections_princeton.csv.gz"
    _write_gz_csv(path, data)

    df = load_flywire_connections(path, neuropil_filter=["MB_CA"], min_synapses=5)
    assert len(df) == 1
    assert df.iloc[0]["pre_root_id"] == 111111111111111111

    df_chunked = load_flywire_connections(path, neuropil_filter=["MB_CA"], min_synapses=5, chunk_size=1)
    assert len(df_chunked) == 1


def test_loader_integrates_all_tables(tmp_path: Path) -> None:
    connections = pd.DataFrame(
        {
            "pre_root_id": [1, 2],
            "post_root_id": [3, 4],
            "neuropil": ["MB_CA", "MB_PEDUNCLE"],
            "syn_count": [7, 8],
            "nt_type": ["Cholinergic", "GABA"],
        }
    )
    cell_types = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5, 6],
            "primary_type": [
                "Olfactory Projection Neuron",
                "Kenyon Cell",
                "Kenyon Cell",
                "Other",
                "Mushroom Body Output Neuron",
                "PAM Dopaminergic Neuron",
            ],
            "additional_type(s)": [
                "ALPN",
                "Mushroom Body Intrinsic",
                "KC",
                pd.NA,
                "MBON",
                "DAN",
            ],
        }
    )
    classification = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5, 6],
            "super_class": [
                "olfactory_projection",
                "intrinsic",
                "intrinsic",
                "glia",
                "mushroom_body_output",
                "dopaminergic",
            ],
            "class": [
                "AL Projection",
                "KC",
                "KC",
                "glia_other",
                "mbon",
                "pam_dan",
            ],
            "sub_class": ["pn", "kenyon", "kenyon", "other", "mbon", "dan"],
        }
    )
    neurons = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5, 6],
            "nt_type": ["Cholinergic", "GABA", "GABA", "Glutamate", "Glutamate", "Dopamine"],
        }
    )
    names = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5, 6],
            "group": ["AL", "MB", "MB", "LX", "MB", "MB"],
        }
    )

    processed_labels = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5, 6],
            "processed_labels": [
                "['Projection neuron', 'DA1 glomerulus']",
                "['Kenyon cell']",
                "['Kenyon cell']",
                "['Other cell']",
                "['MBON neuron']",
                "['DAN neuron']",
            ],
        }
    )

    for filename, df in {
        "connections_princeton.csv.gz": connections,
        "consolidated_cell_types.csv.gz": cell_types,
        "classification.csv.gz": classification,
        "neurons.csv.gz": neurons,
        "names.csv.gz": names,
        "processed_labels.csv.gz": processed_labels,
    }.items():
        _write_gz_csv(tmp_path / filename, df)

    loader = FlyWireLocalDataLoader(tmp_path)
    loaded_connections = loader.load_connections()
    assert len(loaded_connections) == 2

    processed = loader.load_processed_labels()
    assert len(processed) == 6

    kc_frame = get_kc_neurons(
        loader.load_cell_types(),
        loader.load_classification(),
        names_df=loader.load_names(),
        processed_labels_df=loader.load_processed_labels(),
    )
    pn_frame = get_pn_neurons(
        loader.load_cell_types(),
        loader.load_classification(),
        names_df=loader.load_names(),
        neurons_df=loader.load_neurotransmitters(),
        processed_labels_df=loader.load_processed_labels(),
    )
    pn_glomeruli = infer_pn_glomerulus_labels(
        pn_frame,
        processed_labels_df=loader.load_processed_labels(),
    )
    mbon_frame = get_mbon_neurons(loader.load_cell_types(), loader.load_classification())
    dan_frame = get_dan_neurons(loader.load_cell_types(), loader.load_classification())
    assert set(kc_frame["root_id"]) == {2, 3}
    assert set(pn_frame["root_id"]) == {1}
    assert pn_glomeruli.tolist() == ["DA1"]
    assert set(mbon_frame["root_id"]) == {5}
    assert set(dan_frame["root_id"]) == {6}

    neurotransmitters = extract_neurotransmitter_info(loader.load_neurotransmitters(), [1, 2])
    assert set(neurotransmitters["root_id"]) == {1, 2}

    regions = map_brain_regions(loader.load_names(), [1, 3])
    assert set(regions["group"]) == {"AL", "MB"}


def test_build_kc_pn_matrix(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "pre_root_id": [1, 1, 2],
            "post_root_id": [10, 11, 10],
            "neuropil": ["MB_CA", "MB_CA", "MB_PEDUNCLE"],
            "syn_count": [5, 7, 9],
            "nt_type": ["Cholinergic", "Cholinergic", "GABA"],
        }
    )
    filtered = filter_mushroom_body_connections(data)
    matrix = build_kc_pn_matrix(filtered, kc_ids=[10, 11], pn_ids=[1, 2])

    assert matrix.shape == (2, 2)
    assert matrix.nnz == 3
    assert matrix[0, 0] == 5
    assert matrix[1, 0] == 7
    assert matrix[0, 1] == 9


def test_connectome_pipeline_local_cache(tmp_path: Path) -> None:
    data_dir = tmp_path / "inputs"
    cache_dir = tmp_path / "cache"

    connections = pd.DataFrame(
        {
            "pre_root_id": [101, 201, 401, 401],
            "post_root_id": [201, 301, 201, 301],
            "neuropil": ["MB_CA", "MB_PEDUNCLE", "MB_CA", "MB_CA"],
            "syn_count": [12, 7, 3, 2],
            "nt_type": ["ACH", "GABA", "DA", "DA"],
        }
    )
    cell_types = pd.DataFrame(
        {
            "root_id": [101, 201, 301, 401],
            "primary_type": [
                "PN-mALT-DA1",
                "Kenyon Cell",
                "Mushroom Body Output Neuron",
                "PAM Dopaminergic Neuron",
            ],
            "additional_type(s)": [
                "ALPN; DA1 glomerulus",
                "Mushroom Body Intrinsic",
                "MBON",
                "DAN",
            ],
        }
    )
    classification = pd.DataFrame(
        {
            "root_id": [101, 201, 301, 401],
            "super_class": [
                "olfactory_projection",
                "intrinsic",
                "mushroom_body_output",
                "dopaminergic",
            ],
            "class": ["projection neuron", "kc", "mbon", "pam_dan"],
            "sub_class": ["pn_da1", "kc", "mbon", "dan"],
        }
    )
    neurons = pd.DataFrame(
        {
            "root_id": [101, 201, 301, 401],
            "nt_type": ["ACH", "GABA", "GLUT", "DA"],
        }
    )
    names = pd.DataFrame(
        {
            "root_id": [101, 201, 301, 401],
            "group": ["AL", "MB", "MB", "MB"],
        }
    )
    processed_labels = pd.DataFrame(
        {
            "root_id": [101, 201, 301, 401],
            "processed_labels": [
                "['PN', 'DA1 glomerulus']",
                "['KC']",
                "['MBON']",
                "['DAN']",
            ],
        }
    )

    for filename, df in {
        "connections_princeton.csv.gz": connections,
        "consolidated_cell_types.csv.gz": cell_types,
        "classification.csv.gz": classification,
        "neurons.csv.gz": neurons,
        "names.csv.gz": names,
        "processed_labels.csv.gz": processed_labels,
    }.items():
        _write_gz_csv(data_dir / filename, df)

    pipeline = ConnectomePipeline(cache_dir=cache_dir)
    artifacts = pipeline.run(local_data_dir=data_dir)

    nodes = pd.read_parquet(artifacts.nodes)
    edges = pd.read_parquet(artifacts.edges)
    dan_edges = pd.read_parquet(artifacts.dan_edges)
    meta = json.loads(artifacts.meta.read_text())

    assert set(nodes["type"]) == {"PN", "KC", "MBON", "DAN"}
    assert nodes.loc[nodes["type"] == "PN", "glomerulus"].tolist() == ["DA1"]
    assert (edges[["source_id", "target_id"]].values.tolist()) == [[101, 201], [201, 301]]
    assert set(tuple(row) for row in dan_edges[["source_id", "target_id"]].to_numpy()) == {
        (401, 201),
        (401, 301),
    }
    assert meta["source"] == "local_csv"
    assert meta["pn_count"] == 1
    assert meta["kc_count"] == 1
    assert meta["mbon_count"] == 1
    assert meta["dan_count"] == 1
    assert Path(meta["local_data_dir"]).exists()
    assert meta["counts"]["pn_kc_edges"] == 1
    assert meta["counts"]["nodes"] == len(nodes)
