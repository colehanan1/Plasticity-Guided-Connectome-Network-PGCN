from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loaders.connectivity import build_kc_pn_matrix, filter_mushroom_body_connections
from data_loaders.flywire_local import FlyWireLocalDataLoader, load_flywire_connections
from data_loaders.neuron_classification import extract_neurotransmitter_info, get_kc_neurons, get_pn_neurons, map_brain_regions


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
            "root_id": [1, 2, 3, 4],
            "primary_type": [
                "Olfactory Projection Neuron",
                "Kenyon Cell",
                "Kenyon Cell",
                "Other",
            ],
            "additional_type(s)": [
                "ALPN",
                "Mushroom Body Intrinsic",
                "KC",
                pd.NA,
            ],
        }
    )
    classification = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4],
            "super_class": ["olfactory_projection", "intrinsic", "intrinsic", "glia"],
            "class": ["AL Projection", "KC", "KC", "glia_other"],
            "sub_class": ["pn", "kenyon", "kenyon", "other"],
        }
    )
    neurons = pd.DataFrame({"root_id": [1, 2, 3, 4], "nt_type": ["Cholinergic", "GABA", "GABA", "Glutamate"]})
    names = pd.DataFrame({"root_id": [1, 2, 3, 4], "group": ["AL", "MB", "MB", "LX"]})

    processed_labels = pd.DataFrame({"root_id": [1, 2, 3, 4], "label": ["PN", "KC", "KC", "Other"]})

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
    assert len(processed) == 4

    kc_frame = get_kc_neurons(loader.load_cell_types(), loader.load_classification())
    pn_frame = get_pn_neurons(loader.load_cell_types(), loader.load_classification())
    assert set(kc_frame["root_id"]) == {2, 3}
    assert set(pn_frame["root_id"]) == {1}

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
