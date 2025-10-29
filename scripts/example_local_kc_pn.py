"""Example script demonstrating PN→KC matrix construction from local FlyWire CSVs."""

from __future__ import annotations

from pathlib import Path


from config import paths
from data_loaders.connectivity import build_kc_pn_matrix, filter_mushroom_body_connections
from data_loaders.flywire_local import FlyWireLocalDataLoader
from data_loaders.neuron_classification import get_kc_neurons, get_pn_neurons


def main() -> None:
    dataset_dir = Path(paths.DATA_ROOT)
    loader = FlyWireLocalDataLoader(dataset_dir)

    print(f"Loading datasets from {dataset_dir.resolve()}")
    connections = loader.load_connections(neuropil_filter=None, min_synapses=5)
    cell_types = loader.load_cell_types()
    classification = loader.load_classification()

    mushroom_connections = filter_mushroom_body_connections(connections)
    kc_frame = get_kc_neurons(cell_types, classification)
    pn_frame = get_pn_neurons(cell_types, classification)

    matrix = build_kc_pn_matrix(
        mushroom_connections,
        kc_ids=kc_frame["root_id"].tolist(),
        pn_ids=pn_frame["root_id"].tolist(),
    )

    print(f"KC count: {matrix.shape[0]} | PN count: {matrix.shape[1]}")
    print(f"Non-zero synaptic edges: {matrix.nnz}")


if __name__ == "__main__":
    main()
