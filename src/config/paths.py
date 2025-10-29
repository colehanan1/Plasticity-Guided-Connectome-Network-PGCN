"""Centralised filesystem paths for local FlyWire datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

__all__ = [
    "get_dataset_paths",
    "DATA_ROOT",
    "CONNECTIONS_FILE",
    "CELL_TYPES_FILE",
    "CLASSIFICATION_FILE",
    "NEUROTRANSMITTER_FILE",
    "NAMES_FILE",
    "PROCESSED_LABELS_FILE",
]

DATA_ROOT = Path(os.getenv("PGCN_FLYWIRE_DATA", Path("data") / "flywire"))

CONNECTIONS_FILE = DATA_ROOT / "connections_princeton.csv.gz"
CELL_TYPES_FILE = DATA_ROOT / "consolidated_cell_types.csv.gz"
CLASSIFICATION_FILE = DATA_ROOT / "classification.csv.gz"
NEUROTRANSMITTER_FILE = DATA_ROOT / "neurons.csv.gz"
NAMES_FILE = DATA_ROOT / "names.csv.gz"
PROCESSED_LABELS_FILE = DATA_ROOT / "processed_labels.csv.gz"


def get_dataset_paths() -> Dict[str, Path]:
    """Return a mapping of dataset identifiers to their resolved paths."""

    return {
        "connections": CONNECTIONS_FILE,
        "cell_types": CELL_TYPES_FILE,
        "classification": CLASSIFICATION_FILE,
        "neurons": NEUROTRANSMITTER_FILE,
        "names": NAMES_FILE,
        "processed_labels": PROCESSED_LABELS_FILE,
    }
