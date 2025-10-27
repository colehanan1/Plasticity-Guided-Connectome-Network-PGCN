"""Top-level package for the Plasticity-Guided Connectome Network (PGCN)."""

from __future__ import annotations

from . import chemical, data
from .connectome_pipeline import CacheArtifacts, ConnectomePipeline, main as cache_main
from .models import (
    ChemicalSTDP,
    ChemicallyInformedDrosophilaModel,
    DrosophilaReservoir,
)

__all__ = [
    "CacheArtifacts",
    "ConnectomePipeline",
    "cache_main",
    "chemical",
    "data",
    "ChemicalSTDP",
    "ChemicallyInformedDrosophilaModel",
    "DrosophilaReservoir",
]
