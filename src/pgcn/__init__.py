"""Top-level package for the Plasticity-Guided Connectome Network (PGCN)."""

from __future__ import annotations

from .connectome_pipeline import CacheArtifacts, ConnectomePipeline, main as cache_main
from . import chemical
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
    "ChemicalSTDP",
    "ChemicallyInformedDrosophilaModel",
    "DrosophilaReservoir",
]
