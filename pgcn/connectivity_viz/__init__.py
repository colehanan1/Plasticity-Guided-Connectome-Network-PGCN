"""
GRN Downstream Connectivity Analysis Package

Modules:
- grn_downstream_extractor: Load GRNs and fetch connectivity from FlyWire
- grn_connectivity_analyzer: Compute statistics and comparisons
- grn_visualization: Generate publication-ready figures
- main: CLI entry point

Author: Claude AI
Date: 2025-11-03
"""

__version__ = "1.0.0"
__author__ = "Claude AI"

from .grn_downstream_extractor import GRNDownstreamExtractor
from .grn_connectivity_analyzer import GRNConnectivityAnalyzer
from .grn_visualization import GRNConnectivityVisualizer

__all__ = [
    'GRNDownstreamExtractor',
    'GRNConnectivityAnalyzer',
    'GRNConnectivityVisualizer'
]
