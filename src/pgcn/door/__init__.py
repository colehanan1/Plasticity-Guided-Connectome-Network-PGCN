"""DoOR Database Integration for PGCN ORN Pathway Analysis.

This module provides comprehensive integration between the Database of Odorant Responses
(DoOR) and FlyWire connectome labels to enable complete ORN→behavioral pathway analysis.

Key Components
--------------
- DoORDataManager: Access and manage DoOR database responses
- FlyWireORNIdentifier: Identify ORN cells from FlyWire community labels
- ORNBehaviorPathwayAnalyzer: Trace complete ORN→behavior pathways
- DoORConstrainedORNLayer: Neural network layer with biologically realistic responses

Example
-------
>>> from pgcn.door import DoORDataManager, FlyWireORNIdentifier
>>>
>>> # Initialize DoOR database
>>> door_manager = DoORDataManager(method='rpy2')
>>> door_data = door_manager.load_door_data()
>>>
>>> # Identify ORN cells from FlyWire labels
>>> identifier = FlyWireORNIdentifier(door_manager)
>>> olfactory_cells = identifier.identify_olfactory_cells('data/processed_labels.csv.gz')
>>> print(f"Found {len(olfactory_cells)} olfactory cells")
"""

from pgcn.door.door_data_manager import DoORDataManager
from pgcn.door.orn_identifier import FlyWireORNIdentifier
from pgcn.door.pathway_analyzer import ORNBehaviorPathwayAnalyzer

__all__ = [
    'DoORDataManager',
    'FlyWireORNIdentifier',
    'ORNBehaviorPathwayAnalyzer',
]

__version__ = '0.1.0'
