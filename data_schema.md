# Data Schema

## Overview
This document describes the data schema and structure for the FlyWire connectome project.

## Data Directory Structure
```
data/
└── cache/          # Cached connectome data and intermediate results
```

## Cache Directory
The `data/cache/` directory stores:
- Downloaded connectome data from FlyWire
- Processed connectivity matrices
- Intermediate computation results
- Serialized network objects

## Data Format Specifications

### Connectome Data
- **Format**: Parquet/CSV
- **Fields**: 
  - `pre_pt_root_id`: Pre-synaptic neuron ID
  - `post_pt_root_id`: Post-synaptic neuron ID
  - `syn_count`: Number of synapses
  - Additional metadata fields from CAVEclient

### Network Data
- **Format**: NetworkX graph objects (pickled/JSON)
- **Node attributes**: Neuron IDs, cell types, positions
- **Edge attributes**: Synapse counts, weights, plasticity parameters

## Data Sources
- **FlyWire**: Drosophila brain connectome data via CAVEclient
- **Version tracking**: Data versions stored in cache metadata

## Cache Management
- Cache files are gitignored but directory structure is preserved
- Cache invalidation based on data version updates
- Automatic re-download on cache miss
