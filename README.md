# Plasticity-Guided-Connectome-Network-PGCN-

A bio-inspired RNN that fuses Drosophila connectome constraints with valence-modulated plasticity. Uses reservoir computing: fixed realistic circuits for innate odor paths plus learnable dopamine-gated synapses. Captures learning dynamics and aversive-conditioning limits from hardwired biases.

## Project Structure

```
.
├── src/                    # Source code
│   └── connectome_pipeline.py
├── tests/                  # Test suite
│   └── test_cache.py
├── data/                   # Data directory
│   └── cache/             # Cached connectome data
├── README.md              # This file
├── data_schema.md         # Data structure documentation
├── environment.yml        # Conda environment specification
└── .gitignore            # Git ignore rules
```

## Setup

### Creating the Conda Environment

```bash
conda env create -f environment.yml
conda activate PGCN
```

### Running Tests

```bash
pytest tests/
```

## Usage

```python
from src.connectome_pipeline import ConnectomePipeline

# Initialize pipeline
pipeline = ConnectomePipeline(cache_dir="data/cache")

# Fetch connectome data
pipeline.fetch_connectome("flywire")

# Process and build network
pipeline.process_data()
pipeline.build_network()
```

## Data

See [data_schema.md](data_schema.md) for details on data structure and formats.

## Dependencies

- Python 3.11
- caveclient - FlyWire data access
- pandas - Data manipulation
- pyarrow - Efficient data serialization
- networkx - Graph/network analysis
- numpy - Numerical computing
- tqdm - Progress bars
- tenacity - Retry logic
- pydantic - Data validation
- rich - Rich terminal output
- pytest - Testing framework
