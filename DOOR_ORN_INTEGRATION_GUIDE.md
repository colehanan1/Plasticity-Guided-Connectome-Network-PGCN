# DoOR-FlyWire ORN Pathway Analysis System

Complete integration of the Database of Odorant Responses (DoOR) with FlyWire connectome labels for comprehensive olfactory receptor neuron (ORN) identification and pathway analysis in the PGCN framework.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Architecture](#architecture)
- [Biological Context](#biological-context)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Overview

This system enables **biologically accurate** olfactory pathway analysis by integrating:

- **DoOR Database v2.0**: 693 odorants × 78 responding units with 7,381 data points
- **FlyWire Community Labels**: 100,013 neurons with processed cell type annotations
- **PGCN Neural Network**: Existing PN→KC→MBON circuit with dopamine-modulated plasticity

The integration enables mechanistic testing of ORN/PN pathway blocking hypotheses and behavioral predictions.

## Features

### ✅ Core Components

1. **DoORDataManager**: Multi-method DoOR database access
   - rpy2 (primary): Direct R package integration
   - CSV (backup): Preprocessed data files
   - Zenodo (fallback): Automatic download from archive

2. **FlyWireORNIdentifier**: Advanced pattern matching
   - Complete receptor list (60 ORs, 16 IRs, 2 GRs)
   - Sensillum-based identification
   - Anatomical location matching
   - DoOR cross-validation

3. **ORNBehaviorPathwayAnalyzer**: Complete pathway tracing
   - Or42b→attraction pathway analysis
   - Or47b→hexanol→feeding pathway
   - Quantitative behavioral predictions
   - Connectivity-based circuit tracing

4. **DoORConstrainedORNLayer**: Neural network integration
   - PyTorch layer with DoOR response profiles
   - Hill equation concentration-response curves
   - Odor mixture interactions
   - Biological noise modeling

### 📊 Expected Results

- **>500 olfactory cells identified** from FlyWire labels
- **>90% accuracy** with DoOR cross-validation
- **<5 minutes runtime** for complete analysis (100K+ cells)
- **Publication-ready** pathway visualizations

## Installation

### Prerequisites

- Python 3.8+
- R 4.0+ (for DoOR package access)
- 2GB RAM minimum
- Internet connection (for DoOR database)

### Method 1: Automated Installation (Recommended)

```bash
# Clone repository (if not already)
cd Plasticity-Guided-Connectome-Network-PGCN-

# Run installation script
bash scripts/install_door_system.sh
```

This will:
1. Check Python/R versions
2. Install Python dependencies
3. Install DoOR R packages
4. Create directory structure
5. Download FlyWire labels (optional)
6. Run installation tests

### Method 2: Manual Installation

```bash
# Install Python dependencies
pip install -r requirements_door.txt

# Install R (if not present)
# Ubuntu/Debian:
sudo apt-get install r-base r-base-dev

# macOS:
brew install r

# Windows: Download from https://cran.r-project.org/

# Install DoOR R packages
python3 << EOF
from pgcn.door import DoORDataManager
manager = DoORDataManager()
manager.install_door_packages()
EOF

# Download FlyWire labels
wget https://codex.flywire.ai/api/download?dataset=fafb \
     -O data/processed_labels.csv.gz
```

### Verify Installation

```bash
# Run tests
pytest tests/door/ -v

# Check imports
python3 -c "from pgcn.door import DoORDataManager, FlyWireORNIdentifier"
```

## Quick Start

### Example 1: Basic ORN Identification

```python
from pgcn.door import DoORDataManager, FlyWireORNIdentifier

# Initialize DoOR database
door_manager = DoORDataManager(method='rpy2')
door_data = door_manager.load_door_data()

# Create ORN identifier
identifier = FlyWireORNIdentifier(door_manager)

# Identify olfactory cells
cells = identifier.identify_olfactory_cells('data/processed_labels.csv.gz')

print(f"Found {len(cells)} olfactory cells")
print(f"DoOR validated: {cells['door_match'].notna().sum()}")

# Filter for Or42b
or42b_cells = identifier.filter_by_receptor(cells, 'Or42b')
print(f"Or42b cells: {len(or42b_cells)}")
```

### Example 2: Pathway Analysis

```python
from pgcn.door import ORNBehaviorPathwayAnalyzer

# Create pathway analyzer
analyzer = ORNBehaviorPathwayAnalyzer(door_manager, cells)

# Analyze Or42b→attraction pathway
or42b_pathway = analyzer.analyze_or42b_pathway()

print(f"Or42b cells: {or42b_pathway['n_cells']}")
print(f"Top ligands:")
for ligand, response in list(or42b_pathway['best_ligands'].items())[:5]:
    print(f"  {ligand}: {response:.3f}")

# Analyze Or47b→hexanol→feeding pathway
or47b_pathway = analyzer.analyze_or47b_hexanol_pathway()
feeding_prob = or47b_pathway['feeding_prediction']['feeding_probability']
print(f"Feeding probability: {feeding_prob:.2%}")
```

### Example 3: Neural Network Integration

```python
import torch
from pgcn.models.door_constrained_orn import DoORConstrainedORNLayer

# Create DoOR-constrained ORN layer
orn_layer = DoORConstrainedORNLayer(
    door_data=door_data,
    concentration_response=True,
    noise_level=0.1
)

# Encode odor mixture
response = orn_layer.encode_odor_mixture(
    ['ethyl acetate', 'hexanol'],
    concentrations=[0.8, 0.5]
)

print(f"ORN response: {response.shape}")
print(f"Active receptors: {(response > 0.1).sum().item()}")
```

### Example 4: Complete Analysis Script

```bash
# Run complete analysis pipeline
python scripts/complete_orn_analysis.py \
    --labels data/processed_labels.csv.gz \
    --output results/orn_analysis/

# Output:
# - identified_olfactory_cells.csv
# - pathway_analysis_report.json
# - analysis_summary.json
```

## Detailed Usage

### DoORDataManager

```python
# Method 1: rpy2 (requires R and DoOR packages)
manager = DoORDataManager(method='rpy2')

# Method 2: CSV backup (if rpy2 fails)
manager = DoORDataManager(
    method='csv',
    backup_csv_path='data/door_responses.csv.gz'
)

# Load DoOR database
door_data = manager.load_door_data()
# Returns dict with:
#   - response_matrix: Normalized responses (odorants × receptors)
#   - response_matrix_non_normalized: Raw responses
#   - odor_info: Odorant metadata
#   - receptor_info: Receptor metadata

# Get receptor list
receptors = manager.get_complete_receptor_list()
print(f"Total: {len(receptors['all_receptors'])}")
print(f"ORs: {receptors['or_receptors']}")
print(f"IRs: {receptors['ir_receptors']}")

# Get specific receptor profile
or42b_profile = manager.get_receptor_response_profile('Or42b', top_n=10)
```

### FlyWireORNIdentifier

```python
# Create identifier with custom confidence threshold
identifier = FlyWireORNIdentifier(
    door_manager=door_manager,
    confidence_threshold=0.7  # Higher = stricter matching
)

# Identify cells (full analysis)
cells = identifier.identify_olfactory_cells('processed_labels.csv.gz')

# Identify cells (limited for testing)
cells = identifier.identify_olfactory_cells(
    'processed_labels.csv.gz',
    max_cells=10000  # Process first 10K cells only
)

# Access results
print(cells.columns)
# ['root_id', 'processed_label', 'receptor_type', 'receptor_subtype',
#  'anatomical_location', 'cell_type', 'confidence_score', 'door_match']

# Filter by receptor
or_cells = cells[cells['receptor_type'] == 'or_receptors']
ir_cells = cells[cells['receptor_type'] == 'ir_receptors']

# Get summary statistics
summary = identifier.get_summary_statistics()
print(f"Match rate: {summary['match_rate']:.2%}")
print(f"Validation rate: {summary['validation_rate']:.2%}")
```

### ORNBehaviorPathwayAnalyzer

```python
# Create analyzer (with optional connectivity data)
analyzer = ORNBehaviorPathwayAnalyzer(
    door_manager=door_manager,
    identified_cells=cells,
    connectivity_data=connectivity_df  # Optional
)

# Or42b pathway
or42b = analyzer.analyze_or42b_pathway()
if 'behavioral_predictions' in or42b:
    behavior = or42b['behavioral_predictions']
    print(f"Attraction probability: {behavior['attraction_probability']:.2%}")

# Or47b-hexanol pathway
or47b = analyzer.analyze_or47b_hexanol_pathway()
if 'hexanol_responses' in or47b:
    for odorant, response in or47b['hexanol_responses'].items():
        print(f"{odorant}: {response:.3f}")

# Comprehensive report
report = analyzer.generate_comprehensive_report()
# Save to JSON
import json
with open('pathway_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
```

### DoORConstrainedORNLayer (PyTorch)

```python
import torch
import torch.nn as nn
from pgcn.models.door_constrained_orn import DoORConstrainedORNLayer

# Create ORN layer
orn_layer = DoORConstrainedORNLayer(
    door_data=door_data,
    concentration_response=True,  # Enable Hill equation
    noise_level=0.1,  # Biological noise std
    device='cpu'  # or 'cuda'
)

# Single odorant encoding
response1 = orn_layer.encode_odor_mixture(['ethyl acetate'])

# Mixture with concentrations
response2 = orn_layer.encode_odor_mixture(
    ['ethyl acetate', 'hexanol', 'geosmin'],
    concentrations=[0.8, 0.5, 0.3]
)

# Get specific receptor response
or42b_response = orn_layer.get_receptor_response(
    'Or42b',
    'ethyl hexanoate',
    concentration=0.5
)

# Integrate with existing PGCN circuit
class ExtendedOlfactoryCircuit(nn.Module):
    def __init__(self, door_data, pn_to_kc, kc_to_mbon):
        super().__init__()
        self.orn_layer = DoORConstrainedORNLayer(door_data)
        self.pn_to_kc = pn_to_kc
        self.kc_to_mbon = kc_to_mbon

    def forward(self, odorant_names, concentrations):
        # ORN encoding
        orn_response = self.orn_layer.encode_odor_mixture(
            odorant_names, concentrations
        )
        # PN activation (ORN drives PN)
        pn_activity = orn_response  # Or apply ORN→PN transformation

        # KC sparse coding
        kc_activity = self.pn_to_kc(pn_activity)

        # MBON output
        mbon_output = self.kc_to_mbon(kc_activity)

        return mbon_output
```

## Architecture

### System Components

```
DoOR Database (R packages)
    ↓
DoORDataManager (Python/rpy2)
    ↓
┌───────────────────────────────────┐
│  DoOR-FlyWire Integration Layer   │
├───────────────────────────────────┤
│  - FlyWireORNIdentifier           │
│  - ORNBehaviorPathwayAnalyzer     │
│  - DoORConstrainedORNLayer        │
└───────────────────────────────────┘
    ↓
PGCN Neural Network
    ↓
ORN → PN → KC → MBON → Behavior
```

### Data Flow

1. **DoOR Database Access**
   - rpy2 → R DoOR packages → response matrix
   - CSV fallback if rpy2 unavailable
   - Caching for performance

2. **ORN Identification**
   - FlyWire labels → pattern matching → identified cells
   - DoOR validation → confidence scoring
   - Receptor categorization

3. **Pathway Analysis**
   - Identified cells + DoOR responses → pathway tracing
   - FlyWire connectivity → circuit structure
   - Behavioral predictions

4. **Neural Network Integration**
   - DoOR responses → PyTorch layer → ORN encoding
   - Concentration-response curves
   - Integration with PGCN circuit

## Biological Context

### Drosophila Olfactory System

```
Antenna/Palp (ORNs)
    ↓ ~50 glomeruli
Antennal Lobe (PNs)
    ↓ ~2000 connections
Mushroom Body (KCs)
    ↓ sparse coding (~5% active)
Mushroom Body Output (MBONs)
    ↓ dopamine modulation
Behavioral Output
```

### Receptor Types

- **Odorant Receptors (ORx)**: 60 types, general odorants
- **Ionotropic Receptors (IRx)**: 16 types, acids/amines
- **Gustatory Receptors (GRx)**: 2 types, CO2/pheromones

### Key Pathways

**Or42b → Attraction**
- Responds to: Fruit volatiles (ethyl hexanoate, etc.)
- Glomerulus: DM1
- Behavior: Attraction to food sources

**Or47b → Feeding**
- Responds to: Hexanol, palmitoleic acid
- Glomerulus: VA1v
- Behavior: Proboscis extension (feeding)

## API Reference

### DoORDataManager

```python
class DoORDataManager:
    """DoOR database access and management."""

    def __init__(self, method='rpy2', backup_csv_path=None, cache_dir='data/door_cache')

    def install_door_packages() -> bool
    def load_door_data() -> Dict[str, pd.DataFrame]
    def get_complete_receptor_list() -> Dict[str, List[str]]
    def get_receptor_response_profile(receptor_name: str, top_n: int = 10) -> pd.Series
```

### FlyWireORNIdentifier

```python
class FlyWireORNIdentifier:
    """ORN identification from FlyWire labels."""

    def __init__(self, door_manager: DoORDataManager, confidence_threshold: float = 0.5)

    def identify_olfactory_cells(labels_file: str, max_cells: Optional[int] = None) -> pd.DataFrame
    def filter_by_receptor(identified_cells: pd.DataFrame, receptor_name: str) -> pd.DataFrame
    def get_summary_statistics() -> Dict
```

### ORNBehaviorPathwayAnalyzer

```python
class ORNBehaviorPathwayAnalyzer:
    """Pathway analysis and behavioral predictions."""

    def __init__(self, door_manager: DoORDataManager, identified_cells: pd.DataFrame,
                 connectivity_data: Optional[pd.DataFrame] = None)

    def analyze_or42b_pathway() -> Dict
    def analyze_or47b_hexanol_pathway() -> Dict
    def generate_comprehensive_report() -> Dict
```

### DoORConstrainedORNLayer

```python
class DoORConstrainedORNLayer(nn.Module):
    """PyTorch layer for ORN encoding."""

    def __init__(self, door_data: Dict, concentration_response: bool = True,
                 noise_level: float = 0.1, device: str = 'cpu')

    def encode_odor_mixture(odorant_list: List[str],
                           concentrations: Optional[List[float]] = None) -> torch.Tensor
    def get_receptor_response(receptor_name: str, odorant_name: str,
                             concentration: float = 1.0) -> float
    def forward(odor_input: torch.Tensor) -> torch.Tensor
```

## Testing

### Run All Tests

```bash
# Full test suite
pytest tests/door/ -v

# With coverage
pytest tests/door/ --cov=src/pgcn/door --cov-report=html

# Specific test file
pytest tests/door/test_door_data_manager.py -v

# Specific test
pytest tests/door/test_orn_identifier.py::TestFlyWireORNIdentifier::test_pattern_matching_or_receptor -v
```

### Integration Tests

```bash
# Run integration tests (requires R, DoOR packages, FlyWire labels)
pytest tests/door/ -v -m integration

# Skip slow tests
pytest tests/door/ -v -m "not slow"
```

### Expected Test Results

- **DoORDataManager**: 15 tests, ~10 seconds
- **FlyWireORNIdentifier**: 12 tests, ~5 seconds
- **Integration**: 3 tests, ~30 seconds (if data available)

## Troubleshooting

### Issue: rpy2 Installation Fails

**Solution 1**: Install R development libraries
```bash
# Ubuntu/Debian
sudo apt-get install r-base-dev

# macOS
brew install r
```

**Solution 2**: Use CSV fallback
```python
manager = DoORDataManager(
    method='csv',
    backup_csv_path='path/to/door_responses.csv'
)
```

### Issue: DoOR R Packages Won't Install

**Error**: `Error in install.packages...`

**Solution**: Install manually in R
```r
# In R console
install.packages("devtools")
devtools::install_github("ropensci/DoOR.data", ref="v2.0.1")
devtools::install_github("ropensci/DoOR.functions", ref="v2.0.1")
```

### Issue: FlyWire Labels Not Found

**Solution**: Download from FlyWire Codex
```bash
wget https://codex.flywire.ai/api/download?dataset=fafb \
     -O data/processed_labels.csv.gz
```

### Issue: No ORNs Identified

**Check**:
1. FlyWire labels format: `root_id,processed_label`
2. Labels contain receptor names (Or42b, etc.)
3. Confidence threshold not too high (try 0.3)

**Debug**:
```python
identifier = FlyWireORNIdentifier(door_manager, confidence_threshold=0.3)
cells = identifier.identify_olfactory_cells(labels_path, max_cells=1000)
print(f"Found: {len(cells)}")
print(identifier.search_stats)
```

### Issue: PyTorch Not Available

**Solution**: Install PyTorch
```bash
# CPU version
pip install torch

# GPU version (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Fallback**: Use NumPy backend (automatic)

## Citation

If you use this DoOR-FlyWire integration in your research, please cite:

```bibtex
@software{door_flywire_integration,
  title={DoOR-FlyWire ORN Pathway Analysis System},
  author={Your Name},
  year={2024},
  note={Part of PGCN framework},
  url={https://github.com/your-repo/PGCN}
}

@article{muench2016door,
  title={DoOR 2.0--comprehensive mapping of Drosophila melanogaster odorant responses},
  author={M{\"u}nch, Daniel and Galizia, C Giovanni},
  journal={Scientific reports},
  volume={6},
  pages={21841},
  year={2016}
}
```

## License

This integration is part of the PGCN framework. See LICENSE for details.

## Contact

For questions or issues:
- GitHub Issues: [your-repo/issues](https://github.com/your-repo/issues)
- Email: your-email@institution.edu

## Acknowledgments

- **DoOR Database**: Münch & Galizia (2016)
- **FlyWire Connectome**: Dorkenwald et al. (2024)
- **PGCN Framework**: Original PGCN authors
