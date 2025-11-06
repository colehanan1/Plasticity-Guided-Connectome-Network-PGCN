## OR7a Blocking & Cross-Transfer Analysis Guide

## Overview

This guide explains how to use the OR7a pathway analysis system to validate the groundbreaking hypothesis that **OR7a mediates BOTH benzaldehyde learning blocking AND hexanol cross-transfer** through a single unified pathway.

### Scientific Discovery Under Investigation

**Revolutionary Finding:**
1. **Benzaldehyde training** → Benzaldehyde test = **BLOCKED** (learned suppression)
2. **Benzaldehyde training** → Hexanol test = **ENHANCED** (cross-transfer)
3. **Hypothesis**: OR7a pathway is the common mechanism

**Connectome Evidence:**
- 41 OR7a neurons → **2 DL5_adPN projection neurons** (critical bottleneck!)
- 4,733 total synapses OR7a→DL5_adPN (massive pathway strength)
- Just suppressing 2 PNs should eliminate BOTH effects

**DoOR Receptor Evidence:**
- Benzaldehyde Or7a response: **0.551** (strong - training odor)
- Hexanol Or7a response: **0.139** (moderate - predicts 25.2% cross-transfer)
- Control odors: Ethyl butyrate (0.045), 3-Octanol (0.032) - minimal

## Quick Start

### Prerequisites

Ensure you've run the OR7a pathway mapping first:

```bash
# Map OR7a output targets
python scripts/map_or7a_outputs.py --data-source local

# Trace complete pathway
python scripts/map_or7a_complete_pathway.py --data-source local
```

### Basic Analysis Workflow

```bash
# 1. Analyze baseline (pathway + DoOR only)
python scripts/analyze_or7a_baseline.py

# 2. Generate suppression predictions
python scripts/predict_or7a_suppression.py

# 3. Run complete analysis with behavioral data (if available)
python scripts/analyze_or7a_baseline.py \
  --behavioral-data data/behavioral/baseline.csv
```

## Core Analysis Module

### OR7aPathwayAnalyzer Class

The main analysis class integrates:
- FlyWire connectomics (41 OR7a → 2 PN → 312 KC → 65 MBON)
- DoOR receptor responses (Or7a response profiles)
- Behavioral data (blocking + cross-transfer patterns)
- Suppression predictions (expected knockout effects)

**Example usage:**

```python
from pathlib import Path
from src.pgcn.analysis.or7a_pathway import OR7aPathwayAnalyzer
from src.pgcn.door.door_data_manager import DoORDataManager

# Initialize
door_mgr = DoORDataManager()
analyzer = OR7aPathwayAnalyzer(
    flywire_data_dir=Path('data/flywire'),
    pathway_results_dir=Path('results/or7a_complete_pathway'),
    door_manager=door_mgr,
    output_dir=Path('results/or7a_blocking_analysis')
)

# Load data
analyzer.load_or7a_data()
analyzer.load_pathway_data()
analyzer.load_door_responses()

# Analyze pathway architecture
architecture = analyzer.analyze_pathway_architecture()
print(f"Critical bottleneck: {architecture['critical_bottleneck']}")

# Analyze baseline (with or without behavioral data)
baseline = analyzer.analyze_baseline_behavior(behavioral_data=None)

# Predict suppression effects
predictions = analyzer.predict_suppression_effects()

# Generate experimental design
design = analyzer.generate_experimental_design()

# Create visualizations
analyzer.visualize_complete_analysis()
```

## Analysis Scripts

### 1. Baseline Analysis (`analyze_or7a_baseline.py`)

Analyzes baseline behavioral patterns and pathway architecture.

**Command-line options:**

```bash
python scripts/analyze_or7a_baseline.py \
  --flywire-data data/flywire \
  --pathway-results results/or7a_complete_pathway \
  --behavioral-data data/behavioral/baseline.csv \  # Optional
  --output-dir results/or7a_baseline_analysis \
  --odor-panel benzaldehyde 1-hexanol "ethyl butyrate"
```

**Outputs:**
- `or7a_door_responses.csv` - DoOR Or7a responses for odor panel
- `or7a_baseline_complete.json` - Complete baseline analysis
- Analysis summary printed to console

**Key findings reported:**
- Critical bottleneck identification (41 OR7a → 2 PNs)
- OR7a→PN connection strength (4,733 synapses)
- DoOR receptor predictions
- Behavioral blocking/transfer patterns (if data provided)
- DoOR-behavior correlation

### 2. Suppression Predictions (`predict_or7a_suppression.py`)

Generates quantitative predictions for OR7a suppression experiments.

**Command-line options:**

```bash
python scripts/predict_or7a_suppression.py \
  --flywire-data data/flywire \
  --pathway-results results/or7a_complete_pathway \
  --output-dir results/or7a_suppression_predictions
```

**Outputs:**
- `or7a_suppression_predictions.csv` - Experimental predictions table
- `or7a_suppression_predictions_complete.json` - Complete predictions

**Predictions generated:**
1. **Benzaldehyde blocking**: Expected to be eliminated (LI 0.2 → 0.7)
2. **Hexanol cross-transfer**: Expected to be lost (LI 0.5 → 0.2)
3. **Control odors**: Expected to be normal (LI ~0.7)

### 3. Complete Analysis Pipeline

Run the complete analysis programmatically:

```python
from pathlib import Path
from src.pgcn.analysis.or7a_pathway import run_complete_analysis

results = run_complete_analysis(
    flywire_data_dir=Path('data/flywire'),
    pathway_results_dir=Path('results/or7a_complete_pathway'),
    behavioral_data_path=Path('data/behavioral/baseline.csv'),  # Optional
    output_dir=Path('results/or7a_blocking_analysis')
)

# Access results
baseline = results['baseline']
predictions = results['predictions']
design = results['experimental_design']
```

## Behavioral Data Format

If you have behavioral data, it should be a CSV with these columns:

```csv
odor,training_odor,learning_index,condition,genotype
benzaldehyde,benzaldehyde,0.25,control,w1118
1-hexanol,benzaldehyde,0.52,control,w1118
benzaldehyde,none,0.68,untrained,w1118
ethyl butyrate,ethyl butyrate,0.71,control,w1118
```

**Required columns:**
- `odor`: Test odor name
- `training_odor`: Training odor used (or 'none')
- `learning_index`: Performance metric [0, 1]
- `condition`: Experimental condition (e.g., 'control', 'or7a_suppress')
- `genotype`: Fly genotype (e.g., 'w1118', 'Or7a-GAL4>UAS-Kir2.1')

**Optional columns:**
- `trial`: Trial number
- `replicate`: Biological replicate
- `n_flies`: Number of flies tested

## Analysis Outputs

### JSON Outputs

#### `or7a_baseline_complete.json`

Complete baseline analysis with structure:

```json
{
  "pathway_architecture": {
    "levels": {
      "OR7a_ORN": {"neuron_count": 41, ...},
      "DL5_Projection_Neurons": {"neuron_count": 2, ...}
    },
    "critical_bottleneck": {
      "transition": "OR7a_ORN → DL5_Projection_Neurons",
      "source_neurons": 41,
      "target_neurons": 2,
      "expansion_ratio": 0.0488,
      "severity": "High"
    },
    "or7a_pn_strength": {
      "total_synapses": 4733,
      "mean_synapses": 47.3,
      "unique_pns": 2
    }
  },
  "door_predictions": {
    "benzaldehyde_response": 0.551,
    "hexanol_response": 0.139,
    "predicted_cross_transfer": 0.252
  }
}
```

#### `or7a_suppression_predictions_complete.json`

Suppression predictions with structure:

```json
{
  "pathway_elimination": {
    "or7a_pn_synapses": 4733,
    "unique_pns": 2,
    "bottleneck_severity": "CRITICAL"
  },
  "blocking_reduction": {
    "mechanism": "OR7a pathway elimination",
    "expected_change": "Complete loss of benzaldehyde blocking",
    "quantitative": 0.551
  },
  "transfer_reduction": {
    "mechanism": "OR7a pathway elimination",
    "expected_change": "Loss of hexanol cross-transfer",
    "quantitative": 0.252
  },
  "experimental_targets": {
    "primary_pns": [720575940639080700, 720575940617207200],
    "importance_scores": [118531, 106626],
    "targeting_recommendation": "Suppress top 2-3 DL5_adPN neurons"
  }
}
```

### CSV Outputs

#### `or7a_door_responses.csv`

DoOR Or7a responses for experimental odor panel:

```csv
odor_query,odor_door_name,or7a_response,found
benzaldehyde,benzaldehyde,0.551,True
1-hexanol,1-hexanol,0.139,True
ethyl butyrate,ethyl butyrate,0.045,True
linalool,linalool,0.069,True
3-octanol,3-octanol,0.032,True
```

#### `or7a_suppression_predictions.csv`

Experimental predictions table:

```csv
condition,genotype,training,test,predicted_LI,interpretation
Wildtype Control,w1118,benzaldehyde,benzaldehyde,0.2,"Strong blocking (low learning)"
Wildtype Control,w1118,benzaldehyde,1-hexanol,0.5,"Cross-transfer (enhanced)"
OR7a Suppressed,Or7a-GAL4 > UAS-Kir2.1,benzaldehyde,benzaldehyde,0.7,"Blocking eliminated"
OR7a Suppressed,Or7a-GAL4 > UAS-Kir2.1,benzaldehyde,1-hexanol,0.2,"Cross-transfer eliminated"
```

#### `or7a_experimental_design.csv`

Complete experimental protocol design:

```csv
experiment_id,condition,training_odor,test_odor,genotype,expected_outcome,validation_metric,or7a_status
EXP1_CONTROL,Wildtype control,benzaldehyde,benzaldehyde,w1118,Strong blocking,LI < 0.3,Active
EXP2_OR7A_SUPPRESS,OR7a neuron suppression,benzaldehyde,benzaldehyde,Or7a-GAL4 > UAS-Kir2.1,No blocking,LI > 0.6,Suppressed
```

### Visualization Outputs

#### `or7a_complete_analysis.png`

5-panel comprehensive figure:
1. **Pathway schematic** - OR7a → PN → KC → MBON → Behavior
2. **DoOR responses** - Or7a responses for odor panel
3. **Bottleneck analysis** - Circuit convergence/divergence
4. **Suppression predictions** - Expected experimental outcomes
5. **Experimental design** - Protocol summary table

## Integration with PGCN Models

### Using Pathway Data in Models

```python
import pandas as pd
from scipy.sparse import csr_matrix

# Load complete pathway
pathway = pd.read_csv('results/or7a_complete_pathway/or7a_complete_pathway.csv')

# Build OR7a→PN connectivity matrix
or7a_pn = pathway[
    (pathway['source_level'] == 0) &
    (pathway['target_category'] == 'PN')
]

# Use with PGCN learning model
from src.pgcn.models.learning_model import LearningModel

# Initialize with OR7a pathway weights
model = LearningModel(
    initial_weights=or7a_pn_weights,
    plasticity_rule='bcm'
)

# Simulate learning
model.train(odor_patterns, outcomes)
```

### Simulating OR7a Suppression

```python
# Load predictions
import json
with open('results/or7a_suppression_predictions/or7a_suppression_predictions_complete.json') as f:
    predictions = json.load(f)

# Get target PNs
target_pns = predictions['experimental_targets']['primary_pns']

# Suppress OR7a pathway in model
model.suppress_neurons(target_pns)

# Test suppression effects
baseline_performance = model.test(benzaldehyde_pattern)
suppressed_performance = model.test(benzaldehyde_pattern)

print(f"Blocking reduction: {suppressed_performance - baseline_performance:.3f}")
```

## Experimental Validation Protocol

### Recommended Experimental Workflow

1. **Baseline Control Experiments**
   - Genotype: w1118 or Canton-S
   - Train: Benzaldehyde + shock
   - Test: Benzaldehyde (expect blocking, LI ~0.2)
   - Test: 1-Hexanol (expect transfer, LI ~0.5)

2. **OR7a Neuron Suppression**
   - Genotype: Or7a-GAL4 > UAS-Kir2.1
   - Train: Benzaldehyde + shock
   - Test: Benzaldehyde (expect no blocking, LI ~0.7)
   - Test: 1-Hexanol (expect no transfer, LI ~0.2)

3. **DL5_adPN Bottleneck Test**
   - Genotype: DL5-specific-GAL4 > UAS-Kir2.1
   - Same protocol as OR7a suppression
   - Should replicate OR7a suppression effects

4. **Control Odor Validation**
   - Genotype: Or7a-GAL4 > UAS-Kir2.1
   - Train: Ethyl butyrate or 3-Octanol (low Or7a response)
   - Test: Same odor
   - Expect normal learning (LI ~0.7) - confirms OR7a-specificity

### Success Criteria

**Hypothesis confirmed if:**
- ✅ OR7a suppression eliminates benzaldehyde blocking (LI 0.2 → 0.6+)
- ✅ OR7a suppression eliminates hexanol cross-transfer (LI 0.5 → 0.2)
- ✅ DL5_adPN suppression replicates OR7a suppression
- ✅ Control odors unaffected by OR7a suppression

## Troubleshooting

### "DoOR data not available"

**Solution**: Initialize DoOR manager explicitly

```python
from src.pgcn.door.door_data_manager import DoORDataManager

# Try rpy2 first
try:
    door_mgr = DoORDataManager(method='rpy2')
except:
    # Fallback to CSV
    door_mgr = DoORDataManager(method='csv')
```

### "Pathway data not found"

**Cause**: OR7a pathway mapping not run yet

**Solution**: Run pathway mapping first

```bash
python scripts/map_or7a_complete_pathway.py --data-source local
```

### "Behavioral data format error"

**Cause**: Missing required columns

**Solution**: Ensure CSV has columns: `odor`, `training_odor`, `learning_index`, `condition`, `genotype`

## References

### FlyWire Connectome
- Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain. Nature.

### DoOR Database
- Münch & Galizia (2016). DoOR 2.0 - Comprehensive mapping of Drosophila olfactory responses. Scientific Reports.

### Olfactory Learning
- Tully & Quinn (1985). Classical conditioning and retention in normal and mutant Drosophila. Journal of Comparative Physiology A.

## Support

For questions or issues:
- Review pathway mapping guides: [`docs/OR7A_COMPLETE_PATHWAY_GUIDE.md`](OR7A_COMPLETE_PATHWAY_GUIDE.md)
- Check PGCN integration: [`docs/model_integration_status.md`](model_integration_status.md)
- Examine example scripts in [`scripts/`](../scripts/)

## Version History

- **v1.0** (2025-11-05): Initial release
  - Baseline behavioral analysis
  - Suppression predictions
  - DoOR integration
  - Experimental design generation
  - Complete visualization pipeline
