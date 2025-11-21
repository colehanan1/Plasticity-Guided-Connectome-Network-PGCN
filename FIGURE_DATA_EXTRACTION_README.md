# Figure Data Extraction Guide

## Overview

This guide explains how to extract data from your PGCN model outputs for the four key figures in your publication.

## Quick Start

```bash
# Extract all figure data
python extract_figure_data.py --task all

# Extract specific tasks
python extract_figure_data.py --task behavioral
python extract_figure_data.py --task schematic
python extract_figure_data.py --task synapse_map
python extract_figure_data.py --task ml_comparison
```

## Output Structure

All extracted data is saved to `data/extracted_figures/`:

```
data/extracted_figures/
├── behavioral_data.csv              # Figure 1: Behavioral predictions
├── behavioral_data_dict.pkl
├── model_schematic_info.yaml        # Figure 2: Model architecture
├── model_schematic_info.pkl
├── synapse_map_summary.csv          # Figure 3: Critical synapse maps
├── veto_mask.npy
├── synapse_map_data.pkl
├── ml_comparison_data.csv           # Figure 4: ML model comparison
└── ml_comparison_dict.pkl
```

## Detailed Task Descriptions

### Task 1: Behavioral Prediction Data

**Purpose:** Extract memory scores for different experimental groups across training phases.

**Expected input files:**
- `results/behavioral_sim/wildtype_behavioral.csv`
- `results/behavioral_sim/or7a_mutant_behavioral.csv`
- `results/behavioral_sim/control_behavioral.csv`

**Alternative formats:**
- Combined CSV: `results/behavioral_sim/behavioral_results.csv`
- Pickle files: `results/behavioral_sim/*.pkl`

**Output format:**
```python
{
    "wildtype": [0.82, 0.68, 0.55],      # [after_A_train, after_B_train, A_test]
    "or7a_mutant": [0.83, 0.69, 0.21],   # Shows catastrophic forgetting
    "control": [0.80, 0.80, 0.80]        # Stable baseline
}
```

**CSV columns required:**
- `group`: "wildtype", "or7a_mutant", or "control"
- `phase`: "after_A_train", "after_B_train", "A_test"
- `memory_score`: Float value (0-1)

---

### Task 2: Model Schematic Info

**Purpose:** Extract neuron counts and synapse statistics for architecture diagram.

**Expected input files:**
- `configs/penp_model_config.yaml` (architecture definition)
- `results/veto_mask.npy` (optional: protection mask)

**Output format:**
```yaml
n_pn: 50              # Projection Neurons
n_kc: 2000            # Kenyon Cells
n_mbon: 44            # Mushroom Body Output Neurons
n_synapses: 88000     # Total KC→MBON synapses
n_protected: 4400     # Protected synapses
protection_percentage: 5.0
```

**Config file structure:**
```yaml
pgcn_model:
  network_architecture:
    input_layer:
      size: 1756
      type: "olfactory"
    hidden_layer:
      size: 241
      type: "integration"
    output_layer:
      size: 165
      type: "motor"
```

---

### Task 3: Critical Synapse Map Data

**Purpose:** Extract veto gate protection masks for heatmap visualization.

**Expected input files:**
- Single mask: `results/veto_mask.npy`
- Multiple masks: `results/veto_mask_odorpair*.npy`

**Mask format:**
- 2D numpy array: shape `(n_KC, n_MBON)` or `(n_MBON, n_KC)`
- Binary values: 1 = protected, 0 = unprotected
- Example shape: `(2000, 44)`

**Output:**
- Summary table with protection statistics
- Individual mask files (`.npy` format)
- Combined pickle with all masks

---

### Task 4: ML Comparison Data

**Purpose:** Extract forgetting scores for different continual learning methods.

**Expected input files:**
- `results/forgetting_summary.csv`

**Alternative:**
- Individual files: `results/MBON_veto_results.csv`, etc.

**CSV format:**
```csv
model_type,forgetting_score
MBON_veto,0.15
Dense_ANN,0.82
EWC,0.45
SI,0.52
```

**Supported models:**
- `MBON_veto`: Or7a-inspired veto gate (your method)
- `Dense_ANN`: Standard dense neural network
- `EWC`: Elastic Weight Consolidation
- `SI`: Synaptic Intelligence
- `LwF`: Learning without Forgetting
- `GEM`: Gradient Episodic Memory

---

## Customization

### Updating File Paths

If your data is stored in different locations, edit `extract_figure_data.py`:

```python
# Example: Change behavioral data directory
def extract_behavioral_data(
    results_dir: str = "your/custom/path",  # ← Update here
    output_dir: str = "data/extracted_figures"
)
```

### Adding New Data Formats

The extraction functions check multiple file formats automatically. To add support for a new format:

1. Open `extract_figure_data.py`
2. Find the relevant extraction function
3. Add a new `Option X` section following the existing pattern

Example:
```python
# Option 3: HDF5 files
if not found_data:
    h5_file = results_path / "behavioral_data.h5"
    if h5_file.exists():
        import h5py
        with h5py.File(h5_file, 'r') as f:
            behavioral_data = dict(f['behavioral_scores'])
        found_data = True
```

### Adjusting Column Names

If your CSV files use different column names, look for `# TODO: check column name` comments in the code and update accordingly:

```python
# Example: Your CSV uses 'score' instead of 'memory_score'
if 'phase' in df.columns and 'score' in df.columns:  # ← Update here
    phase_scores = df.groupby('phase')['score'].mean().values
```

---

## Example Plotting Usage

After extracting data, use it in your plotting scripts:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 1. Behavioral Data (Figure 1)
df_behavioral = pd.read_csv('data/extracted_figures/behavioral_data.csv')
# Your plotting code...

# 2. Model Schematic (Figure 2)
import yaml
with open('data/extracted_figures/model_schematic_info.yaml', 'r') as f:
    schematic = yaml.safe_load(f)
print(f"KCs: {schematic['n_kc']}, MBONs: {schematic['n_mbon']}")

# 3. Synapse Map (Figure 3)
veto_mask = np.load('data/extracted_figures/veto_mask.npy')
plt.imshow(veto_mask, cmap='RdBu', aspect='auto')
plt.xlabel('MBONs')
plt.ylabel('Kenyon Cells')
plt.colorbar(label='Protected')
plt.show()

# 4. ML Comparison (Figure 4)
df_ml = pd.read_csv('data/extracted_figures/ml_comparison_data.csv')
df_ml.plot(x='model_type', y='forgetting_score', kind='bar')
plt.ylabel('Forgetting Score')
plt.show()
```

See `examples/plot_extracted_figures.py` for complete plotting examples.

---

## Troubleshooting

### "Results directory not found"

**Problem:** `results/` directory doesn't exist yet.

**Solution:**
1. Run your simulation/training scripts first to generate results
2. Or update the `results_dir` parameter to point to your actual output location

### "No compatible data files found"

**Problem:** Data files exist but don't match expected format.

**Solution:**
1. Check the expected formats in this README
2. Update column names in `extract_figure_data.py` (look for `# TODO` comments)
3. Add support for your specific format (see "Adding New Data Formats" above)

### "Unexpected veto mask shape"

**Problem:** Veto mask has wrong dimensions.

**Solution:**
1. Verify mask is 2D: `(n_KC, n_MBON)` or `(n_MBON, n_KC)`
2. Check if mask needs to be transposed: `mask = mask.T`
3. Update expected dimensions in config if using non-standard architecture

### "Placeholder data generated"

**Note:** This is not an error! The script generates realistic placeholder data when real data isn't found yet. This lets you:
1. Test your plotting pipeline before running expensive simulations
2. Verify the extraction logic works
3. Get started on figure layouts

Replace placeholder data with real results when available.

---

## Integration with Existing Scripts

If you already have data extraction code, you can:

1. **Call these functions from your scripts:**
   ```python
   from extract_figure_data import extract_behavioral_data, extract_ml_comparison_data

   # After training
   behavioral_data = extract_behavioral_data(
       results_dir="my_custom_results/",
       output_dir="data/figures/"
   )
   ```

2. **Import extracted data in analysis notebooks:**
   ```python
   import pickle

   with open('data/extracted_figures/behavioral_data_dict.pkl', 'rb') as f:
       behavioral_data = pickle.load(f)
   ```

---

## Next Steps

1. ✅ Run extraction: `python extract_figure_data.py --task all`
2. ✅ Verify outputs in `data/extracted_figures/`
3. ✅ Check for any `TODO` warnings in console output
4. ✅ Update file paths if needed
5. ✅ Create plotting scripts using extracted data
6. ✅ Generate publication-ready figures

---

## Contact

For issues or questions:
- Check `# TODO` comments in `extract_figure_data.py`
- Review expected formats in this README
- Ensure simulation scripts have run and generated outputs

Happy plotting! 📊
