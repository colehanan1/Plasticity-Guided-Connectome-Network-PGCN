# Figure Extraction Pipeline - Complete Implementation

## 🎉 What Was Created

A complete end-to-end pipeline for generating publication-ready figures from your PGCN model.

---

## 📦 Files Created

### Core Scripts (2 files)
1. **`extract_figure_data.py`** (~700 lines)
   - Main data extraction pipeline
   - Handles 4 figure types
   - Multi-format support (CSV, pickle, numpy, YAML)
   - Automatic placeholder generation

2. **`examples/plot_extracted_figures.py`** (~600 lines)
   - Publication-quality plotting
   - 300 DPI PNG + vector PDF output
   - Colorblind-safe palettes
   - Non-interactive mode (no Qt errors)

### Template & Helpers (1 file)
3. **`scripts/generate_figure_data.py`** (~200 lines)
   - Ready-to-customize template
   - Shows how to save your experiment data
   - Simple helper functions
   - Clear TODO markers

### Documentation (5 files)
4. **`QUICKSTART_FIGURES.md`** - Start here! ⭐
   - 3-step workflow
   - Data format examples
   - Quick reference

5. **`EXTRACT_REAL_DATA_GUIDE.md`** - Comprehensive guide
   - Complete code examples for all figures
   - Full integration example
   - 600+ lines of documentation

6. **`FIGURE_DATA_EXTRACTION_README.md`** - User manual
   - Detailed extraction documentation
   - File format specifications
   - Customization guide

7. **`PLOTTING_FIXES.md`** - Troubleshooting
   - Qt backend error fixes
   - File path handling
   - Common issues & solutions

8. **`EXTRACTION_SUMMARY.md`** - Technical docs
   - Implementation details
   - Design decisions
   - Integration examples

---

## 🚀 How to Extract Real Data (Summary)

### The 3-Step Process

#### Step 1: Save Your Experiment Data

Edit `scripts/generate_figure_data.py` to save your results:

```python
# Figure 1: Behavioral scores
behavioral_results = {
    'wildtype': [after_A, after_B, test_A],
    'or7a_mutant': [after_A, after_B, test_A],
    'control': [0.5, 0.5, 0.5]
}
save_behavioral_data_simple(behavioral_results)

# Figure 2 & 3: Veto mask
np.save('results/veto_mask.npy', veto_gate.protection_mask)

# Figure 4: ML comparison
ml_scores = {
    'MBON_veto': forgetting_score,
    'Dense_ANN': forgetting_score,
    ...
}
save_ml_comparison_simple(ml_scores)
```

Run it:
```bash
python scripts/generate_figure_data.py
```

#### Step 2: Extract for Figures

```bash
python extract_figure_data.py --task all
```

#### Step 3: Generate Figures

```bash
python examples/plot_extracted_figures.py --figure all
```

Done! Check `figures/publication/` for your figures.

---

## 📊 What Data You Need

### Figure 1: Behavioral Prediction
- **File:** `results/behavioral_sim/{group}_behavioral.csv`
- **Columns:** `phase`, `memory_score`
- **Groups:** wildtype, or7a_mutant, control
- **Phases:** after_A_train, after_B_train, A_test

### Figure 2: Model Schematic
- **File:** `configs/penp_model_config.yaml` (already exists!)
- **Plus:** `results/veto_mask.npy` (optional)

### Figure 3: Synapse Protection Map
- **File:** `results/veto_mask.npy`
- **Format:** 2D numpy array (n_KC × n_MBON)
- **Values:** Binary (1=protected, 0=unprotected)

### Figure 4: ML Comparison
- **File:** `results/forgetting_summary.csv`
- **Columns:** `model_type`, `forgetting_score`
- **Models:** MBON_veto, Dense_ANN, EWC, SI, LwF, GEM

---

## 💡 Quick Integration Examples

### From Your Training Script

```python
# After running LearningExperiment
from pathlib import Path
import pandas as pd
import numpy as np

# Save behavioral results
data = []
for phase, score in [('after_A_train', 0.85), ('after_B_train', 0.72), ('A_test', 0.68)]:
    data.append({'phase': phase, 'memory_score': score})

Path('results/behavioral_sim').mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(data)
df.to_csv('results/behavioral_sim/wildtype_behavioral.csv', index=False)
```

### From Your Veto Gate Experiments

```python
# After identify_critical_pathways
import numpy as np

np.save('results/veto_mask.npy', veto_gate.protection_mask)
```

### From Your Benchmarks

```python
# After continual learning experiments
import pandas as pd

def compute_forgetting(before, after):
    return (before - after) / before

scores = {
    'MBON_veto': compute_forgetting(perf_A_before, perf_A_after),
    'Dense_ANN': compute_forgetting(...),
}

df = pd.DataFrame([{'model_type': k, 'forgetting_score': v} for k, v in scores.items()])
df.to_csv('results/forgetting_summary.csv', index=False)
```

---

## ✅ What's Fixed

### Qt Backend Errors ✅
- Switched to non-interactive `Agg` backend
- No more "Qt platform plugin" crashes
- Figures save directly to disk

### File Path Issues ✅
- Smart detection for placeholder_mask.npy
- Tries multiple file locations automatically
- Clear error messages

### Syntax Errors ✅
- All parenthesis/bracket mismatches fixed
- Code tested and working

---

## 📚 Documentation Roadmap

**Just getting started?**
→ Read `QUICKSTART_FIGURES.md`

**Need code examples?**
→ Read `EXTRACT_REAL_DATA_GUIDE.md`

**Want full details?**
→ Read `FIGURE_DATA_EXTRACTION_README.md`

**Having issues?**
→ Read `PLOTTING_FIXES.md`

**Want technical details?**
→ Read `EXTRACTION_SUMMARY.md`

---

## 🎯 Next Steps

1. **Test with placeholder data** (works now!)
   ```bash
   python extract_figure_data.py --task all
   python examples/plot_extracted_figures.py --figure all
   ```

2. **Edit template script**
   ```bash
   nano scripts/generate_figure_data.py
   # Add your experiments in the TODO sections
   ```

3. **Run with real data**
   ```bash
   python scripts/generate_figure_data.py
   python extract_figure_data.py --task all
   python examples/plot_extracted_figures.py --figure all
   ```

4. **Customize figures** (optional)
   - Edit colors in `examples/plot_extracted_figures.py`
   - Adjust font sizes for posters vs papers
   - Modify annotations and labels

5. **Use in publication**
   - PNG files: For presentations, preprints
   - PDF files: For final publication submission

---

## 🔧 Command Reference

```bash
# Generate your data
python scripts/generate_figure_data.py

# Extract all figures
python extract_figure_data.py --task all

# Extract specific figure
python extract_figure_data.py --task behavioral

# Generate all figures
python examples/plot_extracted_figures.py --figure all

# Generate specific figure
python examples/plot_extracted_figures.py --figure synapse_map

# View results
xdg-open figures/publication/*.png
```

---

## 📊 Output Files

After running the full pipeline:

```
results/
├── behavioral_sim/
│   ├── wildtype_behavioral.csv
│   ├── or7a_mutant_behavioral.csv
│   └── control_behavioral.csv
├── veto_mask.npy
└── forgetting_summary.csv

data/extracted_figures/
├── behavioral_data.csv
├── behavioral_data_dict.pkl
├── model_schematic_info.yaml
├── model_schematic_info.pkl
├── synapse_map_summary.csv
├── placeholder_mask.npy (or veto_mask.npy)
├── synapse_map_data.pkl
├── ml_comparison_data.csv
└── ml_comparison_dict.pkl

figures/publication/
├── figure1_behavioral_prediction.png (300 DPI)
├── figure1_behavioral_prediction.pdf (vector)
├── figure2_model_schematic.png
├── figure2_model_schematic.pdf
├── figure3_synapse_map.png
├── figure3_synapse_map.pdf
├── figure4_ml_comparison.png
└── figure4_ml_comparison.pdf
```

---

## 🎉 You're All Set!

The complete pipeline is ready to use:
- ✅ All scripts working
- ✅ All bugs fixed
- ✅ Comprehensive documentation
- ✅ Template for your experiments
- ✅ Tested with placeholder data

**Start here:** `QUICKSTART_FIGURES.md`

**Questions?** Check the relevant documentation file above.

---

Last Updated: 2025-11-21
Branch: `claude/extract-figure-data-01Vw8efimVKGeUtpSEw1kbrB`
