# Quick Start: Generating Publication Figures

## 🎯 Goal
Generate 4 publication-ready figures from your PGCN model with real or placeholder data.

---

## ⚡ Quick Start (3 Steps)

### Step 1: Generate Data from Your Model

**Option A: Use Template (Recommended for first time)**
```bash
# Edit the template script with your experiments
nano scripts/generate_figure_data.py

# Run it to save data in the right format
python scripts/generate_figure_data.py
```

**Option B: Use Placeholder Data (Testing)**
```bash
# Skip Step 1, extraction will create placeholder data automatically
```

---

### Step 2: Extract Data
```bash
python extract_figure_data.py --task all
```

Expected output:
```
✓ Saved behavioral data to: data/extracted_figures/behavioral_data.csv
✓ Saved schematic info to: data/extracted_figures/model_schematic_info.yaml
✓ Saved synapse map to: data/extracted_figures/synapse_map_summary.csv
✓ Saved ML comparison to: data/extracted_figures/ml_comparison_data.csv
```

---

### Step 3: Generate Figures
```bash
python examples/plot_extracted_figures.py --figure all
```

Expected output:
```
✓ Saved figure to: figures/publication/figure1_behavioral_prediction.png
✓ Saved PDF to: figures/publication/figure1_behavioral_prediction.pdf
... (similar for figures 2, 3, 4) ...
```

---

## 📊 What You Get

Four publication-ready figures in `figures/publication/`:

1. **figure1_behavioral_prediction.png/pdf**
   - Line plot showing memory retention across training phases
   - Compares wildtype vs or7a_mutant vs control
   - Highlights catastrophic forgetting

2. **figure2_model_schematic.png/pdf**
   - Architecture diagram (PN → KC → MBON)
   - Shows neuron counts and veto gate mechanism
   - Includes plasticity rule equation

3. **figure3_synapse_map.png/pdf**
   - Heatmap of protected vs unprotected synapses
   - Statistics panel showing protection percentage
   - Color-coded visualization

4. **figure4_ml_comparison.png/pdf**
   - Bar chart comparing continual learning methods
   - Shows forgetting scores (lower = better)
   - Highlights your MBON_veto performance

---

## 📁 Data Required

### Figure 1: Behavioral Experiments
**File:** `results/behavioral_sim/{wildtype,or7a_mutant,control}_behavioral.csv`

**Format:**
```csv
phase,memory_score
after_A_train,0.85
after_B_train,0.72
A_test,0.68
```

**How to generate:**
```python
# After running your learning experiment
import pandas as pd

data = [
    {'phase': 'after_A_train', 'memory_score': 0.85},
    {'phase': 'after_B_train', 'memory_score': 0.72},
    {'phase': 'A_test', 'memory_score': 0.68}
]
df = pd.DataFrame(data)
df.to_csv('results/behavioral_sim/wildtype_behavioral.csv', index=False)
```

---

### Figures 2 & 3: Veto Gate Mask
**File:** `results/veto_mask.npy`

**Format:** 2D numpy array (n_KC × n_MBON), binary values (1=protected, 0=unprotected)

**How to generate:**
```python
# After training and identifying critical pathways
import numpy as np

# From your veto gate
veto_mask = veto_gate.protection_mask  # Shape: (2000, 44)
np.save('results/veto_mask.npy', veto_mask)
```

---

### Figure 4: ML Comparison
**File:** `results/forgetting_summary.csv`

**Format:**
```csv
model_type,forgetting_score
MBON_veto,0.15
Dense_ANN,0.82
EWC,0.45
```

**How to generate:**
```python
# After running benchmarks
import pandas as pd

scores = {
    'MBON_veto': 0.15,
    'Dense_ANN': 0.82,
    'EWC': 0.45
}

df = pd.DataFrame([
    {'model_type': k, 'forgetting_score': v}
    for k, v in scores.items()
])
df.to_csv('results/forgetting_summary.csv', index=False)
```

---

## 🔧 Customization

### Change Figure Style
Edit `examples/plot_extracted_figures.py`:

```python
# Colors (line 50)
COLORS = {
    'wildtype': '#0173B2',      # Change to your preference
    'or7a_mutant': '#DE8F05',
    'control': '#029E73',
}

# Font size (line 41)
plt.rcParams['font.size'] = 11  # Increase for posters

# DPI (line 46)
plt.rcParams['figure.dpi'] = 300  # Increase for higher resolution
```

### Change Data Paths
Edit `extract_figure_data.py` or use command line:

```bash
python extract_figure_data.py \
  --task behavioral \
  --output-dir my_custom_output/
```

---

## 📚 Documentation

- **EXTRACT_REAL_DATA_GUIDE.md** - Complete guide with code examples
- **FIGURE_DATA_EXTRACTION_README.md** - Detailed extraction documentation
- **PLOTTING_FIXES.md** - Troubleshooting guide
- **EXTRACTION_SUMMARY.md** - Technical implementation details

---

## 🐛 Troubleshooting

### "No such file or directory: results/..."
**Solution:** Run `python scripts/generate_figure_data.py` first, or let extraction create placeholder data

### "Qt platform plugin error"
**Solution:** ✅ Already fixed! Script uses non-interactive backend

### "FileNotFoundError: veto_mask.npy"
**Solution:** ✅ Already fixed! Script looks for placeholder_mask.npy automatically

### "ModuleNotFoundError: numpy"
**Solution:**
```bash
pip install numpy pandas matplotlib seaborn pyyaml
```

---

## 💡 Tips

1. **Start with placeholder data** to test the pipeline before running long experiments
2. **Customize one figure at a time** using `--figure behavioral` instead of `--figure all`
3. **Use PDF files** for publications (vector graphics, infinite zoom)
4. **Check data extracted** by inspecting `data/extracted_figures/` before plotting
5. **Edit template script** at `scripts/generate_figure_data.py` with your specific experiments

---

## 🚀 Full Workflow Example

```bash
# 1. Edit template with your experiments
nano scripts/generate_figure_data.py

# 2. Generate real data
python scripts/generate_figure_data.py

# Output:
# ✓ Saved wildtype: results/behavioral_sim/wildtype_behavioral.csv
# ✓ Saved veto mask: results/veto_mask.npy
# ✓ Saved ML comparison: results/forgetting_summary.csv

# 3. Extract for figures
python extract_figure_data.py --task all

# Output:
# ✓ All extracted data saved to: data/extracted_figures/

# 4. Generate figures
python examples/plot_extracted_figures.py --figure all

# Output:
# ✓ All figures saved to: figures/publication/

# 5. View results
xdg-open figures/publication/figure1_behavioral_prediction.png
```

---

## ✅ Checklist

- [ ] Install dependencies: `pip install numpy pandas matplotlib seaborn pyyaml`
- [ ] Edit template: `scripts/generate_figure_data.py`
- [ ] Generate data: `python scripts/generate_figure_data.py`
- [ ] Extract data: `python extract_figure_data.py --task all`
- [ ] Create figures: `python examples/plot_extracted_figures.py --figure all`
- [ ] Check output: `ls figures/publication/`
- [ ] Customize styling if needed
- [ ] Use PDF files in your manuscript

---

## 📞 Need Help?

1. Read detailed guides in documentation files
2. Check `# TODO` comments in `scripts/generate_figure_data.py`
3. Verify file formats match examples in guides
4. Test with placeholder data first

---

**You're ready to create publication figures!** 🎉

Last Updated: 2025-11-21
