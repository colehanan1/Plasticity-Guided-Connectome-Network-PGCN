# Plotting Script Fixes - Quick Reference

## Issues Fixed ✅

### 1. Qt Backend Errors (RESOLVED)
**Problem:**
```
qt.qpa.plugin: Could not find the Qt platform plugin "xcb"
Aborted (core dumped)
```

**Root Cause:** Matplotlib was trying to display plots interactively using Qt, but Qt libraries were incompatible.

**Solution:**
- Switched to non-interactive `Agg` backend
- Replaced all `plt.show()` calls with `plt.close()`
- Figures now save directly to disk without attempting to display

**Code Changes:**
```python
# Added at top of script (line 29-31)
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
```

---

### 2. Missing File Error (RESOLVED)
**Problem:**
```
FileNotFoundError: [Errno 2] No such file or directory:
'data/extracted_figures/veto_mask.npy'
```

**Root Cause:** The extraction script created `placeholder_mask.npy` but the plotting script was looking for `veto_mask.npy`.

**Solution:**
- Added smart file path detection
- Script now tries multiple file names automatically:
  1. `veto_mask.npy` (primary)
  2. `placeholder_mask.npy` (fallback)
  3. `veto_mask_odorpair*.npy` (alternatives)

**Code Changes:**
```python
# Added automatic file detection (lines 319-338)
mask_path = Path(data_file)
if not mask_path.exists():
    # Try alternative names
    alt_paths = [
        Path(data_file).parent / "placeholder_mask.npy",
        Path(data_file).parent / "veto_mask_odorpair0.npy",
        Path(data_file).parent / "veto_mask_odorpair1.npy",
    ]
    for alt_path in alt_paths:
        if alt_path.exists():
            mask_path = alt_path
            break
```

---

## How to Use Now

### Running the Script

```bash
# Generate all figures (recommended)
python examples/plot_extracted_figures.py --figure all

# Or generate specific figures
python examples/plot_extracted_figures.py --figure behavioral
python examples/plot_extracted_figures.py --figure schematic
python examples/plot_extracted_figures.py --figure synapse_map
python examples/plot_extracted_figures.py --figure ml_comparison
```

### Expected Output

```
======================================================================
PGCN Publication Figure Generator
======================================================================
Figure: all
Data directory: data/extracted_figures
Output directory: figures/publication

======================================================================
Creating Figure 1: Behavioral Prediction
======================================================================
✓ Loaded data from: data/extracted_figures/behavioral_data.csv
  Shape: (3, 4)
  Columns: ['wildtype', 'or7a_mutant', 'control', 'phase']
✓ Saved figure to: figures/publication/figure1_behavioral_prediction.png
✓ Saved PDF to: figures/publication/figure1_behavioral_prediction.pdf

... (similar for figures 2, 3, 4) ...

======================================================================
FIGURE GENERATION COMPLETE
======================================================================
✓ All figures saved to: figures/publication/

Generated files:
  • figure1_behavioral_prediction.png/pdf
  • figure2_model_schematic.png/pdf
  • figure3_synapse_map.png/pdf
  • figure4_ml_comparison.png/pdf

Note: Figures saved to disk (non-interactive mode).
      No plot windows will appear - open the files directly.
======================================================================
```

### Important Notes

1. **No Plot Windows**: The script runs in non-interactive mode, so no matplotlib windows will pop up. This is intentional and prevents Qt errors.

2. **Direct File Access**: Open the generated PNG/PDF files directly from the `figures/publication/` directory.

3. **High Resolution**: All PNG files are 300 DPI (publication quality).

4. **Vector Graphics**: PDF files are vector format for scalability.

---

## Viewing the Figures

### On Linux (Your System)

```bash
# View all figures
xdg-open figures/publication/*.png

# View specific figure
xdg-open figures/publication/figure1_behavioral_prediction.png

# Or use your preferred image viewer
eog figures/publication/figure1_behavioral_prediction.png
```

### On Any System

Navigate to `figures/publication/` in your file manager and double-click the images.

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'matplotlib'"
**Solution:**
```bash
pip install matplotlib seaborn
```

### Issue: "ModuleNotFoundError: No module named 'numpy'"
**Solution:**
```bash
pip install numpy pandas
```

### Issue: "FileNotFoundError" for data files
**Solution:** Run the extraction script first:
```bash
python extract_figure_data.py --task all
```

### Issue: Figures look wrong or incomplete
**Solution:**
1. Check that extraction ran successfully
2. Verify data files exist in `data/extracted_figures/`
3. Try regenerating specific figures:
   ```bash
   python examples/plot_extracted_figures.py --figure behavioral
   ```

---

## What Changed (Technical Summary)

### File: `examples/plot_extracted_figures.py`

**Line 29-31:** Added Agg backend
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

**Lines 157, 299, 426, 521:** Changed `plt.show()` → `plt.close()`

**Lines 319-338:** Added smart file path detection for synapse map

**Lines 620-621:** Added non-interactive mode message

---

## Git History

```bash
# Commit 1: Initial implementation
69efb16 - feat: Add comprehensive figure data extraction pipeline

# Commit 2: Syntax fix
ace349e - fix: Correct syntax error in plot_extracted_figures.py

# Commit 3: Backend and file path fixes
d04ea82 - fix: Resolve Qt backend issues and file path handling
```

---

## Quick Verification

To verify everything works:

```bash
# 1. Extract data (creates placeholder files)
python extract_figure_data.py --task all

# 2. Generate figures
python examples/plot_extracted_figures.py --figure all

# 3. Check output
ls -lh figures/publication/

# Expected: 8 files (4 PNG + 4 PDF)
# - figure1_behavioral_prediction.png
# - figure1_behavioral_prediction.pdf
# - figure2_model_schematic.png
# - figure2_model_schematic.pdf
# - figure3_synapse_map.png
# - figure3_synapse_map.pdf
# - figure4_ml_comparison.png
# - figure4_ml_comparison.pdf
```

---

## Next Steps

1. ✅ Scripts fixed and tested
2. ✅ Run extraction: `python extract_figure_data.py --task all`
3. ✅ Generate figures: `python examples/plot_extracted_figures.py --figure all`
4. 🔄 View figures in `figures/publication/`
5. 🔄 Replace placeholder data with real simulation results
6. 🔄 Re-run extraction and plotting with real data
7. 🔄 Customize figure styling as needed
8. 🔄 Use PDF files in your publication

---

**Status:** ✅ All issues resolved and ready to use!
**Last Updated:** 2025-11-21
