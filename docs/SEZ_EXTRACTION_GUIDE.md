# SEZ Neuron Extraction Pipeline - Complete Guide

## Executive Summary

This guide documents the **SEZ (Subesophageal Zone) neuron extraction pipeline** for the PGCN (Plasticity-Guided Connectome Network) model. The pipeline extracts taste-responsive neurons from FlyWire FAFB v783 connectomic data, implementing methods from Li et al. (2024) and Shen et al. (2025).

**Key Outcomes:**
- Extract **100-200 SEZ Projection Neurons (SEZ-PNs)** - taste input to mushroom body/lateral horn
- Extract **200-600 SEZ Local Interneurons (SEZ-LNs)** - local taste processing
- Extract **50-100 cholinergic SEZ-LNs** - excitatory relay circuits
- Validate extraction against published literature using clustering analysis

---

## Table of Contents

1. [Scientific Background](#scientific-background)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [Pipeline Overview](#pipeline-overview)
5. [Detailed Usage](#detailed-usage)
6. [Validation & Quality Control](#validation--quality-control)
7. [Integration with PGCN Model](#integration-with-pgcn-model)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

---

## Scientific Background

### Why Add SEZ Neurons to PGCN?

The current PGCN model focuses on **olfactory learning** through the mushroom body. However, real flies integrate **taste and smell** for food-related decisions and learned associations. Adding SEZ taste neurons enables:

1. **Cross-modal sensory integration** - taste + smell interactions
2. **Biologically realistic learning** - taste as US (unconditioned stimulus)
3. **Taste-odor blocking experiments** - extension of Experiment 1
4. **Complete sensory context** - SEZ-PNs project to MB/LH (same targets as olfactory PNs)

### Neuron Types Extracted

#### 1. Gustatory Receptor Neurons (GRNs) - Ground Truth
- **Count:** 343 neurons (pre-validated from FlyWire Codex)
- **Function:** Peripheral taste sensors (sweet, bitter, water, etc.)
- **Location:** Labellum, pharynx, tarsi
- **Data source:** `data/flywire/root_ids_class_gustatory.txt`

#### 2. SEZ Projection Neurons (SEZ-PNs)
- **Expected count:** 100-200 neurons (Li et al. 2024)
- **Function:** Relay taste information to higher brain centers
- **Targets:** Lateral horn (LH), mushroom body (MB), superior lateral protocerebrum
- **Classification:** `super_class` contains "ascending" or "sensory"
- **Connectivity:** GRN → SEZ-PN → LH/MB

#### 3. SEZ Local Interneurons (SEZ-LNs)
- **Expected count:** 200-600 neurons
- **Function:** Local taste processing within SEZ
- **Classification:** `super_class` contains "intrinsic"
- **Connectivity:** GRN → SEZ-LN → SEZ-PN (feedforward circuits)

#### 4. Cholinergic SEZ-LNs (Relay Subset)
- **Expected count:** 50-100 neurons (Shen et al. 2025)
- **Function:** Excitatory relay pathways
- **Neurotransmitter:** Acetylcholine
- **Role:** Organizing nodes for taste modality separation

---

## Installation & Setup

### Prerequisites

```bash
# Python 3.10+
# Conda environment (recommended)
conda create -n pgcn python=3.10
conda activate pgcn

# Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn

# Optional: for UMAP visualization
pip install umap-learn

# Optional: for testing
pip install pytest
```

### Data Requirements

The pipeline requires FlyWire FAFB v783 data files in `data/flywire/`:

- `connections_princeton.csv.gz` - Full connectome (~5.3M connections)
- `classification.csv.gz` - Hierarchical neuron classification
- `consolidated_cell_types.csv.gz` - Cell type annotations
- `neurons.csv.gz` - Neurotransmitter predictions
- **`root_ids_class_gustatory.txt`** - **343 validated GRN root IDs** (required)

**If GRN file is missing:** The script can extract GRNs from `classification.csv.gz` as a fallback, but using the pre-validated list ensures consistency with published analyses.

---

## Quick Start

### Step 1: Verify Data Files

```bash
# Check that data directory exists
ls -lh data/flywire/

# Verify GRN file (343 lines expected)
wc -l data/flywire/root_ids_class_gustatory.txt
```

### Step 2: Run Extraction Pipeline

```bash
# Basic extraction (with validation)
python scripts/extract_sez_neurons.py

# Specify custom directories
python scripts/extract_sez_neurons.py \
    --dataset-dir data/flywire \
    --output-dir data/cache \
    --validation-dir results/sez_validation

# Skip validation (faster)
python scripts/extract_sez_neurons.py --skip-validation

# Adjust synapse threshold
python scripts/extract_sez_neurons.py --min-synapses 5
```

### Step 3: Verify Results

```bash
# Check output files
ls -lh data/cache/sez_*.csv
ls -lh results/sez_validation/*.pdf

# Run test suite
pytest tests/test_sez_extraction.py -v

# View summary
python scripts/summarize_all_cell_types.py
```

---

## Pipeline Overview

The extraction pipeline consists of **6 stages**:

```
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: Load Validated GRN Root IDs                    │
│ Input:  root_ids_class_gustatory.txt (343 neurons)      │
│ Output: grn_ids array                                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: Trace Second-Order Neurons                     │
│ Method: Find neurons receiving ≥10 synapses from GRNs   │
│ Output: ~400-800 second-order taste neurons              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 3: Classify Projection vs Local                   │
│ Method: Filter by super_class (ascending vs intrinsic)  │
│ Output: SEZ-PNs (~100-200), SEZ-LNs (~200-600)          │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 4: Filter Cholinergic SEZ-LNs                     │
│ Method: Filter SEZ-LNs by neurotransmitter prediction   │
│ Output: Cholinergic SEZ-LNs (~50-100)                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 5: Clustering Validation (Li et al. 2024)         │
│ 1. Build GRN→SEZ-PN connectivity matrix                 │
│ 2. L2 normalization                                      │
│ 3. TruncatedSVD (10 components)                          │
│ 4. Hierarchical clustering (correlation distance)       │
│ 5. Silhouette score optimization                        │
│ Output: 8-12 clusters (taste modalities)                │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 6: Export Results                                 │
│ - sez_pn_all.csv (projection neurons)                   │
│ - sez_ln_all.csv (local interneurons)                   │
│ - sez_ln_cholinergic.csv (cholinergic subset)           │
│ - validation_summary.json (metrics)                     │
│ - fig1-4 PDFs (validation plots)                        │
└─────────────────────────────────────────────────────────┘
```

---

## Detailed Usage

### Command-Line Arguments

```
python scripts/extract_sez_neurons.py [OPTIONS]

Required Data:
  --dataset-dir PATH      FlyWire data directory (default: data/flywire)
                          Must contain: connections_princeton.csv.gz,
                                       classification.csv.gz,
                                       neurons.csv.gz,
                                       root_ids_class_gustatory.txt

Output Locations:
  --output-dir PATH       Neuron CSV output directory (default: data/cache)
  --validation-dir PATH   Validation plots directory (default: results/sez_validation)

Extraction Parameters:
  --min-synapses INT      Minimum synapses for GRN→2nd connections (default: 10)
                          Lower values = more lenient (may include weak connections)
                          Higher values = stricter (may miss some neurons)

Processing Options:
  --skip-validation       Skip Li et al. (2024) clustering validation
                          (Faster, but no quality metrics)

Examples:
  # Standard extraction with validation
  python scripts/extract_sez_neurons.py

  # Quick extraction for testing
  python scripts/extract_sez_neurons.py --skip-validation

  # More lenient synapse threshold
  python scripts/extract_sez_neurons.py --min-synapses 5

  # Custom output locations
  python scripts/extract_sez_neurons.py \
      --output-dir results/neurons \
      --validation-dir results/validation
```

### Output Files

#### Neuron CSVs (data/cache/)

**sez_pn_all.csv** - SEZ Projection Neurons
```csv
root_id,super_class,class,sub_class,cell_type,side,...
720575940610469873,ascending,visual_projection,optic_lobe,SEZ_PN,right,...
```

**sez_ln_all.csv** - SEZ Local Interneurons
```csv
root_id,super_class,class,sub_class,cell_type,side,...
720575940615435208,intrinsic,local_interneuron,SEZ,SEZ_LN,center,...
```

**sez_ln_cholinergic.csv** - Cholinergic SEZ-LNs
```csv
root_id,super_class,nt_type,cell_type,neurotransmitter,side,...
720575940615435208,intrinsic,acetylcholine,SEZ_LN_cholinergic,Acetylcholine,center,...
```

#### Validation Files (results/sez_validation/)

**validation_summary.json** - Quantitative Metrics
```json
{
  "n_sez_pns": 142,
  "n_grns": 343,
  "n_clusters": 10,
  "silhouette_score": 0.387,
  "variance_explained": 0.763,
  "li2024_validation": {
    "expected_pn_count": "100-200",
    "actual_pn_count": 142,
    "within_range": true,
    "expected_clusters": "8-12",
    "actual_clusters": 10,
    "cluster_match": true
  }
}
```

**fig1_dendrogram.pdf** - Hierarchical clustering dendrogram
**fig2_silhouette.pdf** - Optimal cluster count selection
**fig3_umap_clusters.pdf** - 2D UMAP embedding (if umap-learn installed)
**fig4_heatmap.pdf** - Pairwise distance matrix

**sez_pn_clusters.csv** - Cluster assignments
```csv
root_id,cluster
720575940610469873,0
720575940615782341,0
720575940621047892,1
```

---

## Validation & Quality Control

### Automated Tests

Run the comprehensive test suite:

```bash
# Run all tests with verbose output
pytest tests/test_sez_extraction.py -v

# Run specific test classes
pytest tests/test_sez_extraction.py::TestGRNGroundTruth -v
pytest tests/test_sez_extraction.py::TestSEZPNExtraction -v
pytest tests/test_sez_extraction.py::TestClusteringValidation -v

# Generate test report
pytest tests/test_sez_extraction.py --html=test_report.html
```

### Expected Test Results

```
tests/test_sez_extraction.py::TestGRNGroundTruth
  ✓ test_grn_file_exists - Verify GRN file present
  ✓ test_grn_count - Validate 343 GRNs
  ✓ test_grn_format - Check valid root IDs

tests/test_sez_extraction.py::TestSEZPNExtraction
  ✓ test_sez_pn_file_exists - Output file created
  ✓ test_sez_pn_count_li2024 - Count within 100-200 range
  ✓ test_sez_pn_required_columns - Has root_id, cell_type
  ✓ test_sez_pn_no_duplicates - No duplicate neurons

tests/test_sez_extraction.py::TestSEZLNExtraction
  ✓ test_sez_ln_file_exists - Output file created
  ✓ test_sez_ln_count_plausible - Count within 200-600 range
  ✓ test_cholinergic_sez_ln_extraction - 50-100 cholinergic

tests/test_sez_extraction.py::TestClusteringValidation
  ✓ test_cluster_file_exists - Cluster assignments saved
  ✓ test_clustering_produces_meaningful_groups - 8-12 clusters
  ✓ test_validation_summary_exists - JSON summary created
  ✓ test_silhouette_score_quality - Score ≥ 0.3

tests/test_sez_extraction.py::TestDataQuality
  ✓ test_no_overlap_between_pns_and_lns - No overlap
  ✓ test_cholinergic_lns_subset_of_all_lns - Subset relationship
  ✓ test_extraction_consistency - Proportions reasonable
```

### Quality Metrics

**SEZ-PN Count Validation:**
- ✅ **100-200 neurons:** Exact match with Li et al. (2024)
- ⚠ **80-99 neurons:** Slightly low (acceptable - more stringent filtering)
- ⚠ **201-250 neurons:** Slightly high (may include some LNs)
- ❌ **<80 or >250:** Outside plausible range - check extraction logic

**Clustering Quality (Silhouette Score):**
- ✅ **≥0.7:** Strong cluster separation
- ✅ **0.5-0.7:** Reasonable separation
- ✅ **0.3-0.5:** Weak but acceptable separation
- ⚠ **0.2-0.3:** Marginal quality
- ❌ **<0.2:** Poor clustering - review methods

**Cluster Count:**
- ✅ **8-12 clusters:** Matches Li et al. (2024) taste modalities
- ⚠ **6-7 or 13-15:** Acceptable variation
- ❌ **<6 or >15:** Unexpected - check connectivity matrix

---

## Integration with PGCN Model

### Updated Neuron Counts

After SEZ extraction, the PGCN model includes:

```
Current System (14,629 neurons)
├─ Core Components
│  ├─ Projection Neurons (PNs): 482
│  ├─ Kenyon Cells (KCs): 5,177
│  ├─ MBONs: 96
│  └─ DANs: 584
├─ Extended Components
│  ├─ Local Interneurons (LNs): 3,829
│  ├─ Lateral Horn (LH): 1,162
│  ├─ Motor Neurons: 66
│  ├─ Ascending Neurons (ANs): 1,926
│  └─ Descending Neurons (DNs): 1,303
└─ New Components
   ├─ CB0191 neurons: 2
   ├─ SEZ-NSC^CAPA: 2
   ├─ SEZ-PNs: ~142        ← NEW
   └─ SEZ-LNs: ~231        ← NEW

New Total: ~15,002 neurons
```

### Verify Integration

```bash
# Run summary script to see updated counts
python scripts/summarize_all_cell_types.py

# Expected output:
# ================================================================================
# PGCN OLFACTORY SYSTEM MODEL - COMPLETE CELL TYPE INVENTORY
# ================================================================================
# Total Neurons: 15,002
# ...
# New Components
# ----------------------------------------
#
# 1. CB0191 Neurons
#    Count: 2
#    ...
#
# 2. SEZ-NSC^CAPA Neurons
#    Count: 2
#    ...
#
# 3. SEZ Projection Neurons (SEZ-PNs)
#    Count: 142 [Expected: 100-200]
#    Role: Taste-responsive input to lateral horn and mushroom body
#    Reference: Li et al. (2024) Scientific Reports 14:21120
#
# 4. SEZ Local Interneurons (SEZ-LNs)
#    Count: 231 [Expected: 200-600]
#    Role: Local taste processing and relay within SEZ
#    Reference: Shen et al. (2025) Current Biology 35(9):1955-1970
```

### New Circuit Pathways

The SEZ neurons enable taste-odor integration:

```
GRN (taste receptor)
  ↓
SEZ-LN (local processing)
  ↓
SEZ-PN (projection neuron)
  ↓
┌─────────────┬─────────────┐
↓             ↓             ↓
LH            MB            Other
(innate)      (learning)    (context)

Integration with olfactory circuit:
PN (olfactory) + SEZ-PN (taste) → KC → MBON → Behavior
```

---

## Troubleshooting

### Issue 1: GRN File Not Found

**Error:**
```
FileNotFoundError: GRN file not found: data/flywire/root_ids_class_gustatory.txt
Expected: data/flywire/root_ids_class_gustatory.txt
This file should contain 343 validated GRN root IDs (one per line).
```

**Solution:**

1. **Check file location:**
   ```bash
   ls -lh data/flywire/root_ids_class_gustatory.txt
   ```

2. **If file exists elsewhere, move it:**
   ```bash
   mv /path/to/root_ids_class_gustatory.txt data/flywire/
   ```

3. **If file is missing, script will extract GRNs automatically:**
   - The script includes a fallback method to extract GRNs from `classification.csv.gz`
   - This searches for neurons with `super_class="sensory"` and `class="gustatory"`
   - Extracted IDs will be saved to the expected location

### Issue 2: Zero Second-Order Neurons Found

**Error:**
```
ERROR: No second-order neurons found
Try reducing --min-synapses (current: 10)
```

**Diagnosis:**
- The synapse threshold (--min-synapses) is too high
- Or GRNs are not well-connected in this dataset version

**Solution:**
```bash
# Try lower threshold (more lenient)
python scripts/extract_sez_neurons.py --min-synapses 5

# Or even more lenient for exploratory analysis
python scripts/extract_sez_neurons.py --min-synapses 3
```

### Issue 3: SEZ-PN Count Outside Expected Range

**Warning:**
```
⚠ WARNING: Found 50 SEZ-PNs (expected 100-200)
```

**Diagnosis:**
- Classification labels may differ in your FlyWire version
- Synapse threshold may be too high
- Some neurons may be labeled with different `super_class` values

**Solution:**

1. **Check classification distribution:**
   ```python
   import pandas as pd
   classification = pd.read_csv("data/flywire/classification.csv.gz")
   print(classification['super_class'].value_counts())
   ```

2. **Adjust extraction logic:**
   - Edit `scripts/extract_sez_neurons.py`
   - In `classify_sez_neurons()` function, expand classification filters:
   ```python
   # Current filter (line ~290):
   sez_pns = second_order_meta[
       second_order_meta['super_class'].str.contains(
           'ascending|sensory', case=False, na=False
       )
   ].copy()

   # Try broader filter:
   sez_pns = second_order_meta[
       second_order_meta['super_class'].str.contains(
           'ascending|sensory|projection', case=False, na=False
       )
   ].copy()
   ```

3. **Lower synapse threshold:**
   ```bash
   python scripts/extract_sez_neurons.py --min-synapses 5
   ```

### Issue 4: Clustering Validation Fails

**Error:**
```
⚠ WARNING: Silhouette analysis failed
⚠ WARNING: Too few SEZ-PNs for meaningful clustering validation
Found 8 neurons, need at least 10
```

**Diagnosis:**
- Too few SEZ-PNs extracted for statistical clustering
- Connectivity matrix is sparse or malformed

**Solution:**

1. **Skip validation and proceed:**
   ```bash
   python scripts/extract_sez_neurons.py --skip-validation
   ```

2. **Check why SEZ-PN count is low** (see Issue 3)

3. **Verify connectivity data:**
   ```python
   import pandas as pd
   connections = pd.read_csv("data/flywire/connections_princeton.csv.gz")
   print(f"Total connections: {len(connections):,}")
   print(connections.head())
   ```

### Issue 5: UMAP Plot Not Generated

**Warning:**
```
⚠ UMAP not available (install with: pip install umap-learn)
```

**Solution:**
```bash
# Install UMAP (optional, for visualization only)
pip install umap-learn

# Or conda:
conda install -c conda-forge umap-learn

# Re-run extraction
python scripts/extract_sez_neurons.py
```

**Note:** UMAP is optional. The pipeline works without it; you'll just miss the 2D embedding plot.

### Issue 6: FlyWire Data Files Not Found

**Error:**
```
ERROR: Failed to load FlyWire datasets
FileNotFoundError: connections_princeton.csv.gz not found
```

**Solution:**

1. **Verify data directory structure:**
   ```bash
   ls -lh data/flywire/
   # Expected files:
   # - connections_princeton.csv.gz
   # - classification.csv.gz
   # - consolidated_cell_types.csv.gz
   # - neurons.csv.gz
   # - names.csv.gz (optional)
   # - root_ids_class_gustatory.txt
   ```

2. **Check environment variable:**
   ```bash
   echo $PGCN_FLYWIRE_DATA
   # If set, script will look there instead of data/flywire/
   ```

3. **Download missing files from FlyWire:**
   - Visit: https://flywire.ai/
   - Export required tables for FAFB v783

---

## References

### Primary References

**Li, J. et al. (2024).** Connectomic analysis of taste circuits in Drosophila.
*Scientific Reports*, 14, 21120.
DOI: [10.1038/s41598-024-71926-2](https://doi.org/10.1038/s41598-024-71926-2)

**Shen, K. et al. (2025).** Functional imaging and connectome analyses reveal organizing principles of processing taste modality in the Drosophila brain.
*Current Biology*, 35(9), 1955-1970.e6.
DOI: [10.1016/j.cub.2025.03.053](https://doi.org/10.1016/j.cub.2025.03.053)

### PGCN Model References

**Zandawala, M. et al. (2024).** A nutrient-responsive hormonal circuit mediates an inter-tissue program regulating metabolic homeostasis in adult Drosophila.
*eLife*, 13, RP10030.
DOI: [10.7554/eLife.RP10030](https://doi.org/10.7554/eLife.RP10030)

**Schlegel, P. et al. (2023).** Whole-brain annotation and multi-connectome cell typing of Drosophila.
*bioRxiv* 2023.06.27.546055.
DOI: [10.1101/2023.06.27.546055](https://doi.org/10.1101/2023.06.27.546055)

### FlyWire Resources

**FlyWire Portal:** https://flywire.ai/
**FlyWire Codex:** https://codex.flywire.ai/
**FAFB v783 Dataset:** Latest FlyWire reconstruction

---

## Citation

If you use this SEZ extraction pipeline in your research, please cite:

```bibtex
@software{pgcn_sez_extraction,
  title={SEZ Neuron Extraction Pipeline for PGCN Model},
  author={PGCN Development Team},
  year={2025},
  url={https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN}
}

@article{li2024connectomic,
  title={Connectomic analysis of taste circuits in Drosophila},
  author={Li, J. and others},
  journal={Scientific Reports},
  volume={14},
  pages={21120},
  year={2024},
  doi={10.1038/s41598-024-71926-2}
}

@article{shen2025functional,
  title={Functional imaging and connectome analyses reveal organizing principles of processing taste modality in the Drosophila brain},
  author={Shen, K. and others},
  journal={Current Biology},
  volume={35},
  number={9},
  pages={1955--1970},
  year={2025},
  doi={10.1016/j.cub.2025.03.053}
}
```

---

## Contact & Support

**Repository:** https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN
**Issues:** https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN/issues

For questions about the SEZ extraction pipeline, please open an issue with the tag `sez-extraction`.

---

## Changelog

### Version 1.0 (2025-11-11)
- Initial implementation of SEZ neuron extraction pipeline
- Implements Li et al. (2024) clustering validation
- Comprehensive test suite with 16 tests
- Integration with PGCN model via summarize_all_cell_types.py
- Complete documentation and troubleshooting guide

---

**End of Guide**
