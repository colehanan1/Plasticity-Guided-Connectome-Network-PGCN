# Extract Taste Circuits from Shen et al. (2025) Paper Data

## Overview

This extraction pipeline loads **calcium imaging-validated** taste circuit neurons from Shen et al. (2025) Current Biology supplementary data, replacing FlyWire query-based extraction with **ground-truth experimental data**.

**Script:** `scripts/extract_from_paper_data.py`

## Why Use Paper Data Instead of FlyWire Queries?

| Feature | Paper Data (This Script) | FlyWire Query-Based |
|---------|--------------------------|---------------------|
| **Functional validation** | ✅ Calcium imaging confirmed | ❌ Connectomic inference only |
| **Taste modality** | ✅ Explicit (sweet/bitter/water) | ⚠️ Must infer from connectivity |
| **Reproducibility** | ✅ Published ground truth | ⚠️ Depends on query logic |
| **Processing speed** | ✅ Pre-built matrices (~1 sec) | ⚠️ Requires FlyWire API queries |
| **Peer review** | ✅ Current Biology peer-reviewed | ⚠️ Internal validation only |

## Scientific Foundation

### Biological Context

The PGCN model implements **appetitive (reward-based) associative learning**:

1. **OR7a odor** (apple vinegar) → Olfactory pathway
2. **+ Sucrose reward** (sweet taste) → Taste pathway
3. **→ Associative learning** in mushroom body (KC→MBON plasticity)

**Critical requirement:** Model must use **ONLY sugar-sensing circuits** that are activated during sucrose reward training.

### Reference

**Shen, K. et al. (2025).** "Functional imaging and connectome analyses reveal organizing principles of processing taste modality in the *Drosophila* brain."
*Current Biology* 35(9):1955-1970.e6
DOI: [10.1016/j.cub.2025.04.066](https://doi.org/10.1016/j.cub.2025.04.066)

**Key findings:**
- Sweet GRNs (Gr5a/Gr64a/Gr61a) drive approach behavior
- Direct SEZ-PN relay to lateral horn and mushroom body
- Cholinergic local interneurons amplify sugar signals
- FlyWire FAFB v783 connectome validation

---

## Data Requirements

### Required Files

Place Shen et al. (2025) supplementary files in `data/10.1016/`:

1. **`Neurons-list-v783.xlsx`** (neuron catalog)
   - Contains GRN identities with taste modality labels
   - FlyWire v783 root IDs for each neuron

2. **`GRN-vs-directly-connected-SEZ-PN-connectivity_final.xlsx`**
   - 143 GRNs × 57 SEZ-PNs connectivity matrix
   - Synapse counts for each GRN→SEZ-PN connection

3. **`GRN-vs-ACh-LNs-connectivity_final.xlsx`**
   - 143 GRNs × 83 ACh-LNs connectivity matrix
   - Cholinergic relay interneurons

4. **`data/flywire/names.csv.gz`** (FlyWire name mapping)
   - Download from: https://codex.flywire.ai/api/download?dataset=fafb
   - File: "Proofread Cell Names And Groups (1,182 KB)"

### Directory Structure

```
data/
├── 10.1016/                                # Shen et al. (2025) paper data
│   ├── Neurons-list-v783.xlsx
│   ├── GRN-vs-directly-connected-SEZ-PN-connectivity_final.xlsx
│   ├── GRN-vs-ACh-LNs-connectivity_final.xlsx
│   └── GRN-vs-GABA-LNs_connectivity_final.xlsx  (optional, for full mode)
│
└── flywire/
    └── names.csv.gz                        # FlyWire Codex name mapping
```

---

## Usage

### Mode 1: Appetitive (Sugar Only) - **Default**

**Use case:** PGCN appetitive learning model (sucrose reward experiments)

```bash
python scripts/extract_from_paper_data.py \
  --mode appetitive \
  --output-dir data/cache
```

**What it does:**
1. Filters to **sweet GRNs only** (Gr5a, Gr64a, Gr61a families)
2. Extracts SEZ-PNs receiving ≥1 synapse from sweet GRNs
3. Extracts ACh-LNs receiving ≥1 synapse from sweet GRNs
4. Maps neuron names to FlyWire root IDs
5. Exports filtered datasets and connectivity matrices

**Expected output:**
```
data/cache/
  shen2025_appetitive_grn.csv                # 30-50 sweet GRNs
  shen2025_appetitive_sez_pn.csv             # 15-35 SEZ-PNs
  shen2025_appetitive_sez_ln_ach.csv         # 25-50 ACh-LNs
  shen2025_appetitive_connectivity_grn_pn.npz
  shen2025_appetitive_connectivity_grn_ach.npz
  shen2025_appetitive_validation_report.json
```

---

### Mode 2: Full Gustatory - **Validation**

**Use case:** Validate extraction logic, enable future bitter/aversive experiments

```bash
python scripts/extract_from_paper_data.py \
  --mode full \
  --output-dir data/cache
```

**What it does:**
1. Loads **all GRNs** (sweet, bitter, IR94e, Ppk23, etc.)
2. Extracts all 57 SEZ-PNs from paper
3. Extracts all 83 ACh-LNs from paper
4. (Optional) Extracts 50 GABA-LNs if file present

**Expected output:**
```
data/cache/
  shen2025_full_grn.csv                      # 120-150 GRNs
  shen2025_full_sez_pn.csv                   # 57 SEZ-PNs
  shen2025_full_sez_ln_ach.csv               # 83 ACh-LNs
  shen2025_full_connectivity_grn_pn.npz
  shen2025_full_connectivity_grn_ach.npz
  shen2025_full_validation_report.json
```

---

## Output File Formats

### CSV Files (Neuron Lists)

**`shen2025_appetitive_grn.csv`**
```csv
v783,root_id,grn_type,side
GNG.1456,720575940612345678,sweet,left
GNG.2567,720575940623456789,sweet,right
...
```

**`shen2025_appetitive_sez_pn.csv`**
```csv
name,total_input_synapses,root_id
SLP.78,245,720575940634567890
GNG.SLP.21,187,720575940645678901
...
```

**`shen2025_appetitive_sez_ln_ach.csv`**
```csv
name,total_input_synapses,root_id
GNG.PRW.24,98,720575940656789012
GNG.285,142,720575940667890123
...
```

### NPZ Files (Connectivity Matrices)

**`shen2025_appetitive_connectivity_grn_pn.npz`**
```python
import numpy as np

data = np.load('data/cache/shen2025_appetitive_connectivity_grn_pn.npz')
connectivity = data['connectivity']  # Shape: (n_grns, n_sez_pns)
grn_ids = data['grn_ids']            # GRN root IDs
sez_pn_ids = data['sez_pn_ids']      # SEZ-PN root IDs

# Example: Get synapses from GRN i to SEZ-PN j
n_synapses = connectivity[i, j]
```

---

## Validation Report

**`shen2025_appetitive_validation_report.json`**
```json
{
  "extraction_mode": "appetitive",
  "timestamp": "2025-11-11T22:30:00",
  "source": "Shen et al. (2025) Current Biology 35(9):1955-1970",
  "flywire_version": "v783",
  "neuron_counts": {
    "grns": 38,
    "sez_pns": 24,
    "ach_lns": 42
  },
  "root_id_mapping": {
    "grns_mapped": 38,
    "sez_pns_mapped": 23,
    "ach_lns_mapped": 41
  },
  "validation_grns": "PASS",
  "validation_sez_pns": "PASS",
  "validation_ach_lns": "PASS",
  "validation_mapping": "CHECK"
}
```

**Validation criteria:**
- ✅ `PASS`: Count within expected range AND mapping rate >95%
- ⚠️ `CHECK`: Count or mapping rate needs review

---

## Integration with PGCN Model

### Replace Existing Extraction

**Old approach (FlyWire query-based):**
```python
# scripts/extract_sez_neurons.py
grn_ids = query_flywire_for_grns(...)
sez_pn_ids = trace_second_order_neurons(...)
```

**New approach (Paper data):**
```python
# scripts/extract_from_paper_data.py
grn_df = pd.read_csv('data/cache/shen2025_appetitive_grn.csv')
sez_pn_df = pd.read_csv('data/cache/shen2025_appetitive_sez_pn.csv')
ach_ln_df = pd.read_csv('data/cache/shen2025_appetitive_sez_ln_ach.csv')
```

### Load Connectivity Matrices

```python
import numpy as np

# Load GRN → SEZ-PN connectivity
conn_data = np.load('data/cache/shen2025_appetitive_connectivity_grn_pn.npz')
W_grn_pn = conn_data['connectivity']  # Shape: (n_grns, n_sez_pns)

# Load GRN → ACh-LN connectivity
conn_data = np.load('data/cache/shen2025_appetitive_connectivity_grn_ach.npz')
W_grn_ln = conn_data['connectivity']  # Shape: (n_grns, n_ach_lns)

# Use in PGCN model
from pgcn.models import EnhancedOlfactoryCircuit

circuit = EnhancedOlfactoryCircuit(
    n_grns=W_grn_pn.shape[0],
    n_sez_pns=W_grn_pn.shape[1],
    grn_to_sez_connectivity=W_grn_pn,
    ...
)
```

---

## Advanced Options

### Adjust Synapse Threshold

```bash
# Require ≥5 synapses instead of ≥1
python scripts/extract_from_paper_data.py \
  --mode appetitive \
  --min-synapses 5
```

**Effect:** Reduces noise by filtering weak connections

### Custom Data Locations

```bash
python scripts/extract_from_paper_data.py \
  --paper-data-dir /mnt/external/shen2025_data \
  --flywire-names /mnt/external/flywire/names.csv.gz \
  --output-dir /mnt/output
```

---

## Troubleshooting

### Error: Required files not found

**Solution:** Download Shen et al. (2025) supplementary files from journal website:
- Visit: https://www.cell.com/current-biology/fulltext/S0960-9822(25)00424-X
- Download supplementary Excel files
- Place in `data/10.1016/`

### Error: FlyWire names file not found

**Solution:** Download from FlyWire Codex:
```bash
wget https://codex.flywire.ai/api/download?dataset=fafb -O data/flywire/names.csv.gz
```

### Warning: X neurons not found in names.csv.gz

**Cause:** Some paper neurons may have been renamed or deleted in FlyWire

**Impact:** Minimal if <5% unmapped

**Action:** Check validation report - if mapping rate >95%, extraction is valid

### Neuron counts outside expected range

**For appetitive mode:**
- GRNs: 30-50 expected (sweet GRNs only)
- SEZ-PNs: 15-35 expected (subset of 57 receiving sweet input)
- ACh-LNs: 25-50 expected (subset of 83 receiving sweet input)

**For full mode:**
- GRNs: 120-150 expected (all taste modalities)
- SEZ-PNs: 57 expected (exact from paper)
- ACh-LNs: 83 expected (exact from paper)

**If counts differ:** Check Excel file versions match paper

---

## Comparison: Query-Based vs Paper Data Extraction

### Neuron Count Comparison

| Extraction Method | GRNs | SEZ-PNs | ACh-LNs |
|------------------|------|---------|---------|
| **FlyWire Query** | 131 (all gustatory) | ~150-200 | ~15-73 |
| **Paper Data (Full)** | 143 (all gustatory) | 57 | 83 |
| **Paper Data (Appetitive)** | ~38 (sweet only) | ~24 | ~42 |

**Key difference:** Query-based includes neurons without functional validation. Paper data uses only calcium imaging-confirmed neurons.

### Biological Accuracy

**Query-based challenges:**
- May include false positives (anatomically connected but not functionally active)
- Cannot distinguish taste modalities reliably
- Depends on synapse count thresholds (arbitrary)

**Paper data advantages:**
- Functional validation via calcium imaging
- Explicit taste modality labels (sweet/bitter)
- Peer-reviewed connectivity matrices

---

## Future Extensions

### Add GABA-LN Extraction (Inhibitory Circuits)

```python
# In extract_from_paper_data.py, add:
def extract_gaba_lns(
    connectivity_file: Path,
    grn_filter: pd.DataFrame,
    names_lookup: pd.DataFrame,
    min_synapses: int = 1
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract GABAergic local neurons."""
    # Same logic as extract_ach_lns
    ...
```

### Support Other Paper Datasets

Extend to other published datasets:
- Engert et al. (2024) - Visual circuits
- Schlegel et al. (2023) - Descending neurons
- Zheng et al. (2018) - Full brain FAFB

---

## Professional Standards

This extraction pipeline implements:

✅ **Experimental validation** - Calcium imaging-confirmed neurons
✅ **Biological accuracy** - Sugar GRNs for sucrose reward experiments
✅ **Reproducibility** - Published paper data with DOI
✅ **Flexibility** - Two modes (appetitive/full)
✅ **Robustness** - Comprehensive validation and error checking
✅ **Documentation** - Inline comments and validation reports

---

## References

1. **Shen, K. et al. (2025).** "Functional imaging and connectome analyses reveal organizing principles of processing taste modality in the *Drosophila* brain." *Current Biology* 35(9):1955-1970.e6. DOI: 10.1016/j.cub.2025.04.066

2. **Li, J. et al. (2024).** "Connectomic analysis of taste circuits in *Drosophila*." *Scientific Reports* 14:21120. DOI: 10.1038/s41598-024-71926-2

3. **Dorkenwald, S. et al. (2024).** "Neuronal wiring diagram of an adult brain." *Nature* 634:124-138. DOI: 10.1038/s41586-024-07558-y

---

## Contact & Support

For issues or questions:
- **GitHub Issues:** https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN/issues
- **Script:** `scripts/extract_from_paper_data.py`
- **Documentation:** `docs/PAPER_DATA_EXTRACTION.md`

---

**Last updated:** November 2025
**Script version:** 1.0.0
**FlyWire version:** v783
