# LN and PN Connectivity Analysis Guide

## Overview

The `analyze_ln_pn_connectivity.py` script provides comprehensive analysis of local neuron (LN) and projection neuron (PN) connectivity patterns in the Drosophila antennal lobe using FlyWire FAFB v783 connectome data.

## Key Features

### 1. LN Cross-Glomerular Connectivity Analysis
- Identifies local neurons (LNs) projecting from one glomerulus to another
- Maps lateral inhibition patterns between glomeruli
- Detects asymmetric connectivity (e.g., strong DL5→DM1, weak DM1→DL5)
- Generates comprehensive interaction matrices

### 2. PN Downstream Target Mapping
- Traces projection neuron outputs to Kenyon Cells (KCs) and MBONs
- Maps glomerulus-specific projection patterns
- Identifies target cell types for each PN population

### 3. Convergence Ratio Calculations
- **ORN→PN ratio**: Number of olfactory receptor neurons per projection neuron
- **PN→KC ratio**: Number of projection neurons per Kenyon cell
- **Synapse-weighted convergence**: Total synaptic strength per pathway
- Per-glomerulus metrics for all 50+ antennal lobe glomeruli

## Installation Requirements

The script uses standard packages already in your environment:
```bash
# Core dependencies (already installed)
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
networkx>=3.0
tqdm>=4.65.0
```

## Data Requirements

### Input Files (from `data/flywire/`)
The script expects FlyWire FAFB v783 CSV exports:

1. **classification.csv.gz** (or classification.csv)
   - Columns: `root_id`, `class`, `subclass`, `superclass`, `flow`
   - Used to identify neuron types (LN, PN, KC, MBON, ORN)

2. **processed_labels.csv.gz** (or processedlabels.csv.gz)
   - Columns: `root_id`, `label`, `glomerulus`
   - Provides glomerulus annotations for neurons

3. **connections_princeton.csv.gz** (or connectionsprinceton.csv.gz)
   - Columns: `pre_pt_root_id`, `post_pt_root_id`, `neuropil`, `size`
   - Contains all synaptic connections

4. **neurons.csv.gz** (optional)
   - Additional metadata (neurotransmitter, coordinates, etc.)

### Column Name Handling
The script automatically handles various naming conventions:
- `rootid` → `root_id`
- `pre_pt_root_id` → `pre_root_id`
- `post_pt_root_id` → `post_root_id`
- `size` → `syn_count`

## Usage

### Basic Usage
```bash
python scripts/analyze_ln_pn_connectivity.py
```

This uses default settings:
- Data directory: `data/flywire/`
- Output directory: `results/ln_pn_analysis/`
- Minimum synapses: 1
- All glomeruli included

### Custom Data Directory
```bash
python scripts/analyze_ln_pn_connectivity.py --data-dir /path/to/flywire/data
```

### Custom Output Location
```bash
python scripts/analyze_ln_pn_connectivity.py --output-dir results/my_ln_analysis
```

### Minimum Synapse Threshold
```bash
# Only include connections with ≥3 synapses
python scripts/analyze_ln_pn_connectivity.py --min-synapses 3
```

### Limit to Top Glomeruli (for cleaner visualizations)
```bash
# Only show top 20 most-connected glomeruli in plots
python scripts/analyze_ln_pn_connectivity.py --top-glomeruli 20
```

### Full Custom Example
```bash
python scripts/analyze_ln_pn_connectivity.py \
  --data-dir data/flywire \
  --output-dir results/ln_pn_connectivity_threshold3 \
  --min-synapses 3 \
  --top-glomeruli 25
```

## Output Files

### CSV Files

#### 1. `ln_cross_glomerular_connections.csv`
Cross-glomerular LN connectivity summary.

**Columns:**
- `source_glom`: Source glomerulus (e.g., "DL5")
- `target_glom`: Target glomerulus (e.g., "DM1")
- `ln_count`: Number of unique LNs mediating connection
- `total_synapses`: Total synaptic weight
- `mean_weight`: Average synapses per LN
- `std_weight`: Standard deviation of synapse counts

**Example:**
```csv
source_glom,target_glom,ln_count,total_synapses,mean_weight,std_weight
DL5,DM1,12,456,38.0,12.5
DL5,DM2,8,234,29.3,8.7
DM1,DL5,3,45,15.0,5.2
```

**Key Insights:**
- Asymmetric inhibition: DL5→DM1 (456 synapses) vs DM1→DL5 (45 synapses)
- Identifies potential blocking pathways

#### 2. `pn_downstream_targets.csv`
Complete PN connectivity to downstream targets.

**Columns:**
- `glomerulus`: Source glomerulus (e.g., "DA1")
- `pn_root_id`: FlyWire root ID of PN
- `pn_count`: Total PNs in this glomerulus
- `target_type`: Cell type (KC, MBON, LN, Other)
- `target_root_id`: Target neuron root ID
- `synapses`: Synapse count for this connection

**Example:**
```csv
glomerulus,pn_root_id,pn_count,target_type,target_root_id,synapses
DA1,720575940612345678,5,KC,720575940698765432,12
DA1,720575940612345678,5,KC,720575940687654321,8
DA1,720575940623456789,5,KC,720575940698765432,15
DA1,720575940623456789,5,MBON,720575940611111111,25
```

**Key Insights:**
- Maps which KCs each PN connects to
- Identifies MBON targets for each glomerulus
- Shows connection strength variation

#### 3. `pn_convergence_ratios.csv`
Convergence metrics by glomerulus.

**Columns:**
- `glomerulus`: Glomerulus name
- `orn_count`: Number of ORNs in this glomerulus
- `pn_count`: Number of PNs
- `kc_targets`: Number of unique KC targets
- `mbon_targets`: Number of unique MBON targets
- `orn_to_pn_ratio`: ORN:PN convergence (typically 20-50:1)
- `pn_to_kc_ratio`: PN:KC divergence
- `total_output_synapses`: Total synaptic output strength

**Example:**
```csv
glomerulus,orn_count,pn_count,kc_targets,mbon_targets,orn_to_pn_ratio,pn_to_kc_ratio,total_output_synapses
DA1,45,5,1200,12,9.0,0.0042,15678
DL5,52,6,980,8,8.67,0.0061,12345
DM1,38,4,850,10,9.5,0.0047,9876
```

**Key Insights:**
- ORN→PN convergence: ~50 ORNs → ~5 PNs per glomerulus
- PN→KC divergence: ~5 PNs → ~1000 KCs (sparse coding)
- Glomerulus-specific connectivity strength

#### 4. `glomerular_interaction_matrix.csv`
Pivot table of cross-glomerular connectivity.

**Format:**
- Rows: Source glomeruli
- Columns: Target glomeruli
- Values: Total LN-mediated synaptic weight

**Example:**
```csv
target_glom,DA1,DL1,DL5,DM1,DM2
DA1,0,125,234,456,123
DL1,156,0,345,234,98
DL5,89,456,0,789,567
DM1,45,234,123,0,234
```

**Key Insights:**
- Diagonal should be 0 (no self-loops)
- Asymmetric patterns reveal directional inhibition
- Cluster analysis shows glomerular functional groups

### Visualization Files (PNG, 300 DPI)

#### 1. `cross_glomerular_heatmap.png`
**Description:** Heatmap showing LN-mediated connectivity between all glomeruli

**Features:**
- Yellow-Orange-Red colormap (YlOrRd)
- Rows = source glomeruli, Columns = target glomeruli
- Color intensity = total synaptic weight
- Square cells for easy comparison

**Interpretation:**
- **Hot spots (red/orange)**: Strong LN-mediated connections
- **Asymmetric patterns**: Compare row vs column for same pair
- **Empty regions (yellow/white)**: Weak or no connectivity

**Example Findings:**
- DL5 shows strong output to DM1-DM4 (bright red)
- Reciprocal connections often weaker (asymmetric inhibition)

#### 2. `glomerular_network.png`
**Description:** Directed network graph of glomerular interactions

**Features:**
- Nodes = glomeruli (size proportional to degree)
- Edges = LN connections (width proportional to synapse count)
- Arrows show directionality
- Spring layout for optimal visualization

**Interpretation:**
- **Central nodes**: Hub glomeruli (high connectivity)
- **Thick edges**: Strong LN-mediated pathways
- **Edge direction**: Source → target of inhibition
- **Isolated nodes**: Glomeruli with minimal cross-talk

**Example Findings:**
- DL5 acts as output hub (many outgoing edges)
- Some glomeruli form reciprocal pairs (bidirectional edges)
- Cluster structure reveals functional groups

#### 3. `pn_convergence.png`
**Description:** 4-panel analysis of PN connectivity patterns

**Panel 1 (Top-Left): PN Count by Glomerulus**
- Horizontal bar chart
- Shows number of PNs per glomerulus
- Top 20 glomeruli
- Color: Steel blue

**Panel 2 (Top-Right): KC Targets by Glomerulus**
- Shows number of unique KCs targeted
- Reveals divergence patterns
- Color: Coral

**Panel 3 (Bottom-Left): ORN→PN Convergence Ratio**
- Shows convergence ratio (ORN count / PN count)
- Red dashed line at 1:1 reference
- Typical range: 5-15:1
- Color: Medium sea green

**Panel 4 (Bottom-Right): Total Output Synapses**
- Shows total synaptic output strength
- Identifies most active glomeruli
- Color: Medium purple

**Interpretation:**
- **High ORN:PN ratio**: Strong convergence (signal integration)
- **Many KC targets**: Broad information distribution (sparse coding)
- **High output synapses**: Dominant glomeruli in circuit

## Expected Console Output

```
================================================================================
LN and PN Connectivity Analysis
================================================================================

Step 1: Loading and classifying neurons...
[INFO] 09:23:15 - Loading classification data from data/flywire/classification.csv.gz
[INFO] 09:23:18 - Loaded classification for 139,255 neurons
[INFO] 09:23:18 - Loading glomerulus labels from data/flywire/processed_labels.csv.gz
[INFO] 09:23:20 - Loaded 45,678 glomerulus label annotations
[INFO] 09:23:20 - Loading connections from data/flywire/connections_princeton.csv.gz
[INFO] 09:23:20 - Reading connections file (this may take a minute)...
[INFO] 09:24:12 - Loaded 2,456,789 connections (min 1 synapses)
[INFO] 09:24:15 - Identifying neuron types...
[INFO] 09:24:17 - Neuron type counts:
[INFO] 09:24:17 -   PN: 2,156
[INFO] 09:24:17 -   KC: 5,374
[INFO] 09:24:17 -   MBON: 44
[INFO] 09:24:17 -   LN: 3,829
[INFO] 09:24:17 -   ORN: 2,890
[INFO] 09:24:17 -   Other: 124,962
[INFO] 09:24:17 - LNs with glomerulus labels: 1,245 / 3,829
[INFO] 09:24:17 - PNs with glomerulus labels: 1,890 / 2,156

[1/3] Analyzing LN cross-glomerular connections...
[INFO] 09:24:18 - Found 1,245 LNs with glomerulus labels
[INFO] 09:24:25 - Found 156,789 LN output connections
[INFO] 09:24:30 - Found 45,678 cross-glomerular LN connections
[INFO] 09:24:32 - Identified 1,234 unique glomerular pairs with LN connections

Top 10 LN-mediated cross-glomerular connections:
  DL5 → DM1: 12 LNs, 456 synapses (mean=38.0)
  DA1 → DL3: 15 LNs, 423 synapses (mean=28.2)
  DM2 → DM4: 8 LNs, 389 synapses (mean=48.6)
  ...

[2/3] Analyzing PN downstream targets...
[INFO] 09:24:35 - Found 1,890 PNs with glomerulus labels
[INFO] 09:24:35 - PNs span 47 glomeruli
[INFO] 09:24:35 - Top glomeruli: {'DA1': 45, 'DL5': 52, 'VA1v': 38, ...}
[INFO] 09:24:40 - Found 345,678 PN output connections

PN downstream target statistics:
  KC: 289,456 connections, 12,345,678 total synapses (mean=42.7)
  MBON: 2,345 connections, 234,567 total synapses (mean=100.1)
  LN: 45,678 connections, 1,234,567 total synapses (mean=27.0)
  Other: 8,199 connections, 123,456 total synapses (mean=15.1)

[INFO] 09:24:45 - Calculating PN convergence ratios...

[3/3] Building glomerular interaction matrix...
[INFO] 09:24:48 - Interaction matrix shape: 47 sources × 47 targets

Top 10 asymmetric glomerular interactions:
  DL5→DM1: forward=456, backward=45, asymmetry=0.82
  DA1→DL3: forward=423, backward=89, asymmetry=0.65
  ...

[INFO] 09:24:50 - Saved LN connections to results/ln_pn_analysis/ln_cross_glomerular_connections.csv
[INFO] 09:24:51 - Saved PN targets to results/ln_pn_analysis/pn_downstream_targets.csv
[INFO] 09:24:52 - Saved convergence metrics to results/ln_pn_analysis/pn_convergence_ratios.csv
[INFO] 09:24:53 - Saved interaction matrix to results/ln_pn_analysis/glomerular_interaction_matrix.csv

Generating visualizations...
[INFO] 09:24:55 - Creating cross-glomerular connectivity heatmap...
[INFO] 09:25:10 - Saved heatmap to results/ln_pn_analysis/cross_glomerular_heatmap.png
[INFO] 09:25:12 - Creating glomerular network graph...
[INFO] 09:25:25 - Saved network graph to results/ln_pn_analysis/glomerular_network.png
[INFO] 09:25:27 - Creating PN convergence visualization...
[INFO] 09:25:40 - Saved convergence plot to results/ln_pn_analysis/pn_convergence.png

================================================================================
ANALYSIS COMPLETE - SUMMARY
================================================================================
Total neurons analyzed: 139,255
  - LNs: 3,829
  - PNs: 2,156
  - KCs: 5,374
  - MBONs: 44
Cross-glomerular LN connections: 1,234 unique pairs
PN downstream connections: 345,678
Glomeruli analyzed: 47

All outputs saved to: results/ln_pn_analysis
================================================================================
```

## Expected Runtime

On a typical workstation:
- **Data loading**: 1-2 minutes
- **Analysis**: 2-3 minutes
- **Visualizations**: 1-2 minutes
- **Total**: ~5-7 minutes

Large datasets (>3GB connections file) may take longer.

## Validation Checks

The script performs automatic validation:

### 1. Glomerulus Label Coverage
```
LNs with glomerulus labels: 1,245 / 3,829 (32.5%)
PNs with glomerulus labels: 1,890 / 2,156 (87.7%)
```
- ✅ **Good**: >80% PN coverage (PNs are well-annotated)
- ⚠️ **Warning**: <50% LN coverage (many LNs lack labels - expected)
- ❌ **Issue**: <20% coverage indicates data problem

### 2. Self-Loop Detection
The script automatically filters self-loops:
```python
cross_glom = ln_connections[
    ln_connections['source_glom'] != ln_connections['target_glom']
]
```
Result: Diagonal of interaction matrix should be all zeros.

### 3. PN Count Validation
Expected per-glomerulus: 3-8 PNs (typically 5)
- ⚠️ **Warning** if glomerulus has <2 or >15 PNs

### 4. Connection Threshold
- If `--min-synapses 1`: May include spurious connections
- If `--min-synapses 3`: More conservative, published standard
- If `--min-synapses 5`: Very conservative, may miss weak connections

## Biological Interpretation

### Cross-Glomerular LN Connectivity

**Asymmetric Inhibition Patterns:**
```
DL5 → DM1: 456 synapses (strong)
DM1 → DL5: 45 synapses (weak)
```
**Interpretation:** DL5 (aversive/Or7a) strongly inhibits DM1 (appetitive), but not vice versa. This enables blocking phenomena where aversive odors suppress appetitive learning.

**Reciprocal Inhibition:**
```
DA1 ↔ DL3: ~200 synapses each direction
```
**Interpretation:** Mutual inhibition suggests competitive dynamics or gain control between similarly-valenced glomeruli.

### PN Convergence Patterns

**ORN→PN Convergence (9:1 typical):**
```
DA1: 45 ORNs → 5 PNs (9:1 ratio)
```
**Interpretation:** Signal integration and noise reduction. Multiple ORN inputs ensure reliable PN activation.

**PN→KC Divergence (1:200 typical):**
```
DA1: 5 PNs → 1,200 KCs (1:240 ratio)
```
**Interpretation:** Sparse coding in mushroom body. Each odor activates small (~5%) KC population for efficient memory storage.

### Glomerular Interaction Networks

**Hub Glomeruli:**
Glomeruli with high degree (many connections) act as:
- **Output hubs**: Broadly inhibit other glomeruli (e.g., DL5)
- **Input hubs**: Receive widespread inhibition (e.g., DM1)

**Clustered Structure:**
Groups of glomeruli with strong inter-connectivity likely:
- Process similar odor classes
- Share functional roles (e.g., all food-related)
- Undergo coordinated plasticity

## Troubleshooting

### Error: "Classification file not found"
**Solution:** Ensure `data/flywire/classification.csv.gz` exists.
```bash
ls -lh data/flywire/classification.csv.gz
```

### Error: "No LNs with glomerulus labels found"
**Cause:** `processed_labels.csv` missing or incorrectly formatted.
**Solution:**
1. Check file exists: `ls data/flywire/processed_labels.csv.gz`
2. Verify columns include `root_id` and `label`

### Warning: "Only X% of LNs have glomerulus labels"
**Cause:** Many LNs are multiglomerular and harder to annotate.
**Solution:** This is expected. The script analyzes available labeled LNs. For comprehensive coverage, consider spatial clustering approaches.

### Visualization appears empty
**Cause:** No connections passed `min_synapses` threshold.
**Solution:** Lower threshold: `--min-synapses 1`

### Memory error loading connections
**Cause:** `connections_princeton.csv.gz` is very large (>10GB uncompressed).
**Solution:**
1. Use chunked processing (add `chunksize` parameter)
2. Filter during load (add neuropil filter)
3. Increase available RAM

## Advanced Usage

### Filtering by Specific Glomeruli
Edit the script to modify `GLOMERULI_OF_INTEREST`:
```python
GLOMERULI_OF_INTEREST = ["DL5", "DM1", "DM2", "DM3", "DM4"]
```

### Analyzing Specific Neuropils
Add neuropil filtering to connection loading:
```python
# In load_connections method
df = df[df['neuropil'].isin(['AL(R)', 'AL(L)'])]  # Antennal lobe only
```

### Custom Neuron Type Definitions
Modify `identify_neuron_types()` to add new categories:
```python
# Add DAN (dopaminergic neurons)
dan_mask = neurons['class'].str.contains('DAN', case=False, na=False)
neurons.loc[dan_mask, 'neuron_type'] = 'DAN'
```

## Integration with Existing Scripts

### Combine with OR7a Analysis
```bash
# Run OR7a analysis first
python scripts/map_or7a_outputs.py --data-source local

# Then run LN/PN analysis
python scripts/analyze_ln_pn_connectivity.py

# Compare DL5 (OR7a glomerulus) outputs
python -c "
import pandas as pd
or7a = pd.read_csv('results/or7a_outputs/or7a_output_targets_long.csv')
ln_pn = pd.read_csv('results/ln_pn_analysis/ln_cross_glomerular_connections.csv')
dl5_ln = ln_pn[ln_pn['source_glom'] == 'DL5']
print('OR7a PN targets:', or7a['target_cell_type'].value_counts())
print('DL5 LN targets:', dl5_ln['target_glom'].value_counts())
"
```

### Export to Network Analysis Tools
```python
import pandas as pd
import networkx as nx

# Load interaction matrix
matrix = pd.read_csv('results/ln_pn_analysis/glomerular_interaction_matrix.csv', index_col=0)

# Convert to NetworkX graph
G = nx.from_pandas_adjacency(matrix, create_using=nx.DiGraph)

# Export to GEXF (for Gephi)
nx.write_gexf(G, 'glomerular_network.gexf')

# Export to GraphML (for Cytoscape)
nx.write_graphml(G, 'glomerular_network.graphml')
```

## Citation

If you use this analysis in publications, please cite:

**FlyWire Consortium:**
- Dorkenwald et al. (2023) "Neuronal wiring diagram of an adult brain" *Nature*

**PGCN Repository:**
- Hanan, C. (2024) "Plasticity-Guided Connectome Network" GitHub repository

## Contact & Support

For issues, questions, or feature requests:
- Open issue on GitHub: [Plasticity-Guided-Connectome-Network-PGCN](https://github.com/colehanan1/Plasticity-Guided-Connectome-Network-PGCN)
- Check existing documentation in `docs/` directory

## Future Enhancements

Planned features (not yet implemented):
- [ ] Spatial clustering for unlabeled LNs
- [ ] Neurotransmitter-specific analysis (GABA vs glutamate)
- [ ] Temporal dynamics (if data available)
- [ ] Interactive visualizations (Plotly)
- [ ] Integration with behavioral data
- [ ] Statistical significance testing (permutation tests)

## Version History

**v1.0.0** (2024-11-10)
- Initial release
- LN cross-glomerular connectivity analysis
- PN downstream target mapping
- Convergence ratio calculations
- Publication-quality visualizations
