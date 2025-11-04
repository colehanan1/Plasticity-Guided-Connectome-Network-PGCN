# GRN Downstream Connectivity Visualization Pipeline

Comprehensive Python pipeline for analyzing and visualizing downstream synaptic connectivity from gustatory receptor neurons (GRNs) in the FlyWire connectome.

## Features

- **Automated GRN identification** from FlyWire Codex labels
- **Downstream connectivity extraction** via FlyWire API
- **Statistical analysis** with effect sizes and hypothesis testing
- **Publication-ready visualizations** (5 figure types)
- **Comprehensive caching** to minimize API calls
- **Demo mode** for quick testing
- **Complete testing suite** with 5 validation tests

## Installation

### Requirements

- Python 3.9+
- FlyWire access (optional - synthetic data available for demo)

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn networkx scipy pyyaml tqdm psutil
pip install fafbseg  # Optional, for real FlyWire connectivity
```

Or install from requirements file:

```bash
cd pgcn/connectivity_viz
pip install -r requirements.txt
```

## Quick Start

### 1. Demo Mode (No Data Required)

Run with synthetic data to test the pipeline:

```bash
cd pgcn/connectivity_viz
python main.py --demo-mode
```

This will:
- Generate synthetic GRN connectivity data
- Analyze 3 sugar GRNs and 5 total GRNs
- Create all 5 visualizations
- Save results to `reports/`

### 2. Full Analysis (With Real Data)

```bash
# Make sure processed_labels.csv.gz is in data/ directory
python main.py --full-analysis --csv-path ../../data/processed_labels.csv.gz
```

### 3. Step-by-Step Execution

```bash
# Step 1: Extract GRNs and fetch connectivity
python main.py --extract-only --csv-path ../../data/processed_labels.csv.gz

# Step 2: Analyze and visualize (uses cached data)
python main.py --visualize-only
```

## Pipeline Phases

### Phase 1: GRN Identification

- Loads `processed_labels.csv.gz` from FlyWire Codex
- Filters for sugar GRNs (keywords: sugar, sweet, gr5a, gr64)
- Filters for all GRNs (keywords: gustatory, taste, grn)
- Saves filtered lists to cache

**Output:**
- `data/cache/sugar_grns_metadata.csv`
- `data/cache/all_grns_metadata.csv`

### Phase 2: Connectivity Extraction

- Fetches downstream synaptic partners for each GRN
- Retrieves synapse counts (weights)
- Attempts to label partners (KC, MBON, LH, etc.)
- Caches all connectivity data

**Output:**
- `data/cache/downstream_connectivity.pkl`

### Phase 3: Statistical Analysis

- Computes per-GRN statistics (partner counts, synapse distributions)
- Population-level statistics
- Convergence analysis (how many GRNs target same neurons)
- Statistical comparison: Sugar GRNs vs All GRNs
  - Mann-Whitney U tests
  - Cohen's d effect sizes
  - 95% confidence intervals

**Output:**
- `reports/sugar_per_grn_stats_TIMESTAMP.csv`
- `reports/sugar_vs_all_comparison_TIMESTAMP.csv`
- `reports/sugar_downstream_partners_TIMESTAMP.csv`
- `reports/sugar_connectivity_summary_TIMESTAMP.txt`

### Phase 4: Visualization

Generates 5 publication-ready figures (300 DPI):

1. **GRN Summary** (`grn_connectivity_overview.png`)
   - Bar chart: downstream partners per GRN
   - Sorted by partner count
   - Color-coded by GRN category

2. **Sugar vs All Comparison** (`sugar_vs_all_grns_comparison.png`)
   - Violin plots: partner count distribution
   - Box plots: partner count distribution
   - Violin plots: synapse count distribution
   - Cumulative distribution curves
   - Statistical test results overlaid

3. **Network Graph** (`grn_downstream_network.png`)
   - Force-directed layout
   - GRNs (red) → Partners (colored by type)
   - Edge width = synapse count (log scale)
   - Node size = convergence degree

4. **Connectivity Heatmap** (`grn_connectivity_heatmap.png`)
   - Rows: GRNs
   - Columns: Top 30 downstream partners
   - Values: Synapse counts (log scale)
   - Hierarchical clustering with dendrograms

5. **Analysis Dashboard** (`grn_analysis_dashboard.png`)
   - 6 subplots:
     - Partner count histogram
     - Synapse count histogram
     - Partner count by GRN category
     - Cumulative distribution
     - Partner type pie chart
     - Statistics summary text box

## Configuration

All parameters are configurable via `pgcn/configs/grn_viz_config.yaml`:

```yaml
data:
  processed_labels_path: 'data/processed_labels.csv.gz'
  cache_dir: 'data/cache'
  output_dir: 'reports'

filters:
  sugar_grn_keywords: ['sugar', 'sweet', 'gr5a', 'gr64']
  gustatory_keywords: ['gustatory', 'grn', 'taste']

visualization:
  figure_dpi: 300
  synapse_log_scale: true
  heatmap_cmap: 'YlOrRd'
  network_layout: 'spring'

# ... and many more options
```

## Testing

Run comprehensive validation tests:

```bash
python test_pipeline.py
```

This executes 5 tests:

1. **Data Integrity**: Validates GRN lists (no duplicates, valid IDs, correct labels)
2. **Connectivity Structure**: Validates connectivity dict format
3. **Statistical Validation**: Checks distributions are reasonable
4. **Comparison Test**: Verifies sugar GRNs ⊂ all GRNs
5. **Visualization Output**: Confirms all figures generated (>50 KB each)

Expected output:
```
================================================================================
TEST SUMMARY
================================================================================
Tests run:    5
Tests passed: 5 ✅
Tests failed: 0 ❌

🎉 ALL TESTS PASSED! 🎉
================================================================================
```

## Module Documentation

### `grn_downstream_extractor.py`

Main class: `GRNDownstreamExtractor`

**Key methods:**
- `load_and_filter_grns(csv_gz_path, sugar_only=False)`: Load and filter GRNs
- `fetch_downstream_connectivity(grn_list, use_cache=True)`: Fetch connectivity
- `save_grn_lists(sugar_grns, all_grns)`: Save metadata to cache

### `grn_connectivity_analyzer.py`

Main class: `GRNConnectivityAnalyzer`

**Key methods:**
- `compute_connectivity_statistics(connectivity_dict, grn_df)`: Compute stats
- `compare_sugar_vs_all(...)`: Statistical comparison
- `generate_partner_table(connectivity_dict, top_n=30)`: Top partners table
- `save_statistics(...)`: Save all results to CSV

### `grn_visualization.py`

Main class: `GRNConnectivityVisualizer`

**Key methods:**
- `plot_grn_summary(...)`: Figure 1 (bar chart)
- `plot_comparison(...)`: Figure 2 (violin/box plots)
- `plot_network_graph(...)`: Figure 3 (network)
- `plot_heatmap_connectivity(...)`: Figure 4 (heatmap)
- `plot_analysis_dashboard(...)`: Figure 5 (dashboard)

### `main.py`

CLI entry point with subcommands:
- `--full-analysis`: Run complete pipeline
- `--demo-mode`: Run with synthetic data
- `--extract-only`: Only extract GRNs and connectivity
- `--visualize-only`: Only generate visualizations (requires cache)

## Expected Output Summary

After running `--full-analysis` or `--demo-mode`:

```
================================================================================
ANALYSIS COMPLETE
================================================================================
Sugar GRNs analyzed:           23
All GRNs analyzed:             47
Total downstream partners:     234 (unique)
Figures saved:                 5
Execution time:                4 min 32 sec
Memory peak:                   450 MB
================================================================================
Output directory: reports/
Key files:
  - grn_connectivity_overview.png
  - sugar_vs_all_grns_comparison.png
  - grn_downstream_network.png
  - grn_connectivity_heatmap.png
  - grn_analysis_dashboard.png
================================================================================
```

## Troubleshooting

### `fafbseg` Not Available

If you don't have FlyWire API access:
1. Run in demo mode: `python main.py --demo-mode`
2. Pipeline will generate synthetic connectivity data
3. All visualizations will still be created

### Missing `processed_labels.csv.gz`

Download from FlyWire Codex:
```bash
# Expected location: data/processed_labels.csv.gz
# Or specify path: --csv-path /path/to/processed_labels.csv.gz
```

### Memory Issues

For large datasets:
1. Increase cache usage: Set `cache.use_cache: true` in config
2. Run in steps: `--extract-only` then `--visualize-only`
3. Reduce network graph nodes: Set `visualization.network.max_nodes: 100`

### Cache Cleanup

```bash
# Clear all cached data
rm -rf data/cache/*
```

## Citation

If you use this pipeline in your research, please cite:

```
GRN Downstream Connectivity Visualization Pipeline
Generated by Claude AI, 2025
https://github.com/your-repo/PGCN
```

## Author

Claude AI (Anthropic)
Date: 2025-11-03

## License

MIT License - See project root for details
