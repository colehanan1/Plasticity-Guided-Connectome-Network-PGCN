#!/bin/bash
# Example script for running LN/PN connectivity analysis
# This demonstrates various usage patterns

set -e  # Exit on error

echo "=========================================="
echo "LN/PN Connectivity Analysis Examples"
echo "=========================================="

# Example 1: Basic analysis with default settings
echo -e "\n[Example 1] Basic analysis with defaults..."
python scripts/analyze_ln_pn_connectivity.py \
  --data-dir data/flywire \
  --output-dir results/ln_pn_analysis_default

# Example 2: Conservative threshold (published standard)
echo -e "\n[Example 2] Conservative analysis (min 3 synapses)..."
python scripts/analyze_ln_pn_connectivity.py \
  --data-dir data/flywire \
  --output-dir results/ln_pn_analysis_threshold3 \
  --min-synapses 3

# Example 3: Top glomeruli only (cleaner visualizations)
echo -e "\n[Example 3] Focus on top 20 glomeruli..."
python scripts/analyze_ln_pn_connectivity.py \
  --data-dir data/flywire \
  --output-dir results/ln_pn_analysis_top20 \
  --top-glomeruli 20

# Example 4: Very conservative (strong connections only)
echo -e "\n[Example 4] Strong connections only (min 5 synapses, top 15)..."
python scripts/analyze_ln_pn_connectivity.py \
  --data-dir data/flywire \
  --output-dir results/ln_pn_analysis_strong \
  --min-synapses 5 \
  --top-glomeruli 15

echo -e "\n=========================================="
echo "Analysis complete! Check results directories:"
echo "  - results/ln_pn_analysis_default/"
echo "  - results/ln_pn_analysis_threshold3/"
echo "  - results/ln_pn_analysis_top20/"
echo "  - results/ln_pn_analysis_strong/"
echo "=========================================="

# Optional: Create comparison summary
echo -e "\nGenerating comparison summary..."
python - <<EOF
import pandas as pd
from pathlib import Path

print("\n=== Analysis Comparison ===\n")

analyses = [
    ('Default (≥1 syn)', 'results/ln_pn_analysis_default'),
    ('Threshold 3', 'results/ln_pn_analysis_threshold3'),
    ('Top 20 Glom', 'results/ln_pn_analysis_top20'),
    ('Strong (≥5 syn)', 'results/ln_pn_analysis_strong'),
]

for name, path in analyses:
    try:
        ln_path = Path(path) / 'ln_cross_glomerular_connections.csv'
        pn_path = Path(path) / 'pn_downstream_targets.csv'

        if ln_path.exists() and pn_path.exists():
            ln_df = pd.read_csv(ln_path)
            pn_df = pd.read_csv(pn_path)

            print(f"{name}:")
            print(f"  LN connections: {len(ln_df):,} pairs")
            print(f"  Total LN synapses: {ln_df['total_synapses'].sum():,}")
            print(f"  PN connections: {len(pn_df):,}")
            print(f"  Glomeruli: {pn_df['glomerulus'].nunique()}")
            print()
    except Exception as e:
        print(f"{name}: Error - {e}\n")

print("=== Top 5 LN-Mediated Connections (Default) ===\n")
try:
    ln_df = pd.read_csv('results/ln_pn_analysis_default/ln_cross_glomerular_connections.csv')
    top5 = ln_df.nlargest(5, 'total_synapses')
    for _, row in top5.iterrows():
        print(f"{row['source_glom']} → {row['target_glom']}: "
              f"{row['ln_count']} LNs, {row['total_synapses']} synapses")
except Exception as e:
    print(f"Error: {e}")

EOF

echo -e "\nDone!"
