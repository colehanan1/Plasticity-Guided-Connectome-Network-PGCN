# Root ID → Downstream Connectivity Guide

Use this guide to map any `root_ids_*.txt` file (e.g., `data/flywire/root_ids_or67b.txt`) to downstream partners, receptor identities, and DoOR response profiles using existing tooling.

## Prerequisites
- Environment: `python -m venv .venv && source .venv/bin/activate` and `pip install -r requirements.txt`
- FlyWire labels: `data/flywire/processed_labels.csv.gz` (must contain `root_id` and `processed_labels`)
- Connectivity (optional but recommended): CSV with `pre_root_id`, `post_root_id`, `syn_count`
- DoOR response matrix (optional): `data/cache/door_response_matrix.csv` or `data/door_cache/door_response_matrix.csv`
- Connectome cache for PN/KC work: run `pgcn-cache --local-data data/flywire --out data/cache/`

## Quick CLI: map a root ID list to receptor types and DoOR tuning
```bash
python scripts/analyze_alrn_mxlbn_orn_feeding.py \
  --root-ids data/flywire/root_ids_or67b.txt \
  --labels data/flywire/processed_labels.csv.gz \
  --door-path data/cache/door_response_matrix.csv \
  --output-dir data/analysis \
  --results-dir results
```
Outputs:
- `data/analysis/orn_alrn_mxlbn_counts.csv` – receptor counts for your root IDs
- `data/analysis/alrn_mxlbn_orn_analysis.csv` – tuning breadth + feeding-odor stats
- `results/alrn_mxlbn_hypothesis_test.png` – quick visualization

## Pathway + downstream partners (programmatic)
```python
import pandas as pd
from pathlib import Path
from pgcn.door import DoORDataManager, ORNBehaviorPathwayAnalyzer

root_ids = [int(r.strip()) for r in Path("data/flywire/root_ids_or67b.txt").read_text().split(",") if r.strip()]
labels = pd.read_csv("data/flywire/processed_labels.csv.gz")
connectivity = pd.read_csv("data/flywire/connectivity.csv")  # columns: pre_root_id, post_root_id, syn_count

door = DoORDataManager(method="csv", backup_csv_path="data/door_cache/door_response_matrix.csv")
analyzer = ORNBehaviorPathwayAnalyzer(door_manager=door, identified_cells=labels, connectivity_data=connectivity)

downstream = analyzer._trace_downstream_connectivity(root_ids)  # reports unique targets and counts
print(downstream)
```
Notes:
- `_trace_downstream_connectivity` expects integer root IDs and returns total/unique targets plus basic categorization.
- Add cell-type labels to `connectivity` to replace the placeholder categorization.
- For multi-hop tracing to motor outputs, extend `_trace_to_motor_neurons` once full connectivity matrices are available.

## Full DoOR-FlyWire report
For receptor validation, ligand ranking, and downstream counts in one shot:
```bash
python scripts/complete_orn_analysis.py \
  --labels data/flywire/processed_labels.csv.gz \
  --connectivity data/flywire/connectivity.csv \
  --door-method csv \
  --door-csv data/door_cache/door_response_matrix.csv \
  --output results/orn_analysis
```
This runs DoOR validation, Or42b/Or47b pathway analysis, and includes downstream target summaries when connectivity is supplied.

## Troubleshooting
- Empty results: ensure root IDs are comma-separated (no newlines) and exist in `processed_labels`.
- Missing connectivity stats: check that the connectivity CSV has `pre_root_id` and `post_root_id` columns with integer types.
- DoOR not found: point `--door-csv` to a local `door_response_matrix.csv` or install via `DoORDataManager.install_door_packages()`.
