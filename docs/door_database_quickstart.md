# DoOR Database Quickstart

Fast instructions for loading and using the DoOR v2.0 database with the existing PGCN tooling.

## Setup
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt` (rpy2 optional but preferred)
- Default cache: `data/door_cache/` (created automatically)

## Choose a load method
- `rpy2` (preferred): pulls data from R packages; fastest once installed.
- `csv`: loads a pre-exported `door_response_matrix.csv` (put in `data/door_cache/`).
- `zenodo`: downloads the official archive when nothing else is available.

## First-time install (rpy2 path)
```bash
python - <<'PY'
from pgcn.door import DoORDataManager
manager = DoORDataManager(method="rpy2")
manager.install_door_packages()  # one-time R install
door = manager.load_door_data()
print(door['response_matrix'].shape)
PY
```

## CSV/offline path
```bash
# Place or symlink your matrix to data/door_cache/door_response_matrix.csv
python - <<'PY'
from pgcn.door import DoORDataManager
door = DoORDataManager(method="csv", backup_csv_path="data/door_cache/door_response_matrix.csv")
data = door.load_door_data()
print("Odorants × receptors:", data["response_matrix"].shape)
print("Or42b top ligands:", data["response_matrix"]["Or42b"].nlargest(5).to_dict())
PY
```

## Where this is used
- `scripts/complete_orn_analysis.py` – end-to-end ORN + DoOR analysis (supports `--door-method` and `--door-csv`)
- `src/pgcn/door/door_data_manager.py` – API docs and logging
- `src/pgcn/door/orn_identifier.py` – receptor matching against FlyWire labels
- `src/pgcn/door/pathway_analyzer.py` – downstream tracing hooks when connectivity is supplied

## Tips and pitfalls
- If rpy2 is missing, switch to `--door-method csv` to stay offline.
- Response values are normalized [0,1]; use thresholds accordingly (e.g., 0.3 for “responsive”).
- Cache size is small; safe to commit paths but not the matrix itself.
- When DoOR is unavailable, degrade gracefully by skipping odor-response-dependent analyses instead of fabricating data.
