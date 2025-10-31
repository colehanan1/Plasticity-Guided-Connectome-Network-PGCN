# Execution Checklist — Plasticity-Guided Connectome Network

## Daily Startup
- [ ] Activate project virtual environment (`source .venv/bin/activate` or `conda activate pgcn`).
- [ ] Sync FlyWire cache location (`export PGCN_FLYWIRE_DATA=/abs/path/to/data/flywire`).
- [ ] Run smoke tests for extraction scripts (`python -m compileall scripts/extract_circuit.py scripts/extract_alpn_projection_neurons.py`).
- [ ] Review `PROJECT_STATUS.md` for outstanding deliverables.

## Phase 1 Focus (Connectivity Backbone)
- [ ] Implement `models/connectivity_matrix.py` and unit tests.
- [ ] Implement `data_loaders/circuit_loader.py` with validation coverage.
- [ ] Implement `models/olfactory_circuit.py` minimal pathway propagation.
- [ ] Add fixtures + tests in `tests/models/` and `tests/integration/`.
- [ ] Create `make connectivity` target (smoke + unit tests).
- [ ] Document new modules in `README.md` and `docs/model_integration_status.md`.

## Quality Gates
- [ ] `pytest tests/models tests/integration -q` passes on local fixtures.
- [ ] Lint check for new modules (`ruff check models data_loaders`).
- [ ] Type hints validated (`mypy models data_loaders`).
- [ ] Updated documentation reviewed for accuracy and command reproducibility.

## Pre-Commit Wrap-Up
- [ ] Re-run extraction scripts if schema changes affect cache format.
- [ ] Update `PROJECT_STATUS.md` progress percentages and counts if new data discovered.
- [ ] Ensure README references remain current (commands, artefact names).
- [ ] Prepare concise PR summary referencing new documentation.
