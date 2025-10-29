# Multi-Task PGCN Usage Guide

This document describes how to operate the multi-task extension of the Plasticity-Guided
Connectome Network (PGCN). The workflow augments the biologically constrained
`DrosophilaReservoir` with dedicated task heads while preserving the FlyWire-derived
connectome. Use this guide in tandem with `docs/model_integration_status.md` to keep
connectome validation and behavioural analysis aligned.

## 1. Environment Preparation

```bash
conda env create -f environment.yml  # includes FastAPI + PyYAML
conda activate PGCN
pip install -e .[models,api]         # install torch + deployment extras
```

The optional `api` extra installs FastAPI/uvicorn for serving predictions. The `models`
extra ensures PyTorch is available for training the readout heads.

## 2. Data Expectations

All tasks consume PN activity tensors aligned to the FlyWire v783 connectome:

- **Input dimensionality:** 10,767 PN features per trial
- **Reservoir KC dimensionality:** 5,177 Kenyon Cells (KC)
- **MBON readout dimensionality:** 96 outputs shared with behavioural conditioning

Feature tables must store PN activity with one row per behavioural trial (440 rows for the
canonical dataset). Provide the columns detailed below:

| Column            | Description                                                   |
|-------------------|---------------------------------------------------------------|
| `pn_0` … `pn_10766` | PN activity vector aligned with FlyWire indices               |
| `prediction`      | Binary conditioning outcome (0/1) for olfactory task          |
| `target` / `label` | Integer or float target specific to the auxiliary task       |
| Additional metadata | Optional contextual columns (ignored by the loader)        |

The table can be stored as Parquet (`.parquet`) or CSV/TSV.

## 3. Configuration (`configs/multi_task_config.yaml`)

Each task entry defines its input/output dimensionality, loader type, and optimisation
hyper-parameters. Example excerpt:

```yaml
reservoir:
  cache_dir: data/cache
  n_pn: 10767
  n_kc: 5177
  n_mbon: 96
  sparsity: 0.05
  freeze_pn_kc: true

tasks:
  olfactory_conditioning:
    input_dim: 10767
    output_dim: 96
    loss_function: binary_crossentropy
    data_loader: behavioral_data
    feature_table: data/cache/olfactory_conditioning_features.parquet
    target_column: prediction
    use_reservoir_head: true
```

Key points:

- `use_reservoir_head: true` reuses the canonical KC→MBON layer to respect plasticity.
- New tasks should typically leverage `parquet_tensor` loader with their own targets.
- Adjust batch size, epochs, and learning rate per task as required.

## 4. Training

```bash
python scripts/train_multi_task.py \
  --config configs/multi_task_config.yaml \
  --output-dir artifacts/multi_task \
  --tasks olfactory_conditioning spatial_navigation
```

The trainer performs sequential optimisation per task, freezing PN→KC projections and
adhering to 5% KC sparsity. Outputs:

- `artifacts/multi_task/multi_task_model.pt` – consolidated state dict
- `artifacts/multi_task/training_history.json` – per-epoch loss curves

## 5. Behaviour–Connectome Alignment

Use the dedicated CLI to correlate behavioural accuracy with connectome structure:

```bash
python analysis/behavior_connectome_analysis.py \
  --cache-dir data/cache \
  --glomerulus-assignments data/connectome/pn_glomerulus.csv \
  --trial-to-glomerulus configs/trial_to_glomerulus.yaml \
  --output-dir artifacts/behavior_connectome
```

The script produces enrichment statistics and Pearson correlations linking glomerular
fan-in with behavioural success rates. Provide a CSV with columns `pn_index` and
`glomerulus`. Optional mapping files (YAML/JSON) associate `trial_label` with glomeruli.

## 6. Deployment

```bash
python scripts/deploy_model_server.py \
  --config configs/multi_task_config.yaml \
  --checkpoint artifacts/multi_task/multi_task_model.pt \
  --host 0.0.0.0 --port 8080
```

Endpoints:

- `GET /tasks` – list available task heads
- `POST /predict` – body `{"task": "reward_prediction", "pn_activity": [[...]]}`

Set `apply_activation` to `false` to retrieve raw logits. Enable `return_kc` to inspect
KC sparsity compliance per query.

## 7. Quality Checks

- `analysis/behavior_connectome_analysis.py` verifies behavioural alignment.
- `scripts/train_multi_task.py` enforces PN→KC freezing and KC sparsity via the
  shared reservoir wrapper.
- Feature tables must contain exactly 10,767 PN columns. The loader raises descriptive
  errors when shapes or row counts drift from expectations.

## 8. Next Steps

- Integrate glomerulus-specific metadata from FlyWire or neuPrint to enrich the
  `BehaviorConnectomeAnalyzer` inputs.
- Extend `TaskDataLoaderFactory` with custom loaders for calcium imaging or optogenetic
  datasets as they become available.
- Automate publishing of enrichment plots via the existing `analysis` tooling to support
  manuscript-ready figures.
