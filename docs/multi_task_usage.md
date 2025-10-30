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

All tasks consume PN activity tensors aligned to the FlyWire v783 connectome. The loader
now probes the cache to determine the PN/KC counts at runtime, so expect the values below
to update when the FlyWire exports change. With the current v783 cache:

- **Input dimensionality:** cache-derived PN count
- **Reservoir KC dimensionality:** cache-derived KC count
- **MBON readout dimensionality:** configuration-specified (96 for v783)

Feature tables must store PN activity with one row per behavioural trial (440 rows for the
canonical dataset). Provide the columns detailed below:

| Column            | Description                                                   |
|-------------------|---------------------------------------------------------------|
| `pn_0` … `pn_{N-1}` | PN activity vector aligned with FlyWire indices               |
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
- Behavioural targets remain single-column in the Parquet export; the loader now expands
  those labels to match the 96 MBON outputs whenever the head reuses the reservoir readout.
- New tasks should typically leverage `parquet_tensor` loader with their own targets.
- Adjust batch size, epochs, and learning rate per task as required.
- Reservoir fields (`n_pn`, `n_kc`, `n_mbon`, `sparsity`) act as hints. When `cache_dir`
  is provided the loader instantiates `DrosophilaReservoir` to read the PN/KC counts
  directly from disk and overrides mismatched configuration values.
- Task `input_dim` values follow the same rule. To pin a bespoke dimension (for example,
  when working with reduced PN projections), set `lock_input_dim: true` within the task
  block; otherwise the cache-derived PN count replaces the configured value.
- `freeze_pn_kc: true` only locks the developmental PN→KC matrix; the shared KC→MBON
  weights remain plastic so the behavioural head continues to update during training.

## 4. Generate feature tables

The trainer expects a Parquet feature table for every configured task. The helper below
emits deterministic, sparse PN representations that satisfy the dimensionality checks in
`TaskDataLoaderFactory`. Provide the behavioural CSV if it lives outside the repository
defaults:

```bash
python scripts/generate_multi_task_features.py \
  --config configs/multi_task_config.yaml \
  --behavior-csv /path/to/model_predictions.csv \
  --report-json artifacts/multi_task/feature_report.json
```

- Behavioural tasks reuse the 440 behavioural trials and attach hashed PN activations.
- Synthetic heads receive reproducible sparse features (2048 rows by default). Use
  `--rows` to align with external datasets and `--overwrite` when regenerating tables.

## 5. Training

```bash
python scripts/train_multi_task.py \
  --config configs/multi_task_config.yaml \
  --output-dir artifacts/multi_task \
  --tasks olfactory_conditioning spatial_navigation
```

The trainer performs sequential optimisation per task, freezing PN→KC projections (while
keeping KC→MBON gradients enabled) and adhering to 5% KC sparsity. Outputs:

- `artifacts/multi_task/multi_task_model.pt` – consolidated state dict
- `artifacts/multi_task/training_history.json` – per-epoch loss curves

## 5. Behaviour–Connectome Alignment

Use the dedicated CLI to correlate behavioural accuracy with connectome structure.
Glomerulus labels are inferred from the FlyWire cache when the CSV argument is
omitted or unavailable, so the option is only mandatory for custom annotations.
You must supply a trial→glomerulus mapping (either by providing `--trial-to-glomerulus`
or embedding a `trial_label` column within the assignments CSV); otherwise the tool
terminates with a descriptive error instead of producing empty reports:

```bash
python analysis/behavior_connectome_analysis.py \
  --cache-dir data/cache \
  --trial-to-glomerulus configs/trial_to_glomerulus.yaml \
  --output-dir artifacts/behavior_connectome
```

The script produces enrichment statistics and Pearson correlations linking glomerular
fan-in with behavioural success rates. When you supply a CSV, ensure it contains
`pn_index` and `glomerulus` columns; otherwise the tool will mine the metadata embedded
within `data/cache/nodes.parquet`. Optional mapping files (YAML/JSON) associate
`trial_label` with glomeruli. The repository ships with a placeholder mapping at
`configs/trial_to_glomerulus.yaml`; customise the values to reflect your FlyWire or
neuPrint annotations before executing the CLI.

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
- Feature tables must match the PN count reported by the cache. Regenerate the Parquet
  files with `scripts/generate_multi_task_features.py --overwrite` whenever `pgcn-cache`
  introduces a new connectome export. The loader raises descriptive errors when shapes
  or row counts drift from expectations.
- Treat `UserWarning: Overriding configured n_pn ...` as a red flag. The tightened
  FlyWire filters only emit those overrides when the cache still contains
  non-olfactory projections or glomerulus-free neurons. Re-run
  `scripts/inspect_flywire_datasets.py` and fix the annotations until the cache reports
  exactly 10 767 projection neurons and 5 177 Kenyon cells.

## 8. Custom Loader Interface

When registering bespoke loaders via `TaskDataLoaderFactory.register`, implement the
signature `def loader(task: TaskSpec, *, shuffle: bool) -> DataLoader`. The factory now
invokes loaders with a keyword argument to make the intention explicit and avoid
positional mismatches. Reuse `_tabular_loader` as a template when adapting future PN
feature sources.

## 9. Next Steps

- Integrate glomerulus-specific metadata from FlyWire or neuPrint to enrich the
  `BehaviorConnectomeAnalyzer` inputs.
- Extend `TaskDataLoaderFactory` with custom loaders for calcium imaging or optogenetic
  datasets as they become available.
- Automate publishing of enrichment plots via the existing `analysis` tooling to support
  manuscript-ready figures.
