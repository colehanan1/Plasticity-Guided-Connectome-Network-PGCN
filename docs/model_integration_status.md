# PGCN Model Integration Status

This document tracks the state of the Plasticity-Guided Connectome Network (PGCN)
when combining FlyWire connectome exports with the behavioral conditioning data.
It clarifies what is already reproducible, what artefacts the tooling produces,
and which validation steps still rely on future work.

## Current data sources

- **Connectome**: FAFB v783 FlyWire CSV exports placed in `data/flywire/`. The
  repository normalises header conventions, infers PN glomerulus labels, and
  builds KC/PN/MBON/DAN node tables without external API access.
- **Behavior**: `data/behavior/model_predictions.csv` (440 trials, 35 flies)
  remains the canonical dataset for odor-conditioned responses. Grouped
  cross-validation still uses `fly` as the grouping key to avoid data leakage.

## Reproducible pipeline (end-to-end)

1. **Dataset inspection** – `python scripts/inspect_flywire_datasets.py --data-dir data/flywire \
     > flywire_dataset_report.txt`
   - Confirms column headers, dtype compatibility, and PN/KC discovery heuristics
     on the downloaded CSVs.
2. **Cache materialisation** – `pgcn-cache --local-data data/flywire --out data/cache/`
   - Emits `nodes.parquet`, `edges.parquet`, `dan_edges.parquet`, and `meta.json`
     with PN/KC/MBON/DAN populations, neurotransmitter annotations, inferred
     glomeruli, and provenance metadata (counts + source directory).
3. **Structural metrics** – `pgcn-metrics --cache-dir data/cache/`
   - Consumes the offline cache without warnings; reports PN/KC overlap tables,
     KC sparsity, MBON/DAN coverage, and writes metrics artefacts alongside the
     cache outputs.
4. **Reservoir initialisation** –
   ```python
   from pathlib import Path
   from pgcn.models.reservoir import DrosophilaReservoir

   reservoir = DrosophilaReservoir(cache_dir=Path("data/cache"))
   print(reservoir.pn_to_kc.weight.shape)  # torch.Size([5177, 10767])
   print(reservoir.pn_kc_mask.sum().item())  # 22_244 PN→KC edges
   ```
   - Confirms the model consumes the cached artefacts and exposes the expected
     connectivity tensors for training loops.
5. **Behavioral evaluation** – `python analysis/cross_validation.py --folds 5`
   - Runs GroupKFold cross-validation with connectome-informed reservoirs to
     verify behavioral predictions remain aligned with reported literature.

## What already works

- Local FlyWire CSVs now produce schema-stable caches identical in structure to
  the authenticated pipeline, including PN glomerulus tags and node/edge counts
  in `meta.json` required by downstream analytics.
- `pgcn-metrics` executes without placeholder warnings because glomerulus
  coverage is inferred offline.
- The behavioral cross-validation script operates unchanged; it reads the cached
  connectivity, constructs the constrained reservoir, and trains MBON output
  heads while logging accuracy/auroc per condition.

## Outstanding items

- **Behavior-to-connectome alignment**: incorporate statistical summaries that
  jointly reference behavioral performance and structural motifs (e.g., PN
  glomerulus enrichment vs. odor-specific success rates).
- **Extended validation**: automate consistency checks comparing the offline
  cache to a reference API-derived cache where access exists, ensuring regression
  protection as FlyWire releases evolve.
- **Documentation**: broaden usage examples that fuse behavior + connectome
  results into unified figures for publication.

When these items are complete, the repository will deliver a fully offline,
biologically-grounded training workflow that aligns connectome connectivity with
observed behavior, closing the loop for KC/PN plasticity studies.
