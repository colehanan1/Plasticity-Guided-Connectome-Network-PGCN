# Plasticity-Guided Connectome Network (PGCN)

The Plasticity-Guided Connectome Network repository provides a reproducible
pipeline for extracting and analysing the FlyWire projection-neuron
subgraph (PN→KC→MBON core plus DAN ancillary pathways). The codebase pins
FlyWire materialization versions, writes schema-stable cache artefacts, and
exposes command line interfaces for cache generation and structural metrics.

## Quickstart

1. **Create the Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate PGCN
   ```

2. **Build the connectome cache**

   ```bash
   pgcn-cache --datastack flywire_fafb_production --mv 783 --out data/cache/
   ```

   Use `--use-sample-data` for an offline deterministic cache suitable for
   testing when FlyWire authentication is unavailable.

3. **Compute structural metrics**

   ```bash
   pgcn-metrics --cache-dir data/cache/
   ```

4. **Run the unit tests**

   ```bash
   pytest -q
   ```

## Chemical Odor Generalisation Toolkit

The repository ships an optional chemical modelling stack for reproducing
odor-generalisation analyses. The modules live under `pgcn.chemical` and
`pgcn.models` and are fully importable once PyTorch is available.

1. **Install the project in editable mode (adds PyTorch via the `models` extra)**

   Run this command from the repository root so `pip` can resolve the local
   package:

   ```bash
   pip install -e ".[models]" --find-links https://download.pytorch.org/whl/cpu
   ```

   The project is not published on PyPI; editable installation ensures the
   `pgcn` package is importable from your working tree. The extra pulls in a
   CPU build of PyTorch. Swap the `--find-links` URL for the CUDA-specific index
   if you require GPU acceleration.

2. **Run the chemical unit tests**

   ```bash
   pytest tests/test_chemical.py -q
   ```

3. **Execute the reference modelling snippet**

   ```bash
   python - <<'PY'
   from pgcn.models import ChemicallyInformedDrosophilaModel

   model = ChemicallyInformedDrosophilaModel()
   prediction = model.predict("ethyl_butyrate", "hexanol")
   print(f"Predicted generalisation probability: {prediction:.3f}")
   PY
   ```

   Substitute any other training/testing odor combination that appears in
   `pgcn.chemical.COMPLETE_ODOR_MAPPINGS` to probe cross-generalisation
   behaviour. The model automatically loads the curated chemical descriptors and
   similarity priors included in the repository.

## Repository Structure

```
PGCN/
├── src/pgcn/                     # Python package
│   ├── __init__.py
│   ├── connectome_pipeline.py     # Cache construction and CLI
│   └── metrics.py                 # Structural metrics and CLI
├── tests/                        # Pytest suite
│   └── test_cache.py
├── data/
│   └── cache/                    # Cache output directory (git-kept, empty)
├── environment.yml               # Conda specification (name=PGCN)
├── pyproject.toml                # Packaging, formatting, and entry points
├── Makefile                      # Common workflows
├── README.md                     # This document
└── data_schema.md                # Cache and metrics schema reference
```

## Cache Outputs

`pgcn-cache` writes the following artefacts into the selected cache directory:

- `nodes.parquet`: node-level metadata with `node_id`, `type`, `glomerulus`,
  `synapse_count`, and centroid coordinates (`x`, `y`, `z`).
- `edges.parquet`: PN→KC and KC→MBON edges with synapse weights.
- `dan_edges.parquet`: DAN→KC and DAN→MBON edges with synapse weights.
- `meta.json`: datastack, materialization version, and table provenance.

All parquet files follow the schemas defined in `data_schema.md`.

## Metrics Outputs

`pgcn-metrics` expects a cache directory and produces:

- `kc_overlap.parquet`: glomerulus-wise Kenyon-cell Jaccard overlaps.
- `pn_kc_mbon_paths.parquet`: PN→KC→MBON two-hop path summary.
- `weighted_centrality.parquet`: weighted in/out degree and betweenness.
- `dan_valence.parquet`: PAM/PPL1/DAN-other valence labels.
- `metrics_meta.json`: row counts for quick validation.

## Authentication Notes

FlyWire access requires a valid CAVE token stored at
`~/.cloudvolume/secrets/cave-secret.json`. The pipeline validates the token,
selects materialization version 783 when available, and records the active
version in the cache metadata. When version 783 is missing the latest
available version is selected and the fallback is logged explicitly.

## Testing Strategy

Unit tests fabricate a deterministic cache to verify schema integrity,
positive weights, and the absence of direct PN→MBON edges. The `--use-sample-data`
flag mirrors this behaviour for developers working offline.
