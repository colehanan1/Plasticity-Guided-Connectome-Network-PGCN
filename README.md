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

   This command requires a FlyWire CAVE token stored as JSON at
   `~/.cloudvolume/secrets/cave-secret.json`. Create the directory (if it does
   not already exist) and place your token inside the `token` field:

   ```bash
   mkdir -p ~/.cloudvolume/secrets
   cat <<'EOF' > ~/.cloudvolume/secrets/cave-secret.json
   {"token": "<paste-your-flywire-token-here>"}
   EOF
   ```

   Use `--use-sample-data` for an offline deterministic cache when a FlyWire
   account or network access is unavailable:

   ```bash
   pgcn-cache --use-sample-data --out data/cache/
   ```

3. **Compute structural metrics**

   ```bash
   pgcn-metrics --cache-dir data/cache/
   ```

4. **Run the unit tests**

   ```bash
   pytest -q
   ```

5. **Validate reservoir weight hydration**

   ```bash
   pytest tests/test_reservoir.py -q
   ```

   The dedicated reservoir tests fabricate a minimal connectome cache and a
   precomputed PN→KC matrix to confirm that weights are normalised, masks are
   preserved, and gradients remain frozen.

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

4. **Hydrate the Drosophila reservoir from a connectome cache**

   With the cache generated via `pgcn-cache`, the reservoir will ingest the
   PN→KC weights directly and respect the native sparsity mask:

   ```bash
   python - <<'PY'
   from pathlib import Path

   from pgcn.models.reservoir import DrosophilaReservoir

   reservoir = DrosophilaReservoir(cache_dir=Path("data/cache"))
   print("PN→KC weight shape:", tuple(reservoir.pn_to_kc.weight.shape))
   density = reservoir.pn_kc_mask.float().mean().item()
   print("Mask density:", density)
   print("Gradients frozen:", not reservoir.pn_to_kc.weight.requires_grad)
   PY
   ```

   The reported mask density reflects the Kenyon cell sparsity encoded in the
   cache (≈5 % for hemibrain-derived datasets), and the PN→KC parameters should
   report frozen gradients.

5. **Load a pre-parsed PN→KC matrix**

   When working with precomputed connectivity matrices (for example, custom
   normalisations or ablation studies), feed them directly into the reservoir:

   ```bash
   python - <<'PY'
   import numpy as np

   # Replace this with your own PN→KC weight matrix construction logic.
   # The saved array must have shape (n_kc, n_pn) and non-negative weights.
   matrix = np.random.rand(2000, 50)
   np.save("custom_pn_to_kc.npy", matrix)
   PY

   python - <<'PY'
   import numpy as np

   from pgcn.models.reservoir import DrosophilaReservoir

   matrix = np.load("custom_pn_to_kc.npy")  # shape = (n_kc, n_pn)
   reservoir = DrosophilaReservoir(pn_kc_matrix=matrix)
   weights = reservoir.pn_to_kc.weight.detach().numpy()
   row_sums = weights.sum(axis=1)
   nonzero = row_sums > 0
   print("Weights sum to 1 per KC row:",
         np.allclose(row_sums[nonzero], 1.0))
   print("Sparsity mask preserved:",
         np.array_equal(reservoir.pn_kc_mask.numpy(), (matrix > 0).astype(float)))
   PY
   ```

   The reservoir auto-resolves `n_pn`/`n_kc` dimensions from the matrix and
   keeps absent connections masked without resampling new sparsity patterns.

## Troubleshooting common setup errors

- **`Authentication secret not found at ~/.cloudvolume/secrets/cave-secret.json`** –
  The FlyWire CLI credentials are missing. Follow the token creation snippet in
  the quickstart to create the JSON file, or rerun `pgcn-cache` with the
  `--use-sample-data` flag to fabricate an offline cache for testing.

- **`FileNotFoundError: 'data/cache/nodes.parquet'` when running `pgcn-metrics` or
  instantiating `DrosophilaReservoir(cache_dir=...)`** – The connectome cache has
  not been generated yet. Execute `pgcn-cache --out data/cache/` (optionally with
  `--use-sample-data`) before invoking downstream commands.

- **`FileNotFoundError: 'custom_pn_to_kc.npy'`** – The matrix file was not saved
  prior to reservoir initialisation. Save the NumPy array with `np.save()` as
  shown above before loading it into `DrosophilaReservoir`.

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
