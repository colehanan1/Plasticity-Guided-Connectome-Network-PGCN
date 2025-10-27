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

2. **Provision your FlyWire token**

   First (re)install the project in editable mode so the latest console scripts
   are registered on your `$PATH`:

   ```bash
   python -m pip install -e .[dev]
   ```

   The `[dev]` extra is optional but convenient if you plan to run tests. When
   you prefer not to install development dependencies, drop the extras suffix.

   You can now write the token to the expected secret path either via the
   installed entry point **or** via the repository-local runner that works even
   before installation:

   ```bash
   pgcn-auth --token "<paste-your-flywire-token-here>"
   # or
   ./scripts/pgcn-auth --token "<paste-your-flywire-token-here>"
   ```

   Replace the placeholder with the token string copied from the
   [FlyWire account portal](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_setup.html).
   The command will create both `~/.cloudvolume/secrets/cave-secret.json` and
   the FlyWire-preferred
   `~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json` if they do not
   exist. When you keep the token in a file, point the helper at it instead:

   ```bash
   pgcn-auth --token-file /path/to/my_token.txt
   # or
   ./scripts/pgcn-auth --token-file /path/to/my_token.txt
   ```

   Use `--force` whenever you intentionally rotate credentials and need to
   overwrite the existing JSON.

3. **Preflight your FlyWire permissions**

   Run the diagnostic CLI before attempting an expensive cache build. The tool
   checks every supported secret location, validates the token against the
   InfoService endpoint, and enumerates available materialization versions so
   you know the datastack is readable:

   ```bash
   pgcn-access-check --datastack flywire_fafb_production
   # or
   ./scripts/pgcn-access-check --datastack flywire_fafb_production
   ```

   A successful probe prints something akin to:

   ```text
   Datastack: flywire_fafb_production
   Token source: file:/home/<user>/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json
   InfoService: OK (dataset=fafb)
   Materialization: OK (27 versions discovered)
   ```

   Anything returning `HTTP 401` means the secret is malformed or expired—mint
   a fresh token at
   `https://global.daf-apis.com/auth/api/v1/user/create_token` and rerun
   `pgcn-auth`. An `HTTP 403` indicates the token is valid but the associated
   FlyWire account lacks FAFB *view* permission; request access via the official
   [FlyWire setup guide](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_setup.html)
   before proceeding. While you wait for approval you can still exercise the
   downstream tooling in offline mode by passing `--use-sample-data` to the
   cache command.

4. **Build the connectome cache**

   ```bash
   pgcn-cache --datastack flywire_fafb_production --mv 783 --out data/cache/
   ```

   > **HTTP 403?** The FlyWire API rejected the token because it lacks `view`
   > permission for the FAFB dataset. Revisit the
   > [FlyWire setup guide](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_setup.html),
   > confirm the correct Google/ORCID identity has access, rerun the
   > `pgcn-access-check` diagnostic to verify, and then repeat the cache build.
   > Until approval lands you can keep moving by supplying `--use-sample-data`.

   Use `--use-sample-data` for an offline deterministic cache when a FlyWire
   account or network access is unavailable:

   ```bash
   pgcn-cache --use-sample-data --out data/cache/
   ```

   ### Working with Codex snapshot 783 exports (public data path)

   While your FlyWire permissions are pending you can bootstrap the cache with
   the public Codex release of snapshot 783:

   1. **Download the Codex neuron and synapse tables**

      Visit [https://codex.flywire-daf.com](https://codex.flywire-daf.com),
      select the FAFB snapshot 783 dataset, and export the following datasets:

      - **Cell Types** – provides neuron metadata and type annotations (used for
        the `--neurons` input).
      - **Connections (Filtered)** *or* **Synapse Table** – provides PN→KC/KC→MBON
        connectivity with weight/count information (used for the `--synapses`
        input). The filtered connections table is lighter (~68 MB) and suffices
        for reservoir hydration; the full synapse table (~2.7 GB) is optional if
        you need per-synapse records.

      Codex exports are delivered as CSV/TSV (often compressed as `.gz`). The
      importer recognises the official column names (`root_id`,
      `primary_type`, `additional_type(s)`, `class`,
      `pre_root_id_720575940`, `post_root_id_720575940`, `size`, …) out of the
      box, so you can feed the downloads directly without renaming headers.
      When using **Classification / Hierarchical Annotations** for the neuron
      metadata, rely on the `class` column: entries labelled `ALPN` are mapped
      to projection neurons and `Kenyon_Cell` is treated as a Kenyon cell. Save
      the downloads into a working directory, for example
      `~/Downloads/fafb_codex_783/`.

   2. **Convert the exports into the PGCN cache layout**

      Use the new Codex importer (remember to reinstall with `pip install -e .`
      after pulling changes) to materialise a local cache:

      ```bash
      pgcn-codex-import \
        --neurons ~/Downloads/fafb_codex_783/neurons.csv.gz \
        --synapses ~/Downloads/fafb_codex_783/synapses.csv.gz \
        --out data/cache/
      ```

      The importer heuristically recognises PN/KC/MBON/DAN types by combining
      Codex `primary_type`, `additional_type(s)`, and `class` annotations before
      applying the regex classifier, so “Projection neuron”, `ALPN`, or
      “Kenyon_Cell” labels in any column count. Root IDs truncated in the
      header (for example
      `pre_root_id_720575940`) are expanded back to the canonical 64-bit values
      even when the synapse table references neurons missing from the Cell
      Types export, ensuring PN→KC edges survive the merge. When Codex provides
      glomerulus annotations they are passed through into the `nodes.parquet`
      output; otherwise the importer emits an empty `glomerulus` column so the
      downstream metrics skip overlap analyses gracefully instead of crashing.
      If your export uses
      project-specific labels, extend the classifier with regular-expression
      overrides:

      ```bash
      pgcn-codex-import \
        --neurons neurons.csv \
        --synapses synapses.csv \
        --pn-pattern "^PN-" \
        --kc-pattern "Kenyon" \
        --out data/cache/
      ```

      The script writes `nodes.parquet`, `edges.parquet`, `dan_edges.parquet`,
      and `meta.json` so downstream tooling (metrics, reservoir hydration)
      works identically to the authenticated pipeline. After conversion run the
      usual sanity checks to confirm PN→KC edges were detected and normalised:

      ```bash
      pgcn-metrics --cache-dir data/cache/
      python - <<'PY'
      from pathlib import Path
      from pgcn.models.reservoir import DrosophilaReservoir
      import json

      reservoir = DrosophilaReservoir(cache_dir=Path("data/cache"))
      print("PN→KC edges:", reservoir.pn_kc_mask.sum().item())
      print("PN→KC weight rows:", reservoir.pn_to_kc.weight.shape[0])
      meta = json.loads((Path("data/cache/meta.json")).read_text())
      print("PN→KC edges in meta:", meta["counts"]["pn_kc_edges"])
      PY
      ```

   3. **Validate the generated cache**

      Reuse the existing diagnostics to ensure the cache can drive the
      reservoir:

      ```bash
      pgcn-metrics --cache-dir data/cache/
      python - <<'PY'
      from pathlib import Path
      from pgcn.models.reservoir import DrosophilaReservoir

      reservoir = DrosophilaReservoir(cache_dir=Path("data/cache"))
      print("PN→KC weight matrix:", reservoir.pn_to_kc.weight.shape)
      print("Mask density:", reservoir.pn_kc_mask.float().mean().item())
      PY
      ```

      Any classification issues (for example if PN→KC edges are missing) can be
      fixed by re-running `pgcn-codex-import` with additional `--*-pattern`
      overrides until PN, KC, MBON, and DAN populations are detected.

5. **Compute structural metrics**

   ```bash
   pgcn-metrics --cache-dir data/cache/
   ```

   The PN glomerulus overlap report now emits a warning (and an empty table)
   if your Codex exports lack glomerulus annotations. Rebuild the cache after
   downloading richer neuron metadata once it becomes available to populate
   those fields.

6. **Run the unit tests**

   ```bash
   pytest -q
   ```

7. **Validate reservoir weight hydration**

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

## Behavioral cross-validation CLI

Week 4 and later checkpoints track behavioural generalisation via grouped
cross-validation. The repository provides a reference driver under
`analysis/cross_validation.py` that reproduces those metrics with fly-aware
splits and ChemicalSTDP fine-tuning confined to the KC→MBON projection.

1. **Run grouped cross-validation**

   ```bash
   python analysis/cross_validation.py \
     --folds 5 \
     --output-dir artifacts/cross_validation \
     --report-prefix week4
   ```

   The command consumes the canonical behavioural CSV (or a custom dataset via
   `--data`) and instantiates a fresh
   `ChemicallyInformedDrosophilaModel` for each fold. During training only the
   KC→MBON weights are updated through the `ChemicalSTDP` plasticity rule while
   all other parameters remain frozen. GroupKFold splitting respects the
   `fly` identifier to prevent leakage across individuals.

   The fold training loop now feeds the model's instantaneous probability into
   `ChemicalSTDP.update_plasticity`, which converts the binary reward into a
   reward-prediction error (`reward - prediction`). Chemical similarity still
   modulates the effective learning rate and expected generalisation, ensuring
   odor pairs with weak literature support learn cautiously. Eligibility traces
   are cached per `(training_odor, test_odor)` pair so repeated presentations
   compound rather than restart plasticity. After each update the KC→MBON
   matrix is clipped to remain non-negative, keeping the synapses biologically
   plausible for downstream metrics that assume excitatory KC drive.

   To verify the update dynamics after a short dry-run you can inspect the
   minimum KC→MBON weight:

   ```bash
   python - <<'PY'
   import torch

   from pgcn.models import ChemicallyInformedDrosophilaModel, ChemicalSTDP
   from pgcn.data.behavioral_data import load_behavioral_dataframe
   from analysis.cross_validation import compute_model_response, resolve_training_odor, resolve_test_odor

   df = load_behavioral_dataframe()
   model = ChemicallyInformedDrosophilaModel()
   stdp = ChemicalSTDP(model.reservoir.n_kc, model.reservoir.n_mbon)
   device = torch.device('cpu')
   model.to(device)
   stdp.to(device)
   model.eval()
   stdp.eval()

   sample = df.iloc[0]
   training = resolve_training_odor(sample['dataset'])
   testing = resolve_test_odor(sample['dataset'], sample['trial_label'])
   reward = float(sample['prediction'])
   prob, kc_activity = compute_model_response(model, training, testing, device=device)
   delta = stdp.update_plasticity(training, testing, reward=reward, predicted_response=prob, kc_activity=kc_activity)
   with torch.no_grad():
       model.reservoir.kc_to_mbon.weight.add_(delta.to(model.reservoir.kc_to_mbon.weight.device))
       model.reservoir.kc_to_mbon.weight.clamp_(min=0.0)
   print('Minimum KC→MBON weight:', float(model.reservoir.kc_to_mbon.weight.min()))
   PY
   ```

   A non-negative minimum confirms the clipping guard is active and prevents
   inhibitory spill-over from the plastic pathway.

   Set the `PGCN_BEHAVIORAL_DATA` environment variable when the canonical
   `data/model_predictions.csv` is unavailable locally:

   ```bash
   export PGCN_BEHAVIORAL_DATA=/home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv
   ```

   The loader will resolve the absolute path, expand `~` if necessary, and
   continue to enforce the 440-trial validation. Confirm your dataset preserves
   one dataset per fly and consistent `trial_label` coverage before running
   large experiments. The odor mapping remains identical to the bundled
   reference, including `hex_control`, which reuses the `opto_hex` test-odor
   assignments; ensure custom exports honour that mapping so the CLI can
   resolve the correct chemical identities.

2. **Inspect per-fold outputs**

   Every fold emits a JSON summary and a companion CSV of generalisation curves
   in the specified output directory. Metrics include overall accuracy,
   trained-odor accuracy, control separation (how sharply the model suppresses
   `hex_control` responses relative to conditioned flies), and AUROC.

3. **Review aggregate reports**

   The script assembles `week4_report.json` and `week4_report.csv` (configurable
   via `--report-prefix`). The JSON report captures per-fold metrics alongside
   fold-wise means and standard deviations; the CSV lists each fold plus
   appended mean/std rows for quick spreadsheet import. If you rerun the CLI and
   supply fold metrics that already include a `fold` column, the exporter now
   preserves that numbering instead of attempting to reinsert it, preventing
   duplicate-column errors during long-running pipelines.

4. **Interpret the aggregate digest**

   After saving the JSON and CSV artefacts the CLI now prints a compact console
   summary that compares the observed means against the Week 4 targets
   (overall ≥0.70, trained-odor ≥0.80, control separation ≥0.90) and a
   50 % chance baseline for accuracy. Metrics that cannot be computed (for
   example when a fold contains only control trials) are explicitly marked so
   you know whether gaps stem from modelling issues or data coverage.

   Example output from the extended behavioural export mentioned above:

   ```text
   === Cross-validation aggregate summary ===
   Folds evaluated: 5
   overall_accuracy: mean=0.495 ±0.072 (5/5 folds)
     ↳ vs. chance (0.500): -0.005
     ↳ below target ≥0.700
   trained_odor_accuracy: mean=0.604 ±0.123 (5/5 folds)
     ↳ below target ≥0.800
   control_separation: mean=0.477 ±0.052 (5/5 folds)
     ↳ below target ≥0.900
   auroc: mean=0.477 ±0.052 (5/5 folds)
   ```

   These results show the current configuration underperforming relative to the
   Week 4 expectations: accuracy sits only marginally above chance, trained-odor
   recognition is inconsistent across folds, and the model fails to suppress
   `hex_control` responses. Treat this digest as the first checkpoint before
   diving into the per-fold JSON files or the generalisation curves when
   debugging.

5. **Tune learning dynamics (optional)**

   Adjust `--learning-rate` or `--decision-threshold` to explore alternative
   plasticity strengths and classification cut-offs. Use `--device` to force CPU
   or GPU execution when the default auto-detection does not match your setup.

## Troubleshooting common setup errors

- **`Authentication secret not found at ~/.cloudvolume/secrets/cave-secret.json`** –
  The FlyWire CLI credentials are missing. Run
  `pgcn-auth --token "<paste-your-flywire-token-here>"` to provision the JSON
  file automatically, or rerun `pgcn-cache` with the `--use-sample-data` flag to
  fabricate an offline cache for testing.

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

## Behavioral Data Utilities

The :mod:`pgcn.data.behavioral_data` module centralises access to
`data/model_predictions.csv`, enforces the canonical row and fly counts, and
exports immutable dataclasses for each behavioural trial. The loaders sort the
raw CSV on `fly` and `trial_label`, reset the index, and propagate that
deterministic ordering into helpers such as ``load_behavioral_tensor``,
``load_behavioral_trial_matrix``, and the modelling-focused tuples described
below. This guarantees that tensors and folds align with the documented
``(fly, trial_label)`` pairs regardless of the storage order on disk.

Helper functions cover both pandas and tensor workflows:

- ``load_behavioral_trials()`` returns a :class:`BehavioralTrialSet` with sorted
  ``(dataset, fly, trial_label)`` ordering and per-trial metadata snapshots.
- ``load_behavioral_model_frames()`` emits ``(features, labels, groups)``
  DataFrames/Series ready for scikit-learn pipelines.
- ``load_behavioral_model_tensors()`` mirrors the pandas helper but yields
  PyTorch tensors with integer-encoded group assignments.
- ``make_group_kfold()`` wraps :class:`sklearn.model_selection.GroupKFold` to
  produce deterministic, per-fly cross-validation splits.

**Verify the ordering and grouping guarantees**

```bash
python -m pytest tests/data/test_behavioral_data.py -v
```

The suite asserts the reset index, the stable ordering in tensor exports, the
behaviour of the modelling helpers, and the `fly × trial` pivot used for
exploratory analysis.

## Authentication Notes

Unit tests fabricate a deterministic cache to verify schema integrity,
positive weights, and the absence of direct PN→MBON edges. The `--use-sample-data`
flag mirrors this behaviour for developers working offline.
