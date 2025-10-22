# 1) Mission

Create a reproducible research repo that (a) builds a FlyWire-backed connectome cache for AL→MB circuits, (b) computes glomerulus-wise structural priors and graph metrics, (c) aligns and analyzes neural/behavioral data, and (d) trains/evaluates biologically constrained models with an optional Streamlit explorer. All code must be unit-tested, version-pinned, and one-command runnable.

Authoritative refs to follow for APIs/concepts: CAVEclient materialization/versioning, neuPrint hemibrain access, connectome-based reservoir computing (conn2res), Drosophila MB innate/learned valence circuitry, and light-weight Drosophila SNN modeling.

# 2) Repo scaffold (create on init)
```````
PGCN/
├── src/pgcn/
│   ├── __init__.py
│   ├── connectome_pipeline.py      # FlyWire/CAVE pull; PN→KC→MBON (+DAN edges)
│   ├── metrics.py                  # Jaccard, path lengths, weighted centrality, DAN valence
│   ├── etl_recordings.py           # session schema & QC (Weeks 6–7)
│   ├── priors.py                   # structural prior vectors (Week 8)
│   ├── models/
│   │   ├── reservoir.py            # baseline reservoir; MB-inspired sparsity
│   │   ├── plasticity_meta.py      # meta-learned plasticity hooks (Week 10)
│   │   └── constraints.py          # homeostasis/Dale/targets (Week 11)
│   └── viz/
│       └── figures.py              # paper-quality plots
├── app/
│   └── Streamlit_app.py            # explorer v0.1–0.2
├── tests/
│   ├── test_cache.py
│   ├── test_metrics.py
│   └── test_priors.py
├── data/
│   ├── cache/                      # parquet snapshots + metrics
│   ├── sessions/                   # canonical session files
│   └── manifests/
├── notebooks/
│   ├── 01_prepare_data.ipynb
│   ├── 02_structural_metrics.ipynb
│   ├── 03_decoder_baseline.ipynb
│   ├── 04_decoder_with_priors.ipynb
│   └── 05_plasticity_hotspots.ipynb
├── environment.yml
├── pyproject.toml
├── Makefile
├── README.md
├── data_schema.md
├── METHODS.md
└── LICENSE
```````

### Conda env (name exactly PGCN):

##### environment.yml
name: PGCN
channels: [conda-forge, defaults]
dependencies:
  - python=3.11
  - pandas
  - pyarrow
  - numpy
  - networkx
  - tqdm
  - pydantic
  - rich
  - pytest
  - pip
  - pip:
      - caveclient            # CAVE/Materialization APIs
      - neuprint-python       # hemibrain validation
      - tenacity
      - streamlit
      - scikit-learn

### Makefile (minimum targets):

- conda:        # create env
- cache:        # build FlyWire cache
- metrics:      # compute Jaccard/paths/centralities/DAN labels
- priors:       # build structural_priors.parquet
- baseline:     # train reservoir baseline + CV
- app:          # run Streamlit explorer
- test:         # pytest -q
- format:       # ruff/black (add to pyproject)

### pyproject.toml: enforce ruff/black, mypy optional; set pgcn package and console scripts:

[project.scripts]

- pgcn-cache = "pgcn.connectome_pipeline:main"

- pgcn-metrics = "pgcn.metrics:cli"

# 3) Ground rules & quality bar

Versioning: Always pin a materialization_version (prefer 783; otherwise latest, recorded in meta). Use CAVEclient materialization with explicit version to ensure reproducibility. 


Tables: 
- Discover synapse/type tables programmatically; do not hard-code names. Respect that materialized tables are snapshot-only and may change across versions. 


Schema contracts:

- nodes.parquet: node_id, type{PN,KC,MBON,DAN}, glomerulus?, synapse_count?, x,y,z

- edges.parquet: source_id, target_id, synapse_weight

- dan_edges.parquet: same schema (DAN→KC/MBON)

- meta.json: datastack, materialization_version, synapse_table, cell_tables

- Unit tests: Non-empty graphs; PN/KC/MBON/DAN presence; PN→MBON direct edges excluded in core subgraph; weights >0; DAN labeler returns PAM/PPL1/other.

- CLI: Every pipeline step must be invocable via CLI and Makefile.

# 4) Month-by-Month build (what to implement)
#### Month 1 — Research bootstrap (Weeks 1–4)

- Capture METHODS.md notes tying model choices to literature:

- Reservoir computing on real connectomes (use as baseline framing).

- MB innate vs learned valence organization for architecture/labels.

- Lightweight Drosophila SNN scale & design cues (~2.2k neurons). 

#### Month 2 — Data & cache (Weeks 5–8)

##### Connectome cache (Week 5):

- Auth CAVE; pin snapshot; pull PN→KC→MBON and DAN→(KC|MBON) edges; store nodes/edges as Parquet + meta.

- Compute per-glomerulus metrics: KC set Jaccard, 2-hop PN→KC→MBON path summaries, synapse-weighted degree/betweenness, and DAN valence (PAM vs PPL1 via type strings).

- Version cache as flywire_v{version}_cache.*.

- Tests: node/edge counts, weight sanity, connectivity pattern sanity.

- Validation (Weekend): sanity-check counts against hemibrain via neuPrint (MBON/DAN/KC tallies/ratios).

- Recording ETL (Week 6): implement session schema (trial_id, fly_id, date, odor_id, concentration, condition, opto_on; t, features; labels), QC (motion/SNR), and manifest.yaml.

- Behavioral alignment (Week 7): align timestamps, derive labels, grouped splits (leave-one-fly-out).
- Structural priors (Week 8): build structural_priors.parquet with features like KC overlap to appetitive/aversive pools, path lengths to PAM/PPL1, LH vs MB strengths (document in 02_structural_metrics.ipynb).

#### Month 3 — Models (Weeks 9–12)

- Reservoir baseline: sparse KC-like layer (≈5% active), linear readout to MBON outputs; AUROC/AUPRC + calibration.

- Meta-learned plasticity (optional): evolvable local rule; dopamine/valence-conditioned; fitness by cross-fly performance.

- Homeostatic constraints: synaptic scaling / target activity; Dale’s-style sign handling for readout.

- Plasticity hotspot inference: ΔW maps (pre→post), group Lasso + stability selection.

#### Month 4 — App & packaging (Weeks 13–16)

- Streamlit app v0.1–0.2: circuit view, traces, overlays (Δ-activity, priors, SHAP).

- Reproducibility: make_figs.py, CI (lint, tests), README quick-start, data_schema.md, archiving instructions.

# 5) Minimal implementations the agent must produce
## 5.1 Connectome cache (core)

connectome_pipeline.py exposes a CLI:

- pgcn-cache --datastack flywire_fafb_production --mv 783 --out data/cache/

Behavior: initialize CAVEclient; choose materialization version (default 783, else latest); discover synapse and cell-type tables; select PN/KC/MBON/DAN populations; filter PN by optional glomerulus mapping; compute edges (PN→KC, KC→MBON, DAN→KC/MBON); estimate node coordinates from synapse loci; write Parquet + meta.
Why MV pinning: guarantees frozen annotations and stable IDs.

## 5.2 Metrics

metrics.py provides:

- jaccard_kc_overlap(e_pn_kc, pn_nodes) -> df[g1,g2,jaccard,...]

- path_lengths_pn_kc_mbon(edges) -> df[pn_id,mbon_id,path_strength,...]

- weighted_centralities(nodes, edges) -> df[node_id,deg_in_w,deg_out_w,betweenness_w]

- label_dan_valence(dan_nodes) -> df[node_id,dan_cluster∈{PAM,PPL1,DAN_other}]

CLI:

- pgcn-metrics --cache-stem flywire_v783_cache

## 5.3 Tests (must pass)

- test_cache.py: schema presence, non-emptiness, no PN→MBON direct edges in core subgraph, positive weights.

- test_metrics.py: deterministic metrics on toy graphs; DAN labeler finds PAM/PPL1 from type strings.

- pytest -q wired to make test.

# 6) CI, style, and docs

CI: GitHub Actions to run conda env update, pytest, ruff/black.

Style: ruff (E/F/I/B) + black; type hints everywhere; logging with rich.

Docs:

- README.md: 5-minute quick-start (create PGCN, run pgcn-cache, run metrics, launch app).

- data_schema.md: exact Parquet/JSON schemas and units; note that coordinates are nm; cite CAVE materialization snapshot semantics.

- METHODS.md: short rationale per component with literature hooks (conn2res, Eschbach MB circuits, SNN scale).

# 7) Non-negotiables

- Reproducibility > speed. Prefer explicit versions and cached Parquet to ad-hoc queries.

- No silent schema drift: fail fast if a required column is missing.

- All metrics and models must be callable from CLI and importable in notebooks.

# Appendix: API quick links (for implementers)

- CAVEclient (materialization/versioning): docs & API.

- neuPrint (hemibrain) docs & API: explorer, Python, and manual. 

- connectome-neuprint.github.io

- Connectome-based reservoir computing (conn2res): paper & code.

- MB innate/learned valence circuits: eLife study integrating MB and LH.

- Lightweight Drosophila olfactory SNN: open-access article.