# Project Status

## 1. Executive Summary
- **Project goal:** Build a biologically grounded Plasticity-Guided Connectome Network that predicts Drosophila olfactory learning and cross-odor generalization from FlyWire FAFB v783 data.
- **Current completion:** ~35% — data extraction and neuropil-aware preprocessing are stable, but circuit modeling, learning dynamics, and experiment replication layers remain outstanding.
- **Key milestone achieved:** Offline extraction of the olfactory learning circuit (ALPN → KC → MBON with DAN modulation) including neuropil annotations derived from the 5.3 M-edge connectivity table.

## 2. Completed Components
### Data Extraction Pipeline
- ✅ **ALPN extraction** — 353 projection neurons with glomerulus assignments via `scripts/extract_alpn_projection_neurons.py`; outputs `data/cache/alpn_extracted.csv` and `pn_to_kc_connectivity.csv` when source data is present.
- ✅ **Kenyon cell extraction** — 5,177 neurons spanning eight subtypes using neuropil-aware filtering in `scripts/extract_circuit.py`; emits dedicated CSVs (`kc_ab.csv`, `kc_ab_p.csv`, `kc_g_main.csv`, `kc_g_dorsal.csv`, `kc_g_sparse.csv`, `kc_apbp_main.csv`, `kc_apbp_ap1.csv`, `kc_apbp_ap2.csv`).
- ✅ **MBON extraction** — 96 neurons with calyx and medial-lobe input annotations derived from the connections table; exports `mbon_all.csv`, `mbon_calyx.csv`, `mbon_ml.csv`, `mbon_glut.csv`.
- ✅ **DAN extraction** — 584 dopaminergic neurons with mushroom-body targeting metadata via neuropil aggregation; exports `dan_all.csv`, `dan_mb.csv`, `dan_calyx.csv`, `dan_ml.csv`.
- ✅ **Connectivity availability** — `connections_princeton.csv.gz` integrated for presynaptic/postsynaptic neuropil discovery; PN→KC synapses already cached by the ALPN pipeline.

### Support Infrastructure
- ✅ Scripts documented in `README.md` with invocation commands and expected sanity checks.
- ✅ FlyWire CSV loaders normalise schema discrepancies and provide consistent metadata joins for downstream use.

## 3. In-Place Components
- `scripts/extract_alpn_projection_neurons.py` and `scripts/extract_circuit.py` operate end-to-end with informative logging, fallback behaviours, and gzip-aware IO.
- FlyWire exports (`classification.csv.gz`, `consolidated_cell_types.csv.gz`, `neurons.csv.gz`, `connections_princeton.csv.gz`) are recognised and merged through shared helper utilities.
- Cache directory structure (`data/cache/`) is standardised for downstream model loading.

## 4. Outstanding Work (⏳)
| Component | Objective | Status | Dependencies |
| --- | --- | --- | --- |
| `models/connectivity_matrix.py` | Construct sparse PN×KC, KC×MBON, and DAN targeting matrices. | ⏳ | `data/cache/*.csv`, connections table |
| `models/olfactory_circuit.py` | Define circuit container with activation/propagation methods. | ⏳ | `connectivity_matrix.py`, extracted neuron tables |
| `models/learning_model.py` | Implement dopamine-modulated plasticity and learning routines. | ⏳ | `olfactory_circuit.py` |
| `data_loaders/circuit_loader.py` | Transform cached CSVs into typed objects for models. | ⏳ | Extracted CSVs |
| `experiments/optogenetic_conditioning.py` | Reproduce GR5a conditioning protocols. | ⏳ | `learning_model.py` |
| `experiments/validation.py` | Benchmark model predictions vs. behavioural datasets. | ⏳ | Experiment outputs, learning model |
| Documentation refresh | End-to-end README + API docs + tutorials. | ⏳ | Core modules |
| Automated tests | Unit tests for loaders, matrices, learning rules. | ⏳ | Core modules |

## 5. Recommended Next Steps (🎯)
### Phase 1 — Core Circuit Model (Target: 1 week)
1. Implement `models/connectivity_matrix.py` to load cached neuron tables and build SciPy sparse matrices with helper queries for subtype subsets.
2. Develop `models/olfactory_circuit.py` exposing PN activation, KC sparse expansion, MBON projection, and DAN modulation hooks.
3. Ship an integration script or notebook that instantiates the circuit and performs an odor-to-behaviour trace to validate the data flow.

### Phase 2 — Learning Dynamics (Target: 1 week)
1. Build `models/learning_model.py` with Hebbian + dopamine-gated plasticity, eligibility traces, and configurable learning rates.
2. Create regression tests comparing learning curves against published behavioural data; tune plasticity constants accordingly.

### Phase 3 — Experimental Replication (Target: 1 week)
1. Implement optogenetic conditioning simulations mirroring GR5a activation schedules, including parameter sweeps for light timing and reward pairing.
2. Add validation scripts juxtaposing simulated behaviour against experimental benchmarks with statistical summaries.

### Phase 4 — Documentation & Tooling (Target: 1 week)
1. Produce comprehensive README/tutorial coverage, notebooks for exploratory analysis, and diagrams illustrating circuit information flow.
2. Harden CI with unit/integration tests and lightweight benchmarks on sampled connectivity data.

## 6. Integration Points (🔗)
- **Data loaders → Models:** CSV outputs from extraction scripts will feed `data_loaders/circuit_loader.py`, which in turn should initialise `ConnectivityMatrix` objects consumed by circuit and learning modules.
- **Circuit ↔ Learning:** The learning model will reference the circuit for forward passes while mutating KC→MBON weights based on DAN signals.
- **Experiments ↔ Documentation:** Optogenetic simulations and validation scripts must emit artefacts summarised in `docs/` and referenced in the main README for reproducibility.
- **External datasets:** Behavioural datasets (e.g., `model_predictions.csv`) must align with odor mappings and chemical descriptors already defined in the repo.

## 7. Data Pipeline Summary (📊)
```
FlyWire FAFB v783 exports
    │
    ├─ scripts/extract_alpn_projection_neurons.py
    │    ├─ ALPN metadata (glomeruli, NT type)
    │    └─ PN→KC synapse cache (calyx restricted)
    │
    └─ scripts/extract_circuit.py
         ├─ KC subtypes (8 CSVs, neuropil aware)
         ├─ MBON populations with input neuropils
         └─ DAN populations with output neuropils
            │
            ▼
        data/cache/ (CSV artefacts with root IDs)
            │
            ▼
        models/connectivity_matrix.py (next)
            │
            ▼
        models/olfactory_circuit.py
            │
            ▼
        models/learning_model.py
            │
            ▼
        experiments/* (conditioning + validation)
```

## 8. Immediate Action Checklist & Metrics
### Immediate Checklist
- [ ] Stand up `models/connectivity_matrix.py` with sparse matrix loaders.
- [ ] Draft `data_loaders/circuit_loader.py` to hydrate neuron classes from cache.
- [ ] Define interface for `OlfactoryCircuit` to unblock learning module design.
- [ ] Capture sample outputs (counts, neuropil summaries) in regression fixtures for testing.

### Success Metrics
- ✅ Extraction scripts load all required CSVs without error and emit stable counts.
- ✅ Connectivity matrices remain sparse (<5% density) and match published partner statistics.
- ✅ Circuit simulation reproduces odor-driven MBON responses without plasticity.
- ✅ Learning model matches experimental conditioning curves (target correlation r > 0.7).
- ✅ Documentation enumerates run commands and data dependencies for every stage.
