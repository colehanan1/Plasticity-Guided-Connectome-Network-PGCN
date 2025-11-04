# Phase 1 Implementation Guide — Connectivity Backbone

This guide expands Phase 1 of the roadmap into precise engineering steps. Complete each subsection in order; later tasks assume earlier artefacts exist and pass tests.

## 1. Connectivity Matrix Module (`models/connectivity_matrix.py`)
### 1.1 File structure
```text
models/
  __init__.py
  connectivity_matrix.py  # new
```

### 1.2 Class design
```python
@dataclass
class ConnectivityMatrix:
    pn_ids: np.ndarray
    kc_ids: np.ndarray
    mbon_ids: np.ndarray
    dan_ids: np.ndarray
    pn_to_kc: sp.csr_matrix
    kc_to_mbon: sp.csr_matrix
    dan_to_targets: Dict[str, sp.csr_matrix]

    def slice_kc_subtypes(self, subtype: str) -> "ConnectivityMatrix": ...
    def pn_fan_in(self, kc_index: int) -> sp.csr_matrix: ...
    def mbon_fan_in(self, mbon_index: int) -> sp.csr_matrix: ...
```
- Use `scipy.sparse.csr_matrix` for PN→KC and KC→MBON weights; normalise synapse counts to floats (synapses or probabilities configurable via keyword).
- `dan_to_targets` should include at least `"kc"` and `"mbon"` matrices, keyed by target class.
- Provide constructor helpers (`from_cache(cache_dir: Path, *, min_synapses: int = 1)`) that load CSVs and build matrices in one call.

### 1.3 Loading logic
1. Parse CSVs from `data/cache/` using pandas with explicit dtypes (`Int64`, `category`) to minimise RAM.
2. Map neuron root IDs to contiguous indices; persist mapping dictionaries on the class for downstream reference.
3. Convert synapse tables (PN→KC, KC→MBON, DAN→KC/MBON) into COO format then CSR for efficient arithmetic.
4. Validate dimensions: PN count should match ALPN table, KC count should match aggregate KC CSVs, MBON and DAN counts should match circuit exports.
5. Add invariants (`assert matrix.shape == (len(pn_ids), len(kc_ids))`, etc.) and informative error messages when caches are missing.

### 1.4 Unit tests
- Location: `tests/models/test_connectivity_matrix.py`.
- Fixtures: create small synthetic CSVs (5–10 rows) mimicking cache schema.
- Tests: constructor loads data, fan-in methods return expected values, subtype slicing reduces matrices appropriately.

## 2. Circuit Container (`models/olfactory_circuit.py`)
### 2.1 Class design
```python
class OlfactoryCircuit:
    def __init__(self, connectivity: ConnectivityMatrix, *, kc_sparsity: float = 0.05): ...
    def activate_pns(self, glomeruli: Sequence[str]) -> np.ndarray: ...
    def kc_response(self, pn_activity: np.ndarray) -> np.ndarray: ...
    def mbon_response(self, kc_activity: np.ndarray) -> np.ndarray: ...
    def dan_response(self, pn_activity: np.ndarray, reward: float) -> np.ndarray: ...
```
- Keep methods deterministic; return numpy arrays.
- Accept glomerulus metadata from ALPN table to build PN activation masks.
- Inject KC sparsity mask (5%) inside `kc_response` if not supplied externally.

### 2.2 Implementation steps
1. Import ALPN/KC metadata via a loader (`data_loaders/circuit_loader.py` placeholder) to map glomeruli to PN indices.
2. Implement PN activation as binary mask over PN indices, scaled by optional firing rates.
3. Multiply PN→KC CSR matrix by PN activity vector; enforce sparsity by zeroing all but top `sparsity` fraction per odor.
4. Project KC activity to MBON outputs via KC→MBON CSR matrix; normalise outputs for interpretability.
5. Derive DAN output neuropil groups to produce modulatory signals (e.g., average activity for MB-targeting DANs).
6. Provide `simulate_trial` helper returning `(pn_activity, kc_activity, mbon_activity, dan_activity)` for integration tests.

### 2.3 Tests
- Create `tests/models/test_olfactory_circuit.py` using miniature matrices to verify PN activation, KC sparsity enforcement, and MBON output ranges.

## 3. Circuit Loader (`data_loaders/circuit_loader.py`)
### 3.1 Responsibilities
- Load cached CSVs (`data/cache/*.csv`).
- Merge subtype tables into a single KC dataframe with subtype labels.
- Provide typed DTOs (e.g., `NamedTuple` or `dataclass`) for ALPN, KC, MBON, DAN populations.
- Supply convenience methods (`load_connectivity_inputs`, `load_glomerulus_index`) consumed by `ConnectivityMatrix.from_cache` and `OlfactoryCircuit`.

### 3.2 Implementation outline
```python
@dataclass
class CircuitData:
    alpn: pd.DataFrame
    kenyon: pd.DataFrame
    mbon: pd.DataFrame
    dan: pd.DataFrame
    pn_to_kc: pd.DataFrame
    kc_to_mbon: pd.DataFrame
    dan_to_kc: pd.DataFrame
    dan_to_mbon: pd.DataFrame
```
- Include validation to ensure CSV presence; raise descriptive `FileNotFoundError` with remediation instructions.
- Normalise neuropil columns to uppercase strings and split pipe-delimited fields into lists for richer querying.

### 3.3 Tests
- Use synthetic CSVs to confirm loader merges subtypes, enforces required columns, and handles missing files gracefully.

## 4. Integration Smoke Test
1. Write `tests/integration/test_circuit_pipeline.py` that:
   - Loads synthetic cache fixtures.
   - Builds `ConnectivityMatrix` via `from_cache`.
   - Instantiates `OlfactoryCircuit` and runs `simulate_trial` with a sample odor.
   - Asserts shapes and non-zero outputs at each stage.
2. Wire this test into CI (update `pyproject.toml` or `pytest.ini` if needed).

## 5. Tooling & Developer Experience
- Add `make connectivity` target to `Makefile` executing smoke tests relevant to the new modules.
- Document usage in `README.md` and `docs/model_integration_status.md` once modules exist.
- Capture profiling notes if matrix construction exceeds 1 GB RAM; consider chunked loading.

## 6. Deliverables Checklist
- [ ] `models/connectivity_matrix.py` implemented with sparse matrices and helper methods.
- [ ] `data_loaders/circuit_loader.py` implemented with robust validation.
- [ ] `models/olfactory_circuit.py` providing PN→KC→MBON propagation and DAN hooks.
- [ ] New unit tests and integration test passing locally (`pytest tests/models tests/integration`).
- [ ] README + docs updated with run instructions and module descriptions.
