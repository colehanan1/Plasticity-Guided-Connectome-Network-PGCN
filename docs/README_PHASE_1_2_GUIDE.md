# Plasticity-Guided Connectome Network (PGCN) — Phase 1 & 2 Implementation

## 🎯 Project Overview

The Plasticity-Guided Connectome Network builds a **biologically grounded neural network** that instantiates the Drosophila olfactory learning circuit (PN→KC→MBON pathway) directly from FlyWire connectome data. The model implements dopamine-modulated learning dynamics to test mechanistic hypotheses about how the brain implements associative learning.

**Key biological features**:
- **Sparse expansion coding**: ~5% KC activation per odor (pattern separation)
- **Structured connectivity**: 376 PNs → 5,177 KCs → 96 MBONs, with 584 DANs providing reward signals
- **Three-factor Hebbian rule**: dW ∝ (presynaptic KC) × (postsynaptic MBON) × (dopamine/RPE)
- **KC sparsity via APL inhibition**: k-winners-take-all modeling of lateral inhibition
- **Causal experiments**: Veto gates, ablations, eligibility traces, and Shapley analysis to prove blocking mechanisms

---

## ✅ Phase 1 Status: Connectivity Backbone Complete

**What's implemented**:
- ✅ `ConnectivityMatrix`: Immutable sparse connectivity loaded from FlyWire v783
- ✅ `CircuitLoader`: Robust cache loader with validation and normalization strategies
- ✅ `OlfactoryCircuit`: Feedforward propagation (PN→KC→MBON) with k-WTA sparsity
- ✅ 59 unit & integration tests (100% passing)
- ✅ Full biological documentation for every class and method

**Verified dimensions**:
```
PNs (connected):    376 / 482 ALPNs
KCs (8 subtypes):   5,177 neurons
MBONs:              96 neurons
DANs:               584 dopaminergic neurons
PN→KC edges:        22,806 synapses
KC→MBON edges:      2,153 synapses
```

### Run Phase 1 Tests (Quick Start)

#### Option 1: Run all Phase 1 tests in one command

```bash
# Activate environment
conda activate PGCN

# Run Phase 1 tests with coverage report
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests/models tests/integration -v --tb=short
```

**Expected output**:
```
tests/models/test_connectivity_matrix.py::test_immutability PASSED
tests/models/test_connectivity_matrix.py::test_load_shapes PASSED
...
tests/integration/test_phase1_integration.py::test_end_to_end_forward_pass PASSED

====== 59 passed in 29.38s ======
```

#### Option 2: Run specific test modules

```bash
# Test connectivity matrix loading
PYTHONPATH=src pytest tests/models/test_connectivity_matrix.py -v

# Test circuit loader (CSV parsing, normalization)
PYTHONPATH=src pytest tests/models/test_circuit_loader.py -v

# Test olfactory circuit (forward propagation)
PYTHONPATH=src pytest tests/models/test_olfactory_circuit.py -v

# Test end-to-end integration
PYTHONPATH=src pytest tests/integration/test_phase1_integration.py -v
```

#### Option 3: Run with detailed output and coverage

```bash
# Full coverage report
PYTHONPATH=src pytest tests/models tests/integration \
  --cov=src/pgcn/models \
  --cov=src/data_loaders \
  --cov-report=term-missing \
  -v
```

### Verify Phase 1 Works: Example Usage

```python
# Example: Load circuit and run forward pass
from pathlib import Path
from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.olfactory_circuit import OlfactoryCircuit
import numpy as np

# 1. Load connectivity from FlyWire cache
loader = CircuitLoader(cache_dir=Path("data/cache"))
connectivity = loader.load_connectivity_matrix(normalize_weights="row")

print(f"✓ Loaded {len(connectivity.pn_ids)} PNs")
print(f"✓ Loaded {len(connectivity.kc_ids)} KCs")
print(f"✓ Loaded {len(connectivity.mbon_ids)} MBONs")

# 2. Create circuit model
circuit = OlfactoryCircuit(
    connectivity=connectivity,
    kc_sparsity_target=0.05  # 5% of KCs active
)

# 3. Simulate a trial: activate a glomerulus, observe MBON response
pn_activity = circuit.activate_pns_by_glomeruli(["DA1"], firing_rate=1.0)
mbon_output, diag = circuit.forward_pass(pn_activity, return_intermediates=True)

print(f"✓ PN activity: {pn_activity.sum():.1f} neurons active")
print(f"✓ KC sparsity: {diag['sparsity_fraction']:.1%}")
print(f"✓ MBON valence output: {mbon_output[0]:.3f}")
```

---

## 🚀 Phase 2 Status: Learning Dynamics (In Progress)

**What's coming in Phase 2**:

### Core Learning Components
- `DopamineModulatedPlasticity`: Three-factor Hebbian rule with RPE gating
  - dW = α × (KC × MBON × dopamine) with configurable time constants
  - Eligibility traces for synaptic tagging-and-capture
  - Weight decay and gating modes (hard threshold, soft modulation)
  
- `LearningExperiment`: Trial-by-trial simulation orchestrator
  - Run odor conditioning protocols (CS+/CS-, reward pairing)
  - Compute RPE = reward - predicted_value
  - Record learning curves and memory retention

### Causal Experiments
- **Experiment 1 - Veto Gate**: Single-pathway blockade of learning
  - Gate: v = V^T × PN; Learning: dW = α × KC × MBON × DAN × (1 - v)
  - Predicts: Can one glomerulus block Odor A learning while preserving Odor B?

- **Experiment 2 - Counterfactual Microsurgery**: Three editing variants
  - (i) Ablate PN→KC inputs from veto glomerulus
  - (ii) Freeze KC→MBON synapses in veto pathway
  - (iii) Sign-flip dopamine coupling for veto neurons
  - Predicts: Editing any variant should reverse blocking

- **Experiment 3 - Eligibility Traces**: Soft vs. hard memory protection
  - Compare: eligibility traces vs. hard weight freezing vs. no protection
  - During OdorB learning, which method best preserves OdorA memory?

- **Experiment 6 - Shapley Analysis**: Causal neuron identification
  - Compute Shapley values for each KC: contribution to blocking?
  - Edit top-k blockers (prune, sign-flip, reweight)
  - Predicts: Minimal edits to top blockers should reverse learning deficit

### Test Suite
- 50+ new unit tests across plasticity, experiments
- Integration tests validating learning curves and memory retention
- Full biological documentation with rationale for every design choice

### Quick Start: Phase 2 Prompt for Claude

To generate Phase 2 code, copy the prompt from [28] and paste into Claude:

```bash
# View the Phase 2 prompt
cat phase2-implementation-prompt.md
```

Then follow Claude's "explore, plan, code, commit" workflow:
1. **Explore**: Claude reads Phase 1 code and NEXT_STEPS_DETAILED.md
2. **Plan**: Claude proposes Phase 2 architecture (get your approval)
3. **Code**: Claude implements all modules + tests
4. **Commit**: Claude creates git commit with all changes

---

## 📋 Complete Test Instructions

### 1. Initial Setup

```bash
# Clone repository
git clone git@github.com:colehanan1/Plasticity-Guided-Connectome-Network-PGCN.git
cd Plasticity-Guided-Connectome-Network-PGCN
git checkout wow-im-tired

# Create environment
conda env create -f environment.yml
conda activate PGCN

# Install project (editable)
pip install -e .[dev]
```

### 2. Verify Data Files Present

```bash
# Check that cache files exist
ls -lh data/cache/ | head -20

# Expected files (Phase 1 cache):
# - nodes.parquet (431K)
# - edges.parquet (202K)
# - alpn_extracted.csv
# - kc_ab.csv, kc_ab_p.csv, kc_apb.csv, ... (8 KC subtypes)
# - mbon_all.csv
# - dan_all.csv, dan_mb.csv, dan_calyx.csv, dan_ml.csv
```

### 3. Run Phase 1 Full Test Suite

```bash
# Run all Phase 1 tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest \
  tests/models/test_connectivity_matrix.py \
  tests/models/test_circuit_loader.py \
  tests/models/test_olfactory_circuit.py \
  tests/integration/test_phase1_integration.py \
  -v --tb=short

# Expected: 59 passed in ~30s
```

### 4. Run Individual Test Categories

```bash
# Connectivity loading tests (18 tests)
PYTHONPATH=src pytest tests/models/test_connectivity_matrix.py -v --tb=short

# Circuit loader tests (12 tests)
PYTHONPATH=src pytest tests/models/test_circuit_loader.py -v --tb=short

# Forward propagation tests (18 tests)
PYTHONPATH=src pytest tests/models/test_olfactory_circuit.py -v --tb=short

# End-to-end pipeline tests (11 tests)
PYTHONPATH=src pytest tests/integration/test_phase1_integration.py -v --tb=short
```

### 5. Inspect Test Coverage

```bash
# Generate coverage report
PYTHONPATH=src pytest tests/models tests/integration \
  --cov=src/pgcn/models/connectivity_matrix \
  --cov=src/pgcn/models/olfactory_circuit \
  --cov=src/data_loaders/circuit_loader \
  --cov-report=term-missing \
  -v

# Expected: >85% coverage for each module
```

### 6. Verify Specific Phase 1 Functionality

#### Test Connectivity Matrix (Immutability, Sparse Format)
```python
python - <<'EOF'
import sys
sys.path.insert(0, 'src')
from data_loaders.circuit_loader import CircuitLoader
from pathlib import Path

# Load
loader = CircuitLoader(cache_dir=Path("data/cache"))
conn = loader.load_connectivity_matrix()

# Verify shapes
assert conn.pn_to_kc.shape[1] == len(conn.pn_ids), "PN→KC shape mismatch"
assert conn.kc_to_mbon.shape[0] == len(conn.mbon_ids), "KC→MBON shape mismatch"

# Verify sparse format
import scipy.sparse as sp
assert sp.issparse(conn.pn_to_kc), "pn_to_kc not sparse"
assert sp.issparse(conn.kc_to_mbon), "kc_to_mbon not sparse"

# Verify immutability
try:
    conn.pn_ids[0] = 999  # Try to modify
    print("❌ FAILED: ConnectivityMatrix is NOT immutable!")
except (ValueError, TypeError):
    print("✓ ConnectivityMatrix is immutable (frozen dataclass)")

print(f"✓ PN→KC matrix: {conn.pn_to_kc.shape}, sparsity {(1 - conn.pn_to_kc.nnz / (conn.pn_to_kc.shape[0] * conn.pn_to_kc.shape[1])):.1%}")
print(f"✓ KC→MBON matrix: {conn.kc_to_mbon.shape}, sparsity {(1 - conn.kc_to_mbon.nnz / (conn.kc_to_mbon.shape[0] * conn.kc_to_mbon.shape[1])):.1%}")
EOF
```

#### Test Forward Propagation (k-WTA Sparsity)
```python
python - <<'EOF'
import sys
sys.path.insert(0, 'src')
from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.olfactory_circuit import OlfactoryCircuit
import numpy as np
from pathlib import Path

# Load circuit
loader = CircuitLoader(cache_dir=Path("data/cache"))
conn = loader.load_connectivity_matrix()
circuit = OlfactoryCircuit(conn, kc_sparsity_target=0.05)

# Test 100 random inputs
for trial in range(100):
    pn_activity = np.random.rand(len(conn.pn_ids))
    mbon_output, diag = circuit.forward_pass(pn_activity, return_intermediates=True)
    
    # Check sparsity is enforced
    sparsity = diag['sparsity_fraction']
    assert 0.04 <= sparsity <= 0.06, f"Sparsity {sparsity:.3f} out of range [0.04, 0.06]"
    
    # Check outputs are finite
    assert np.all(np.isfinite(mbon_output)), f"MBON output contains NaN/Inf"

print(f"✓ 100 trials completed")
print(f"✓ KC sparsity maintained within 5% ± 1%")
print(f"✓ All MBON outputs finite (no NaN/Inf)")
EOF
```

#### Test Normalization Strategies
```python
python - <<'EOF'
import sys
sys.path.insert(0, 'src')
from data_loaders.circuit_loader import CircuitLoader
from pathlib import Path
import numpy as np

loader = CircuitLoader(cache_dir=Path("data/cache"))

# Test three normalization modes
for mode in ["row", "global", "none"]:
    conn = loader.load_connectivity_matrix(normalize_weights=mode)
    
    # Check all weights are non-negative
    assert (conn.pn_to_kc.data >= 0).all(), f"{mode}: Found negative weights in PN→KC"
    assert (conn.kc_to_mbon.data >= 0).all(), f"{mode}: Found negative weights in KC→MBON"
    
    # Row normalization: rows sum to 1
    if mode == "row":
        row_sums = np.array(conn.pn_to_kc.sum(axis=1)).flatten()
        nonzero_rows = row_sums > 0
        assert np.allclose(row_sums[nonzero_rows], 1.0), f"{mode}: Rows don't sum to 1"
    
    print(f"✓ Normalization mode '{mode}' verified")
EOF
```

### 7. View Detailed Test Results

```bash
# Show full output for one test file
PYTHONPATH=src pytest tests/models/test_connectivity_matrix.py -vv --tb=long

# Show only failed tests (if any)
PYTHONPATH=src pytest tests/models tests/integration -v --tb=short -x
```

---

## 📊 Phase 1 Deliverables Summary

| Component | Lines | Status | Tests |
|-----------|-------|--------|-------|
| `connectivity_matrix.py` | 550 | ✅ | 18 |
| `circuit_loader.py` | 650 | ✅ | 12 |
| `olfactory_circuit.py` | 500 | ✅ | 18 |
| Integration suite | — | ✅ | 11 |
| **Total** | **2,985** | ✅ **COMPLETE** | **59/59** |

### Biological Grounding Examples

Every function includes rationale like:

```python
def _apply_k_winners_take_all(self, activations: np.ndarray, k: int) -> np.ndarray:
    """Keep top-k activations, zero the rest (k-WTA lateral inhibition).
    
    Biological Rationale:
    The MB circuit implements lateral inhibition via GABAergic APL:
    1. Strong PN input → some KCs depolarize
    2. Depolarized KCs activate APL (feedforward excitation)
    3. APL releases GABA onto all KCs (feedback inhibition)
    4. Weakly-depolarized KCs suppressed below threshold
    5. Only top ~5% of KCs escape inhibition and fire
    
    This is approximated here as static k-WTA: rank by activation, keep top k.
    """
```

---

## 🔧 Troubleshooting

### Issue: "Cannot find cache files"

**Solution**: Verify data directory structure:
```bash
ls -la data/cache/
# Should show: nodes.parquet, edges.parquet, alpn_extracted.csv, kc_*.csv, etc.

# If empty, check if extraction scripts have been run:
PYTHONPATH=src python scripts/extract_alpn_projection_neurons.py \
  --dataset-dir data/flywire \
  --output-dir data/cache
```

### Issue: "Plugin conflicts" during pytest

**Solution**: Use environment variables to disable napari/npe2:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest tests/
```

### Issue: "pytest: command not found"

**Solution**: Install dev dependencies:
```bash
pip install -e .[dev]
# OR
pip install pytest pytest-cov
```

### Issue: Tests run but shapes don't match

**Solution**: Verify cache was generated from correct FlyWire version:
```bash
# Check cache metadata
python -c "import json; print(json.load(open('data/cache/meta.json')))"

# Expected: FlyWire v783 with ~376 PNs, 5177 KCs, 96 MBONs, 584 DANs
```

---

## 📖 Project Structure

```
src/
├── pgcn/
│   ├── models/
│   │   ├── connectivity_matrix.py    (Phase 1: Immutable connectivity)
│   │   ├── olfactory_circuit.py      (Phase 1: Forward propagation)
│   │   └── learning_model.py         (Phase 2: Plasticity)
│   └── experiments/
│       ├── experiment_1_veto_gate.py           (Phase 2)
│       ├── experiment_2_counterfactual.py      (Phase 2)
│       ├── experiment_3_eligibility_traces.py  (Phase 2)
│       └── experiment_6_shapley_analysis.py    (Phase 2)
├── data_loaders/
│   └── circuit_loader.py            (Phase 1: Cache loading)
└── utils/
    └── ...

tests/
├── models/
│   ├── test_connectivity_matrix.py   (18 tests, Phase 1)
│   ├── test_circuit_loader.py        (12 tests, Phase 1)
│   └── test_olfactory_circuit.py     (18 tests, Phase 1)
├── experiments/
│   ├── test_experiment_1_veto.py           (Phase 2)
│   ├── test_experiment_2_microsurgery.py   (Phase 2)
│   └── ...
└── integration/
    ├── test_phase1_integration.py    (11 tests, Phase 1)
    └── test_phase2_full_pipeline.py  (Phase 2)

data/
├── cache/               (FlyWire connectome artifacts)
│   ├── nodes.parquet
│   ├── edges.parquet
│   ├── alpn_extracted.csv
│   └── ...
└── flywire/            (Raw FlyWire exports)
    ├── connections_princeton.csv.gz
    ├── neurons.csv.gz
    └── ...
```

---

## 🎓 Next Steps

### To generate Phase 2 code:

1. **Copy Phase 2 prompt** [28]
2. **Paste into Claude**
3. **Claude will**:
   - Explore Phase 1 code (confirm interfaces)
   - Propose learning architecture (wait for approval)
   - Implement all Phase 2 modules + tests
   - Run pytest to verify all pass
   - Create git commit

### After Phase 2:

- Phase 3: Optogenetic conditioning experiments
- Phase 4: Behavioral validation (compare to real fly learning curves)
- Phase 5: Documentation and benchmarking

---

## 📞 Questions?

Refer to:
- `PROJECT_STATUS.md` — Overall project roadmap and completion status
- `NEXT_STEPS_DETAILED.md` — Detailed build plan for each phase
- `CHECKLIST.md` — Daily execution checklist
- `docs/model_integration_status.md` — Connectome-behavior integration progress

---

**Last updated**: October 30, 2025  
**Phase 1 Status**: ✅ Complete (59/59 tests passing)  
**Phase 2 Status**: 🚀 Ready for implementation
