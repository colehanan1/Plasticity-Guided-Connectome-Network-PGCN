# PGCN Phase 1-2-3: Quick Reference & Implementation Guide

## 📋 At a Glance

| Phase | Prompt ID | Status | Code | Tests | Files | Key Focus |
|-------|-----------|--------|------|-------|-------|-----------|
| 1 | [27] | ✅ Complete | 2,985 lines | 59 ✅ | 6 | Connectivity |
| 2 | [28] | ✅ Complete | 3,000+ lines | 60+ ✅ | 8 | Learning |
| 3 | [31] | 🚀 Ready | 1,500+ lines | 20+ 🚀 | 6 | Optogenetics + README |

---

## 🎯 What Each Prompt Does

### Phase 1 Prompt [27]
**Purpose**: Build the connectivity backbone
- Loads FlyWire connectome into sparse matrices
- Implements k-WTA KC sparsity (~5%)
- Tests all 59 tests pass

### Phase 2 Prompt [28]
**Purpose**: Add dopamine-gated learning
- Three-factor Hebbian rule (KC × MBON × dopamine)
- Veto gates, ablations, eligibility traces, Shapley analysis
- Tests all 60+ tests pass

### Phase 3 Prompt [31]
**Purpose**: Optogenetics + validation + README updates ⭐
- Optogenetic perturbations (silence/activate neurons)
- Behavioral validation against real fly data
- Multi-task learning analysis
- **CRITICAL**: Updates README with test commands for ALL phases
- Tests all 20+ tests pass + 140+ total tests pass

---

## 🚀 How to Use (Step by Step)

### Option A: Implement All 3 Phases Sequentially

**Week 1 - Phase 1**:
```bash
# 1. Copy prompt [27]
# 2. Paste into Claude
# 3. Claude implements Phase 1 (59 tests pass)
# 4. Verify:
pytest tests/models tests/integration -v
# Result: 59 passed ✅
```

**Week 2 - Phase 2**:
```bash
# 1. Copy prompt [28]
# 2. Paste into Claude
# 3. Claude implements Phase 2 (60+ tests pass)
# 4. Verify Phase 1 + 2 together:
pytest tests/models tests/experiments tests/integration -v
# Result: 120+ passed ✅
```

**Week 3 - Phase 3** ⭐ (Different from Phase 1 & 2):
```bash
# 1. Copy prompt [31]
# 2. Paste into Claude
# 3. Claude implements Phase 3 + UPDATES README
# 4. Verify ALL phases together:
pytest tests/ -v
# Result: 140+ passed ✅

# 5. Test README instructions by copying from README:
python - << 'EOF'
... (end-to-end verification script from Phase 3 README section)
EOF
# Result: Phase 1 ✓ Phase 2 ✓ Phase 3 ✓
```

### Option B: Give Claude All 3 Prompts At Once

```bash
# 1. Concatenate all three prompts
cat phase1-implementation-prompt.md phase2-implementation-prompt.md phase3-implementation-prompt.md > all-phases.md

# 2. Ask Claude: "Implement phases 1, 2, 3 in sequence (not parallel)"
# 3. Claude will take longer but deliver all at once
```

---

## 🎯 What Phase 3 Prompt Requires (Critical Difference)

### Phases 1 & 2: "Just make the code work"
- Implement classes + methods ✓
- Write tests ✓
- All tests pass ✓

### Phase 3: "Make the code work AND make it easy for users"
- Implement classes + methods ✓
- Write tests ✓
- All tests pass ✓
- **Update README with Phase 2 test section** ← NEW
- **Update README with Phase 3 test section** ← NEW
- **Update README with end-to-end test section** ← NEW
- **Include copy-paste commands that users can run without modification** ← NEW
- **Include expected outputs so users know success** ← NEW
- **Include verification script that exercises Phase 1→2→3 together** ← NEW

**Why**: Phase 3 is about making sure users can actually verify everything works end-to-end.

---

## ✅ Success Criteria Checklist

### For Phase 1:
- [ ] 59 tests pass
- [ ] All code has type hints + docstrings
- [ ] ConnectivityMatrix is immutable (frozen)
- [ ] Sparse matrices used throughout
- [ ] Biological rationale in every docstring
- [ ] `pytest tests/models tests/integration -v` works

### For Phase 2:
- [ ] Phase 1 still passes (59 tests)
- [ ] Phase 2 tests pass (60+ tests)
- [ ] Learning curves are smooth (no NaN/Inf)
- [ ] RPE computation correct
- [ ] Three-factor rule: dW = α * KC * MBON * dopamine
- [ ] `pytest tests/ -v` shows 120+ passed

### For Phase 3 (Additional Requirements):
- [ ] Phase 1 + 2 still pass (120+ tests)
- [ ] Phase 3 tests pass (20+ tests)
- [ ] **README updated with Phase 2 test section**
- [ ] **README updated with Phase 3 test section**
- [ ] **README updated with end-to-end test section**
- [ ] Copy-paste commands in README work without modification
- [ ] Expected outputs shown in README match actual test runs
- [ ] End-to-end verification script in README works
- [ ] `pytest tests/ -v` shows 140+ passed
- [ ] Users can verify all phases work by following README instructions

---

## 📚 Prompt Contents Quick Reference

### Prompt [27] - Phase 1
**Sections**:
- Context & Background
- Phase 1 Modules Specification (3 modules)
- Testing Specification (7 test files)
- Workflow: Explore, Plan, Code, Commit
- Quality Checklist

**Key Files Specified**:
- src/pgcn/models/connectivity_matrix.py
- src/data_loaders/circuit_loader.py
- src/pgcn/models/olfactory_circuit.py
- tests/models/test_connectivity_matrix.py
- tests/models/test_circuit_loader.py
- tests/models/test_olfactory_circuit.py
- tests/integration/test_phase1_integration.py

### Prompt [28] - Phase 2
**Sections**:
- Phase 1 Summary (Context)
- Phase 2 Architecture: Learning Dynamics Overview
- Phase 2 Modules Specification (5 modules)
- Testing Specification (5 test files)
- Workflow: Explore, Plan, Code, Commit
- Quality Checklist

**Key Files Specified**:
- src/pgcn/models/learning_model.py
- src/pgcn/experiments/experiment_1_veto_gate.py
- src/pgcn/experiments/experiment_2_counterfactual_microsurgery.py
- src/pgcn/experiments/experiment_3_eligibility_traces.py
- src/pgcn/experiments/experiment_6_shapley_analysis.py
- tests/models/test_learning_model.py
- tests/experiments/test_experiment_*.py
- tests/integration/test_phase2_full_pipeline.py

### Prompt [31] - Phase 3 ⭐
**Sections**:
- Phase 1 + 2 Summary (Context)
- Phase 3 Architecture: Optogenetic Experiments & Behavioral Validation
- Phase 3 Modules Specification (3 modules)
- Testing Specification (3+ test files)
- **🚨 CRITICAL: README Update Requirements** ← UNIQUE TO PHASE 3
  - Phase 2 Test Commands Section
  - Phase 3 Test Commands Section
  - Full End-to-End Test Section
  - Verification Script Section
- Workflow: Explore, Plan, Code, Commit
- Quality Checklist

**Key Files Specified**:
- src/pgcn/experiments/optogenetic_perturbations.py
- src/pgcn/analysis/behavioral_validation.py
- src/pgcn/analysis/multi_task_analysis.py
- tests/experiments/test_optogenetic_experiments.py
- tests/analysis/test_behavioral_validation.py
- tests/analysis/test_multi_task_analysis.py
- **README.md (UPDATED)** ← Important!

---

## 🔍 Verifying Everything Works

### Phase 1 Verification
```bash
# Run Phase 1 tests
PYTHONPATH=src pytest tests/models tests/integration -v --tb=short
# Expected: 59 passed
```

### Phase 2 Verification
```bash
# Run Phase 1 + Phase 2 tests
PYTHONPATH=src pytest tests/models tests/experiments tests/integration -v --tb=short
# Expected: 120+ passed
```

### Phase 3 Verification ⭐
```bash
# Run ALL tests (Phase 1 + 2 + 3)
PYTHONPATH=src pytest tests/ -v --tb=short
# Expected: 140+ passed

# Verify README instructions work
# Find "Phase 2 Test Commands" section in README
# Copy one command and run it
# Verify it works ✓

# Find "Phase 3 Test Commands" section in README
# Copy one command and run it
# Verify it works ✓

# Find "End-to-End Verification Script" section in README
# Copy script and run it
# Verify it shows: Phase 1 ✓ Phase 2 ✓ Phase 3 ✓
```

---

## 📞 Troubleshooting

### Issue: Tests fail with plugin errors
**Solution**: Use environment variable
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest tests/ -v
```

### Issue: Missing data files
**Solution**: Ensure data/cache/ is populated
```bash
ls data/cache/ | grep parquet
# Should show: nodes.parquet, edges.parquet
```

### Issue: Phase 3 README not updated
**Solution**: Check that prompt [31] includes "README Update Requirements"
- Phase 3 prompt MUST include explicit README update instructions
- Claude MUST be asked to update README
- Verify README has new sections:
  - "Phase 2 Status:" section
  - "Phase 3 Status:" section
  - "Full Pipeline: Phase 1 → Phase 2 → Phase 3" section

---

## 🎓 Learning Path for Users

### After Phase 1:
Users understand how FlyWire connectome is loaded and forward-passed

### After Phase 2:
Users understand how learning occurs with dopamine-gated plasticity

### After Phase 3:
Users understand:
- How optogenetic perturbations test causality
- How model predictions validate against real fly data
- How multiple tasks can be learned on shared circuit
- How to verify all phases work together with single command

---

## 💡 Key Innovation of This Approach

**Traditional software**: "Here's the code, test it yourself"

**PGCN Phases 1-3**: "Here's the code, the tests, and easy copy-paste instructions to verify everything works end-to-end"

**Result**: Users can clone repo and run ONE command to verify all phases work together ✓

---

## 📅 Implementation Timeline (Recommended)

| Week | Phase | Tasks | Expected Result |
|------|-------|-------|-----------------|
| 1 | Phase 1 | Give Claude [27], implement, test | 59/59 tests pass |
| 2 | Phase 2 | Give Claude [28], implement, test | 120+/120+ tests pass |
| 3 | Phase 3 | Give Claude [31], implement, test, **update README** | 140+/140+ tests pass + README updated |
| 4 | Documentation | Write tutorials, analysis, papers | Full project documented |

---

## 🎯 Final Deliverable

After all three phases:

```bash
# Single command users can run to verify everything
pytest tests/ -v

# Expected output:
# tests/models/test_connectivity_matrix.py::... 18 PASSED
# tests/models/test_circuit_loader.py::... 12 PASSED
# tests/models/test_olfactory_circuit.py::... 18 PASSED
# tests/integration/test_phase1_integration.py::... 11 PASSED
# tests/models/test_learning_model.py::... 15 PASSED
# tests/experiments/test_experiment_*.py::... 40 PASSED
# tests/integration/test_phase2_full_pipeline.py::... 15 PASSED
# tests/experiments/test_optogenetic_experiments.py::... 10 PASSED
# tests/analysis/test_behavioral_validation.py::... 8 PASSED
# tests/analysis/test_multi_task_analysis.py::... 10 PASSED

# ====== 140+ passed in ~120s ======
```

Plus:
- README with copy-paste test commands for each phase
- End-to-end verification script
- Biological documentation throughout
- Full type hints + docstrings

---

**Status**: ✅ Phase 1 & 2 complete, 🚀 Phase 3 ready to implement

**Use [31] to implement Phase 3 with full README updates!**
