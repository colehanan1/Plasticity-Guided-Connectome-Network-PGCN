# AGENTS.md

## Project Overview

**Drosophila Cross-Odor Generalization Model** - A biologically-constrained neural network that predicts fruit fly olfactory learning and cross-odor generalization using real chemical properties and connectome data. This 16-week research implementation follows a "Build Now, Validate Later" philosophy to model how flies trained with one odor generalize to chemically similar/dissimilar test odors.

> **Offline FlyWire reminder**: The authenticated Codex pipeline is now optional. Local FAFB v783 CSV exports (see `data/flywire/`) are the primary source for KC/PN connectivity. Always run `pgcn-cache --local-data data/flywire --out data/cache/` before behavioural analyses and consult `docs/model_integration_status.md` for the latest integration status and verification commands.

## Core Commands

```bash
# Development
python -m venv .venv && source .venv/bin/activate  # Setup environment
pip install -r requirements.txt                   # Install dependencies  
python train.py --config configs/baseline.yaml    # Train baseline model
python evaluate.py --model checkpoints/best.pt    # Evaluate model

# File-scoped checks (preferred for fast feedback)
python -m pytest tests/test_reservoir.py -v       # Test specific module
python -m flake8 models/chemical_model.py         # Lint single file
python -m mypy models/reservoir.py               # Type check single file
python -c "import models.reservoir; print('✓')"   # Quick import test

# Data validation
python data/validate_behavioral_data.py           # Verify CSV integrity
python analysis/cross_validation.py --folds 5     # Run CV with statistical tests (all datasets)
python analysis/cross_validation.py --dataset opto_EB --folds 5  # Run CV for specific dataset
python analysis/cross_validation.py --per-dataset --folds 5  # Run separate CV for each dataset
python analysis/cross_validation.py --skip-stats  # Run CV without statistical tests
python analysis/run_statistical_tests.py --artifacts-dir artifacts/cross_validation  # Standalone statistical analysis

# Full suite (use sparingly - takes 5-10 minutes)
python -m pytest tests/ --coverage               # All tests with coverage
python train_all_conditions.py                   # Train all experimental conditions
```

## Do

- **Data Integrity**: Always validate that `len(df) == 440` and fly groupings are intact
- **Chemical Constraints**: Use real molecular properties from `CHEMICAL_PROPERTIES` dict, never arbitrary embeddings
- **Biological Realism**: Implement 5% KC sparse activation, only train KC→MBON weights
- **Cross-Validation**: Always use `GroupKFold` with `groups=df['fly'].values` to prevent data leakage
- **Performance Targets**: Overall accuracy >70%, trained odor recognition >80%, control separation >90%
- **Small Focused Models**: Default to modular components, single responsibility classes
- **Real Connectome Data**: Use hemibrain PN→KC connectivity when available (`neuprint-python`)
- **Chemical Similarity**: Base on functional groups + molecular weight + literature values
- **Proper Validation**: Report accuracy, AUROC, cross-fly generalization for every model
- **Scientific Rigor**: Include statistical significance testing, confidence intervals

## Don't  

- **No Data Leakage**: Never split randomly - always group by `fly_id` 
- **No Synthetic Data**: Don't generate fake behavioral responses or chemical properties
- **No Arbitrary Architectures**: Don't make all connections trainable - biological constraints matter
- **No Missing Controls**: Always validate `hex_control` shows 0% response to "trained" odor
- **No Unrealistic Performance**: Don't claim >95% accuracy without exceptional evidence
- **No Hard-Coded Values**: Use constants, config files, or environment variables
- **No Skipping Baselines**: Always compare against chance performance (52% reaction rate)
- **No Ignoring Failed Runs**: Document and analyze failures, don't just report successes

## Project Structure

```
drosophila-learning-model/
├── models/                    # Neural architectures
│   ├── reservoir.py          # Core DrosophilaReservoir (2000 KC, 10 MBON)
│   ├── chemical_model.py     # ChemicallyInformedDrosophilaModel  
│   ├── dual_pathway.py       # MB (plastic) + LH (innate) integration
│   └── plasticity_rules.py   # STDP, dopamine modulation, homeostasis
├── data/                     # Data loading and biological constraints  
│   ├── behavioral_data.py    # Load model_predictions.csv (440 trials, 35 flies)
│   ├── chemical_properties.py # 7 odors with molecular descriptors
│   ├── odor_mappings.py      # testing_X → chemical identity mappings
│   └── connectome_data.py    # Hemibrain PN→KC connectivity (neuPrint API)
├── analysis/                 # Model interpretation and validation
│   ├── cross_validation.py   # GroupKFold, leave-one-fly-out, transfer learning
│   ├── plasticity_analysis.py # KC→MBON hotspot identification
│   ├── statistical_tests.py  # Permutation tests, bootstrap CIs, correlations, effect sizes
│   └── run_statistical_tests.py # Standalone statistical analysis script
├── configs/                  # Hyperparameters and experimental settings
│   ├── model_config.yaml     # Architecture: n_pn=50, n_kc=2000, n_mbon=10
│   └── experiment_config.yaml # Training: lr=0.001, batch_size=32, epochs=100
├── app/                      # Interactive analysis dashboard
│   ├── streamlit_dashboard.py # Chemical generalization explorer
│   └── visualization_utils.py # Plotting with consistent color schemes
├── docs/                     # Documentation and specifications
│   └── statistical_reporting_format.md # Statistical testing format and Week 12+ templates
└── cache/                    # Downloaded data (neuPrint, literature)
    ├── pn_kc_connectivity.npy # Hemibrain connectome weights
    └── chemical_similarity.json # Literature-derived similarity matrix
```

## Experimental Data Context (CRITICAL)

### Input Data: `model_predictions.csv` (440 trials, 35 flies)
- **dataset**: Training condition (`opto_EB`, `opto_benz_1`, `opto_hex`, `hex_control`)
- **fly**: Individual identifier (35 unique flies, grouped for CV)
- **trial_label**: Test odor (`testing_1` through `testing_10`)  
- **prediction**: Binary outcome (0=no reaction, 1=reaction/approach)

### Key Experimental Design
- **Trained Odors**: `testing_2`, `testing_4`, `testing_5` = Same odor used for training
- **Cross-Odor Tests**: `testing_1`, `testing_3`, `testing_6-10` = Novel odors
- **Response Rates**: opto_EB=80%, opto_hex=84%, opto_benz_1=26%, hex_control=0%

### Complete Chemical Mappings
```python
# These mappings are GROUND TRUTH - never modify
ODOR_MAPPINGS = {
    'opto_EB': {  # Ethyl butyrate training (80% response to trained)
        'testing_2/4/5': 'ethyl_butyrate',    # TRAINED ODOR
        'testing_1/3': 'hexanol',             # Similar alcohol  
        'testing_6': 'apple_cider_vinegar',   # Strongly avoided (4.5%)
        'testing_7': '3-octanol',             # Similar alcohol (50%)
        'testing_8': 'benzaldehyde',          # Different class (38.6%)
        'testing_9': 'citral', 'testing_10': 'linalool'  # Terpenes (31.8%, 40.9%)
    },
    'opto_hex': {  # Hexanol training (84% response to trained)  
        'testing_2/4/5': 'hexanol',           # TRAINED ODOR
        'testing_1/3': 'apple_cider_vinegar', # Different class
        'testing_6': 'benzaldehyde',          # Different class  
        'testing_7': '3-octanol',             # Similar alcohol
        'testing_8': 'ethyl_butyrate',        # Different class
        'testing_9': 'citral', 'testing_10': 'linalool'  # Terpenes
    },
    'opto_benz_1': {  # Benzaldehyde training (26% response - poor learning)
        'testing_2/4/5': 'benzaldehyde',      # TRAINED ODOR  
        'testing_1/3': 'hexanol',             # Different class
        'testing_6': 'apple_cider_vinegar',   # Different class
        'testing_7': '3-octanol',             # Different class
        'testing_8': 'ethyl_butyrate',        # Different class  
        'testing_9': 'citral', 'testing_10': 'linalool'  # Terpenes
    }
}
```

## Chemical Properties Implementation

Always use these molecular descriptors for similarity calculations:
```python
CHEMICAL_PROPERTIES = {
    'ethyl_butyrate': {
        'class': 'ester', 'mw': 116.16, 'carbon_length': 6,
        'functional_groups': ['ester'], 'odor_descriptor': ['fruity', 'sweet']
    },
    'hexanol': {
        'class': 'alcohol', 'mw': 102.17, 'carbon_length': 6, 
        'functional_groups': ['alcohol'], 'odor_descriptor': ['floral', 'green']
    },
    'benzaldehyde': {
        'class': 'aldehyde', 'mw': 106.12, 'carbon_length': 7,
        'functional_groups': ['aldehyde', 'aromatic'], 'odor_descriptor': ['almond', 'sweet']
    },
    # ... complete chemical database in data/chemical_properties.py
}

def compute_chemical_similarity(odor1, odor2):
    """ALWAYS use this function - don't create arbitrary similarity metrics"""
    props1, props2 = CHEMICAL_PROPERTIES[odor1], CHEMICAL_PROPERTIES[odor2]
    
    # Functional group overlap
    groups1 = set(props1['functional_groups'])
    groups2 = set(props2['functional_groups'])
    functional_sim = len(groups1 & groups2) / len(groups1 | groups2)
    
    # Molecular weight similarity  
    mw_sim = 1 - abs(props1['mw'] - props2['mw']) / max(props1['mw'], props2['mw'])
    
    # Literature similarity (from published behavioral studies)
    lit_sim = LITERATURE_SIMILARITY.get((odor1, odor2), 0.2)
    
    return 0.4 * functional_sim + 0.3 * mw_sim + 0.3 * lit_sim
```

## Required Model Architecture

```python
class DrosophilaReservoir(nn.Module):
    def __init__(self, n_pn=50, n_kc=2000, n_mbon=10):
        # Fixed PN→KC connectivity (biological constraint)
        self.pn_kc_weights = self.load_hemibrain_connectivity()  # Don't train these!
        self.pn_kc_weights.requires_grad = False
        
        # Only trainable weights: KC→MBON readout
        self.kc_mbon_weights = nn.Linear(n_kc, n_mbon)
        
    def forward(self, odor_input):
        # Sparse KC activation (5% active)
        kc_activity = torch.relu(self.pn_kc_weights @ odor_input)  
        kc_sparse = self.enforce_sparsity(kc_activity, sparsity=0.05)
        
        # Only this layer learns
        return self.kc_mbon_weights(kc_sparse)
    
    def enforce_sparsity(self, activity, sparsity=0.05):
        k = int(sparsity * len(activity))
        topk_values, topk_indices = torch.topk(activity, k)
        sparse_activity = torch.zeros_like(activity)
        sparse_activity[topk_indices] = topk_values
        return sparse_activity
```

## Validation Requirements

### Mandatory Checks (include in every script)
```python
# Data integrity  
assert len(df) == 440, "Expected 440 total trials"
assert set(df['dataset'].unique()) == {'opto_EB', 'opto_benz_1', 'opto_hex', 'hex_control'}
assert df['prediction'].isin([0, 1]).all(), "Binary predictions only"

# Cross-validation setup
groups = df['fly'].values  # NEVER split without grouping
cv = GroupKFold(n_splits=5)

# Model architecture  
assert kc_activity.sum() / len(kc_activity) <= 0.06, "KC activation must be ~5% sparse"
assert hasattr(model, 'kc_mbon_weights'), "Only KC→MBON should be trainable"

# Performance benchmarks
assert accuracy > 0.52, "Must beat baseline reaction rate (52%)"
assert control_accuracy < 0.10, "hex_control should show ~0% response"
```

### Performance Targets
```python
PERFORMANCE_TARGETS = {
    'overall_accuracy': 0.70,              # Significantly above chance
    'trained_odor_auroc': 0.80,            # Strong trained odor recognition
    'cross_fly_generalization': 0.65,      # Leave-one-fly-out performance
    'control_separation': 0.90,            # hex_control vs trained flies
    'chemical_similarity_correlation': 0.4  # Similarity predicts generalization
}
```

### Statistical Testing Requirements

All model validation MUST include statistical significance testing and effect sizes:

```python
# Automatically included in cross-validation
python analysis/cross_validation.py --folds 5
# Generates: artifacts/cross_validation/{prefix}_statistical_report.json

# Standalone statistical analysis (re-run without re-training)
python analysis/run_statistical_tests.py --artifacts-dir artifacts/cross_validation

# Custom parameters
python analysis/cross_validation.py \
    --n-permutations 10000 \
    --n-bootstrap-samples 10000 \
    --skip-stats  # For quick debugging runs only
```

**Required Statistical Tests:**
1. **Permutation tests vs chance level (52%)**: Overall accuracy, trained odor accuracy, AUROC
2. **Between-condition comparisons**: opto_EB vs opto_hex vs opto_benz_1
3. **Bootstrap confidence intervals**: 95% and 99% CIs for all metrics
4. **Chemical similarity correlations**: Pearson r and Spearman ρ with significance tests
5. **Effect sizes**: Cohen's d and eta-squared for all comparisons

**Reporting Standards:**
- Always report p-values AND effect sizes (never p-value alone)
- Include confidence intervals for point estimates
- Use α = 0.05 for significance, but report exact p-values
- Interpret effect sizes: small (d=0.2), medium (d=0.5), large (d=0.8)
- See `docs/statistical_reporting_format.md` for complete specifications

**Example Output:**
```
overall_accuracy:
  Observed: 0.7300 vs chance 0.52
  p-value: 0.0012 ✓✓
  Cohen's d: 2.100 (large)

Confidence Intervals (95%):
  overall_accuracy: [0.7000, 0.7600]
  trained_odor_accuracy: [0.7800, 0.8600]
```

### Per-Dataset Analysis

Run statistical tests separately for each training condition:

```python
# Analyze specific dataset only
python analysis/cross_validation.py --dataset opto_EB --folds 5
# Output: artifacts/cross_validation/week4_statistical_report.json (opto_EB only)

# Analyze multiple datasets separately
python analysis/cross_validation.py --dataset opto_EB opto_hex --folds 5
# Output: artifacts/cross_validation/week4_statistical_report.json (combined opto_EB + opto_hex)

# Run SEPARATE analysis for each dataset
python analysis/cross_validation.py --per-dataset --folds 5
# Output structure:
#   artifacts/cross_validation/opto_EB/week4_statistical_report.json
#   artifacts/cross_validation/opto_hex/week4_statistical_report.json
#   artifacts/cross_validation/opto_benz_1/week4_statistical_report.json
#   artifacts/cross_validation/EB_control/week4_statistical_report.json
#   artifacts/cross_validation/hex_control/week4_statistical_report.json
```

**Use Cases:**
- **Single dataset:** Compare one training condition against chance (e.g., how well does opto_EB perform?)
- **Per-dataset:** Compare performance characteristics across training conditions independently
- **Combined:** Pool all datasets for overall model performance (default behavior)

**Example per-dataset output:**
```
================================================================================
Running cross-validation for dataset: opto_EB
================================================================================

overall_accuracy: mean=0.547, p=0.0000 ✓✓, Cohen's d=3.407 (large)
trained_odor_accuracy: mean=0.479, p=0.2532 ✗

================================================================================
Running cross-validation for dataset: opto_hex
================================================================================

overall_accuracy: mean=0.456, p=0.0001 ✓✓, Cohen's d=2.845 (large)
trained_odor_accuracy: mean=0.771, p=0.0012 ✓✓
```

## Safety and Permissions

### Allowed without approval:
- Read any project files and documentation
- Run single-file checks: `python -m pytest tests/test_file.py`
- Load and analyze behavioral data (`model_predictions.csv`)  
- Train models with existing hyperparameters
- Generate plots and analysis notebooks
- Create/modify model files following established patterns

### Require approval before:
- Installing new packages (`pip install`)
- Downloading large datasets (>100MB)
- Running full training sweeps (>1 hour compute)
- Modifying core data files or experimental mappings
- Changing fundamental model architecture constraints
- Publishing results or creating external connections

## Good and Bad Examples

### ✅ Follow These Patterns:
- **Data Loading**: `data/behavioral_data.py` - proper GroupKFold splits
- **Model Architecture**: `models/reservoir.py` - biological constraints implemented
- **Chemical Similarity**: `data/chemical_properties.py` - real molecular descriptors
- **Cross-Validation**: `analysis/cross_validation.py` - grouped splits, proper metrics
- **Visualization**: `app/visualization_utils.py` - consistent color schemes, publication-ready

### ❌ Avoid These Patterns:
- Random train/test splits (causes data leakage)
- Arbitrary odor embeddings (use chemical properties)
- Fully trainable networks (violates biological constraints)
- Missing control validations (hex_control must show 0% response)
- Unrealistic performance claims without proper validation

## External APIs and Data Sources

### Hemibrain Connectome (neuPrint API)
```python
from neuprint import Client
client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1')

# Cache downloaded data locally (never re-download)
cache_dir = 'cache/'
if not os.path.exists(f'{cache_dir}/pn_kc_connectivity.npy'):
    connectivity = query_pn_kc_connections(client)
    np.save(f'{cache_dir}/pn_kc_connectivity.npy', connectivity)
```

### Required Libraries
```python
# Core ML stack
torch>=1.9.0              # Neural networks
scikit-learn>=1.0.0        # Cross-validation, metrics, AUROC
pandas>=1.3.0              # Data manipulation
numpy>=1.21.0              # Numerical operations

# Statistical analysis
scipy>=1.7.0               # Permutation tests, correlations, statistical functions

# Neuroscience-specific
neuprint-python>=0.4.0     # Connectome data access

# Visualization
matplotlib>=3.4.0          # Basic plotting
seaborn>=0.11.0            # Statistical visualization
streamlit>=1.0.0           # Interactive dashboard
```

## Milestone Validation (16-Week Timeline)

### Week 4 Checkpoint: "Behavioral Baseline"
- [ ] Model accuracy >60% on cross-validated test set
- [ ] Trained odors show >70% response rate across conditions
- [ ] hex_control shows <10% response (demonstrates learning specificity)
- [ ] Chemical similarity correlates with behavioral responses (r>0.3)

### Week 8 Checkpoint: "Biological Integration"
- [ ] Hemibrain connectivity improves performance >10% vs random
- [ ] STDP plasticity rules converge within 100 epochs  
- [ ] Cross-fly generalization achieves >65% accuracy
- [ ] Plasticity hotspots identified in <10% of KC→MBON connections

### Week 12 Checkpoint: "Scientific Analysis"
- [ ] Chemical feature importance analysis completed with permutation tests for significance
- [ ] Ablation studies demonstrate component necessity with statistical tests (p-values, effect sizes)
- [ ] Statistical significance testing confirms key findings (accuracy > chance, chemical similarity correlations)
- [ ] Bootstrap confidence intervals (95%, 99%) reported for all primary metrics
- [ ] Effect sizes (Cohen's d, eta-squared) computed for all comparisons
- [ ] Between-condition comparisons include permutation tests and effect sizes
- [ ] Statistical report generated automatically after cross-validation
- [ ] Plasticity maps are biologically interpretable

### Week 16 Checkpoint: "Complete System"  
- [ ] Streamlit dashboard functional with all experimental conditions
- [ ] Automated figure generation produces publication-quality plots
- [ ] Full documentation with usage examples and API reference
- [ ] Neural data interface prepared for future calcium imaging integration

## Color Schemes and Visualization Standards

```python
# Use these consistent color schemes across all plots
COLOR_SCHEMES = {
    'training_conditions': {
        'opto_EB': '#1f77b4',      # Blue - ethyl butyrate
        'opto_benz_1': '#ff7f0e',  # Orange - benzaldehyde  
        'opto_hex': '#2ca02c',     # Green - hexanol
        'hex_control': '#d62728'   # Red - control
    },
    'chemical_classes': {
        'alcohol': '#2ca02c',      # Green
        'ester': '#1f77b4',        # Blue
        'aldehyde': '#ff7f0e',     # Orange
        'acid': '#d62728',         # Red
        'terpene': '#9467bd'       # Purple
    },
    'performance': {
        'high': '#2ca02c',         # >70% accuracy  
        'medium': '#ff7f0e',       # 50-70% accuracy
        'low': '#d62728'           # <50% accuracy
    }
}
```

## When Stuck

If uncertain about implementation details:
1. **Ask clarifying questions** about experimental requirements
2. **Reference similar patterns** in existing codebase  
3. **Create minimal working example** first, then expand
4. **Validate against biological literature** before implementing
5. **Test on subset of data** before full training runs

**Never make large speculative changes without confirmation.**

## Scientific Context and Literature

This model implements cross-odor generalization as described in:
- Eschbach et al. (2021) - Dual pathway architecture (MB plastic, LH innate)
- Dolan et al. (2019) - Glomerulus-specific connectivity patterns  
- Springer & Nawrot (2021) - Dopamine-modulated plasticity rules
- Spanier et al. (2024) - Lightweight Drosophila olfactory network architecture

**Key Biological Constraints:**
- ~7 PN inputs per KC (connectivity sparsity)
- ~5% KC activation (population sparsity)  
- Only KC→MBON weights plastic (PN→KC fixed by development)
- Dopamine modulates plasticity based on reward prediction error
- Chemical similarity drives cross-odor generalization strength

---

*This AGENTS.md follows industry standards for AI agent guidance while maintaining scientific rigor appropriate for computational neuroscience research. Update as project evolves and new patterns emerge.*