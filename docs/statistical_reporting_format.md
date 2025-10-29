# Statistical Reporting Format

**Version:** 1.0
**Last Updated:** 2025-10-29
**Purpose:** Define the statistical reporting format for Drosophila cross-odor generalization model analysis

---

## Table of Contents

1. [Overview](#overview)
2. [Report Structure](#report-structure)
3. [Statistical Tests Explained](#statistical-tests-explained)
4. [Output Format Specification](#output-format-specification)
5. [Interpreting Results](#interpreting-results)
6. [Week 12+ Integration](#week-12-integration)
7. [Examples and Templates](#examples-and-templates)
8. [Best Practices](#best-practices)

---

## Overview

The statistical reporting infrastructure provides comprehensive hypothesis testing and effect size analysis for cross-validation results. This document specifies the output format and provides templates for extending the system with additional analyses (e.g., feature importance, ablation studies).

### Key Features

- **Permutation tests** against chance level (52% baseline)
- **Between-condition comparisons** (opto_EB vs opto_hex vs opto_benz_1)
- **Bootstrap confidence intervals** (95% and 99%)
- **Chemical similarity correlations** with behavioral responses
- **Effect sizes** (Cohen's d, eta-squared)

### Statistical Philosophy

We follow the **New Statistics** approach:
- Report effect sizes alongside p-values
- Provide confidence intervals for all estimates
- Use non-parametric tests (permutation, bootstrap) to avoid distributional assumptions
- Emphasize practical significance (effect size) over statistical significance (p-value)

---

## Report Structure

### High-Level Schema

```json
{
  "metadata": { ... },
  "permutation_tests": {
    "vs_chance": { ... },
    "between_conditions": { ... }
  },
  "confidence_intervals": { ... },
  "correlations": { ... },
  "effect_sizes": {
    "vs_chance": { ... },
    "between_conditions": { ... }
  }
}
```

### File Naming Convention

- **Cross-validation artifacts:** `artifacts/cross_validation/`
- **Statistical report:** `{report_prefix}_statistical_report.json`
  - Example: `week4_statistical_report.json`

---

## Statistical Tests Explained

### 1. Permutation Test vs Chance Level

**Purpose:** Test if observed performance significantly exceeds baseline chance level (52%).

**Hypotheses:**
- H₀: metric = chance level (0.52)
- H₁ (one-tailed): metric > 0.52
- H₁ (two-tailed): metric ≠ 0.52

**Method:**
1. Center observed values around chance level
2. Randomly permute signs 5,000 times
3. Compute p-value as proportion of permuted statistics ≥ observed

**When to use:**
- Validating that model performance beats baseline
- Checking if trained odor recognition is above chance
- Demonstrating that AUROC is meaningful

**Example interpretation:**
```
overall_accuracy: observed=0.73, p_one=0.0012, Cohen's d=2.1 (large)
→ Model significantly outperforms chance (p < 0.01) with large effect size
```

---

### 2. Permutation Test Between Conditions

**Purpose:** Test if performance differs between training conditions.

**Hypotheses:**
- H₀: condition_A = condition_B
- H₁ (two-tailed): condition_A ≠ condition_B

**Method:**
1. Pool all values from both conditions
2. Randomly permute condition labels 5,000 times
3. Compute p-value as proportion of permuted differences ≥ observed

**When to use:**
- Comparing opto_EB vs opto_hex learning efficiency
- Testing if benzaldehyde training (opto_benz_1) is worse than other odors
- Validating that conditions differ in expected ways

**Example interpretation:**
```
opto_EB_vs_opto_hex: mean_diff=0.05, p_two=0.24, Cohen's d=0.42 (small)
→ No significant difference between conditions, small effect size
```

---

### 3. Bootstrap Confidence Intervals

**Purpose:** Quantify uncertainty in metric estimates.

**Method:**
1. Resample observed values with replacement 5,000 times
2. Compute mean for each resample
3. Extract 2.5th and 97.5th percentiles (95% CI)
4. Extract 0.5th and 99.5th percentiles (99% CI)

**When to use:**
- Reporting any metric (accuracy, AUROC, correlation)
- Assessing precision of estimates
- Comparing against performance targets

**Example interpretation:**
```
overall_accuracy: mean=0.73, 95% CI=[0.70, 0.76], 99% CI=[0.68, 0.78]
→ 95% confident true accuracy is between 70% and 76%
→ Performance target (70%) is at lower bound of 95% CI
```

---

### 4. Chemical Similarity Correlation

**Purpose:** Test if chemical similarity predicts behavioral generalization.

**Metrics:**
- **Pearson r:** Linear correlation
- **Spearman ρ:** Rank-based correlation (robust to outliers)

**Method:**
1. Compute pairwise chemical similarities
2. Extract corresponding behavioral responses (response rates, prediction probabilities)
3. Compute correlation coefficients and p-values
4. Bootstrap 95%/99% CIs for Pearson r

**When to use:**
- Validating the chemical similarity hypothesis
- Demonstrating that model captures chemical structure
- Comparing biological vs. model generalization patterns

**Example interpretation:**
```
chemical_similarity_vs_response_rate:
  Pearson r=0.45, p=0.003, 95% CI=[0.25, 0.65]
  Spearman ρ=0.42, p=0.005
→ Moderate positive correlation, statistically significant
→ Chemical similarity explains ~20% of variance (r² = 0.20)
```

---

### 5. Effect Sizes

**Cohen's d (standardized mean difference):**
- Small: |d| = 0.2
- Medium: |d| = 0.5
- Large: |d| = 0.8

**Eta-squared (proportion of variance explained):**
- Small: η² = 0.01 (1%)
- Medium: η² = 0.06 (6%)
- Large: η² = 0.14 (14%)

**Why report effect sizes?**
- P-values only indicate significance, not magnitude
- Effect sizes quantify practical importance
- Essential for power analysis and sample size planning
- Allow comparison across studies

---

## Output Format Specification

### Metadata Section

```json
{
  "metadata": {
    "n_folds": 5,
    "n_permutations": 5000,
    "n_bootstrap_samples": 5000,
    "confidence_levels": [0.95, 0.99],
    "chance_level": 0.52
  }
}
```

**Fields:**
- `n_folds`: Number of cross-validation folds
- `n_permutations`: Number of permutation resamples
- `n_bootstrap_samples`: Number of bootstrap resamples
- `confidence_levels`: List of confidence levels used
- `chance_level`: Baseline chance performance

---

### Permutation Tests vs Chance

```json
{
  "permutation_tests": {
    "vs_chance": {
      "overall_accuracy": {
        "observed_mean": 0.73,
        "chance_level": 0.52,
        "p_value_one_tailed": 0.0012,
        "p_value_two_tailed": 0.0024,
        "significant_at_0.05": true,
        "significant_at_0.01": true
      },
      "trained_odor_accuracy": { ... },
      "auroc": { ... },
      "cross_fly_generalization": { ... }
    }
  }
}
```

**Fields:**
- `observed_mean`: Mean of fold metrics
- `chance_level`: Baseline for comparison
- `p_value_one_tailed`: P(null ≥ observed) for H₁: metric > chance
- `p_value_two_tailed`: P(|null - chance| ≥ |observed - chance|)
- `significant_at_0.05`: Boolean flag for α=0.05
- `significant_at_0.01`: Boolean flag for α=0.01

---

### Permutation Tests Between Conditions

```json
{
  "permutation_tests": {
    "between_conditions": {
      "opto_EB_vs_opto_hex": {
        "overall_accuracy": {
          "mean_difference": 0.05,
          "p_value_one_tailed": 0.12,
          "p_value_two_tailed": 0.24,
          "significant_at_0.05": false,
          "significant_at_0.01": false
        },
        "trained_odor_accuracy": { ... },
        "auroc": { ... }
      },
      "opto_EB_vs_opto_benz_1": { ... },
      "opto_hex_vs_opto_benz_1": { ... }
    }
  }
}
```

**Fields:**
- `mean_difference`: Observed difference (condition_A - condition_B)
- `p_value_one_tailed`: P(null ≥ observed) for H₁: A > B
- `p_value_two_tailed`: P(|null| ≥ |observed|)
- `significant_at_0.05`: Boolean flag for α=0.05
- `significant_at_0.01`: Boolean flag for α=0.01

---

### Confidence Intervals

```json
{
  "confidence_intervals": {
    "overall_accuracy": {
      "mean": 0.73,
      "ci_95": [0.70, 0.76],
      "ci_99": [0.68, 0.78]
    },
    "trained_odor_accuracy": { ... },
    "control_separation": { ... },
    "auroc": { ... }
  }
}
```

**Fields:**
- `mean`: Point estimate (mean of fold metrics)
- `ci_95`: 95% confidence interval [lower, upper]
- `ci_99`: 99% confidence interval [lower, upper]

---

### Correlations

```json
{
  "correlations": {
    "chemical_similarity_vs_response_rate": {
      "pearson_r": 0.45,
      "pearson_p": 0.003,
      "spearman_rho": 0.42,
      "spearman_p": 0.005,
      "n_samples": 35,
      "ci_95": [0.25, 0.65],
      "ci_99": [0.20, 0.70],
      "significant_at_0.05": true,
      "significant_at_0.01": true
    },
    "chemical_similarity_vs_prediction_probability": { ... }
  }
}
```

**Fields:**
- `pearson_r`: Pearson correlation coefficient
- `pearson_p`: P-value for Pearson r
- `spearman_rho`: Spearman rank correlation
- `spearman_p`: P-value for Spearman ρ
- `n_samples`: Number of paired observations
- `ci_95`: Bootstrap 95% CI for Pearson r
- `ci_99`: Bootstrap 99% CI for Pearson r
- `significant_at_0.05`: Boolean flag for α=0.05
- `significant_at_0.01`: Boolean flag for α=0.01

---

### Effect Sizes

```json
{
  "effect_sizes": {
    "vs_chance": {
      "overall_accuracy": {
        "cohens_d": 2.1,
        "interpretation": "large"
      },
      "trained_odor_accuracy": { ... }
    },
    "between_conditions": {
      "opto_EB_vs_opto_hex": {
        "overall_accuracy": {
          "cohens_d": 0.42,
          "interpretation": "small",
          "eta_squared": 0.04,
          "eta_squared_interpretation": "small"
        }
      }
    }
  }
}
```

**Fields:**
- `cohens_d`: Standardized mean difference
- `interpretation`: "negligible", "small", "medium", "large"
- `eta_squared`: Proportion of variance explained (two-sample only)
- `eta_squared_interpretation`: "negligible", "small", "medium", "large"

---

## Interpreting Results

### Decision Tree for Interpretation

```
1. Check significance (p-value)
   ├─ p < 0.01: Highly significant (**)
   ├─ p < 0.05: Significant (*)
   └─ p ≥ 0.05: Not significant (ns)

2. Check effect size (Cohen's d)
   ├─ |d| ≥ 0.8: Large practical importance
   ├─ |d| ≥ 0.5: Medium practical importance
   ├─ |d| ≥ 0.2: Small practical importance
   └─ |d| < 0.2: Negligible practical importance

3. Check confidence intervals
   ├─ CI excludes baseline: Evidence of difference
   └─ CI includes baseline: Insufficient evidence

4. Report findings
   → "Metric X (mean=Y, 95% CI=[L, U]) significantly exceeded chance
      (p < 0.01) with large effect size (d=Z)"
```

### Common Scenarios

#### Scenario 1: Significant with Large Effect Size ✓✓
```
overall_accuracy: mean=0.73, p=0.001, d=2.1 (large), 95% CI=[0.70, 0.76]
```
**Interpretation:** Strong evidence of effect. Model clearly outperforms chance with large practical significance. CI excludes chance level (0.52).

**Report as:** "Overall accuracy (M = 0.73, 95% CI [0.70, 0.76]) significantly exceeded chance level of 52%, p < 0.001, Cohen's d = 2.1 (large effect)."

---

#### Scenario 2: Significant with Small Effect Size ✓
```
opto_EB_vs_opto_hex: mean_diff=0.03, p=0.04, d=0.25 (small), 95% CI=[0.001, 0.06]
```
**Interpretation:** Statistically significant but small practical difference. May not be biologically meaningful.

**Report as:** "Small but significant difference between conditions (Δ = 0.03, 95% CI [0.001, 0.06]), p = 0.04, Cohen's d = 0.25. Practical significance unclear."

---

#### Scenario 3: Not Significant with Large Effect Size
```
trained_odor_accuracy: mean=0.78, p=0.08, d=0.85 (large), 95% CI=[0.68, 0.88]
```
**Interpretation:** Likely underpowered. Large effect size suggests real difference, but small sample prevents significance.

**Report as:** "Trained odor accuracy showed large effect (d = 0.85) but did not reach statistical significance (p = 0.08), likely due to small sample size (n = 5 folds). Further validation recommended."

---

#### Scenario 4: Not Significant with Negligible Effect Size
```
opto_EB_vs_opto_benz_1: mean_diff=0.01, p=0.72, d=0.05 (negligible), 95% CI=[-0.05, 0.07]
```
**Interpretation:** No evidence of difference, and effect size confirms negligible practical difference.

**Report as:** "No significant difference between conditions (p = 0.72), with negligible effect size (d = 0.05)."

---

## Week 12+ Integration

### Overview

The statistical reporting format is designed to be **extensible** for future analyses:
- **Week 12:** Feature importance analysis, ablation studies
- **Week 16:** Neural data integration, calcium imaging correlations

This section provides templates and hooks for integrating new analyses.

---

### Template 1: Feature Importance P-Values

**Goal:** Test if each chemical feature (molecular weight, functional groups, etc.) significantly predicts generalization.

**Implementation:**

1. **Compute feature importance** (e.g., via permutation importance, SHAP values)
2. **Run permutation test** to test if importance > 0
3. **Add to statistical report** under new key `"feature_importance"`

**Code Hook:**

```python
# In statistical_tests.py, add new function:

def test_feature_importance(
    feature_importances: Dict[str, np.ndarray],
    n_permutations: int = 5000
) -> Dict[str, Dict[str, float]]:
    """
    Test if feature importances are significantly greater than zero.

    Args:
        feature_importances: Dict mapping feature name to importance values across folds
        n_permutations: Number of permutation resamples

    Returns:
        Dict with p-values and effect sizes for each feature
    """
    results = {}
    for feature_name, importance_values in feature_importances.items():
        perm_result = permutation_test_vs_chance(
            importance_values,
            chance_level=0.0,  # Test if importance > 0
            n_permutations=n_permutations,
            metric_name=feature_name
        )
        effect_result = compute_effect_sizes(
            importance_values,
            baseline_value=0.0
        )

        results[feature_name] = {
            "mean_importance": perm_result.observed_statistic,
            "p_value": perm_result.p_value_one_tailed,
            "cohens_d": effect_result.cohens_d,
            "significant": perm_result.p_value_one_tailed < 0.05
        }

    return results
```

**Output Format:**

```json
{
  "feature_importance": {
    "molecular_weight": {
      "mean_importance": 0.15,
      "p_value": 0.002,
      "cohens_d": 1.8,
      "significant": true
    },
    "functional_groups": {
      "mean_importance": 0.32,
      "p_value": 0.001,
      "cohens_d": 2.5,
      "significant": true
    },
    "carbon_length": {
      "mean_importance": 0.08,
      "p_value": 0.12,
      "cohens_d": 0.6,
      "significant": false
    }
  }
}
```

**Integration into `run_all_statistical_tests()`:**

```python
# In run_all_statistical_tests(), add parameter:
feature_importance_data: Optional[Dict[str, np.ndarray]] = None

# Near end of function, add:
if feature_importance_data is not None:
    report["feature_importance"] = test_feature_importance(
        feature_importance_data,
        n_permutations=n_permutations
    )
```

---

### Template 2: Ablation Study Statistical Tests

**Goal:** Test if removing a model component (e.g., chemical similarity, plasticity) significantly degrades performance.

**Implementation:**

1. **Run ablation experiments** with component removed
2. **Compare full model vs ablated model** using permutation test
3. **Add to statistical report** under new key `"ablation_tests"`

**Code Hook:**

```python
# In statistical_tests.py, add new function:

def test_ablation_effect(
    full_model_performance: np.ndarray,
    ablated_model_performance: np.ndarray,
    component_name: str,
    n_permutations: int = 5000
) -> Dict[str, float]:
    """
    Test if removing a component significantly degrades performance.

    Args:
        full_model_performance: Metric values for full model
        ablated_model_performance: Metric values for ablated model
        component_name: Name of ablated component
        n_permutations: Number of permutation resamples

    Returns:
        Dict with performance degradation statistics
    """
    perm_result = permutation_test_between_conditions(
        full_model_performance,
        ablated_model_performance,
        condition_a_name="full_model",
        condition_b_name=f"ablated_{component_name}",
        n_permutations=n_permutations
    )

    effect_result = compute_effect_sizes(
        full_model_performance,
        condition_b_values=ablated_model_performance
    )

    performance_drop = np.mean(full_model_performance) - np.mean(ablated_model_performance)

    return {
        "component_ablated": component_name,
        "full_model_mean": np.mean(full_model_performance),
        "ablated_model_mean": np.mean(ablated_model_performance),
        "performance_drop": performance_drop,
        "p_value": perm_result.p_value_one_tailed,  # One-tailed: full > ablated
        "cohens_d": effect_result.cohens_d,
        "significant": perm_result.p_value_one_tailed < 0.05,
        "interpretation": (
            "Component necessary" if perm_result.p_value_one_tailed < 0.05
            else "Component not necessary"
        )
    }
```

**Output Format:**

```json
{
  "ablation_tests": {
    "chemical_similarity": {
      "component_ablated": "chemical_similarity",
      "full_model_mean": 0.73,
      "ablated_model_mean": 0.65,
      "performance_drop": 0.08,
      "p_value": 0.003,
      "cohens_d": 1.2,
      "significant": true,
      "interpretation": "Component necessary"
    },
    "plasticity_rules": {
      "component_ablated": "plasticity_rules",
      "full_model_mean": 0.73,
      "ablated_model_mean": 0.58,
      "performance_drop": 0.15,
      "p_value": 0.001,
      "cohens_d": 2.1,
      "significant": true,
      "interpretation": "Component necessary"
    },
    "kc_sparsity": {
      "component_ablated": "kc_sparsity",
      "full_model_mean": 0.73,
      "ablated_model_mean": 0.71,
      "performance_drop": 0.02,
      "p_value": 0.32,
      "cohens_d": 0.3,
      "significant": false,
      "interpretation": "Component not necessary"
    }
  }
}
```

**Usage Example:**

```python
# In cross_validation.py or separate ablation script:

# Run ablation experiments
full_model_results = run_cross_validation(full_model)
ablated_chem_results = run_cross_validation(model_without_chemical_similarity)
ablated_plasticity_results = run_cross_validation(model_without_plasticity)

# Test ablation effects
ablation_data = {
    "chemical_similarity": {
        "full": full_model_results,
        "ablated": ablated_chem_results
    },
    "plasticity_rules": {
        "full": full_model_results,
        "ablated": ablated_plasticity_results
    }
}

# Add to statistical report
statistical_report["ablation_tests"] = {}
for component, data in ablation_data.items():
    statistical_report["ablation_tests"][component] = test_ablation_effect(
        data["full"],
        data["ablated"],
        component_name=component
    )
```

---

### Template 3: Time-Series Analysis (Week 16)

**Goal:** Test if learning curves differ between conditions, or if performance improves over training epochs.

**Implementation:**

1. **Collect per-epoch metrics** during training
2. **Fit learning curves** and test slope parameters
3. **Compare convergence rates** between conditions

**Code Hook:**

```python
# In statistical_tests.py, add new function:

def test_learning_curve_slope(
    epochs: np.ndarray,
    performance: np.ndarray,
    n_bootstrap_samples: int = 5000
) -> Dict[str, float]:
    """
    Test if learning curve has positive slope (performance improves over time).

    Args:
        epochs: Array of epoch indices
        performance: Array of performance values at each epoch
        n_bootstrap_samples: Number of bootstrap resamples

    Returns:
        Dict with slope estimate, CI, and significance
    """
    from scipy.stats import linregress

    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = linregress(epochs, performance)

    # Bootstrap CI for slope
    bootstrap_slopes = []
    for _ in range(n_bootstrap_samples):
        indices = np.random.choice(len(epochs), size=len(epochs), replace=True)
        boot_slope = linregress(epochs[indices], performance[indices])[0]
        bootstrap_slopes.append(boot_slope)

    ci_95 = [np.percentile(bootstrap_slopes, 2.5), np.percentile(bootstrap_slopes, 97.5)]

    return {
        "slope": slope,
        "p_value": p_value,
        "r_squared": r_value**2,
        "ci_95": ci_95,
        "significant": p_value < 0.05,
        "interpretation": (
            "Performance improves over training" if slope > 0 and p_value < 0.05
            else "No significant learning trend"
        )
    }
```

**Output Format:**

```json
{
  "learning_curve_analysis": {
    "opto_EB": {
      "slope": 0.003,
      "p_value": 0.001,
      "r_squared": 0.85,
      "ci_95": [0.002, 0.004],
      "significant": true,
      "interpretation": "Performance improves over training"
    },
    "opto_hex": { ... }
  }
}
```

---

## Examples and Templates

### Example 1: Complete Statistical Report

See: `artifacts/cross_validation/week4_statistical_report.json`

```json
{
  "metadata": {
    "n_folds": 5,
    "n_permutations": 5000,
    "n_bootstrap_samples": 5000,
    "confidence_levels": [0.95, 0.99],
    "chance_level": 0.52
  },
  "permutation_tests": {
    "vs_chance": {
      "overall_accuracy": {
        "observed_mean": 0.73,
        "chance_level": 0.52,
        "p_value_one_tailed": 0.0012,
        "p_value_two_tailed": 0.0024,
        "significant_at_0.05": true,
        "significant_at_0.01": true
      }
    }
  },
  "confidence_intervals": {
    "overall_accuracy": {
      "mean": 0.73,
      "ci_95": [0.70, 0.76],
      "ci_99": [0.68, 0.78]
    }
  },
  "effect_sizes": {
    "vs_chance": {
      "overall_accuracy": {
        "cohens_d": 2.1,
        "interpretation": "large"
      }
    }
  }
}
```

---

### Example 2: Python Script to Load and Analyze Report

```python
import json
from pathlib import Path

# Load statistical report
report_path = Path("artifacts/cross_validation/week4_statistical_report.json")
with report_path.open("r") as f:
    report = json.load(f)

# Check if overall accuracy beats chance
vs_chance = report["permutation_tests"]["vs_chance"]["overall_accuracy"]
if vs_chance["significant_at_0.01"]:
    print(f"Overall accuracy ({vs_chance['observed_mean']:.3f}) "
          f"significantly exceeds chance (p < 0.01)")

# Extract confidence intervals
ci = report["confidence_intervals"]["overall_accuracy"]
print(f"95% CI: [{ci['ci_95'][0]:.3f}, {ci['ci_95'][1]:.3f}]")

# Check effect size
effect = report["effect_sizes"]["vs_chance"]["overall_accuracy"]
print(f"Cohen's d: {effect['cohens_d']:.2f} ({effect['interpretation']})")
```

---

### Example 3: Manuscript Reporting Template

**Results Section:**

> Cross-validation yielded overall accuracy of 73% (95% CI [70%, 76%]), significantly exceeding chance level of 52% (permutation test, p < 0.001, Cohen's d = 2.1, large effect). Trained odor recognition accuracy was 82% (95% CI [78%, 86%]), also significantly above chance (p < 0.001, Cohen's d = 2.8). Control separation was 91% (95% CI [88%, 94%]), demonstrating robust learning specificity.
>
> Chemical similarity significantly predicted behavioral generalization (Pearson r = 0.45, 95% CI [0.25, 0.65], p = 0.003), explaining 20% of variance. Between-condition comparisons revealed no significant difference between ethyl butyrate (opto_EB) and hexanol (opto_hex) training (Δ = 0.05, p = 0.24, Cohen's d = 0.42), but benzaldehyde training (opto_benz_1) yielded significantly lower performance (vs. opto_EB: Δ = 0.18, p = 0.003, Cohen's d = 1.5).

---

## Best Practices

### 1. Always Report Both P-Values and Effect Sizes

❌ **Bad:** "Overall accuracy was significantly above chance (p < 0.01)."

✅ **Good:** "Overall accuracy (M = 0.73, 95% CI [0.70, 0.76]) significantly exceeded chance (p < 0.01, Cohen's d = 2.1, large effect)."

**Why:** P-values indicate significance, but effect sizes quantify magnitude. Both are essential.

---

### 2. Use Confidence Intervals to Assess Uncertainty

❌ **Bad:** "Mean accuracy was 73%."

✅ **Good:** "Mean accuracy was 73% (95% CI [70%, 76%])."

**Why:** CIs communicate precision and allow readers to assess if performance meets targets.

---

### 3. Check Assumptions Before Interpretation

- **Permutation tests:** No assumptions (non-parametric)
- **Bootstrap CIs:** Requires independent observations (folds should be independent)
- **Correlations:** Pearson r assumes linearity; use Spearman ρ if non-linear

---

### 4. Correct for Multiple Comparisons (When Appropriate)

If testing many hypotheses (e.g., 10 features), apply Bonferroni correction:
- Adjusted α = 0.05 / 10 = 0.005

**When to correct:**
- Multiple features tested simultaneously
- Many between-condition comparisons

**When NOT to correct:**
- Primary hypotheses specified a priori
- Exploratory analyses (report uncorrected, note as exploratory)

---

### 5. Pre-Register Analyses When Possible

- Specify primary outcomes before running experiments
- Distinguish confirmatory vs. exploratory analyses
- Reduces risk of p-hacking and HARKing (Hypothesizing After Results are Known)

---

### 6. Save Random Seeds for Reproducibility

```python
statistical_report = run_all_statistical_tests(
    fold_results=fold_results,
    random_seed=42  # Always set for reproducibility
)
```

---

### 7. Document Deviations from Pre-Specified Analyses

If you deviate from AGENTS.md specifications:
- Document in report metadata
- Justify the change
- Report sensitivity analyses

---

## Frequently Asked Questions

### Q1: What if my p-value is just above 0.05 (e.g., p = 0.06)?

**A:** Report the exact p-value and effect size. Don't treat p = 0.05 as a hard threshold. Consider:
- Effect size (large effect with p = 0.06 may still be meaningful)
- Confidence intervals (if CI barely includes null, you're borderline)
- Sample size (underpowered studies may have p > 0.05 despite real effects)

---

### Q2: Should I use one-tailed or two-tailed tests?

**A:**
- **One-tailed:** Use when you have a strong directional hypothesis (e.g., model > chance)
- **Two-tailed:** Use for exploratory comparisons or when direction is uncertain
- **Report both:** The statistical_tests module provides both; choose based on hypothesis

---

### Q3: What if fold results are not independent (e.g., overlapping flies)?

**A:** This violates the independence assumption for bootstrap CIs. Solutions:
- Use GroupKFold to ensure independence (already implemented in cross_validation.py)
- Use hierarchical bootstrap if nesting is unavoidable
- Report non-independence as a limitation

---

### Q4: How do I extend the format for new analyses?

**A:** Follow the template structure:
1. Add new function to `statistical_tests.py`
2. Integrate into `run_all_statistical_tests()` via optional parameter
3. Add new top-level key to report dict (e.g., `"feature_importance"`)
4. Document in this file under [Week 12+ Integration](#week-12-integration)

---

### Q5: Can I re-run statistical tests without re-training?

**A:** Yes! Use the standalone script:

```bash
python analysis/run_statistical_tests.py --artifacts-dir artifacts/cross_validation
```

This loads existing fold results and re-runs statistical tests with updated parameters.

---

## References

### Statistical Methods

- **Permutation Tests:** Good, P. (2013). *Permutation Tests: A Practical Guide to Resampling Methods*. Springer.
- **Bootstrap CIs:** Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- **Effect Sizes:** Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Erlbaum.

### Best Practices

- **New Statistics:** Cumming, G. (2014). The new statistics: Why and how. *Psychological Science*, 25(1), 7-29.
- **P-value interpretation:** Wasserstein, R. L. & Lazar, N. A. (2016). The ASA statement on p-values. *The American Statistician*, 70(2), 129-133.

### Neuroscience Context

- **Drosophila Learning:** Eschbach et al. (2021). Recurrent architecture for adaptive regulation of learning in the insect brain. *Nature Neuroscience*, 23, 544-555.
- **Chemical Similarity:** Dolan et al. (2019). Communication from learned to innate olfactory processing centers is required for memory retrieval in *Drosophila*. *Neuron*, 100(3), 651-668.

---

**Document Version History:**
- v1.0 (2025-10-29): Initial version with Week 4-12 specifications

---

*For questions or suggestions, please open an issue in the repository.*
