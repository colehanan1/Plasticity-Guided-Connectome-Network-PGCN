"""
Statistical Testing Module for Drosophila Cross-Odor Generalization Model

Provides permutation tests, bootstrap confidence intervals, correlation analysis,
and effect size calculations for model validation and scientific reporting.

This module implements:
1. Permutation tests against chance level (52% baseline)
2. Permutation tests for between-condition comparisons
3. Bootstrap confidence intervals (95% and 99%)
4. Chemical similarity correlation analysis
5. Effect size calculations (Cohen's d, eta-squared)

Usage:
    from analysis.statistical_tests import run_all_statistical_tests

    # After cross-validation
    statistical_report = run_all_statistical_tests(
        fold_results=cv_results,
        predictions_data=predictions,
        chemical_similarity_matrix=chem_sim_matrix,
        n_permutations=5000,
        n_bootstrap_samples=5000
    )

Mathematical Background:
    - Permutation test: Non-parametric hypothesis testing via resampling
    - Bootstrap CI: Empirical confidence intervals via resampling with replacement
    - Pearson r: Linear correlation coefficient
    - Spearman ρ: Rank-based correlation coefficient
    - Cohen's d: Standardized mean difference effect size
    - Eta-squared: Proportion of variance explained (ANOVA effect size)

References:
    - Good, P. (2013). Permutation Tests: A Practical Guide to Resampling Methods
    - Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap
    - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy import stats
from scipy.stats import bootstrap
import warnings
from dataclasses import dataclass


# Constants for interpretation
COHEN_D_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}
ETA_SQUARED_THRESHOLDS = {"small": 0.01, "medium": 0.06, "large": 0.14}
ALPHA_05 = 0.05
ALPHA_01 = 0.01


@dataclass
class PermutationTestResult:
    """Results from a permutation test."""
    observed_statistic: float
    p_value_one_tailed: float
    p_value_two_tailed: float
    null_distribution: np.ndarray
    n_permutations: int
    test_description: str


@dataclass
class BootstrapCIResult:
    """Results from bootstrap confidence interval estimation."""
    mean: float
    ci_95_lower: float
    ci_95_upper: float
    ci_99_lower: float
    ci_99_upper: float
    n_bootstrap_samples: int


@dataclass
class CorrelationResult:
    """Results from correlation analysis."""
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    n_samples: int
    ci_95_lower: float
    ci_95_upper: float
    ci_99_lower: float
    ci_99_upper: float


@dataclass
class EffectSizeResult:
    """Results from effect size calculations."""
    cohens_d: float
    cohens_d_interpretation: str
    eta_squared: Optional[float] = None
    eta_squared_interpretation: Optional[str] = None


def permutation_test_vs_chance(
    observed_values: np.ndarray,
    chance_level: float = 0.52,
    n_permutations: int = 5000,
    metric_name: str = "metric",
    random_seed: Optional[int] = None
) -> PermutationTestResult:
    """
    Test if observed metric significantly exceeds chance level using permutation test.

    This test evaluates whether the observed performance (e.g., accuracy, AUROC)
    is significantly different from a baseline chance level (52% for behavioral data).

    Null Hypothesis (H0): The observed metric equals the chance level
    Alternative Hypothesis (H1):
        - One-tailed: metric > chance_level
        - Two-tailed: metric ≠ chance_level

    Method:
        1. Compute observed mean statistic
        2. Generate null distribution by:
           - Centering data around chance level
           - Randomly permuting centered values
           - Computing mean for each permutation
        3. Calculate p-values as proportion of permuted statistics
           as or more extreme than observed

    Args:
        observed_values: Array of observed metric values (e.g., per-fold accuracies)
        chance_level: Baseline chance performance (default: 0.52 for 52% reaction rate)
        n_permutations: Number of permutation resamples (default: 5000)
        metric_name: Name of metric for reporting (default: "metric")
        random_seed: Random seed for reproducibility (optional)

    Returns:
        PermutationTestResult containing:
            - observed_statistic: Mean of observed values
            - p_value_one_tailed: P-value for H1: metric > chance
            - p_value_two_tailed: P-value for H1: metric ≠ chance
            - null_distribution: Array of permuted statistics
            - n_permutations: Number of permutations performed
            - test_description: Human-readable test description

    Example:
        >>> fold_accuracies = np.array([0.72, 0.75, 0.71, 0.74, 0.73])
        >>> result = permutation_test_vs_chance(
        ...     fold_accuracies,
        ...     chance_level=0.52,
        ...     metric_name="overall_accuracy"
        ... )
        >>> print(f"p-value (one-tailed): {result.p_value_one_tailed:.4f}")
        p-value (one-tailed): 0.0002
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    observed_values = np.asarray(observed_values)
    observed_mean = np.mean(observed_values)

    # Center observations around chance level for permutation
    centered_values = observed_values - observed_mean + chance_level

    # Generate null distribution
    null_distribution = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Permute signs (equivalent to permuting under null)
        permuted_values = centered_values * np.random.choice([-1, 1], size=len(centered_values))
        null_distribution[i] = np.mean(permuted_values)

    # Calculate p-values
    # One-tailed: proportion of null >= observed (testing if observed > chance)
    p_one_tailed = np.mean(null_distribution >= observed_mean)

    # Two-tailed: proportion of null as or more extreme than observed
    p_two_tailed = np.mean(np.abs(null_distribution - chance_level) >= np.abs(observed_mean - chance_level))

    test_description = (
        f"{metric_name} vs chance level ({chance_level:.2f}): "
        f"observed={observed_mean:.4f}, p_one={p_one_tailed:.4f}, p_two={p_two_tailed:.4f}"
    )

    return PermutationTestResult(
        observed_statistic=observed_mean,
        p_value_one_tailed=p_one_tailed,
        p_value_two_tailed=p_two_tailed,
        null_distribution=null_distribution,
        n_permutations=n_permutations,
        test_description=test_description
    )


def permutation_test_between_conditions(
    condition_a_values: np.ndarray,
    condition_b_values: np.ndarray,
    condition_a_name: str = "condition_A",
    condition_b_name: str = "condition_B",
    n_permutations: int = 5000,
    random_seed: Optional[int] = None
) -> PermutationTestResult:
    """
    Test if two conditions differ significantly using permutation test.

    This test evaluates whether performance differs between training conditions
    (e.g., opto_EB vs opto_hex).

    Null Hypothesis (H0): The two conditions have the same distribution
    Alternative Hypothesis (H1):
        - One-tailed: condition_A > condition_B
        - Two-tailed: condition_A ≠ condition_B

    Method:
        1. Compute observed mean difference
        2. Generate null distribution by:
           - Pooling all values
           - Randomly permuting condition labels
           - Computing mean difference for each permutation
        3. Calculate p-values as proportion of permuted differences
           as or more extreme than observed

    Args:
        condition_a_values: Array of metric values for condition A
        condition_b_values: Array of metric values for condition B
        condition_a_name: Name of condition A (for reporting)
        condition_b_name: Name of condition B (for reporting)
        n_permutations: Number of permutation resamples (default: 5000)
        random_seed: Random seed for reproducibility (optional)

    Returns:
        PermutationTestResult containing difference statistics and p-values

    Example:
        >>> opto_eb_acc = np.array([0.75, 0.78, 0.76])
        >>> opto_hex_acc = np.array([0.70, 0.72, 0.71])
        >>> result = permutation_test_between_conditions(
        ...     opto_eb_acc,
        ...     opto_hex_acc,
        ...     condition_a_name="opto_EB",
        ...     condition_b_name="opto_hex"
        ... )
        >>> print(f"Mean difference: {result.observed_statistic:.4f}")
        Mean difference: 0.0500
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    condition_a_values = np.asarray(condition_a_values)
    condition_b_values = np.asarray(condition_b_values)

    observed_diff = np.mean(condition_a_values) - np.mean(condition_b_values)

    # Pool all values
    pooled_values = np.concatenate([condition_a_values, condition_b_values])
    n_a = len(condition_a_values)
    n_total = len(pooled_values)

    # Generate null distribution
    null_distribution = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Randomly permute condition labels
        permuted_indices = np.random.permutation(n_total)
        permuted_a = pooled_values[permuted_indices[:n_a]]
        permuted_b = pooled_values[permuted_indices[n_a:]]
        null_distribution[i] = np.mean(permuted_a) - np.mean(permuted_b)

    # Calculate p-values
    # One-tailed: proportion of null >= observed (testing if A > B)
    p_one_tailed = np.mean(null_distribution >= observed_diff)

    # Two-tailed: proportion of null as or more extreme than observed
    p_two_tailed = np.mean(np.abs(null_distribution) >= np.abs(observed_diff))

    test_description = (
        f"{condition_a_name} vs {condition_b_name}: "
        f"mean_diff={observed_diff:.4f}, p_one={p_one_tailed:.4f}, p_two={p_two_tailed:.4f}"
    )

    return PermutationTestResult(
        observed_statistic=observed_diff,
        p_value_one_tailed=p_one_tailed,
        p_value_two_tailed=p_two_tailed,
        null_distribution=null_distribution,
        n_permutations=n_permutations,
        test_description=test_description
    )


def bootstrap_ci(
    observed_values: np.ndarray,
    confidence_levels: Tuple[float, float] = (0.95, 0.99),
    n_bootstrap_samples: int = 5000,
    random_seed: Optional[int] = None
) -> BootstrapCIResult:
    """
    Compute bootstrap confidence intervals for a metric.

    Bootstrap resampling provides empirical confidence intervals without
    assuming a specific parametric distribution. This is particularly useful
    for metrics like accuracy where the underlying distribution may be non-normal.

    Method:
        1. Resample observed values with replacement
        2. Compute mean for each resample
        3. Use percentile method to extract confidence intervals

    Args:
        observed_values: Array of observed metric values
        confidence_levels: Tuple of confidence levels (default: (0.95, 0.99))
        n_bootstrap_samples: Number of bootstrap resamples (default: 5000)
        random_seed: Random seed for reproducibility (optional)

    Returns:
        BootstrapCIResult containing mean and confidence interval bounds

    Example:
        >>> fold_accuracies = np.array([0.72, 0.75, 0.71, 0.74, 0.73])
        >>> result = bootstrap_ci(fold_accuracies)
        >>> print(f"Mean: {result.mean:.4f}, 95% CI: [{result.ci_95_lower:.4f}, {result.ci_95_upper:.4f}]")
        Mean: 0.7300, 95% CI: [0.7120, 0.7480]
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    observed_values = np.asarray(observed_values)
    mean_val = np.mean(observed_values)

    # Generate bootstrap resamples
    bootstrap_means = np.zeros(n_bootstrap_samples)
    for i in range(n_bootstrap_samples):
        resample = np.random.choice(observed_values, size=len(observed_values), replace=True)
        bootstrap_means[i] = np.mean(resample)

    # Calculate confidence intervals using percentile method
    ci_95_lower = np.percentile(bootstrap_means, (1 - confidence_levels[0]) / 2 * 100)
    ci_95_upper = np.percentile(bootstrap_means, (1 + confidence_levels[0]) / 2 * 100)

    ci_99_lower = np.percentile(bootstrap_means, (1 - confidence_levels[1]) / 2 * 100)
    ci_99_upper = np.percentile(bootstrap_means, (1 + confidence_levels[1]) / 2 * 100)

    return BootstrapCIResult(
        mean=mean_val,
        ci_95_lower=ci_95_lower,
        ci_95_upper=ci_95_upper,
        ci_99_lower=ci_99_lower,
        ci_99_upper=ci_99_upper,
        n_bootstrap_samples=n_bootstrap_samples
    )


def compute_chemical_similarity_correlation(
    chemical_similarities: np.ndarray,
    behavioral_responses: np.ndarray,
    response_type: str = "response_rate",
    n_bootstrap_samples: int = 5000,
    random_seed: Optional[int] = None
) -> CorrelationResult:
    """
    Compute correlation between chemical similarity and behavioral responses.

    This analysis tests whether chemically similar odors elicit similar behavioral
    responses, a key prediction of the model. Both parametric (Pearson) and
    non-parametric (Spearman) correlations are computed, along with bootstrap CIs.

    Args:
        chemical_similarities: Array of pairwise chemical similarity values (0-1)
        behavioral_responses: Array of corresponding behavioral responses
            (response rates or prediction probabilities)
        response_type: Type of behavioral response ("response_rate" or "prediction_probability")
        n_bootstrap_samples: Number of bootstrap samples for CI (default: 5000)
        random_seed: Random seed for reproducibility (optional)

    Returns:
        CorrelationResult containing correlation coefficients, p-values, and CIs

    Example:
        >>> chem_sim = np.array([0.8, 0.6, 0.3, 0.7, 0.4])
        >>> response_rate = np.array([0.75, 0.65, 0.45, 0.70, 0.50])
        >>> result = compute_chemical_similarity_correlation(chem_sim, response_rate)
        >>> print(f"Pearson r: {result.pearson_r:.4f}, p: {result.pearson_p:.4f}")
        Pearson r: 0.9850, p: 0.0015
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    chemical_similarities = np.asarray(chemical_similarities)
    behavioral_responses = np.asarray(behavioral_responses)

    # Remove any NaN values
    valid_mask = ~(np.isnan(chemical_similarities) | np.isnan(behavioral_responses))
    chem_sim_clean = chemical_similarities[valid_mask]
    behavior_clean = behavioral_responses[valid_mask]

    if len(chem_sim_clean) < 3:
        warnings.warn(
            f"Insufficient valid samples for correlation ({len(chem_sim_clean)} < 3). "
            "Returning NaN values."
        )
        return CorrelationResult(
            pearson_r=np.nan,
            pearson_p=np.nan,
            spearman_rho=np.nan,
            spearman_p=np.nan,
            n_samples=len(chem_sim_clean),
            ci_95_lower=np.nan,
            ci_95_upper=np.nan,
            ci_99_lower=np.nan,
            ci_99_upper=np.nan
        )

    # Compute Pearson correlation (linear relationship)
    pearson_r, pearson_p = stats.pearsonr(chem_sim_clean, behavior_clean)

    # Compute Spearman correlation (monotonic relationship)
    spearman_rho, spearman_p = stats.spearmanr(chem_sim_clean, behavior_clean)

    # Bootstrap confidence intervals for Pearson r
    bootstrap_correlations = np.zeros(n_bootstrap_samples)
    for i in range(n_bootstrap_samples):
        indices = np.random.choice(len(chem_sim_clean), size=len(chem_sim_clean), replace=True)
        bootstrap_correlations[i] = stats.pearsonr(
            chem_sim_clean[indices],
            behavior_clean[indices]
        )[0]

    ci_95_lower = np.percentile(bootstrap_correlations, 2.5)
    ci_95_upper = np.percentile(bootstrap_correlations, 97.5)
    ci_99_lower = np.percentile(bootstrap_correlations, 0.5)
    ci_99_upper = np.percentile(bootstrap_correlations, 99.5)

    return CorrelationResult(
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        n_samples=len(chem_sim_clean),
        ci_95_lower=ci_95_lower,
        ci_95_upper=ci_95_upper,
        ci_99_lower=ci_99_lower,
        ci_99_upper=ci_99_upper
    )


def compute_effect_sizes(
    condition_a_values: np.ndarray,
    condition_b_values: Optional[np.ndarray] = None,
    baseline_value: Optional[float] = None
) -> EffectSizeResult:
    """
    Compute effect sizes (Cohen's d and eta-squared) for group comparisons.

    Effect sizes quantify the magnitude of differences, complementing p-values
    which only indicate statistical significance. Effect sizes are essential for:
    - Assessing practical significance (beyond statistical significance)
    - Comparing results across studies
    - Power analysis for future experiments

    Cohen's d interpretation:
        - Small: |d| = 0.2
        - Medium: |d| = 0.5
        - Large: |d| = 0.8

    Eta-squared interpretation (proportion of variance explained):
        - Small: η² = 0.01 (1%)
        - Medium: η² = 0.06 (6%)
        - Large: η² = 0.14 (14%)

    Args:
        condition_a_values: Array of metric values for condition A (or single condition)
        condition_b_values: Array of metric values for condition B (optional, for two-group comparison)
        baseline_value: Baseline value for one-sample comparison (e.g., chance level 0.52)

    Returns:
        EffectSizeResult containing Cohen's d and eta-squared (if applicable)

    Example:
        >>> # One-sample comparison (vs chance)
        >>> accuracies = np.array([0.72, 0.75, 0.71, 0.74, 0.73])
        >>> result = compute_effect_sizes(accuracies, baseline_value=0.52)
        >>> print(f"Cohen's d: {result.cohens_d:.4f} ({result.cohens_d_interpretation})")
        Cohen's d: 3.1250 (large)

        >>> # Two-sample comparison
        >>> opto_eb = np.array([0.75, 0.78, 0.76])
        >>> opto_hex = np.array([0.70, 0.72, 0.71])
        >>> result = compute_effect_sizes(opto_eb, condition_b_values=opto_hex)
        >>> print(f"Cohen's d: {result.cohens_d:.4f}")
        Cohen's d: 2.1909
    """
    condition_a_values = np.asarray(condition_a_values)

    # Two-sample Cohen's d (between conditions)
    if condition_b_values is not None:
        condition_b_values = np.asarray(condition_b_values)

        mean_a = np.mean(condition_a_values)
        mean_b = np.mean(condition_b_values)
        std_a = np.std(condition_a_values, ddof=1)
        std_b = np.std(condition_b_values, ddof=1)

        # Pooled standard deviation
        n_a = len(condition_a_values)
        n_b = len(condition_b_values)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

        cohens_d = (mean_a - mean_b) / pooled_std

        # Eta-squared (proportion of variance explained)
        # For two groups: η² = d² / (d² + (n_a + n_b) / (n_a * n_b))
        eta_squared = cohens_d**2 / (cohens_d**2 + (n_a + n_b) / (n_a * n_b))

    # One-sample Cohen's d (vs baseline)
    elif baseline_value is not None:
        mean_a = np.mean(condition_a_values)
        std_a = np.std(condition_a_values, ddof=1)

        # Handle zero or near-zero standard deviation
        if std_a < 1e-10:
            cohens_d = np.inf if mean_a != baseline_value else 0.0
        else:
            cohens_d = (mean_a - baseline_value) / std_a
        eta_squared = None  # Not applicable for one-sample

    else:
        raise ValueError("Must provide either condition_b_values or baseline_value")

    # Interpret Cohen's d
    abs_d = abs(cohens_d)
    if abs_d >= COHEN_D_THRESHOLDS["large"]:
        d_interpretation = "large"
    elif abs_d >= COHEN_D_THRESHOLDS["medium"]:
        d_interpretation = "medium"
    elif abs_d >= COHEN_D_THRESHOLDS["small"]:
        d_interpretation = "small"
    else:
        d_interpretation = "negligible"

    # Interpret eta-squared
    eta_interpretation = None
    if eta_squared is not None:
        if eta_squared >= ETA_SQUARED_THRESHOLDS["large"]:
            eta_interpretation = "large"
        elif eta_squared >= ETA_SQUARED_THRESHOLDS["medium"]:
            eta_interpretation = "medium"
        elif eta_squared >= ETA_SQUARED_THRESHOLDS["small"]:
            eta_interpretation = "small"
        else:
            eta_interpretation = "negligible"

    return EffectSizeResult(
        cohens_d=cohens_d,
        cohens_d_interpretation=d_interpretation,
        eta_squared=eta_squared,
        eta_squared_interpretation=eta_interpretation
    )


def run_all_statistical_tests(
    fold_results: List[Dict[str, Any]],
    chance_level: float = 0.52,
    n_permutations: int = 5000,
    n_bootstrap_samples: int = 5000,
    chemical_similarity_data: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run comprehensive statistical analysis on cross-validation results.

    This is the main orchestration function that runs all statistical tests:
    1. Permutation tests vs chance level for each metric
    2. Permutation tests between training conditions
    3. Bootstrap confidence intervals for all metrics
    4. Chemical similarity correlation analysis (if data provided)
    5. Effect sizes for all comparisons

    Args:
        fold_results: List of per-fold result dictionaries with keys:
            - "metrics": Dict with "overall_accuracy", "trained_odor_accuracy",
              "auroc", "control_separation"
            - "dataset": Training condition (e.g., "opto_EB", "opto_hex")
        chance_level: Baseline chance performance (default: 0.52)
        n_permutations: Number of permutation resamples (default: 5000)
        n_bootstrap_samples: Number of bootstrap resamples (default: 5000)
        chemical_similarity_data: Optional dict with:
            - "similarities": Array of pairwise chemical similarities
            - "response_rates": Array of behavioral response rates
            - "prediction_probabilities": Array of model prediction probabilities
        random_seed: Random seed for reproducibility (optional)

    Returns:
        Comprehensive statistical report dictionary with structure:
        {
            "metadata": {...},
            "permutation_tests": {
                "vs_chance": {...},
                "between_conditions": {...}
            },
            "confidence_intervals": {...},
            "correlations": {...},
            "effect_sizes": {...}
        }

    Example:
        >>> cv_results = [
        ...     {"metrics": {"overall_accuracy": 0.72, ...}, "dataset": "opto_EB"},
        ...     {"metrics": {"overall_accuracy": 0.75, ...}, "dataset": "opto_hex"},
        ...     # ... more folds
        ... ]
        >>> report = run_all_statistical_tests(cv_results)
        >>> print(report["permutation_tests"]["vs_chance"]["overall_accuracy"]["p_value_one_tailed"])
        0.0012
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Extract metrics by fold
    metrics_by_fold = {}
    metrics_by_condition = {}

    for fold_result in fold_results:
        metrics = fold_result["metrics"]
        dataset = fold_result.get("dataset", "unknown")

        # Aggregate across all folds (skip None values)
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:  # Skip None values
                if metric_name not in metrics_by_fold:
                    metrics_by_fold[metric_name] = []
                metrics_by_fold[metric_name].append(metric_value)

        # Organize by condition (skip None values)
        if dataset not in metrics_by_condition:
            metrics_by_condition[dataset] = {}
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:  # Skip None values
                if metric_name not in metrics_by_condition[dataset]:
                    metrics_by_condition[dataset][metric_name] = []
                metrics_by_condition[dataset][metric_name].append(metric_value)

    # Initialize report structure
    report = {
        "metadata": {
            "n_folds": len(fold_results),
            "n_permutations": n_permutations,
            "n_bootstrap_samples": n_bootstrap_samples,
            "confidence_levels": [0.95, 0.99],
            "chance_level": chance_level
        },
        "permutation_tests": {
            "vs_chance": {},
            "between_conditions": {}
        },
        "confidence_intervals": {},
        "correlations": {},
        "effect_sizes": {
            "vs_chance": {},
            "between_conditions": {}
        }
    }

    # 1. Permutation tests vs chance level
    for metric_name, metric_values in metrics_by_fold.items():
        if metric_name in ["overall_accuracy", "trained_odor_accuracy", "auroc", "cross_fly_generalization"]:
            # Skip if insufficient data
            if len(metric_values) < 2:
                continue

            perm_result = permutation_test_vs_chance(
                np.array(metric_values),
                chance_level=chance_level,
                n_permutations=n_permutations,
                metric_name=metric_name,
                random_seed=random_seed
            )

            report["permutation_tests"]["vs_chance"][metric_name] = {
                "observed_mean": perm_result.observed_statistic,
                "chance_level": chance_level,
                "p_value_one_tailed": perm_result.p_value_one_tailed,
                "p_value_two_tailed": perm_result.p_value_two_tailed,
                "significant_at_0.05": perm_result.p_value_one_tailed < ALPHA_05,
                "significant_at_0.01": perm_result.p_value_one_tailed < ALPHA_01
            }

            # Effect size vs chance
            effect_result = compute_effect_sizes(
                np.array(metric_values),
                baseline_value=chance_level
            )
            report["effect_sizes"]["vs_chance"][metric_name] = {
                "cohens_d": effect_result.cohens_d,
                "interpretation": effect_result.cohens_d_interpretation
            }

    # 2. Bootstrap confidence intervals
    for metric_name, metric_values in metrics_by_fold.items():
        ci_result = bootstrap_ci(
            np.array(metric_values),
            n_bootstrap_samples=n_bootstrap_samples,
            random_seed=random_seed
        )

        report["confidence_intervals"][metric_name] = {
            "mean": ci_result.mean,
            "ci_95": [ci_result.ci_95_lower, ci_result.ci_95_upper],
            "ci_99": [ci_result.ci_99_lower, ci_result.ci_99_upper]
        }

    # 3. Between-condition permutation tests
    conditions = [c for c in metrics_by_condition.keys() if c != "hex_control"]

    for i, cond_a in enumerate(conditions):
        for cond_b in conditions[i+1:]:
            comparison_key = f"{cond_a}_vs_{cond_b}"
            report["permutation_tests"]["between_conditions"][comparison_key] = {}
            report["effect_sizes"]["between_conditions"][comparison_key] = {}

            for metric_name in ["overall_accuracy", "trained_odor_accuracy", "auroc"]:
                if (metric_name in metrics_by_condition[cond_a] and
                    metric_name in metrics_by_condition[cond_b]):

                    values_a = np.array(metrics_by_condition[cond_a][metric_name])
                    values_b = np.array(metrics_by_condition[cond_b][metric_name])

                    # Permutation test
                    perm_result = permutation_test_between_conditions(
                        values_a,
                        values_b,
                        condition_a_name=cond_a,
                        condition_b_name=cond_b,
                        n_permutations=n_permutations,
                        random_seed=random_seed
                    )

                    report["permutation_tests"]["between_conditions"][comparison_key][metric_name] = {
                        "mean_difference": perm_result.observed_statistic,
                        "p_value_one_tailed": perm_result.p_value_one_tailed,
                        "p_value_two_tailed": perm_result.p_value_two_tailed,
                        "significant_at_0.05": perm_result.p_value_two_tailed < ALPHA_05,
                        "significant_at_0.01": perm_result.p_value_two_tailed < ALPHA_01
                    }

                    # Effect size
                    effect_result = compute_effect_sizes(values_a, condition_b_values=values_b)
                    report["effect_sizes"]["between_conditions"][comparison_key][metric_name] = {
                        "cohens_d": effect_result.cohens_d,
                        "interpretation": effect_result.cohens_d_interpretation,
                        "eta_squared": effect_result.eta_squared,
                        "eta_squared_interpretation": effect_result.eta_squared_interpretation
                    }

    # 4. Chemical similarity correlation analysis
    if chemical_similarity_data is not None:
        # Response rate correlation
        if "response_rates" in chemical_similarity_data:
            corr_result = compute_chemical_similarity_correlation(
                np.array(chemical_similarity_data["similarities"]),
                np.array(chemical_similarity_data["response_rates"]),
                response_type="response_rate",
                n_bootstrap_samples=n_bootstrap_samples,
                random_seed=random_seed
            )

            report["correlations"]["chemical_similarity_vs_response_rate"] = {
                "pearson_r": corr_result.pearson_r,
                "pearson_p": corr_result.pearson_p,
                "spearman_rho": corr_result.spearman_rho,
                "spearman_p": corr_result.spearman_p,
                "n_samples": corr_result.n_samples,
                "ci_95": [corr_result.ci_95_lower, corr_result.ci_95_upper],
                "ci_99": [corr_result.ci_99_lower, corr_result.ci_99_upper],
                "significant_at_0.05": corr_result.pearson_p < ALPHA_05,
                "significant_at_0.01": corr_result.pearson_p < ALPHA_01
            }

        # Prediction probability correlation
        if "prediction_probabilities" in chemical_similarity_data:
            corr_result = compute_chemical_similarity_correlation(
                np.array(chemical_similarity_data["similarities"]),
                np.array(chemical_similarity_data["prediction_probabilities"]),
                response_type="prediction_probability",
                n_bootstrap_samples=n_bootstrap_samples,
                random_seed=random_seed
            )

            report["correlations"]["chemical_similarity_vs_prediction_probability"] = {
                "pearson_r": corr_result.pearson_r,
                "pearson_p": corr_result.pearson_p,
                "spearman_rho": corr_result.spearman_rho,
                "spearman_p": corr_result.spearman_p,
                "n_samples": corr_result.n_samples,
                "ci_95": [corr_result.ci_95_lower, corr_result.ci_95_upper],
                "ci_99": [corr_result.ci_99_lower, corr_result.ci_99_upper],
                "significant_at_0.05": corr_result.pearson_p < ALPHA_05,
                "significant_at_0.01": corr_result.pearson_p < ALPHA_01
            }

    return report


# CLI support for standalone usage
if __name__ == "__main__":
    import sys
    print("This module is intended to be imported, not run directly.")
    print("Use: from analysis.statistical_tests import run_all_statistical_tests")
    print("Or run: python analysis/run_statistical_tests.py --artifacts-dir <dir>")
    sys.exit(0)
