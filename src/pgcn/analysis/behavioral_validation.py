"""Behavioral validation: Compare model predictions to real Drosophila learning data.

This module validates PGCN predictions against empirical behavioral data from
optogenetic conditioning experiments. The core challenge in computational neuroscience
is ensuring that models capture real neural and behavioral phenomena, not just
abstract patterns. This validator implements rigorous comparison metrics used in
the Drosophila learning field.

Biological Context
------------------
Drosophila olfactory learning is measured using classical conditioning protocols:

1. **Training phase**: Flies receive odor (CS) paired with reward/punishment (US)
   - CS+ (e.g., ethyl butyrate) → shock (aversive) or sugar (appetitive)
   - CS- (e.g., methylcyclohexanol) → no outcome
   - Typical: 12 training trials alternating CS+ and CS-

2. **Test phase**: Flies choose between CS+ and CS- in T-maze
   - Learning Index (LI) = (# approach CS- - # approach CS+) / total
   - Range: [-1, 1] where LI > 0.5 indicates strong aversive learning

3. **Memory retention**: Test again at 3h, 24h to measure memory decay
   - Short-term memory (STM): 0-30 min, γ KC-dependent
   - Long-term memory (LTM): >24h, α/β KC-dependent

4. **Optogenetic perturbations**: Silencing or activating specific neurons
   during training to test causal role in learning

This module compares model learning curves (MBON outputs over trials) to real
fly behavioral learning curves (learning index over trials or sessions).

Key Validation Metrics
----------------------
1. **Root Mean Square Error (RMSE)**: Euclidean distance between curves
2. **Pearson correlation (r)**: Linear relationship between model and data
3. **Saturation similarity**: How well model matches final learning asymptote
4. **Learning rate similarity**: How well model matches early trial dynamics

Example
-------
>>> from pathlib import Path
>>> from pgcn.analysis.behavioral_validation import BehavioralValidator
>>> import pandas as pd
>>> import numpy as np
>>>
>>> # Load real fly data
>>> fly_data = pd.read_csv("data/behavioral/model_predictions.csv")
>>>
>>> # Create synthetic model learning curves
>>> model_curves = {
...     'control': np.linspace(0.0, 0.8, 20),
...     'opto_silencing': np.linspace(0.0, 0.3, 20),
... }
>>>
>>> # Validate
>>> validator = BehavioralValidator(model_curves, fly_data)
>>> metrics = validator.compare_learning_curves('control')
>>> print(f"RMSE: {metrics['rmse']:.3f}")
>>> print(f"Pearson r: {metrics['pearson_r']:.3f}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats


class BehavioralValidator:
    """Validate PGCN predictions against real Drosophila behavioral data.

    This class implements comparison metrics between model learning curves
    (MBON outputs over trials) and real fly behavioral data (learning index
    or choice probability over trials). It enables rigorous validation that
    the model captures biologically realistic learning dynamics.

    Parameters
    ----------
    model_learning_curves : Dict[str, np.ndarray]
        Dictionary mapping condition name to model learning curve.
        Each curve is a 1D array of MBON outputs (or learning index) over trials.
        Example: {'control': [0.1, 0.2, ..., 0.8], 'opto_pn_silence': [0.1, ..., 0.3]}
    real_fly_data : pd.DataFrame
        DataFrame with real behavioral data. Expected columns:
        - 'dataset': Condition name (e.g., 'EB_control', 'opto_hex')
        - 'fly': Fly identifier (e.g., 'october_28_batch_1')
        - 'fly_number': Individual fly number within batch
        - 'trial_label': Trial identifier (e.g., 'training_1', 'testing_1')
        - 'prediction': Binary outcome (0 or 1)
        - 'probability': Model probability or learning index [0, 1]

    Attributes
    ----------
    model_curves : Dict[str, np.ndarray]
        Stored model learning curves.
    fly_data : pd.DataFrame
        Stored real fly behavioral data.

    Examples
    --------
    Load real data and compare to model:
    >>> fly_data = pd.read_csv("/path/to/model_predictions.csv")
    >>> model_curves = {'control': np.linspace(0, 0.8, 20)}
    >>> validator = BehavioralValidator(model_curves, fly_data)
    >>> metrics = validator.compare_learning_curves('control')
    """

    def __init__(
        self,
        model_learning_curves: Dict[str, np.ndarray],
        real_fly_data: pd.DataFrame,
    ) -> None:
        """Initialize validator with model and real data."""
        self.model_curves = model_learning_curves
        self.fly_data = real_fly_data

        # Validate fly_data has required columns
        required_cols = ['dataset', 'fly', 'trial_label', 'prediction', 'probability']
        missing_cols = set(required_cols) - set(real_fly_data.columns)
        if missing_cols:
            raise ValueError(
                f"real_fly_data missing required columns: {missing_cols}"
            )

    @staticmethod
    def compute_learning_index(
        mbon_cs_plus: np.ndarray,
        mbon_cs_minus: np.ndarray,
    ) -> float:
        """Compute learning index from MBON outputs for CS+ and CS-.

        The Learning Index (LI) is a standard metric in Drosophila conditioning:
        LI = (response to CS- - response to CS+) / (|CS-| + |CS+|)

        For aversive learning (shock paired with CS+):
        - Flies learn to avoid CS+ → approach CS-
        - LI > 0 indicates learning

        For appetitive learning (sugar paired with CS+):
        - Flies learn to approach CS+ → avoid CS-
        - LI < 0 indicates learning (often reported as -LI to make positive)

        In model terms: MBON output represents valence, where:
        - High MBON output = approach behavior
        - Low MBON output = avoidance behavior

        Parameters
        ----------
        mbon_cs_plus : np.ndarray
            MBON outputs for CS+ trials (1D array).
        mbon_cs_minus : np.ndarray
            MBON outputs for CS- trials (1D array).

        Returns
        -------
        float
            Learning Index in range [-1, 1].
            - LI > 0: Approach CS- more than CS+ (aversive learning)
            - LI < 0: Approach CS+ more than CS- (appetitive learning)
            - LI ≈ 0: No discrimination learned

        Examples
        --------
        Strong aversive learning (avoid CS+):
        >>> mbon_cs_plus = np.array([0.1, 0.1, 0.1])  # Low valence → avoidance
        >>> mbon_cs_minus = np.array([0.9, 0.9, 0.9])  # High valence → approach
        >>> li = BehavioralValidator.compute_learning_index(mbon_cs_plus, mbon_cs_minus)
        >>> print(f"LI: {li:.2f}")  # Expect ~0.8 (strong learning)
        """
        mean_cs_plus = mbon_cs_plus.mean()
        mean_cs_minus = mbon_cs_minus.mean()

        numerator = mean_cs_minus - mean_cs_plus
        denominator = np.abs(mean_cs_minus) + np.abs(mean_cs_plus)

        if denominator < 1e-9:
            return 0.0

        return numerator / denominator

    def compare_learning_curves(
        self,
        condition: str,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compare model learning curve to real fly data for a condition.

        This method computes multiple comparison metrics between model predictions
        and empirical behavioral data. It's the core validation function for
        assessing whether the model captures real learning dynamics.

        Parameters
        ----------
        condition : str
            Condition name in model_curves (e.g., 'control', 'opto_pn_silence').
        dataset_name : Optional[str]
            Dataset name in fly_data to compare against (e.g., 'EB_control').
            If None, uses condition name to find matching dataset.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'condition': Condition name
            - 'dataset': Dataset name from fly_data
            - 'rmse': Root mean square error between curves
            - 'pearson_r': Pearson correlation coefficient
            - 'pearson_p': p-value for correlation
            - 'saturation_similarity': How well final values match (1 = perfect, 0 = max diff)
            - 'learning_rate_similarity': How well early slopes match (1 = perfect, 0 = max diff)
            - 'model_final_value': Model's final learning value
            - 'fly_final_value': Real fly data's final learning value
            - 'n_trials_compared': Number of trials used in comparison

        Raises
        ------
        ValueError
            If condition not in model_curves or dataset not found in fly_data.

        Notes
        -----
        - Curves are interpolated to common length for fair comparison
        - RMSE penalizes large deviations at any trial
        - Pearson r captures overall shape similarity
        - Saturation similarity focuses on final learning asymptote
        - Learning rate similarity focuses on early trial dynamics

        Examples
        --------
        >>> validator = BehavioralValidator(model_curves, fly_data)
        >>> metrics = validator.compare_learning_curves('control', dataset_name='EB_control')
        >>> print(f"Model matches data: r = {metrics['pearson_r']:.3f}, RMSE = {metrics['rmse']:.3f}")
        """
        if condition not in self.model_curves:
            raise ValueError(
                f"Condition '{condition}' not found in model_curves. "
                f"Available: {list(self.model_curves.keys())}"
            )

        # Get model curve
        model_curve = self.model_curves[condition]

        # Find matching dataset in fly_data
        if dataset_name is None:
            dataset_name = condition

        fly_subset = self.fly_data[self.fly_data['dataset'] == dataset_name]
        if len(fly_subset) == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in fly_data. "
                f"Available: {self.fly_data['dataset'].unique().tolist()}"
            )

        # Aggregate fly data across flies for this dataset
        # Use 'probability' column as learning measure
        fly_curve = fly_subset.groupby('trial_label')['probability'].mean().values

        # Interpolate to common length
        if len(model_curve) != len(fly_curve):
            # Resample fly_curve to match model_curve length
            x_fly = np.linspace(0, 1, len(fly_curve))
            x_model = np.linspace(0, 1, len(model_curve))
            fly_curve = np.interp(x_model, x_fly, fly_curve)

        # Compute comparison metrics
        # 1. Root mean square error
        rmse = np.sqrt(np.mean((model_curve - fly_curve) ** 2))

        # 2. Pearson correlation
        if len(model_curve) > 2:
            pearson_r, pearson_p = scipy.stats.pearsonr(model_curve, fly_curve)
        else:
            pearson_r, pearson_p = 0.0, 1.0

        # 3. Saturation similarity (how well final values match)
        model_final = model_curve[-1]
        fly_final = fly_curve[-1]
        max_final_diff = max(1.0, np.abs(fly_final))  # Normalize by fly final value
        saturation_similarity = 1.0 - np.abs(model_final - fly_final) / max_final_diff

        # 4. Learning rate similarity (early trial slopes)
        early_trials = min(5, len(model_curve))
        if early_trials >= 2:
            model_slope = np.polyfit(np.arange(early_trials), model_curve[:early_trials], 1)[0]
            fly_slope = np.polyfit(np.arange(early_trials), fly_curve[:early_trials], 1)[0]
            max_slope_diff = max(0.1, np.abs(fly_slope))  # Avoid division by zero
            learning_rate_similarity = 1.0 - np.abs(model_slope - fly_slope) / max_slope_diff
        else:
            learning_rate_similarity = 0.0

        return {
            'condition': condition,
            'dataset': dataset_name,
            'rmse': float(rmse),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'saturation_similarity': float(np.clip(saturation_similarity, 0.0, 1.0)),
            'learning_rate_similarity': float(np.clip(learning_rate_similarity, 0.0, 1.0)),
            'model_final_value': float(model_final),
            'fly_final_value': float(fly_final),
            'n_trials_compared': int(len(model_curve)),
        }

    def predict_optogenetic_outcome(
        self,
        perturbation_type: str,
        target_neurons: str,
        control_condition: str = "control",
        efficacy: float = 1.0,
    ) -> float:
        """Predict learning deficit caused by optogenetic perturbation.

        This method provides a simple heuristic for predicting how optogenetic
        manipulations will affect learning based on the target neuron population.
        It can be used to generate testable predictions before running experiments.

        Parameters
        ----------
        perturbation_type : str
            Type of perturbation: "silence" or "activate"
        target_neurons : str
            Which population to perturb: "pn", "kc", "mbon", "dan"
        control_condition : str
            Name of control condition in model_curves to use as baseline
        efficacy : float
            Perturbation efficacy [0, 1]. 1.0 = complete silencing/activation

        Returns
        -------
        float
            Predicted final learning value with perturbation applied.
            E.g., if control reaches LI=0.8 and PN silencing causes 60% deficit,
            returns 0.32 (40% of control learning).

        Notes
        -----
        This is a simplified heuristic based on circuit position:
        - PN silencing: ~30% deficit (input layer)
        - KC silencing: ~60% deficit (expansion layer)
        - MBON silencing: ~90% deficit (output layer)
        - DAN silencing: ~100% deficit (teaching signal)

        Activation effects are estimated as opposite polarity.

        Examples
        --------
        Predict effect of DA1 PN silencing:
        >>> validator = BehavioralValidator(model_curves, fly_data)
        >>> predicted = validator.predict_optogenetic_outcome(
        ...     perturbation_type="silence",
        ...     target_neurons="pn",
        ...     control_condition="control",
        ...     efficacy=1.0
        ... )
        >>> print(f"Predicted learning with PN silencing: {predicted:.3f}")
        """
        if control_condition not in self.model_curves:
            raise ValueError(
                f"Control condition '{control_condition}' not in model_curves"
            )

        control_final = self.model_curves[control_condition][-1]

        # Heuristic deficit magnitudes based on circuit position
        if perturbation_type == "silence":
            if target_neurons == "pn":
                deficit = 0.30 * efficacy  # Modest deficit (redundant input)
            elif target_neurons == "kc":
                deficit = 0.60 * efficacy  # Larger deficit (sparse expansion)
            elif target_neurons == "mbon":
                deficit = 0.90 * efficacy  # Severe deficit (output readout)
            elif target_neurons == "dan":
                deficit = 1.00 * efficacy  # Complete deficit (teaching signal)
            else:
                deficit = 0.50 * efficacy

            predicted = control_final * (1.0 - deficit)

        elif perturbation_type == "activate":
            # Activation may enhance learning (heuristic: opposite of silencing)
            if target_neurons == "pn":
                enhancement = 0.20 * efficacy
            elif target_neurons == "kc":
                enhancement = 0.30 * efficacy
            elif target_neurons == "mbon":
                enhancement = 0.40 * efficacy
            elif target_neurons == "dan":
                enhancement = 0.50 * efficacy
            else:
                enhancement = 0.25 * efficacy

            predicted = control_final * (1.0 + enhancement)

        else:
            # Unknown perturbation type, return control
            predicted = control_final

        return float(predicted)

    def compute_aggregate_validation_metrics(
        self,
        conditions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute validation metrics for multiple conditions.

        Parameters
        ----------
        conditions : Optional[List[str]]
            List of conditions to compare. If None, compare all conditions
            in model_curves.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per condition, columns for all metrics.

        Examples
        --------
        >>> validator = BehavioralValidator(model_curves, fly_data)
        >>> aggregate = validator.compute_aggregate_validation_metrics()
        >>> print(aggregate[['condition', 'rmse', 'pearson_r']])
        """
        if conditions is None:
            conditions = list(self.model_curves.keys())

        results = []
        for condition in conditions:
            # Try to find matching dataset in fly_data
            matching_datasets = self.fly_data['dataset'].unique()
            dataset_name = None
            for ds in matching_datasets:
                if condition.lower() in ds.lower() or ds.lower() in condition.lower():
                    dataset_name = ds
                    break

            if dataset_name is None:
                # Use first dataset as fallback
                dataset_name = matching_datasets[0]

            try:
                metrics = self.compare_learning_curves(condition, dataset_name)
                results.append(metrics)
            except Exception as e:
                # Skip conditions that fail comparison
                print(f"Warning: Could not compare condition '{condition}': {e}")
                continue

        return pd.DataFrame(results)

    @staticmethod
    def load_behavioral_data(
        data_path: Path,
    ) -> pd.DataFrame:
        """Load behavioral data from CSV file.

        Parameters
        ----------
        data_path : Path
            Path to behavioral data CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded behavioral data with validated columns.

        Raises
        ------
        FileNotFoundError
            If data_path does not exist.
        ValueError
            If CSV missing required columns.

        Examples
        --------
        >>> data_path = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv")
        >>> fly_data = BehavioralValidator.load_behavioral_data(data_path)
        >>> print(f"Loaded {len(fly_data)} trials from {fly_data['dataset'].nunique()} datasets")
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Behavioral data file not found: {data_path}")

        fly_data = pd.read_csv(data_path)

        # Validate required columns
        required_cols = ['dataset', 'fly', 'trial_label', 'prediction', 'probability']
        missing = set(required_cols) - set(fly_data.columns)
        if missing:
            raise ValueError(
                f"Behavioral data missing required columns: {missing}\n"
                f"Found columns: {fly_data.columns.tolist()}"
            )

        return fly_data
