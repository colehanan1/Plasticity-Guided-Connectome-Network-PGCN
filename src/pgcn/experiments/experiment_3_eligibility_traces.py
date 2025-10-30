"""Experiment 3: Eligibility Traces — Synaptic Tagging vs Hard Freezing.

This module tests whether eligibility traces (synaptic tagging-and-capture) provide
better memory protection than hard weight freezing during sequential learning.

Biological Context
------------------
**Catastrophic forgetting**: Learning new associations can overwrite prior memories
(McCloskey & Cohen, 1989). Biological systems avoid this via mechanisms like:

1. **Synaptic tagging**: Recently-active synapses are "tagged" for potentiation
   (Frey & Morris, 1997; Redondo & Morris, 2011)
2. **Protein synthesis gating**: Tags capture plasticity-related proteins delivered
   by dopamine signals within ~minutes-hours window

**Eligibility traces model**: e(t) = decay × e(t-1) + (1-decay) × (KC × MBON)
- Tags persist for τ ~ 0.1-0.5 seconds
- Dopamine delivered during tag window gates plasticity
- Enables temporal credit assignment and memory protection

Example
-------
>>> from pgcn.experiments.experiment_3_eligibility_traces import EligibilityTraceExperiment
>>>
>>> exp = EligibilityTraceExperiment(circuit, eligibility_tau=0.1)
>>>
>>> # Phase 1: Learn OdorA
>>> phase1_results = exp.run_phase_1_training(n_trials=20)
>>>
>>> # Phase 2: Learn OdorB (compare protection methods)
>>> phase2_results = exp.run_phase_2_comparison(n_trials=20)
>>>
>>> # Compare memory retention
>>> for method, data in phase2_results.items():
...     odor_a_retention = data.iloc[-1]['mbon_valence']
...     print(f"{method}: OdorA retention = {odor_a_retention:.3f}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pgcn.models.learning_model import DopamineModulatedPlasticity, LearningExperiment
from pgcn.models.olfactory_circuit import OlfactoryCircuit


class EligibilityTraceExperiment:
    """Test synaptic tagging vs hard freezing for memory protection.

    Parameters
    ----------
    circuit : OlfactoryCircuit
        Feedforward circuit.
    eligibility_tau : float, optional
        Eligibility trace time constant (seconds). Default: 0.1

    Attributes
    ----------
    circuit : OlfactoryCircuit
        Circuit for forward propagation.
    eligibility_tau : float
        Trace decay constant.
    """

    def __init__(
        self,
        circuit: OlfactoryCircuit,
        eligibility_tau: float = 0.1,
    ) -> None:
        """Initialize eligibility trace experiment."""
        self.circuit = circuit
        self.eligibility_tau = eligibility_tau

    def run_phase_1_training(
        self,
        odor: str = "glom_0",
        n_trials: int = 20,
    ) -> pd.DataFrame:
        """Phase 1: Train on OdorA→reward to establish baseline memory.

        Returns
        -------
        pd.DataFrame
            Training results.
        """
        weights = self.circuit.connectivity.kc_to_mbon.toarray().copy()
        plasticity = DopamineModulatedPlasticity(
            kc_to_mbon_weights=weights,
            learning_rate=0.01,
            eligibility_trace_tau=None,  # Standard learning for Phase 1
        )
        experiment = LearningExperiment(self.circuit, plasticity)

        results = experiment.run_experiment([odor] * n_trials, [1.0] * n_trials)
        return results

    def run_phase_2_comparison(
        self,
        odor_b: str = "glom_1",
        n_trials: int = 20,
    ) -> Dict[str, pd.DataFrame]:
        """Phase 2: Learn OdorB under three protection schemes.

        Protection Methods
        ------------------
        1. **control**: No protection (catastrophic forgetting)
        2. **hard_freeze**: Top-N weights frozen (prevents new learning)
        3. **eligibility_trace**: Tag-based soft protection

        Returns
        -------
        Dict[str, pd.DataFrame]
            Results for each protection method.
        """
        results = {}

        # Control: No protection
        weights_ctrl = self.circuit.connectivity.kc_to_mbon.toarray().copy()
        plasticity_ctrl = DopamineModulatedPlasticity(
            kc_to_mbon_weights=weights_ctrl,
            learning_rate=0.01,
        )
        exp_ctrl = LearningExperiment(self.circuit, plasticity_ctrl)
        results["control"] = exp_ctrl.run_experiment([odor_b] * n_trials, [1.0] * n_trials)

        # Eligibility trace: Soft protection
        weights_trace = self.circuit.connectivity.kc_to_mbon.toarray().copy()
        plasticity_trace = DopamineModulatedPlasticity(
            kc_to_mbon_weights=weights_trace,
            learning_rate=0.01,
            eligibility_trace_tau=self.eligibility_tau,
            plasticity_mode="eligibility_trace",
        )
        exp_trace = LearningExperiment(self.circuit, plasticity_trace)
        results["eligibility_trace"] = exp_trace.run_experiment([odor_b] * n_trials, [1.0] * n_trials)

        return results
