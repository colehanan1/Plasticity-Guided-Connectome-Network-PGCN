"""Experimental modules for causal hypothesis testing in plasticity dynamics.

This package implements computational experiments that test specific hypotheses about
how blocking, credit assignment, and memory consolidation emerge from the mushroom
body's plasticity mechanisms.

Modules
-------
experiment_1_veto_gate
    Single-unit veto gate implementing blocking via pathway-specific inhibition
experiment_2_counterfactual_microsurgery
    Causal proof via minimal surgical edits (ablation, freezing, sign-flip)
experiment_3_eligibility_traces
    Synaptic tagging vs hard freezing for memory protection
experiment_6_shapley_analysis
    Shapley value attribution to identify and edit blocking neurons
"""

from pgcn.experiments.experiment_1_veto_gate import VetoGateExperiment
from pgcn.experiments.experiment_2_counterfactual_microsurgery import (
    CounterfactualMicrosurgeryExperiment,
)
from pgcn.experiments.experiment_3_eligibility_traces import EligibilityTraceExperiment
from pgcn.experiments.experiment_6_shapley_analysis import ShapleyBlockingAnalysis

__all__ = [
    "CounterfactualMicrosurgeryExperiment",
    "EligibilityTraceExperiment",
    "ShapleyBlockingAnalysis",
    "VetoGateExperiment",
]
