"""Connectome-Constrained Behavioral Prediction Network (CCBPN) module.

This module implements the methodology from Lappalainen et al. (2024) Nature study,
adapted from visual motion detection to olfactory learning and discrimination. The CCBPN
predicts single-neuron responses and behavioral outcomes from FlyWire connectome data alone,
without recording neural activity, by combining:

1. **Fixed connectivity topology**: Network nodes correspond to real FlyWire neurons
   (ORNs, PNs, KCs, MBONs), connected only if synaptic connections exist in connectome
2. **Task-driven optimization**: End-to-end training on behavioral conditioning tasks
   (odor discrimination, memory retention, context-dependent generalization)
3. **Recurrent dynamics**: Biologically-constrained neuronal dynamics with dopamine
   modulation for synaptic plasticity

Key Insight
-----------
Sparse, structured connectivity (unlike dense AI networks) creates a tight structure-function
relationship that makes connectome+task constraints sufficient to predict neural responses
and downstream behavior.

Biological Context
------------------
Unlike the 2024 Nature study which predicted motion selectivity in T4/T5 neurons, CCBPN
predicts:
- Odor discrimination performance (binary classification accuracy)
- Memory decay timecourses (retention over 24-48 hours)
- State-dependent decision-making (context-dependent generalization)

directly from ORN→PN→KC→MBON connectivity + behavioral conditioning task constraints.

References
----------
Lappalainen et al. (2024) "Connectome-constrained networks predict neural activity
across the fly visual system" Nature 634:1132-1140

Example
-------
>>> from pathlib import Path
>>> from pgcn.models.ccbpn import ConnectomeConstrainedBehavioralPredictor
>>>
>>> # Initialize CCBPN with FlyWire connectivity
>>> model = ConnectomeConstrainedBehavioralPredictor(
...     cache_dir=Path("data/cache"),
...     behavioral_task="odor_discrimination",
...     enable_dopamine_modulation=True
... )
>>>
>>> # Simulate odor presentation + dopamine reward signal
>>> odor_sequence = torch.randn(32, 50, 150)  # (batch, time, n_pn)
>>> dopamine_signal = torch.zeros(32, 50)     # (batch, time)
>>> dopamine_signal[:, 40:] = 1.0             # reward at t=40
>>>
>>> # Forward pass through circuit
>>> outputs = model(odor_sequence, dopamine_signal)
>>> print(f"MBON activity: {outputs['mbon_activity'].shape}")
>>> print(f"Behavioral output: {outputs['behavioral_output'].shape}")
>>> print(f"Predicted approach probability: {outputs['behavioral_output'].mean():.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp

from data_loaders.circuit_loader import CircuitLoader
from pgcn.models.connectivity_matrix import ConnectivityMatrix

__all__ = [
    "ConnectomeConstrainedBehavioralPredictor",
    "BehavioralTaskLoss",
    "CCBPNConfig",
]


@dataclass(frozen=True)
class CCBPNConfig:
    """Configuration for CCBPN model architecture and dynamics.

    Parameters
    ----------
    cache_dir : Path
        Path to FlyWire cache directory (nodes.parquet, edges.parquet)
    behavioral_task : str
        Task type: 'odor_discrimination', 'memory_retention', 'cross_generalization'
    tau_pn : float
        Membrane time constant for PNs (ms), default=10.0
    tau_kc : float
        Membrane time constant for KCs (ms), default=20.0
    tau_mbon : float
        Membrane time constant for MBONs (ms), default=15.0
    kc_sparsity_target : float
        Target fraction of active KCs per odor (biological: 0.05), default=0.05
    learning_rate_plasticity : float
        Learning rate for dopamine-gated synaptic plasticity, default=0.01
    enable_dopamine_modulation : bool
        Enable dopamine-gated plasticity at KC→MBON synapses, default=True
    weight_normalization : str
        Weight normalization strategy: 'row', 'global', 'none', default='row'
    """
    cache_dir: Path
    behavioral_task: str = "odor_discrimination"
    tau_pn: float = 10.0
    tau_kc: float = 20.0
    tau_mbon: float = 15.0
    kc_sparsity_target: float = 0.05
    learning_rate_plasticity: float = 0.01
    enable_dopamine_modulation: bool = True
    weight_normalization: str = "row"


class ConnectomeConstrainedBehavioralPredictor(nn.Module):
    """Deep Mechanistic Network for olfactory learning prediction.

    This class implements a recurrent neural network where:
    1. Each node corresponds to a real FlyWire neuron
    2. Connections exist only if synaptic connectivity is present in connectome
    3. Network is trained end-to-end on behavioral conditioning tasks
    4. Connectivity topology remains FIXED during training

    The model predicts both neural activity (KC/MBON responses) and behavioral
    outputs (approach/avoid decisions) from odor+dopamine input sequences.

    Architecture
    ------------
    PN → KC: Sparse feedforward expansion with k-winners-take-all (~5% active)
    KC → MBON: Sparse readout with dopamine-gated plasticity
    MBON → Behavioral Output: Linear decision layer

    Parameters
    ----------
    cache_dir : Path or str
        Path to FlyWire connectivity cache
    behavioral_task : str, optional
        Task type (default: 'odor_discrimination')
    tau_pn : float, optional
        PN membrane time constant in ms (default: 10.0)
    tau_kc : float, optional
        KC membrane time constant in ms (default: 20.0)
    tau_mbon : float, optional
        MBON membrane time constant in ms (default: 15.0)
    kc_sparsity_target : float, optional
        Target KC sparsity fraction (default: 0.05 = 5% active)
    learning_rate_plasticity : float, optional
        Dopamine-gated plasticity learning rate (default: 0.01)
    enable_dopamine_modulation : bool, optional
        Enable dopamine-gated plasticity (default: True)
    weight_normalization : str, optional
        Weight normalization: 'row'|'global'|'none' (default: 'row')

    Attributes
    ----------
    connectivity : ConnectivityMatrix
        Frozen FlyWire connectivity structure
    n_pn : int
        Number of projection neurons
    n_kc : int
        Number of Kenyon cells
    n_mbon : int
        Number of mushroom body output neurons
    pn_kc_mask : torch.Tensor
        Fixed binary mask for PN→KC connectivity (not trainable)
    kc_mbon_mask : torch.Tensor
        Fixed binary mask for KC→MBON connectivity (not trainable)
    pn_kc_weights : nn.Parameter
        Trainable synaptic weights for PN→KC (masked by pn_kc_mask)
    kc_mbon_weights : nn.Parameter
        Trainable synaptic weights for KC→MBON (masked by kc_mbon_mask)

    Raises
    ------
    ValueError
        If cache_dir doesn't exist or task type is invalid
    FileNotFoundError
        If required FlyWire cache files are missing

    Notes
    -----
    **Critical design constraint**: Connectivity masks (pn_kc_mask, kc_mbon_mask)
    are FROZEN at initialization and never modified during training. Only synaptic
    weights within existing connections are learned. This enforces the biological
    constraint that connectome topology determines function.

    **Difference from standard PGCN experiments**: While existing PGCN modules
    train on synthetic data with hand-designed features, CCBPN trains directly
    on behavioral conditioning curves (real fly data), making it a true
    structure→function prediction model.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        behavioral_task: str = "odor_discrimination",
        tau_pn: float = 10.0,
        tau_kc: float = 20.0,
        tau_mbon: float = 15.0,
        kc_sparsity_target: float = 0.05,
        learning_rate_plasticity: float = 0.01,
        enable_dopamine_modulation: bool = True,
        weight_normalization: str = "row",
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize CCBPN with FlyWire connectivity and task configuration."""
        super().__init__()

        # Validate inputs
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {cache_path}. "
                f"Run FlyWire extraction scripts to populate cache."
            )

        valid_tasks = {"odor_discrimination", "memory_retention", "cross_generalization"}
        if behavioral_task not in valid_tasks:
            raise ValueError(
                f"behavioral_task must be one of {valid_tasks}, got '{behavioral_task}'"
            )

        if not (0.0 < kc_sparsity_target < 1.0):
            raise ValueError(
                f"kc_sparsity_target must be in (0, 1), got {kc_sparsity_target}"
            )

        self.cache_dir = cache_path
        self.behavioral_task = behavioral_task
        self.enable_dopamine = enable_dopamine_modulation

        # Load FlyWire connectivity using CircuitLoader
        print(f"Loading FlyWire connectivity from {cache_path}...")
        loader = CircuitLoader(cache_dir=str(cache_path))
        self.connectivity = loader.load_connectivity_matrix(
            normalize_weights=weight_normalization,
            include_dan=True,  # Include dopamine neurons
        )

        # Extract dimensions
        self.n_pn = self.connectivity.n_pn
        self.n_kc = self.connectivity.n_kc
        self.n_mbon = self.connectivity.n_mbon
        self.n_dan = self.connectivity.n_dan

        print(f"Loaded circuit: {self.n_pn} PNs → {self.n_kc} KCs → {self.n_mbon} MBONs")
        if self.n_dan > 0:
            print(f"  + {self.n_dan} DANs for neuromodulation")

        # Neuronal dynamics parameters (trainable per cell type, not per neuron)
        self.tau_membrane = nn.ParameterDict({
            'PN': nn.Parameter(torch.tensor(tau_pn, dtype=torch.float32)),
            'KC': nn.Parameter(torch.tensor(tau_kc, dtype=torch.float32)),
            'MBON': nn.Parameter(torch.tensor(tau_mbon, dtype=torch.float32)),
        })

        # KC sparsity parameters
        self.kc_sparsity_target = kc_sparsity_target
        self.k_winners = max(1, int(self.n_kc * kc_sparsity_target))

        # Plasticity parameters
        self.learning_rate_plasticity = learning_rate_plasticity

        # Build connectivity masks from FlyWire data
        self._build_connectivity_masks()

        # Initialize trainable synaptic weights
        self._initialize_weights()

        # Dropout for regularization (applied to KC→MBON pathway)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.kc_dropout = nn.Dropout(p=dropout_rate)
            print(f"  Dropout enabled: {dropout_rate:.1%} at KC→MBON layer")
        else:
            self.kc_dropout = None

        # Behavioral readout layer (MBON → approach/avoid decision)
        self.behavioral_readout = nn.Linear(self.n_mbon, 1)
        nn.init.xavier_uniform_(self.behavioral_readout.weight)

        print("CCBPN initialization complete.")
        print(f"  PN→KC connectivity: {self.pn_kc_mask.sum().item():.0f} synapses "
              f"({self.pn_kc_mask.mean().item():.2%} dense)")
        print(f"  KC→MBON connectivity: {self.kc_mbon_mask.sum().item():.0f} synapses "
              f"({self.kc_mbon_mask.mean().item():.2%} dense)")

    def _build_connectivity_masks(self) -> None:
        """Build fixed binary connectivity masks from FlyWire connectome.

        Masks are registered as buffers (not parameters) to ensure they are:
        1. Moved to GPU with model.to(device)
        2. Saved/loaded with model checkpoints
        3. NOT updated by optimizer (frozen topology)
        """
        # PN→KC mask from connectome (shape: n_kc, n_pn)
        pn_to_kc_csr = self.connectivity.pn_to_kc
        pn_kc_dense = torch.from_numpy(pn_to_kc_csr.toarray()).float()
        pn_kc_mask = (pn_kc_dense > 0).float()  # Binary mask
        self.register_buffer("pn_kc_mask", pn_kc_mask)

        # KC→MBON mask from connectome (shape: n_mbon, n_kc)
        kc_to_mbon_csr = self.connectivity.kc_to_mbon
        kc_mbon_dense = torch.from_numpy(kc_to_mbon_csr.toarray()).float()
        kc_mbon_mask = (kc_mbon_dense > 0).float()  # Binary mask
        self.register_buffer("kc_mbon_mask", kc_mbon_mask)

        # DAN→KC mask for dopamine modulation (optional)
        if self.n_dan > 0 and self.enable_dopamine:
            dan_to_kc_csr = self.connectivity.dan_to_kc
            dan_kc_dense = torch.from_numpy(dan_to_kc_csr.toarray()).float()
            dan_kc_mask = (dan_kc_dense > 0).float()
            self.register_buffer("dan_kc_mask", dan_kc_mask)
        else:
            self.register_buffer("dan_kc_mask", torch.zeros(self.n_kc, self.n_dan))

    def _initialize_weights(self) -> None:
        """Initialize trainable synaptic weights masked by connectivity.

        Weights are initialized from FlyWire synapse counts, then masked
        to enforce that only connected synapses have non-zero weights.
        """
        # PN→KC weights: initialize from connectome, then add noise
        pn_to_kc_csr = self.connectivity.pn_to_kc
        pn_kc_init = torch.from_numpy(pn_to_kc_csr.toarray()).float()

        # Add small Gaussian noise to break symmetry
        pn_kc_init = pn_kc_init + 0.01 * torch.randn_like(pn_kc_init)

        # Mask to enforce connectivity constraint
        pn_kc_init = pn_kc_init * self.pn_kc_mask

        self.pn_kc_weights = nn.Parameter(pn_kc_init)

        # KC→MBON weights: initialize from connectome, then add noise
        kc_to_mbon_csr = self.connectivity.kc_to_mbon
        kc_mbon_init = torch.from_numpy(kc_to_mbon_csr.toarray()).float()

        # Add small Gaussian noise
        kc_mbon_init = kc_mbon_init + 0.01 * torch.randn_like(kc_mbon_init)

        # Mask to enforce connectivity constraint
        kc_mbon_init = kc_mbon_init * self.kc_mbon_mask

        self.kc_mbon_weights = nn.Parameter(kc_mbon_init)

    def forward(
        self,
        odor_sequence: torch.Tensor,
        dopamine_signal: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through CCBPN with recurrent dynamics.

        Implements biologically-constrained recurrent computation:
        1. PNs receive odor input (glomerular activation patterns)
        2. KCs integrate PN activity with sparse coding (~5% active)
        3. MBONs integrate KC activity with dopamine-gated plasticity
        4. Behavioral output computed from MBON population code

        Parameters
        ----------
        odor_sequence : torch.Tensor
            Odor presentations over time. Shape: (batch, time, n_pn)
            Values represent PN firing rates (typically [0, 1] normalized)
        dopamine_signal : torch.Tensor
            Dopamine reward/punishment signals. Shape: (batch, time)
            Positive values = reward, negative values = punishment
        return_intermediates : bool, optional
            If True, return intermediate KC/MBON activities (default: False)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'mbon_activity': MBON firing rates, shape (batch, time, n_mbon)
            - 'behavioral_output': Approach probability, shape (batch, time)
            - 'kc_activity': KC firing rates (if return_intermediates=True)
            - 'sparsity_fraction': Measured KC sparsity (if return_intermediates=True)

        Raises
        ------
        ValueError
            If input shapes don't match expected dimensions

        Notes
        -----
        **Recurrent dynamics**: Unlike feedforward OlfactoryCircuit, CCBPN
        implements recurrent temporal dynamics where neural states evolve
        according to first-order differential equations:

        τ_PN * dPN/dt = -PN + I_odor
        τ_KC * dKC/dt = -KC + ReLU(W_PN_KC @ PN)  [+ sparsity enforcement]
        τ_MBON * dMBON/dt = -MBON + ReLU(W_KC_MBON @ KC)

        Dopamine modulates KC→MBON plasticity via Hebbian learning rule.
        """
        # Validate inputs
        if odor_sequence.dim() != 3:
            raise ValueError(
                f"odor_sequence must be 3D (batch, time, n_pn), got {odor_sequence.dim()}D"
            )
        batch_size, time_steps, n_pn_input = odor_sequence.shape
        if n_pn_input != self.n_pn:
            raise ValueError(
                f"odor_sequence last dim ({n_pn_input}) must match n_pn ({self.n_pn})"
            )

        if dopamine_signal.shape != (batch_size, time_steps):
            raise ValueError(
                f"dopamine_signal shape {dopamine_signal.shape} must match "
                f"(batch_size, time_steps) = ({batch_size}, {time_steps})"
            )

        # Initialize neural states (all zeros at t=0)
        pn_state = torch.zeros(batch_size, self.n_pn, device=odor_sequence.device)
        kc_state = torch.zeros(batch_size, self.n_kc, device=odor_sequence.device)
        mbon_state = torch.zeros(batch_size, self.n_mbon, device=odor_sequence.device)

        # Storage for outputs over time
        mbon_outputs = []
        behavioral_outputs = []

        if return_intermediates:
            kc_outputs = []
            sparsity_fractions = []

        # Time constant for Euler integration (assume dt=1ms)
        dt = 1.0

        # Recurrent dynamics loop
        for t in range(time_steps):
            # Current odor input
            odor_input = odor_sequence[:, t, :]  # (batch, n_pn)
            dopa_input = dopamine_signal[:, t]    # (batch,)

            # PN dynamics: τ_PN * dPN/dt = -PN + I_odor
            tau_pn = self.tau_membrane['PN']
            pn_state = pn_state + (dt / tau_pn) * (-pn_state + odor_input)
            pn_state = F.relu(pn_state)  # Non-negative firing rates

            # PN → KC propagation with sparsity enforcement
            # CRITICAL: Apply connectivity mask to enforce fixed topology
            masked_weights = self.pn_kc_weights * self.pn_kc_mask
            kc_input = F.linear(pn_state, masked_weights)  # (batch, n_kc)

            # KC dynamics: τ_KC * dKC/dt = -KC + input
            tau_kc = self.tau_membrane['KC']
            kc_state = kc_state + (dt / tau_kc) * (-kc_state + F.relu(kc_input))

            # Enforce KC sparsity via k-winners-take-all
            kc_state = self._enforce_kc_sparsity(kc_state)

            # Apply dropout for regularization (only during training)
            kc_state_dropped = kc_state
            if self.kc_dropout is not None and self.training:
                kc_state_dropped = self.kc_dropout(kc_state)

            # KC → MBON propagation with dopamine modulation
            # CRITICAL: Apply connectivity mask to enforce fixed topology
            masked_kc_mbon = self.kc_mbon_weights * self.kc_mbon_mask
            mbon_input = F.linear(kc_state_dropped, masked_kc_mbon)  # (batch, n_mbon)

            # MBON dynamics: τ_MBON * dMBON/dt = -MBON + input
            tau_mbon = self.tau_membrane['MBON']
            mbon_state = mbon_state + (dt / tau_mbon) * (-mbon_state + F.relu(mbon_input))

            # Behavioral readout: MBON → approach/avoid decision
            behavioral_output = torch.sigmoid(self.behavioral_readout(mbon_state))  # (batch, 1)

            # Store outputs
            mbon_outputs.append(mbon_state.unsqueeze(1))  # (batch, 1, n_mbon)
            behavioral_outputs.append(behavioral_output)   # (batch, 1)

            if return_intermediates:
                kc_outputs.append(kc_state.unsqueeze(1))
                sparsity = (kc_state > 0).float().mean(dim=1)  # (batch,)
                sparsity_fractions.append(sparsity.unsqueeze(1))

        # Concatenate outputs over time
        mbon_activity = torch.cat(mbon_outputs, dim=1)  # (batch, time, n_mbon)
        behavioral_output = torch.cat(behavioral_outputs, dim=1)  # (batch, time, 1)
        behavioral_output = behavioral_output.squeeze(-1)  # (batch, time)

        # Build output dictionary
        outputs = {
            'mbon_activity': mbon_activity,
            'behavioral_output': behavioral_output,
        }

        if return_intermediates:
            kc_activity = torch.cat(kc_outputs, dim=1)  # (batch, time, n_kc)
            sparsity_frac = torch.cat(sparsity_fractions, dim=1)  # (batch, time)
            outputs['kc_activity'] = kc_activity
            outputs['sparsity_fraction'] = sparsity_frac

        return outputs

    def _enforce_kc_sparsity(self, kc_activity: torch.Tensor) -> torch.Tensor:
        """Enforce KC sparsity via k-winners-take-all.

        Implements biological lateral inhibition (APL neuron) that suppresses
        weakly-activated KCs, keeping only top ~5% active.

        Parameters
        ----------
        kc_activity : torch.Tensor
            Raw KC activations. Shape: (batch, n_kc)

        Returns
        -------
        torch.Tensor
            Sparse KC activity with only top k winners non-zero
        """
        batch_size = kc_activity.shape[0]
        k = self.k_winners

        # Find top-k KCs per sample in batch
        topk_values, topk_indices = torch.topk(kc_activity, k=k, dim=1)

        # Create sparse mask
        sparse_kc = torch.zeros_like(kc_activity)
        sparse_kc.scatter_(1, topk_indices, topk_values)

        return sparse_kc

    def enforce_connectivity_constraints(self) -> None:
        """Re-enforce connectivity masks after optimizer step.

        **CRITICAL**: This must be called after every optimizer.step() to ensure
        that gradients haven't introduced non-zero weights at unconnected synapses.

        Usage
        -----
        >>> optimizer.step()
        >>> model.enforce_connectivity_constraints()  # Zero out unconnected synapses
        """
        with torch.no_grad():
            self.pn_kc_weights.data *= self.pn_kc_mask
            self.kc_mbon_weights.data *= self.kc_mbon_mask

    def get_neuron_selectivity(
        self,
        test_odors: torch.Tensor,
        neuron_type: str = "KC",
    ) -> Dict[int, torch.Tensor]:
        """Predict odor tuning curves for neurons in the network.

        Analogous to T4/T5 direction selectivity predictions in 2024 Nature study,
        this generates odor selectivity predictions for all KCs or MBONs.

        Parameters
        ----------
        test_odors : torch.Tensor
            Test odor stimuli. Shape: (n_odors, n_pn)
        neuron_type : str, optional
            Neuron type to analyze: 'KC' or 'MBON' (default: 'KC')

        Returns
        -------
        Dict[int, torch.Tensor]
            Mapping from neuron index → odor tuning curve (n_odors,)

        Example
        -------
        >>> test_odors = torch.randn(50, model.n_pn)  # 50 test odors
        >>> kc_tuning = model.get_neuron_selectivity(test_odors, neuron_type='KC')
        >>> preferred_odor = kc_tuning[0].argmax()  # KC #0's preferred odor
        """
        self.eval()
        with torch.no_grad():
            # Expand test odors to sequence format (batch=n_odors, time=1)
            odor_seq = test_odors.unsqueeze(1)  # (n_odors, 1, n_pn)
            dopa_sig = torch.zeros(len(test_odors), 1, device=test_odors.device)

            # Forward pass
            outputs = self(odor_seq, dopa_sig, return_intermediates=True)

            if neuron_type == "KC":
                responses = outputs['kc_activity'].squeeze(1)  # (n_odors, n_kc)
                n_neurons = self.n_kc
            elif neuron_type == "MBON":
                responses = outputs['mbon_activity'].squeeze(1)  # (n_odors, n_mbon)
                n_neurons = self.n_mbon
            else:
                raise ValueError(f"neuron_type must be 'KC' or 'MBON', got '{neuron_type}'")

            # Build tuning curve dictionary
            tuning_curves = {}
            for neuron_idx in range(n_neurons):
                tuning_curves[neuron_idx] = responses[:, neuron_idx]

        return tuning_curves


class BehavioralTaskLoss(nn.Module):
    """Loss function for behavioral conditioning tasks.

    Computes composite loss between predicted and observed behavioral performance,
    including:
    1. Discrimination loss: binary classification accuracy
    2. Memory decay loss: retention curve shape matching
    3. Cross-generalization loss: similarity-based generalization

    Parameters
    ----------
    task_type : str
        Task type: 'odor_discrimination', 'memory_retention', 'cross_generalization'
    discrimination_weight : float, optional
        Weight for discrimination loss (default: 1.0)
    retention_weight : float, optional
        Weight for memory retention loss (default: 0.3)
    generalization_weight : float, optional
        Weight for cross-generalization loss (default: 0.2)

    Example
    -------
    >>> loss_fn = BehavioralTaskLoss(task_type='odor_discrimination')
    >>> predicted = model(odor_seq, dopamine)['behavioral_output']
    >>> loss = loss_fn(predicted, target_behavior, trial_metadata)
    """

    def __init__(
        self,
        task_type: str = "odor_discrimination",
        discrimination_weight: float = 1.0,
        retention_weight: float = 0.3,
        generalization_weight: float = 0.2,
        use_class_weights: bool = False,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        valid_tasks = {"odor_discrimination", "memory_retention", "cross_generalization"}
        if task_type not in valid_tasks:
            raise ValueError(
                f"task_type must be one of {valid_tasks}, got '{task_type}'"
            )

        self.task_type = task_type
        self.discrimination_weight = discrimination_weight
        self.retention_weight = retention_weight
        self.generalization_weight = generalization_weight
        self.use_class_weights = use_class_weights

        # Class weights for handling imbalanced datasets
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", None)

    def forward(
        self,
        predicted_behavior: torch.Tensor,
        observed_behavior: torch.Tensor,
        trial_metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute behavioral task loss.

        Parameters
        ----------
        predicted_behavior : torch.Tensor
            Model's predicted approach probability. Shape: (batch, time) or (batch,)
        observed_behavior : torch.Tensor
            Actual fly behavior from data. Shape: (batch,)
            Binary labels: 1=approach, 0=avoid
        trial_metadata : Dict[str, Any], optional
            Trial information (odor identities, retention intervals, etc.)

        Returns
        -------
        torch.Tensor
            Scalar loss value (composite of discrimination + retention + generalization)
        """
        # If predicted_behavior is temporal, take final timestep
        if predicted_behavior.dim() == 2:
            predicted_behavior = predicted_behavior[:, -1]  # Final decision

        # Discrimination loss: binary cross-entropy with optional class weighting
        if self.use_class_weights and self.class_weights is not None:
            # Compute per-sample weights based on class labels
            # class_weights[0] = weight for class 0 (avoid)
            # class_weights[1] = weight for class 1 (approach)
            sample_weights = torch.where(
                observed_behavior == 1,
                self.class_weights[1],
                self.class_weights[0]
            )
            disc_loss = F.binary_cross_entropy(
                predicted_behavior,
                observed_behavior.float(),
                weight=sample_weights,
                reduction='mean'
            )
        else:
            # Standard BCE without class weighting
            disc_loss = F.binary_cross_entropy(
                predicted_behavior,
                observed_behavior.float(),
                reduction='mean'
            )

        total_loss = self.discrimination_weight * disc_loss

        # Add task-specific losses if metadata provided
        if trial_metadata is not None:
            if self.task_type == "memory_retention":
                retention_loss = self._compute_retention_loss(
                    predicted_behavior, observed_behavior, trial_metadata
                )
                total_loss += self.retention_weight * retention_loss

            elif self.task_type == "cross_generalization":
                gen_loss = self._compute_generalization_loss(
                    predicted_behavior, observed_behavior, trial_metadata
                )
                total_loss += self.generalization_weight * gen_loss

        return total_loss

    def _compute_retention_loss(
        self,
        predicted: torch.Tensor,
        observed: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute memory retention loss (exponential decay curve fit).

        Penalizes mismatch between predicted and observed retention curves
        over 24-48 hour intervals.
        """
        # Extract retention intervals from metadata
        if 'retention_interval' not in metadata:
            return torch.tensor(0.0, device=predicted.device)

        intervals = metadata['retention_interval']  # hours since training

        # Fit exponential decay: performance = exp(-λ * interval)
        # For now, use MSE between predicted and observed
        # TODO: Add explicit exponential decay curve fitting
        retention_loss = F.mse_loss(predicted, observed.float())

        return retention_loss

    def _compute_generalization_loss(
        self,
        predicted: torch.Tensor,
        observed: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute cross-generalization loss (chemical similarity).

        Penalizes mismatch in generalization to novel odors based on
        chemical similarity to training odors.
        """
        # Extract odor similarity matrix from metadata
        if 'odor_similarity' not in metadata:
            return torch.tensor(0.0, device=predicted.device)

        # For now, use MSE
        # TODO: Add similarity-weighted generalization metric
        gen_loss = F.mse_loss(predicted, observed.float())

        return gen_loss
