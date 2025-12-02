"""Connectome-Constrained Behavioral Predictor with Recurrent Context Memory.

Extends the base DrosophilaReservoir (PN→KC→MBON) with an LSTM that maintains
trial-to-trial memory, enabling context-dependent learning across multiple datasets.

Biological Motivation
---------------------
Real Drosophila maintain trial-to-trial memory through synaptic tags and dopaminergic
plasticity that accumulates across experiences. This module implements recurrent context
memory where:

1. **Context accumulation**: LSTM integrates previous trial outcomes and MBON activity
2. **Context-dependent modulation**: Learned context modulates current trial processing
3. **Gated memory**: Network learns when to rely on memory vs. current input
4. **Trial-to-trial learning**: Maintains separate associations for different contexts
   (e.g., hexanol=CS+ in opto_hex but CS- in opto_benz)

This enables the model to learn context-specific odor-outcome associations without
requiring explicit context labels at test time.

Example
-------
>>> from pgcn.models.ccbpn_recurrent import CCBPNWithRecurrentContext
>>>
>>> # Initialize model
>>> model = CCBPNWithRecurrentContext(
...     n_pn=150,
...     n_kc=2000,
...     n_mbon=44,
...     cache_dir="data/cache",
...     kc_sparsity=0.05,
...     context_dim=64
... )
>>>
>>> # Process sequential trials for one fly
>>> hidden_state = None
>>> previous_outcome = None
>>>
>>> for trial_idx, (odor_seq, dopamine_signal, label) in enumerate(fly_trials):
...     outputs = model(
...         odor_sequences=odor_seq,
...         dopamine_signals=dopamine_signal,
...         hidden_state=hidden_state,
...         previous_outcome=previous_outcome
...     )
...
...     prediction = outputs['behavioral_output']
...     hidden_state = outputs['hidden_state']  # Carry forward
...     previous_outcome = torch.tensor([label])  # Use true label
...
>>> print(f"Trial {trial_idx}: predicted {prediction.item():.3f}, actual {label}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .reservoir import DrosophilaReservoir

__all__ = ["CCBPNWithRecurrentContext"]


class CCBPNWithRecurrentContext(nn.Module):
    """CCBPN with recurrent context memory for multi-dataset learning.

    Architecture
    ------------
    1. **Base CCBPN** (PN → KC → MBON): Processes current trial using connectome
    2. **Recurrent context**: LSTM accumulates context from previous trials
    3. **Context modulation**: Context modulates MBON → behavior mapping
    4. **Gated integration**: Learns when to rely on memory vs. current input

    This enables the model to maintain separate associations for different training
    contexts without requiring explicit context labels.

    Parameters
    ----------
    n_pn : int
        Number of projection neurons (PNs)
    n_kc : int
        Number of Kenyon cells (KCs)
    n_mbon : int
        Number of mushroom body output neurons (MBONs)
    cache_dir : str or Path
        Path to FlyWire connectivity cache
    kc_sparsity : float, optional
        KC activation sparsity (fraction active). Default: 0.05 (5%)
    context_dim : int, optional
        Dimensionality of context embedding. Default: 64
    use_gate : bool, optional
        If True, use learned gating; if False, always use context. Default: True
    dropout : float, optional
        Dropout probability for context processing. Default: 0.2

    Attributes
    ----------
    ccbpn_core : DrosophilaReservoir
        Base CCBPN (stateless trial processing)
    context_memory : nn.LSTM
        LSTM for accumulating trial-to-trial context
    context_gate : nn.Sequential
        Network that learns when to use memory
    context_modulation : nn.Sequential
        Network that modulates MBON signals with context
    """

    def __init__(
        self,
        n_pn: int,
        n_kc: int,
        n_mbon: int,
        cache_dir: str | Path,
        kc_sparsity: float = 0.05,
        context_dim: int = 64,
        use_gate: bool = True,
        dropout: float = 0.2,
    ) -> None:
        """Initialize CCBPN with recurrent context memory.

        Parameters
        ----------
        n_pn : int
            Number of projection neurons
        n_kc : int
            Number of Kenyon cells
        n_mbon : int
            Number of mushroom body output neurons
        cache_dir : str or Path
            Path to FlyWire connectivity cache
        kc_sparsity : float
            KC activation sparsity (biological: 0.05-0.10)
        context_dim : int
            Dimensionality of context embedding
        use_gate : bool
            Whether to use learned gating for context integration
        dropout : float
            Dropout probability for regularization
        """
        super().__init__()

        # Base CCBPN (stateless trial processing)
        self.ccbpn_core = DrosophilaReservoir(
            n_pn=n_pn,
            n_kc=n_kc,
            n_mbon=n_mbon,
            kc_sparsity=kc_sparsity,
            cache_dir=cache_dir,
        )

        self.n_pn = n_pn
        self.n_kc = n_kc
        self.n_mbon = n_mbon
        self.context_dim = context_dim
        self.use_gate = use_gate

        # Recurrent context memory
        # Input: MBON activity + mean dopamine + previous outcome
        self.context_memory = nn.LSTM(
            input_size=n_mbon + 2,  # MBON + dopamine + outcome
            hidden_size=context_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # Single layer, so no dropout
        )

        # Context gating (learns when to use memory)
        if self.use_gate:
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim + n_mbon, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        else:
            self.context_gate = None

        # Context modulation of MBON signals
        self.context_modulation = nn.Sequential(
            nn.Linear(context_dim, n_mbon),
            nn.Tanh(),  # Bounded modulation [-1, 1]
        )

        # Final behavior prediction
        self.behavior_head = nn.Sequential(
            nn.Linear(n_mbon, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize recurrent and modulation layers with appropriate scales."""
        # LSTM initialization (PyTorch default is good)
        for name, param in self.context_memory.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Context modulation initialization
        for module in self.context_modulation.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Gate initialization (if used)
        if self.context_gate is not None:
            for module in self.context_gate.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # Behavior head initialization
        for module in self.behavior_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        odor_sequences: torch.Tensor,
        dopamine_signals: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        previous_outcome: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with recurrent context.

        Parameters
        ----------
        odor_sequences : torch.Tensor
            PN activity sequences. Shape: (batch, time, n_pn)
        dopamine_signals : torch.Tensor
            Dopamine signals. Shape: (batch, time)
        hidden_state : tuple of torch.Tensor, optional
            LSTM hidden state from previous trial: (h, c).
            Each shape: (1, batch, context_dim). If None, initializes to zeros.
        previous_outcome : torch.Tensor, optional
            Outcome of previous trial. Shape: (batch,). If None, uses zeros.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'behavioral_output': Predicted behavior (batch,)
            - 'hidden_state': Updated LSTM state for next trial
            - 'context': Context vector (batch, context_dim)
            - 'gate_value': Gate value (batch, 1) if using gating
            - 'kc_activity': KC activity (batch, time, n_kc)
            - 'mbon_output': MBON activity (batch, n_mbon)
        """
        batch_size = odor_sequences.shape[0]
        time_steps = odor_sequences.shape[1]
        device = odor_sequences.device

        # 1. Process current trial through base CCBPN
        # Reshape for processing: (batch, time, n_pn) → (batch*time, n_pn)
        odor_flat = odor_sequences.reshape(batch_size * time_steps, self.n_pn)

        # Forward through PN→KC→MBON
        mbon_flat = self.ccbpn_core(odor_flat)  # (batch*time, n_mbon)

        # Reshape back: (batch*time, n_mbon) → (batch, time, n_mbon)
        mbon_output = mbon_flat.reshape(batch_size, time_steps, self.n_mbon)

        # Aggregate MBON activity over time (mean)
        mbon_mean = mbon_output.mean(dim=1)  # (batch, n_mbon)

        # 2. Initialize or retrieve context from previous trial
        if hidden_state is None:
            h_0 = torch.zeros(1, batch_size, self.context_dim, device=device)
            c_0 = torch.zeros(1, batch_size, self.context_dim, device=device)
            hidden_state = (h_0, c_0)

        if previous_outcome is None:
            previous_outcome = torch.zeros(batch_size, device=device)

        # 3. Update context memory with previous trial's outcome
        # Input to LSTM: [MBON activity | dopamine | outcome] from PREVIOUS trial
        dopamine_mean = dopamine_signals.mean(dim=1, keepdim=True)  # (batch, 1)
        previous_outcome_expanded = previous_outcome.unsqueeze(1)  # (batch, 1)

        # Concatenate features
        lstm_input = torch.cat([
            mbon_mean,
            dopamine_mean,
            previous_outcome_expanded,
        ], dim=1)  # (batch, n_mbon + 2)

        # LSTM expects (batch, seq_len=1, input_size)
        lstm_input_seq = lstm_input.unsqueeze(1)  # (batch, 1, n_mbon+2)

        # Update context
        context_seq, new_hidden_state = self.context_memory(lstm_input_seq, hidden_state)
        context = context_seq.squeeze(1)  # (batch, context_dim)

        # 4. Gate: How much to use context vs. current input
        if self.use_gate:
            gate_input = torch.cat([context, mbon_mean], dim=1)  # (batch, context_dim+n_mbon)
            gate_value = self.context_gate(gate_input)  # (batch, 1)
        else:
            gate_value = torch.ones(batch_size, 1, device=device)

        # 5. Modulate MBON output with context
        context_signal = self.context_modulation(context)  # (batch, n_mbon)

        # Combine context and current MBON activity
        mbon_modulated = gate_value * context_signal + (1 - gate_value) * mbon_mean

        # 6. Final behavioral decision
        behavioral_output = self.behavior_head(mbon_modulated).squeeze(1)  # (batch,)

        # 7. Prepare outputs
        outputs = {
            'behavioral_output': behavioral_output,
            'hidden_state': new_hidden_state,
            'context': context,
            'gate_value': gate_value,
            'kc_activity': None,  # Not exposed in current implementation
            'mbon_output': mbon_output,
            'mbon_mean': mbon_mean,
        }

        return outputs

    def reset_context(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset context for new fly/session.

        Parameters
        ----------
        batch_size : int
            Batch size for hidden state
        device : torch.device, optional
            Device for tensors. If None, uses CPU.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Initial hidden state (h_0, c_0) for LSTM
        """
        if device is None:
            device = torch.device('cpu')

        h_0 = torch.zeros(1, batch_size, self.context_dim, device=device)
        c_0 = torch.zeros(1, batch_size, self.context_dim, device=device)

        return (h_0, c_0)

    def freeze_ccbpn_core(self) -> None:
        """Freeze PN→KC→MBON weights (only train context/modulation)."""
        for param in self.ccbpn_core.parameters():
            param.requires_grad = False

    def unfreeze_ccbpn_core(self) -> None:
        """Unfreeze PN→KC→MBON weights (train everything)."""
        for param in self.ccbpn_core.parameters():
            param.requires_grad = True
