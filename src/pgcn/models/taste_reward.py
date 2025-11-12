"""Simple taste reward circuit - provides dopamine signal for learning.

Based on Shen et al. (2025) Current Biology data.
Extracted from: data/cache/shen2025_appetitive_*.csv

Purpose: Convert sugar input → reward signal (dopamine)
NOT for: Taste learning experiments, GABA testing, benzaldehyde

Architecture:
    Sugar GRNs (90)
        ↓
    ACh-LNs (60) [excitatory relay]
        ↓
    SEZ-PNs (21) [projection]
        ↓
    Reward signal (scalar)

This is a SIMPLIFIED version focused on providing reward signals only.
For GABA veto gate experiments, see taste_circuit.py instead.

Example
-------
>>> from pgcn.models.taste_reward import TasteRewardCircuit
>>> taste = TasteRewardCircuit()
>>> reward = taste(sugar_input=1.0)  # Full sugar reward
>>> print(f"Reward: {reward.item():.3f}")

Reference
---------
Shen, K. et al. (2025). Functional imaging and connectome analyses reveal
organizing principles of taste circuits in Drosophila.
Current Biology, 35(9), 1955-1970.e6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TasteRewardCircuit(nn.Module):
    """Taste circuit that converts sugar stimulus to reward signal.

    This is a simplified version that extracts reward information from
    the taste pathway without modeling GABA inhibition or veto gates.

    Architecture:
        Sugar GRNs (90) → ACh-LNs (60) → SEZ-PNs (21) → Reward

    The circuit processes sugar through gustatory receptor neurons (GRNs),
    relays through cholinergic local neurons (ACh-LNs), and projects via
    SEZ projection neurons (SEZ-PNs) to provide a scalar reward signal.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing extracted taste circuit data. Default: data/cache
    use_synapse_weights : bool, optional
        If True, use actual synapse counts as weights. Default: True

    Attributes
    ----------
    n_grns : int
        Number of gustatory receptor neurons (90)
    n_ach_lns : int
        Number of cholinergic local neurons (60)
    n_sez_pns : int
        Number of SEZ projection neurons (21)
    W_grn_to_ach : torch.Tensor
        GRN → ACh-LN connectivity weights, shape (n_ach_lns, n_grns)
    W_grn_to_pn : torch.Tensor
        GRN → SEZ-PN connectivity weights, shape (n_sez_pns, n_grns)
    W_ach_to_pn : nn.Parameter
        ACh-LN → SEZ-PN connectivity weights, shape (n_sez_pns, n_ach_lns)
        This is learnable since it's not directly in the paper data.
    """

    def __init__(
        self,
        data_dir: Path = Path('data/cache'),
        use_synapse_weights: bool = True
    ) -> None:
        """Initialize taste reward circuit from extracted paper data."""
        super().__init__()

        self.data_dir = data_dir
        self._load_neuron_data()
        self._load_connectivity_matrices(use_synapse_weights)

        # Activation functions
        self.grn_nonlinearity = nn.ReLU()
        self.ln_nonlinearity = nn.ReLU()
        self.pn_nonlinearity = nn.ReLU()

        print("[TasteRewardCircuit] Ready!")

    def _load_neuron_data(self) -> None:
        """Load neuron lists from extracted CSV files."""
        try:
            self.grn_data = pd.read_csv(self.data_dir / "shen2025_appetitive_grn.csv")
            self.sez_pn_data = pd.read_csv(self.data_dir / "shen2025_appetitive_sez_pn.csv")
            self.ach_ln_data = pd.read_csv(self.data_dir / "shen2025_appetitive_sez_ln_ach.csv")

            # Store dimensions
            self.n_grns = len(self.grn_data)
            self.n_ach_lns = len(self.ach_ln_data)
            self.n_sez_pns = len(self.sez_pn_data)

            print(f"[TasteRewardCircuit] Loaded neuron data:")
            print(f"  GRNs: {self.n_grns}")
            print(f"  ACh-LNs: {self.n_ach_lns}")
            print(f"  SEZ-PNs: {self.n_sez_pns}")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not load taste circuit data from {self.data_dir}. "
                f"Run 'python scripts/extract_from_paper_data.py --mode appetitive' first. "
                f"Error: {e}"
            )

    def _load_connectivity_matrices(self, use_synapse_weights: bool) -> None:
        """Load connectivity matrices from extracted NPZ files."""
        try:
            # Load connectivity files
            conn_ach = np.load(self.data_dir / "shen2025_appetitive_connectivity_grn_ach.npz")
            conn_pn = np.load(self.data_dir / "shen2025_appetitive_connectivity_grn_pn.npz")

            # Extract connectivity matrices
            W_grn_ach_raw = conn_ach['connectivity'].astype(np.float32)
            W_grn_pn_raw = conn_pn['connectivity'].astype(np.float32)

            if use_synapse_weights:
                # Use actual synapse counts (normalized to [0, 1])
                W_grn_ach = W_grn_ach_raw / W_grn_ach_raw.max() if W_grn_ach_raw.max() > 0 else W_grn_ach_raw
                W_grn_pn = W_grn_pn_raw / W_grn_pn_raw.max() if W_grn_pn_raw.max() > 0 else W_grn_pn_raw
                print(f"[TasteRewardCircuit] Using synapse-weighted connectivity")
            else:
                # Binary connectivity
                W_grn_ach = (W_grn_ach_raw > 0).astype(np.float32)
                W_grn_pn = (W_grn_pn_raw > 0).astype(np.float32)
                print(f"[TasteRewardCircuit] Using binary connectivity")

            # Register as fixed buffers (connectome-constrained, not learnable)
            self.register_buffer('W_grn_to_ach', torch.from_numpy(W_grn_ach.T))  # (60, 90)
            self.register_buffer('W_grn_to_pn', torch.from_numpy(W_grn_pn.T))    # (21, 90)

            # Count connections
            n_grn_ach = int((W_grn_ach > 0).sum())
            n_grn_pn = int((W_grn_pn > 0).sum())

            print(f"  GRN→ACh connections: {n_grn_ach}")
            print(f"  GRN→PN connections: {n_grn_pn}")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not load connectivity matrices from {self.data_dir}. "
                f"Run 'python scripts/extract_from_paper_data.py --mode appetitive' first. "
                f"Error: {e}"
            )

        # ACh-LN → SEZ-PN weights (not directly in paper data, so learnable)
        # Initialize with small random weights
        W_ach_to_pn = np.random.randn(self.n_sez_pns, self.n_ach_lns) * 0.05
        self.W_ach_to_pn = nn.Parameter(torch.from_numpy(W_ach_to_pn.astype(np.float32)))

    def forward(
        self,
        sugar_input: Union[float, torch.Tensor],
        return_details: bool = False
    ) -> Union[torch.Tensor, dict]:
        """Convert sugar stimulus to reward signal.

        Parameters
        ----------
        sugar_input : float or torch.Tensor
            Sugar stimulus strength. Can be:
            - Scalar float (e.g., 1.0 for full sugar, 0.0 for none)
            - Tensor of shape (batch, n_grns) for explicit GRN activation
            Values should be in [0, 1] range.
        return_details : bool, optional
            If True, return dict with intermediate activations. Default: False

        Returns
        -------
        reward_signal : torch.Tensor
            Scalar reward strength per sample, shape (batch,)
            Range: ~[0, 1] depending on connectivity strength

        OR (if return_details=True):

        dict with keys:
            - 'reward_signal': Reward strength, shape (batch,)
            - 'grn_activity': GRN activation, shape (batch, n_grns)
            - 'ach_ln_activity': ACh-LN activation, shape (batch, n_ach_lns)
            - 'sez_pn_activity': SEZ-PN activation, shape (batch, n_sez_pns)
        """
        # Handle scalar float input (convert to batched tensor)
        if isinstance(sugar_input, (int, float)):
            # Broadcast scalar to all GRNs
            sugar_input = torch.ones(1, self.n_grns) * float(sugar_input)
        elif not isinstance(sugar_input, torch.Tensor):
            sugar_input = torch.tensor(sugar_input, dtype=torch.float32)

        # Ensure 2D (add batch dimension if needed)
        if sugar_input.ndim == 1:
            sugar_input = sugar_input.unsqueeze(0)

        # [1] Sugar GRN activation
        grn_activity = self.grn_nonlinearity(sugar_input)  # (batch, n_grns)

        # [2] GRN → ACh-LN pathway (excitatory relay)
        ach_ln_input = torch.matmul(grn_activity, self.W_grn_to_ach.T)  # (batch, n_ach_lns)
        ach_ln_activity = self.ln_nonlinearity(ach_ln_input)

        # [3] Direct GRN → SEZ-PN pathway
        pn_input_direct = torch.matmul(grn_activity, self.W_grn_to_pn.T)  # (batch, n_sez_pns)

        # [4] Indirect ACh-LN → SEZ-PN pathway
        pn_input_indirect = torch.matmul(ach_ln_activity, self.W_ach_to_pn.T)  # (batch, n_sez_pns)

        # [5] Total SEZ-PN activity (sum pathways)
        pn_input_total = pn_input_direct + pn_input_indirect
        sez_pn_activity = self.pn_nonlinearity(pn_input_total)  # (batch, n_sez_pns)

        # [6] Aggregate to scalar reward signal
        # Simple average across SEZ-PNs
        reward_signal = sez_pn_activity.mean(dim=1)  # (batch,)

        if return_details:
            return {
                'reward_signal': reward_signal,
                'grn_activity': grn_activity,
                'ach_ln_activity': ach_ln_activity,
                'sez_pn_activity': sez_pn_activity,
            }
        else:
            return reward_signal

    def compute_dopamine(
        self,
        sugar_input: Union[float, torch.Tensor],
        predicted_reward: torch.Tensor
    ) -> torch.Tensor:
        """Compute dopamine signal (reward prediction error).

        This is the learning signal that gates KC→MBON plasticity.

        Parameters
        ----------
        sugar_input : float or torch.Tensor
            Actual sugar stimulus present
        predicted_reward : torch.Tensor
            Model's current prediction of reward, shape (batch,)

        Returns
        -------
        dopamine : torch.Tensor
            Reward prediction error (RPE) signal, shape (batch,)
            Positive = better than expected (increase weights)
            Negative = worse than expected (decrease weights)
            Range: typically [-1, 1]
        """
        # Get actual reward from taste circuit
        actual_reward = self.forward(sugar_input)

        # Compute prediction error
        rpe = actual_reward - predicted_reward

        # Clip to reasonable range for stability
        dopamine = torch.clamp(rpe, -1.0, 1.0)

        return dopamine

    def get_statistics(self) -> dict:
        """Get statistics about the taste reward circuit.

        Returns
        -------
        dict
            Statistics including neuron counts, connection counts, and weights.
        """
        stats = {
            'n_grns': self.n_grns,
            'n_ach_lns': self.n_ach_lns,
            'n_sez_pns': self.n_sez_pns,
            'grn_to_ach_connections': int((self.W_grn_to_ach > 0).sum()),
            'grn_to_pn_connections': int((self.W_grn_to_pn > 0).sum()),
            'mean_grn_to_ach_weight': float(
                self.W_grn_to_ach[self.W_grn_to_ach > 0].mean()
                if (self.W_grn_to_ach > 0).any() else 0
            ),
            'mean_grn_to_pn_weight': float(
                self.W_grn_to_pn[self.W_grn_to_pn > 0].mean()
                if (self.W_grn_to_pn > 0).any() else 0
            ),
        }
        return stats
