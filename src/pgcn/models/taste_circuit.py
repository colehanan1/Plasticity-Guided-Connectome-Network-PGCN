"""Taste processing circuit with dual excitatory/inhibitory pathways.

This module implements the gustatory (taste) processing circuit based on
Shen et al. (2025) paper data, with explicit excitatory (ACh) and inhibitory
(GABA) pathways that create a veto gate mechanism.

Architecture:
    Sugar GRNs (90)
        ├→ ACh-LNs (60) → SEZ-PNs (21) [EXCITATORY PATH]
        └→ GABA-LNs (36) → Veto signal [INHIBITORY PATH]

The GABA veto gate can operate in three modes:
1. 'direct': GABA-LN → SEZ-PN (inhibit projection neurons)
2. 'feedforward': GABA-LN → ACh-LN (inhibit relay neurons)
3. 'neuromod': GABA creates scalar veto signal (for dopamine modulation)

Reference:
    Shen, K. et al. (2025). Functional imaging and connectome analyses reveal
    organizing principles of taste circuits in Drosophila.
    Current Biology, 35(9), 1955-1970.e6.

Example:
    >>> from pathlib import Path
    >>> taste = TasteCircuit(
    ...     data_dir=Path('data/cache'),
    ...     gaba_veto_mode='direct',
    ...     gaba_gain=1.0
    ... )
    >>> sugar_input = torch.ones(1, 90) * 0.5
    >>> sez_pn, ach_ln, gaba_ln, veto = taste(sugar_input)
    >>> print(f"Veto signal: {veto.item():.3f}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TasteCircuit(nn.Module):
    """Taste processing circuit with dual excitatory/inhibitory pathways.

    This module implements the gustatory (taste) circuit with explicit
    ACh (excitatory) and GABA (inhibitory) pathways that create a
    veto gate mechanism for contextual control of reward signaling.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing extracted taste circuit data from Shen et al. (2025).
        Default: Path('data/cache')
    gaba_veto_mode : str, optional
        GABA inhibition mode:
        - 'direct': GABA-LN → SEZ-PN (inhibit projection neurons)
        - 'feedforward': GABA-LN → ACh-LN (inhibit relay neurons)
        - 'neuromod': GABA creates scalar veto signal
        Default: 'direct'
    gaba_gain : float, optional
        Scaling factor for GABA inhibition strength (learnable parameter).
        Default: 1.0
    use_synapse_weights : bool, optional
        If True, use actual synapse counts as weights.
        If False, use binary connectivity.
        Default: True

    Attributes
    ----------
    n_grns : int
        Number of gustatory receptor neurons (~90 for sugar/water)
    n_sez_pns : int
        Number of SEZ projection neurons (~21)
    n_ach_lns : int
        Number of cholinergic local neurons (~60)
    n_gaba_lns : int
        Number of GABAergic local neurons (~36)
    W_grn_to_pn : torch.Tensor
        GRN → SEZ-PN connectivity (n_sez_pns × n_grns)
    W_grn_to_ach : torch.Tensor
        GRN → ACh-LN connectivity (n_ach_lns × n_grns)
    W_grn_to_gaba : torch.Tensor
        GRN → GABA-LN connectivity (n_gaba_lns × n_grns)
    """

    def __init__(
        self,
        data_dir: Path = Path("data/cache"),
        gaba_veto_mode: str = "direct",
        gaba_gain: float = 1.0,
        use_synapse_weights: bool = True,
    ) -> None:
        """Initialize taste circuit from extracted paper data."""
        super().__init__()

        # Load extracted data
        self.data_dir = data_dir
        self._load_neuron_data()
        self._load_connectivity_matrices(use_synapse_weights)

        # GABA veto configuration
        if gaba_veto_mode not in ["direct", "feedforward", "neuromod"]:
            raise ValueError(
                f"Invalid gaba_veto_mode: {gaba_veto_mode}. "
                f"Must be 'direct', 'feedforward', or 'neuromod'"
            )
        self.gaba_veto_mode = gaba_veto_mode

        # Learnable GABA gain parameter
        self.gaba_gain = nn.Parameter(torch.tensor(gaba_gain, dtype=torch.float32))

        # Initialize GABA output weights (mode-dependent)
        self._init_gaba_outputs()

        # Activation functions
        self.grn_nonlinearity = nn.ReLU()
        self.ln_nonlinearity = nn.ReLU()
        self.pn_nonlinearity = nn.ReLU()

    def _load_neuron_data(self) -> None:
        """Load neuron lists from extracted CSV files."""
        try:
            self.grn_data = pd.read_csv(self.data_dir / "shen2025_appetitive_grn.csv")
            self.sez_pn_data = pd.read_csv(self.data_dir / "shen2025_appetitive_sez_pn.csv")
            self.ach_ln_data = pd.read_csv(self.data_dir / "shen2025_appetitive_sez_ln_ach.csv")
            self.gaba_ln_data = pd.read_csv(self.data_dir / "shen2025_appetitive_sez_ln_gaba.csv")

            # Store dimensions
            self.n_grns = len(self.grn_data)
            self.n_sez_pns = len(self.sez_pn_data)
            self.n_ach_lns = len(self.ach_ln_data)
            self.n_gaba_lns = len(self.gaba_ln_data)

            print(f"[TasteCircuit] Loaded neuron data:")
            print(f"  GRNs: {self.n_grns}")
            print(f"  SEZ-PNs: {self.n_sez_pns}")
            print(f"  ACh-LNs: {self.n_ach_lns}")
            print(f"  GABA-LNs: {self.n_gaba_lns}")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not load taste circuit data from {self.data_dir}. "
                f"Run 'python scripts/extract_from_paper_data.py --mode appetitive' first. "
                f"Error: {e}"
            )

    def _load_connectivity_matrices(self, use_synapse_weights: bool) -> None:
        """Load connectivity matrices from extracted NPZ files.

        Parameters
        ----------
        use_synapse_weights : bool
            If True, normalize synapse counts to [0, 1] range.
            If False, binarize to 0/1.
        """
        try:
            conn_pn = np.load(self.data_dir / "shen2025_appetitive_connectivity_grn_pn.npz")
            conn_ach = np.load(self.data_dir / "shen2025_appetitive_connectivity_grn_ach.npz")
            conn_gaba = np.load(self.data_dir / "shen2025_appetitive_connectivity_grn_gaba.npz")

            # Extract connectivity matrices
            W_grn_pn = conn_pn["connectivity"].astype(np.float32)
            W_grn_ach = conn_ach["connectivity"].astype(np.float32)
            W_grn_gaba = conn_gaba["connectivity"].astype(np.float32)

            if use_synapse_weights:
                # Normalize by max synapses (preserves relative strengths)
                if W_grn_pn.max() > 0:
                    W_grn_pn = W_grn_pn / W_grn_pn.max()
                if W_grn_ach.max() > 0:
                    W_grn_ach = W_grn_ach / W_grn_ach.max()
                if W_grn_gaba.max() > 0:
                    W_grn_gaba = W_grn_gaba / W_grn_gaba.max()

                print(f"[TasteCircuit] Using synapse-weighted connectivity")
                print(f"  GRN→PN: {(W_grn_pn > 0).sum()} connections")
                print(f"  GRN→ACh: {(W_grn_ach > 0).sum()} connections")
                print(f"  GRN→GABA: {(W_grn_gaba > 0).sum()} connections")
            else:
                # Binarize connectivity
                W_grn_pn = (W_grn_pn > 0).astype(np.float32)
                W_grn_ach = (W_grn_ach > 0).astype(np.float32)
                W_grn_gaba = (W_grn_gaba > 0).astype(np.float32)
                print(f"[TasteCircuit] Using binary connectivity")

            # Register as fixed buffers (connectome-constrained, not trainable)
            # Shape: (n_postsynaptic, n_grns) for efficient matrix multiplication
            self.register_buffer("W_grn_to_pn", torch.from_numpy(W_grn_pn.T))
            self.register_buffer("W_grn_to_ach", torch.from_numpy(W_grn_ach.T))
            self.register_buffer("W_grn_to_gaba", torch.from_numpy(W_grn_gaba.T))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not load connectivity matrices from {self.data_dir}. "
                f"Run 'python scripts/extract_from_paper_data.py --mode appetitive' first. "
                f"Error: {e}"
            )

    def _init_gaba_outputs(self) -> None:
        """Initialize GABA output weights based on veto mode.

        For 'direct' mode: GABA-LN → SEZ-PN
        For 'feedforward' mode: GABA-LN → ACh-LN
        For 'neuromod' mode: No explicit weights (scalar output)
        """
        if self.gaba_veto_mode == "direct":
            # GABA-LN → SEZ-PN (inhibit projection neurons)
            W_gaba_to_pn = np.random.randn(self.n_sez_pns, self.n_gaba_lns) * 0.1
            self.register_buffer("W_gaba_to_pn", torch.from_numpy(W_gaba_to_pn.astype(np.float32)))
            print(f"[TasteCircuit] GABA veto mode: direct (GABA→SEZ-PN)")

        elif self.gaba_veto_mode == "feedforward":
            # GABA-LN → ACh-LN (inhibit relay neurons)
            W_gaba_to_ach = np.random.randn(self.n_ach_lns, self.n_gaba_lns) * 0.1
            self.register_buffer(
                "W_gaba_to_ach", torch.from_numpy(W_gaba_to_ach.astype(np.float32))
            )
            print(f"[TasteCircuit] GABA veto mode: feedforward (GABA→ACh-LN)")

        else:  # 'neuromod'
            print(f"[TasteCircuit] GABA veto mode: neuromod (scalar signal)")

        # ACh-LN → SEZ-PN connectivity (learned or random)
        # TODO: Extract from paper data if available
        W_ach_to_pn = np.random.randn(self.n_sez_pns, self.n_ach_lns) * 0.1
        self.register_buffer("W_ach_to_pn", torch.from_numpy(W_ach_to_pn.astype(np.float32)))

    def forward(
        self, sugar_input, odor_context: Optional[torch.Tensor] = None
    ) -> dict:
        """Forward pass through taste circuit.

        Parameters
        ----------
        sugar_input : float or torch.Tensor
            Sugar reward strength (scalar) or GRN activation (batch, n_grns).
            Scalar values will be broadcast to all GRNs.
            Values should be in [0, 1] range.
        odor_context : torch.Tensor, optional
            Optional odor context for gating, shape (batch, context_dim)
            Currently unused but reserved for future context-dependent gating

        Returns
        -------
        dict
            Dictionary with keys:
            - 'sez_pn_activity': SEZ-PN projection neuron activity, shape (batch, n_sez_pns)
            - 'ach_ln_activity': ACh-LN excitatory relay activity, shape (batch, n_ach_lns)
            - 'gaba_ln_activity': GABA-LN inhibitory gate activity, shape (batch, n_gaba_lns)
            - 'veto_signal': Scalar veto strength per sample, shape (batch,)
            - 'grn_activity': GRN activity, shape (batch, n_grns)

        Notes
        -----
        The veto signal strength increases with GABA-LN activity and is scaled
        by the learnable gaba_gain parameter. High veto signal suppresses
        downstream reward processing.
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

        batch_size = sugar_input.shape[0]

        # [1] Sugar GRN activation
        grn_activity = self.grn_nonlinearity(sugar_input)  # (batch, n_grns)

        # [2] Excitatory pathway: GRN → ACh-LN
        ach_ln_input = torch.matmul(grn_activity, self.W_grn_to_ach.T)  # (batch, n_ach_lns)
        ach_ln_activity = self.ln_nonlinearity(ach_ln_input)

        # [3] Inhibitory pathway: GRN → GABA-LN
        gaba_ln_input = torch.matmul(grn_activity, self.W_grn_to_gaba.T)  # (batch, n_gaba_lns)
        gaba_ln_activity = self.ln_nonlinearity(gaba_ln_input)

        # [4] Direct GRN → SEZ-PN pathway
        pn_input_direct = torch.matmul(grn_activity, self.W_grn_to_pn.T)  # (batch, n_sez_pns)

        # [5] Indirect ACh-LN → SEZ-PN pathway (excitatory relay)
        pn_input_ach = torch.matmul(ach_ln_activity, self.W_ach_to_pn.T)  # (batch, n_sez_pns)

        # [6] GABA veto gate (mode-dependent)
        if self.gaba_veto_mode == "direct":
            # GABA-LN directly inhibits SEZ-PNs
            pn_input_gaba = torch.matmul(gaba_ln_activity, self.W_gaba_to_pn.T)  # (batch, n_sez_pns)
            pn_input_total = pn_input_direct + pn_input_ach - self.gaba_gain * pn_input_gaba

        elif self.gaba_veto_mode == "feedforward":
            # GABA-LN inhibits ACh-LNs (feed-forward inhibition)
            ach_inhibition = torch.matmul(
                gaba_ln_activity, self.W_gaba_to_ach.T
            )  # (batch, n_ach_lns)
            ach_ln_gated = torch.relu(ach_ln_activity - self.gaba_gain * ach_inhibition)
            pn_input_ach_gated = torch.matmul(ach_ln_gated, self.W_ach_to_pn.T)
            pn_input_total = pn_input_direct + pn_input_ach_gated

        else:  # 'neuromod'
            # GABA creates a scalar veto signal (for neuromodulation elsewhere)
            pn_input_total = pn_input_direct + pn_input_ach

        # [7] SEZ-PN output
        sez_pn_output = self.pn_nonlinearity(pn_input_total)  # (batch, n_sez_pns)

        # [8] Compute veto strength (for analysis and neuromodulation)
        # Higher GABA activity → stronger veto
        # Normalize by mean activation to get veto in [0, 1] range
        gaba_mean_activity = gaba_ln_activity.mean(dim=1)  # (batch,)
        veto_signal = torch.sigmoid(self.gaba_gain * gaba_mean_activity)  # (batch,)

        return {
            'sez_pn_activity': sez_pn_output,
            'ach_ln_activity': ach_ln_activity,
            'gaba_ln_activity': gaba_ln_activity,
            'veto_signal': veto_signal,
            'grn_activity': grn_activity,
        }

    def get_synapse_statistics(self) -> dict:
        """Get statistics about connectivity and synapse weights.

        Returns
        -------
        dict
            Dictionary containing:
            - 'grn_to_pn_connections': Number of GRN→PN connections
            - 'grn_to_ach_connections': Number of GRN→ACh connections
            - 'grn_to_gaba_connections': Number of GRN→GABA connections
            - 'grn_to_pn_mean_weight': Mean weight of GRN→PN connections
            - 'grn_to_ach_mean_weight': Mean weight of GRN→ACh connections
            - 'grn_to_gaba_mean_weight': Mean weight of GRN→GABA connections
        """
        stats = {
            "grn_to_pn_connections": int((self.W_grn_to_pn > 0).sum()),
            "grn_to_ach_connections": int((self.W_grn_to_ach > 0).sum()),
            "grn_to_gaba_connections": int((self.W_grn_to_gaba > 0).sum()),
            "grn_to_pn_mean_weight": float(
                self.W_grn_to_pn[self.W_grn_to_pn > 0].mean() if (self.W_grn_to_pn > 0).any() else 0
            ),
            "grn_to_ach_mean_weight": float(
                self.W_grn_to_ach[self.W_grn_to_ach > 0].mean()
                if (self.W_grn_to_ach > 0).any()
                else 0
            ),
            "grn_to_gaba_mean_weight": float(
                self.W_grn_to_gaba[self.W_grn_to_gaba > 0].mean()
                if (self.W_grn_to_gaba > 0).any()
                else 0
            ),
            "gaba_gain": float(self.gaba_gain.item()),
            "gaba_veto_mode": self.gaba_veto_mode,
        }
        return stats
