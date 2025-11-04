"""Enhanced neural network layers for extended PGCN components.

This module provides PyTorch-based neural network layers that implement the extended
circuit components required for blocking experiments:

1. LocalInterneuronLayer: GABAergic/Cholinergic processing in antennal lobe
2. LateralHornLayer: Innate valence computation
3. MotorSystemLayer: Proboscis extension reflex (PER) measurement
4. BrainVNCInterface: Behavioral state and motor command integration

All layers are designed to integrate with the existing OlfactoryCircuit while
maintaining biological realism through sparse connectivity and realistic parameters.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class LocalInterneuronLayer(nn.Module):
    """Antennal lobe local interneuron processing layer.

    Biological Context
    ------------------
    Local interneurons (LNs) in the antennal lobe implement lateral interactions
    between glomeruli, providing gain control, normalization, and decorrelation.
    Two main functional types:

    1. GABAergic LNs (inhibitory): Lateral inhibition for odor sharpening
    2. Cholinergic LNs (excitatory): Signal amplification and synchronization

    This layer implements a veto mechanism where GABAergic LNs can suppress
    PN→KC transmission when enhanced, enabling blocking experiments.

    Parameters
    ----------
    n_pn : int
        Number of projection neurons
    n_ln : int
        Number of local interneurons
    n_kc : int
        Number of Kenyon cells
    pn_to_ln : sp.csr_matrix
        PN→LN connectivity (sparse)
    ln_to_pn : Optional[sp.csr_matrix]
        LN→PN feedback connectivity (sparse)
    ln_to_kc : Optional[sp.csr_matrix]
        LN→KC direct veto connectivity (sparse)
    ln_neurotransmitters : Dict[int, str]
        Neurotransmitter types for each LN (GABA vs ACH)
    gaba_strength : float
        Scaling factor for GABAergic inhibition (default: 1.0)
    chol_strength : float
        Scaling factor for cholinergic excitation (default: 1.0)
    """

    def __init__(
        self,
        n_pn: int,
        n_ln: int,
        n_kc: int,
        pn_to_ln: sp.csr_matrix,
        ln_to_pn: Optional[sp.csr_matrix] = None,
        ln_to_kc: Optional[sp.csr_matrix] = None,
        ln_neurotransmitters: Optional[Dict[int, str]] = None,
        gaba_strength: float = 1.0,
        chol_strength: float = 1.0,
    ) -> None:
        super().__init__()

        self.n_pn = n_pn
        self.n_ln = n_ln
        self.n_kc = n_kc
        self.gaba_strength = gaba_strength
        self.chol_strength = chol_strength

        # Convert sparse matrices to PyTorch tensors
        self.pn_to_ln = self._sparse_to_tensor(pn_to_ln)

        if ln_to_pn is not None:
            self.ln_to_pn = self._sparse_to_tensor(ln_to_pn)
        else:
            self.ln_to_pn = None

        if ln_to_kc is not None:
            self.ln_to_kc = self._sparse_to_tensor(ln_to_kc)
        else:
            self.ln_to_kc = None

        # Separate GABAergic and Cholinergic LN masks
        self.gaba_mask = torch.zeros(n_ln, dtype=torch.bool)
        self.chol_mask = torch.zeros(n_ln, dtype=torch.bool)

        if ln_neurotransmitters:
            for idx, (neuron_id, nt_type) in enumerate(ln_neurotransmitters.items()):
                if nt_type == "GABA":
                    self.gaba_mask[idx] = True
                elif nt_type in ["ACH", "CHOL"]:
                    self.chol_mask[idx] = True

        # Learnable gain parameters for each LN type (optional)
        self.gaba_gain = nn.Parameter(torch.tensor(gaba_strength, dtype=torch.float32))
        self.chol_gain = nn.Parameter(torch.tensor(chol_strength, dtype=torch.float32))

    def _sparse_to_tensor(self, sparse_matrix: sp.csr_matrix) -> torch.Tensor:
        """Convert scipy sparse matrix to dense PyTorch tensor."""
        return torch.from_numpy(sparse_matrix.toarray()).float()

    def forward(
        self,
        pn_activity: torch.Tensor,
        blocking_strength: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through local interneuron layer.

        Parameters
        ----------
        pn_activity : torch.Tensor
            PN activity vector, shape (batch_size, n_pn) or (n_pn,)
        blocking_strength : float
            Enhancement factor for GABAergic veto (0.0 = normal, 1.0 = full block)

        Returns
        -------
        ln_activity : torch.Tensor
            LN activity vector, shape (batch_size, n_ln) or (n_ln,)
        modulated_pn_activity : torch.Tensor
            PN activity after LN feedback, shape matches input
        diagnostics : Dict[str, torch.Tensor]
            Diagnostic information (gaba_activity, chol_activity, veto_strength)
        """
        # Ensure pn_activity has batch dimension
        if pn_activity.ndim == 1:
            pn_activity = pn_activity.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = pn_activity.shape[0]

        # PN → LN feedforward
        ln_activity = torch.matmul(pn_activity, self.pn_to_ln.T)  # (batch, n_ln)
        ln_activity = torch.relu(ln_activity)  # Rectify

        # Separate GABAergic and Cholinergic activity
        gaba_activity = ln_activity * self.gaba_mask.float()
        chol_activity = ln_activity * self.chol_mask.float()

        # Apply neurotransmitter-specific gains
        gaba_activity = gaba_activity * self.gaba_gain
        chol_activity = chol_activity * self.chol_gain

        # Apply blocking enhancement to GABAergic LNs
        gaba_activity = gaba_activity * (1.0 + blocking_strength)

        # Recombine
        ln_activity = gaba_activity + chol_activity

        # LN → PN feedback modulation (if connectivity exists)
        if self.ln_to_pn is not None:
            # GABAergic feedback: inhibition (subtract)
            gaba_feedback = torch.matmul(gaba_activity, self.ln_to_pn.T)
            # Cholinergic feedback: excitation (add)
            chol_feedback = torch.matmul(chol_activity, self.ln_to_pn.T)

            modulated_pn_activity = pn_activity + chol_feedback - gaba_feedback
            modulated_pn_activity = torch.clamp(modulated_pn_activity, min=0.0)  # No negative firing
        else:
            modulated_pn_activity = pn_activity

        # Compute veto strength for diagnostics
        if self.ln_to_kc is not None:
            veto_strength = torch.matmul(gaba_activity, self.ln_to_kc.T)  # (batch, n_kc)
        else:
            veto_strength = torch.zeros(batch_size, self.n_kc)

        # Squeeze if input was 1D
        if squeeze_output:
            ln_activity = ln_activity.squeeze(0)
            modulated_pn_activity = modulated_pn_activity.squeeze(0)
            veto_strength = veto_strength.squeeze(0)

        diagnostics = {
            "gaba_activity": gaba_activity,
            "chol_activity": chol_activity,
            "veto_strength": veto_strength,
        }

        return ln_activity, modulated_pn_activity, diagnostics


class LateralHornLayer(nn.Module):
    """Lateral horn innate valence computation layer.

    Biological Context
    ------------------
    The lateral horn (LH) mediates innate behavioral responses to odors, parallel
    to learned responses in the mushroom body. LH neurons receive direct PN input
    and compute hardwired attraction/aversion valence.

    This layer enables separation of innate vs. learned valence during blocking
    experiments, testing predictions about pathway interactions.

    Parameters
    ----------
    n_pn : int
        Number of projection neurons
    n_lh : int
        Number of lateral horn neurons
    pn_to_lh : sp.csr_matrix
        PN→LH connectivity (sparse)
    lh_cell_types : Dict[int, str]
        Cell type annotations (LHLN vs LHCENT)
    attraction_bias : float
        Baseline attraction valence (default: 0.5)
    aversion_bias : float
        Baseline aversion valence (default: -0.5)
    """

    def __init__(
        self,
        n_pn: int,
        n_lh: int,
        pn_to_lh: sp.csr_matrix,
        lh_cell_types: Optional[Dict[int, str]] = None,
        attraction_bias: float = 0.5,
        aversion_bias: float = -0.5,
    ) -> None:
        super().__init__()

        self.n_pn = n_pn
        self.n_lh = n_lh

        # Convert sparse matrix to PyTorch tensor
        self.pn_to_lh = torch.from_numpy(pn_to_lh.toarray()).float()

        # Initialize valence readout weights (learnable)
        self.valence_weights = nn.Parameter(torch.randn(n_lh) * 0.01)

        # Biases for attraction/aversion
        self.attraction_bias = nn.Parameter(torch.tensor(attraction_bias, dtype=torch.float32))
        self.aversion_bias = nn.Parameter(torch.tensor(aversion_bias, dtype=torch.float32))

    def forward(
        self,
        pn_activity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through lateral horn layer.

        Parameters
        ----------
        pn_activity : torch.Tensor
            PN activity vector, shape (batch_size, n_pn) or (n_pn,)

        Returns
        -------
        lh_activity : torch.Tensor
            LH activity vector, shape (batch_size, n_lh) or (n_lh,)
        innate_valence : torch.Tensor
            Innate valence score, shape (batch_size,) or scalar
        diagnostics : Dict[str, torch.Tensor]
            Diagnostic information (attraction_score, aversion_score)
        """
        # Ensure pn_activity has batch dimension
        if pn_activity.ndim == 1:
            pn_activity = pn_activity.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # PN → LH feedforward
        lh_activity = torch.matmul(pn_activity, self.pn_to_lh.T)  # (batch, n_lh)
        lh_activity = torch.relu(lh_activity)  # Rectify

        # Compute innate valence as weighted sum of LH activity
        innate_valence = torch.matmul(lh_activity, self.valence_weights)  # (batch,)

        # Add biases (attraction positive, aversion negative)
        attraction_score = innate_valence + self.attraction_bias
        aversion_score = innate_valence + self.aversion_bias

        # Combined innate valence
        innate_valence = torch.tanh(attraction_score + aversion_score)  # Range [-1, 1]

        # Squeeze if input was 1D
        if squeeze_output:
            lh_activity = lh_activity.squeeze(0)
            innate_valence = innate_valence.squeeze(0)
            attraction_score = attraction_score.squeeze(0)
            aversion_score = aversion_score.squeeze(0)

        diagnostics = {
            "attraction_score": attraction_score,
            "aversion_score": aversion_score,
        }

        return lh_activity, innate_valence, diagnostics


class MotorSystemLayer(nn.Module):
    """Motor system layer for proboscis extension reflex (PER) measurement.

    Biological Context
    ------------------
    Motor neurons execute behavioral outputs. For olfactory learning, proboscis
    extension reflex (PER) is the primary readout:
    - PER magnitude reflects learned + innate odor valence
    - Blocking experiments measure PER to quantify learning deficits

    This layer integrates learned valence (MBON→DN→Motor) and innate valence
    (LH→Motor) to produce a measurable PER response.

    Parameters
    ----------
    n_motor : int
        Number of motor neurons (proboscis-specific)
    n_dn : int
        Number of descending neurons
    n_lh : int
        Number of lateral horn neurons (for innate pathway)
    dn_to_motor : Optional[sp.csr_matrix]
        DN→Motor connectivity (learned pathway)
    lh_to_motor : Optional[sp.csr_matrix]
        LH→Motor connectivity (innate pathway)
    per_threshold : float
        Threshold for PER response (default: 0.5)
    """

    def __init__(
        self,
        n_motor: int,
        n_dn: int = 0,
        n_lh: int = 0,
        dn_to_motor: Optional[sp.csr_matrix] = None,
        lh_to_motor: Optional[sp.csr_matrix] = None,
        per_threshold: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_motor = n_motor
        self.n_dn = n_dn
        self.n_lh = n_lh
        self.per_threshold = per_threshold

        # DN → Motor (learned pathway)
        if dn_to_motor is not None:
            self.dn_to_motor = torch.from_numpy(dn_to_motor.toarray()).float()
        else:
            self.dn_to_motor = torch.zeros(n_motor, n_dn)

        # LH → Motor (innate pathway)
        if lh_to_motor is not None:
            self.lh_to_motor = torch.from_numpy(lh_to_motor.toarray()).float()
        else:
            self.lh_to_motor = torch.zeros(n_motor, n_lh)

        # Learnable integration weights
        self.learned_weight = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))  # Learned pathway
        self.innate_weight = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))  # Innate pathway

    def forward(
        self,
        dn_activity: Optional[torch.Tensor] = None,
        lh_activity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through motor system layer.

        Parameters
        ----------
        dn_activity : Optional[torch.Tensor]
            DN activity from learned pathway, shape (batch_size, n_dn) or (n_dn,)
        lh_activity : Optional[torch.Tensor]
            LH activity from innate pathway, shape (batch_size, n_lh) or (n_lh,)

        Returns
        -------
        motor_activity : torch.Tensor
            Motor neuron activity, shape (batch_size, n_motor) or (n_motor,)
        per_response : torch.Tensor
            PER magnitude [0, 1], shape (batch_size,) or scalar
        diagnostics : Dict[str, torch.Tensor]
            Diagnostic information (learned_contribution, innate_contribution)
        """
        squeeze_output = False

        # Initialize motor activity
        if dn_activity is not None:
            if dn_activity.ndim == 1:
                dn_activity = dn_activity.unsqueeze(0)
                squeeze_output = True
            batch_size = dn_activity.shape[0]
            learned_contribution = torch.matmul(dn_activity, self.dn_to_motor.T)
        else:
            batch_size = 1
            learned_contribution = torch.zeros(batch_size, self.n_motor)

        if lh_activity is not None:
            if lh_activity.ndim == 1:
                lh_activity = lh_activity.unsqueeze(0)
            innate_contribution = torch.matmul(lh_activity, self.lh_to_motor.T)
        else:
            innate_contribution = torch.zeros(batch_size, self.n_motor)

        # Weighted integration
        motor_activity = (
            self.learned_weight * learned_contribution + self.innate_weight * innate_contribution
        )
        motor_activity = torch.relu(motor_activity)  # Rectify

        # Compute PER response (sum motor activity above threshold)
        per_response = torch.sum(motor_activity, dim=-1)  # (batch,)
        per_response = torch.sigmoid(per_response - self.per_threshold)  # Range [0, 1]

        # Squeeze if needed
        if squeeze_output:
            motor_activity = motor_activity.squeeze(0)
            per_response = per_response.squeeze(0)
            learned_contribution = learned_contribution.squeeze(0)
            innate_contribution = innate_contribution.squeeze(0)

        diagnostics = {
            "learned_contribution": learned_contribution,
            "innate_contribution": innate_contribution,
        }

        return motor_activity, per_response, diagnostics


class BrainVNCInterface(nn.Module):
    """Brain-VNC communication interface for behavioral state integration.

    Biological Context
    ------------------
    Ascending neurons (ANs) carry sensory/state information from VNC to brain,
    while descending neurons (DNs) carry motor commands from brain to VNC.
    This bidirectional communication enables:
    - Context-dependent learning modulation (AN→MBON, AN→DAN)
    - Behavioral execution of learned responses (MBON→DN→Motor)

    Parameters
    ----------
    n_an : int
        Number of ascending neurons
    n_dn : int
        Number of descending neurons
    n_mbon : int
        Number of MBONs
    n_dan : int
        Number of DANs
    an_to_mbon : Optional[sp.csr_matrix]
        AN→MBON behavioral state modulation
    an_to_dan : Optional[sp.csr_matrix]
        AN→DAN state-dependent reinforcement
    mbon_to_dn : Optional[sp.csr_matrix]
        MBON→DN learned valence to motor commands
    """

    def __init__(
        self,
        n_an: int,
        n_dn: int,
        n_mbon: int,
        n_dan: int,
        an_to_mbon: Optional[sp.csr_matrix] = None,
        an_to_dan: Optional[sp.csr_matrix] = None,
        mbon_to_dn: Optional[sp.csr_matrix] = None,
    ) -> None:
        super().__init__()

        self.n_an = n_an
        self.n_dn = n_dn
        self.n_mbon = n_mbon
        self.n_dan = n_dan

        # AN → MBON (behavioral state modulation)
        if an_to_mbon is not None:
            self.an_to_mbon = torch.from_numpy(an_to_mbon.toarray()).float()
        else:
            self.an_to_mbon = torch.zeros(n_mbon, n_an)

        # AN → DAN (state-dependent reinforcement)
        if an_to_dan is not None:
            self.an_to_dan = torch.from_numpy(an_to_dan.toarray()).float()
        else:
            self.an_to_dan = torch.zeros(n_dan, n_an)

        # MBON → DN (learned valence to motor commands)
        if mbon_to_dn is not None:
            self.mbon_to_dn = torch.from_numpy(mbon_to_dn.toarray()).float()
        else:
            self.mbon_to_dn = torch.zeros(n_dn, n_mbon)

        # Learnable modulation gains
        self.state_modulation_gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.valence_to_command_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        an_activity: Optional[torch.Tensor] = None,
        mbon_activity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through brain-VNC interface.

        Parameters
        ----------
        an_activity : Optional[torch.Tensor]
            Ascending neuron activity (behavioral state), shape (batch, n_an) or (n_an,)
        mbon_activity : Optional[torch.Tensor]
            MBON activity (learned valence), shape (batch, n_mbon) or (n_mbon,)

        Returns
        -------
        mbon_modulation : torch.Tensor
            State-dependent MBON modulation, shape (batch, n_mbon) or (n_mbon,)
        dan_modulation : torch.Tensor
            State-dependent DAN modulation, shape (batch, n_dan) or (n_dan,)
        dn_activity : torch.Tensor
            Descending neuron activity (motor commands), shape (batch, n_dn) or (n_dn,)
        diagnostics : Dict[str, torch.Tensor]
            Diagnostic information
        """
        squeeze_output = False

        # Process AN activity (ascending: VNC → Brain)
        if an_activity is not None:
            if an_activity.ndim == 1:
                an_activity = an_activity.unsqueeze(0)
                squeeze_output = True
            batch_size = an_activity.shape[0]

            # AN → MBON modulation
            mbon_modulation = torch.matmul(an_activity, self.an_to_mbon.T)
            mbon_modulation = self.state_modulation_gain * mbon_modulation

            # AN → DAN modulation
            dan_modulation = torch.matmul(an_activity, self.an_to_dan.T)
            dan_modulation = self.state_modulation_gain * dan_modulation
        else:
            batch_size = 1
            mbon_modulation = torch.zeros(batch_size, self.n_mbon)
            dan_modulation = torch.zeros(batch_size, self.n_dan)

        # Process MBON activity (descending: Brain → VNC)
        if mbon_activity is not None:
            if mbon_activity.ndim == 1:
                mbon_activity = mbon_activity.unsqueeze(0)

            # MBON → DN conversion (valence to motor commands)
            dn_activity = torch.matmul(mbon_activity, self.mbon_to_dn.T)
            dn_activity = self.valence_to_command_gain * dn_activity
            dn_activity = torch.relu(dn_activity)
        else:
            dn_activity = torch.zeros(batch_size, self.n_dn)

        # Squeeze if needed
        if squeeze_output:
            mbon_modulation = mbon_modulation.squeeze(0)
            dan_modulation = dan_modulation.squeeze(0)
            dn_activity = dn_activity.squeeze(0)

        diagnostics = {
            "state_modulation_strength": self.state_modulation_gain.item(),
            "valence_to_command_strength": self.valence_to_command_gain.item(),
        }

        return mbon_modulation, dan_modulation, dn_activity, diagnostics
