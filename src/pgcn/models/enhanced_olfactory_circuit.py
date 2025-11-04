"""Enhanced olfactory circuit integrating all FlyWire components.

This module provides the master EnhancedOlfactoryCircuit class that combines:
- Core circuit: PN → KC → MBON (with DAN modulation)
- Antennal lobe: PN ↔ LN interactions (GABAergic veto gate)
- Lateral horn: PN → LH → Motor (innate valence)
- Motor output: MBON → DN → Motor (learned responses)
- Behavioral state: AN → MBON/DAN (context modulation)

This integrates all 13,172+ extracted FlyWire neurons into a unified PyTorch model
suitable for blocking experiments and olfactory learning simulations.

Example
-------
>>> from data_loaders.circuit_loader import CircuitLoader
>>> from pgcn.models.enhanced_olfactory_circuit import EnhancedOlfactoryCircuit
>>>
>>> # Load full circuit with extended components
>>> loader = CircuitLoader(cache_dir="data/cache")
>>> conn_matrix = loader.load_connectivity_matrix(
...     normalize_weights="row",
...     include_extended=True
... )
>>>
>>> # Instantiate enhanced circuit
>>> circuit = EnhancedOlfactoryCircuit(
...     connectivity=conn_matrix,
...     kc_sparsity_target=0.05,
...     enable_ln_modulation=True,
...     enable_lh_pathway=True,
...     enable_motor_output=True
... )
>>>
>>> # Run odor presentation with veto blocking
>>> pn_activity = circuit.activate_pns_by_glomeruli(["DA1", "DL3"])
>>> results = circuit.forward_pass_full(
...     pn_activity,
...     blocking_strength=0.8,  # 80% GABAergic veto
...     return_diagnostics=True
... )
>>>
>>> print(f"PER response: {results['per_response']:.3f}")
>>> print(f"Veto strength: {results['diagnostics']['veto_strength'].mean():.3f}")
>>> print(f"KC sparsity: {results['diagnostics']['kc_sparsity']:.2%}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pgcn.models.connectivity_matrix import ConnectivityMatrix
from pgcn.models.enhanced_layers import (
    BrainVNCInterface,
    LateralHornLayer,
    LocalInterneuronLayer,
    MotorSystemLayer,
)
from pgcn.models.olfactory_circuit import OlfactoryCircuit


class EnhancedOlfactoryCircuit(nn.Module):
    """Master integration class for complete Drosophila olfactory learning circuit.

    This class extends the base OlfactoryCircuit with:
    1. Local interneuron (LN) modulation in the antennal lobe
    2. Lateral horn (LH) innate valence pathway
    3. Motor system output (proboscis extension reflex)
    4. Brain-VNC interface (ascending/descending neurons)

    The architecture enables blocking experiments where GABAergic LNs can veto
    PN→KC transmission, testing hypotheses about pathway-specific plasticity gating.

    Parameters
    ----------
    connectivity : ConnectivityMatrix
        Connectivity matrix with extended components loaded (include_extended=True).
    kc_sparsity_target : float, optional
        Target KC sparsity fraction (default: 0.05 = 5% active).
    enable_ln_modulation : bool, optional
        Enable local interneuron modulation (default: True).
    enable_lh_pathway : bool, optional
        Enable lateral horn innate pathway (default: True).
    enable_motor_output : bool, optional
        Enable motor system and PER measurement (default: True).
    enable_vnc_interface : bool, optional
        Enable brain-VNC ascending/descending pathways (default: True).
    gaba_strength : float, optional
        GABAergic LN inhibition strength (default: 1.0).
    chol_strength : float, optional
        Cholinergic LN excitation strength (default: 1.0).

    Attributes
    ----------
    core_circuit : OlfactoryCircuit
        Base PN→KC→MBON forward pass implementation.
    ln_layer : Optional[LocalInterneuronLayer]
        LN modulation layer (if enabled).
    lh_layer : Optional[LateralHornLayer]
        LH innate valence layer (if enabled).
    motor_layer : Optional[MotorSystemLayer]
        Motor output layer (if enabled).
    vnc_interface : Optional[BrainVNCInterface]
        Brain-VNC communication layer (if enabled).
    """

    def __init__(
        self,
        connectivity: ConnectivityMatrix,
        kc_sparsity_target: float = 0.05,
        enable_ln_modulation: bool = True,
        enable_lh_pathway: bool = True,
        enable_motor_output: bool = True,
        enable_vnc_interface: bool = True,
        gaba_strength: float = 1.0,
        chol_strength: float = 1.0,
    ) -> None:
        """Initialize enhanced olfactory circuit with all components."""
        super().__init__()

        self.connectivity = connectivity

        # Core circuit (PN → KC → MBON)
        self.core_circuit = OlfactoryCircuit(
            connectivity=connectivity,
            kc_sparsity_target=kc_sparsity_target,
            kc_sparsity_mode="k_winners_take_all",
        )

        # Local interneuron layer (PN ↔ LN ↔ KC)
        if enable_ln_modulation and connectivity.n_ln > 0:
            self.ln_layer = LocalInterneuronLayer(
                n_pn=connectivity.n_pn,
                n_ln=connectivity.n_ln,
                n_kc=connectivity.n_kc,
                pn_to_ln=connectivity.pn_to_ln,
                ln_to_pn=connectivity.ln_to_pn,
                ln_to_kc=connectivity.ln_to_kc,
                ln_neurotransmitters=connectivity.ln_neurotransmitters,
                gaba_strength=gaba_strength,
                chol_strength=chol_strength,
            )
        else:
            self.ln_layer = None

        # Lateral horn layer (PN → LH → Motor)
        if enable_lh_pathway and connectivity.n_lh > 0:
            self.lh_layer = LateralHornLayer(
                n_pn=connectivity.n_pn,
                n_lh=connectivity.n_lh,
                pn_to_lh=connectivity.pn_to_lh,
                lh_cell_types=connectivity.lh_cell_types,
            )
        else:
            self.lh_layer = None

        # Motor system layer (DN/LH → Motor → PER)
        if enable_motor_output and connectivity.n_motor > 0:
            self.motor_layer = MotorSystemLayer(
                n_motor=connectivity.n_motor,
                n_dn=connectivity.n_dn,
                n_lh=connectivity.n_lh,
                dn_to_motor=connectivity.dn_to_motor,
                lh_to_motor=connectivity.lh_to_motor,
            )
        else:
            self.motor_layer = None

        # Brain-VNC interface (AN ↔ MBON/DAN, MBON → DN)
        if enable_vnc_interface and connectivity.n_an > 0:
            self.vnc_interface = BrainVNCInterface(
                n_an=connectivity.n_an,
                n_dn=connectivity.n_dn,
                n_mbon=connectivity.n_mbon,
                n_dan=connectivity.n_dan,
                an_to_mbon=connectivity.an_to_mbon,
                an_to_dan=connectivity.an_to_dan,
                mbon_to_dn=connectivity.mbon_to_dn,
            )
        else:
            self.vnc_interface = None

    def forward_pass_full(
        self,
        pn_activity: np.ndarray,
        blocking_strength: float = 0.0,
        an_activity: Optional[np.ndarray] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Complete forward pass through all circuit components.

        This orchestrates the full circuit computation:
        1. PN → LN modulation (with optional GABAergic veto)
        2. Modulated PN → KC → MBON (core circuit)
        3. PN → LH → innate valence
        4. AN → MBON/DAN context modulation
        5. MBON → DN → Motor commands
        6. LH + DN → Motor → PER response

        Parameters
        ----------
        pn_activity : np.ndarray
            PN activity vector, shape (n_pn,).
        blocking_strength : float, optional
            GABAergic veto strength (0.0 = none, 1.0 = max). Default: 0.0
        an_activity : Optional[np.ndarray], optional
            Ascending neuron activity for context, shape (n_an,). Default: None
        return_diagnostics : bool, optional
            Return detailed diagnostics. Default: False

        Returns
        -------
        Dict[str, Any]
            Results dictionary with keys:
            - mbon_output : np.ndarray, shape (n_mbon,)
            - per_response : float (if motor enabled)
            - innate_valence : float (if LH enabled)
            - diagnostics : Dict (if return_diagnostics=True)
        """
        # Convert numpy to torch
        pn_activity_torch = torch.from_numpy(pn_activity).float()
        if an_activity is not None:
            an_activity_torch = torch.from_numpy(an_activity).float()
        else:
            an_activity_torch = None

        diagnostics = {}

        # Step 1: LN modulation (if enabled)
        if self.ln_layer is not None:
            ln_activity, modulated_pn_activity, ln_diag = self.ln_layer(
                pn_activity_torch, blocking_strength=blocking_strength
            )
            diagnostics["ln_activity"] = ln_activity.detach().numpy()
            diagnostics["veto_strength"] = ln_diag["veto_strength"].detach().numpy()
            pn_for_kc = modulated_pn_activity.detach().numpy()
        else:
            pn_for_kc = pn_activity

        # Step 2: Core circuit (PN → KC → MBON)
        kc_activity = self.core_circuit.propagate_pn_to_kc(pn_for_kc)
        mbon_output = self.core_circuit.propagate_kc_to_mbon(kc_activity)

        # Compute KC sparsity
        kc_sparsity = self.core_circuit.compute_kc_sparsity_fraction(kc_activity)
        diagnostics["kc_activity"] = kc_activity
        diagnostics["kc_sparsity"] = kc_sparsity

        # Step 3: Lateral horn innate pathway (if enabled)
        innate_valence = None
        lh_activity_np = None
        if self.lh_layer is not None:
            lh_activity, innate_valence_torch, lh_diag = self.lh_layer(
                pn_activity_torch
            )
            innate_valence = innate_valence_torch.detach().numpy()
            lh_activity_np = lh_activity.detach().numpy()
            if isinstance(innate_valence, np.ndarray) and innate_valence.shape == ():
                innate_valence = float(innate_valence)
            diagnostics["lh_activity"] = lh_activity_np
            diagnostics["innate_valence"] = innate_valence

        # Step 4: Brain-VNC interface (if enabled)
        dn_activity_np = None
        if self.vnc_interface is not None:
            mbon_output_torch = torch.from_numpy(mbon_output).float()
            mbon_mod, dan_mod, dn_activity, vnc_diag = self.vnc_interface(
                an_activity_torch, mbon_output_torch
            )
            dn_activity_np = dn_activity.detach().numpy()
            diagnostics["dn_activity"] = dn_activity_np

        # Step 5: Motor output and PER (if enabled)
        per_response = None
        if self.motor_layer is not None:
            dn_activity_torch = (
                torch.from_numpy(dn_activity_np).float()
                if dn_activity_np is not None
                else None
            )
            lh_activity_torch = (
                torch.from_numpy(lh_activity_np).float()
                if lh_activity_np is not None
                else None
            )

            motor_activity, per_response_torch, motor_diag = self.motor_layer(
                dn_activity=dn_activity_torch, lh_activity=lh_activity_torch
            )
            per_response = per_response_torch.detach().numpy()
            if isinstance(per_response, np.ndarray) and per_response.shape == ():
                per_response = float(per_response)
            diagnostics["motor_activity"] = motor_activity.detach().numpy()
            diagnostics["per_response"] = per_response

        results = {
            "mbon_output": mbon_output,
            "per_response": per_response,
            "innate_valence": innate_valence,
        }

        if return_diagnostics:
            results["diagnostics"] = diagnostics

        return results

    def activate_pns_by_glomeruli(
        self, glomeruli: List[str], firing_rate: float = 1.0
    ) -> np.ndarray:
        """Activate PNs by glomerulus names (delegates to core circuit).

        Parameters
        ----------
        glomeruli : List[str]
            Glomerulus names (e.g., ["DA1", "DL3"]).
        firing_rate : float, optional
            PN firing rate. Default: 1.0

        Returns
        -------
        np.ndarray
            PN activity vector, shape (n_pn,).
        """
        return self.core_circuit.activate_pns_by_glomeruli(glomeruli, firing_rate)

    def simulate_odor_response(
        self,
        glomeruli: List[str],
        blocking_strength: float = 0.0,
        return_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Simulate complete odor response including all pathways.

        Parameters
        ----------
        glomeruli : List[str]
            Glomeruli to activate.
        blocking_strength : float, optional
            GABAergic veto strength. Default: 0.0
        return_diagnostics : bool, optional
            Return detailed diagnostics. Default: True

        Returns
        -------
        Dict[str, Any]
            Full circuit response including PER and valence.
        """
        pn_activity = self.activate_pns_by_glomeruli(glomeruli)
        return self.forward_pass_full(
            pn_activity,
            blocking_strength=blocking_strength,
            return_diagnostics=return_diagnostics,
        )

    def measure_blocking_effect(
        self,
        test_glomeruli: List[str],
        blocking_strengths: List[float] = [0.0, 0.5, 1.0],
    ) -> Dict[str, Any]:
        """Measure how blocking strength affects circuit responses.

        Parameters
        ----------
        test_glomeruli : List[str]
            Glomeruli to test.
        blocking_strengths : List[float], optional
            Veto strengths to test. Default: [0.0, 0.5, 1.0]

        Returns
        -------
        Dict[str, Any]
            Blocking curves for PER, MBON, KC sparsity, etc.
        """
        results = {
            "blocking_strengths": blocking_strengths,
            "per_responses": [],
            "mbon_responses": [],
            "kc_sparsities": [],
            "veto_strengths": [],
        }

        pn_activity = self.activate_pns_by_glomeruli(test_glomeruli)

        for strength in blocking_strengths:
            response = self.forward_pass_full(
                pn_activity, blocking_strength=strength, return_diagnostics=True
            )

            results["per_responses"].append(response.get("per_response", 0.0))
            results["mbon_responses"].append(
                float(np.mean(np.abs(response["mbon_output"])))
            )
            results["kc_sparsities"].append(response["diagnostics"]["kc_sparsity"])
            if "veto_strength" in response["diagnostics"]:
                veto = response["diagnostics"]["veto_strength"]
                results["veto_strengths"].append(float(np.mean(veto)))

        return results
