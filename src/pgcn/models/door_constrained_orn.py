"""DoOR-Constrained ORN Layer - Biologically Accurate Olfactory Input Layer.

This module implements a neural network layer for ORN responses constrained by
DoOR database receptor-odorant response profiles, enabling biologically realistic
odor encoding for the PGCN framework.

The layer implements:
- DoOR-based receptor response profiles
- Concentration-response curves (Hill equation)
- Odor mixture interactions (competitive inhibition)
- Biological noise for realistic neural dynamics

Example
-------
>>> import torch
>>> from pgcn.door import DoORDataManager
>>> from pgcn.models.door_constrained_orn import DoORConstrainedORNLayer
>>>
>>> # Initialize DoOR data
>>> door_manager = DoORDataManager()
>>> door_data = door_manager.load_door_data()
>>>
>>> # Create ORN layer
>>> orn_layer = DoORConstrainedORNLayer(
...     door_data=door_data,
...     concentration_response=True,
...     noise_level=0.1
... )
>>>
>>> # Encode odor mixture
>>> response = orn_layer.encode_odor_mixture(
...     ['ethyl acetate', 'hexanol'],
...     concentrations=[0.5, 0.3]
... )
>>> print(f"ORN population response: {response.shape}")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# PyTorch imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")


if TORCH_AVAILABLE:
    class DoORConstrainedORNLayer(nn.Module):
        """Biologically accurate ORN layer using DoOR response profiles.

        This neural network module implements realistic ORN encoding based on
        DoOR database receptor-odorant response profiles, with support for:
        - Concentration-dependent responses (Hill equation)
        - Odor mixture interactions
        - Biological noise
        - Sparse receptor expression patterns

        Biological Context
        ------------------
        ORNs transduce chemical odorants into neural signals through olfactory
        receptors. Each ORN expresses typically 1-2 receptor types, creating a
        combinatorial code where odor identity is encoded by the pattern of
        activated ORNs.

        The DoOR database provides normalized response strengths for 693 odorants
        across 78 receptor types, enabling biologically constrained simulations.

        Concentration-Response Dynamics
        --------------------------------
        Real ORNs show dose-dependent responses following Hill kinetics:

        R(C) = R_max * (C^n / (EC50^n + C^n))

        where:
        - C: odorant concentration
        - R_max: maximal response (from DoOR)
        - EC50: half-maximal concentration
        - n: Hill coefficient (cooperativity)

        Parameters
        ----------
        door_data : Dict[str, pd.DataFrame]
            Complete DoOR database from DoORDataManager
        concentration_response : bool, optional
            Enable concentration-dependent Hill equation. Default: True
        noise_level : float, optional
            Gaussian noise standard deviation for biological realism. Default: 0.1
        device : str, optional
            PyTorch device ('cpu' or 'cuda'). Default: 'cpu'

        Attributes
        ----------
        response_matrix : torch.Tensor
            DoOR response matrix (odorants × receptors)
        odorant_names : List[str]
            Ordered list of odorant names
        receptor_names : List[str]
            Ordered list of receptor names
        hill_coefficients : nn.Parameter
            Learnable Hill coefficients for each receptor
        ec50_values : nn.Parameter
            Learnable EC50 values for each receptor

        Examples
        --------
        >>> # Create ORN layer
        >>> orn_layer = DoORConstrainedORNLayer(door_data)
        >>>
        >>> # Encode single odorant
        >>> response = orn_layer.encode_odor_mixture(['ethyl acetate'])
        >>> print(f"Active receptors: {(response > 0.1).sum().item()}")
        >>>
        >>> # Encode odor mixture with concentrations
        >>> mixture_response = orn_layer.encode_odor_mixture(
        ...     ['ethyl acetate', 'hexanol'],
        ...     concentrations=[0.8, 0.5]
        ... )
        """

        def __init__(
            self,
            door_data: Dict[str, pd.DataFrame],
            concentration_response: bool = True,
            noise_level: float = 0.1,
            device: str = 'cpu'
        ):
            """Initialize DoOR-constrained ORN layer."""
            super().__init__()

            self.device = device
            self.concentration_response = concentration_response
            self.noise_level = noise_level

            # Extract response matrix from DoOR data
            response_matrix = door_data['response_matrix']

            # Store odorant and receptor names
            self.odorant_names = response_matrix.index.tolist()
            self.receptor_names = response_matrix.columns.tolist()

            # Convert to tensor
            response_tensor = torch.tensor(
                response_matrix.values,
                dtype=torch.float32,
                device=device
            )

            # Handle NaN values (replace with 0)
            response_tensor = torch.nan_to_num(response_tensor, nan=0.0)

            # Register as buffer (not trainable)
            self.register_buffer('response_matrix', response_tensor)

            # Initialize concentration-response parameters if enabled
            if concentration_response:
                # Hill coefficients (typically 1-4 for olfactory receptors)
                self.hill_coefficients = nn.Parameter(
                    torch.ones(len(self.receptor_names), device=device) * 2.0
                )

                # EC50 values (half-maximal concentration, typically 10^-6 to 10^-3 M)
                self.ec50_values = nn.Parameter(
                    torch.ones(len(self.receptor_names), device=device) * 0.5
                )

            # Create odorant name to index mapping
            self.odorant_to_idx = {
                name: idx for idx, name in enumerate(self.odorant_names)
            }

        def encode_odor_mixture(
            self,
            odorant_list: List[str],
            concentrations: Optional[List[float]] = None
        ) -> torch.Tensor:
            """Encode complex odor mixtures with realistic interactions.

            Parameters
            ----------
            odorant_list : List[str]
                List of odorant names to mix
            concentrations : List[float], optional
                Concentration of each odorant (0-1). Default: all 1.0

            Returns
            -------
            torch.Tensor
                ORN population response vector (n_receptors,)

            Examples
            --------
            >>> # Single odorant
            >>> response = orn_layer.encode_odor_mixture(['ethyl acetate'])
            >>>
            >>> # Mixture with concentrations
            >>> response = orn_layer.encode_odor_mixture(
            ...     ['ethyl acetate', 'hexanol'],
            ...     concentrations=[0.8, 0.3]
            ... )
            """
            if concentrations is None:
                concentrations = [1.0] * len(odorant_list)

            # Initialize mixture response
            mixture_response = torch.zeros(
                len(self.receptor_names),
                device=self.device
            )

            # Process each odorant in mixture
            for odorant, conc in zip(odorant_list, concentrations):
                if odorant in self.odorant_to_idx:
                    odorant_idx = self.odorant_to_idx[odorant]

                    # Get base response from DoOR
                    base_response = self.response_matrix[odorant_idx]

                    # Apply concentration-response curve if enabled
                    if self.concentration_response:
                        conc_response = self._apply_hill_equation(base_response, conc)
                    else:
                        conc_response = base_response * conc

                    # Model mixture interactions (competitive inhibition)
                    # Each new odorant reduces sensitivity to subsequent ones
                    mixture_response = mixture_response + conc_response / (
                        1.0 + 0.1 * mixture_response
                    )

            # Add biological noise if in training mode
            if self.training and self.noise_level > 0:
                noise = torch.randn_like(mixture_response) * self.noise_level
                mixture_response = mixture_response + noise

            # Clamp to biologically realistic range [0, 1]
            mixture_response = torch.clamp(mixture_response, 0.0, 1.0)

            return mixture_response

        def _apply_hill_equation(
            self,
            base_response: torch.Tensor,
            concentration: float
        ) -> torch.Tensor:
            """Apply Hill equation for realistic concentration-response curves.

            Hill equation: R(C) = R_max * (C^n / (EC50^n + C^n))

            Parameters
            ----------
            base_response : torch.Tensor
                Maximal response (R_max) from DoOR
            concentration : float
                Odorant concentration (0-1)

            Returns
            -------
            torch.Tensor
                Concentration-adjusted response
            """
            # Extract Hill parameters
            n = self.hill_coefficients
            ec50 = self.ec50_values

            # Hill equation
            conc_n = concentration ** n
            ec50_n = ec50 ** n

            # Avoid division by zero
            response = base_response * (conc_n / (ec50_n + conc_n + 1e-6))

            return response

        def get_receptor_response(
            self,
            receptor_name: str,
            odorant_name: str,
            concentration: float = 1.0
        ) -> float:
            """Get response of specific receptor to specific odorant.

            Parameters
            ----------
            receptor_name : str
                Receptor name (e.g., 'Or42b')
            odorant_name : str
                Odorant name
            concentration : float, optional
                Concentration (0-1). Default: 1.0

            Returns
            -------
            float
                Receptor response strength

            Examples
            --------
            >>> response = orn_layer.get_receptor_response('Or42b', 'ethyl hexanoate')
            >>> print(f"Or42b response: {response:.3f}")
            """
            if receptor_name not in self.receptor_names:
                raise ValueError(f"Receptor '{receptor_name}' not found")

            if odorant_name not in self.odorant_to_idx:
                raise ValueError(f"Odorant '{odorant_name}' not found")

            receptor_idx = self.receptor_names.index(receptor_name)
            odorant_idx = self.odorant_to_idx[odorant_name]

            # Get base response
            base_response = self.response_matrix[odorant_idx, receptor_idx]

            # Apply Hill equation if enabled
            if self.concentration_response:
                response = self._apply_hill_equation(
                    base_response.unsqueeze(0),
                    concentration
                )[0]
            else:
                response = base_response * concentration

            return response.item()

        def forward(self, odor_input: torch.Tensor) -> torch.Tensor:
            """Forward pass for neural network integration.

            Parameters
            ----------
            odor_input : torch.Tensor
                Input tensor (batch_size, n_odorants) or (n_odorants,)

            Returns
            -------
            torch.Tensor
                ORN response (batch_size, n_receptors) or (n_receptors,)
            """
            # Handle batch dimension
            if odor_input.dim() == 1:
                odor_input = odor_input.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False

            # Matrix multiplication: (batch, odorants) @ (odorants, receptors)
            orn_response = torch.matmul(odor_input, self.response_matrix)

            # Add noise if training
            if self.training and self.noise_level > 0:
                noise = torch.randn_like(orn_response) * self.noise_level
                orn_response = orn_response + noise

            # Clamp to [0, 1]
            orn_response = torch.clamp(orn_response, 0.0, 1.0)

            if squeeze_output:
                orn_response = orn_response.squeeze(0)

            return orn_response

else:
    # Fallback implementation without PyTorch
    class DoORConstrainedORNLayer:
        """NumPy-based DoOR-constrained ORN layer (PyTorch not available)."""

        def __init__(
            self,
            door_data: Dict[str, pd.DataFrame],
            concentration_response: bool = True,
            noise_level: float = 0.1,
            device: str = 'cpu'
        ):
            """Initialize ORN layer with NumPy backend."""
            print("Warning: Using NumPy backend (PyTorch not available)")

            self.concentration_response = concentration_response
            self.noise_level = noise_level

            # Extract response matrix
            response_matrix = door_data['response_matrix']
            self.odorant_names = response_matrix.index.tolist()
            self.receptor_names = response_matrix.columns.tolist()

            # Convert to NumPy
            self.response_matrix = response_matrix.values
            self.response_matrix = np.nan_to_num(self.response_matrix, nan=0.0)

            # Initialize Hill parameters
            if concentration_response:
                self.hill_coefficients = np.ones(len(self.receptor_names)) * 2.0
                self.ec50_values = np.ones(len(self.receptor_names)) * 0.5

            self.odorant_to_idx = {
                name: idx for idx, name in enumerate(self.odorant_names)
            }

        def encode_odor_mixture(
            self,
            odorant_list: List[str],
            concentrations: Optional[List[float]] = None
        ) -> np.ndarray:
            """Encode odor mixture (NumPy implementation)."""
            if concentrations is None:
                concentrations = [1.0] * len(odorant_list)

            mixture_response = np.zeros(len(self.receptor_names))

            for odorant, conc in zip(odorant_list, concentrations):
                if odorant in self.odorant_to_idx:
                    odorant_idx = self.odorant_to_idx[odorant]
                    base_response = self.response_matrix[odorant_idx]

                    if self.concentration_response:
                        conc_response = self._apply_hill_equation_numpy(base_response, conc)
                    else:
                        conc_response = base_response * conc

                    mixture_response = mixture_response + conc_response / (1.0 + 0.1 * mixture_response)

            if self.noise_level > 0:
                noise = np.random.randn(len(self.receptor_names)) * self.noise_level
                mixture_response = mixture_response + noise

            return np.clip(mixture_response, 0.0, 1.0)

        def _apply_hill_equation_numpy(
            self,
            base_response: np.ndarray,
            concentration: float
        ) -> np.ndarray:
            """Apply Hill equation (NumPy)."""
            n = self.hill_coefficients
            ec50 = self.ec50_values

            conc_n = concentration ** n
            ec50_n = ec50 ** n

            response = base_response * (conc_n / (ec50_n + conc_n + 1e-6))

            return response
