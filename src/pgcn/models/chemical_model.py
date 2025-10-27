"""Chemically constrained neural modules for odor generalisation."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - handled gracefully
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


BaseModule = nn.Module if nn is not None else object

from ..chemical.features import CHEMICAL_FEATURE_NAMES, get_chemical_features
from ..chemical.mappings import COMPLETE_ODOR_MAPPINGS
from ..chemical.similarity import compute_chemical_similarity_constraint
from .reservoir import DrosophilaReservoir, ReservoirConfig


class ChemicallyInformedDrosophilaModel(BaseModule):
    """Hybrid model mixing chemical descriptors with a sparse reservoir."""

    def __init__(
        self,
        training_conditions: Optional[Sequence[str]] = None,
        reservoir_config: ReservoirConfig | None = None,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError("PyTorch must be installed to use ChemicallyInformedDrosophilaModel.")
        super().__init__()
        self.training_conditions = (
            [cond.lower() for cond in training_conditions]
            if training_conditions is not None
            else sorted(COMPLETE_ODOR_MAPPINGS)
        )
        self.condition_lookup = {condition: idx for idx, condition in enumerate(self.training_conditions)}

        feature_dim = len(CHEMICAL_FEATURE_NAMES)
        self.chemical_encoder = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.training_encoder = nn.Embedding(len(self.training_conditions), 8)
        self.condition_projector = nn.Linear(8, 32)
        self.interaction_layer = nn.Bilinear(16, 16, 32)

        reservoir_config = reservoir_config or ReservoirConfig()
        self.interaction_projector = nn.Linear(32, reservoir_config.n_pn)
        self.reservoir = DrosophilaReservoir(
            n_pn=reservoir_config.n_pn,
            n_kc=reservoir_config.n_kc,
            n_mbon=reservoir_config.n_mbon,
            kc_sparsity=reservoir_config.kc_sparsity,
            connectivity=reservoir_config.connectivity,
        )
        self.classifier = nn.Linear(reservoir_config.n_mbon, 1)

    def forward(self, training_odor: str, test_odor: str) -> torch.Tensor:  # type: ignore[override]
        train_features = get_chemical_features(training_odor, as_tensor=True).unsqueeze(0)
        test_features = get_chemical_features(test_odor, as_tensor=True).unsqueeze(0)

        train_repr = self.chemical_encoder(train_features)
        test_repr = self.chemical_encoder(test_features)
        condition_index = torch.tensor(
            [self.condition_lookup.get(training_odor.lower(), 0)],
            dtype=torch.long,
            device=train_repr.device,
        )
        condition_embedding = self.training_encoder(condition_index)
        condition_projection = self.condition_projector(condition_embedding)

        interaction = self.interaction_layer(train_repr, test_repr) + condition_projection
        pn_drive = self.interaction_projector(interaction)
        reservoir_out = self.reservoir(pn_drive)
        logits = self.classifier(reservoir_out)
        return torch.sigmoid(logits.squeeze(-1))

    def predict(self, training_odor: str, test_odor: str) -> float:
        with torch.no_grad():
            prediction = self.forward(training_odor, test_odor)
        return float(prediction.squeeze())


class ChemicalSTDP(BaseModule):
    """Dopamine-modulated plasticity scaled by chemical similarity."""

    def __init__(self, kc_dim: int, mbon_dim: int, base_lr: float = 0.01) -> None:
        if torch is None or nn is None:
            raise ImportError("PyTorch must be installed to use ChemicalSTDP.")
        super().__init__()
        if base_lr <= 0:
            raise ValueError("base_lr must be positive.")
        self.base_lr = base_lr
        self.kc_dim = kc_dim
        self.mbon_dim = mbon_dim
        self.register_buffer("zero_trace", torch.zeros(kc_dim, mbon_dim))
        self.eligibility_traces: Dict[Tuple[str, str], torch.Tensor] = {}

    def update_plasticity(
        self,
        training_odor: str,
        test_odor: str,
        *,
        reward: float,
        predicted_response: float,
        kc_activity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KC→MBON weight updates using reward prediction error."""

        similarity = compute_chemical_similarity_constraint(training_odor, test_odor)
        effective_lr = float(self.base_lr * similarity["learning_rate_modifier"])
        expected_response = float(similarity["expected_generalization"])

        reward = float(reward)
        predicted_response = float(predicted_response)
        reward_prediction_error = reward - predicted_response
        baseline_error = reward - expected_response
        combined_error = reward_prediction_error + 0.5 * baseline_error

        if kc_activity.dim() == 1:
            kc_activity = kc_activity.unsqueeze(0)
        avg_activity = kc_activity.mean(dim=0).clamp_min(0.0)
        odor_pair = (training_odor.lower(), test_odor.lower())
        trace = self.eligibility_traces.get(odor_pair)
        if trace is None:
            trace = self.zero_trace.clone()
        if trace.device != avg_activity.device or trace.dtype != avg_activity.dtype:
            trace = trace.to(device=avg_activity.device, dtype=avg_activity.dtype)
        self.eligibility_traces[odor_pair] = trace

        ones = torch.ones(self.mbon_dim, device=avg_activity.device, dtype=trace.dtype)
        avg_activity = avg_activity.to(trace.dtype)

        with torch.no_grad():
            trace.add_(torch.outer(avg_activity, ones))
            lr = trace.new_tensor(effective_lr)
            error_term = trace.new_tensor(combined_error)
            delta_w = -lr * error_term * trace
            delta_w = torch.nan_to_num(delta_w)
        return delta_w.transpose(0, 1).contiguous()


__all__ = ["ChemicallyInformedDrosophilaModel", "ChemicalSTDP"]
