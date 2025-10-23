"""Reservoir module approximating mushroom body sparsity."""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - handled gracefully
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


BaseModule = nn.Module if nn is not None else object


@dataclass(frozen=True)
class ReservoirConfig:
    n_pn: int = 50
    n_kc: int = 2000
    n_mbon: int = 10
    kc_sparsity: float = 0.05
    connectivity: str = "hemibrain"


class DrosophilaReservoir(BaseModule):
    """Simple PN→KC→MBON reservoir with configurable sparsity."""

    def __init__(
        self,
        n_pn: int = ReservoirConfig.n_pn,
        n_kc: int = ReservoirConfig.n_kc,
        n_mbon: int = ReservoirConfig.n_mbon,
        kc_sparsity: float = ReservoirConfig.kc_sparsity,
        connectivity: str = ReservoirConfig.connectivity,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError("PyTorch is required to instantiate DrosophilaReservoir.")
        if not 0.0 < kc_sparsity <= 1.0:
            raise ValueError("kc_sparsity must lie within (0, 1].")
        super().__init__()
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.n_mbon = n_mbon
        self.kc_sparsity = kc_sparsity
        self.connectivity = connectivity

        self.pn_to_kc = nn.Linear(n_pn, n_kc, bias=False)
        self.kc_to_mbon = nn.Linear(n_kc, n_mbon, bias=True)

        self._initialise_connectivity()

    def _initialise_connectivity(self) -> None:
        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            weight = torch.randn(self.n_kc, self.n_pn, generator=generator) * 0.1
            mask = torch.rand_like(weight, generator=generator)
            threshold = torch.quantile(mask, 1.0 - self.kc_sparsity)
            mask = (mask >= threshold).float()
            self.pn_to_kc.weight.copy_(weight * mask)
        for param in self.pn_to_kc.parameters():
            param.requires_grad = False

    def forward(self, pn_activity):  # type: ignore[override]
        if pn_activity.dim() == 1:
            pn_activity = pn_activity.unsqueeze(0)
        kc = F.relu(self.pn_to_kc(pn_activity))
        kc = self._enforce_sparsity(kc)
        return F.relu(self.kc_to_mbon(kc))

    def _enforce_sparsity(self, kc_activity: torch.Tensor) -> torch.Tensor:
        if self.kc_sparsity >= 1.0:
            return kc_activity
        k = max(1, int(round(self.n_kc * self.kc_sparsity)))
        _, indices = torch.topk(kc_activity, k=k, dim=-1)
        mask = torch.zeros_like(kc_activity)
        mask.scatter_(-1, indices, 1.0)
        return kc_activity * mask


__all__ = ["DrosophilaReservoir", "ReservoirConfig"]
