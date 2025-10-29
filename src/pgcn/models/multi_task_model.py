"""Multi-task extensions that wrap the biologically-constrained reservoir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .reservoir import DrosophilaReservoir

__all__ = ["MultiTaskDrosophilaModel", "TaskHeadConfig", "validate_biological_constraints"]


@dataclass(frozen=True)
class TaskHeadConfig:
    """Configuration metadata for a task specific readout head."""

    output_dim: int
    activation: Optional[str] = None
    loss: Optional[str] = None
    dropout: float = 0.0
    use_reservoir_head: bool = False


def _ensure_2d(tensor: Tensor) -> Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise ValueError("PN activity tensors must be one or two dimensional.")
    return tensor


def _apply_activation(tensor: Tensor, activation: Optional[str]) -> Tensor:
    if activation is None:
        return tensor
    if activation == "sigmoid":
        return torch.sigmoid(tensor)
    if activation == "softmax":
        return torch.softmax(tensor, dim=-1)
    if activation == "tanh":
        return torch.tanh(tensor)
    raise ValueError(f"Unsupported activation '{activation}'.")


def validate_biological_constraints(kc_activity: Tensor, *, expected_sparsity: float = 0.05) -> bool:
    """Validate that KC activity respects the FlyWire sparsity constraint."""

    if kc_activity.dim() != 2:
        raise ValueError("KC activity must be a 2D tensor with shape [batch, n_kc].")
    expected_active = int(round(expected_sparsity * kc_activity.size(-1)))
    actual_active = kc_activity.count_nonzero(dim=-1)
    if torch.any(actual_active > expected_active * 1.2):
        raise ValueError(
            "KC activation exceeds biological limit: "
            f"observed {actual_active.max().item()}, expected <= {expected_active * 1.2:.0f}"
        )
    return True


class MultiTaskDrosophilaModel(nn.Module):
    """Reservoir backed architecture with task specific linear readouts."""

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        *,
        reservoir_params: Optional[Mapping[str, object]] = None,
        task_configs: Optional[Mapping[str, TaskHeadConfig | Mapping[str, object]]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        resolved_params: MutableMapping[str, object] = {}
        if reservoir_params is not None:
            resolved_params.update(reservoir_params)
        if cache_dir is not None:
            resolved_params.setdefault("cache_dir", cache_dir)
        if "sparsity" in resolved_params and "kc_sparsity" not in resolved_params:
            resolved_params["kc_sparsity"] = resolved_params.pop("sparsity")
        filtered_params = {key: value for key, value in resolved_params.items() if value is not None}
        self.reservoir = DrosophilaReservoir(**filtered_params)
        self.n_kc = self.reservoir.n_kc
        self._head_configs: MutableMapping[str, TaskHeadConfig] = {}
        self.task_heads = nn.ModuleDict()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        defaults: Mapping[str, TaskHeadConfig] = {
            "olfactory_conditioning": TaskHeadConfig(
                self.reservoir.n_mbon,
                activation=None,
                loss="binary_crossentropy",
                use_reservoir_head=True,
            ),
            "spatial_navigation": TaskHeadConfig(8, activation="softmax", loss="categorical_crossentropy"),
            "visual_pattern": TaskHeadConfig(16, activation="softmax", loss="categorical_crossentropy"),
            "temporal_sequence": TaskHeadConfig(12, activation="softmax", loss="categorical_crossentropy"),
            "reward_prediction": TaskHeadConfig(1, activation="sigmoid", loss="binary_crossentropy"),
        }

        configs = dict(defaults)
        if task_configs is not None:
            for name, config in task_configs.items():
                if isinstance(config, TaskHeadConfig):
                    configs[name] = config
                else:
                    output_dim = int(config["output_dim"])
                    activation = config.get("activation")
                    loss = config.get("loss")
                    dropout_override = float(config.get("dropout", dropout))
                    configs[name] = TaskHeadConfig(
                        output_dim,
                        activation,
                        loss,
                        dropout_override,
                        bool(config.get("use_reservoir_head", False)),
                    )

        for name, config in configs.items():
            self.add_task_head(name, config)

    @property
    def head_configs(self) -> Mapping[str, TaskHeadConfig]:
        return dict(self._head_configs)

    def add_task_head(self, name: str, config: TaskHeadConfig) -> None:
        if name in self.task_heads:
            raise ValueError(f"Task head '{name}' already registered.")
        if config.use_reservoir_head:
            if config.output_dim != self.reservoir.n_mbon:
                raise ValueError(
                    "Reservoir head must match MBON dimensionality: "
                    f"expected {self.reservoir.n_mbon}, received {config.output_dim}."
                )
            head_module: nn.Module = self.reservoir.kc_to_mbon
        else:
            head_module = nn.Linear(self.n_kc, config.output_dim)

        if config.dropout > 0:
            self.task_heads[name] = nn.Sequential(nn.Dropout(config.dropout), head_module)
        else:
            self.task_heads[name] = head_module
        self._head_configs[name] = config

    def forward(
        self,
        pn_activity: Tensor,
        *,
        tasks: Optional[Sequence[str]] = None,
        return_kc: bool = False,
    ) -> dict[str, Tensor]:
        pn_activity = _ensure_2d(pn_activity)
        with torch.no_grad():
            kc = self.reservoir.pn_to_kc(pn_activity)
            kc = F.relu(kc)
            kc = self.reservoir._enforce_sparsity(kc)
        validate_biological_constraints(kc)
        kc = self.dropout(kc)
        outputs: dict[str, Tensor] = {}
        requested = tasks if tasks is not None else self.task_heads.keys()
        for task in requested:
            if task not in self.task_heads:
                raise KeyError(f"Task '{task}' is not registered.")
            head = self.task_heads[task]
            task_out = head(kc)
            outputs[task] = task_out
        if return_kc:
            outputs["kc_activity"] = kc
        outputs["mbon_activity"] = self.reservoir.forward(pn_activity)
        return outputs

    def predict_task(self, task: str, pn_activity: Tensor) -> Tensor:
        outputs = self.forward(pn_activity, tasks=[task])
        activation = self._head_configs[task].activation
        return _apply_activation(outputs[task], activation)

    def available_tasks(self) -> Sequence[str]:
        return tuple(self.task_heads.keys())

    def freeze_reservoir(self) -> None:
        for parameter in self.reservoir.parameters():
            parameter.requires_grad = False

    def task_parameters(self, tasks: Optional[Iterable[str]] = None) -> Iterable[nn.Parameter]:
        selected = tasks if tasks is not None else self.task_heads.keys()
        for task in selected:
            yield from self.task_heads[task].parameters()

    def create_optimizer(self, tasks: Optional[Sequence[str]] = None, lr: float = 1e-3) -> torch.optim.Optimizer:
        params = list(self.task_parameters(tasks))
        if not params:
            raise ValueError("No task parameters available for optimisation.")
        return torch.optim.Adam(params, lr=lr)
