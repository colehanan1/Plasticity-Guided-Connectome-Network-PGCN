"""Train the multi-task Drosophila reservoir using shared connectome features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import torch
from rich.console import Console
from rich.progress import Progress

from pgcn.data import TaskDataLoaderFactory, load_multi_task_config
from pgcn.models import MultiTaskDrosophilaModel, TaskHeadConfig


def _resolve_loss(loss_function: str) -> torch.nn.Module:
    if loss_function == "binary_crossentropy":
        return torch.nn.BCEWithLogitsLoss()
    if loss_function == "categorical_crossentropy":
        return torch.nn.CrossEntropyLoss()
    if loss_function == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"Unsupported loss function '{loss_function}'.")


def _build_model(config, tasks: Mapping[str, TaskHeadConfig]) -> MultiTaskDrosophilaModel:
    model = MultiTaskDrosophilaModel(cache_dir=config.reservoir.cache_dir, task_configs=tasks)
    if config.reservoir.freeze_pn_kc:
        model.freeze_reservoir()
    return model


def _train_single_task(
    model: MultiTaskDrosophilaModel,
    task_name: str,
    task_spec,
    dataloader,
    device: torch.device,
    console: Console,
) -> list[float]:
    model.train()
    optimizer = model.create_optimizer(tasks=[task_name], lr=task_spec.learning_rate)
    loss_fn = _resolve_loss(task_spec.loss_function)
    epoch_losses: list[float] = []
    for epoch in range(task_spec.epochs):
        running_loss = 0.0
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model.forward(features, tasks=[task_name])
            logits = outputs[task_name]
            if task_spec.loss_function == "binary_crossentropy" and logits.ndim > targets.ndim:
                targets = targets.view_as(logits)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        epoch_loss = running_loss / max(1, len(dataloader))
        epoch_losses.append(epoch_loss)
        console.log(f"[{task_name}] epoch {epoch + 1}/{task_spec.epochs} loss={epoch_loss:.4f}")
    return epoch_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multi_task_config.yaml", help="YAML configuration file")
    parser.add_argument("--output-dir", default="artifacts/multi_task", help="Directory to store checkpoints and logs")
    parser.add_argument("--tasks", nargs="*", help="Optional subset of tasks to train")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even when CUDA is available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    config = load_multi_task_config(args.config)

    selected_tasks: Mapping[str, TaskHeadConfig] = {}
    for name, spec in config.tasks.items():
        if args.tasks and name not in args.tasks:
            continue
        selected_tasks[name] = TaskHeadConfig(
            output_dim=spec.output_dim,
            activation=spec.activation,
            loss=spec.loss_function,
            dropout=spec.dropout,
            use_reservoir_head=spec.use_reservoir_head,
        )
    if not selected_tasks:
        raise ValueError("No tasks selected for training. Check the --tasks argument or configuration file.")

    model = _build_model(config, selected_tasks)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    factory = TaskDataLoaderFactory()

    history: dict[str, list[float]] = {}
    with Progress() as progress:
        task_progress = progress.add_task("training", total=len(selected_tasks))
        for name, spec in config.tasks.items():
            if name not in selected_tasks:
                continue
            dataloader = factory.create(spec, shuffle=True)
            epoch_losses = _train_single_task(model, name, spec, dataloader, device, console)
            history[name] = epoch_losses
            progress.advance(task_progress)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "multi_task_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    console.print(f"Model checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
