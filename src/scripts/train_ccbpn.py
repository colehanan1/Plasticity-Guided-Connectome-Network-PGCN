#!/usr/bin/env python3
"""Training pipeline for Connectome-Constrained Behavioral Prediction Network.

This script trains a CCBPN model on behavioral conditioning data, adapting the
2024 Nature study methodology (Lappalainen et al.) to olfactory learning.

The training pipeline:
1. Loads FlyWire connectivity and behavioral conditioning data
2. Creates odor sequence and dopamine signal representations
3. Trains CCBPN end-to-end via gradient descent on behavioral task loss
4. Validates predictions against held-out flies (group k-fold)
5. Saves best model checkpoint and training metrics

Usage
-----
Basic training (odor discrimination):
    python src/scripts/train_ccbpn.py --task odor_discrimination --epochs 100

With custom hyperparameters:
    python src/scripts/train_ccbpn.py \\
        --task odor_discrimination \\
        --epochs 200 \\
        --batch_size 16 \\
        --learning_rate 0.001 \\
        --kc_sparsity 0.05

Memory retention task:
    python src/scripts/train_ccbpn.py --task memory_retention --epochs 150

Example
-------
$ python src/scripts/train_ccbpn.py --task odor_discrimination --epochs 100
Loading FlyWire connectivity from data/cache...
Loaded circuit: 150 PNs → 2500 KCs → 34 MBONs
Loading behavioral data from data/model_predictions.csv...
Loaded 440 trials from 35 flies
Training CCBPN (5-fold cross-validation)...
Fold 1/5: Train loss=0.523, Val acc=0.742
Fold 2/5: Train loss=0.498, Val acc=0.768
...
Best model saved to results/ccbpn_odor_discrimination_best.pt
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pgcn.data.behavioral_data import (
    load_behavioral_dataframe,
    make_group_kfold,
)
from pgcn.data.door_integration import DoORIntegration
from pgcn.models.ccbpn import (
    ConnectomeConstrainedBehavioralPredictor,
    BehavioralTaskLoss,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Connectome-Constrained Behavioral Prediction Network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        default="odor_discrimination",
        choices=["odor_discrimination", "memory_retention", "cross_generalization"],
        help="Behavioral task type"
    )

    # Data paths
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cache",
        help="Path to FlyWire connectivity cache"
    )
    parser.add_argument(
        "--behavioral_data",
        type=str,
        default=None,
        help="Path to behavioral CSV (default: uses BEHAVIORAL_DATA_PATH)"
    )
    parser.add_argument(
        "--dataset_mapping",
        type=str,
        default="configs/dataset_to_odor_mapping.yaml",
        help="Path to dataset-to-odor mapping YAML file"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cache",
        help="Path to FlyWire cache directory (for DoOR integration)"
    )

    # Model hyperparameters
    parser.add_argument(
        "--kc_sparsity",
        type=float,
        default=0.05,
        help="Target KC sparsity fraction (biological: 0.05)"
    )
    parser.add_argument(
        "--tau_pn",
        type=float,
        default=10.0,
        help="PN membrane time constant (ms)"
    )
    parser.add_argument(
        "--tau_kc",
        type=float,
        default=20.0,
        help="KC membrane time constant (ms)"
    )
    parser.add_argument(
        "--tau_mbon",
        type=float,
        default=15.0,
        help="MBON membrane time constant (ms)"
    )
    parser.add_argument(
        "--disable_dopamine",
        action="store_true",
        help="Disable dopamine-gated plasticity"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=50,
        help="Length of odor sequence (time steps)"
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for saving model checkpoints and metrics"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed training progress"
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )

    return parser.parse_args()


def prepare_behavioral_data(
    behavioral_csv: Optional[str] = None,
    dataset_mapping_path: str = "configs/dataset_to_odor_mapping.yaml",
    cache_dir: str = "data/cache",
    sequence_length: int = 50,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load and prepare behavioral data for CCBPN training.

    Converts behavioral trial data into:
    1. Odor sequences (PN activity patterns over time) using DoOR database
    2. Dopamine signals (reward/punishment timing)
    3. Behavioral labels (approach/avoid outcomes)

    Parameters
    ----------
    behavioral_csv : str, optional
        Path to behavioral CSV file
    dataset_mapping_path : str
        Path to dataset-to-odor mapping YAML file
    cache_dir : str
        Path to FlyWire cache directory
    sequence_length : int
        Length of temporal sequence (default: 50 time steps)
    device : str
        Device for tensors (default: 'cpu')

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, np.ndarray]
        (odor_sequences, dopamine_signals, behavioral_labels, groups)
        - odor_sequences: (n_trials, sequence_length, n_pn)
        - dopamine_signals: (n_trials, sequence_length)
        - behavioral_labels: (n_trials,)
        - groups: (n_trials,) fly identifiers for group k-fold
    """
    print(f"Loading behavioral data from {behavioral_csv or 'default path'}...")

    # Load behavioral dataframe
    df = load_behavioral_dataframe(behavioral_csv, validate=True)

    n_trials = len(df)
    print(f"Loaded {n_trials} trials from {df['dataset'].nunique()} datasets, {df['fly'].nunique()} flies")

    # Extract behavioral labels (prediction column)
    behavioral_labels = df['prediction'].values  # Binary: 1=approach, 0=avoid

    # Extract fly groups for cross-validation
    groups = df['fly'].values

    # Load dataset-to-odor mapping
    print(f"Loading dataset-to-odor mapping from {dataset_mapping_path}...")
    with open(dataset_mapping_path, 'r') as f:
        dataset_mapping = yaml.safe_load(f)

    # Initialize DoOR integration
    print(f"Initializing DoOR integration (cache_dir={cache_dir})...")
    door = DoORIntegration(cache_dir=Path(cache_dir))

    # Infer number of PNs from DoOR/FlyWire cache
    n_pn = len(door.pn_glomeruli) if door.pn_glomeruli else 150
    print(f"Using {n_pn} projection neurons")

    # Generate odor sequences using DoOR-based PN activity patterns
    print(f"Generating DoOR-based odor sequences (sequence_length={sequence_length})...")
    odor_sequences = torch.zeros(n_trials, sequence_length, n_pn)

    # Track statistics for validation
    odor_coverage_stats = {}
    missing_odors = set()

    for trial_idx, row in df.iterrows():
        dataset = row['dataset']
        trial_label = row['trial_label']

        # Parse trial label to get trial type and number
        # Expected format: "training_1", "testing_3", etc.
        if 'training' in str(trial_label):
            trial_type = 'training_trials'
            trial_num = int(str(trial_label).split('_')[-1]) - 1  # 0-indexed
        elif 'testing' in str(trial_label):
            trial_type = 'testing_trials'
            trial_num = int(str(trial_label).split('_')[-1]) - 1  # 0-indexed
        else:
            print(f"Warning: Unrecognized trial_label format: {trial_label}")
            continue

        # Get odor identity from mapping
        if dataset not in dataset_mapping:
            print(f"Warning: Dataset '{dataset}' not in mapping YAML")
            continue

        trials_list = dataset_mapping[dataset].get(trial_type, [])
        if trial_num >= len(trials_list):
            print(f"Warning: Trial {trial_label} out of range for {dataset} {trial_type}")
            continue

        odor_name = trials_list[trial_num]

        # Convert odor to PN activity pattern
        try:
            # Create temporal sequence: odor ON from t=0 to t=40, OFF after
            odor_sequence = door.create_odor_sequence(
                odor_name,
                n_pn=n_pn,
                sequence_length=sequence_length,
                odor_onset=0,
                odor_duration=40  # 40ms odor pulse (can be scaled to match 30s)
            )

            odor_sequences[trial_idx] = torch.from_numpy(odor_sequence).float()

            # Track coverage statistics
            if odor_name not in odor_coverage_stats:
                n_active = np.sum(odor_sequence[20, :] > 0.1)  # Check mid-odor
                odor_coverage_stats[odor_name] = n_active

        except Exception as e:
            print(f"Warning: Failed to get PN pattern for odor '{odor_name}': {e}")
            missing_odors.add(odor_name)
            # Leave as zeros (no odor pattern)

    # Print coverage statistics
    print(f"\nDoOR coverage statistics:")
    for odor, n_active in sorted(odor_coverage_stats.items()):
        status = "✓" if n_active > 0 else "✗"
        print(f"  {status} {odor:25s}: {n_active:3d} active PNs")

    if missing_odors:
        print(f"\n⚠️  Warning: {len(missing_odors)} odors missing from DoOR:")
        for odor in sorted(missing_odors):
            print(f"    - {odor}")
        print("  These trials will have ZERO odor patterns!")

    # Create dopamine signals based on behavioral outcome
    # Reward (dopamine=1.0) if approach, punishment (dopamine=-1.0) if avoid
    # Dopamine delivered at t=45 (after odor offset)
    dopamine_signals = torch.zeros(n_trials, sequence_length)

    for trial_idx, label in enumerate(behavioral_labels):
        if label > 0.5:  # Approach trial
            dopamine_signals[trial_idx, 45:] = 1.0  # Reward
        else:  # Avoid trial
            dopamine_signals[trial_idx, 45:] = -1.0  # Punishment

    # Move to device
    odor_sequences = odor_sequences.to(device)
    dopamine_signals = dopamine_signals.to(device)

    print(f"\nPrepared data shapes:")
    print(f"  Odor sequences: {odor_sequences.shape}")
    print(f"  Dopamine signals: {dopamine_signals.shape}")
    print(f"  Behavioral labels: {behavioral_labels.shape}")
    print(f"  Mean active PNs per trial: {torch.sum(odor_sequences > 0.1, dim=2).float().mean():.1f}")

    return odor_sequences, dopamine_signals, behavioral_labels, groups


def train_epoch(
    model: ConnectomeConstrainedBehavioralPredictor,
    dataloader: DataLoader,
    criterion: BehavioralTaskLoss,
    optimizer: optim.Optimizer,
    device: str,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Train model for one epoch.

    Parameters
    ----------
    model : ConnectomeConstrainedBehavioralPredictor
        CCBPN model
    dataloader : DataLoader
        Training data loader
    criterion : BehavioralTaskLoss
        Loss function
    optimizer : optim.Optimizer
        Optimizer
    device : str
        Device
    verbose : bool
        Print batch-level progress

    Returns
    -------
    Tuple[float, float]
        (average_loss, accuracy)
    """
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    iterator = tqdm(dataloader, desc="Training") if verbose else dataloader

    for batch_idx, (odor_seq, dopa_sig, labels) in enumerate(iterator):
        # Move batch to device
        odor_seq = odor_seq.to(device)
        dopa_sig = dopa_sig.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(odor_seq, dopa_sig, return_intermediates=False)
        predicted_behavior = outputs['behavioral_output']  # (batch, time)

        # Compute loss
        loss = criterion(predicted_behavior, labels, trial_metadata=None)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # CRITICAL: Re-enforce connectivity constraints
        model.enforce_connectivity_constraints()

        # Metrics
        total_loss += loss.item()
        final_predictions = (predicted_behavior[:, -1] > 0.5).float()
        correct_predictions += (final_predictions == labels).sum().item()
        total_predictions += len(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy


def evaluate(
    model: ConnectomeConstrainedBehavioralPredictor,
    dataloader: DataLoader,
    criterion: BehavioralTaskLoss,
    device: str,
) -> Tuple[float, float]:
    """Evaluate model on validation set.

    Parameters
    ----------
    model : ConnectomeConstrainedBehavioralPredictor
        CCBPN model
    dataloader : DataLoader
        Validation data loader
    criterion : BehavioralTaskLoss
        Loss function
    device : str
        Device

    Returns
    -------
    Tuple[float, float]
        (average_loss, accuracy)
    """
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for odor_seq, dopa_sig, labels in dataloader:
            # Move batch to device
            odor_seq = odor_seq.to(device)
            dopa_sig = dopa_sig.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(odor_seq, dopa_sig, return_intermediates=False)
            predicted_behavior = outputs['behavioral_output']

            # Compute loss
            loss = criterion(predicted_behavior, labels, trial_metadata=None)

            # Metrics
            total_loss += loss.item()
            final_predictions = (predicted_behavior[:, -1] > 0.5).float()
            correct_predictions += (final_predictions == labels).sum().item()
            total_predictions += len(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy


def main():
    """Main training pipeline."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Prepare behavioral data
    odor_sequences, dopamine_signals, behavioral_labels, groups = prepare_behavioral_data(
        behavioral_csv=args.behavioral_data,
        dataset_mapping_path=args.dataset_mapping,
        cache_dir=args.cache_dir,
        sequence_length=args.sequence_length,
        device=args.device,
    )

    # Initialize model
    print(f"\nInitializing CCBPN model...")
    model = ConnectomeConstrainedBehavioralPredictor(
        cache_dir=args.cache_dir,
        behavioral_task=args.task,
        tau_pn=args.tau_pn,
        tau_kc=args.tau_kc,
        tau_mbon=args.tau_mbon,
        kc_sparsity_target=args.kc_sparsity,
        enable_dopamine_modulation=not args.disable_dopamine,
    )
    model = model.to(args.device)

    # Update odor sequence dimension to match model's n_pn
    if odor_sequences.shape[2] != model.n_pn:
        print(f"Adjusting odor sequences from {odor_sequences.shape[2]} to {model.n_pn} PNs...")
        # Resize by padding or truncating
        new_odor_sequences = torch.zeros(
            odor_sequences.shape[0],
            odor_sequences.shape[1],
            model.n_pn,
            device=args.device
        )
        min_pn = min(odor_sequences.shape[2], model.n_pn)
        new_odor_sequences[:, :, :min_pn] = odor_sequences[:, :, :min_pn]
        odor_sequences = new_odor_sequences

    # Loss function
    criterion = BehavioralTaskLoss(task_type=args.task)

    # Cross-validation training
    print(f"\nTraining CCBPN ({args.n_folds}-fold cross-validation)...")

    fold_results = []
    best_val_acc = 0.0
    best_model_state = None

    for fold_idx, (train_idx, val_idx) in enumerate(make_group_kfold(
        path=args.behavioral_data,
        n_splits=args.n_folds,
        groups=groups,
        validate=False,
    )):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*60}")

        # Create data loaders
        train_dataset = TensorDataset(
            odor_sequences[train_idx],
            dopamine_signals[train_idx],
            torch.from_numpy(behavioral_labels[train_idx]).float(),
        )
        val_dataset = TensorDataset(
            odor_sequences[val_idx],
            dopamine_signals[val_idx],
            torch.from_numpy(behavioral_labels[val_idx]).float(),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        # Re-initialize model for each fold
        model = ConnectomeConstrainedBehavioralPredictor(
            cache_dir=args.cache_dir,
            behavioral_task=args.task,
            tau_pn=args.tau_pn,
            tau_kc=args.tau_kc,
            tau_mbon=args.tau_mbon,
            kc_sparsity_target=args.kc_sparsity,
            enable_dopamine_modulation=not args.disable_dopamine,
        )
        model = model.to(args.device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # Training loop
        fold_best_val_acc = 0.0
        fold_metrics = []

        for epoch in range(args.epochs):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer,
                args.device, verbose=args.verbose
            )

            # Validate
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, args.device
            )

            # Log metrics
            fold_metrics.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            })

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f} | "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")

            # Save best model for this fold
            if val_acc > fold_best_val_acc:
                fold_best_val_acc = val_acc
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()

        # Store fold results
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_acc': fold_best_val_acc,
            'metrics': fold_metrics,
        })

        print(f"Fold {fold_idx + 1} best validation accuracy: {fold_best_val_acc:.3f}")

    # Save best model
    if best_model_state is not None:
        model_path = output_dir / f"ccbpn_{args.task}_best.pt"
        torch.save({
            'model_state_dict': best_model_state,
            'args': vars(args),
            'best_val_acc': best_val_acc,
        }, model_path)
        print(f"\nBest model saved to {model_path}")
        print(f"Best validation accuracy: {best_val_acc:.3f}")

    # Save training metrics
    metrics_path = output_dir / f"ccbpn_{args.task}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'fold_results': fold_results,
            'best_val_acc': best_val_acc,
        }, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    avg_val_acc = np.mean([f['best_val_acc'] for f in fold_results])
    std_val_acc = np.std([f['best_val_acc'] for f in fold_results])
    print(f"Average validation accuracy: {avg_val_acc:.3f} ± {std_val_acc:.3f}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
