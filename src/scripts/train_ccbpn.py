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
        help="Path to FlyWire connectivity and DoOR data cache"
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

    # Model hyperparameters
    parser.add_argument(
        "--kc_sparsity",
        type=float,
        default=0.10,
        help="Target KC sparsity fraction (biological: 0.05, recommended: 0.10)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30,
        help="Dropout rate for KC→MBON layer (0.30=balanced - prevents overfitting while allowing learning)"
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
        default=0.01,
        help="Initial learning rate for optimizer (allows faster learning)"
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
        help="Use learning rate scheduler (warmup + cosine decay)"
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class-balanced loss to handle imbalanced datasets"
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

    # Dataset filtering (for task complexity test)
    parser.add_argument(
        "--single_dataset",
        type=str,
        default=None,
        help="Train on single dataset only (e.g., 'opto_hex'). If None, use all datasets."
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
    add_input_noise: bool = True,
    noise_std: float = 0.15,
    single_dataset: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Load and prepare behavioral data for CCBPN training.

    CRITICAL CHANGES (Biological Realism Fixes):
    1. Dopamine assigned based on TRAINING PROTOCOL (CS+ identity), not behavioral outcome
    2. Input noise ADDED to create trial-to-trial variability
    3. Control datasets INCLUDED to capture innate preferences

    Converts behavioral trial data into:
    1. Odor sequences (PN activity patterns over time) using DoOR database WITH NOISE
    2. Dopamine signals (reward timing based on CS+ identity)
    3. Behavioral labels (approach/avoid outcomes)

    Parameters
    ----------
    behavioral_csv : str, optional
        Path to behavioral CSV file (includes control + conditioned datasets)
    dataset_mapping_path : str
        Path to dataset-to-odor mapping YAML file (must include dataset_reward_mapping)
    cache_dir : str
        Path to FlyWire cache directory
    sequence_length : int
        Length of temporal sequence (default: 50 time steps)
    device : str
        Device for tensors (default: 'cpu')
    add_input_noise : bool
        If True, add biological noise to odor inputs (default: True)
    noise_std : float
        Standard deviation of noise (default: 0.15 = 15% variability)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
        (odor_sequences, dopamine_signals, behavioral_labels, groups)
        - odor_sequences: (n_trials, sequence_length, n_pn) WITH trial variability
        - dopamine_signals: (n_trials, sequence_length) based on CS+ identity
        - behavioral_labels: (n_trials,)
        - groups: (n_trials,) fly identifiers for group k-fold
    """
    print(f"Loading behavioral data from {behavioral_csv or 'default path'}...")

    # Load behavioral dataframe
    df = load_behavioral_dataframe(behavioral_csv, validate=True)

    n_trials = len(df)
    print(f"Loaded {n_trials} trials from {df['dataset'].nunique()} datasets, {df['fly'].nunique()} flies")

    # Filter to single dataset if specified (for task complexity test)
    if single_dataset is not None:
        print(f"\n⚠️  SINGLE-DATASET MODE: Filtering to dataset '{single_dataset}' only")
        df = df[df['dataset'] == single_dataset].copy()

        if len(df) == 0:
            raise ValueError(
                f"No trials found for dataset '{single_dataset}'. "
                f"Available datasets: {sorted(load_behavioral_dataframe(behavioral_csv, validate=True)['dataset'].unique())}"
            )

        # CRITICAL: Reset index to avoid index out of bounds errors in trial_odors_array
        df = df.reset_index(drop=True)

        n_trials_filtered = len(df)
        print(f"  Retained {n_trials_filtered} trials from {df['fly'].nunique()} flies")
        print(f"  Class distribution: {np.sum(df['prediction'] == 0)} avoid, {np.sum(df['prediction'] == 1)} approach")
        n_trials = n_trials_filtered

    # Extract behavioral labels (prediction column)
    behavioral_labels = df['prediction'].values  # Binary: 1=approach, 0=avoid

    # Extract fly groups for cross-validation
    groups = df['fly'].values

    # Load dataset-to-odor mapping AND reward mapping
    print(f"Loading dataset-to-odor mapping from {dataset_mapping_path}...")
    with open(dataset_mapping_path, 'r') as f:
        config = yaml.safe_load(f)

    # Separate trial mappings from reward mappings
    reward_mapping = config.get('dataset_reward_mapping', {})
    dataset_mapping = {k: v for k, v in config.items() if k != 'dataset_reward_mapping'}

    print(f"  Found {len(dataset_mapping)} datasets and {len(reward_mapping)} reward assignments")

    # Initialize DoOR integration
    print(f"Initializing DoOR integration (cache_dir={cache_dir})...")
    door = DoORIntegration(cache_dir=Path(cache_dir))

    # Infer number of PNs from DoOR/FlyWire cache
    n_pn = len(door.pn_glomeruli) if door.pn_glomeruli else 150
    print(f"Using {n_pn} projection neurons")

    # Generate odor sequences using DoOR-based PN activity patterns WITH NOISE
    noise_status = "WITH biological noise" if add_input_noise else "deterministic"
    print(f"Generating DoOR-based odor sequences ({noise_status}, sequence_length={sequence_length})...")
    odor_sequences = torch.zeros(n_trials, sequence_length, n_pn)

    # Track statistics for validation
    odor_coverage_stats = {}
    missing_odors = set()
    trial_odors = []  # Track odor identity for dopamine assignment

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
        trial_odors.append(odor_name)  # Track for dopamine assignment

        # Convert odor to PN activity pattern WITH biological noise
        try:
            # Create temporal sequence: odor ON from t=0 to t=40, OFF after
            odor_sequence = door.create_odor_sequence(
                odor_name,
                n_pn=n_pn,
                sequence_length=sequence_length,
                odor_onset=0,
                odor_duration=40,  # 40ms odor pulse (can be scaled to match 30s)
                add_noise=add_input_noise,  # Enable trial-to-trial variability
                noise_std=noise_std,        # 15% noise (default)
                temporal_jitter=3           # ±3ms timing variability
            )

            odor_sequences[trial_idx] = torch.from_numpy(odor_sequence).float()

            # Track coverage statistics (only for unique odors, before noise)
            if odor_name not in odor_coverage_stats:
                # Check coverage on canonical pattern (without noise for consistency)
                canonical_seq = door.create_odor_sequence(
                    odor_name, n_pn=n_pn, sequence_length=sequence_length,
                    odor_onset=0, odor_duration=40, add_noise=False
                )
                n_active = np.sum(canonical_seq[20, :] > 0.1)  # Check mid-odor
                odor_coverage_stats[odor_name] = n_active

        except Exception as e:
            print(f"Warning: Failed to get PN pattern for odor '{odor_name}': {e}")
            missing_odors.add(odor_name)
            trial_odors[-1] = None  # Mark as failed
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

    # CRITICAL: Create dopamine signals based on TRAINING PROTOCOL (CS+ identity), NOT behavioral outcome!
    # This is the key biological realism fix - dopamine reflects reward contingency, not fly's choice
    print(f"\nAssigning dopamine signals based on CS+ identity (training protocol)...")
    dopamine_signals = torch.zeros(n_trials, sequence_length)

    # Convert trial_odors to numpy array for easier indexing
    trial_odors_array = np.array(trial_odors, dtype=object)

    # Track dopamine statistics
    dopamine_stats = {dataset: {'total': 0, 'rewarded': 0} for dataset in df['dataset'].unique()}

    for trial_idx, row in df.iterrows():
        dataset = row['dataset']
        odor_name = trial_odors_array[trial_idx]

        if odor_name is None:
            continue  # Skip trials with failed odor mapping

        # Get the CS+ (rewarded odor) for this dataset
        rewarded_odor = reward_mapping.get(dataset, None)

        dopamine_stats[dataset]['total'] += 1

        if rewarded_odor is not None and odor_name == rewarded_odor:
            # This odor was the CS+ (rewarded) in this dataset
            # Dopamine signal during/after odor presentation (40-50ms window)
            dopamine_signals[trial_idx, 40:50] = 1.0  # Reward
            dopamine_stats[dataset]['rewarded'] += 1
        else:
            # This odor was CS- (not rewarded) OR from control dataset
            # NO dopamine signal
            dopamine_signals[trial_idx, :] = 0.0  # No reward

    # Print dopamine assignment statistics
    print(f"\nDopamine assignment statistics:")
    total_rewarded = 0
    for dataset in sorted(dopamine_stats.keys()):
        n_total = dopamine_stats[dataset]['total']
        n_rewarded = dopamine_stats[dataset]['rewarded']
        total_rewarded += n_rewarded
        pct = (n_rewarded / n_total * 100) if n_total > 0 else 0
        cs_plus = reward_mapping.get(dataset, 'none')
        print(f"  {dataset:20s}: {n_rewarded:3d}/{n_total:3d} trials ({pct:5.1f}%) | CS+: {cs_plus}")

    print(f"\n  TOTAL: {total_rewarded}/{n_trials} trials ({total_rewarded/n_trials*100:.1f}%) received dopamine")

    # Validation: Check that control datasets have ZERO dopamine
    control_datasets = [d for d, cs in reward_mapping.items() if cs is None]
    for dataset in control_datasets:
        dataset_trials = df[df['dataset'] == dataset].index
        if len(dataset_trials) > 0:
            dataset_dopamine = dopamine_signals[dataset_trials].max().item()
            if dataset_dopamine > 0:
                print(f"  ⚠️  WARNING: Control dataset '{dataset}' has dopamine signal!")
            else:
                print(f"  ✓ Control dataset '{dataset}' has ZERO dopamine (correct)")

    # Move to device
    odor_sequences = odor_sequences.to(device)
    dopamine_signals = dopamine_signals.to(device)

    print(f"\nPrepared data shapes:")
    print(f"  Odor sequences: {odor_sequences.shape}")
    print(f"  Dopamine signals: {dopamine_signals.shape}")
    print(f"  Behavioral labels: {behavioral_labels.shape}")
    print(f"  Mean active PNs per trial: {torch.sum(odor_sequences > 0.1, dim=2).float().mean():.1f}")

    # VALIDATION: Check that noise created trial-to-trial variability
    if add_input_noise:
        print(f"\nValidating trial-to-trial variability...")

        # Find trials of the same odor
        for odor in set(trial_odors_array) - {None}:
            odor_trials = np.where(trial_odors_array == odor)[0]

            if len(odor_trials) >= 2:
                # Compute pairwise correlations between trials of same odor
                correlations = []
                for i in range(min(5, len(odor_trials))):
                    for j in range(i+1, min(5, len(odor_trials))):
                        trial_i = odor_sequences[odor_trials[i]].flatten().cpu().numpy()
                        trial_j = odor_sequences[odor_trials[j]].flatten().cpu().numpy()

                        if np.sum(trial_i) > 0 and np.sum(trial_j) > 0:
                            corr = np.corrcoef(trial_i, trial_j)[0, 1]
                            correlations.append(corr)

                if correlations:
                    mean_corr = np.mean(correlations)
                    print(f"  {odor:25s}: mean correlation = {mean_corr:.3f} (expect 0.90-0.95)")

                    if mean_corr > 0.98:
                        print(f"    ⚠️  WARNING: Correlation too high ({mean_corr:.3f})! Noise may be insufficient.")
                    elif mean_corr < 0.85:
                        print(f"    ⚠️  WARNING: Correlation too low ({mean_corr:.3f})! Noise may be excessive.")
                break  # Just check one odor as example

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


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Stops training when validation loss stops improving for `patience` epochs.

    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping (default: 15)
    min_delta : float
        Minimum change in validation loss to qualify as improvement (default: 0.001)
    verbose : bool
        If True, print messages when validation loss improves (default: True)

    Example
    -------
    >>> early_stopping = EarlyStopping(patience=15)
    >>> for epoch in range(100):
    ...     train_loss = train_one_epoch(...)
    ...     val_loss = validate(...)
    ...     if early_stopping(val_loss):
    ...         print(f"Early stopping at epoch {epoch}")
    ...         break
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Parameters
        ----------
        val_loss : float
            Current validation loss

        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Validation loss improved
            if self.verbose:
                print(f"    Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0
            self.should_stop = False
        else:
            # No improvement
            self.counter += 1
            if self.verbose and self.counter > 0:
                print(f"    No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                if self.verbose:
                    print(f"    Early stopping triggered (no improvement for {self.patience} epochs)")
                self.should_stop = True

        return self.should_stop


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

    # Prepare behavioral data WITH biological noise and correct dopamine assignment
    odor_sequences, dopamine_signals, behavioral_labels, groups = prepare_behavioral_data(
        behavioral_csv=args.behavioral_data,
        dataset_mapping_path=args.dataset_mapping,
        cache_dir=args.cache_dir,
        sequence_length=args.sequence_length,
        device=args.device,
        add_input_noise=True,  # Enable trial-to-trial variability
        noise_std=0.08,        # REDUCED to 8% additive only (was 15% with multiplicative/dropout/jitter)
        single_dataset=args.single_dataset,  # Filter to single dataset if specified
    )

    # Compute class weights for imbalanced datasets
    class_weights = None
    if args.use_class_weights:
        print(f"\nComputing class weights for imbalanced dataset...")
        n_avoid = np.sum(behavioral_labels == 0)
        n_approach = np.sum(behavioral_labels == 1)
        n_total = len(behavioral_labels)

        # Inverse frequency weighting
        weight_avoid = n_total / (2.0 * n_avoid) if n_avoid > 0 else 1.0
        weight_approach = n_total / (2.0 * n_approach) if n_approach > 0 else 1.0

        class_weights = torch.tensor([weight_avoid, weight_approach], device=args.device)

        print(f"  Class distribution:")
        print(f"    Avoid (0):    {n_avoid:4d} ({n_avoid/n_total:.1%}) -> weight={weight_avoid:.3f}")
        print(f"    Approach (1): {n_approach:4d} ({n_approach/n_total:.1%}) -> weight={weight_approach:.3f}")

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
        dropout_rate=args.dropout,
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

    # Loss function with optional class weighting
    criterion = BehavioralTaskLoss(
        task_type=args.task,
        use_class_weights=args.use_class_weights,
        class_weights=class_weights,
    )

    # Cross-validation training
    print(f"\nTraining CCBPN ({args.n_folds}-fold cross-validation)...")

    fold_results = []
    best_val_acc = 0.0
    best_model_state = None

    # Handle cross-validation splits
    # If single_dataset is used, we can't use make_group_kfold with path (it reloads full data)
    # Instead, use GroupKFold directly on filtered data
    if args.single_dataset is not None:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=args.n_folds)
        cv_splits = splitter.split(np.arange(len(groups)), groups=groups)
    else:
        cv_splits = make_group_kfold(
            path=args.behavioral_data,
            n_splits=args.n_folds,
            groups=groups,
            validate=False,
        )

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
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
            dropout_rate=args.dropout,
        )
        model = model.to(args.device)

        # Optimizer with light L2 regularization to prevent overfitting
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.002)

        # Learning rate scheduler (optional)
        scheduler = None
        if args.use_lr_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            # Warmup for 10 epochs, then cosine decay
            warmup_epochs = 10
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - warmup_epochs,
                eta_min=0.0001
            )
            scheduler = SequentialLR(
                optimizer,
                [warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            print(f"  Learning rate scheduler: warmup({warmup_epochs}) + cosine decay")

        # Training loop
        fold_best_val_acc = 0.0
        fold_metrics = []

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(patience=20, min_delta=0.001, verbose=args.verbose)

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

            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()

            # Check early stopping
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"  Best validation loss: {early_stopping.best_loss:.4f}")
                print(f"  No improvement for {early_stopping.patience} epochs")
                break

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
