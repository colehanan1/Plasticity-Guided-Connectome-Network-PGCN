"""Training script for CCBPN with recurrent context memory.

This script implements sequential training where trials are processed in temporal
order within each fly, maintaining LSTM hidden state to enable trial-to-trial learning.

Biological Motivation
---------------------
Real Drosophila maintain context across trials through synaptic tags and dopaminergic
plasticity. This training procedure mimics sequential learning by:

1. Processing all trials for a fly in order
2. Maintaining LSTM hidden state across trials (within fly)
3. Resetting context between flies
4. Using truncated BPTT to prevent gradient vanishing

This enables the model to learn context-dependent associations (e.g., hexanol=CS+
in opto_hex but CS- in opto_benz) without explicit context labels.

Usage
-----
# Quick test with 10 flies
python src/scripts/train_ccbpn_recurrent.py \\
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \\
    --cache-dir data/cache \\
    --output-dir results/ccbpn_recurrent_test \\
    --epochs 10 \\
    --context-dim 32 \\
    --max-flies 10

# Full training
python src/scripts/train_ccbpn_recurrent.py \\
    --behavioral-data ~/Documents/cole/Data/Opto/Combined/model_predictions.csv \\
    --cache-dir data/cache \\
    --output-dir results/ccbpn_recurrent_final \\
    --epochs 100 \\
    --context-dim 64 \\
    --lr 0.001 \\
    --use-class-weights \\
    --n-folds 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pgcn.models.ccbpn_recurrent import CCBPNWithRecurrentContext
from pgcn.data.behavioral_data import load_behavioral_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequentialBehavioralDataset:
    """Dataset that preserves trial order within flies.

    Each sample is an entire sequence for one fly rather than individual trials.
    This enables sequential training with maintained hidden state.

    Parameters
    ----------
    behavioral_csv : str or Path
        Path to behavioral data CSV
    n_pn : int
        Number of projection neurons (for odor encoding)
    odor_duration : int
        Duration of odor presentation in timesteps
    use_noise : bool
        Whether to add noise to odor representations
    noise_std : float
        Standard deviation of Gaussian noise

    Attributes
    ----------
    fly_sequences : dict
        Maps fly_id → list of (odor_seq, dopamine, label, trial_info)
    fly_ids : list
        List of all fly IDs in dataset
    """

    def __init__(
        self,
        behavioral_csv: str | Path,
        n_pn: int = 150,
        odor_duration: int = 50,
        use_noise: bool = True,
        noise_std: float = 0.08,
    ):
        self.n_pn = n_pn
        self.odor_duration = odor_duration
        self.use_noise = use_noise
        self.noise_std = noise_std

        # Load behavioral data
        df = load_behavioral_dataframe(behavioral_csv, validate=False)

        # Sort by fly and trial to ensure temporal order
        sort_cols = ['fly']
        if 'fly_number' in df.columns:
            sort_cols.append('fly_number')
            df['fly_id'] = df['fly'] + '_' + df['fly_number'].astype(str)
        else:
            df['fly_id'] = df['fly']

        if 'trial_order' in df.columns:
            sort_cols.append('trial_order')
        elif 'trial_label' in df.columns:
            sort_cols.append('trial_label')

        df = df.sort_values(sort_cols).reset_index(drop=True)

        logger.info(f"Loaded {len(df)} trials from {df['fly_id'].nunique()} flies")

        # Group trials by fly
        self.fly_sequences = {}
        for fly_id, fly_df in df.groupby('fly_id'):
            sequence = self._prepare_fly_sequence(fly_df)
            if len(sequence) > 0:  # Only include flies with valid trials
                self.fly_sequences[fly_id] = sequence

        self.fly_ids = list(self.fly_sequences.keys())
        logger.info(f"Prepared {len(self.fly_ids)} fly sequences")

    def _prepare_fly_sequence(self, fly_df: pd.DataFrame) -> List[Tuple]:
        """Convert fly's trials to sequence of (odor, dopamine, label, info) tuples."""
        sequence = []

        for idx, row in fly_df.iterrows():
            # Generate odor sequence (simplified - using random patterns)
            # In production, this should use DoOR integration
            odor_seq = self._generate_odor_sequence(row)

            # Generate dopamine signal
            dopamine = self._generate_dopamine_signal(row)

            # Get label
            label = float(row['prediction'])

            # Store trial info for debugging
            trial_info = {
                'dataset': row['dataset'],
                'trial_label': row.get('trial_label', 'unknown'),
                'fly': row.get('fly', 'unknown'),
            }

            sequence.append((odor_seq, dopamine, label, trial_info))

        return sequence

    def _generate_odor_sequence(self, row: pd.Series) -> np.ndarray:
        """Generate temporal odor sequence for one trial.

        In production, this should use DoOR integration to get realistic
        PN activation patterns. For now, uses simplified random patterns.

        Returns
        -------
        np.ndarray
            Shape: (odor_duration, n_pn)
        """
        # Simplified odor encoding (should be replaced with DoOR)
        # Use hash of odor name to get reproducible patterns
        odor_name = row.get('trial_label', 'unknown')
        np.random.seed(hash(odor_name) % (2**32))

        # Create baseline odor pattern (sparse)
        odor_pattern = np.zeros(self.n_pn)
        n_active = int(0.15 * self.n_pn)  # 15% active PNs
        active_indices = np.random.choice(self.n_pn, size=n_active, replace=False)
        odor_pattern[active_indices] = np.random.uniform(0.5, 1.0, size=n_active)

        # Expand to temporal sequence
        odor_seq = np.tile(odor_pattern, (self.odor_duration, 1))

        # Add temporal dynamics (ramp up at start, plateau, decay at end)
        ramp_duration = 10
        decay_duration = 5
        temporal_profile = np.ones(self.odor_duration)
        temporal_profile[:ramp_duration] = np.linspace(0, 1, ramp_duration)
        temporal_profile[-decay_duration:] = np.linspace(1, 0, decay_duration)
        odor_seq *= temporal_profile[:, np.newaxis]

        # Add noise
        if self.use_noise:
            noise = np.random.randn(self.odor_duration, self.n_pn) * self.noise_std
            odor_seq = np.maximum(0, odor_seq + noise)

        return odor_seq.astype(np.float32)

    def _generate_dopamine_signal(self, row: pd.Series) -> np.ndarray:
        """Generate dopamine signal for one trial.

        CS+ trials get dopamine pulse after odor offset.
        CS- trials get no dopamine.

        Returns
        -------
        np.ndarray
            Shape: (odor_duration,)
        """
        dopamine = np.zeros(self.odor_duration, dtype=np.float32)

        # Check if this is a CS+ trial (should be in data, but infer if missing)
        # In production, this should come from the dataset
        is_cs_plus = row.get('prediction', 0) > 0.5

        if is_cs_plus:
            # Dopamine pulse after odor offset (realistic timing)
            dopamine_start = int(0.8 * self.odor_duration)  # 80% through trial
            dopamine_end = min(self.odor_duration, dopamine_start + 10)
            dopamine[dopamine_start:dopamine_end] = 1.0

        return dopamine

    def __len__(self) -> int:
        return len(self.fly_ids)

    def __getitem__(self, idx: int) -> Tuple[List, str]:
        """Return entire sequence for one fly.

        Returns
        -------
        Tuple[List, str]
            (fly_sequence, fly_id) where fly_sequence is list of
            (odor_seq, dopamine, label, trial_info) tuples
        """
        fly_id = self.fly_ids[idx]
        return self.fly_sequences[fly_id], fly_id

    def get_subset(self, fly_ids: List[str]) -> 'SequentialBehavioralDataset':
        """Create a subset dataset with only specified flies."""
        subset = SequentialBehavioralDataset.__new__(SequentialBehavioralDataset)
        subset.n_pn = self.n_pn
        subset.odor_duration = self.odor_duration
        subset.use_noise = self.use_noise
        subset.noise_std = self.noise_std
        subset.fly_sequences = {fid: self.fly_sequences[fid] for fid in fly_ids if fid in self.fly_sequences}
        subset.fly_ids = list(subset.fly_sequences.keys())
        return subset


def train_one_epoch(
    model: CCBPNWithRecurrentContext,
    dataset: SequentialBehavioralDataset,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """Train for one epoch on sequential data.

    Parameters
    ----------
    model : CCBPNWithRecurrentContext
        Model to train
    dataset : SequentialBehavioralDataset
        Training dataset
    criterion : nn.Module
        Loss function (e.g., BCELoss)
    optimizer : Optimizer
        Optimizer
    device : torch.device
        Device to use
    max_grad_norm : float
        Maximum gradient norm for clipping

    Returns
    -------
    Tuple[float, float]
        (average_loss, average_accuracy)
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_trials = 0

    # Shuffle flies (but keep within-fly order)
    fly_indices = torch.randperm(len(dataset)).tolist()

    for idx in tqdm(fly_indices, desc="Training flies", leave=False):
        fly_sequence, fly_id = dataset[idx]

        # Reset context for new fly
        hidden_state = None
        previous_outcome = None

        # Process trials in order for this fly
        for trial_idx, (odor_seq, dopamine, label, trial_info) in enumerate(fly_sequence):
            # Convert to tensors
            odor_tensor = torch.tensor(odor_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, time, n_pn)
            dopamine_tensor = torch.tensor(dopamine, dtype=torch.float32).unsqueeze(0).to(device)  # (1, time)
            label_tensor = torch.tensor([label], dtype=torch.float32).to(device)  # (1,)

            # Forward pass with recurrent context
            outputs = model(
                odor_sequences=odor_tensor,
                dopamine_signals=dopamine_tensor,
                hidden_state=hidden_state,
                previous_outcome=previous_outcome,
            )

            prediction = outputs['behavioral_output']
            new_hidden_state = outputs['hidden_state']

            # Compute loss
            loss = criterion(prediction, label_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for RNNs!)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            # Update for next trial
            previous_outcome = label_tensor.detach()

            # Detach hidden state to prevent backprop through entire sequence
            # (Truncated BPTT - important for long sequences!)
            hidden_state = tuple(h.detach() for h in new_hidden_state)

            # Metrics
            total_loss += loss.item()
            correct = ((prediction > 0.5).float() == label_tensor).float().sum().item()
            total_correct += correct
            total_trials += 1

    avg_loss = total_loss / max(total_trials, 1)
    avg_acc = total_correct / max(total_trials, 1)

    return avg_loss, avg_acc


def validate(
    model: CCBPNWithRecurrentContext,
    dataset: SequentialBehavioralDataset,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model on sequential data.

    Parameters
    ----------
    model : CCBPNWithRecurrentContext
        Model to validate
    dataset : SequentialBehavioralDataset
        Validation dataset
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to use

    Returns
    -------
    Tuple[float, float]
        (average_loss, average_accuracy)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_trials = 0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Validating", leave=False):
            fly_sequence, fly_id = dataset[idx]

            # Reset context for new fly
            hidden_state = None
            previous_outcome = None

            # Process trials in order
            for odor_seq, dopamine, label, trial_info in fly_sequence:
                # Convert to tensors
                odor_tensor = torch.tensor(odor_seq, dtype=torch.float32).unsqueeze(0).to(device)
                dopamine_tensor = torch.tensor(dopamine, dtype=torch.float32).unsqueeze(0).to(device)
                label_tensor = torch.tensor([label], dtype=torch.float32).to(device)

                # Forward pass
                outputs = model(
                    odor_sequences=odor_tensor,
                    dopamine_signals=dopamine_tensor,
                    hidden_state=hidden_state,
                    previous_outcome=previous_outcome,
                )

                prediction = outputs['behavioral_output']
                hidden_state = outputs['hidden_state']

                # Compute loss
                loss = criterion(prediction, label_tensor)

                # Update for next trial
                previous_outcome = label_tensor

                # Metrics
                total_loss += loss.item()
                correct = ((prediction > 0.5).float() == label_tensor).float().sum().item()
                total_correct += correct
                total_trials += 1

    avg_loss = total_loss / max(total_trials, 1)
    avg_acc = total_correct / max(total_trials, 1)

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data parameters
    parser.add_argument('--behavioral-data', type=str, required=True,
                        help='Path to behavioral data CSV')
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Path to FlyWire connectivity cache')

    # Model parameters
    parser.add_argument('--n-pn', type=int, default=150,
                        help='Number of projection neurons')
    parser.add_argument('--n-kc', type=int, default=2000,
                        help='Number of Kenyon cells')
    parser.add_argument('--n-mbon', type=int, default=44,
                        help='Number of MBONs')
    parser.add_argument('--kc-sparsity', type=float, default=0.05,
                        help='KC activation sparsity')
    parser.add_argument('--context-dim', type=int, default=64,
                        help='Context embedding dimension')
    parser.add_argument('--no-gate', action='store_true',
                        help='Disable context gating')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay for regularization')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use class weights for imbalanced data')
    parser.add_argument('--use-lr-scheduler', action='store_true',
                        help='Use learning rate scheduler')

    # Cross-validation
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of cross-validation folds')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Debugging
    parser.add_argument('--max-flies', type=int, default=None,
                        help='Limit to N flies for quick testing')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    logger.info("Loading behavioral data...")
    full_dataset = SequentialBehavioralDataset(
        behavioral_csv=args.behavioral_data,
        n_pn=args.n_pn,
        odor_duration=50,
        use_noise=True,
        noise_std=0.08,
    )

    # Limit flies if debugging
    if args.max_flies is not None:
        fly_ids = full_dataset.fly_ids[:args.max_flies]
        full_dataset = full_dataset.get_subset(fly_ids)
        logger.info(f"Limited to {len(full_dataset)} flies for testing")

    # Setup cross-validation
    logger.info(f"Setting up {args.n_folds}-fold cross-validation...")
    kfold = GroupKFold(n_splits=args.n_folds)

    # Get fly groups for splitting
    fly_groups = np.arange(len(full_dataset.fly_ids))

    # Store results
    fold_results = []

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(fly_groups, groups=fly_groups)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold + 1}/{args.n_folds}")
        logger.info(f"{'='*60}")

        # Create train/val splits
        train_flies = [full_dataset.fly_ids[i] for i in train_idx]
        val_flies = [full_dataset.fly_ids[i] for i in val_idx]

        train_dataset = full_dataset.get_subset(train_flies)
        val_dataset = full_dataset.get_subset(val_flies)

        logger.info(f"Train: {len(train_dataset)} flies, Val: {len(val_dataset)} flies")

        # Initialize model
        model = CCBPNWithRecurrentContext(
            n_pn=args.n_pn,
            n_kc=args.n_kc,
            n_mbon=args.n_mbon,
            cache_dir=args.cache_dir,
            kc_sparsity=args.kc_sparsity,
            context_dim=args.context_dim,
            use_gate=not args.no_gate,
            dropout=args.dropout,
        ).to(device)

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Setup optimizer
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Setup loss (with class weights if requested)
        if args.use_class_weights:
            # Compute class weights from training data
            labels = []
            for fly_seq, _ in [train_dataset[i] for i in range(len(train_dataset))]:
                labels.extend([label for _, _, label, _ in fly_seq])
            pos_weight = (len(labels) - sum(labels)) / max(sum(labels), 1)
            criterion = nn.BCELoss(weight=torch.tensor([pos_weight]).to(device))
            logger.info(f"Using class weights: pos_weight={pos_weight:.3f}")
        else:
            criterion = nn.BCELoss()

        # Setup learning rate scheduler
        if args.use_lr_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        else:
            scheduler = None

        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        train_history = []
        val_history = []

        for epoch in range(args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_dataset, criterion, optimizer, device, args.max_grad_norm
            )

            # Validate
            val_loss, val_acc = validate(model, val_dataset, criterion, device)

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            train_history.append({'epoch': epoch + 1, 'loss': train_loss, 'acc': train_acc})
            val_history.append({'epoch': epoch + 1, 'loss': val_loss, 'acc': val_acc})

            # Update learning rate
            if scheduler is not None:
                scheduler.step(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, output_dir / f'best_model_fold{fold}.pt')
                logger.info(f"✓ Saved best model (val_acc={val_acc:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, output_dir / f'checkpoint_fold{fold}_epoch{epoch+1}.pt')

        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'train_history': train_history,
            'val_history': val_history,
        })

        # Save fold results
        with open(output_dir / f'fold{fold}_results.json', 'w') as f:
            json.dump(fold_results[-1], f, indent=2)

    # Compute overall statistics
    val_accs = [fold['best_val_acc'] for fold in fold_results]
    logger.info(f"\n{'='*60}")
    logger.info("Cross-Validation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Mean Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    logger.info(f"Best Val Acc: {np.max(val_accs):.4f}")
    logger.info(f"Min Val Acc: {np.min(val_accs):.4f}")

    # Save overall results
    overall_results = {
        'args': vars(args),
        'fold_results': fold_results,
        'summary': {
            'mean_val_acc': float(np.mean(val_accs)),
            'std_val_acc': float(np.std(val_accs)),
            'best_val_acc': float(np.max(val_accs)),
            'min_val_acc': float(np.min(val_accs)),
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(overall_results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
