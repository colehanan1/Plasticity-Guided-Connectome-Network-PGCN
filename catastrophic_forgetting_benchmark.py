"""
Catastrophic Forgetting Benchmark: Biological Veto Gate vs ML Protection Strategies

This module implements a biologically-constrained neural network modeling the
Drosophila PN→KC→MBON pathway to test catastrophic forgetting mitigation strategies.

Architecture:
- Input: 51 Projection Neurons (PNs) encoding odor identity
- Hidden: 2000 Kenyon Cells (KCs) with sparse, fixed connectivity and top-k activation
- Output: 44 Mushroom Body Output Neurons (MBONs) for behavioral readout

Protection Strategies:
1. Baseline: No protection (standard gradient descent)
2. Veto Gate: Biological mechanism protecting 2.6% of critical synapses
3. Synaptic Freezing: Freeze top 2.6% weights by magnitude
4. EWC: Elastic Weight Consolidation with Fisher Information regularization

Author: Generated with Claude Code
Date: 2025-11-21
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import copy


# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class NetworkConfig:
    """Configuration parameters for the PN→KC→MBON network."""
    n_pn: int = 51          # Number of Projection Neurons (glomeruli)
    n_kc: int = 2000        # Number of Kenyon Cells
    n_mbon: int = 44        # Number of Mushroom Body Output Neurons
    pn_kc_sparsity: float = 0.03  # PN→KC connection probability
    kc_topk_frac: float = 0.05    # Fraction of KCs that activate (top-k)
    veto_frac: float = 0.026      # Fraction of synapses protected by veto gate
    learning_rate: float = 0.01
    task1_epochs: int = 200
    task2_epochs: int = 200
    device: str = 'cpu'


class DrosophilaOlfactoryNetwork(nn.Module):
    """
    Biologically-constrained feedforward network modeling Drosophila olfaction.

    Architecture:
        PN (input) → KC (sparse expansion) → MBON (readout)

    Key features:
    - Fixed, sparse PN→KC connectivity (no learning)
    - Top-k winner-take-all sparsification at KC layer
    - Learnable KC→MBON synapses (plasticity site)
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Layer 1: PN → KC (fixed, sparse connectivity)
        self.W_PK = self._initialize_pn_kc_weights()

        # Layer 2: KC → MBON (learnable)
        self.W_KM = nn.Linear(config.n_kc, config.n_mbon, bias=True)
        nn.init.xavier_normal_(self.W_KM.weight, gain=0.5)
        nn.init.zeros_(self.W_KM.bias)

        # Top-k parameter for KC sparsification
        self.k_kc = int(config.n_kc * config.kc_topk_frac)

        self.to(self.device)

    def _initialize_pn_kc_weights(self) -> torch.Tensor:
        """
        Initialize fixed PN→KC connectivity matrix.

        Uses sparse random connectivity (3% connection probability) to mimic
        biological connectome. In production, this would load real FlyWire data.

        Returns:
            Fixed weight matrix W_PK of shape (n_kc, n_pn)
        """
        cfg = self.config

        # Create sparse random connectivity mask
        mask = torch.rand(cfg.n_kc, cfg.n_pn) < cfg.pn_kc_sparsity

        # Initialize weights where connections exist
        W_PK = torch.randn(cfg.n_kc, cfg.n_pn) * mask.float()

        # Normalize each KC's input weights to unit norm (biological constraint)
        row_norms = W_PK.norm(dim=1, keepdim=True)
        row_norms[row_norms == 0] = 1.0  # Avoid division by zero
        W_PK = W_PK / row_norms

        # Make non-trainable
        W_PK = W_PK.to(self.device)
        W_PK.requires_grad = False

        return W_PK

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input odor vector (batch_size, n_pn)

        Returns:
            h: KC activations after top-k sparsification (batch_size, n_kc)
            y: MBON output (batch_size, n_mbon)
        """
        # PN → KC: sparse projection
        h_raw = torch.matmul(x, self.W_PK.T)  # (batch, n_kc)

        # Apply ReLU
        h_raw = torch.relu(h_raw)

        # Top-k sparsification (winner-take-all)
        h = self._topk_sparsify(h_raw)

        # KC → MBON: learnable readout
        y = self.W_KM(h)

        return h, y

    def _topk_sparsify(self, h_raw: torch.Tensor) -> torch.Tensor:
        """
        Apply top-k winner-take-all sparsification.

        Only the top k% of KCs remain active; others are set to zero.
        This mimics biological lateral inhibition in the mushroom body.

        Args:
            h_raw: Raw KC activations (batch_size, n_kc)

        Returns:
            Sparsified activations (batch_size, n_kc)
        """
        # Get top-k values and indices
        topk_vals, topk_idxs = torch.topk(h_raw, self.k_kc, dim=1)

        # Create sparse tensor with only top-k values
        h_sparse = torch.zeros_like(h_raw)
        h_sparse.scatter_(1, topk_idxs, topk_vals)

        return h_sparse


class ProtectionStrategy:
    """Base class for synaptic protection strategies."""

    def __init__(self, network: DrosophilaOlfactoryNetwork):
        self.network = network
        self.task1_weights = None
        self.protection_mask = None

    def after_task1(self, x_task1: torch.Tensor, y_task1: torch.Tensor):
        """Called after task 1 training to set up protection."""
        # Save task 1 weights
        self.task1_weights = copy.deepcopy(self.network.W_KM.weight.data)

    def get_loss_modifier(self, loss: torch.Tensor) -> torch.Tensor:
        """Modify loss during task 2 training (e.g., for EWC regularization)."""
        return loss

    def apply_weight_constraints(self):
        """Apply weight constraints after gradient step (e.g., for veto gate)."""
        pass

    def get_name(self) -> str:
        """Return strategy name for plotting."""
        raise NotImplementedError


class BaselineStrategy(ProtectionStrategy):
    """No protection - standard gradient descent."""

    def get_name(self) -> str:
        return "Baseline (No Protection)"


class VetoGateStrategy(ProtectionStrategy):
    """
    Biological veto gate mechanism (Or7a pathway).

    Identifies and protects the most critical 2.6% of KC→MBON synapses
    from modification during task 2 training.
    """

    def after_task1(self, x_task1: torch.Tensor, y_task1: torch.Tensor):
        super().after_task1(x_task1, y_task1)

        # Identify critical synapses by weight magnitude
        weight_magnitudes = torch.abs(self.task1_weights)

        # Flatten and get top 2.6% indices
        n_protect = int(self.network.config.veto_frac * weight_magnitudes.numel())
        flat_weights = weight_magnitudes.flatten()
        _, top_indices = torch.topk(flat_weights, n_protect)

        # Create protection mask (1 = protected, 0 = modifiable)
        self.protection_mask = torch.zeros_like(weight_magnitudes).flatten()
        self.protection_mask[top_indices] = 1.0
        self.protection_mask = self.protection_mask.reshape(weight_magnitudes.shape)

        print(f"Veto Gate: Protecting {n_protect} synapses ({self.network.config.veto_frac*100:.1f}%)")

    def apply_weight_constraints(self):
        """Restore protected synapses to their task 1 values."""
        if self.protection_mask is not None:
            with torch.no_grad():
                # Restore protected weights: W = W_task1 * mask + W_current * (1 - mask)
                self.network.W_KM.weight.data = (
                    self.task1_weights * self.protection_mask +
                    self.network.W_KM.weight.data * (1 - self.protection_mask)
                )

    def get_name(self) -> str:
        return "Veto Gate (Biological)"


class SynapticFreezingStrategy(ProtectionStrategy):
    """
    ML baseline: Freeze top 2.6% of synapses by magnitude.

    Similar to veto gate but applied uniformly without biological constraint.
    """

    def after_task1(self, x_task1: torch.Tensor, y_task1: torch.Tensor):
        super().after_task1(x_task1, y_task1)

        # Identify top synapses by magnitude (same as veto gate)
        weight_magnitudes = torch.abs(self.task1_weights)
        n_freeze = int(self.network.config.veto_frac * weight_magnitudes.numel())
        flat_weights = weight_magnitudes.flatten()
        _, top_indices = torch.topk(flat_weights, n_freeze)

        # Create freeze mask
        self.protection_mask = torch.zeros_like(weight_magnitudes).flatten()
        self.protection_mask[top_indices] = 1.0
        self.protection_mask = self.protection_mask.reshape(weight_magnitudes.shape)

        print(f"Synaptic Freezing: Freezing {n_freeze} synapses ({self.network.config.veto_frac*100:.1f}%)")

    def apply_weight_constraints(self):
        """Restore frozen synapses to their task 1 values."""
        if self.protection_mask is not None:
            with torch.no_grad():
                self.network.W_KM.weight.data = (
                    self.task1_weights * self.protection_mask +
                    self.network.W_KM.weight.data * (1 - self.protection_mask)
                )

    def get_name(self) -> str:
        return "Synaptic Freezing (ML)"


class EWCStrategy(ProtectionStrategy):
    """
    Elastic Weight Consolidation.

    Adds L2 regularization weighted by Fisher Information to penalize
    changes to weights important for task 1.
    """

    def __init__(self, network: DrosophilaOlfactoryNetwork, ewc_lambda: float = 400.0):
        super().__init__(network)
        self.ewc_lambda = ewc_lambda
        self.fisher_information = None

    def after_task1(self, x_task1: torch.Tensor, y_task1: torch.Tensor):
        super().after_task1(x_task1, y_task1)

        # Compute Fisher Information Matrix (diagonal approximation)
        self.fisher_information = self._compute_fisher(x_task1, y_task1)

        print(f"EWC: Computed Fisher Information (λ={self.ewc_lambda})")

    def _compute_fisher(self, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal Fisher Information Matrix.

        Fisher ≈ E[∇log p(y|x; θ)²] for each weight.
        """
        self.network.eval()

        # Forward pass
        _, y_pred = self.network(x)

        # Use MSE loss for Fisher computation
        loss = nn.functional.mse_loss(y_pred, y_target)

        # Compute gradients
        self.network.zero_grad()
        loss.backward()

        # Fisher is squared gradient (diagonal approximation)
        fisher = self.network.W_KM.weight.grad.data.clone() ** 2

        self.network.train()

        return fisher

    def get_loss_modifier(self, loss: torch.Tensor) -> torch.Tensor:
        """Add EWC regularization penalty to task 2 loss."""
        if self.fisher_information is not None and self.task1_weights is not None:
            # EWC penalty: (λ/2) * Σ F_ij * (w_ij - w_ij^*)²
            weight_diff = self.network.W_KM.weight - self.task1_weights
            ewc_penalty = (self.ewc_lambda / 2) * torch.sum(
                self.fisher_information * (weight_diff ** 2)
            )
            return loss + ewc_penalty
        return loss

    def get_name(self) -> str:
        return f"EWC (λ={self.ewc_lambda})"


def generate_mock_odors(config: NetworkConfig) -> Dict[str, torch.Tensor]:
    """
    Generate mock odor representations.

    In production, these would be loaded from DoOR database.

    Returns:
        Dictionary mapping odor names to PN activation vectors
    """
    device = torch.device(config.device)

    # Benzaldehyde: strong Or7a activation (index 0)
    benzaldehyde = torch.zeros(1, config.n_pn, device=device)
    benzaldehyde[0, 0] = 1.0  # Or7a
    benzaldehyde[0, 1:4] = torch.tensor([0.2, 0.15, 0.1], device=device)

    # Hexanol: weak Or7a, strong Or67b activation (index 5)
    hexanol = torch.zeros(1, config.n_pn, device=device)
    hexanol[0, 5] = 1.0  # Or67b
    hexanol[0, 6:9] = torch.tensor([0.3, 0.25, 0.15], device=device)
    hexanol[0, 0] = 0.05  # Minimal Or7a overlap

    # Normalize
    benzaldehyde = benzaldehyde / (benzaldehyde.norm() + 1e-8)
    hexanol = hexanol / (hexanol.norm() + 1e-8)

    return {
        'benzaldehyde': benzaldehyde,
        'hexanol': hexanol
    }


def train_task(
    network: DrosophilaOlfactoryNetwork,
    x_odor: torch.Tensor,
    y_target: torch.Tensor,
    n_epochs: int,
    optimizer: optim.Optimizer,
    strategy: Optional[ProtectionStrategy] = None,
    task_name: str = "Task"
) -> List[float]:
    """
    Train network on a single odor-reward association task.

    Args:
        network: The olfactory network
        x_odor: Input odor vector (1, n_pn)
        y_target: Target MBON output (1, n_mbon)
        n_epochs: Number of training epochs
        optimizer: PyTorch optimizer
        strategy: Protection strategy (if task 2)
        task_name: Name for logging

    Returns:
        List of loss values per epoch
    """
    network.train()
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass
        _, y_pred = network(x_odor)

        # Compute loss
        loss = loss_fn(y_pred, y_target)

        # Apply protection strategy modifications (e.g., EWC penalty)
        if strategy is not None:
            loss = strategy.get_loss_modifier(loss)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Apply weight constraints (e.g., veto gate restoration)
        if strategy is not None:
            strategy.apply_weight_constraints()

        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"{task_name} - Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    return losses


def evaluate_retention(
    network: DrosophilaOlfactoryNetwork,
    x_odor: torch.Tensor,
    baseline_response: torch.Tensor
) -> float:
    """
    Compute retention metric for an odor.

    Retention = current_response / baseline_response (normalized)

    Args:
        network: Trained network
        x_odor: Input odor vector
        baseline_response: MBON response after initial training

    Returns:
        Retention score (1.0 = full retention, 0.0 = complete forgetting)
    """
    network.eval()
    with torch.no_grad():
        _, current_response = network(x_odor)

        # Normalize by baseline (element-wise)
        retention = torch.abs(current_response / (baseline_response + 1e-8))

        # Average across MBON population
        retention_score = retention.mean().item()

    return retention_score


def run_catastrophic_forgetting_experiment(
    config: NetworkConfig,
    strategy: ProtectionStrategy
) -> Dict[str, any]:
    """
    Run complete sequential learning experiment with one protection strategy.

    Protocol:
        1. Train on odor A (benzaldehyde) + reward
        2. Record MBON response to A
        3. Train on odor B (hexanol) + reward (with protection)
        4. Test retention of odor A

    Args:
        config: Network configuration
        strategy: Protection strategy to test

    Returns:
        Dictionary containing results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {strategy.get_name()}")
    print(f"{'='*60}\n")

    # Initialize network
    network = DrosophilaOlfactoryNetwork(config)
    strategy.network = network

    # Generate odors
    odors = generate_mock_odors(config)
    x_A = odors['benzaldehyde']
    x_B = odors['hexanol']

    # Target: approach behavior (positive MBON output)
    y_target = torch.ones(1, config.n_mbon, device=torch.device(config.device))

    # Optimizer
    optimizer = optim.Adam(network.W_KM.parameters(), lr=config.learning_rate)

    # === TASK 1: Learn Odor A (Benzaldehyde) ===
    print("\n--- TASK 1: Learning Benzaldehyde + Reward ---")
    task1_losses = train_task(
        network, x_A, y_target, config.task1_epochs, optimizer,
        strategy=None, task_name="Task 1"
    )

    # Evaluate after task 1
    network.eval()
    with torch.no_grad():
        _, y_A_after_task1 = network(x_A)
        _, y_B_before_task2 = network(x_B)

    # Setup protection strategy
    strategy.after_task1(x_A, y_target)

    # === TASK 2: Learn Odor B (Hexanol) ===
    print("\n--- TASK 2: Learning Hexanol + Reward ---")
    task2_losses = train_task(
        network, x_B, y_target, config.task2_epochs, optimizer,
        strategy=strategy, task_name="Task 2"
    )

    # === TEST PHASE: Measure Forgetting ===
    print("\n--- TEST PHASE: Measuring Retention ---")
    network.eval()
    with torch.no_grad():
        _, y_A_after_task2 = network(x_A)
        _, y_B_after_task2 = network(x_B)

    # Compute retention metrics
    retention_A = evaluate_retention(network, x_A, y_A_after_task1)
    retention_B = evaluate_retention(network, x_B, y_target)

    # Compute forgetting percentage
    forgetting_A = (1 - retention_A) * 100

    print(f"\nResults for {strategy.get_name()}:")
    print(f"  Retention A (Benzaldehyde): {retention_A*100:.1f}%")
    print(f"  Forgetting A: {forgetting_A:.1f}%")
    print(f"  Retention B (Hexanol): {retention_B*100:.1f}%")

    return {
        'strategy_name': strategy.get_name(),
        'task1_losses': task1_losses,
        'task2_losses': task2_losses,
        'y_A_after_task1': y_A_after_task1.cpu().numpy(),
        'y_A_after_task2': y_A_after_task2.cpu().numpy(),
        'y_B_after_task2': y_B_after_task2.cpu().numpy(),
        'retention_A': retention_A,
        'retention_B': retention_B,
        'forgetting_A': forgetting_A,
        'weight_changes': (network.W_KM.weight.data - strategy.task1_weights).cpu().numpy(),
        'protection_mask': strategy.protection_mask.cpu().numpy() if strategy.protection_mask is not None else None
    }


def plot_results(all_results: List[Dict], config: NetworkConfig, save_path: str = "forgetting_benchmark.png"):
    """
    Generate comprehensive visualization of catastrophic forgetting benchmark.

    Creates 4-panel figure:
        1. MBON population histograms (before/after tasks)
        2. Training loss curves
        3. Retention comparison bar chart
        4. Weight change heatmaps

    Args:
        all_results: List of result dictionaries from each strategy
        config: Network configuration
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))

    # Panel 1: MBON Population Histograms
    ax1 = plt.subplot(2, 3, 1)
    for result in all_results:
        y_after_task1 = result['y_A_after_task1'].flatten()
        y_after_task2 = result['y_A_after_task2'].flatten()

        ax1.hist(y_after_task2, bins=30, alpha=0.5, label=result['strategy_name'])

    ax1.set_xlabel('MBON Output', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('MBON Population Response to Odor A\n(After Task 2)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: Training Loss Curves
    ax2 = plt.subplot(2, 3, 2)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for i, result in enumerate(all_results):
        epochs_task1 = range(1, len(result['task1_losses']) + 1)
        epochs_task2 = range(len(result['task1_losses']) + 1,
                            len(result['task1_losses']) + len(result['task2_losses']) + 1)

        ax2.plot(epochs_task1, result['task1_losses'], color=colors[i],
                linestyle='--', alpha=0.6, linewidth=1.5)
        ax2.plot(epochs_task2, result['task2_losses'], color=colors[i],
                label=result['strategy_name'], linewidth=2)

    ax2.axvline(config.task1_epochs, color='red', linestyle=':', linewidth=2,
               label='Task Switch', alpha=0.7)
    ax2.set_xlabel('Training Epoch', fontsize=11)
    ax2.set_ylabel('Loss (MSE)', fontsize=11)
    ax2.set_title('Training Loss Curves', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')

    # Panel 3: Retention Bar Chart
    ax3 = plt.subplot(2, 3, 3)
    strategy_names = [r['strategy_name'] for r in all_results]
    retention_scores = [r['retention_A'] * 100 for r in all_results]

    bars = ax3.barh(strategy_names, retention_scores, color=colors)
    ax3.set_xlabel('Retention (%)', fontsize=11)
    ax3.set_title('Odor A Retention After Task 2', fontsize=12, fontweight='bold')
    ax3.axvline(100, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Retention')
    ax3.set_xlim(0, 120)
    ax3.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, retention_scores)):
        ax3.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

    # Panel 4: Forgetting Percentage
    ax4 = plt.subplot(2, 3, 4)
    forgetting_scores = [r['forgetting_A'] for r in all_results]
    bars2 = ax4.barh(strategy_names, forgetting_scores, color=colors)
    ax4.set_xlabel('Forgetting (%)', fontsize=11)
    ax4.set_title('Catastrophic Forgetting (Odor A)', fontsize=12, fontweight='bold')
    ax4.invert_xaxis()  # Lower is better
    ax4.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars2, forgetting_scores)):
        ax4.text(val - 2, i, f'{val:.1f}%', va='center', ha='right', fontsize=10, fontweight='bold')

    # Panel 5: Weight Change Heatmap (Veto Gate)
    ax5 = plt.subplot(2, 3, 5)
    veto_result = next((r for r in all_results if 'Veto' in r['strategy_name']), all_results[0])
    weight_changes = veto_result['weight_changes']

    im1 = ax5.imshow(np.abs(weight_changes), aspect='auto', cmap='hot', interpolation='nearest')
    ax5.set_xlabel('KC Index (sample)', fontsize=10)
    ax5.set_ylabel('MBON Index', fontsize=10)
    ax5.set_title(f'Weight Changes: {veto_result["strategy_name"]}', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax5, label='|ΔW|')

    # Panel 6: Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    table_data = []
    table_data.append(['Strategy', 'Retention A', 'Retention B', 'Forgetting A'])
    for r in all_results:
        table_data.append([
            r['strategy_name'][:20],  # Truncate long names
            f"{r['retention_A']*100:.1f}%",
            f"{r['retention_B']*100:.1f}%",
            f"{r['forgetting_A']:.1f}%"
        ])

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax6.set_title('Quantitative Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.close()


def main():
    """
    Main experiment runner.

    Compares four protection strategies on catastrophic forgetting benchmark:
        1. Baseline (no protection)
        2. Veto Gate (biological Or7a mechanism)
        3. Synaptic Freezing (ML baseline)
        4. EWC (Elastic Weight Consolidation)
    """
    print("="*80)
    print("CATASTROPHIC FORGETTING BENCHMARK")
    print("Biological Veto Gate vs ML Protection Strategies")
    print("="*80)

    # Configuration
    config = NetworkConfig(
        n_pn=51,
        n_kc=2000,
        n_mbon=44,
        pn_kc_sparsity=0.03,
        kc_topk_frac=0.05,
        veto_frac=0.026,
        learning_rate=0.01,
        task1_epochs=200,
        task2_epochs=200,
        device='cpu'
    )

    print(f"\nNetwork Configuration:")
    print(f"  PNs: {config.n_pn}")
    print(f"  KCs: {config.n_kc} (top-{config.kc_topk_frac*100:.0f}% active)")
    print(f"  MBONs: {config.n_mbon}")
    print(f"  Protection: {config.veto_frac*100:.1f}% of synapses")
    print(f"  Training: {config.task1_epochs} + {config.task2_epochs} epochs")

    # Run experiments for each strategy
    all_results = []

    # 1. Baseline
    network_baseline = DrosophilaOlfactoryNetwork(config)
    strategy_baseline = BaselineStrategy(network_baseline)
    results_baseline = run_catastrophic_forgetting_experiment(config, strategy_baseline)
    all_results.append(results_baseline)

    # 2. Veto Gate
    network_veto = DrosophilaOlfactoryNetwork(config)
    strategy_veto = VetoGateStrategy(network_veto)
    results_veto = run_catastrophic_forgetting_experiment(config, strategy_veto)
    all_results.append(results_veto)

    # 3. Synaptic Freezing
    network_freeze = DrosophilaOlfactoryNetwork(config)
    strategy_freeze = SynapticFreezingStrategy(network_freeze)
    results_freeze = run_catastrophic_forgetting_experiment(config, strategy_freeze)
    all_results.append(results_freeze)

    # 4. EWC
    network_ewc = DrosophilaOlfactoryNetwork(config)
    strategy_ewc = EWCStrategy(network_ewc, ewc_lambda=400.0)
    results_ewc = run_catastrophic_forgetting_experiment(config, strategy_ewc)
    all_results.append(results_ewc)

    # Generate comparison visualizations
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    plot_results(all_results, config)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    best_strategy = max(all_results, key=lambda x: x['retention_A'])
    print(f"\nBest Protection Strategy: {best_strategy['strategy_name']}")
    print(f"  Retention: {best_strategy['retention_A']*100:.1f}%")
    print(f"  Forgetting Reduction: {best_strategy['forgetting_A']:.1f}%")

    # Compare veto gate vs baseline
    veto_forgetting = next(r['forgetting_A'] for r in all_results if 'Veto' in r['strategy_name'])
    baseline_forgetting = next(r['forgetting_A'] for r in all_results if 'Baseline' in r['strategy_name'])
    improvement = ((baseline_forgetting - veto_forgetting) / baseline_forgetting) * 100

    print(f"\nVeto Gate vs Baseline:")
    print(f"  Baseline Forgetting: {baseline_forgetting:.1f}%")
    print(f"  Veto Gate Forgetting: {veto_forgetting:.1f}%")
    print(f"  Improvement: {improvement:.1f}% reduction in forgetting")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
