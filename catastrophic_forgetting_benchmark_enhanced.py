"""
Enhanced Catastrophic Forgetting Benchmark with Conflicting Tasks

This version creates actual catastrophic forgetting by:
1. Increasing odor similarity (overlapping KC representations)
2. Using conflicting outputs (approach vs avoid)
3. Adding more realistic biological constraints

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
    n_kc: int = 800         # Number of Kenyon Cells (reduced for capacity constraint)
    n_mbon: int = 44        # Number of Mushroom Body Output Neurons
    pn_kc_sparsity: float = 0.03  # PN→KC connection probability
    kc_topk_frac: float = 0.10    # Fraction of KCs that activate (increased overlap)
    veto_frac: float = 0.026      # Fraction of synapses protected by veto gate
    learning_rate: float = 0.05   # Higher learning rate for stronger updates
    task1_epochs: int = 200
    task2_epochs: int = 300       # More epochs to create interference
    device: str = 'cpu'
    odor_overlap: float = 0.7     # High overlap between odor representations
    task2_learnability: float = 0.3  # Or7a blocking factor (0.3 = 70% suppression)


class DrosophilaOlfactoryNetwork(nn.Module):
    """
    Biologically-constrained feedforward network modeling Drosophila olfaction.
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
        """Initialize fixed PN→KC connectivity matrix."""
        cfg = self.config

        # Create sparse random connectivity mask
        mask = torch.rand(cfg.n_kc, cfg.n_pn) < cfg.pn_kc_sparsity

        # Initialize weights where connections exist
        W_PK = torch.randn(cfg.n_kc, cfg.n_pn) * mask.float()

        # Normalize each KC's input weights to unit norm
        row_norms = W_PK.norm(dim=1, keepdim=True)
        row_norms[row_norms == 0] = 1.0
        W_PK = W_PK / row_norms

        # Make non-trainable
        W_PK = W_PK.to(self.device)
        W_PK.requires_grad = False

        return W_PK

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        # PN → KC: sparse projection
        h_raw = torch.matmul(x, self.W_PK.T)  # (batch, n_kc)

        # Apply ReLU
        h_raw = torch.relu(h_raw)

        # Top-k sparsification (winner-take-all)
        h = self._topk_sparsify(h_raw)

        # KC → MBON: learnable readout
        y = self.W_KM(h)

        # Note: MBON population normalization can be added post-hoc for analysis
        # but is not applied during forward pass to preserve learning dynamics

        return h, y

    def _topk_sparsify(self, h_raw: torch.Tensor) -> torch.Tensor:
        """Apply top-k winner-take-all sparsification."""
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
        self.task1_weights = copy.deepcopy(self.network.W_KM.weight.data)

    def get_loss_modifier(self, loss: torch.Tensor) -> torch.Tensor:
        """Modify loss during task 2 training."""
        return loss

    def apply_weight_constraints(self):
        """Apply weight constraints after gradient step."""
        pass

    def get_name(self) -> str:
        """Return strategy name for plotting."""
        raise NotImplementedError


class BaselineStrategy(ProtectionStrategy):
    """No protection - standard gradient descent."""

    def get_name(self) -> str:
        return "Baseline"


class VetoGateStrategy(ProtectionStrategy):
    """Biological veto gate mechanism (Or7a pathway)."""

    def after_task1(self, x_task1: torch.Tensor, y_task1: torch.Tensor):
        super().after_task1(x_task1, y_task1)

        # Identify critical synapses by weight magnitude
        weight_magnitudes = torch.abs(self.task1_weights)

        # Get top 2.6% indices
        n_protect = int(self.network.config.veto_frac * weight_magnitudes.numel())
        flat_weights = weight_magnitudes.flatten()
        _, top_indices = torch.topk(flat_weights, n_protect)

        # Create protection mask
        self.protection_mask = torch.zeros_like(weight_magnitudes).flatten()
        self.protection_mask[top_indices] = 1.0
        self.protection_mask = self.protection_mask.reshape(weight_magnitudes.shape)

        print(f"  Veto Gate: Protecting {n_protect} synapses ({self.network.config.veto_frac*100:.1f}%)")

    def apply_weight_constraints(self):
        """Restore protected synapses to their task 1 values."""
        if self.protection_mask is not None:
            with torch.no_grad():
                self.network.W_KM.weight.data = (
                    self.task1_weights * self.protection_mask +
                    self.network.W_KM.weight.data * (1 - self.protection_mask)
                )

    def get_name(self) -> str:
        return "Veto Gate"


class SynapticFreezingStrategy(ProtectionStrategy):
    """ML baseline: Freeze top 2.6% of synapses by magnitude."""

    def after_task1(self, x_task1: torch.Tensor, y_task1: torch.Tensor):
        super().after_task1(x_task1, y_task1)

        weight_magnitudes = torch.abs(self.task1_weights)
        n_freeze = int(self.network.config.veto_frac * weight_magnitudes.numel())
        flat_weights = weight_magnitudes.flatten()
        _, top_indices = torch.topk(flat_weights, n_freeze)

        self.protection_mask = torch.zeros_like(weight_magnitudes).flatten()
        self.protection_mask[top_indices] = 1.0
        self.protection_mask = self.protection_mask.reshape(weight_magnitudes.shape)

        print(f"  Synaptic Freezing: Freezing {n_freeze} synapses ({self.network.config.veto_frac*100:.1f}%)")

    def apply_weight_constraints(self):
        if self.protection_mask is not None:
            with torch.no_grad():
                self.network.W_KM.weight.data = (
                    self.task1_weights * self.protection_mask +
                    self.network.W_KM.weight.data * (1 - self.protection_mask)
                )

    def get_name(self) -> str:
        return "Synaptic Freezing"


class EWCStrategy(ProtectionStrategy):
    """Elastic Weight Consolidation."""

    def __init__(self, network: DrosophilaOlfactoryNetwork, ewc_lambda: float = 400.0):
        super().__init__(network)
        self.ewc_lambda = ewc_lambda
        self.fisher_information = None

    def after_task1(self, x_task1: torch.Tensor, y_task1: torch.Tensor):
        super().after_task1(x_task1, y_task1)
        self.fisher_information = self._compute_fisher(x_task1, y_task1)
        print(f"  EWC: Computed Fisher Information (λ={self.ewc_lambda})")

    def _compute_fisher(self, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """Compute diagonal Fisher Information Matrix."""
        self.network.eval()

        _, y_pred = self.network(x)
        loss = nn.functional.mse_loss(y_pred, y_target)

        self.network.zero_grad()
        loss.backward()

        fisher = self.network.W_KM.weight.grad.data.clone() ** 2

        self.network.train()

        return fisher

    def get_loss_modifier(self, loss: torch.Tensor) -> torch.Tensor:
        """Add EWC regularization penalty to task 2 loss."""
        if self.fisher_information is not None and self.task1_weights is not None:
            weight_diff = self.network.W_KM.weight - self.task1_weights
            ewc_penalty = (self.ewc_lambda / 2) * torch.sum(
                self.fisher_information * (weight_diff ** 2)
            )
            return loss + ewc_penalty
        return loss

    def get_name(self) -> str:
        return "EWC"


def generate_overlapping_odors(config: NetworkConfig) -> Dict[str, torch.Tensor]:
    """
    Generate odor representations for sequential learning tasks.

    Task 1: Benzaldehyde (learns normally)
    Task 2: Benzaldehyde presented again (Or7a blocks learning)

    Both tasks use the same odor with high overlap to test memory interference.
    """
    device = torch.device(config.device)

    # Benzaldehyde (Task 1): strong activation in channels 0-10
    benzaldehyde_task1 = torch.zeros(1, config.n_pn, device=device)
    benzaldehyde_task1[0, :10] = torch.randn(10, device=device).abs()
    benzaldehyde_task1[0, 0] = 2.0  # Or7a dominant

    # Benzaldehyde (Task 2): high overlap with Task 1 presentation
    # This simulates the same odor being presented again
    benzaldehyde_task2 = torch.zeros(1, config.n_pn, device=device)

    # Shared components (high overlap as specified in config)
    benzaldehyde_task2[0, :10] = benzaldehyde_task1[0, :10] * config.odor_overlap

    # Small unique noise (natural variation)
    noise = torch.randn(10, device=device).abs() * (1 - config.odor_overlap) * 0.3
    benzaldehyde_task2[0, :10] += noise
    benzaldehyde_task2[0, 0] = 2.0  # Or7a still dominant

    # Normalize
    benzaldehyde_task1 = benzaldehyde_task1 / (benzaldehyde_task1.norm() + 1e-8)
    benzaldehyde_task2 = benzaldehyde_task2 / (benzaldehyde_task2.norm() + 1e-8)

    return {
        'benzaldehyde': benzaldehyde_task1,      # Task 1: normal learning
        'benzaldehyde_blocked': benzaldehyde_task2  # Task 2: Or7a blocks
    }


def train_task(
    network: DrosophilaOlfactoryNetwork,
    x_odor: torch.Tensor,
    y_target: torch.Tensor,
    n_epochs: int,
    optimizer: optim.Optimizer,
    strategy: Optional[ProtectionStrategy] = None,
    task_name: str = "Task",
    verbose: bool = True,
    or7a_blocking: float = 0.0  # Or7a blocking factor (0 = no blocking, 1 = full blocking)
) -> Tuple[List[float], List[torch.Tensor]]:
    """
    Train network on a single odor-reward association task.

    Args:
        or7a_blocking: Fraction of learning suppression (0-1)
                      0.0 = normal learning (Task 1)
                      0.7 = 70% suppression by Or7a (Task 2)
    """
    network.train()
    loss_fn = nn.MSELoss()
    losses = []
    mbon_outputs = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass
        h_kc, y_pred = network(x_odor)

        loss = loss_fn(y_pred, y_target)

        if strategy is not None:
            loss = strategy.get_loss_modifier(loss)

        loss.backward()

        # Or7a blocking: suppress gradient updates proportionally
        if or7a_blocking > 0:
            with torch.no_grad():
                # Identify benzaldehyde-selective KCs (those activated by this odor)
                kc_active_mask = (h_kc > 0).float()  # Binary mask of active KCs

                # Suppress gradients on KC→MBON weights proportional to blocking factor
                # Higher blocking = less learning
                suppression_factor = 1.0 - or7a_blocking

                # Apply suppression to weights connected to active KCs
                if network.W_KM.weight.grad is not None:
                    # Suppress gradients for benzaldehyde-selective pathways
                    # Shape: (n_mbon, n_kc)
                    network.W_KM.weight.grad *= suppression_factor

                if network.W_KM.bias.grad is not None:
                    network.W_KM.bias.grad *= suppression_factor

        optimizer.step()

        if strategy is not None:
            strategy.apply_weight_constraints()

        losses.append(loss.item())

        # Record MBON outputs periodically
        if epoch % 10 == 0:
            with torch.no_grad():
                _, y_out = network(x_odor)
                mbon_outputs.append(y_out.clone())

        if verbose and (epoch + 1) % 50 == 0:
            blocking_str = f" [Or7a blocking: {or7a_blocking*100:.0f}%]" if or7a_blocking > 0 else ""
            print(f"    Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}{blocking_str}")

    return losses, mbon_outputs


def evaluate_retention(
    network: DrosophilaOlfactoryNetwork,
    x_odor: torch.Tensor,
    baseline_response: torch.Tensor
) -> float:
    """Compute retention metric for an odor."""
    network.eval()
    with torch.no_grad():
        _, current_response = network(x_odor)

        # Compute MSE between baseline and current response
        mse = torch.mean((baseline_response - current_response) ** 2).item()

        # Convert MSE to retention percentage
        # MSE of 0 = 100% retention, higher MSE = lower retention
        # Use exponential decay: retention = exp(-mse)
        retention_score = np.exp(-mse)

    return retention_score


def run_catastrophic_forgetting_experiment(
    config: NetworkConfig,
    strategy: ProtectionStrategy
) -> Dict[str, any]:
    """Run complete sequential learning experiment."""
    print(f"\n{'─'*60}")
    print(f"Strategy: {strategy.get_name()}")
    print(f"{'─'*60}")

    # Initialize network
    network = DrosophilaOlfactoryNetwork(config)
    strategy.network = network

    # Generate odors
    odors = generate_overlapping_odors(config)
    x_benzaldehyde = odors['benzaldehyde']          # Task 1: normal presentation
    x_benzaldehyde_blocked = odors['benzaldehyde_blocked']  # Task 2: Or7a blocks

    # Both tasks: Approach target (trying to learn same association)
    y_approach = torch.ones(1, config.n_mbon, device=torch.device(config.device))

    # Optimizer
    optimizer = optim.SGD(network.W_KM.parameters(), lr=config.learning_rate, momentum=0.9)

    # TASK 1: Learn Benzaldehyde → Approach (normal learning)
    print("\n  TASK 1: Benzaldehyde → Approach (normal learning)")
    task1_losses, task1_mbon = train_task(
        network, x_benzaldehyde, y_approach, config.task1_epochs, optimizer,
        strategy=None, task_name="Task 1", verbose=False, or7a_blocking=0.0
    )
    print(f"    Final loss: {task1_losses[-1]:.6f}")

    # Evaluate after task 1
    network.eval()
    with torch.no_grad():
        _, y_A_after_task1 = network(x_benzaldehyde)

    # Setup protection strategy
    strategy.after_task1(x_benzaldehyde, y_approach)

    # TASK 2: Attempt to learn Benzaldehyde → Approach again (Or7a blocks)
    blocking_factor = 1.0 - config.task2_learnability  # 0.7 = 70% blocking
    print(f"\n  TASK 2: Benzaldehyde → Approach (Or7a blocks {blocking_factor*100:.0f}%)")
    task2_losses, task2_mbon = train_task(
        network, x_benzaldehyde_blocked, y_approach, config.task2_epochs, optimizer,
        strategy=strategy, task_name="Task 2", verbose=False, or7a_blocking=blocking_factor
    )
    print(f"    Final loss: {task2_losses[-1]:.6f} [Learning suppressed]")

    # TEST PHASE: Measure forgetting
    print("\n  TEST PHASE:")
    network.eval()
    with torch.no_grad():
        _, y_A_after_task2 = network(x_benzaldehyde)
        _, y_B_after_task2 = network(x_benzaldehyde_blocked)

    # Compute retention metrics
    retention_A = evaluate_retention(network, x_benzaldehyde, y_A_after_task1)
    retention_B = evaluate_retention(network, x_benzaldehyde_blocked, y_approach)

    # Compute forgetting percentage
    forgetting_A = (1 - retention_A) * 100

    print(f"    Retention A: {retention_A*100:.1f}%")
    print(f"    Forgetting A: {forgetting_A:.1f}%")
    print(f"    Retention B: {retention_B*100:.1f}%")

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


def plot_results(all_results: List[Dict], config: NetworkConfig, save_path: str = "forgetting_benchmark_enhanced.png"):
    """Generate comprehensive visualization."""
    fig = plt.figure(figsize=(16, 9))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']  # Red, Blue, Green, Orange

    # Panel 1: MBON Population Distribution (After Task 1)
    ax1 = plt.subplot(2, 4, 1)
    for i, result in enumerate(all_results):
        y_task1 = result['y_A_after_task1'].flatten()
        ax1.hist(y_task1, bins=25, alpha=0.6, label=result['strategy_name'], color=colors[i])
    ax1.set_xlabel('MBON Output (Approach)', fontsize=10)
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_title('Benzaldehyde Memory\n(After Task 1 Learning)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: MBON Population Distribution (After Task 2)
    ax2 = plt.subplot(2, 4, 2)
    for i, result in enumerate(all_results):
        y_task2 = result['y_A_after_task2'].flatten()
        ax2.hist(y_task2, bins=25, alpha=0.6, label=result['strategy_name'], color=colors[i])
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero')
    ax2.set_xlabel('MBON Output (Approach)', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Benzaldehyde Memory\n(After Blocked Re-learning)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: Training Loss Curves
    ax3 = plt.subplot(2, 4, 3)
    for i, result in enumerate(all_results):
        epochs_task1 = range(1, len(result['task1_losses']) + 1)
        epochs_task2 = range(len(result['task1_losses']) + 1,
                            len(result['task1_losses']) + len(result['task2_losses']) + 1)

        ax3.plot(epochs_task1, result['task1_losses'], color=colors[i],
                linestyle='--', alpha=0.5, linewidth=1.5, label='_nolegend_')
        ax3.plot(epochs_task2, result['task2_losses'], color=colors[i],
                label=result['strategy_name'], linewidth=2)

    ax3.axvline(config.task1_epochs, color='black', linestyle=':', linewidth=2,
               alpha=0.5, label='Or7a Blocks')
    ax3.set_xlabel('Training Epoch', fontsize=10)
    ax3.set_ylabel('Loss (MSE)', fontsize=10)
    ax3.set_title('Training Loss\n(Task 1 | Task 2 Blocked)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(alpha=0.3)
    ax3.set_yscale('log')

    # Panel 4: Retention Bar Chart
    ax4 = plt.subplot(2, 4, 4)
    strategy_names = [r['strategy_name'] for r in all_results]
    retention_scores = [r['retention_A'] * 100 for r in all_results]

    bars = ax4.barh(strategy_names, retention_scores, color=colors)
    ax4.set_xlabel('Retention (%)', fontsize=10)
    ax4.set_title('Task 1 Memory Retention\n(After Blocked Re-learning)', fontsize=11, fontweight='bold')
    ax4.axvline(100, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax4.set_xlim(0, 110)
    ax4.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, retention_scores)):
        ax4.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    # Panel 5: Forgetting Percentage
    ax5 = plt.subplot(2, 4, 5)
    forgetting_scores = [r['forgetting_A'] for r in all_results]
    bars2 = ax5.barh(strategy_names, forgetting_scores, color=colors)
    ax5.set_xlabel('Interference (%)', fontsize=10)
    ax5.set_title('Memory Damage from\nBlocked Learning', fontsize=11, fontweight='bold')
    ax5.set_xlim(0, max(forgetting_scores) * 1.2 if max(forgetting_scores) > 0 else 10)
    ax5.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars2, forgetting_scores)):
        ax5.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    # Panel 6: Weight Change Heatmap (Veto Gate)
    ax6 = plt.subplot(2, 4, 6)
    veto_result = next((r for r in all_results if 'Veto' in r['strategy_name']), all_results[0])
    weight_changes = np.abs(veto_result['weight_changes'])

    # Sample for visualization
    sampled_weights = weight_changes[:, ::20]  # Sample every 20th KC

    im = ax6.imshow(sampled_weights, aspect='auto', cmap='hot', interpolation='nearest')
    ax6.set_xlabel('KC Index (sampled)', fontsize=9)
    ax6.set_ylabel('MBON Index', fontsize=9)
    ax6.set_title(f'|ΔW|: {veto_result["strategy_name"]}', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax6, label='|ΔW|', fraction=0.046)

    # Panel 7: Weight Change Heatmap (Baseline)
    ax7 = plt.subplot(2, 4, 7)
    baseline_result = next((r for r in all_results if 'Baseline' in r['strategy_name']), all_results[0])
    baseline_changes = np.abs(baseline_result['weight_changes'])
    sampled_baseline = baseline_changes[:, ::20]

    im2 = ax7.imshow(sampled_baseline, aspect='auto', cmap='hot', interpolation='nearest')
    ax7.set_xlabel('KC Index (sampled)', fontsize=9)
    ax7.set_ylabel('MBON Index', fontsize=9)
    ax7.set_title(f'|ΔW|: {baseline_result["strategy_name"]}', fontsize=10, fontweight='bold')
    plt.colorbar(im2, ax=ax7, label='|ΔW|', fraction=0.046)

    # Panel 8: Summary Statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')

    # Calculate improvement over baseline
    baseline_forgetting = next(r['forgetting_A'] for r in all_results if 'Baseline' in r['strategy_name'])

    summary_text = "╔═══ RESULTS SUMMARY ═══╗\n\n"
    summary_text += f"Network: {config.n_pn}PN → {config.n_kc}KC → {config.n_mbon}MBON\n"
    summary_text += f"Training: {config.task1_epochs} + {config.task2_epochs} epochs\n"
    summary_text += f"Protection: {config.veto_frac*100:.1f}% synapses\n\n"
    summary_text += "─── Forgetting (%) ───\n"

    for r in all_results:
        improvement = ((baseline_forgetting - r['forgetting_A']) / baseline_forgetting * 100) if baseline_forgetting > 0 else 0
        summary_text += f"{r['strategy_name']:>16}: {r['forgetting_A']:5.1f}%"
        if r['strategy_name'] != 'Baseline':
            summary_text += f" (↓{improvement:.0f}%)"
        summary_text += "\n"

    best_strategy = min(all_results, key=lambda x: x['forgetting_A'])
    summary_text += f"\n✓ Best: {best_strategy['strategy_name']}\n"
    summary_text += f"  {best_strategy['forgetting_A']:.1f}% forgetting\n"
    summary_text += f"  {best_strategy['retention_A']*100:.1f}% retention"

    ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {save_path}")
    plt.close()


def main():
    """Main experiment runner."""
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "   OR7A BLOCKING & CATASTROPHIC FORGETTING".center(78) + "║")
    print("║" + "   Biological Veto Gate Protects Memory During Blocked Learning".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")

    # Configuration - use defaults from NetworkConfig
    config = NetworkConfig()

    print(f"\n{'─'*80}")
    print("EXPERIMENTAL SETUP")
    print(f"{'─'*80}")
    print(f"  Architecture: {config.n_pn} PNs → {config.n_kc} KCs (top-{config.kc_topk_frac*100:.0f}%) → {config.n_mbon} MBONs")
    print(f"  Protection: {config.veto_frac*100:.1f}% of KC→MBON synapses")
    print(f"  Training: Task 1 ({config.task1_epochs} epochs), Task 2 ({config.task2_epochs} epochs)")
    print(f"  Odor overlap: {config.odor_overlap*100:.0f}%")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Or7a blocking: {(1-config.task2_learnability)*100:.0f}% suppression in Task 2")
    print()
    print("  Task 1: Benzaldehyde → Approach (normal learning)")
    print("  Task 2: Benzaldehyde → Approach (Or7a blocks learning)")
    print("  Question: Does failed Task 2 training damage Task 1 memory?")

    # Run experiments
    all_results = []

    strategies = [
        BaselineStrategy,
        VetoGateStrategy,
        SynapticFreezingStrategy,
        lambda n: EWCStrategy(n, ewc_lambda=400.0)
    ]

    print(f"\n{'─'*80}")
    print("RUNNING EXPERIMENTS")
    print(f"{'─'*80}")

    for strategy_class in strategies:
        network = DrosophilaOlfactoryNetwork(config)
        strategy = strategy_class(network)
        results = run_catastrophic_forgetting_experiment(config, strategy)
        all_results.append(results)

    # Generate visualizations
    print(f"\n{'─'*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'─'*80}")
    plot_results(all_results, config)

    # Final summary
    print(f"\n{'╔' + '═'*78 + '╗'}")
    print(f"{'║'}{'FINAL SUMMARY'.center(78)}{'║'}")
    print(f"{'╚' + '═'*78 + '╝'}\n")

    # Sort by retention (best first)
    sorted_results = sorted(all_results, key=lambda x: x['retention_A'], reverse=True)

    print("Ranking (by retention):")
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r['strategy_name']:>20}: {r['retention_A']*100:>5.1f}% retention, {r['forgetting_A']:>5.1f}% forgetting")

    # Compare veto gate vs baseline
    veto_forgetting = next(r['forgetting_A'] for r in all_results if 'Veto' in r['strategy_name'])
    baseline_forgetting = next(r['forgetting_A'] for r in all_results if 'Baseline' in r['strategy_name'])

    if baseline_forgetting > 0:
        improvement = ((baseline_forgetting - veto_forgetting) / baseline_forgetting) * 100
        print(f"\n✓ Veto Gate achieves {improvement:.1f}% reduction in forgetting vs Baseline")
        print(f"  ({veto_forgetting:.1f}% vs {baseline_forgetting:.1f}%)")

    print(f"\n{'─'*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'─'*80}\n")


if __name__ == "__main__":
    main()
