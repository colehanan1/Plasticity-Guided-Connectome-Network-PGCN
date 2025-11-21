"""Sanity check tests for CCBPN with recurrent context memory.

These tests verify that:
1. Hidden state propagation works correctly
2. Gradients flow through the LSTM
3. Context affects predictions
4. Model can overfit a small dataset (sanity check)
"""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pgcn.models.ccbpn_recurrent import CCBPNWithRecurrentContext


def test_hidden_state_propagation():
    """Test 1: Verify that hidden state carries information across trials."""
    print("\n" + "="*70)
    print("Test 1: Hidden State Propagation")
    print("="*70)

    # Create dummy model (no cache_dir needed for this test)
    model = CCBPNWithRecurrentContext(
        n_pn=50,
        n_kc=200,
        n_mbon=10,
        cache_dir=None,  # Will use random weights
        kc_sparsity=0.05,
        context_dim=32,
    )
    model.eval()

    # Create dummy sequence
    batch_size = 1
    time_steps = 20
    odor1 = torch.randn(batch_size, time_steps, 50)
    dopamine1 = torch.ones(batch_size, time_steps)

    # First trial (no context)
    out1 = model(odor1, dopamine1, hidden_state=None, previous_outcome=None)
    h1 = out1['hidden_state']
    pred1_no_context = out1['behavioral_output'].item()

    # Second trial with same input but WITH context from first trial
    out2_with_context = model(odor1, dopamine1, hidden_state=h1, previous_outcome=torch.tensor([1.0]))
    pred2_with_context = out2_with_context['behavioral_output'].item()

    # Second trial with same input but WITHOUT context (reset)
    out2_no_context = model(odor1, dopamine1, hidden_state=None, previous_outcome=None)
    pred2_no_context = out2_no_context['behavioral_output'].item()

    print(f"Trial 1 (no context):        {pred1_no_context:.4f}")
    print(f"Trial 2 (with context):      {pred2_with_context:.4f}")
    print(f"Trial 2 (without context):   {pred2_no_context:.4f}")

    # Predictions should differ when context is present
    context_effect = abs(pred2_with_context - pred2_no_context)
    print(f"\nContext effect: {context_effect:.6f}")

    if context_effect > 1e-4:
        print("✅ PASS: Context affects predictions")
        return True
    else:
        print("❌ FAIL: Context is not affecting predictions!")
        return False


def test_gradient_flow():
    """Test 2: Verify gradients flow through LSTM."""
    print("\n" + "="*70)
    print("Test 2: Gradient Flow")
    print("="*70)

    # Create model
    model = CCBPNWithRecurrentContext(
        n_pn=50,
        n_kc=200,
        n_mbon=10,
        cache_dir=None,
        kc_sparsity=0.05,
        context_dim=32,
    )
    model.train()

    # Create dummy inputs
    odor = torch.randn(1, 20, 50, requires_grad=False)
    dopamine = torch.ones(1, 20)
    label = torch.tensor([1.0])

    # Forward pass
    outputs = model(odor, dopamine, hidden_state=None, previous_outcome=None)
    prediction = outputs['behavioral_output']

    # Compute loss
    loss = nn.BCELoss()(prediction, label)

    # Backward pass
    loss.backward()

    # Check LSTM has gradients
    lstm_has_grad = False
    lstm_grad_norm = 0.0
    for name, param in model.context_memory.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                lstm_has_grad = True
                lstm_grad_norm += grad_norm

    print(f"LSTM gradient norm: {lstm_grad_norm:.6f}")

    # Check context modulation has gradients
    context_has_grad = False
    context_grad_norm = 0.0
    for param in model.context_modulation.parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                context_has_grad = True
                context_grad_norm += grad_norm

    print(f"Context modulation gradient norm: {context_grad_norm:.6f}")

    if lstm_has_grad and context_has_grad:
        print("✅ PASS: Gradients flow to LSTM and context modulation")
        return True
    else:
        print("❌ FAIL: No gradients flowing to recurrent components!")
        print(f"  LSTM has grad: {lstm_has_grad}")
        print(f"  Context has grad: {context_has_grad}")
        return False


def test_context_learning():
    """Test 3: Verify model can learn to use context."""
    print("\n" + "="*70)
    print("Test 3: Context Learning (Overfitting Check)")
    print("="*70)

    # Create model
    model = CCBPNWithRecurrentContext(
        n_pn=50,
        n_kc=200,
        n_mbon=10,
        cache_dir=None,
        kc_sparsity=0.05,
        context_dim=32,
    )
    model.train()

    # Create a simple task: predict based on previous outcome
    # Trial 1: Random odor → outcome = 0
    # Trial 2: Same odor → outcome = 1 if previous was 0, else 0
    # This requires using context (previous outcome) to predict correctly

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # Training data: 10 sequences, each with 5 trials
    n_sequences = 10
    trials_per_sequence = 5

    print("\nTraining model to use context...")
    initial_loss = None
    final_loss = None

    for epoch in range(50):
        total_loss = 0.0

        for seq_idx in range(n_sequences):
            # Reset context for new sequence
            hidden_state = None
            previous_outcome = None

            # Random odor for this sequence
            odor_pattern = torch.randn(1, 20, 50)

            for trial_idx in range(trials_per_sequence):
                # Label depends on previous outcome
                if previous_outcome is None:
                    label = torch.tensor([0.0])  # First trial always 0
                else:
                    # Flip previous outcome
                    label = 1.0 - previous_outcome

                # Forward
                outputs = model(
                    odor_pattern,
                    torch.ones(1, 20),
                    hidden_state=hidden_state,
                    previous_outcome=previous_outcome,
                )

                prediction = outputs['behavioral_output']
                new_hidden_state = outputs['hidden_state']

                # Loss
                loss = criterion(prediction, label)
                total_loss += loss.item()

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update for next trial
                previous_outcome = label
                hidden_state = tuple(h.detach() for h in new_hidden_state)

        avg_loss = total_loss / (n_sequences * trials_per_sequence)

        if epoch == 0:
            initial_loss = avg_loss
        if epoch == 49:
            final_loss = avg_loss

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d}: Loss = {avg_loss:.4f}")

    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Improvement:  {initial_loss - final_loss:.4f}")

    if final_loss < initial_loss * 0.5:
        print("✅ PASS: Model learned to use context (loss decreased by >50%)")
        return True
    else:
        print("❌ FAIL: Model did not learn effectively")
        return False


def test_model_forward_shapes():
    """Test 4: Verify output shapes are correct."""
    print("\n" + "="*70)
    print("Test 4: Output Shape Verification")
    print("="*70)

    model = CCBPNWithRecurrentContext(
        n_pn=50,
        n_kc=200,
        n_mbon=10,
        cache_dir=None,
        kc_sparsity=0.05,
        context_dim=32,
    )
    model.eval()

    batch_size = 4
    time_steps = 20
    n_pn = 50

    odor = torch.randn(batch_size, time_steps, n_pn)
    dopamine = torch.ones(batch_size, time_steps)

    outputs = model(odor, dopamine)

    print(f"Input shapes:")
    print(f"  odor: {odor.shape}")
    print(f"  dopamine: {dopamine.shape}")

    print(f"\nOutput shapes:")
    print(f"  behavioral_output: {outputs['behavioral_output'].shape}")
    print(f"  context: {outputs['context'].shape}")
    print(f"  gate_value: {outputs['gate_value'].shape}")
    print(f"  mbon_output: {outputs['mbon_output'].shape}")

    # Check shapes
    checks = [
        (outputs['behavioral_output'].shape == (batch_size,), "behavioral_output"),
        (outputs['context'].shape == (batch_size, 32), "context"),
        (outputs['gate_value'].shape == (batch_size, 1), "gate_value"),
        (outputs['mbon_output'].shape == (batch_size, time_steps, 10), "mbon_output"),
        (outputs['hidden_state'][0].shape == (1, batch_size, 32), "hidden_state[0]"),
        (outputs['hidden_state'][1].shape == (1, batch_size, 32), "hidden_state[1]"),
    ]

    all_pass = True
    for check, name in checks:
        if check:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} - WRONG SHAPE!")
            all_pass = False

    if all_pass:
        print("\n✅ PASS: All output shapes correct")
        return True
    else:
        print("\n❌ FAIL: Some output shapes incorrect")
        return False


def test_reset_context():
    """Test 5: Verify context reset functionality."""
    print("\n" + "="*70)
    print("Test 5: Context Reset")
    print("="*70)

    model = CCBPNWithRecurrentContext(
        n_pn=50,
        n_kc=200,
        n_mbon=10,
        cache_dir=None,
        kc_sparsity=0.05,
        context_dim=32,
    )

    # Reset context
    h, c = model.reset_context(batch_size=4)

    print(f"Reset hidden state shapes:")
    print(f"  h: {h.shape}")
    print(f"  c: {c.shape}")

    # Check all zeros
    h_zero = torch.allclose(h, torch.zeros_like(h))
    c_zero = torch.allclose(c, torch.zeros_like(c))

    print(f"\nHidden state all zeros: {h_zero}")
    print(f"Cell state all zeros: {c_zero}")

    if h_zero and c_zero and h.shape == (1, 4, 32):
        print("✅ PASS: Context reset works correctly")
        return True
    else:
        print("❌ FAIL: Context reset incorrect")
        return False


def run_all_tests():
    """Run all sanity check tests."""
    print("\n" + "="*70)
    print("CCBPN Recurrent Context - Sanity Check Tests")
    print("="*70)

    tests = [
        ("Output Shape Verification", test_model_forward_shapes),
        ("Context Reset", test_reset_context),
        ("Hidden State Propagation", test_hidden_state_propagation),
        ("Gradient Flow", test_gradient_flow),
        ("Context Learning", test_context_learning),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! Model is ready for training.")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please review.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
