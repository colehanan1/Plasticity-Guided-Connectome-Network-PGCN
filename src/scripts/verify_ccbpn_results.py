#!/usr/bin/env python
"""Verify and analyze CCBPN recurrent training results.

This script checks:
1. Are the results real or is the model cheating?
2. How many samples were tested?
3. Is the model actually using context?
4. Performance breakdown by dataset
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
from pgcn.models.ccbpn_recurrent import CCBPNWithRecurrentContext


def load_results(results_dir: str):
    """Load training results from directory."""
    results_path = Path(results_dir)

    # Load overall results
    with open(results_path / 'results.json', 'r') as f:
        results = json.load(f)

    return results


def analyze_results(results_dir: str):
    """Analyze training results and check for issues."""
    print("="*70)
    print("CCBPN Recurrent Context - Results Analysis")
    print("="*70)

    results = load_results(results_dir)

    # 1. Check overall performance
    print("\n📊 Overall Performance:")
    print(f"   Mean Accuracy: {results['summary']['mean_val_acc']:.1%}")
    print(f"   Std Dev:       {results['summary']['std_val_acc']:.1%}")
    print(f"   Best Fold:     {results['summary']['best_val_acc']:.1%}")
    print(f"   Worst Fold:    {results['summary']['min_val_acc']:.1%}")

    # Check if too good to be true
    mean_acc = results['summary']['mean_val_acc']
    std_acc = results['summary']['std_val_acc']

    if mean_acc >= 0.99:
        print("\n⚠️  WARNING: 99%+ accuracy is suspiciously high!")
        print("   Possible issues:")
        print("   - Model might be overfitting")
        print("   - Data leakage (train/val contamination)")
        print("   - Test set too small/easy")
        print("   - Labels might be predictable from metadata")

    if std_acc == 0.0:
        print("\n⚠️  WARNING: Zero variance across folds!")
        print("   This is extremely unusual. Possible causes:")
        print("   - All validation sets are identical")
        print("   - Model always predicts the same thing")
        print("   - Something is wrong with evaluation")

    # 2. Check each fold
    print("\n📋 Fold-by-Fold Breakdown:")
    print("   Fold | Train Acc | Val Acc | Epochs | Early Stop?")
    print("   " + "-"*55)

    for fold_result in results['fold_results']:
        fold = fold_result['fold']
        best_val = fold_result['best_val_acc']

        # Get final training accuracy
        if fold_result['train_history']:
            final_train = fold_result['train_history'][-1]['acc']
            n_epochs = len(fold_result['train_history'])
        else:
            final_train = 0.0
            n_epochs = 0

        # Check if early stopped
        early_stopped = n_epochs < results['args']['epochs']

        print(f"   {fold:4d} | {final_train:9.1%} | {best_val:7.1%} | {n_epochs:6d} | {'Yes' if early_stopped else 'No'}")

    # 3. Check for overfitting
    print("\n🔍 Overfitting Check:")
    for fold_result in results['fold_results']:
        fold = fold_result['fold']
        train_history = fold_result['train_history']
        val_history = fold_result['val_history']

        if train_history and val_history:
            final_train_acc = train_history[-1]['acc']
            final_val_acc = val_history[-1]['acc']
            gap = final_train_acc - final_val_acc

            print(f"   Fold {fold}: Train={final_train_acc:.1%}, Val={final_val_acc:.1%}, Gap={gap:+.1%}")

            if gap > 0.15:
                print(f"             ⚠️  Large train-val gap suggests overfitting!")

    # 4. Count samples
    print("\n📈 Dataset Info:")
    args = results['args']
    print(f"   Total flies:    {args.get('max_flies', 'Unknown') if args.get('max_flies') else 'All (120)'}")
    print(f"   Cross-val:      {args['n_folds']}-fold")
    print(f"   Context dim:    {args['context_dim']}")
    print(f"   Learning rate:  {args['lr']}")
    print(f"   Max epochs:     {args['epochs']}")

    # 5. Learning curves
    print("\n📉 Learning Curves (Fold 1):")
    fold1 = results['fold_results'][0]

    if fold1['val_history']:
        print("   Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
        print("   " + "-"*60)

        # Show first, middle, and last epochs
        epochs_to_show = [0, len(fold1['val_history'])//2, -1]

        for idx in epochs_to_show:
            train = fold1['train_history'][idx]
            val = fold1['val_history'][idx]

            print(f"   {train['epoch']:5d} | {train['loss']:10.4f} | {train['acc']:9.1%} | "
                  f"{val['loss']:8.4f} | {val['acc']:7.1%}")

    # 6. Recommendations
    print("\n💡 Recommendations:")

    if mean_acc >= 0.99 and std_acc == 0.0:
        print("   ❌ Results look suspicious - investigate further")
        print("   → Check if validation data is leaking into training")
        print("   → Try on a completely held-out test set")
        print("   → Verify labels are correct")
    elif mean_acc >= 0.85:
        print("   ✅ Excellent performance!")
        print("   → Compare with baseline (~70%)")
        print("   → Visualize what the model learned")
        print("   → Try on new data to confirm generalization")
    elif mean_acc >= 0.75:
        print("   ✅ Good performance!")
        print("   → Meets target improvement (+5-10pp over baseline)")
        print("   → Ready for publication")
    else:
        print("   ⚠️  Performance below expectations")
        print("   → Try longer training")
        print("   → Increase context dimension")
        print("   → Check hyperparameters")

    return results


def check_model_predictions(results_dir: str, model_path: str = None):
    """Load model and check what it's actually predicting."""
    results_path = Path(results_dir)

    # Find best model checkpoint
    if model_path is None:
        checkpoints = list(results_path.glob('best_model_fold*.pt'))
        if not checkpoints:
            print("\n⚠️  No model checkpoints found!")
            return
        model_path = checkpoints[0]

    print(f"\n🔬 Analyzing model predictions from: {model_path}")

    # Load results to get args
    with open(results_path / 'results.json', 'r') as f:
        results = json.load(f)
    args = results['args']

    # Load model
    model = CCBPNWithRecurrentContext(
        n_pn=args['n_pn'],
        n_kc=args['n_kc'],
        n_mbon=args['n_mbon'],
        cache_dir=args['cache_dir'],
        kc_sparsity=args['kc_sparsity'],
        context_dim=args['context_dim'],
    )

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Checkpoint from epoch {checkpoint['epoch']}")
    print(f"   ✓ Checkpoint val_acc: {checkpoint['val_acc']:.1%}")

    # Test with dummy input to see if context matters
    print("\n🧪 Testing context effect:")

    # Create dummy odor sequence
    odor = torch.randn(1, 50, args['n_pn'])
    dopamine = torch.ones(1, 50)

    # Trial 1: No context
    with torch.no_grad():
        out1 = model(odor, dopamine, hidden_state=None, previous_outcome=None)
        pred1 = out1['behavioral_output'].item()

    # Trial 2: With context (previous outcome = 1)
    with torch.no_grad():
        out2 = model(odor, dopamine,
                    hidden_state=out1['hidden_state'],
                    previous_outcome=torch.tensor([1.0]))
        pred2 = out2['behavioral_output'].item()

    # Trial 3: With context (previous outcome = 0)
    with torch.no_grad():
        out3 = model(odor, dopamine,
                    hidden_state=out1['hidden_state'],
                    previous_outcome=torch.tensor([0.0]))
        pred3 = out3['behavioral_output'].item()

    print(f"   Trial 1 (no context):                {pred1:.3f}")
    print(f"   Trial 2 (context + outcome=1):       {pred2:.3f}")
    print(f"   Trial 3 (context + outcome=0):       {pred3:.3f}")

    context_effect = abs(pred2 - pred3)
    print(f"\n   Context effect: {context_effect:.4f}")

    if context_effect > 0.01:
        print("   ✅ Model IS using context (predictions change based on history)")
    else:
        print("   ⚠️  Model NOT using context effectively")

    # Check gate values
    gate1 = out1['gate_value'].item()
    gate2 = out2['gate_value'].item()

    print(f"\n   Gate values (how much to use memory):")
    print(f"   Trial 1: {gate1:.3f}")
    print(f"   Trial 2: {gate2:.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_results.py <results_directory>")
        print("\nExample:")
        print("  python verify_results.py results/ccbpn_recurrent_final")
        sys.exit(1)

    results_dir = sys.argv[1]

    # Check if directory exists
    if not Path(results_dir).exists():
        print(f"❌ Directory not found: {results_dir}")
        sys.exit(1)

    # Analyze results
    results = analyze_results(results_dir)

    # Check model predictions
    try:
        check_model_predictions(results_dir)
    except Exception as e:
        print(f"\n⚠️  Could not load model: {e}")
        print("   (This is OK - results analysis above is still valid)")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()
