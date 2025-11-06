# Monday Experiment Results

## Experiment: veto_gate

### Key Results

- **Blocking Index**: 0.994
  - > 0: Blocking successful (OdorB learned more than OdorA)
  - ≈ 0: No blocking (equal learning)
  - < 0: Blocking failed (OdorA learned more)

- **Veto Efficacy**: 1.000
  - Measures how strongly veto pathway activates (0-1 range)

- **Mean Gating Suppression**: 1.000
  - Fraction of plasticity blocked by veto (0-1 range)

### Test Responses

- **DA1**: 63.920
- **DL3**: 0.205



### Files Generated

1. **experiment_summary.json** - JSON summary of all metrics
2. **phase2_trials.csv** - Trial-by-trial data for Phase 2
3. **experiment_1_veto_gate_results.png** - Visualization plots
4. **README.md** - This file

### Interpretation

✓ **Blocking effect detected!** The GABAergic veto successfully suppressed learning in the blocked pathway.
