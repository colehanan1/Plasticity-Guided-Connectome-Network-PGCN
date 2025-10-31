# PGCN → FlyWire Codex Visualization: Complete Execution Guide

## Overview

This guide walks you through generating and executing a **FlyWire-integrated visualization** of your PGCN circuit using **Codex (FlyOne)** to produce publication-ready connectome renderings.

**Two files to use:**
1. `quick-chatgpt-prompt.md` — **START HERE** (copy/paste into ChatGPT)
2. `flywire-codex-prompt.md` — Detailed technical reference

---

## 🚀 Quick Start (5 Minutes)

### Phase 1: Generate the Script (With ChatGPT/Claude)

1. Open **ChatGPT** (or Claude, or Perplexity)
2. Navigate to [**quick-chatgpt-prompt.md**](quick-chatgpt-prompt.md)
3. Copy the entire prompt block under "⚡ Use This Direct Prompt"
4. Paste into ChatGPT and hit Send

**What to expect:** ChatGPT will generate a complete, working Python script within 2–3 minutes.

### Phase 2: Save and Test the Script

```bash
# Copy the generated code and save it:
vim scripts/visualize_pgcn_connectome.py

# Or use cat heredoc:
cat > scripts/visualize_pgcn_connectome.py << 'EOF'
[paste the full script from ChatGPT here]
EOF

# Test that it runs:
python scripts/visualize_pgcn_connectome.py --help
```

**Expected output:**
```
usage: visualize_pgcn_connectome.py [-h] [--cache-dir CACHE_DIR] [--output-dir OUTPUT_DIR] ...
```

### Phase 3: Install Dependencies (if needed)

```bash
pip install plotly networkx pandas pyarrow
```

Or add to `pyproject.toml` / `requirements.txt`:
```
plotly>=5.0
networkx>=3.0
pandas>=1.3
pyarrow>=10.0
```

### Phase 4: Run the Visualization

```bash
# Basic usage (reads data/cache/, outputs to reports/):
python scripts/visualize_pgcn_connectome.py --cache-dir data/cache --output-dir reports

# Advanced: Show top 300 edges, color by glomerulus:
python scripts/visualize_pgcn_connectome.py \
  --cache-dir data/cache \
  --output-dir reports \
  --top-edges 300 \
  --neuron-colors-by glomerulus \
  --include-dan

# Test mode (uses synthetic small graph, no cache needed):
python scripts/visualize_pgcn_connectome.py --test-mode
```

### Phase 5: View and Share Outputs

**Three outputs are generated:**

#### A) Interactive HTML Visualization (`reports/pgcn_circuit_interactive.html`)
```bash
# Open in your browser:
open reports/pgcn_circuit_interactive.html
# or
firefox reports/pgcn_circuit_interactive.html
```

**Features:**
- 4-layer hierarchical network (PN | KC | MBON | DAN)
- Hover for neuron metadata
- Zoom/pan for exploration
- Edge thickness ∝ synapse count
- Ready for publication or sharing

#### B) FlyWire Manifest JSON (`reports/pgcn_circuit_manifest.json`)
```json
{
  "neurons_by_type": {
    "PN": ["648518346496101905", "648518346496101906", ...],
    "KC": [...],
    "MBON": [...],
    "DAN": [...]
  },
  "instructions": "Follow steps 1-6 below to load in FlyWire...",
  "edges": [...]
}
```

**Use this to load into FlyWire (see below).**

#### C) Connectivity Statistics (`reports/circuit_stats.csv`)
```
layer,node_count,edge_count,avg_in_degree,avg_out_degree,sparsity
PN,153,12456,0.0,81.4,0.02
KC,5177,12456,2.4,23.7,0.00094
MBON,96,10284,107.1,0.0,0.11
DAN,584,3245,0.0,5.6,0.009
```

---

## 🎯 Visualization Output Examples

### HTML Graph Layout
```
                            DANs
                          (y = 3)
                            |
                          MBONs
                          (y = 1)
                         /    |    \
                        /     |     \
                       /      |      \
                      KCs
                    (y = -1)
                   /  |  |  \
                  /   |  |   \
                 /    |  |    \
               PNs
             (y = -3)
```

**Visual encoding:**
- Node **size** = in/out degree
- Node **color** = type (or glomerulus for PNs, subtype for KCs)
- Edge **thickness** = synapse_weight
- Edge **opacity** = normalized frequency

---

## 🌐 Load into FlyWire Web Viewer (Codex Integration)

Once you have `pgcn_circuit_manifest.json`, you can load neurons into the **FlyWire 3D viewer** for publication-quality renders.

### Option 1: Manual Load via FlyWire Web UI

1. Go to **https://flywire.ai/**
2. Login with Google account
3. Click **"Segment ID Loader"** (or search bar)
4. Open `pgcn_circuit_manifest.json` in a text editor
5. Copy the first `PN` root ID list
6. Paste into FlyWire loader
7. Repeat for KC, MBON, DAN
8. Use FlyWire's **color picker** to assign per-type colors:
   - PNs → Blue
   - KCs → Green
   - MBONs → Yellow
   - DANs → Red
9. Enable **"Brain mesh"** for anatomical context (spacebar to toggle)
10. Export 3D render or take screenshot

### Option 2: Programmatic Load (via Codex API, if available)

If Codex exposes a REST API for batch neuron loading, you can extend the script with:

```python
def load_into_codex(manifest_json, flywire_session=None):
    """
    POST manifest neurons to Codex API for direct visualization.
    Requires valid FlyWire credentials.
    """
    # This would be implemented if Codex API is available
    pass
```

*Contact FlyWire team for API access.*

---

## 📊 Integration with PGCN Workflow

### Add to Makefile

```makefile
.PHONY: visualize
visualize:
	python scripts/visualize_pgcn_connectome.py \
	  --cache-dir data/cache \
	  --output-dir reports \
	  --top-edges 500 \
	  --neuron-colors-by type

.PHONY: visualize-by-glomerulus
visualize-by-glomerulus:
	python scripts/visualize_pgcn_connectome.py \
	  --cache-dir data/cache \
	  --output-dir reports \
	  --top-edges 500 \
	  --neuron-colors-by glomerulus

.PHONY: visualize-test
visualize-test:
	python scripts/visualize_pgcn_connectome.py --test-mode
```

Then run:
```bash
make visualize
make visualize-by-glomerulus
make visualize-test
```

### Add to CI/CD (GitHub Actions, etc.)

```yaml
- name: Generate circuit visualizations
  run: |
    python scripts/visualize_pgcn_connectome.py --cache-dir data/cache --output-dir reports
    
- name: Upload visualization artifacts
  uses: actions/upload-artifact@v3
  with:
    name: circuit-visualizations
    path: reports/pgcn_circuit_*.html
```

### Add to README

```markdown
## Visualize the Circuit

Generate interactive network graphs and FlyWire manifests:

```bash
# Quick start
python scripts/visualize_pgcn_connectome.py --cache-dir data/cache --output-dir reports

# Explore interactively
open reports/pgcn_circuit_interactive.html
```

See `reports/pgcn_circuit_manifest.json` for loading neurons into FlyWire.
```

---

## 🔧 Troubleshooting

### Issue: "FileNotFoundError: data/cache/nodes.parquet not found"

**Solution:** Ensure your cache is populated:
```bash
ls -la data/cache/
# Should show: nodes.parquet, edges.parquet, alpn_extracted.csv, etc.

# If missing, run the extraction pipeline:
python scripts/extract_alpn_projection_neurons.py
python scripts/extract_kc_neurons.py
```

### Issue: "plotly not installed"

**Solution:**
```bash
pip install plotly>=5.0
```

### Issue: HTML file is huge / slow to load

**Solution:** Reduce `--top-edges`:
```bash
python scripts/visualize_pgcn_connectome.py --top-edges 200  # default 500
```

### Issue: "Can't see neuron IDs in FlyWire manifest"

**Solution:** Check JSON syntax:
```bash
python -m json.tool reports/pgcn_circuit_manifest.json | head -20
```

---

## 📚 Reference

### FlyWire Resources
- **FlyWire web viewer:** https://flywire.ai/
- **Codex explorer:** https://codex.flywire.ai/
- **FAFB dataset:** FAFB v783 (139,255 neurons, FAFB voxel coordinates)
- **Neuroglancer:** Backend 3D viewer (handles mesh rendering)

### FlyWire Addons for Enhanced Visualization
- **Batch Processor:** https://blog.flywire.ai/2022/08/11/flywire-addons/
- **Bulk color tool:** Assign colors to many neurons at once
- **Connectivity explorer:** Find upstream/downstream partners

### PGCN Context
- **PN → KC → MBON + DAN** olfactory learning circuit
- **~353 PNs** across 50–58 glomeruli
- **~5,177 KCs** (8 subtypes)
- **~96 MBONs** (output neurons)
- **~584 DANs** (dopaminergic modulators)

---

## 🎓 Next Steps

### Immediate
- [ ] Use `quick-chatgpt-prompt.md` to generate the script
- [ ] Run `python scripts/visualize_pgcn_connectome.py`
- [ ] View HTML output in browser
- [ ] Load manifest into FlyWire

### Short-term
- [ ] Add visualization target to Makefile
- [ ] Integrate into CI/CD pipeline
- [ ] Customize node colors/labels for your specific experiments
- [ ] Share interactive HTML with collaborators

### Long-term
- [ ] Animate circuit dynamics (layer-by-layer activation during learning)
- [ ] Overlay experimental data (Ca²⁺ imaging, behavioral outcomes)
- [ ] Generate figure-quality renders for publications
- [ ] Create Shapley-value edge importance ranking for veto mechanisms (Exps 1–6)

---

## 📝 Questions?

If the script generation fails or you hit blockers:

1. **Check the detailed prompt:** See `flywire-codex-prompt.md` for full technical specs
2. **Ask ChatGPT for help:** Paste error messages into ChatGPT with context
3. **Consult FlyWire docs:** https://blog.flywire.ai/
4. **Reach out:** Contact FlyWire team at flywire@princeton.edu for API questions

---

**Good luck! 🚀**
