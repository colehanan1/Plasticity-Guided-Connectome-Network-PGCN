# FlyWire Codex Best Practices for PGCN Visualization

## Why FlyWire/Codex for PGCN?

Your PGCN instantiates the **PN → KC → MBON + DAN circuit** extracted directly from **FlyWire FAFB v783**. Using FlyWire's native viewer (Codex) ensures:

1. **Anatomical accuracy** – 3D neuron meshes from the same dataset you extracted connectivity from
2. **Full neuropil context** – see where each layer projects (calyx, medial lobe, pedunculus, etc.)
3. **Community validation** – neurons are proofread by FlyWire community
4. **Publication-ready** – professional 3D renders for figures and supplementary materials
5. **Bidirectional traceability** – your PN/KC/MBON root IDs link back to FlyWire metadata

---

## Workflow: Cache → Visualization → FlyWire

```
┌─────────────────────────────────────────────────────────┐
│ 1. Your PGCN data/cache/ (Parquet/CSV)                 │
│    - nodes.parquet (PN/KC/MBON/DAN root_ids)          │
│    - edges.parquet (connectivity + synapse_counts)    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. visualize_pgcn_connectome.py (Python)               │
│    - Loads cache files                                 │
│    - Builds graph: PN→KC→MBON→DAN                      │
│    - Outputs JSON manifest + HTML viz                  │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌──────────────────────────────┬─────────────┐
        ↓                              ↓             ↓
  ┌──────────────┐           ┌──────────────────┐  ┌──────────────┐
  │  .html       │           │  manifest.json   │  │  stats.csv   │
  │  (Plotly)    │           │  (root IDs)      │  │  (metrics)   │
  │  Local graph │           │  FlyWire loader  │  │              │
  └──────────────┘           └──────────────────┘  └──────────────┘
        ↓                              ↓
   Quick                      FlyWire Web UI
   exploration                (https://flywire.ai/)
                                     ↓
                         ┌─────────────────────┐
                         │ 3D mesh rendering   │
                         │ Color-coded neurons │
                         │ Export to figures   │
                         └─────────────────────┘
```

---

## Key Decisions: Visualization Parameters

### 1. What to Show: Layer-by-Layer Filtering

| Layer | Count | Show All? | Recommendation |
|-------|-------|-----------|-----------------|
| PN    | ~353  | Yes       | All PNs; color by glomerulus |
| KC    | ~5,177| Maybe     | Top 2,000 by in-degree; color by subtype |
| MBON  | ~96   | Yes       | All MBONs; color by output neuropil |
| DAN   | ~584  | Maybe     | Only PAM/PPL1 dopaminergic neurons (color by valence) |

**Rationale:** KCs are extremely sparse (2% PN→KC connectivity); showing all creates visual clutter. DANs are auxiliary; focus on core learning pathway first.

### 2. Edge Filtering: Top-K by Synapse Count

```
--top-edges 500   # Default: show only 500 strongest PN→KC→MBON edges
--top-edges 200   # Minimal: for quick desktop rendering
--top-edges 1000  # Full: for HPC cluster or patience
--top-edges 0     # All edges (warning: may crash browser)
```

**Why:** ~12k PN→KC edges → if you show all, graph becomes unreadable hairball.

### 3. Node Coloring Strategies

#### Option A: By Type (Simplest)
```bash
--neuron-colors-by type
# Output:
#   PN = blue
#   KC = green
#   MBON = yellow
#   DAN = red
```
**Best for:** Overview; showing circuit layers clearly.

#### Option B: By Glomerulus (For PNs)
```bash
--neuron-colors-by glomerulus
# Output:
#   Each PN colored by its olfactory receptor zone
#   ~50-58 colors assigned to glomeruli
```
**Best for:** Understanding olfactory tuning; seeing which glomeruli project strongly to KCs.

#### Option C: By Subtype (For KCs/MBONs)
```bash
--neuron-colors-by subtype
# Output:
#   KCs: α/β lobe (dorsal/ventral), γ lobe, etc.
#   MBONs: input neuropil (calyx vs medial lobe)
```
**Best for:** Anatomical organization; identifying sub-circuits.

---

## FlyWire Manifest: How to Load

### 1. Get Your Manifest

Run the visualization script:
```bash
python scripts/visualize_pgcn_connectome.py --cache-dir data/cache --output-dir reports
```

Output: `reports/pgcn_circuit_manifest.json`

### 2. Inspect the Manifest

```bash
cat reports/pgcn_circuit_manifest.json | python -m json.tool | head -50
```

Expected structure:
```json
{
  "metadata": {
    "circuit": "PN→KC→MBON+DAN",
    "dataset": "FAFB v783",
    "n_neurons": 6210,
    "n_edges": 12456
  },
  "neurons_by_type": {
    "PN": [
      "648518346496101905",
      "648518346496101906",
      ...
    ],
    "KC": [...],
    "MBON": [...],
    "DAN": [...]
  },
  "edge_examples": [
    {"source": "648518346496101905", "target": "648518346489234567", "synapses": 42}
  ],
  "instructions": "..."
}
```

### 3. Load into FlyWire

**Step-by-step:**

1. Navigate to **https://flywire.ai/**
2. Ensure you're in the **production** dataset (FAFB v783 or latest)
3. Click **Search** (magnifying glass icon)
4. Select **"Segment ID Loader"** or paste directly into search bar
5. Copy the PN root IDs from the manifest (e.g., first 10):
   ```
   648518346496101905, 648518346496101906, 648518346496101907, ...
   ```
6. Paste and press Enter; FlyWire will highlight all PNs in blue
7. Wait ~5 seconds for meshes to render
8. Repeat steps 4–7 for KC, MBON, DAN (using different colors)

**Coloring per type:**
- In the left panel, find "Segment List"
- For each type, right-click → "Change color"
- Assign: PN=blue, KC=green, MBON=yellow, DAN=red

### 4. Enhance Visualization

Once neurons are loaded:

- **Toggle "Show slices"** (keyboard: `S`) to see EM sections
- **Toggle "Brain mesh"** (keyboard: `M`) to add anatomical context
- **Zoom in on specific neuropils:** calyx, medial lobe, pedunculus
- **Hover over neurons** to see root ID and synapse count (if available in FlyWire UI)
- **Export screenshot** or **record a fly-through** for presentations

### 5. Save for Publication

```bash
# In FlyWire UI, once happy with visualization:
# Press Ctrl+P or go to menu → Export
# Choose: PNG (screenshot) or MP4 (3D fly-through)
# Save to figures/ folder

# Tip: Use consistent camera angles for multi-panel figures
```

---

## Advanced: Overlaying Experimental Data

Once your 3D circuit is visualized in FlyWire, you can layer on PGCN outputs:

### Overlay 1: Connectivity Strength Heatmap
Implement in the HTML script:
```python
# Edge opacity ∝ learned KC→MBON weight (after training)
# This shows which KC→MBON connections potentiated (experiment 1–6)
```

### Overlay 2: Single-Unit Veto Highlighting (Exp 1)
```python
# Highlight the A-selective PN and its KC partners in red
# Show KC→MBON blocking synapses in bold
# Annotate: "Veto gate blocks A-reward learning"
```

### Overlay 3: Shaley Importance Ranking (Exp 6)
```python
# Color edges by Shapley value (red = blocks A, green = promotes A)
# Edge thickness ∝ |Shapley value|
# Identifies minimal intervention targets
```

---

## Exporting Multi-View Figures

### Figure 1: Circuit Overview (4-layer graph)
```bash
# 1. Generate HTML with --neuron-colors-by type
python scripts/visualize_pgcn_connectome.py --neuron-colors-by type --output-dir figures

# 2. Screenshot in browser or use Plotly export
# → figures/pgcn_circuit_interactive.html (open, export as PNG)
```

### Figure 2: Glomerular Organization
```bash
# 1. Generate HTML colored by glomerulus
python scripts/visualize_pgcn_connectome.py --neuron-colors-by glomerulus --output-dir figures

# 2. Export
# → figures/pgcn_glomerular_organization.png
```

### Figure 3: 3D FlyWire Rendering
```bash
# 1. Load manifest into FlyWire (see steps above)
# 2. Rotate to nice angle (isometric view recommended)
# 3. Export 3D render (Ctrl+P)
# → figures/pgcn_3d_flywire_render.png
```

### Figure 4: Experiment-Specific Overlays
For each experiment (1, 2, 3, 6):
```bash
# Run experiment output through visualization script
python scripts/visualize_pgcn_connectome.py \
  --experiment 1 \
  --highlight-veto-path \
  --output-dir figures

# → figures/pgcn_exp1_veto_blocking.png
```

---

## Quality Checklist

Before sharing your visualization:

- [ ] **Connectivity sums make sense:** PN→KC (~12k edges) >> KC→MBON (~10k edges) ✓
- [ ] **Sparsity is realistic:** PN→KC = 2%, KC→MBON = 11% (matches biology) ✓
- [ ] **Node counts match expected:** PNs ~353, KCs ~5k, MBONs ~96, DANs ~584 ✓
- [ ] **FlyWire manifest IDs are valid:** Paste into FlyWire and neurons appear ✓
- [ ] **Colors are distinguishable:** Blue (PN) ≠ green (KC) ≠ yellow (MBON) ≠ red (DAN) ✓
- [ ] **HTML graph is responsive:** Zoom/pan works smoothly ✓
- [ ] **3D FlyWire render shows anatomical neuropils:** Calyx, medial lobe, pedunculus distinct ✓

---

## Troubleshooting: Common Issues

### Issue: "Neurons don't appear in FlyWire"
**Cause:** Root IDs are from old materialization; FAFB v783 may have been updated.
**Solution:**
1. Check `meta.json` in your cache for materialization version
2. If outdated, re-extract: `python scripts/extract_alpn_projection_neurons.py --materialization latest`
3. Confirm root IDs by searching one in Codex: `https://codex.flywire.ai/`

### Issue: "Graph is unreadable mess of edges"
**Cause:** Too many edges displayed (try --top-edges 500, but showing 5000).
**Solution:**
```bash
# Reduce edge threshold
python scripts/visualize_pgcn_connectome.py --top-edges 100
```

### Issue: "HTML file is 500 MB, won't open in browser"
**Cause:** Plotly serializes full graph structure into standalone HTML.
**Solution:**
```bash
# Filter edges more aggressively
python scripts/visualize_pgcn_connectome.py --top-edges 50 --kc-subset 1000

# Or use Cytoscape.js instead of Plotly (lighter format)
python scripts/visualize_pgcn_connectome.py --plot-engine cytoscape
```

### Issue: "FlyWire UI slow when loading 6k neurons"
**Cause:** Browser/GPU limits on mesh rendering.
**Solution:**
1. Load neurons in 2–3 batches (e.g., PN batch, KC batch, MBON batch)
2. Use FlyWire "Batch Processor" addon for bulk coloring
3. Simplify mesh LOD (level of detail) in FlyWire settings

---

## Best Practices Summary

### Do ✅
- **Use FlyWire's 3D meshes** for publication figures (more credible than cartoons)
- **Color by functional logic** (glomerulus for PNs, valence for DANs)
- **Filter edges** to top-K to maintain readability
- **Include scale bar** and neuropil labels in exported figures
- **Link root IDs** in figure captions so readers can verify in FlyWire/Codex
- **Test manifest in FlyWire** before sharing; ensure neurons load

### Don't ❌
- **Don't show all ~50k edges** (hairball, uninterpretable)
- **Don't use arbitrary colors** (makes KC subtypes indistinguishable)
- **Don't mix materialization versions** (old PN IDs won't find KCs if KC extraction was newer)
- **Don't assume HTML is portable** (Plotly embeds full graph; huge file size)
- **Don't forget root IDs** in figure legends (enables reproducibility)

---

## Reference: FlyWire Dataset Specs

| Property | Value |
|----------|-------|
| **Dataset** | FAFB (Full Adult Female Brain) |
| **Materialization** | v783 (check your `meta.json`) |
| **Total neurons** | 139,255 |
| **Total synapses** | ~15.1 million |
| **Central brain subset** | ~130k neurons (your PGCN focus) |
| **Coordinates** | Nanometers (FAFB voxel space) |
| **PN subsets** | Olfactory (ALPNs), other sensory, etc. |
| **KC count** | ~5,177 (intrinsic to mushroom body) |
| **MBON count** | ~96 (calyx & medial lobe outputs) |
| **DAN count** | ~584 (PAM, PPL1, others) |

---

## Next Actions

1. **Immediate:** Use the ChatGPT prompts to generate your visualization script
2. **Short-term:** Load manifest into FlyWire and export 3D renders
3. **Integration:** Add visualizations to PGCN reports/ folder and version control
4. **Publication:** Use FlyWire renders in figures; cite FAFB v783 dataset + FlyWire consortium

---

**Happy visualizing! 🧠 → 🐍 → 🌐 (FlyWire)**
