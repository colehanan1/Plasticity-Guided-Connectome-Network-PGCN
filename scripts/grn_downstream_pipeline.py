#!/usr/bin/env python3
"""
PGCN GRN Connectivity & Visualization Pipeline
==============================================

Production-ready command line tool that loads gustatory receptor neuron (GRN)
annotations derived from FlyWire classification data, fetches downstream
connectivity, and produces publication-ready interactive visualizations.

Key features
------------
* Guaranteed use of structured `sub_class == 'sugar/water'` filtering.
* Supports sugar-specific or full gustatory cohorts with validation.
* Fallback to synthetic demo connectivity when the FlyWire API is unavailable
  or when explicitly requested (useful for offline development).
* Resilient FlyWire querying with exponential backoff and informative logging.
* Generates four interactive Plotly HTML reports alongside a cached JSON export
  of downstream connectivity metadata.
* Optional FAFB14 brain mesh overlay (requires `flybrains`) for anatomical
  context, scaled to match the network layout space.
* Comprehensive statistics and output validation suitable for a production
  workflow.

Example usage
-------------
    python scripts/grn_downstream_pipeline.py --grn-type sugar
    python scripts/grn_downstream_pipeline.py --grn-type all --include-demo true
    python scripts/grn_downstream_pipeline.py --grn-type sugar --output-dir reports/grn_network_html

The script completes in under ten minutes when using cached connectivity and
produces responsive HTML files that can be shared without additional
dependencies (Plotly served via CDN).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from tqdm import tqdm

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - plotly is required for runtime
    raise SystemExit("plotly is required to run this pipeline") from exc

try:
    from fafbseg import flywire  # type: ignore

    FAFBSEG_AVAILABLE = True
except ImportError:
    FAFBSEG_AVAILABLE = False

try:
    import navis  # noqa: F401  # pragma: no cover - used indirectly for brain mesh
    import flybrains  # type: ignore

    FLYBRAINS_AVAILABLE = True
except ImportError:
    FLYBRAINS_AVAILABLE = False

DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_CONNECTIVITY_PATH = Path("data/flywire/downstream_connectivity.json")

# Color palette aligned with navis/NavisMorphologyVisualizer conventions.
NODE_COLORS = {
    "GRN": "#FF0000",
    "KC": "#9370DB",
    "MBON": "#FFD700",
    "PN": "#1E90FF",
    "LN": "#00CED1",
    "other": "#808080",
}

PATHWAY_COLORS = {
    "sugar/water": "#0066CC",
    "bitter": "#FF9933",
    "other": "#00AA00",
}

MAX_DOWNSTREAM_PER_GRN = 75  # limits layout complexity for large partner sets


def configure_logging(verbosity: int) -> None:
    """Configure root logger with stream handler and leveled output."""

    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="End-to-end GRN connectivity extraction and visualization pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--grn-type",
        choices=("sugar", "all"),
        default="sugar",
        help="Subset of GRNs to analyze (sugar: 131 neurons, all: 343 neurons).",
    )
    parser.add_argument(
        "--include-demo",
        default="false",
        choices=("true", "false"),
        help="Use synthetic demo connectivity instead of FlyWire API.",
    )
    parser.add_argument(
        "--include-brain-mesh",
        default="false",
        choices=("true", "false"),
        help="Overlay scaled FAFB14 brain mesh on network visualization (requires flybrains).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/grn_network_html"),
        help="Directory for HTML reports and summary artifacts.",
    )
    parser.add_argument(
        "--connectivity-cache",
        type=Path,
        default=DEFAULT_CONNECTIVITY_PATH,
        help="Path to cache downstream connectivity JSON output.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum FlyWire API retry attempts per neuron (ignored for demo mode).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Base seconds for exponential backoff between FlyWire retries.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Logging verbosity: 0=warnings, 1=info, 2=debug.",
    )

    return parser.parse_args(argv)


def load_grn_tables(report_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load sugar/water and all GRN CSV tables.

    Args:
        report_dir: Directory containing precomputed CSV exports.

    Returns:
        Tuple of (sugar_df, all_df) DataFrames.
    """

    sugar_path = report_dir / "sugar_water_grns_131.csv"
    all_path = report_dir / "all_grns_343.csv"

    logging.info("Loading GRN CSVs from %s", report_dir.resolve())
    sugar_df = pd.read_csv(sugar_path)
    all_df = pd.read_csv(all_path)

    logging.debug("Sugar/water columns: %s", list(sugar_df.columns))
    logging.debug("All GRNs columns: %s", list(all_df.columns))

    return sugar_df, all_df


def validate_grn_tables(sugar_df: pd.DataFrame, all_df: pd.DataFrame) -> None:
    """
    Ensure GRN tables satisfy expected counts and classifications.

    Raises:
        AssertionError if validation fails.
    """

    logging.info("Validating GRN extraction consistency...")

    assert len(sugar_df) == 131, f"Expected 131 sugar GRNs, found {len(sugar_df)}"
    assert (sugar_df["sub_class"] == "sugar/water").all(), "Sugar GRNs must have sub_class == 'sugar/water'"

    assert len(all_df) == 343, f"Expected 343 total GRNs, found {len(all_df)}"
    assert set(all_df["class"].unique()) == {"gustatory"}, "All GRNs must have class == 'gustatory'"

    sugar_breakdown = sugar_df["sub_class"].value_counts().to_dict()
    total_breakdown = all_df["sub_class"].value_counts().to_dict()

    logging.info("  ✓ Sugar GRNs: 131 (sub_class breakdown: %s)", sugar_breakdown)
    logging.info("  ✓ All GRNs: 343 (sub_class breakdown: %s)", total_breakdown)


def select_grn_population(grn_type: str, sugar_df: pd.DataFrame, all_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """
    Select the GRN population to analyze.

    Args:
        grn_type: 'sugar' or 'all'.
        sugar_df: Sugar/water GRN DataFrame.
        all_df: All GRN DataFrame.

    Returns:
        Tuple of (selected_df, list_of_root_ids).
    """

    if grn_type == "sugar":
        logging.info("Using sugar/water GRNs (131 neurons).")
        selected_df = sugar_df.copy()
    else:
        logging.info("Using all gustatory GRNs (343 neurons).")
        selected_df = all_df.copy()

    root_ids = selected_df["root_id"].astype(int).tolist()
    logging.info("  ✓ Selected %d GRNs.", len(root_ids))
    return selected_df, root_ids


def load_cached_connectivity(cache_path: Path) -> Optional[Dict[str, List[Dict[str, object]]]]:
    """
    Load previously cached connectivity JSON if available.

    Returns:
        Parsed JSON dictionary or None if file missing.
    """

    if cache_path.exists():
        try:
            with cache_path.open("r") as fh:
                data = json.load(fh)
            logging.info("Loaded cached connectivity from %s", cache_path)
            return data
        except json.JSONDecodeError as exc:
            logging.warning("Failed to parse cached connectivity (%s): %s", cache_path, exc)
    return None


def coerce_downstream_records(raw: object) -> List[Dict[str, object]]:
    """
    Convert FlyWire API responses into a normalized record list.

    Handles pandas DataFrames, lists of tuples, or lists of dicts.
    """

    records: List[Dict[str, object]] = []

    if raw is None:
        return records

    if isinstance(raw, pd.DataFrame):
        for row in raw.itertuples(index=False):
            downstream_id = int(getattr(row, "downstream_id", getattr(row, "root_id", getattr(row, "id", 0))))
            synapses = int(getattr(row, "synapses", getattr(row, "weight", 1)) or 1)
            neuron_class = getattr(row, "class", getattr(row, "type", "unknown"))
            neuropil = getattr(row, "neuropil", getattr(row, "region", "unknown"))
            records.append(
                {
                    "downstream_id": downstream_id,
                    "synapses": synapses,
                    "class": str(neuron_class),
                    "neuropil": str(neuropil),
                }
            )
        return records

    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                downstream_id = int(entry.get("downstream_id", entry.get("root_id", entry.get("id", 0))))
                synapses = int(entry.get("synapses", entry.get("weight", 1)) or 1)
                neuron_class = entry.get("class", entry.get("type", "unknown"))
                neuropil = entry.get("neuropil", entry.get("region", "unknown"))
            else:
                # Allow tuple/list responses following (id, synapses, class, neuropil) pattern.
                parts = list(entry)
                downstream_id = int(parts[0])
                synapses = int(parts[1] if len(parts) > 1 and parts[1] is not None else 1)
                neuron_class = parts[2] if len(parts) > 2 else "unknown"
                neuropil = parts[3] if len(parts) > 3 else "unknown"

            records.append(
                {
                    "downstream_id": downstream_id,
                    "synapses": synapses,
                    "class": str(neuron_class),
                    "neuropil": str(neuropil),
                }
            )
        return records

    logging.debug("Received unsupported downstream record type: %s", type(raw))
    return records


def generate_synthetic_connectivity(grn_ids: List[int]) -> Dict[str, List[Dict[str, object]]]:
    """
    Generate reproducible synthetic connectivity for offline demos.

    The distributions mimic expected downstream classes and synapse counts.
    """

    rng = np.random.default_rng(seed=42)
    logging.warning("Using synthetic demo connectivity (fafbseg not available or demo requested).")

    downstream_classes = ["KC", "MBON", "PN", "LN", "other"]
    class_probabilities = np.array([0.45, 0.2, 0.15, 0.1, 0.1])

    connectivity: Dict[str, List[Dict[str, object]]] = {}
    base_id = 720575940700000000

    for idx, grn_id in enumerate(tqdm(grn_ids, desc="Generating synthetic data")):
        partner_count = int(rng.integers(30, 100))
        classes = rng.choice(downstream_classes, size=partner_count, p=class_probabilities)
        neuropils = rng.choice(["GNG", "PRW", "SAD", "MB", "LH"], size=partner_count)
        synapses = rng.poisson(lam=12, size=partner_count) + 1

        records = []
        for i in range(partner_count):
            records.append(
                {
                    "downstream_id": int(base_id + idx * 1000 + i),
                    "synapses": int(synapses[i]),
                    "class": str(classes[i]),
                    "neuropil": str(neuropils[i]),
                }
            )

        connectivity[str(int(grn_id))] = records

    return connectivity


def fetch_downstream_connectivity(
    grn_ids: List[int],
    use_demo: bool,
    cache_path: Path,
    max_retries: int,
    backoff: float,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Fetch downstream connectivity for each GRN with retry logic and caching.
    """

    cached = load_cached_connectivity(cache_path)
    if cached:
        missing = [gid for gid in grn_ids if str(gid) not in cached]
        if not missing:
            logging.info("Using cached connectivity for all GRNs.")
            return cached
        logging.info("Cache incomplete (%d missing). Continuing with fetch.", len(missing))
    else:
        cached = {}

    if use_demo or not FAFBSEG_AVAILABLE:
        connectivity = generate_synthetic_connectivity(grn_ids)
    else:
        logging.info("Fetching downstream connectivity from FlyWire (fafbseg).")
        connectivity = {}
        for grn_id in tqdm(grn_ids, desc="Querying FlyWire"):
            attempt = 0
            success = False
            while attempt <= max_retries and not success:
                try:
                    raw = flywire.get_downstream_neurons(int(grn_id), dataset="783")  # type: ignore[arg-type]
                    connectivity[str(int(grn_id))] = coerce_downstream_records(raw)
                    success = True
                except Exception as exc:  # noqa: BLE001 - broad catch for resiliency
                    attempt += 1
                    wait = backoff * (2 ** (attempt - 1))
                    if attempt > max_retries:
                        logging.error("  ✗ Failed to fetch %s after %d attempts (%s).", grn_id, attempt - 1, exc)
                        connectivity[str(int(grn_id))] = []
                    else:
                        logging.warning(
                            "  ⚠ FlyWire error for %s (attempt %d/%d): %s. Retrying in %.1fs...",
                            grn_id,
                            attempt,
                            max_retries,
                            exc,
                            wait,
                        )
                        time.sleep(wait)

    # Merge with cached data (prefer freshly fetched entries to ensure updates).
    cached.update(connectivity)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as fh:
        json.dump(cached, fh, indent=2)
    logging.info("Saved connectivity cache to %s", cache_path)

    return cached


def build_network_graph(
    grn_ids: List[int],
    connectivity: Dict[str, List[Dict[str, object]]],
    grn_df: pd.DataFrame,
) -> nx.DiGraph:
    """
    Construct a directed networkx graph representing GRNs and their downstream partners.
    """

    logging.info("Building network graph...")
    graph = nx.DiGraph()

    subclass_lookup = grn_df.set_index("root_id")["sub_class"].to_dict()

    for grn_id in grn_ids:
        sub_class = subclass_lookup.get(grn_id, "other")
        graph.add_node(
            int(grn_id),
            node_type="GRN",
            sub_class=sub_class,
            color=NODE_COLORS["GRN"],
            size=22,
        )

    downstream_counts: Dict[str, int] = {}

    for grn_id in grn_ids:
        records = connectivity.get(str(int(grn_id)), [])
        if MAX_DOWNSTREAM_PER_GRN and len(records) > MAX_DOWNSTREAM_PER_GRN:
            logging.debug(
                "  • Trimming downstream partners for %s from %d to %d (by synapses).",
                grn_id,
                len(records),
                MAX_DOWNSTREAM_PER_GRN,
            )
            records = sorted(records, key=lambda rec: int(rec.get("synapses", 1)), reverse=True)[
                :MAX_DOWNSTREAM_PER_GRN
            ]
        sub_class = graph.nodes[int(grn_id)]["sub_class"]
        pathway_color = PATHWAY_COLORS.get(sub_class, PATHWAY_COLORS["other"])

        for record in records:
            downstream_id = int(record.get("downstream_id", 0))
            downstream_class = str(record.get("class", "other")) or "other"
            synapses = int(record.get("synapses", 1) or 1)

            if downstream_id not in graph:
                graph.add_node(
                    downstream_id,
                    node_type="downstream",
                    class_label=downstream_class,
                    color=NODE_COLORS.get(downstream_class, NODE_COLORS["other"]),
                    size=16,
                )

            graph.add_edge(
                int(grn_id),
                downstream_id,
                weight=synapses,
                color=pathway_color,
                width=float(max(1.0, min(10.0, np.log1p(synapses) * 2))),
            )

            downstream_counts[downstream_class] = downstream_counts.get(downstream_class, 0) + 1

    logging.info("  ✓ GRN nodes: %d", sum(1 for _, data in graph.nodes(data=True) if data["node_type"] == "GRN"))
    logging.info(
        "  ✓ Downstream nodes: %d",
        sum(1 for _, data in graph.nodes(data=True) if data["node_type"] == "downstream"),
    )
    logging.info("  ✓ Edges: %d", graph.number_of_edges())
    logging.info("  ✓ Downstream class counts: %s", downstream_counts)

    return graph


def compute_force_layout(graph: nx.DiGraph, seed: int = 42) -> Dict[int, np.ndarray]:
    """
    Compute a deterministic 3D force-directed layout for the network graph.
    """

    logging.info("Computing spring layout (dim=3, seed=%d)...", seed)
    positions = nx.spring_layout(graph, dim=3, iterations=60, seed=seed, weight="weight")
    return {int(node): np.asarray(coord, dtype=float) for node, coord in positions.items()}


def add_brain_mesh_trace(fig: go.Figure, layout_positions: Dict[int, np.ndarray], include_brain_mesh: bool) -> None:
    """
    Add a scaled FAFB14 brain mesh for anatomical context when requested.

    The mesh vertices are min-max normalized and scaled to match the bounding
    box of the network layout. This preserves relative positioning while keeping
    the brain context lightweight for HTML export.
    """

    if not include_brain_mesh:
        return

    if not FLYBRAINS_AVAILABLE:
        logging.warning("Brain mesh requested but flybrains is unavailable. Skipping overlay.")
        return

    try:
        logging.info("Adding FAFB14 brain mesh overlay (scaled to layout bounds).")
        brain_mesh = flybrains.FAFB14.mesh  # type: ignore[attr-defined]

        layout_coords = np.vstack(list(layout_positions.values()))
        layout_min = layout_coords.min(axis=0)
        layout_max = layout_coords.max(axis=0)

        vertices = brain_mesh.vertices.astype(float)
        mesh_min = vertices.min(axis=0)
        mesh_ptp = np.clip(vertices.max(axis=0) - mesh_min, a_min=1e-6, a_max=None)
        vertices_norm = (vertices - mesh_min) / mesh_ptp
        scaled_vertices = vertices_norm * (layout_max - layout_min) + layout_min

        fig.add_trace(
            go.Mesh3d(
                x=scaled_vertices[:, 0],
                y=scaled_vertices[:, 1],
                z=scaled_vertices[:, 2],
                i=brain_mesh.faces[:, 0],
                j=brain_mesh.faces[:, 1],
                k=brain_mesh.faces[:, 2],
                color="lightgray",
                opacity=0.1,
                name="FAFB14 mesh (scaled)",
                hoverinfo="skip",
            )
        )
    except Exception as exc:  # noqa: BLE001 - ensure pipeline continues
        logging.warning("Failed to add brain mesh overlay: %s", exc)


def create_interactive_network_html(
    graph: nx.DiGraph,
    layout_positions: Dict[int, np.ndarray],
    title: str,
    output_file: Path,
    include_brain_mesh: bool,
) -> None:
    """
    Create a Plotly 3D network visualization and export to HTML.
    """

    logging.info("Rendering 3D network visualization → %s", output_file)

    edge_traces: List[go.Scatter3d] = []
    for source, target, data in graph.edges(data=True):
        start = layout_positions[int(source)]
        end = layout_positions[int(target)]
        edge_traces.append(
            go.Scatter3d(
                x=[start[0], end[0], None],
                y=[start[1], end[1], None],
                z=[start[2], end[2], None],
                mode="lines",
                line=dict(width=data.get("width", 2.0), color=data.get("color", "#808080")),
                hoverinfo="text",
                text=f"Synapses: {data.get('weight', 1)}",
                showlegend=False,
            )
        )

    node_x, node_y, node_z, node_color, node_size, node_text = [], [], [], [], [], []
    for node, attrs in graph.nodes(data=True):
        xyz = layout_positions[int(node)]
        node_x.append(xyz[0])
        node_y.append(xyz[1])
        node_z.append(xyz[2])
        node_color.append(attrs.get("color", NODE_COLORS["other"]))
        node_size.append(attrs.get("size", 15))
        label_parts = [attrs.get("node_type", "node")]
        if attrs.get("node_type") == "GRN":
            label_parts.append(str(attrs.get("sub_class", "unknown")))
        else:
            label_parts.append(str(attrs.get("class_label", "unknown")))
        node_text.append(f"{'_'.join(label_parts)} | id={node}")

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        marker=dict(size=node_size, color=node_color, line=dict(width=1.5, color="white"), opacity=0.92),
        hoverinfo="text",
        text=node_text,
        name="Neurons",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    add_brain_mesh_trace(fig, layout_positions, include_brain_mesh=include_brain_mesh)

    fig.update_layout(
        title=title,
        width=1400,
        height=1000,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.6)),
        ),
        hovermode="closest",
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_file),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )


def create_connectivity_heatmap(
    connectivity: Dict[str, List[Dict[str, object]]],
    grn_df: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Create combined heatmap + scatter dashboard illustrating connectivity composition.
    """

    logging.info("Rendering connectivity heatmap → %s", output_file)

    agg: Dict[Tuple[str, str], int] = {}
    scatter_classes: List[str] = []
    scatter_synapses: List[int] = []
    scatter_text: List[str] = []

    for grn_id, records in connectivity.items():
        root_id = int(grn_id)
        df_row = grn_df[grn_df["root_id"] == root_id]
        if df_row.empty:
            continue
        sub_class = df_row["sub_class"].iloc[0]
        for record in records:
            downstream_class = str(record.get("class", "other")) or "other"
            synapses = int(record.get("synapses", 1) or 1)
            key = (sub_class, downstream_class)
            agg[key] = agg.get(key, 0) + 1
            scatter_classes.append(downstream_class)
            scatter_synapses.append(synapses)
            scatter_text.append(
                f"GRN {grn_id} ({sub_class}) → {downstream_class}<br>Synapses: {synapses}"
            )

    sub_classes = sorted({key[0] for key in agg.keys()})
    downstream_classes = sorted({key[1] for key in agg.keys()})

    z_matrix = []
    for sub_class in sub_classes:
        row = [agg.get((sub_class, downstream), 0) for downstream in downstream_classes]
        z_matrix.append(row)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=(
            "GRN Sub-class → Downstream Class",
            "Synapse Counts per Connection",
        ),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}]],
    )

    fig.add_trace(
        go.Heatmap(
            z=z_matrix,
            x=downstream_classes,
            y=sub_classes,
            colorscale="YlOrRd",
            text=z_matrix,
            texttemplate="%{text}",
            colorbar=dict(title="# Connections"),
        ),
        row=1,
        col=1,
    )

    if scatter_classes:
        fig.add_trace(
            go.Scattergl(
                x=scatter_classes,
                y=scatter_synapses,
                mode="markers",
                marker=dict(size=5, color="rgba(0, 102, 204, 0.35)"),
                text=scatter_text,
                hovertemplate="%{text}<extra></extra>",
                name="Synapse distribution",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="GRN Connectivity Heatmap & Synapse Distribution",
        width=1300,
        height=650,
        margin=dict(l=60, r=40, t=70, b=60),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Downstream Class", row=1, col=1)
    fig.update_yaxes(title_text="GRN Sub-class", row=1, col=1)
    fig.update_xaxes(title_text="Downstream Class", row=1, col=2)
    fig.update_yaxes(title_text="Synapses per Connection", row=1, col=2)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_file),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )


def create_degree_dashboard(graph: nx.DiGraph, output_file: Path) -> None:
    """
    Create four-panel dashboard showing degree and synapse distributions.
    """

    logging.info("Rendering degree distribution dashboard → %s", output_file)

    node_ids = list(graph.nodes())
    in_degrees = [graph.in_degree(node) for node in node_ids]
    out_degrees = [graph.out_degree(node) for node in node_ids]
    weights = [data.get("weight", 1) for _, _, data in graph.edges(data=True)]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "In-degree Distribution",
            "Out-degree Distribution",
            "Synapse Count Distribution",
            "Network Summary",
        ),
    )

    fig.add_trace(
        go.Histogram(x=in_degrees, nbinsx=30, marker_color="#1f77b4", name="In-degree"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=out_degrees, nbinsx=30, marker_color="#ff7f0e", name="Out-degree"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=weights, nbinsx=30, marker_color="#2ca02c", name="Synapses"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=out_degrees,
            y=in_degrees,
            mode="markers",
            marker=dict(size=5, color="rgba(148, 103, 189, 0.55)"),
            text=[f"Node {node}" for node in node_ids],
            hovertemplate="%{text}<br>Out-degree=%{x}<br>In-degree=%{y}<extra></extra>",
            name="Degree scatter",
        ),
        row=2,
        col=2,
    )

    stats_text = (
        "<b>Network Statistics</b><br>"
        f"Nodes: {graph.number_of_nodes()}<br>"
        f"Edges: {graph.number_of_edges()}<br>"
        f"Mean in-degree: {np.mean(in_degrees):.2f}<br>"
        f"Mean out-degree: {np.mean(out_degrees):.2f}<br>"
        f"Mean synapses/edge: {np.mean(weights):.2f}"
    )

    fig.add_annotation(
        text=stats_text,
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#444",
        borderwidth=1,
        showarrow=False,
        row=2,
        col=2,
        font=dict(size=12),
    )

    fig.update_xaxes(title_text="Out-degree", row=2, col=2)
    fig.update_yaxes(title_text="In-degree", row=2, col=2)

    fig.update_layout(height=900, width=1200, showlegend=False, title="GRN Network Degree Distribution Dashboard")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_file),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )


def create_connectivity_summary(
    connectivity: Dict[str, List[Dict[str, object]]],
    output_file: Path,
) -> None:
    """
    Create a dashboard summarising downstream class/neuropil composition plus top connections table.
    """

    logging.info("Rendering connectivity summary dashboard → %s", output_file)

    class_counts: Dict[str, int] = {}
    neuropil_counts: Dict[str, int] = {}
    connection_records: List[Tuple[int, int, str, str, int]] = []

    for grn_id_str, records in connectivity.items():
        grn_id_int = int(grn_id_str)
        for record in records:
            neuron_class = str(record.get("class", "other")) or "other"
            neuropil = str(record.get("neuropil", "unknown")) or "unknown"
            downstream_id = int(record.get("downstream_id", 0))
            synapses = int(record.get("synapses", 1) or 1)
            class_counts[neuron_class] = class_counts.get(neuron_class, 0) + 1
            neuropil_counts[neuropil] = neuropil_counts.get(neuropil, 0) + 1
            connection_records.append((grn_id_int, downstream_id, neuron_class, neuropil, synapses))

    class_items = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
    neuropil_items = sorted(neuropil_counts.items(), key=lambda item: item[1], reverse=True)
    sorted_connections = sorted(connection_records, key=lambda item: item[4], reverse=True)
    top_n = min(2000, len(sorted_connections))
    top_connections = sorted_connections[:top_n]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Downstream Class Distribution",
            "Downstream Neuropil Distribution",
            f"Top {top_n} Connections by Synapse Count",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "table", "colspan": 2}, None]],
        vertical_spacing=0.12,
    )

    if class_items:
        fig.add_trace(
            go.Bar(
                x=[item[0] for item in class_items],
                y=[item[1] for item in class_items],
                marker_color=[NODE_COLORS.get(item[0], NODE_COLORS["other"]) for item in class_items],
                hovertemplate="%{x}: %{y} connections<extra></extra>",
                name="Class distribution",
            ),
            row=1,
            col=1,
        )

    if neuropil_items:
        fig.add_trace(
            go.Bar(
                x=[item[0] for item in neuropil_items],
                y=[item[1] for item in neuropil_items],
                marker_color="#9467bd",
                hovertemplate="%{x}: %{y} connections<extra></extra>",
                name="Neuropil distribution",
            ),
            row=1,
            col=2,
        )

    if top_connections:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Rank", "GRN ID", "Downstream ID", "Class", "Neuropil", "Synapses"],
                    fill_color="#2ca02c",
                    font=dict(color="white", size=12),
                    align="center",
                ),
                cells=dict(
                    values=[
                        list(range(1, top_n + 1)),
                        [str(item[0]) for item in top_connections],
                        [str(item[1]) for item in top_connections],
                        [item[2] for item in top_connections],
                        [item[3] for item in top_connections],
                        [item[4] for item in top_connections],
                    ],
                    fill_color="white",
                    align="center",
                    font=dict(size=11),
                ),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        width=1300,
        height=900,
        showlegend=False,
        title="Connectivity Composition Overview",
        margin=dict(l=60, r=30, t=70, b=60),
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_file),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )


def summarise_connectivity(connectivity: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    """
    Compute summary statistics for downstream connectivity.
    """

    total_edges = sum(len(records) for records in connectivity.values())
    unique_targets = {int(record["downstream_id"]) for records in connectivity.values() for record in records}
    synapse_counts = [int(record.get("synapses", 1)) for records in connectivity.values() for record in records]
    class_counts: Dict[str, int] = {}
    neuropil_counts: Dict[str, int] = {}

    for records in connectivity.values():
        for record in records:
            neuron_class = str(record.get("class", "other")) or "other"
            neuropil = str(record.get("neuropil", "unknown")) or "unknown"
            class_counts[neuron_class] = class_counts.get(neuron_class, 0) + 1
            neuropil_counts[neuropil] = neuropil_counts.get(neuropil, 0) + 1

    return {
        "grn_count": len(connectivity),
        "total_connections": total_edges,
        "unique_targets": len(unique_targets),
        "mean_synapses_per_edge": float(np.mean(synapse_counts)) if synapse_counts else 0.0,
        "median_synapses_per_edge": float(np.median(synapse_counts)) if synapse_counts else 0.0,
        "class_counts": class_counts,
        "neuropil_counts": neuropil_counts,
    }


def validate_output_files(html_files: List[Path]) -> None:
    """
    Ensure generated HTML files exist and meet minimum size requirements.
    """

    for html_file in html_files:
        if not html_file.exists():
            raise FileNotFoundError(f"Expected output HTML not found: {html_file}")
        size_kb = html_file.stat().st_size / 1024
        if size_kb < 100:
            raise ValueError(f"{html_file} appears too small ({size_kb:.1f} KB) – did Plotly export fail?")
        logging.info("  ✓ %s (%.1f KB)", html_file.name, size_kb)


def save_summary(summary: Dict[str, object], output_dir: Path) -> Path:
    """
    Persist summary statistics to JSON for downstream analyses.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "grn_connectivity_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logging.info("Saved summary statistics → %s", summary_path)
    return summary_path


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    """
    Execute the end-to-end GRN pipeline and return summary statistics.
    """

    configure_logging(args.verbosity)

    logging.info("=== PGCN GRN Connectivity Pipeline ===")
    logging.info("Options: %s", vars(args))

    sugar_df, all_df = load_grn_tables(DEFAULT_REPORT_DIR)
    validate_grn_tables(sugar_df, all_df)

    selected_df, grn_ids = select_grn_population(args.grn_type, sugar_df, all_df)

    use_demo = args.include_demo.lower() == "true"
    include_brain_mesh = args.include_brain_mesh.lower() == "true"

    connectivity = fetch_downstream_connectivity(
        grn_ids=grn_ids,
        use_demo=use_demo,
        cache_path=args.connectivity_cache,
        max_retries=args.max_retries,
        backoff=args.retry_backoff,
    )

    # Subset connectivity to selected GRNs to avoid stale cache entries from other runs.
    connectivity_subset = {str(int(grn_id)): connectivity.get(str(int(grn_id)), []) for grn_id in grn_ids}

    graph = build_network_graph(grn_ids, connectivity_subset, selected_df)
    layout_positions = compute_force_layout(graph)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    network_html = args.output_dir / "grn_downstream_network.html"
    heatmap_html = args.output_dir / "grn_connectivity_heatmap.html"
    degree_html = args.output_dir / "grn_degree_distribution_dashboard.html"
    summary_html = args.output_dir / "grn_connectivity_summary_dashboard.html"

    create_interactive_network_html(
        graph=graph,
        layout_positions=layout_positions,
        title=(
            "Sugar/Water GRN Connectivity Network"
            if args.grn_type == "sugar"
            else "All Gustatory GRN Connectivity Network"
        ),
        output_file=network_html,
        include_brain_mesh=include_brain_mesh,
    )
    create_connectivity_heatmap(connectivity_subset, selected_df, heatmap_html)
    create_degree_dashboard(graph, degree_html)
    create_connectivity_summary(connectivity_subset, summary_html)

    summary = summarise_connectivity(connectivity_subset)
    summary["grn_type"] = args.grn_type
    summary["use_demo_data"] = use_demo
    summary["include_brain_mesh"] = include_brain_mesh
    summary_path = save_summary(summary, args.output_dir)

    logging.info("\nOutput validation:")
    validate_output_files([network_html, heatmap_html, degree_html, summary_html])

    logging.info("\nPipeline complete.")
    logging.info("  ✓ Total GRNs analysed: %d", summary["grn_count"])
    logging.info("  ✓ Downstream connections: %d", summary["total_connections"])
    logging.info("  ✓ Unique targets: %d", summary["unique_targets"])
    logging.info("  ✓ Mean synapses per connection: %.2f", summary["mean_synapses_per_edge"])

    return {
        "summary": summary,
        "summary_path": summary_path,
        "html_outputs": [network_html, heatmap_html, degree_html, summary_html],
    }


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    results = run_pipeline(args)

    logging.info("\nArtifacts:")
    for html_file in results["html_outputs"]:
        size_kb = html_file.stat().st_size / 1024
        logging.info("  • %s (%.1f KB)", html_file, size_kb)
    logging.info("  • %s", results["summary_path"])
    logging.info("\nNext steps:")
    logging.info("  1. Review interactive HTML files for interpretation.")
    logging.info("  2. Feed connectivity JSON into downstream PGCN modules.")
    logging.info("  3. Commit artifacts for reproducibility once verified.")


if __name__ == "__main__":
    main()
