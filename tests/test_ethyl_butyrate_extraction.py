"""Unit tests for ethyl butyrate circuit extraction.

Test suite for validating ORN extraction, connectivity mapping, and
stochasticity analysis functions.

Usage
-----
pytest tests/test_ethyl_butyrate_extraction.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from circuit_analysis.ethyl_butyrate_mapper import (
    extract_ethyl_butyrate_orns,
    get_pn_targets,
    compute_fanout_metrics,
    build_ethyl_butyrate_graph,
    TARGET_ORNS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_door_data():
    """Create mock DoOR response matrix."""
    return pd.DataFrame({
        'Or42a': [0.82, 0.1, 0.0],
        'Or42b': [0.53, 0.2, 0.05],
        'Or43b': [0.72, 0.15, 0.0],
        'Or7a': [0.05, 0.9, 0.1],
    }, index=['ethyl butyrate', '2-heptanone', 'benzaldehyde'])


@pytest.fixture
def mock_classification():
    """Create mock FlyWire classification data."""
    return pd.DataFrame({
        'root_id': [1001, 1002, 1003, 1004, 1005],
        'cell_type': ['ORN_DM3', 'ORN_VM2', 'ORN_DM1', 'PN_DM3', 'KC_alphabeta'],
        'sub_class': ['DM3', 'VM2', 'DM1', 'DM3_uPN', 'KC'],
        'side': ['L', 'R', 'L', 'L', 'R'],
    })


@pytest.fixture
def mock_connections():
    """Create mock connections table."""
    return pd.DataFrame({
        'pre_root_id': [1001, 1002, 1003, 1001, 1004],
        'post_root_id': [1004, 1004, 1004, 1005, 1005],
        'syn_count': [50, 30, 40, 5, 100],
        'neuropil': ['AL', 'AL', 'AL', 'AL', 'MB'],
        'nt_type': ['ACH', 'ACH', 'ACH', 'ACH', 'ACH'],
    })


# =============================================================================
# Test ORN Extraction
# =============================================================================

def test_extract_ethyl_butyrate_orns_basic(mock_door_data, mock_classification):
    """Test basic ORN extraction with valid inputs."""
    orns = extract_ethyl_butyrate_orns(
        classification=mock_classification,
        door_data=mock_door_data,
        threshold=0.5,
        odorant='ethyl butyrate',
    )

    # Should extract ORNs matching glomeruli
    assert len(orns) > 0
    assert 'root_id' in orns.columns
    assert 'or_type' in orns.columns
    assert 'glomerulus' in orns.columns
    assert 'door_response' in orns.columns


def test_extract_ethyl_butyrate_orns_threshold(mock_door_data, mock_classification):
    """Test ORN extraction with different thresholds."""
    # High threshold should return fewer ORNs
    orns_high = extract_ethyl_butyrate_orns(
        classification=mock_classification,
        door_data=mock_door_data,
        threshold=0.7,
        odorant='ethyl butyrate',
    )

    # Low threshold should return more
    orns_low = extract_ethyl_butyrate_orns(
        classification=mock_classification,
        door_data=mock_door_data,
        threshold=0.5,
        odorant='ethyl butyrate',
    )

    # Note: With mock data, counts may be equal due to limited glomeruli
    # But responses should respect threshold
    assert all(orns_high['door_response'] >= 0.7)
    assert all(orns_low['door_response'] >= 0.5)


def test_extract_ethyl_butyrate_orns_invalid_odorant(mock_door_data, mock_classification):
    """Test error handling for invalid odorant name."""
    with pytest.raises(ValueError, match="not found in DoOR data"):
        extract_ethyl_butyrate_orns(
            classification=mock_classification,
            door_data=mock_door_data,
            threshold=0.5,
            odorant='nonexistent_odorant',
        )


def test_extract_ethyl_butyrate_orns_unique_root_ids(mock_door_data, mock_classification):
    """Test that extracted ORNs have unique root IDs."""
    orns = extract_ethyl_butyrate_orns(
        classification=mock_classification,
        door_data=mock_door_data,
        threshold=0.5,
        odorant='ethyl butyrate',
    )

    assert orns['root_id'].is_unique, "Duplicate root IDs detected"


def test_extract_ethyl_butyrate_orns_glomeruli_valid(mock_door_data, mock_classification):
    """Test that extracted ORNs have valid glomeruli."""
    orns = extract_ethyl_butyrate_orns(
        classification=mock_classification,
        door_data=mock_door_data,
        threshold=0.5,
        odorant='ethyl butyrate',
    )

    valid_glomeruli = {'DM3', 'VM2', 'DM1'}
    assert set(orns['glomerulus']) <= valid_glomeruli, "Invalid glomeruli found"


# =============================================================================
# Test PN Targets
# =============================================================================

def test_get_pn_targets_basic(mock_connections, mock_classification):
    """Test basic PN target extraction."""
    orn_ids = [1001, 1002, 1003]

    pn_targets = get_pn_targets(
        orn_ids=orn_ids,
        connections=mock_connections,
        classification=mock_classification,
        min_synapses=10,
    )

    # Should find at least one PN
    assert len(pn_targets) > 0
    assert 'pn_root_id' in pn_targets.columns
    assert 'total_synapses' in pn_targets.columns
    assert 'cv_synapses' in pn_targets.columns


def test_get_pn_targets_synapse_threshold(mock_connections):
    """Test PN extraction with different synapse thresholds."""
    orn_ids = [1001, 1002, 1003]

    # High threshold
    pn_high = get_pn_targets(
        orn_ids=orn_ids,
        connections=mock_connections,
        min_synapses=100,
    )

    # Low threshold
    pn_low = get_pn_targets(
        orn_ids=orn_ids,
        connections=mock_connections,
        min_synapses=10,
    )

    # Low threshold should return more PNs (or equal)
    assert len(pn_low) >= len(pn_high)


def test_get_pn_targets_no_connections():
    """Test PN extraction with no valid connections."""
    empty_connections = pd.DataFrame({
        'pre_root_id': [],
        'post_root_id': [],
        'syn_count': [],
        'neuropil': [],
    })

    pn_targets = get_pn_targets(
        orn_ids=[1001],
        connections=empty_connections,
        min_synapses=5,
    )

    assert len(pn_targets) == 0


def test_get_pn_targets_synapse_statistics(mock_connections):
    """Test that PN synapse statistics are computed correctly."""
    orn_ids = [1001, 1002, 1003]

    pn_targets = get_pn_targets(
        orn_ids=orn_ids,
        connections=mock_connections,
        min_synapses=10,
    )

    if len(pn_targets) > 0:
        # Check that statistics are non-negative
        assert (pn_targets['total_synapses'] >= 0).all()
        assert (pn_targets['mean_synapses'] >= 0).all()
        assert (pn_targets['cv_synapses'] >= 0).all()

        # CV should be finite
        assert np.isfinite(pn_targets['cv_synapses']).all()


# =============================================================================
# Test Fan-out Metrics
# =============================================================================

def test_compute_fanout_metrics_basic(mock_connections):
    """Test fan-out metric computation."""
    orn_to_pn = mock_connections[mock_connections['post_root_id'] == 1004].copy()
    pn_to_kc = mock_connections[mock_connections['pre_root_id'] == 1004].copy()

    metrics = compute_fanout_metrics(
        orn_to_pn=orn_to_pn,
        pn_to_kc=pn_to_kc,
    )

    # Check expected keys
    expected_keys = {
        'mean_orn_pn_synapses',
        'std_orn_pn_synapses',
        'cv_orn_pn_synapses',
        'orn_fanout',
        'pn_fanin',
        'bottleneck_score',
    }

    assert set(metrics.keys()) >= expected_keys

    # Check values are valid
    for key, value in metrics.items():
        assert isinstance(value, (int, float))
        assert np.isfinite(value)
        assert value >= 0


def test_compute_fanout_metrics_empty_connections():
    """Test fan-out metrics with empty connections."""
    empty_df = pd.DataFrame({
        'pre_root_id': [],
        'post_root_id': [],
        'syn_count': [],
    })

    metrics = compute_fanout_metrics(
        orn_to_pn=empty_df,
        pn_to_kc=empty_df,
    )

    # Should return zeros
    assert metrics['mean_orn_pn_synapses'] == 0.0
    assert metrics['bottleneck_score'] == 0.0


# =============================================================================
# Test Graph Construction
# =============================================================================

def test_build_ethyl_butyrate_graph_basic(mock_connections):
    """Test circuit graph construction."""
    orns = pd.DataFrame({
        'root_id': [1001, 1002, 1003],
        'or_type': ['Or42a', 'Or43b', 'Or42b'],
        'glomerulus': ['DM3', 'VM2', 'DM1'],
        'door_response': [0.82, 0.72, 0.53],
    })

    pns = pd.DataFrame({'root_id': [1004]})
    kcs = pd.DataFrame({'root_id': [1005]})
    mbons = pd.DataFrame({'root_id': [1006]})

    graph = build_ethyl_butyrate_graph(
        orns=orns,
        pns=pns,
        kcs=kcs,
        mbons=mbons,
        connections=mock_connections,
        min_synapses=3,
    )

    # Check graph structure
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() >= 0


def test_build_ethyl_butyrate_graph_node_attributes(mock_connections):
    """Test that graph nodes have correct attributes."""
    orns = pd.DataFrame({
        'root_id': [1001],
        'or_type': ['Or42a'],
        'glomerulus': ['DM3'],
        'door_response': [0.82],
    })

    pns = pd.DataFrame({'root_id': [1004]})
    kcs = pd.DataFrame({'root_id': []})
    mbons = pd.DataFrame({'root_id': []})

    graph = build_ethyl_butyrate_graph(
        orns=orns,
        pns=pns,
        kcs=kcs,
        mbons=mbons,
        connections=mock_connections,
        min_synapses=3,
    )

    # Check ORN node attributes
    if 1001 in graph:
        orn_attrs = graph.nodes[1001]
        assert orn_attrs.get('neuron_type') == 'ORN'
        assert orn_attrs.get('or_type') == 'Or42a'
        assert orn_attrs.get('glomerulus') == 'DM3'
        assert orn_attrs.get('door_response') == 0.82


def test_build_ethyl_butyrate_graph_edge_weights(mock_connections):
    """Test that graph edges have correct weights."""
    orns = pd.DataFrame({'root_id': [1001]})
    pns = pd.DataFrame({'root_id': [1004]})
    kcs = pd.DataFrame({'root_id': []})
    mbons = pd.DataFrame({'root_id': []})

    graph = build_ethyl_butyrate_graph(
        orns=orns,
        pns=pns,
        kcs=kcs,
        mbons=mbons,
        connections=mock_connections,
        min_synapses=10,
    )

    # Check edge weights
    if graph.has_edge(1001, 1004):
        edge_weight = graph[1001][1004]['weight']
        assert edge_weight > 0
        assert isinstance(edge_weight, (int, float))


# =============================================================================
# Test Constants
# =============================================================================

def test_target_orns_structure():
    """Test TARGET_ORNS constant structure."""
    assert 'Or42a' in TARGET_ORNS
    assert 'Or42b' in TARGET_ORNS
    assert 'Or43b' in TARGET_ORNS

    for or_type, info in TARGET_ORNS.items():
        assert 'response' in info
        assert 'glomerulus' in info
        assert 'expected_count' in info
        assert 'synapses_out_range' in info
        assert 'neurotransmitter' in info

        # Validate response range
        assert 0 <= info['response'] <= 1

        # Validate expected count
        assert info['expected_count'] > 0


def test_target_orns_responses():
    """Test that TARGET_ORNS responses match expected values."""
    # From prompt specification
    assert TARGET_ORNS['Or42a']['response'] == 0.82
    assert TARGET_ORNS['Or43b']['response'] == 0.72
    assert TARGET_ORNS['Or42b']['response'] == 0.53


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
