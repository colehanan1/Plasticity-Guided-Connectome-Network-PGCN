import logging

import pandas as pd

from pgcn.metrics import jaccard_kc_overlap


def test_jaccard_kc_overlap_handles_missing_glomerulus(caplog) -> None:
    edges = pd.DataFrame({"source_id": [1], "target_id": [10]})
    pn_nodes = pd.DataFrame({"node_id": [1]})

    with caplog.at_level(logging.WARNING):
        result = jaccard_kc_overlap(edges, pn_nodes)

    assert "missing 'glomerulus'" in " ".join(caplog.messages)
    assert list(result.columns) == [
        "glomerulus_a",
        "glomerulus_b",
        "kc_count_a",
        "kc_count_b",
        "intersection",
        "union",
        "jaccard",
    ]
    assert result.empty


def test_jaccard_kc_overlap_computes_pairwise_statistics() -> None:
    edges = pd.DataFrame(
        {
            "source_id": [1, 1, 2, 2],
            "target_id": [10, 11, 10, 12],
        }
    )
    pn_nodes = pd.DataFrame(
        {
            "node_id": [1, 2],
            "glomerulus": ["A", "B"],
        }
    )

    result = jaccard_kc_overlap(edges, pn_nodes)

    assert len(result) == 3
    lookup = {
        (row.glomerulus_a, row.glomerulus_b): row
        for row in result.itertuples(index=False)
    }
    assert lookup[("A", "A")].jaccard == 1.0
    assert lookup[("B", "B")].jaccard == 1.0
    assert lookup[("A", "B")].intersection == 1
    assert lookup[("A", "B")].union == 3
