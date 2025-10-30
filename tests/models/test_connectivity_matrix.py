"""Unit tests for ConnectivityMatrix."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pgcn.models.connectivity_matrix import ConnectivityMatrix


@pytest.fixture
def sample_connectivity_matrix():
    """Create small synthetic ConnectivityMatrix for testing."""
    n_pn, n_kc, n_mbon, n_dan = 10, 20, 5, 8

    # Create neuron ID arrays
    pn_ids = np.arange(1000, 1000 + n_pn, dtype=np.int64)
    kc_ids = np.arange(2000, 2000 + n_kc, dtype=np.int64)
    mbon_ids = np.arange(3000, 3000 + n_mbon, dtype=np.int64)
    dan_ids = np.arange(4000, 4000 + n_dan, dtype=np.int64)

    # Create sparse connectivity matrices
    # PN→KC: ~30% density (each KC receives from ~3 PNs)
    pn_to_kc_data = np.random.rand(60)
    pn_to_kc_row = np.random.randint(0, n_kc, size=60)
    pn_to_kc_col = np.random.randint(0, n_pn, size=60)
    pn_to_kc = sp.coo_matrix(
        (pn_to_kc_data, (pn_to_kc_row, pn_to_kc_col)),
        shape=(n_kc, n_pn),
    ).tocsr()

    # KC→MBON: ~40% density
    kc_to_mbon_data = np.random.rand(40)
    kc_to_mbon_row = np.random.randint(0, n_mbon, size=40)
    kc_to_mbon_col = np.random.randint(0, n_kc, size=40)
    kc_to_mbon = sp.coo_matrix(
        (kc_to_mbon_data, (kc_to_mbon_row, kc_to_mbon_col)),
        shape=(n_mbon, n_kc),
    ).tocsr()

    # DAN→KC: ~20% density
    dan_to_kc_data = np.random.rand(32)
    dan_to_kc_row = np.random.randint(0, n_kc, size=32)
    dan_to_kc_col = np.random.randint(0, n_dan, size=32)
    dan_to_kc = sp.coo_matrix(
        (dan_to_kc_data, (dan_to_kc_row, dan_to_kc_col)),
        shape=(n_kc, n_dan),
    ).tocsr()

    # DAN→MBON: ~25% density
    dan_to_mbon_data = np.random.rand(10)
    dan_to_mbon_row = np.random.randint(0, n_mbon, size=10)
    dan_to_mbon_col = np.random.randint(0, n_dan, size=10)
    dan_to_mbon = sp.coo_matrix(
        (dan_to_mbon_data, (dan_to_mbon_row, dan_to_mbon_col)),
        shape=(n_mbon, n_dan),
    ).tocsr()

    # Create metadata
    pn_glomeruli = {pn_id: f"glom_{i % 5}" for i, pn_id in enumerate(pn_ids)}
    kc_subtypes = {kc_id: ["ab", "g_main", "apbp_main"][i % 3] for i, kc_id in enumerate(kc_ids)}
    mbon_neuropils = {mbon_id: [f"CA_L", f"lobe_{i}"] for i, mbon_id in enumerate(mbon_ids)}
    dan_neuropils = {dan_id: [f"MB_CA", f"region_{i}"] for i, dan_id in enumerate(dan_ids)}

    return ConnectivityMatrix(
        pn_ids=pn_ids,
        kc_ids=kc_ids,
        mbon_ids=mbon_ids,
        dan_ids=dan_ids,
        pn_to_kc=pn_to_kc,
        kc_to_mbon=kc_to_mbon,
        dan_to_kc=dan_to_kc,
        dan_to_mbon=dan_to_mbon,
        pn_glomeruli=pn_glomeruli,
        kc_subtypes=kc_subtypes,
        mbon_neuropils=mbon_neuropils,
        dan_neuropils=dan_neuropils,
    )


def test_connectivity_matrix_immutability(sample_connectivity_matrix):
    """Test that ConnectivityMatrix is frozen (immutable)."""
    conn = sample_connectivity_matrix

    # Biological rationale: Connectivity should be frozen after development
    # Adult fly MB connectivity is stable; plasticity occurs at weights, not topology
    with pytest.raises(Exception):  # FrozenInstanceError from dataclass
        conn.pn_ids = np.array([999])


def test_connectivity_matrix_shapes(sample_connectivity_matrix):
    """Test that matrix shapes match neuron counts."""
    conn = sample_connectivity_matrix

    # Biological validation: Dimensions must be consistent
    assert conn.pn_to_kc.shape == (conn.n_kc, conn.n_pn)
    assert conn.kc_to_mbon.shape == (conn.n_mbon, conn.n_kc)
    assert conn.dan_to_kc.shape == (conn.n_kc, conn.n_dan)
    assert conn.dan_to_mbon.shape == (conn.n_mbon, conn.n_dan)


def test_connectivity_matrix_sparse_format(sample_connectivity_matrix):
    """Test that matrices use CSR sparse format."""
    conn = sample_connectivity_matrix

    # Biological rationale: MB circuits are sparse (~95-97% absent connections)
    # CSR format is optimal for row-slicing and matrix-vector products
    assert isinstance(conn.pn_to_kc, sp.csr_matrix)
    assert isinstance(conn.kc_to_mbon, sp.csr_matrix)
    assert isinstance(conn.dan_to_kc, sp.csr_matrix)
    assert isinstance(conn.dan_to_mbon, sp.csr_matrix)


def test_connectivity_matrix_properties(sample_connectivity_matrix):
    """Test n_pn, n_kc, n_mbon, n_dan properties."""
    conn = sample_connectivity_matrix

    assert conn.n_pn == 10
    assert conn.n_kc == 20
    assert conn.n_mbon == 5
    assert conn.n_dan == 8


def test_slice_kc_subtypes(sample_connectivity_matrix):
    """Test slicing by KC subtype."""
    conn = sample_connectivity_matrix

    # Biological rationale: Different KC subtypes support distinct memory functions
    # (γ for short-term, α/β for long-term). Slicing isolates memory systems.
    sliced = conn.slice_kc_subtypes(["ab", "g_main"])

    # PNs, MBONs, DANs unchanged
    assert sliced.n_pn == conn.n_pn
    assert sliced.n_mbon == conn.n_mbon
    assert sliced.n_dan == conn.n_dan

    # KCs reduced to only "ab" and "g_main" subtypes
    expected_kc_count = sum(
        1 for subtype in conn.kc_subtypes.values() if subtype in ["ab", "g_main"]
    )
    assert sliced.n_kc == expected_kc_count

    # Matrix shapes updated
    assert sliced.pn_to_kc.shape == (sliced.n_kc, sliced.n_pn)
    assert sliced.kc_to_mbon.shape == (sliced.n_mbon, sliced.n_kc)


def test_slice_kc_subtypes_invalid(sample_connectivity_matrix):
    """Test that slicing with invalid subtypes raises error."""
    conn = sample_connectivity_matrix

    with pytest.raises(ValueError, match="No KCs found"):
        conn.slice_kc_subtypes(["invalid_subtype"])


def test_pn_fan_in(sample_connectivity_matrix):
    """Test PN→KC fan-in extraction."""
    conn = sample_connectivity_matrix

    # Biological rationale: Each KC receives ~6-8 PN inputs (the "claw")
    # This method extracts inputs to a single KC for analysis
    kc_idx = 5
    pn_inputs = conn.pn_fan_in(kc_idx)

    assert pn_inputs.shape == (conn.n_pn,)
    assert isinstance(pn_inputs, np.ndarray)

    # Check consistency: should match row from pn_to_kc matrix
    expected = conn.pn_to_kc[kc_idx, :].toarray().ravel()
    np.testing.assert_array_equal(pn_inputs, expected)


def test_pn_fan_in_out_of_bounds(sample_connectivity_matrix):
    """Test that out-of-bounds KC index raises error."""
    conn = sample_connectivity_matrix

    with pytest.raises(IndexError):
        conn.pn_fan_in(999)  # Beyond n_kc


def test_mbon_fan_in(sample_connectivity_matrix):
    """Test KC→MBON fan-in extraction."""
    conn = sample_connectivity_matrix

    # Biological rationale: MBONs integrate across hundreds-thousands of KCs
    # This method extracts KC inputs to a single MBON
    mbon_idx = 2
    kc_inputs = conn.mbon_fan_in(mbon_idx)

    assert kc_inputs.shape == (conn.n_kc,)
    assert isinstance(kc_inputs, np.ndarray)

    # Check consistency
    expected = conn.kc_to_mbon[mbon_idx, :].toarray().ravel()
    np.testing.assert_array_equal(kc_inputs, expected)


def test_mbon_fan_in_out_of_bounds(sample_connectivity_matrix):
    """Test that out-of-bounds MBON index raises error."""
    conn = sample_connectivity_matrix

    with pytest.raises(IndexError):
        conn.mbon_fan_in(999)


def test_get_pn_indices_all(sample_connectivity_matrix):
    """Test getting all PN indices (no filter)."""
    conn = sample_connectivity_matrix

    pn_indices = conn.get_pn_indices(glomeruli=None)

    assert len(pn_indices) == conn.n_pn
    np.testing.assert_array_equal(pn_indices, np.arange(conn.n_pn))


def test_get_pn_indices_filtered(sample_connectivity_matrix):
    """Test filtering PNs by glomerulus."""
    conn = sample_connectivity_matrix

    # Biological rationale: Glomeruli define odorant receptor specificity
    # Filtering enables targeted PN activation by chemical structure
    target_glom = "glom_0"
    pn_indices = conn.get_pn_indices(glomeruli=[target_glom])

    # Verify all returned indices belong to target glomerulus
    for idx in pn_indices:
        pn_id = conn.pn_ids[idx]
        assert conn.pn_glomeruli[pn_id] == target_glom


def test_pn_to_kc_sparsity(sample_connectivity_matrix):
    """Test PN→KC sparsity calculation."""
    conn = sample_connectivity_matrix

    # Biological rationale: PN→KC should be ~95-97% sparse (only ~3-5% connections present)
    sparsity = conn.pn_to_kc_sparsity()

    assert 0.0 <= sparsity <= 1.0
    # For sample data: should be fairly sparse
    assert sparsity > 0.5  # At least 50% sparse


def test_kc_to_mbon_sparsity(sample_connectivity_matrix):
    """Test KC→MBON sparsity calculation."""
    conn = sample_connectivity_matrix

    # Biological rationale: KC→MBON less sparse than PN→KC but still sparse (~90-95%)
    sparsity = conn.kc_to_mbon_sparsity()

    assert 0.0 <= sparsity <= 1.0


def test_to_dict(sample_connectivity_matrix):
    """Test serialization to dictionary."""
    conn = sample_connectivity_matrix

    summary = conn.to_dict()

    # Check expected keys
    assert "n_pn" in summary
    assert "n_kc" in summary
    assert "n_mbon" in summary
    assert "n_dan" in summary
    assert "pn_to_kc_sparsity" in summary
    assert "kc_to_mbon_sparsity" in summary
    assert "n_glomeruli" in summary
    assert "kc_subtypes" in summary

    # Validate values
    assert summary["n_pn"] == conn.n_pn
    assert summary["n_kc"] == conn.n_kc
    assert summary["n_glomeruli"] == len(set(conn.pn_glomeruli.values()))
    assert isinstance(summary["kc_subtypes"], list)


def test_repr_includes_diagnostics(sample_connectivity_matrix):
    """Test __repr__ includes shape and sparsity info."""
    conn = sample_connectivity_matrix

    repr_str = repr(conn)

    # Should include neuron counts
    assert "PNs: 10" in repr_str
    assert "KCs: 20" in repr_str
    assert "MBONs: 5" in repr_str
    assert "DANs: 8" in repr_str

    # Should include matrix info
    assert "PN→KC" in repr_str
    assert "KC→MBON" in repr_str
    assert "sparse" in repr_str


def test_validation_wrong_pn_to_kc_shape():
    """Test that wrong PN→KC shape raises error."""
    # Biological validation: Matrix dimensions must match neuron counts
    pn_ids = np.arange(10, dtype=np.int64)
    kc_ids = np.arange(20, dtype=np.int64)
    mbon_ids = np.arange(5, dtype=np.int64)
    dan_ids = np.arange(8, dtype=np.int64)

    # Create wrong-shaped matrix (should be (20, 10), not (10, 20))
    wrong_pn_to_kc = sp.csr_matrix((10, 20), dtype=np.float64)

    with pytest.raises(ValueError, match="pn_to_kc shape"):
        ConnectivityMatrix(
            pn_ids=pn_ids,
            kc_ids=kc_ids,
            mbon_ids=mbon_ids,
            dan_ids=dan_ids,
            pn_to_kc=wrong_pn_to_kc,
            kc_to_mbon=sp.csr_matrix((5, 20)),
            dan_to_kc=sp.csr_matrix((20, 8)),
            dan_to_mbon=sp.csr_matrix((5, 8)),
        )


def test_validation_wrong_matrix_type():
    """Test that non-CSR matrices raise error."""
    pn_ids = np.arange(10, dtype=np.int64)
    kc_ids = np.arange(20, dtype=np.int64)
    mbon_ids = np.arange(5, dtype=np.int64)
    dan_ids = np.arange(8, dtype=np.int64)

    # Use COO format instead of CSR (should fail)
    wrong_format = sp.coo_matrix((20, 10), dtype=np.float64)

    with pytest.raises(TypeError, match="csr_matrix"):
        ConnectivityMatrix(
            pn_ids=pn_ids,
            kc_ids=kc_ids,
            mbon_ids=mbon_ids,
            dan_ids=dan_ids,
            pn_to_kc=wrong_format,
            kc_to_mbon=sp.csr_matrix((5, 20)),
            dan_to_kc=sp.csr_matrix((20, 8)),
            dan_to_mbon=sp.csr_matrix((5, 8)),
        )
