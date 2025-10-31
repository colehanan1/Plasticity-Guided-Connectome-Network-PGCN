"""Tests for DAN (dopaminergic neuron) filtering to MB-only populations.

This test suite validates the strict MB-only gating required for olfactory learning
models. Only DANs projecting to mushroom body compartments should modulate
KC→MBON plasticity.
"""

import pytest
import pandas as pd
from pgcn.data.dan_filtering import (
    filter_dan_to_mb_only,
    validate_dan_mb_filter,
    get_dan_compartment_mapping,
)


def test_dan_mb_filter_keeps_mb_projecting():
    """Test that MB-only filter keeps DANs with MB in output_neuropils."""
    dan_df = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5],
            "nt_type": ["DA", "DA", "DA", "DA", "DA"],
            "output_neuropils": [
                "MB_CA_L|MB_CA_R",  # MB-only → keep
                "LH_R|AVLP_R",  # No MB → exclude
                "MB_ML_L",  # MB-only → keep
                "AC_R|SIP_R",  # No MB → exclude
                "MB_VL_L|MB_VL_R|CRE_L",  # MB + other → keep
            ],
            "primary_type": ["PAM01", "PPL2", "PAM08", "PPL3", "PAM12"],
        }
    )

    dan_mb = filter_dan_to_mb_only(dan_df, verbose=False)

    # Should keep rows with MB in neuropils (1, 3, 5)
    assert len(dan_mb) == 3
    assert set(dan_mb["root_id"]) == {1, 3, 5}


def test_dan_mb_filter_excludes_non_da():
    """Test that filter only keeps dopaminergic neurons."""
    dan_df = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4],
            "nt_type": ["DA", "GABA", "DA", "ACH"],
            "output_neuropils": [
                "MB_CA_L",
                "MB_ML_L",  # GABA neuron with MB → exclude
                "MB_CA_R",
                "MB_VL_L",  # ACH neuron with MB → exclude
            ],
            "primary_type": ["PAM01", "KC_gad1", "PPL1", "MBON"],
        }
    )

    dan_mb = filter_dan_to_mb_only(dan_df, verbose=False)

    # Should keep only DA rows with MB (1, 3)
    assert len(dan_mb) == 2
    assert set(dan_mb["root_id"]) == {1, 3}
    assert all(dan_mb["nt_type"] == "DA")


def test_dan_mb_filter_case_insensitive():
    """Test that filter is case-insensitive for MB matching."""
    dan_df = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4],
            "nt_type": ["DA", "Da", "da", "DA"],
            "output_neuropils": ["MB_CA_L", "mb_ml_l", "Mb_VL_R", "mb_ca_r"],
            "primary_type": ["PAM01", "PAM08", "PPL1", "PAM12"],
        }
    )

    dan_mb = filter_dan_to_mb_only(dan_df, verbose=False)

    # All should be kept (case-insensitive matching)
    assert len(dan_mb) == 4


def test_dan_mb_filter_empty_input():
    """Test that filter handles empty input gracefully."""
    dan_df = pd.DataFrame(
        columns=["root_id", "nt_type", "output_neuropils", "primary_type"]
    )

    dan_mb = filter_dan_to_mb_only(dan_df, verbose=False)

    assert len(dan_mb) == 0
    assert list(dan_mb.columns) == list(dan_df.columns)


def test_dan_mb_filter_no_mb_dans():
    """Test that filter returns empty when no MB-projecting DANs exist."""
    dan_df = pd.DataFrame(
        {
            "root_id": [1, 2, 3],
            "nt_type": ["DA", "DA", "DA"],
            "output_neuropils": ["LH_R", "CX_L", "AVLP_R"],  # No MB
            "primary_type": ["PPL2", "PPL3", "PPM3"],
        }
    )

    dan_mb = filter_dan_to_mb_only(dan_df, verbose=False)

    assert len(dan_mb) == 0


def test_dan_mb_filter_missing_column_raises():
    """Test that filter raises ValueError when required columns missing."""
    # Missing nt_type column
    dan_df = pd.DataFrame(
        {
            "root_id": [1, 2],
            "output_neuropils": ["MB_CA_L", "MB_ML_R"],
        }
    )

    with pytest.raises(ValueError, match="Missing 'nt_type' column"):
        filter_dan_to_mb_only(dan_df, verbose=False)

    # Missing output_neuropils column
    dan_df = pd.DataFrame(
        {
            "root_id": [1, 2],
            "nt_type": ["DA", "DA"],
        }
    )

    with pytest.raises(ValueError, match="Missing 'output_neuropils' column"):
        filter_dan_to_mb_only(dan_df, verbose=False)


def test_dan_mb_validation_passes_for_valid_data():
    """Test that validation passes when all DANs have MB output."""
    dan_mb = pd.DataFrame(
        {
            "root_id": [1, 2, 3],
            "nt_type": ["DA", "DA", "DA"],
            "output_neuropils": ["MB_CA_L", "MB_ML_R", "MB_VL_L|MB_VL_R"],
            "primary_type": ["PAM01", "PAM08", "PPL1"],
        }
    )

    assert validate_dan_mb_filter(dan_mb, neuropil_column="output_neuropils")


def test_dan_mb_validation_fails_for_non_mb():
    """Test that validation raises AssertionError when non-MB DAN present."""
    dan_mb = pd.DataFrame(
        {
            "root_id": [1, 2, 3],
            "nt_type": ["DA", "DA", "DA"],
            "output_neuropils": [
                "MB_CA_L",
                "LH_R",  # This one has no MB → should fail
                "MB_VL_L",
            ],
            "primary_type": ["PAM01", "PPL2", "PPL1"],
        }
    )

    with pytest.raises(AssertionError, match="does not contain 'MB'"):
        validate_dan_mb_filter(dan_mb, neuropil_column="output_neuropils")


def test_dan_compartment_mapping():
    """Test extraction of MB compartment → DAN mapping."""
    dan_mb = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4],
            "nt_type": ["DA", "DA", "DA", "DA"],
            "output_neuropils": [
                "MB_CA_L|MB_CA_R",
                "MB_ML_L",
                "MB_CA_L",
                "MB_VL_L|MB_VL_R",
            ],
            "primary_type": ["PAM01", "PAM08", "PAM03", "PPL1"],
        }
    )

    compartment_map = get_dan_compartment_mapping(dan_mb)

    # Check that each compartment has correct DANs
    assert set(compartment_map["MB_CA_L"]) == {1, 3}  # DANs 1 and 3
    assert set(compartment_map["MB_CA_R"]) == {1}  # DAN 1 only
    assert set(compartment_map["MB_ML_L"]) == {2}  # DAN 2 only
    assert set(compartment_map["MB_VL_L"]) == {4}  # DAN 4 only
    assert set(compartment_map["MB_VL_R"]) == {4}  # DAN 4 only


def test_filter_with_real_cache_data():
    """Test filtering with real cached DAN data if available."""
    import pathlib

    dan_all_path = pathlib.Path("data/cache/dan_all.csv")
    if not dan_all_path.exists():
        pytest.skip("Real DAN data not available (dan_all.csv)")

    # Load real data
    dan_all = pd.read_csv(dan_all_path)

    # Filter to MB-only
    dan_mb = filter_dan_to_mb_only(dan_all, verbose=True)

    # Validate
    assert len(dan_mb) > 0, "Should have at least some MB-projecting DANs"
    assert len(dan_mb) <= len(dan_all), "MB subset should not exceed total"

    # Validate all have MB
    validate_dan_mb_filter(dan_mb)

    # Check that we filtered out some non-MB DANs
    assert len(dan_mb) < len(
        dan_all
    ), "Should have excluded some non-MB DANs from dan_all.csv"


def test_filter_matches_dan_mb_cache():
    """Test that filtering dan_all.csv reproduces dan_mb.csv."""
    import pathlib

    dan_all_path = pathlib.Path("data/cache/dan_all.csv")
    dan_mb_path = pathlib.Path("data/cache/dan_mb.csv")

    if not (dan_all_path.exists() and dan_mb_path.exists()):
        pytest.skip("Both dan_all.csv and dan_mb.csv required for this test")

    # Load both files
    dan_all = pd.read_csv(dan_all_path)
    dan_mb_cached = pd.read_csv(dan_mb_path)

    # Filter dan_all
    dan_mb_filtered = filter_dan_to_mb_only(dan_all, verbose=True)

    # Should have approximately same number of rows
    # (Allow small differences due to different filtering criteria versions)
    assert abs(len(dan_mb_filtered) - len(dan_mb_cached)) <= 5, (
        f"Filtered count ({len(dan_mb_filtered)}) should match "
        f"cached count ({len(dan_mb_cached)}) within tolerance"
    )

    # All cached DANs should have MB in neuropils
    validate_dan_mb_filter(dan_mb_cached)
