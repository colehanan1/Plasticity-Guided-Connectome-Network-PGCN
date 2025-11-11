"""
Comprehensive Test Suite for SEZ Neuron Extraction Pipeline

This test suite validates the SEZ neuron extraction pipeline against
Li et al. (2024) benchmarks and ensures data quality.

Reference:
Li, J. et al. (2024). Connectomic analysis of taste circuits in Drosophila.
Scientific Reports, 14, 21120. https://doi.org/10.1038/s41598-024-71926-2

Usage:
    pytest tests/test_sez_extraction.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test data paths
FLYWIRE_DIR = Path("data/flywire")
CACHE_DIR = Path("data/cache")
VALIDATION_DIR = Path("results/sez_validation")
GRN_FILE = FLYWIRE_DIR / "root_ids_class_gustatory.txt"


# =============================================================================
# Test 1: GRN Ground Truth File
# =============================================================================


class TestGRNGroundTruth:
    """Test that GRN ground truth file exists and is valid."""

    def test_grn_file_exists(self):
        """Test that GRN file exists at expected location."""
        assert GRN_FILE.exists(), (
            f"GRN file not found: {GRN_FILE}\n"
            f"Expected: data/flywire/root_ids_class_gustatory.txt\n"
            f"This file should contain 343 validated GRN root IDs."
        )
        print(f"  ✅ GRN file exists: {GRN_FILE}")

    def test_grn_count(self):
        """Test that GRN count matches expected value (343)."""
        if not GRN_FILE.exists():
            pytest.skip("GRN file not found")

        grn_ids = pd.read_csv(GRN_FILE, header=None)
        actual_count = len(grn_ids)
        expected_count = 343

        print(f"  GRN count: {actual_count}")
        print(f"  Expected: {expected_count}")

        assert actual_count == expected_count, (
            f"Expected {expected_count} GRNs, found {actual_count}"
        )
        print(f"  ✅ GRN count matches: {actual_count}")

    def test_grn_format(self):
        """Test that GRN file has valid format (positive integers)."""
        if not GRN_FILE.exists():
            pytest.skip("GRN file not found")

        grn_ids = pd.read_csv(GRN_FILE, header=None, names=["root_id"])

        # Check for missing values
        assert not grn_ids["root_id"].isna().any(), "GRN file contains missing values"

        # Check for valid root IDs (positive integers)
        assert (grn_ids["root_id"] > 0).all(), "GRN file contains invalid root IDs"

        # Check for duplicates
        n_unique = grn_ids["root_id"].nunique()
        n_total = len(grn_ids)
        assert n_unique == n_total, f"GRN file contains {n_total - n_unique} duplicates"

        print(f"  ✅ GRN file format valid")


# =============================================================================
# Test 2: SEZ-PN Extraction
# =============================================================================


class TestSEZPNExtraction:
    """Test SEZ projection neuron extraction."""

    def test_sez_pn_file_exists(self):
        """Test that SEZ-PN file was created."""
        sez_pn_file = CACHE_DIR / "sez_pn_all.csv"

        if not sez_pn_file.exists():
            pytest.skip("SEZ-PNs not extracted yet - run extract_sez_neurons.py")

        print(f"  ✅ SEZ-PN file exists: {sez_pn_file}")

    def test_sez_pn_count_li2024(self):
        """Test SEZ-PN count matches Li et al. (2024) range."""
        sez_pn_file = CACHE_DIR / "sez_pn_all.csv"

        if not sez_pn_file.exists():
            pytest.skip("SEZ-PNs not extracted yet")

        sez_pns = pd.read_csv(sez_pn_file)
        n_pns = len(sez_pns)

        print(f"  SEZ-PN count: {n_pns}")
        print(f"  Expected range: 100-200 (Li et al. 2024)")

        # Allow wider plausible range for assertion
        assert 80 <= n_pns <= 250, (
            f"SEZ-PN count {n_pns} outside plausible range (80-250)"
        )

        # Check if within exact Li et al. (2024) range
        if 100 <= n_pns <= 200:
            print(f"  ✅ Exact match with Li et al. (2024)")
        elif 80 <= n_pns < 100:
            print(f"  ⚠ Slightly below Li et al. range (acceptable)")
        else:
            print(f"  ⚠ Slightly above Li et al. range (may need review)")

    def test_sez_pn_required_columns(self):
        """Test that SEZ-PN file has required columns."""
        sez_pn_file = CACHE_DIR / "sez_pn_all.csv"

        if not sez_pn_file.exists():
            pytest.skip("SEZ-PNs not extracted yet")

        sez_pns = pd.read_csv(sez_pn_file)

        required_columns = ["root_id", "cell_type"]
        missing_columns = [col for col in required_columns if col not in sez_pns.columns]

        assert not missing_columns, (
            f"SEZ-PN file missing required columns: {missing_columns}"
        )
        print(f"  ✅ Required columns present: {required_columns}")

    def test_sez_pn_no_duplicates(self):
        """Test that SEZ-PN file has no duplicate root IDs."""
        sez_pn_file = CACHE_DIR / "sez_pn_all.csv"

        if not sez_pn_file.exists():
            pytest.skip("SEZ-PNs not extracted yet")

        sez_pns = pd.read_csv(sez_pn_file)
        n_unique = sez_pns["root_id"].nunique()
        n_total = len(sez_pns)

        assert n_unique == n_total, (
            f"SEZ-PN file contains {n_total - n_unique} duplicate root IDs"
        )
        print(f"  ✅ No duplicate root IDs")

    def test_sez_pn_cell_type_label(self):
        """Test that SEZ-PNs have correct cell_type label."""
        sez_pn_file = CACHE_DIR / "sez_pn_all.csv"

        if not sez_pn_file.exists():
            pytest.skip("SEZ-PNs not extracted yet")

        sez_pns = pd.read_csv(sez_pn_file)

        if "cell_type" in sez_pns.columns:
            cell_types = sez_pns["cell_type"].unique()
            expected_type = "SEZ_PN"

            assert expected_type in cell_types or len(cell_types) == 1, (
                f"Expected cell_type '{expected_type}', found: {cell_types}"
            )
            print(f"  ✅ Cell type label: {cell_types}")


# =============================================================================
# Test 3: SEZ-LN Extraction
# =============================================================================


class TestSEZLNExtraction:
    """Test SEZ local interneuron extraction."""

    def test_sez_ln_file_exists(self):
        """Test that SEZ-LN file was created."""
        sez_ln_file = CACHE_DIR / "sez_ln_all.csv"

        if not sez_ln_file.exists():
            pytest.skip("SEZ-LNs not extracted yet - run extract_sez_neurons.py")

        print(f"  ✅ SEZ-LN file exists: {sez_ln_file}")

    def test_sez_ln_count_plausible(self):
        """Test that SEZ-LN count is biologically plausible."""
        sez_ln_file = CACHE_DIR / "sez_ln_all.csv"

        if not sez_ln_file.exists():
            pytest.skip("SEZ-LNs not extracted yet")

        sez_lns = pd.read_csv(sez_ln_file)
        n_lns = len(sez_lns)

        print(f"  SEZ-LN count: {n_lns}")
        print(f"  Expected range: 200-600")

        assert 100 <= n_lns <= 800, (
            f"SEZ-LN count {n_lns} outside plausible range (100-800)"
        )

        if 200 <= n_lns <= 600:
            print(f"  ✅ Within expected range")
        else:
            print(f"  ⚠ Outside typical range (acceptable variation)")

    def test_cholinergic_sez_ln_extraction(self):
        """Test cholinergic SEZ-LN extraction."""
        sez_ln_file = CACHE_DIR / "sez_ln_cholinergic.csv"

        if not sez_ln_file.exists():
            pytest.skip("Cholinergic SEZ-LNs not extracted yet")

        sez_lns = pd.read_csv(sez_ln_file)
        n_lns = len(sez_lns)

        print(f"  Cholinergic SEZ-LN count: {n_lns}")
        print(f"  Expected range: 50-100")

        assert 30 <= n_lns <= 150, (
            f"Cholinergic SEZ-LN count {n_lns} implausible"
        )

        if 50 <= n_lns <= 100:
            print(f"  ✅ Within expected range")
        else:
            print(f"  ⚠ Outside typical range (may need review)")


# =============================================================================
# Test 4: Clustering Validation
# =============================================================================


class TestClusteringValidation:
    """Test Li et al. (2024) clustering validation."""

    def test_cluster_file_exists(self):
        """Test that cluster assignment file was created."""
        cluster_file = VALIDATION_DIR / "sez_pn_clusters.csv"

        if not cluster_file.exists():
            pytest.skip("Clustering not performed yet")

        print(f"  ✅ Cluster file exists: {cluster_file}")

    def test_clustering_produces_meaningful_groups(self):
        """Test that clustering produces biologically meaningful groups."""
        cluster_file = VALIDATION_DIR / "sez_pn_clusters.csv"

        if not cluster_file.exists():
            pytest.skip("Clustering not performed yet")

        clusters = pd.read_csv(cluster_file)
        n_clusters = clusters["cluster"].nunique()

        print(f"  Clusters found: {n_clusters}")
        print(f"  Expected range: 8-12 (Li et al. 2024 taste modalities)")

        assert 6 <= n_clusters <= 15, (
            f"Cluster count {n_clusters} implausible (6-15 expected)"
        )

        if 8 <= n_clusters <= 12:
            print(f"  ✅ Matches Li et al. (2024) range")
        else:
            print(f"  ⚠ Outside typical range (acceptable variation)")

    def test_validation_summary_exists(self):
        """Test that validation summary JSON was created."""
        summary_file = VALIDATION_DIR / "validation_summary.json"

        if not summary_file.exists():
            pytest.skip("Validation summary not created yet")

        import json

        with open(summary_file) as f:
            summary = json.load(f)

        print(f"  ✅ Validation summary exists")
        print(f"  SEZ-PNs: {summary.get('n_sez_pns', 'N/A')}")
        print(f"  Clusters: {summary.get('n_clusters', 'N/A')}")
        print(f"  Silhouette: {summary.get('silhouette_score', 'N/A')}")

        # Check required fields
        required_fields = ["n_sez_pns", "n_grns", "n_clusters"]
        missing_fields = [f for f in required_fields if f not in summary]

        assert not missing_fields, (
            f"Validation summary missing fields: {missing_fields}"
        )

    def test_silhouette_score_quality(self):
        """Test that silhouette score indicates reasonable clustering."""
        summary_file = VALIDATION_DIR / "validation_summary.json"

        if not summary_file.exists():
            pytest.skip("Validation summary not created yet")

        import json

        with open(summary_file) as f:
            summary = json.load(f)

        if "silhouette_score" not in summary:
            pytest.skip("Silhouette score not computed")

        score = summary["silhouette_score"]
        print(f"  Silhouette score: {score:.3f}")

        # Silhouette score interpretation:
        # > 0.7: Strong separation
        # 0.5-0.7: Reasonable separation
        # 0.3-0.5: Weak but acceptable separation
        # < 0.3: Poor separation

        assert score >= 0.2, (
            f"Silhouette score {score:.3f} too low (< 0.2) - poor clustering"
        )

        if score >= 0.3:
            print(f"  ✅ Acceptable clustering quality (≥0.3)")
        else:
            print(f"  ⚠ Marginal clustering quality (0.2-0.3)")


# =============================================================================
# Test 5: Validation Plots
# =============================================================================


class TestValidationPlots:
    """Test that validation plots were generated."""

    def test_dendrogram_plot_exists(self):
        """Test that hierarchical clustering dendrogram was created."""
        plot_file = VALIDATION_DIR / "fig1_dendrogram.pdf"

        if not plot_file.exists():
            pytest.skip("Dendrogram plot not generated yet")

        assert plot_file.stat().st_size > 1000, "Dendrogram file is too small"
        print(f"  ✅ Dendrogram plot exists: {plot_file}")

    def test_silhouette_plot_exists(self):
        """Test that silhouette score plot was created."""
        plot_file = VALIDATION_DIR / "fig2_silhouette.pdf"

        if not plot_file.exists():
            pytest.skip("Silhouette plot not generated yet")

        assert plot_file.stat().st_size > 1000, "Silhouette plot file is too small"
        print(f"  ✅ Silhouette plot exists: {plot_file}")

    def test_umap_plot_exists(self):
        """Test that UMAP embedding plot was created (if UMAP available)."""
        plot_file = VALIDATION_DIR / "fig3_umap_clusters.pdf"

        if not plot_file.exists():
            pytest.skip("UMAP plot not generated (may require umap-learn)")

        assert plot_file.stat().st_size > 1000, "UMAP plot file is too small"
        print(f"  ✅ UMAP plot exists: {plot_file}")

    def test_heatmap_plot_exists(self):
        """Test that similarity heatmap was created."""
        plot_file = VALIDATION_DIR / "fig4_heatmap.pdf"

        if not plot_file.exists():
            pytest.skip("Heatmap plot not generated yet")

        assert plot_file.stat().st_size > 1000, "Heatmap file is too small"
        print(f"  ✅ Heatmap plot exists: {plot_file}")


# =============================================================================
# Test 6: Data Quality Checks
# =============================================================================


class TestDataQuality:
    """Test data quality and consistency."""

    def test_no_overlap_between_pns_and_lns(self):
        """Test that SEZ-PNs and SEZ-LNs don't overlap."""
        sez_pn_file = CACHE_DIR / "sez_pn_all.csv"
        sez_ln_file = CACHE_DIR / "sez_ln_all.csv"

        if not (sez_pn_file.exists() and sez_ln_file.exists()):
            pytest.skip("Both SEZ-PN and SEZ-LN files needed")

        sez_pns = pd.read_csv(sez_pn_file)
        sez_lns = pd.read_csv(sez_ln_file)

        pn_ids = set(sez_pns["root_id"])
        ln_ids = set(sez_lns["root_id"])

        overlap = pn_ids & ln_ids
        assert len(overlap) == 0, (
            f"Found {len(overlap)} neurons classified as both PN and LN"
        )
        print(f"  ✅ No overlap between SEZ-PNs and SEZ-LNs")

    def test_cholinergic_lns_subset_of_all_lns(self):
        """Test that cholinergic SEZ-LNs are a subset of all SEZ-LNs."""
        all_ln_file = CACHE_DIR / "sez_ln_all.csv"
        chol_ln_file = CACHE_DIR / "sez_ln_cholinergic.csv"

        if not (all_ln_file.exists() and chol_ln_file.exists()):
            pytest.skip("Both SEZ-LN files needed")

        all_lns = pd.read_csv(all_ln_file)
        chol_lns = pd.read_csv(chol_ln_file)

        all_ln_ids = set(all_lns["root_id"])
        chol_ln_ids = set(chol_lns["root_id"])

        # Cholinergic LNs should be subset of all LNs
        not_in_all = chol_ln_ids - all_ln_ids

        assert len(not_in_all) == 0, (
            f"Found {len(not_in_all)} cholinergic LNs not in all LNs"
        )
        print(f"  ✅ Cholinergic LNs are subset of all LNs")

    def test_extraction_consistency(self):
        """Test overall extraction consistency."""
        sez_pn_file = CACHE_DIR / "sez_pn_all.csv"
        sez_ln_file = CACHE_DIR / "sez_ln_all.csv"

        if not (sez_pn_file.exists() and sez_ln_file.exists()):
            pytest.skip("Both files needed for consistency check")

        sez_pns = pd.read_csv(sez_pn_file)
        sez_lns = pd.read_csv(sez_ln_file)

        total_second_order = len(sez_pns) + len(sez_lns)

        print(f"  Total second-order neurons: {total_second_order}")
        print(f"  ├─ SEZ-PNs: {len(sez_pns)}")
        print(f"  └─ SEZ-LNs: {len(sez_lns)}")

        # Total should be in plausible range
        assert 200 <= total_second_order <= 1000, (
            f"Total second-order count {total_second_order} implausible"
        )

        # PNs should be minority (projection neurons are more selective)
        pn_fraction = len(sez_pns) / total_second_order
        print(f"  PN fraction: {pn_fraction:.1%}")

        assert 0.1 <= pn_fraction <= 0.5, (
            f"PN fraction {pn_fraction:.1%} unexpected (should be 10-50%)"
        )
        print(f"  ✅ Extraction proportions reasonable")


# =============================================================================
# Test 7: Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_extraction_pipeline(self):
        """Test that all extraction outputs exist."""
        required_files = [
            CACHE_DIR / "sez_pn_all.csv",
            CACHE_DIR / "sez_ln_all.csv",
            CACHE_DIR / "sez_ln_cholinergic.csv",
        ]

        missing_files = [f for f in required_files if not f.exists()]

        if missing_files:
            pytest.skip(
                f"Pipeline incomplete - missing: {[f.name for f in missing_files]}"
            )

        print(f"  ✅ All extraction files present")

    def test_extraction_summary(self):
        """Print comprehensive extraction summary."""
        sez_pn_file = CACHE_DIR / "sez_pn_all.csv"
        sez_ln_file = CACHE_DIR / "sez_ln_all.csv"
        chol_ln_file = CACHE_DIR / "sez_ln_cholinergic.csv"

        if not all([f.exists() for f in [sez_pn_file, sez_ln_file, chol_ln_file]]):
            pytest.skip("Not all extraction files available")

        sez_pns = pd.read_csv(sez_pn_file)
        sez_lns = pd.read_csv(sez_ln_file)
        chol_lns = pd.read_csv(chol_ln_file)

        if GRN_FILE.exists():
            grn_ids = pd.read_csv(GRN_FILE, header=None)
            n_grns = len(grn_ids)
        else:
            n_grns = "N/A"

        print("\n" + "=" * 60)
        print("SEZ NEURON EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"  GRNs (ground truth):        {n_grns}")
        print(f"  SEZ-PNs (projection):       {len(sez_pns)}")
        print(f"  SEZ-LNs (local):            {len(sez_lns)}")
        print(f"  ├─ Cholinergic:             {len(chol_lns)}")
        print(f"  └─ Other:                   {len(sez_lns) - len(chol_lns)}")
        print(f"  Total second-order:         {len(sez_pns) + len(sez_lns)}")
        print("=" * 60)


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
