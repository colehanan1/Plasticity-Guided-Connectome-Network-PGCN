"""Extract olfactory projection neurons (ALPNs) and PN→KC connectivity statistics."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from config import paths
from data_loaders.flywire_local import FlyWireLocalDataLoader
from data_loaders.neuron_classification import get_kc_neurons

# Canonical glomerulus catalogue adapted from community standards and hemibrain exports.
_CANONICAL_GLOMERULI_SET = {
    "DA1",
    "DA2",
    "DA3",
    "DA4",
    "DA4l",
    "DA4m",
    "DA5",
    "DA6",
    "DA7",
    "DL1",
    "DL2",
    "DL2d",
    "DL2v",
    "DL3",
    "DL4",
    "DL5",
    "DM1",
    "DM2",
    "DM3",
    "DM4",
    "DM5",
    "DM6",
    "DM7",
    "DM8",
    "DP1",
    "DP1l",
    "DP1m",
    "DP2",
    "DP3",
    "DP4",
    "DP5",
    "DP6",
    "DP7",
    "VA1d",
    "VA1v",
    "VA2",
    "VA3",
    "VA4",
    "VA5",
    "VA6",
    "VA7",
    "VA7d",
    "VA7l",
    "VA7m",
    "VA7v",
    "VA8",
    "VA9",
    "VC1",
    "VC2",
    "VC3",
    "VC4",
    "VC5",
    "VL1",
    "VL2",
    "VL3",
    "VM1",
    "VM2",
    "VM3",
    "VM4",
    "VM5",
    "VM5d",
    "VM5v",
    "VM6",
    "VM7",
    "VM7d",
    "VM7v",
    "VM8",
    "VM9",
    "VP1",
    "VP2",
    "VP3",
    "VP4",
    "VP5",
    "DC1",
    "DC2",
    "DC3",
    "DC4",
    "DC5",
    "DC6",
    "DC7",
}


def _glomerulus_sort_key(value: str) -> tuple[str, int, str]:
    match = re.match(r"([A-Z]+)(\d+)(.*)", value)
    if match:
        prefix, digits, suffix = match.groups()
        return (prefix, int(digits), suffix)
    return (value, 0, "")


_CANONICAL_GLOMERULI: Sequence[str] = tuple(
    sorted(_CANONICAL_GLOMERULI_SET, key=_glomerulus_sort_key)
)

_CANONICAL_LOOKUP = {value.upper(): value for value in _CANONICAL_GLOMERULI}
_ALLOWED_PREFIXES = {
    match.group(1)
    for value in _CANONICAL_GLOMERULI
    if (match := re.match(r"([A-Z]+)\d", value))
}
_GLOMERULUS_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


@dataclass(slots=True)
class ExtractionConfig:
    """Runtime configuration for ALPN extraction."""

    dataset_dir: Path
    output_dir: Path
    min_synapses: int = 5


def _load_dataframe(loader: FlyWireLocalDataLoader, name: str, load_fn) -> pd.DataFrame:
    try:
        df = load_fn()
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        msg = f"Required file for '{name}' not found: {exc}"
        raise SystemExit(msg) from exc
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Failed to load dataset '{name}': {exc}"
        raise SystemExit(msg) from exc
    print(f"Loaded {name}: {len(df):,} rows")
    return df


def _normalise_string_series(series: pd.Series) -> pd.Series:
    """Standardise string columns for reliable comparisons."""

    normalised = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )
    return normalised.replace({"": pd.NA})


def _parse_processed_label_entry(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, bytes))]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if isinstance(item, (str, bytes))]
        return [value]
    return []


def _build_processed_label_lookup(processed_df: pd.DataFrame | None) -> Dict[int, List[str]]:
    if processed_df is None or processed_df.empty:
        return {}
    if "root_id" not in processed_df.columns:
        return {}

    label_column = None
    for candidate in ("processed_labels", "labels", "annotations"):
        if candidate in processed_df.columns:
            label_column = candidate
            break
    if label_column is None:
        # Fallback: first object column that is not root_id
        for column in processed_df.columns:
            if column == "root_id":
                continue
            if processed_df[column].dtype == object:
                label_column = column
                break
    if label_column is None:
        return {}

    lookup: Dict[int, List[str]] = {}
    for row in processed_df.itertuples(index=False):
        root_id = getattr(row, "root_id", None)
        if pd.isna(root_id):
            continue
        labels = _parse_processed_label_entry(getattr(row, label_column))
        if labels:
            try:
                lookup[int(root_id)] = labels
            except (TypeError, ValueError):
                continue
    return lookup


def _normalise_glomerulus_token(token: str) -> str | None:
    token = token.strip()
    if not token:
        return None
    token = token.replace("glomerulus", "").replace("-", "").replace("_", "")
    token = token.strip()
    if not token:
        return None
    upper = token.upper()
    if upper in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[upper]

    prefix = "".join(char for char in upper if char.isalpha())
    digits = "".join(char for char in upper if char.isdigit())
    suffix = upper[len(prefix) + len(digits) :]
    if not prefix or not digits:
        return None
    if prefix[:2] not in _ALLOWED_PREFIXES:
        return None
    canonical_key = f"{prefix}{digits}{suffix.lower()}".upper()
    if canonical_key in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[canonical_key]
    suffix_formatted = suffix.lower()
    if suffix_formatted and suffix_formatted not in {"d", "v", "l", "m"}:
        # Unsupported suffix, drop to base form
        suffix_formatted = ""
    return f"{prefix}{digits}{suffix_formatted}"


def _extract_glomeruli_from_candidates(candidates: Iterable[str]) -> List[str]:
    glomeruli: List[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        for segment in _GLOMERULUS_SPLIT_RE.split(candidate):
            if not segment:
                continue
            normalised = _normalise_glomerulus_token(segment)
            if normalised and normalised not in glomeruli:
                glomeruli.append(normalised)
    return glomeruli


def _infer_glomeruli(pn_df: pd.DataFrame, processed_lookup: Dict[int, List[str]]) -> pd.DataFrame:
    glomerulus_lists: List[List[str]] = []
    primary_labels: List[str | pd.NA] = []

    for row in pn_df.itertuples(index=False):
        candidates: List[str] = []
        for column in (
            "cell_type",
            "cell_type_aliases",
            "super_class",
            "class",
            "sub_class",
            "group",
        ):
            value = getattr(row, column, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        root_id = row.root_id
        if not pd.isna(root_id):
            labels = processed_lookup.get(int(root_id), [])
            candidates.extend(labels)
        glomeruli = _extract_glomeruli_from_candidates(candidates)
        glomerulus_lists.append(glomeruli)
        primary_labels.append(glomeruli[0] if glomeruli else pd.NA)

    pn_df = pn_df.copy()
    pn_df["glomeruli"] = glomerulus_lists
    pn_df["primary_glomerulus"] = primary_labels
    pn_df["has_multiple_glomeruli"] = pn_df["glomeruli"].apply(lambda values: len(values) > 1)
    return pn_df


def extract_alpns(
    classification_df: pd.DataFrame,
    neurons_df: pd.DataFrame,
    cell_types_df: Optional[pd.DataFrame],
    processed_labels_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Filter ALPNs with neurotransmitter and glomerulus annotations."""

    required_columns = {"root_id", "class"}
    missing_columns = required_columns - set(classification_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise SystemExit(f"classification table missing required columns: {missing}")

    classification_df = classification_df.copy()
    classification_df["class_normalised"] = _normalise_string_series(
        classification_df["class"]
    )

    if "super_class" in classification_df.columns:
        classification_df["super_class_normalised"] = _normalise_string_series(
            classification_df["super_class"]
        )
    else:
        classification_df["super_class_normalised"] = pd.NA

    if "flow" in classification_df.columns:
        classification_df["flow_normalised"] = _normalise_string_series(
            classification_df["flow"]
        )
    else:
        classification_df["flow_normalised"] = pd.NA

    class_mask = classification_df["class_normalised"] == "ALPN"
    pn_class_only = classification_df[class_mask].copy()
    print(f"ALPN candidates with class=='ALPN': {len(pn_class_only):,}")
    if pn_class_only.empty:
        print(
            "Warning: classification export lacks rows with class=='ALPN' after normalisation."
        )

    super_class_mask = classification_df["super_class_normalised"].isin({"ASCENDING"})
    pn_candidates = classification_df[class_mask & super_class_mask].copy()
    print(
        "ALPN candidates after class & super_class filters: "
        f"{len(pn_candidates):,}"
    )

    if pn_candidates.empty and not pn_class_only.empty:
        print(
            "Super-class filter removed all ALPN candidates. Falling back to class-only "
            f"selection ({len(pn_class_only):,} rows)."
        )
        pn_candidates = pn_class_only

    if pn_candidates.empty:
        print(
            "No ALPN candidates matched the class criteria even after relaxing filters."
        )

    if not pn_candidates.empty:
        print("Unique 'flow' values in ALPN candidates (normalised):")
        flow_counts = pn_candidates["flow_normalised"].value_counts(dropna=False)
        if flow_counts.empty:
            print("  <no flow annotations present>")
        else:
            print(flow_counts.to_string())

    join_columns = ["root_id", "nt_type", "group", "output_neuropils"]
    available_join_columns = [column for column in join_columns if column in neurons_df.columns]
    neurotransmitters = neurons_df.loc[:, available_join_columns].drop_duplicates(
        subset=["root_id"]
    )
    pn_enriched = pn_candidates.merge(
        neurotransmitters, on="root_id", how="left", indicator=True
    )

    print(f"ALPN candidates after merging neurons table: {len(pn_enriched):,}")

    missing_nt = int((pn_enriched["_merge"] == "left_only").sum())
    if missing_nt:
        print(
            f"Warning: {missing_nt} ALPN candidates lack neurotransmitter annotations;"
            " treating them as UNKNOWN."
        )
    pn_enriched = pn_enriched.drop(columns=["_merge"])

    if "nt_type" not in pn_enriched.columns:
        raise SystemExit("neurons table is missing 'nt_type' column")

    pn_enriched["nt_type"] = pn_enriched["nt_type"].astype("string")
    pn_enriched["nt_type_normalised"] = (
        pn_enriched["nt_type"].str.strip().str.upper().replace({"": pd.NA})
    )

    nt_series = pn_enriched["nt_type_normalised"]
    print("\nUnique 'nt_type' values in ALPN candidates (normalised):")
    nt_counts = nt_series.value_counts(dropna=False)
    if nt_counts.empty:
        print("  <no neurotransmitter annotations present>")
    else:
        print(nt_counts.to_string())
    ach_mask = nt_series == "ACH"
    cholinergic_mask = nt_series == "CHOLINERGIC"
    glut_mask = nt_series.str.contains("GLUT", case=False, na=False)
    excitatory_mask = ach_mask | cholinergic_mask | glut_mask
    gabanergic_mask = nt_series.str.contains("GABA", case=False, na=False)
    unknown_mask = nt_series.isna()

    gaba_excluded = int(gabanergic_mask.sum())
    non_excitatory_excluded = int((~excitatory_mask & ~gabanergic_mask).sum())

    pn_filtered = pn_enriched[excitatory_mask & ~gabanergic_mask].copy()
    unknown_reintroduced = False
    if pn_filtered.empty and unknown_mask.any():
        pn_filtered = pn_enriched[(excitatory_mask | unknown_mask) & ~gabanergic_mask].copy()
        unknown_reintroduced = True

    print(
        "After nt_type filtering (ACH/GLUT only): "
        f"{len(pn_filtered):,} (excluded {gaba_excluded} GABAergic entries, "
        f"{non_excitatory_excluded} non-excitatory entries)"
    )
    if unknown_reintroduced:
        unknown_retained = int((pn_filtered["nt_type_normalised"].isna()).sum())
        print(
            f"  Included {unknown_retained} ALPNs with missing nt_type annotations due to "
            "the absence of labelled cholinergic/glutamatergic entries."
        )

    pn_filtered["nt_type_normalised"] = pn_filtered["nt_type_normalised"].fillna("UNKNOWN")

    pn_validated = pn_filtered
    if "output_neuropils" in pn_filtered.columns:
        neuropil_series = pn_filtered["output_neuropils"].astype("string")
        calyx_mask = neuropil_series.str.contains("CA", case=False, na=False)
        pn_validated = pn_filtered[calyx_mask].copy()
        removed = len(pn_filtered) - len(pn_validated)
        print(
            f"After calyx neuropil validation: {len(pn_validated):,} "
            f"(removed {removed} lacking CA outputs)"
        )
        if pn_validated.empty:
            print(
                "Calyx validation removed all candidates; reverting to neurotransmitter "
                "filtered set."
            )
            pn_validated = pn_filtered
    else:
        print("output_neuropils column not available; skipping calyx neuropil validation.")

    # Merge optional cell-type annotations for richer metadata
    if cell_types_df is not None and not cell_types_df.empty:
        ct_columns = [
            column
            for column in ("cell_type", "cell_type_aliases", "sub_class")
            if column in cell_types_df.columns
        ]
        if ct_columns:
            cell_type_subset = cell_types_df.loc[
                :, ["root_id", *ct_columns]
            ].drop_duplicates(subset=["root_id"])
            pn_validated = pn_validated.merge(cell_type_subset, on="root_id", how="left")
    else:
        print("Cell-type table missing or empty; glomerulus inference will rely on labels only.")

    processed_lookup = _build_processed_label_lookup(processed_labels_df)
    pn_with_glomeruli = _infer_glomeruli(pn_validated, processed_lookup)

    pn_with_glomeruli = pn_with_glomeruli.drop_duplicates(subset=["root_id"]).reset_index(drop=True)
    pn_with_glomeruli = pn_with_glomeruli.drop(
        columns=[
            column
            for column in (
                "class_normalised",
                "super_class_normalised",
                "flow_normalised",
            )
            if column in pn_with_glomeruli.columns
        ]
    )
    return pn_with_glomeruli


def summarise_alpn_population(pn_df: pd.DataFrame) -> None:
    print("\n=== ALPN Population Summary ===")
    total = len(pn_df)
    print(f"Total ALPNs extracted: {total:,}")
    if total == 0:
        print("No ALPN population statistics available.")
        return

    if "primary_glomerulus" in pn_df.columns:
        glomeruli = pn_df["primary_glomerulus"].dropna()
        unique_glomeruli = sorted(glomeruli.unique())
        print(f"Unique glomeruli: {len(unique_glomeruli):,}")
    else:
        glomeruli = pd.Series(dtype=object)
        print(
            "primary_glomerulus column missing; glomerulus summaries unavailable."
        )

    nt_column = "nt_type_normalised" if "nt_type_normalised" in pn_df.columns else "nt_type"
    neurotransmitter_counts = pn_df[nt_column].value_counts(dropna=False).sort_values(
        ascending=False
    )
    print("Neurotransmitter distribution:")
    for nt, count in neurotransmitter_counts.items():
        label = nt if pd.notna(nt) else "<NA>"
        percentage = (count / total) * 100 if total else 0
        print(f"  {label}: {count} ({percentage:.1f}%)")

    if "primary_glomerulus" in pn_df.columns and not pn_df.empty:
        sample_counts = pn_df.groupby("primary_glomerulus", dropna=True)["root_id"].apply(list)
        sample_counts = sample_counts.sort_values(
            key=lambda series: series.apply(len), ascending=False
        )
        for glomerulus, root_ids in sample_counts.head(10).items():
            preview_ids = [str(value) for value in root_ids[:5]]
            preview = ", ".join(preview_ids)
            if len(root_ids) > 5:
                preview += ", ..."
            print(f"  {glomerulus}: {len(root_ids)} PNs (root_ids: [{preview}])")

        missing = pn_df[pn_df["primary_glomerulus"].isna()]
        if not missing.empty and total:
            print(
                "PNs missing glomerulus assignments: "
                f"{len(missing):,} ({len(missing) / total:.1%})"
            )
        elif total:
            print("No PNs missing glomerulus assignments.")

    if "group" in pn_df.columns:
        hemispheres = pn_df["group"].dropna().astype(str).str.extract(r"(L|R)", expand=False)
        counts = Counter(hemispheres)
        print("Hemisphere distribution (by group column):")
        for hemisphere, count in counts.items():
            label = "Left" if hemisphere == "L" else "Right" if hemisphere == "R" else "Unknown"
            print(f"  {label}: {count}")


def compute_pn_kc_connectivity(
    loader: FlyWireLocalDataLoader,
    pn_ids: Iterable[int],
    kc_ids: Iterable[int],
    *,
    min_synapses: int,
) -> pd.DataFrame:
    pn_ids_set = {int(node) for node in pn_ids}
    kc_ids_set = {int(node) for node in kc_ids}

    print(
        f"Evaluating connectivity for {len(pn_ids_set):,} PNs and "
        f"{len(kc_ids_set):,} KCs"
    )

    print("\nLoading PN→KC connectivity table (filtered to calyx neuropils)...")
    connections = loader.load_connections(
        neuropil_filter=("CA_L", "CA_R"),
        min_synapses=min_synapses,
    )
    print(f"Total calyx connections (all neurons): {len(connections):,}")

    mask = connections["pre_root_id"].isin(pn_ids_set) & connections["post_root_id"].isin(
        kc_ids_set
    )
    pn_kc_connections = connections.loc[mask].reset_index(drop=True)
    print(f"PN→KC connections after filtering: {len(pn_kc_connections):,}")

    observed_pns = pn_kc_connections["pre_root_id"].nunique()
    if observed_pns < len(pn_ids_set):
        print(
            f"  {len(pn_ids_set) - observed_pns} PNs lacked calyx outputs "
            "above the threshold."
        )
    observed_kcs = pn_kc_connections["post_root_id"].nunique()
    if observed_kcs < len(kc_ids_set):
        print(
            f"  {len(kc_ids_set) - observed_kcs} KCs received no calyx input "
            "from the filtered PNs."
        )

    return pn_kc_connections


def summarise_connectivity(
    pn_kc_df: pd.DataFrame,
    pn_df: pd.DataFrame,
    kc_df: pd.DataFrame,
) -> None:
    print("\n=== PN→KC Connectivity Summary ===")
    if pn_kc_df.empty:
        print("No PN→KC connections found with the specified filters.")
        return

    print(f"Total PN→KC connections: {len(pn_kc_df):,}")
    total_synapses = int(pn_kc_df["syn_count"].sum())
    unique_pairs = len(pn_kc_df[["pre_root_id", "post_root_id"]].drop_duplicates())
    avg_synapses = pn_kc_df["syn_count"].mean()

    print(f"Total PN→KC synapses: {total_synapses:,}")
    print(f"Unique PN→KC pairs: {unique_pairs:,}")
    print(f"Average synapses per connection: {avg_synapses:.2f}")

    pn_per_kc = pn_kc_df.groupby("post_root_id")["pre_root_id"].nunique().reindex(
        kc_df["root_id"],
        fill_value=0,
    )
    avg_pn_per_kc = pn_per_kc.mean()
    print(f"Average PNs per KC: {avg_pn_per_kc:.2f}")

    kc_missing_inputs = (pn_per_kc == 0).sum()
    if kc_missing_inputs:
        proportion = kc_missing_inputs / len(kc_df)
        print(f"KCs without PN input in calyx: {kc_missing_inputs} ({proportion:.1%})")

    pn_out_degree = pn_kc_df.groupby("pre_root_id")["post_root_id"].nunique()
    pn_synapse_totals = pn_kc_df.groupby("pre_root_id")["syn_count"].sum()
    sample_stats = (
        pd.DataFrame({"kc_targets": pn_out_degree, "synapses": pn_synapse_totals})
        .join(pn_df.set_index("root_id")["primary_glomerulus"], how="left")
        .sort_values("synapses", ascending=False)
        .head(10)
    )
    if not sample_stats.empty:
        print("Top PN outputs by synapse count:")
        for row in sample_stats.itertuples():
            print(
                f"  PN {row.Index} ({getattr(row, 'primary_glomerulus', 'NA')}): "
                f"{row.synapses} synapses to {row.kc_targets} KCs"
            )


def save_outputs(pn_df: pd.DataFrame, pn_kc_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pn_path = output_dir / "alpn_extracted.csv"
    connectivity_path = output_dir / "pn_to_kc_connectivity.csv"

    pn_export = pn_df.copy()
    pn_export["glomeruli"] = pn_export["glomeruli"].apply(
        lambda values: "|".join(values) if values else ""
    )
    pn_export.to_csv(pn_path, index=False)
    pn_kc_df.to_csv(connectivity_path, index=False)

    print(f"\nSaved ALPN annotations to {pn_path}")
    print(f"Saved PN→KC connectivity to {connectivity_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract ALPNs and PN→KC connectivity statistics")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=paths.DATA_ROOT,
        help="Directory containing FlyWire FAFB v783 CSV exports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cache"),
        help="Directory to save extracted CSVs",
    )
    parser.add_argument(
        "--min-synapses",
        type=int,
        default=5,
        help="Minimum synapse count threshold for PN→KC connections",
    )
    args = parser.parse_args(argv)

    config = ExtractionConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_synapses=args.min_synapses,
    )

    print("Starting ALPN extraction pipeline")
    print(f"Dataset directory: {config.dataset_dir.resolve()}")
    print(f"Output directory: {config.output_dir.resolve()}")

    loader = FlyWireLocalDataLoader(dataset_dir=config.dataset_dir)

    cell_types = _load_dataframe(loader, "cell_types", loader.load_cell_types)
    classification = _load_dataframe(loader, "classification", loader.load_classification)
    neurons = _load_dataframe(loader, "neurons", loader.load_neurotransmitters)
    processed_labels = _load_dataframe(loader, "processed_labels", loader.load_processed_labels)
    names_df = _load_dataframe(loader, "names", loader.load_names)

    pn_df = extract_alpns(classification, neurons, cell_types, processed_labels)
    summarise_alpn_population(pn_df)

    if pn_df.empty:
        print(
            "No ALPNs extracted after filtering; skipping PN→KC connectivity computation."
        )
        save_outputs(pn_df, pd.DataFrame(), config.output_dir)
        return

    kc_df = get_kc_neurons(
        cell_types,
        classification,
        names_df=names_df,
        processed_labels_df=processed_labels,
    )
    print(f"\nIdentified Kenyon cells: {len(kc_df):,}")

    pn_kc_df = compute_pn_kc_connectivity(
        loader,
        pn_df["root_id"].tolist(),
        kc_df["root_id"].tolist(),
        min_synapses=config.min_synapses,
    )
    summarise_connectivity(pn_kc_df, pn_df, kc_df)

    save_outputs(pn_df, pn_kc_df, config.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
