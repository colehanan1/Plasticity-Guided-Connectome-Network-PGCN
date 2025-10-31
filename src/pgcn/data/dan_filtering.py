"""
Dopaminergic neuron (DAN) filtering for mushroom body (MB) learning circuits.

This module implements strict MB-only DAN filtering to ensure that only DANs
projecting to mushroom body compartments are included in olfactory learning models.

Biological Basis
----------------
Olfactory associative learning in Drosophila is mediated by dopaminergic modulation
of Kenyon cell (KC) → mushroom body output neuron (MBON) synapses. Only DANs that
project to MB compartments (calyx, peduncle, lobes) can modulate MB plasticity:

1. **PAM neurons** (Protocol-associated memory): Reward-coding DANs targeting MB lobes
   - PAM α/α', PAM β/β', PAM γ compartments
   - Signal positive reinforcement (sugar reward, escape from aversive stimuli)
   - Induce long-term potentiation (LTP) at KC→MBON synapses

2. **PPL1 neurons** (Posterior Paired Lateral 1): Punishment-coding DANs
   - Target specific MB compartments (γ1pedc, α2α'2, α3)
   - Signal aversive outcomes (electric shock, bitter taste)
   - Induce long-term depression (LTD) at KC→MBON synapses

3. **Non-MB DANs** (excluded by this filter):
   - DANs projecting to lateral horn (LH) → olfactory innate behaviors
   - DANs targeting central complex (CX) → visual/motor learning
   - DANs to antennal mechanosensory center (AMMC) → courtship song learning

Filtering Criteria
-------------------
A DAN is considered MB-projecting if:
1. Neurotransmitter type is dopamine (DA)
2. Output neuropils contain "MB" (mushroom body compartment codes)

MB compartment codes (FlyWire/hemibrain conventions):
- MB_CA_L, MB_CA_R (calyx, left/right)
- MB_PED_L, MB_PED_R (peduncle)
- MB_ML_L, MB_ML_R (medial lobe)
- MB_VL_L, MB_VL_R (vertical lobe)

References
----------
- Aso et al. (2014): "The neuronal architecture of the mushroom body provides a logic
  for associative learning." eLife 3:e04577.
- Boto et al. (2019): "Dopaminergic modulation of cAMP drives nonlinear plasticity
  across the Drosophila mushroom body lobes." Current Biology 29(11):1802-1817.
- Cognigni et al. (2018): "Do the right thing: neural network mechanisms of memory
  valence in Drosophila." Current Opinion in Neurobiology 49:91-97.

Example
-------
>>> import pandas as pd
>>> from pgcn.data.dan_filtering import filter_dan_to_mb_only, validate_dan_mb_filter
>>>
>>> # Load all DANs
>>> dan_df = pd.read_csv('data/cache/dan_all.csv')
>>> print(f"Total DANs: {len(dan_df)}")
>>>
>>> # Filter to MB-only
>>> dan_mb = filter_dan_to_mb_only(dan_df)
>>> print(f"MB-projecting DANs: {len(dan_mb)}")
>>>
>>> # Validate
>>> validate_dan_mb_filter(dan_mb)
"""

from __future__ import annotations

from typing import Set

import pandas as pd


def filter_dan_to_mb_only(
    dan_df: pd.DataFrame,
    nt_column: str = "nt_type",
    neuropil_column: str = "output_neuropils",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter dopaminergic neurons to only those projecting to mushroom body (MB).

    This function implements the strict MB-only gate required for olfactory learning
    models. Only DANs with MB in their output neuropils can modulate KC→MBON plasticity.

    Parameters
    ----------
    dan_df : pd.DataFrame
        DataFrame containing DAN metadata. Must include columns for neurotransmitter
        type and output neuropils.
    nt_column : str, optional
        Column name for neurotransmitter type. Default: "nt_type"
    neuropil_column : str, optional
        Column name for output neuropils (pipe-separated or contains "MB").
        Default: "output_neuropils"
    verbose : bool, optional
        If True, print filtering diagnostics. Default: True

    Returns
    -------
    pd.DataFrame
        Subset of dan_df containing only MB-projecting DANs. Includes all original
        columns, with rows filtered to MB-only neurons.

    Raises
    ------
    ValueError
        If required columns are missing from dan_df.

    Notes
    -----
    Filtering logic:
    1. Keep rows where nt_type contains "DA" (case-insensitive)
    2. Keep rows where output_neuropils contains "MB" (case-insensitive)
    3. Return intersection of both criteria

    Edge cases:
    - If no DANs have MB output: returns empty DataFrame (logs warning)
    - If neuropil column is missing: raises ValueError
    - If all DANs project to MB: returns full dan_df

    Example
    -------
    >>> dan_all = pd.read_csv('dan_all.csv')
    >>> dan_mb = filter_dan_to_mb_only(dan_all)
    >>> # Output:
    >>> # DAN filtering result:
    >>> #   Total DA neurons: 585
    >>> #   DA neurons with MB output: 368
    >>> #   Percentage kept: 62.9%
    """
    # Validate required columns
    if nt_column not in dan_df.columns:
        raise ValueError(
            f"Missing '{nt_column}' column in DAN dataframe. "
            f"Available columns: {list(dan_df.columns)}"
        )

    if neuropil_column not in dan_df.columns:
        raise ValueError(
            f"Missing '{neuropil_column}' column in DAN dataframe. "
            f"Available columns: {list(dan_df.columns)}"
        )

    # Filter by neurotransmitter type (dopamine)
    da_mask = dan_df[nt_column].astype(str).str.contains("DA", case=False, na=False)
    dan_candidates = dan_df[da_mask].copy()

    if len(dan_candidates) == 0:
        if verbose:
            print(f"WARNING: No DA neurons found in input ({len(dan_df)} rows total)")
        return pd.DataFrame(columns=dan_df.columns)

    # Filter by output neuropils containing MB
    mb_mask = dan_candidates[neuropil_column].astype(str).str.contains(
        "MB", case=False, na=False
    )
    dan_mb = dan_candidates[mb_mask].copy()

    # Diagnostic output
    if verbose:
        n_da_total = len(dan_candidates)
        n_da_mb = len(dan_mb)
        pct_mb = 100 * n_da_mb / n_da_total if n_da_total > 0 else 0

        print(f"DAN filtering result:")
        print(f"  Total DA neurons: {n_da_total}")
        print(f"  DA neurons with MB output: {n_da_mb}")
        print(f"  Percentage kept: {pct_mb:.1f}%")

        if n_da_mb == 0:
            print(
                f"  WARNING: No MB-projecting DANs found! Check {neuropil_column} column."
            )

    return dan_mb


def validate_dan_mb_filter(
    dan_mb_df: pd.DataFrame,
    neuropil_column: str = "output_neuropils",
) -> bool:
    """
    Validate that all DANs in filtered set have MB in output_neuropils.

    This function performs post-filtering validation to ensure the MB-only constraint
    is satisfied. Useful for catching bugs in filtering logic or data issues.

    Parameters
    ----------
    dan_mb_df : pd.DataFrame
        Filtered DataFrame that should contain only MB-projecting DANs.
    neuropil_column : str, optional
        Column name for output neuropils. Default: "output_neuropils"

    Returns
    -------
    bool
        True if all rows pass validation (all have "MB" in neuropils).

    Raises
    ------
    AssertionError
        If any row does not contain "MB" in output_neuropils.

    Example
    -------
    >>> dan_mb = filter_dan_to_mb_only(dan_all)
    >>> validate_dan_mb_filter(dan_mb)
    >>> # Output: ✓ Validated: all 368 DANs have MB output
    """
    if neuropil_column not in dan_mb_df.columns:
        raise ValueError(
            f"Missing '{neuropil_column}' column in DAN dataframe. "
            f"Available columns: {list(dan_mb_df.columns)}"
        )

    # Check each row
    for idx, row in dan_mb_df.iterrows():
        neuropils = str(row[neuropil_column]).upper()
        if "MB" not in neuropils:
            # Get identifier for error message
            if "root_id" in row:
                neuron_id = f"root_id={row['root_id']}"
            elif "cell_id" in row:
                neuron_id = f"cell_id={row['cell_id']}"
            else:
                neuron_id = f"row {idx}"

            raise AssertionError(
                f"DAN {neuron_id} does not contain 'MB' in {neuropil_column}: "
                f"{row[neuropil_column]}"
            )

    print(f"✓ Validated: all {len(dan_mb_df)} DANs have MB output")
    return True


def get_dan_compartment_mapping(dan_mb_df: pd.DataFrame) -> dict:
    """
    Extract MB compartment → DAN mapping from filtered DAN DataFrame.

    Useful for downstream analyses requiring knowledge of which DANs innervate
    which MB compartments (e.g., compartment-specific plasticity rules).

    Parameters
    ----------
    dan_mb_df : pd.DataFrame
        Filtered DataFrame of MB-projecting DANs (from filter_dan_to_mb_only).

    Returns
    -------
    dict
        Mapping: {compartment_code: list of DAN root_ids}
        Example: {"MB_CA_L": [720575940604479590, ...], ...}

    Example
    -------
    >>> dan_mb = filter_dan_to_mb_only(dan_all)
    >>> compartment_map = get_dan_compartment_mapping(dan_mb)
    >>> print(f"DANs innervating MB_CA_L: {len(compartment_map['MB_CA_L'])}")
    """
    compartment_mapping = {}

    for _, row in dan_mb_df.iterrows():
        neuropils = str(row.get("output_neuropils", ""))
        root_id = row.get("root_id", row.get("cell_id", None))

        # Split pipe-separated neuropils
        for neuropil in neuropils.split("|"):
            neuropil = neuropil.strip()
            if "MB" in neuropil.upper():
                if neuropil not in compartment_mapping:
                    compartment_mapping[neuropil] = []
                if root_id is not None:
                    compartment_mapping[neuropil].append(root_id)

    return compartment_mapping
