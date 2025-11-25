#!/usr/bin/env python
"""Export DoOR responses for a specific odor × ORN panel to CSV.

Outputs `data/analysis/selected_orn_responses.csv` with:
- Rows: requested odors (+ a final row with receptor means across odors)
- Columns: requested receptors (missing receptors kept as NaN) and a per-odor mean
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DOOR_MATRIX_CANDIDATES = [
    Path("data/cache/door_response_matrix.csv"),
    Path("data/door_cache/door_response_matrix.csv"),
    Path("door_cache/door_response_matrix.csv"),
]

# User-specified panel
ODORS = [
    "1-hexanol",
    "ethyl butyrate",
    "benzaldehyde",
    "citral",
    "linalool",
    "3-octanol",
    "acetic acid",
]

RECEPTORS = ["Or42a", "Or85d", "Or85d2", "Or71a", "Or33c", "Or59c", "Or46a"]

OUTPUT_PATH = Path("data/analysis/selected_orn_responses.csv")


def find_door_matrix(user_path: Optional[Path] = None) -> pd.DataFrame:
    candidates: List[Path] = []
    if user_path is not None:
        candidates.append(user_path)
    candidates.extend(DOOR_MATRIX_CANDIDATES)
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return pd.read_csv(path, index_col=0)
    raise FileNotFoundError("Could not find DoOR response matrix in expected locations.")


def map_rows(df: pd.DataFrame, odors: List[str]) -> Dict[str, Optional[str]]:
    rows_lower = {r.lower(): r for r in df.index}

    def try_match(q: str) -> Optional[str]:
        ql = q.lower()
        if ql in rows_lower:
            return rows_lower[ql]
        # Simple normalizations
        replacements = [
            ("butyrate", "butanoate"),
            ("1-", "(+/-)-1-"),
            (" ", "-"),
        ]
        for old, new in replacements:
            cand = ql.replace(old, new)
            if cand in rows_lower:
                return rows_lower[cand]
        matches = [r for r in df.index if ql in r.lower()]
        return matches[0] if matches else None

    return {q: try_match(q) for q in odors}


def main() -> None:
    door = find_door_matrix()
    row_map = map_rows(door, ODORS)
    col_lookup = {c.lower(): c for c in door.columns}

    records = []
    for odor, row_name in row_map.items():
        if row_name is None:
            continue
        row = door.loc[row_name]
        for receptor in RECEPTORS:
            col = col_lookup.get(receptor.lower())
            val = row[col] if col is not None else float("nan")
            records.append(
                {
                    "odor": odor,
                    "DoOR_row": row_name,
                    "receptor": receptor,
                    "response": val,
                }
            )

    df = pd.DataFrame(records)
    pivot = df.pivot(index="odor", columns="receptor", values="response").reindex(ODORS)
    pivot["mean_across_receptors"] = pivot.mean(axis=1, skipna=True)

    # Append receptor means across odors as final row
    receptor_means = pivot.drop(columns=["mean_across_receptors"]).mean(axis=0, skipna=True)
    receptor_means.name = "mean_across_odors"
    pivot = pd.concat([pivot, receptor_means.to_frame().T], axis=0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(OUTPUT_PATH)
    print(f"Saved odor × receptor responses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
