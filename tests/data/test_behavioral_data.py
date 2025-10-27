from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from pgcn.data.behavioral_data import (
    load_behavioral_dataframe,
    load_behavioral_tensor,
    load_behavioral_trial_matrix,
)


@pytest.fixture()
def behavioral_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "fly": ["fly_2", "fly_1", "fly_2", "fly_1"],
            "trial_label": ["testing_2", "testing_1", "testing_1", "testing_2"],
            "prediction": [0, 1, 1, 0],
            "confidence": [0.2, 0.9, 0.8, 0.1],
        }
    )
    csv_path = tmp_path / "behavioral.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_behavioral_dataframe_sorts_and_resets_index(behavioral_csv: Path) -> None:
    df_sorted = load_behavioral_dataframe(behavioral_csv)

    expected = pd.read_csv(behavioral_csv).sort_values(["fly", "trial_label"], kind="mergesort").reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(df_sorted, expected)
    assert list(df_sorted.index) == list(range(len(df_sorted)))


def test_load_behavioral_tensor_uses_sorted_order(behavioral_csv: Path) -> None:
    tensor = load_behavioral_tensor(behavioral_csv, columns=("prediction", "confidence"))
    expected_df = load_behavioral_dataframe(behavioral_csv)
    expected_tensor = torch.as_tensor(expected_df[["prediction", "confidence"]].to_numpy())

    assert torch.equal(tensor, expected_tensor)


def test_load_behavioral_trial_matrix_preserves_sorted_keys(behavioral_csv: Path) -> None:
    matrix = load_behavioral_trial_matrix(behavioral_csv)
    expected_df = load_behavioral_dataframe(behavioral_csv)

    expected_fly_order = expected_df["fly"].drop_duplicates().tolist()
    expected_trial_order = expected_df["trial_label"].drop_duplicates().tolist()

    assert list(matrix.index) == expected_fly_order
    assert list(matrix.columns) == expected_trial_order

    for fly in matrix.index:
        for trial_label in matrix.columns:
            filtered = expected_df[(expected_df["fly"] == fly) & (expected_df["trial_label"] == trial_label)]
            if filtered.empty:
                assert pd.isna(matrix.loc[fly, trial_label])
            else:
                assert matrix.loc[fly, trial_label] == filtered["prediction"].iloc[0]
