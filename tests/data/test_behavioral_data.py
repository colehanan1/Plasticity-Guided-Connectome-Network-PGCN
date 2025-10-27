from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from pgcn.data.behavioral_data import (
    BehavioralTrial,
    BehavioralTrialSet,
    load_behavioral_dataframe,
    load_behavioral_model_frames,
    load_behavioral_model_tensors,
    load_behavioral_tensor,
    load_behavioral_trial_matrix,
    load_behavioral_trials,
    make_group_kfold,
)


@pytest.fixture()
def behavioral_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "dataset": ["opto_EB", "opto_EB", "opto_benz_1", "opto_benz_1"],
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
    df_sorted = load_behavioral_dataframe(behavioral_csv, validate=False)

    expected = pd.read_csv(behavioral_csv).sort_values(["fly", "trial_label"], kind="mergesort").reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(df_sorted, expected)
    assert list(df_sorted.index) == list(range(len(df_sorted)))


def test_load_behavioral_tensor_uses_sorted_order(behavioral_csv: Path) -> None:
    tensor = load_behavioral_tensor(behavioral_csv, columns=("prediction", "confidence"))
    expected_df = load_behavioral_dataframe(behavioral_csv, validate=False)
    expected_tensor = torch.as_tensor(expected_df[["prediction", "confidence"]].to_numpy())

    assert torch.equal(tensor, expected_tensor)


def test_load_behavioral_trial_matrix_preserves_sorted_keys(behavioral_csv: Path) -> None:
    matrix = load_behavioral_trial_matrix(behavioral_csv)
    expected_df = load_behavioral_dataframe(behavioral_csv, validate=False)

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


def test_load_behavioral_trials_returns_dataclasses(behavioral_csv: Path) -> None:
    trial_set = load_behavioral_trials(behavioral_csv, validate=False)

    assert isinstance(trial_set, BehavioralTrialSet)
    assert len(trial_set.trials) == 4
    assert trial_set.fly_order == ("fly_1", "fly_2")
    assert trial_set.trial_order == ("testing_1", "testing_2")

    first_trial = trial_set.trials[0]
    assert isinstance(first_trial, BehavioralTrial)
    assert first_trial.fly == "fly_1"
    assert first_trial.metadata["confidence"] == pytest.approx(0.9)


def test_model_frames_split_columns(behavioral_csv: Path) -> None:
    features, labels, groups = load_behavioral_model_frames(
        behavioral_csv,
        feature_columns=["dataset", "confidence"],
        label_column="prediction",
        group_column="fly",
        validate=False,
    )

    assert list(features.columns) == ["dataset", "confidence"]
    assert labels.tolist() == [1, 0, 1, 0]
    assert groups.tolist() == ["fly_1", "fly_1", "fly_2", "fly_2"]


def test_model_tensors_are_numeric_and_group_encoded(behavioral_csv: Path) -> None:
    features, labels, groups = load_behavioral_model_tensors(
        behavioral_csv,
        feature_columns=["confidence"],
        label_column="prediction",
        group_column="fly",
        validate=False,
    )

    assert features.shape == (4, 1)
    assert labels.shape == (4,)
    assert groups.dtype == torch.long
    assert set(groups.tolist()) == {0, 1}


def test_make_group_kfold_respects_fly_groups(behavioral_csv: Path) -> None:
    pytest.importorskip("sklearn")
    folds = list(make_group_kfold(behavioral_csv, n_splits=2, validate=False))

    assert len(folds) == 2
    # Each split should keep fly_1 and fly_2 separate between train/test
    for train_idx, test_idx in folds:
        train_flies = set(
            load_behavioral_dataframe(behavioral_csv, validate=False).iloc[train_idx]["fly"].tolist()
        )
        test_flies = set(
            load_behavioral_dataframe(behavioral_csv, validate=False).iloc[test_idx]["fly"].tolist()
        )
        assert train_flies.isdisjoint(test_flies)
