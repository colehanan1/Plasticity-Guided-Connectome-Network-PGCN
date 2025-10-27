import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pgcn.data import behavioral_data as bd


@pytest.fixture()
def behavioral_csv(tmp_path):
    path = tmp_path / "model_predictions.csv"
    df = _build_mock_behavioral_dataframe()
    df.to_csv(path, index=False)
    return path, df


def _build_mock_behavioral_dataframe() -> pd.DataFrame:
    datasets = ["opto_EB", "opto_hex", "opto_benz_1", "hex_control"]
    trial_labels = [f"testing_{i}" for i in range(1, 11)]
    exposures_per_fly = [14] * 5 + [13] * 10 + [12] * 20

    records = []
    for fly_idx, exposures in enumerate(exposures_per_fly):
        dataset = datasets[fly_idx % len(datasets)]
        fly_name = f"fly_{fly_idx:02d}"
        fly_number = fly_idx + 1
        base_predictions = [int((fly_idx + idx) % 2 == 0) for idx in range(len(trial_labels))]
        for label_idx, (label, prediction) in enumerate(zip(trial_labels, base_predictions)):
            records.append(
                {
                    "dataset": dataset,
                    "fly": fly_name,
                    "fly_number": fly_number,
                    "trial_label": label,
                    "prediction": prediction,
                    "probability": _mock_probability(fly_idx, label_idx, prediction),
                }
            )
        extra_trials = exposures - len(trial_labels)
        for extra_idx in range(extra_trials):
            label = trial_labels[extra_idx % len(trial_labels)]
            prediction = base_predictions[extra_idx % len(trial_labels)]
            records.append(
                {
                    "dataset": dataset,
                    "fly": fly_name,
                    "fly_number": fly_number,
                    "trial_label": label,
                    "prediction": prediction,
                    "probability": _mock_probability(
                        fly_idx,
                        len(trial_labels) + extra_idx,
                        prediction,
                    ),
                }
            )

    df = pd.DataFrame.from_records(records)
    assert len(df) == bd.EXPECTED_TRIAL_COUNT
    assert df["fly"].nunique() == bd.EXPECTED_FLY_COUNT
    return df


def _mock_probability(fly_idx: int, trial_idx: int, prediction: int) -> float:
    base = 0.25 + 0.5 * prediction
    modulation = 0.02 * ((fly_idx + trial_idx) % 5)
    return float(min(base + modulation, 0.99))


def test_load_behavioral_dataframe_validates(behavioral_csv):
    csv_path, _ = behavioral_csv
    df = bd.load_behavioral_dataframe(path=csv_path)

    assert len(df) == bd.EXPECTED_TRIAL_COUNT
    assert df["fly"].nunique() == bd.EXPECTED_FLY_COUNT
    assert set(df["trial_label"].unique()) <= set(bd.EXPECTED_TRIAL_LABELS)
    assert df["fly_number"].dtype == "int64"
    assert df["probability"].between(0, 1).all()
    sorted_index = df.sort_values(["fly", "trial_label"]).index.to_list()
    assert list(df.index) == sorted_index


def test_load_behavioral_dataframe_rejects_invalid_length(tmp_path, behavioral_csv):
    csv_path, _ = behavioral_csv
    df = pd.read_csv(csv_path)
    truncated_path = tmp_path / "truncated.csv"
    df.iloc[:-1].to_csv(truncated_path, index=False)

    with pytest.raises(ValueError):
        bd.load_behavioral_dataframe(path=truncated_path)


def test_load_behavioral_trials_returns_dataclasses(behavioral_csv):
    csv_path, df_raw = behavioral_csv
    trials = bd.load_behavioral_trials(path=csv_path)
    assert len(trials) == bd.EXPECTED_TRIAL_COUNT
    first_trial = trials[0]
    assert isinstance(first_trial, bd.BehavioralTrial)
    expected_first_dataset = df_raw.sort_values(["fly", "trial_label"]).iloc[0]["dataset"]
    assert first_trial.dataset == expected_first_dataset
    assert first_trial.fly_number is not None
    assert 0 <= first_trial.probability <= 1


def test_get_model_ready_dataframe_and_tensors(behavioral_csv):
    csv_path, _ = behavioral_csv
    df = bd.load_behavioral_dataframe(path=csv_path)
    features_df, labels, groups = bd.get_model_ready_dataframe(df)

    assert features_df.shape[0] == bd.EXPECTED_TRIAL_COUNT
    assert labels.shape[0] == bd.EXPECTED_TRIAL_COUNT
    assert groups.shape[0] == bd.EXPECTED_TRIAL_COUNT
    assert "fly_number" in features_df.columns
    assert "probability" in features_df.columns

    feature_tensor, label_tensor, group_tensor = bd.get_model_ready_tensors(df)
    assert feature_tensor.shape[0] == bd.EXPECTED_TRIAL_COUNT
    assert label_tensor.dtype == torch.float32
    assert group_tensor.dtype == torch.long
    assert torch.unique(group_tensor).numel() == bd.EXPECTED_FLY_COUNT


def test_make_group_kfold_generates_stratified_splits(behavioral_csv):
    csv_path, _ = behavioral_csv
    df = bd.load_behavioral_dataframe(path=csv_path)
    splits = list(bd.make_group_kfold(n_splits=5, groups=df["fly"]))
    assert len(splits) == 5
    all_indices = []
    for train_idx, test_idx in splits:
        train_groups = set(df.iloc[train_idx]["fly"])
        test_groups = set(df.iloc[test_idx]["fly"])
        assert train_groups.isdisjoint(test_groups)
        all_indices.extend(test_idx.tolist())
    assert sorted(all_indices) == sorted(set(all_indices))


def test_environment_variable_overrides_path(monkeypatch, behavioral_csv):
    csv_path, _ = behavioral_csv
    monkeypatch.setenv(bd.BEHAVIORAL_DATA_ENV_VAR, str(csv_path))

    df = bd.load_behavioral_dataframe()

    assert len(df) == bd.EXPECTED_TRIAL_COUNT
    assert df["fly"].nunique() == bd.EXPECTED_FLY_COUNT
