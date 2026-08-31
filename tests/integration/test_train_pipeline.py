import yaml
import pytest

import train as train_module


def _write_config(tmp_path, config, name="config.yaml"):
    path = tmp_path / name
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


def _write_data(tmp_path, df, name="data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_train_raises_before_touching_data_when_feature_is_denylisted(
    tmp_path, binary_config, binary_dataframe
):
    leaky_config = dict(binary_config)
    leaky_config["feature_columns"] = binary_config["feature_columns"] + ["contract_signed_date"]

    config_path = _write_config(tmp_path, leaky_config)
    data_path = _write_data(tmp_path, binary_dataframe)
    out_dir = tmp_path / "model_out"

    # The leaky column isn't even in the dataframe, so if leakage_checks did
    # NOT run first, this would fail later with a KeyError (or worse, train
    # silently on whatever data is there) instead of the leakage ValueError.
    # Asserting the specific message pins the failure to the leakage guard.
    with pytest.raises(ValueError, match="Potential label leakage"):
        train_module.train(config_path, data_path, str(out_dir))

    assert not out_dir.exists()


def test_train_binary_pipeline_end_to_end(tmp_path, binary_config, binary_dataframe):
    config_path = _write_config(tmp_path, binary_config)
    data_path = _write_data(tmp_path, binary_dataframe)
    out_dir = tmp_path / "model_out"

    train_module.train(config_path, data_path, str(out_dir))

    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "calibrator.joblib").exists()
    metrics = yaml.safe_load((out_dir / "metrics.yaml").read_text())
    assert metrics["n_classes"] == 2
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["precision_at_10pct"] <= 1.0


def test_train_multiclass_pipeline_end_to_end(tmp_path, multiclass_config, multiclass_dataframe):
    config_path = _write_config(tmp_path, multiclass_config)
    data_path = _write_data(tmp_path, multiclass_dataframe)
    out_dir = tmp_path / "model_out"

    train_module.train(config_path, data_path, str(out_dir))

    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "calibrator.joblib").exists()
    metrics = yaml.safe_load((out_dir / "metrics.yaml").read_text())
    assert metrics["n_classes"] == 4
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["precision_at_10pct"] <= 1.0
