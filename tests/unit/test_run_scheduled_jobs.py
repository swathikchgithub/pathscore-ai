import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "monitoring"))
import run_scheduled_jobs as jobs  # noqa: E402


def test_use_cases_lists_yaml_basenames_sorted(tmp_path):
    (tmp_path / "b.yaml").write_text("use_case: b\n")
    (tmp_path / "a.yaml").write_text("use_case: a\n")
    (tmp_path / "notes.txt").write_text("ignored\n")

    assert jobs._use_cases(str(tmp_path)) == ["a", "b"]


def test_retrain_all_builds_and_trains_every_use_case(tmp_path, monkeypatch):
    (tmp_path / "uc1.yaml").write_text("use_case: uc1\n")
    (tmp_path / "uc2.yaml").write_text("use_case: uc2\n")

    calls = []
    monkeypatch.setattr(jobs.build_features, "build_features", lambda uc: calls.append(("build", uc)))
    monkeypatch.setattr(
        jobs.train_module,
        "train",
        lambda config_path, data_path, out_dir: calls.append(("train", config_path, data_path, out_dir)),
    )

    jobs.retrain_all(config_dir=str(tmp_path), models_dir="models")

    assert calls == [
        ("build", "uc1"),
        ("train", f"{tmp_path}/uc1.yaml", "data/features/uc1.csv", "models/uc1"),
        ("build", "uc2"),
        ("train", f"{tmp_path}/uc2.yaml", "data/features/uc2.csv", "models/uc2"),
    ]


def test_retrain_all_continues_past_a_failure_then_raises(tmp_path, monkeypatch):
    (tmp_path / "bad.yaml").write_text("use_case: bad\n")
    (tmp_path / "good.yaml").write_text("use_case: good\n")

    def fake_build(use_case):
        if use_case == "bad":
            raise RuntimeError("boom")

    trained = []
    monkeypatch.setattr(jobs.build_features, "build_features", fake_build)
    monkeypatch.setattr(jobs.train_module, "train", lambda *a, **k: trained.append(a))

    with pytest.raises(SystemExit, match="bad"):
        jobs.retrain_all(config_dir=str(tmp_path), models_dir="models")

    # "bad" failed during build (before train would've run); "good" still
    # ran despite "bad" failing first alphabetically.
    assert trained == [(f"{tmp_path}/good.yaml", "data/features/good.csv", "models/good")]


def test_drift_check_all_builds_and_checks_every_use_case(tmp_path, monkeypatch):
    (tmp_path / "uc1.yaml").write_text("use_case: uc1\n")

    calls = []
    monkeypatch.setattr(jobs.build_features, "build_features", lambda uc: calls.append(("build", uc)))
    monkeypatch.setattr(
        jobs.drift_check,
        "check_drift",
        lambda config_path, data_path, models_dir: calls.append(("check", config_path, data_path, models_dir)),
    )

    jobs.drift_check_all(config_dir=str(tmp_path), models_dir="models")

    assert calls == [
        ("build", "uc1"),
        ("check", f"{tmp_path}/uc1.yaml", "data/features/uc1.csv", "models"),
    ]


def test_drift_check_all_continues_past_a_failure_then_raises(tmp_path, monkeypatch):
    (tmp_path / "bad.yaml").write_text("use_case: bad\n")
    (tmp_path / "good.yaml").write_text("use_case: good\n")

    checked = []
    monkeypatch.setattr(jobs.build_features, "build_features", lambda uc: None)

    def fake_check(config_path, data_path, models_dir):
        if "bad" in config_path:
            raise RuntimeError("boom")
        checked.append(config_path)

    monkeypatch.setattr(jobs.drift_check, "check_drift", fake_check)

    with pytest.raises(SystemExit, match="bad"):
        jobs.drift_check_all(config_dir=str(tmp_path), models_dir="models")

    assert checked == [f"{tmp_path}/good.yaml"]
