"""
Scheduled-job entrypoint -- the actual place each config's `retrain.cadence`
gets enforced. Until now nothing called train.py or drift_check.py on a
schedule; an external scheduler (GitHub Actions -- see
.github/workflows/retrain.yml -- cron, etc.) invokes this script. It doesn't
track time internally: "when" is the scheduler's job, "what to run for every
registered use case" is this script's.

Two jobs, same "one script, N configs" principle as train.py:
  retrain      rebuild features + retrain every use case, overwriting its
               model and baseline metrics. Meant to run on the cadence
               every config currently declares (weekly).
  drift-check  rebuild features + compare against the existing baseline,
               no retrain. Cheaper, meant to run more often than retrain as
               an early warning -- see src/monitoring/drift_check.py.

Usage:
  python src/monitoring/run_scheduled_jobs.py retrain
  python src/monitoring/run_scheduled_jobs.py drift-check
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scoring"))
sys.path.insert(0, os.path.dirname(__file__))
import build_features  # noqa: E402
import train as train_module  # noqa: E402
import drift_check  # noqa: E402

CONFIG_DIR_DEFAULT = "src/config"
MODELS_DIR_DEFAULT = "models"


def _use_cases(config_dir: str) -> list:
    return sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(f"{config_dir}/*.yaml")
    )


def retrain_all(config_dir: str = CONFIG_DIR_DEFAULT, models_dir: str = MODELS_DIR_DEFAULT):
    failures = []
    for use_case in _use_cases(config_dir):
        print(f"=== retrain: {use_case} ===")
        try:
            build_features.build_features(use_case)
            train_module.train(
                f"{config_dir}/{use_case}.yaml",
                f"data/features/{use_case}.csv",
                f"{models_dir}/{use_case}",
            )
        except Exception as exc:
            print(f"[{use_case}] retrain FAILED: {exc}")
            failures.append(use_case)
    if failures:
        raise SystemExit(f"Retrain failed for: {', '.join(failures)}")


def drift_check_all(config_dir: str = CONFIG_DIR_DEFAULT, models_dir: str = MODELS_DIR_DEFAULT):
    failures = []
    for use_case in _use_cases(config_dir):
        print(f"=== drift-check: {use_case} ===")
        try:
            build_features.build_features(use_case)
            drift_check.check_drift(
                f"{config_dir}/{use_case}.yaml",
                f"data/features/{use_case}.csv",
                models_dir=models_dir,
            )
        except Exception as exc:
            print(f"[{use_case}] drift-check FAILED: {exc}")
            failures.append(use_case)
    if failures:
        raise SystemExit(f"Drift check failed for: {', '.join(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["retrain", "drift-check"])
    parser.add_argument("--config-dir", default=CONFIG_DIR_DEFAULT)
    parser.add_argument("--models-dir", default=MODELS_DIR_DEFAULT)
    args = parser.parse_args()

    if args.job == "retrain":
        retrain_all(args.config_dir, args.models_dir)
    else:
        drift_check_all(args.config_dir, args.models_dir)
