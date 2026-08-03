from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

from cofitok.generation_class_fidelity import (
    CLASS_FIDELITY_QUALIFICATION_ROLE,
    CLASS_FIDELITY_QUALIFICATION_SCHEMA_VERSION,
    validate_class_fidelity_qualification,
    validate_class_fidelity_report,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound matched class-conditional fidelity qualification "
            "for CoFiTok and dense identity samples."
        )
    )
    parser.add_argument("--cofitok-report", required=True)
    parser.add_argument("--dense-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("scaling", "full"), required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--min-top1", type=float, required=True)
    parser.add_argument("--min-top5", type=float, required=True)
    parser.add_argument("--min-predicted-class-fraction", type=float, required=True)
    parser.add_argument("--min-normalized-predicted-entropy", type=float, required=True)
    parser.add_argument("--max-top1-regression", type=float, required=True)
    parser.add_argument("--max-top5-regression", type=float, required=True)
    parser.add_argument(
        "--allow-hold",
        action="store_true",
        help="Write a source-valid hold report and exit successfully.",
    )
    return parser.parse_args()


def _finite_fraction(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite fraction")
    return value


def _paired_sampling_contract(
    cofitok: dict[str, Any],
    dense: dict[str, Any],
) -> dict[str, Any]:
    cofitok_sample = cofitok["sample_provenance"]
    dense_sample = dense["sample_provenance"]
    if cofitok_sample.get("git") != dense_sample.get("git"):
        raise ValueError("class-fidelity sampling Git identities are not matched")
    if cofitok.get("git") != dense.get("git"):
        raise ValueError("class-fidelity evaluator Git identities are not matched")
    cofitok_sampling = copy.deepcopy(cofitok_sample["sampling"])
    dense_sampling = copy.deepcopy(dense_sample["sampling"])
    cofitok_budgets = cofitok_sampling.pop("prefix_budgets", None)
    dense_budgets = dense_sampling.pop("prefix_budgets", None)
    if cofitok_sampling != dense_sampling:
        raise ValueError("class-fidelity sampling protocols are not matched")
    if cofitok_budgets != [8] or dense_budgets != [1]:
        raise ValueError("class-fidelity prefix budgets do not identify the matched pair")
    if (
        cofitok_sample.get("selected_prefix_budget") != 8
        or dense_sample.get("selected_prefix_budget") != 1
        or cofitok_sample.get("weights") != "ema"
        or dense_sample.get("weights") != "ema"
        or cofitok_sample.get("sampling", {}).get("class_schedule")
        != "balanced_modulo"
        or dense_sample.get("sampling", {}).get("class_schedule")
        != "balanced_modulo"
    ):
        raise ValueError("class-fidelity matched sample identities differ")
    if int(cofitok_sampling["num_samples"]) != int(dense_sampling["num_samples"]):
        raise ValueError("class-fidelity sample counts differ")
    return {
        "sampling": cofitok_sampling,
        "cofitok_prefix_budget": 8,
        "dense_prefix_budget": 1,
        "weights": "ema",
        "sampling_git": copy.deepcopy(cofitok_sample["git"]),
        "evaluator_git": copy.deepcopy(cofitok["git"]),
        "evaluator_runtime_environment_sha256": cofitok[
            "runtime_environment_sha256"
        ],
        "sample_count_per_method": int(cofitok_sampling["num_samples"]),
        "cofitok_checkpoint_sha256": cofitok_sample["checkpoint_sha256"],
        "dense_checkpoint_sha256": dense_sample["checkpoint_sha256"],
        "cofitok_sample_set_sha256": cofitok_sample["sample_set_sha256"],
        "dense_sample_set_sha256": dense_sample["sample_set_sha256"],
    }


def _check(
    name: str,
    *,
    observed: float,
    threshold: float,
    passed: bool,
    comparison: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "hold",
        "observed": observed,
        "threshold": threshold,
        "comparison": comparison,
    }


def build_qualification(
    *,
    cofitok: dict[str, Any],
    dense: dict[str, Any],
    cofitok_path: str | Path,
    dense_path: str | Path,
    stage: str,
    expected_revision: str,
    expected_branch: str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    validate_class_fidelity_report(cofitok)
    validate_class_fidelity_report(dense)
    evaluator_git = git_provenance(PROJECT_ROOT)
    if evaluator_git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("class-fidelity qualification Git identity differs")
    for name, report in (("cofitok", cofitok), ("dense", dense)):
        if report.get("git") != evaluator_git:
            raise ValueError(f"{name} class-fidelity evaluator Git identity differs")
    if cofitok["classifier"] != dense["classifier"]:
        raise ValueError("class-fidelity classifier identities differ")
    if (
        cofitok.get("runtime_environment_sha256")
        != dense.get("runtime_environment_sha256")
    ):
        raise ValueError("class-fidelity evaluator runtime identities differ")
    sampling_contract = _paired_sampling_contract(cofitok, dense)

    cm = cofitok["metrics"]
    dm = dense["metrics"]
    checks = []
    for method, metrics in (("cofitok", cm), ("dense_identity", dm)):
        for metric_name, threshold_name in (
            ("top1_accuracy", "min_top1"),
            ("top5_accuracy", "min_top5"),
            ("predicted_class_fraction", "min_predicted_class_fraction"),
            (
                "normalized_predicted_class_entropy",
                "min_normalized_predicted_entropy",
            ),
        ):
            observed = float(metrics[metric_name])
            threshold = thresholds[threshold_name]
            checks.append(
                _check(
                    f"{method}_{metric_name}",
                    observed=observed,
                    threshold=threshold,
                    passed=observed >= threshold,
                    comparison=">=",
                )
            )
    for metric_name, threshold_name in (
        ("top1_accuracy", "max_top1_regression"),
        ("top5_accuracy", "max_top5_regression"),
    ):
        regression = float(dm[metric_name]) - float(cm[metric_name])
        threshold = thresholds[threshold_name]
        checks.append(
            _check(
                f"cofitok_{metric_name}_regression_vs_dense",
                observed=regression,
                threshold=threshold,
                passed=regression <= threshold,
                comparison="<=",
            )
        )
    status = "pass" if all(row["status"] == "pass" for row in checks) else "hold"
    result = {
        "schema_version": CLASS_FIDELITY_QUALIFICATION_SCHEMA_VERSION,
        "role": CLASS_FIDELITY_QUALIFICATION_ROLE,
        "status": status,
        "stage": stage,
        "git": evaluator_git,
        "classifier": cofitok["classifier"],
        "sampling_contract": sampling_contract,
        "thresholds": thresholds,
        "checks": checks,
        "metrics": {
            "cofitok": cm,
            "dense_identity": dm,
            "cofitok_minus_dense": {
                key: float(cm[key]) - float(dm[key])
                for key in (
                    "top1_accuracy",
                    "top5_accuracy",
                    "mean_target_probability",
                    "predicted_class_fraction",
                    "normalized_predicted_class_entropy",
                )
            },
        },
        "sources": {
            "cofitok": file_identity(cofitok_path),
            "dense_identity": file_identity(dense_path),
        },
        "claim_boundary": {
            "class_conditional_quality_evaluated": True,
            "unconditional_distribution_quality_evaluated": False,
            "standalone_generation_quality_claim_allowed": False,
            "full_training_launch_allowed": False,
            "release_authorization_allowed": False,
            "interpretation": (
                "Class fidelity complements but cannot replace FID, IS, precision, "
                "recall, rollout stability, or visual review."
            ),
        },
    }
    validate_class_fidelity_qualification(
        result,
        expected_stage=stage,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        require_pass=False,
    )
    return result


def main() -> None:
    args = parse_args()
    thresholds = {
        "min_top1": _finite_fraction(args.min_top1, name="min-top1"),
        "min_top5": _finite_fraction(args.min_top5, name="min-top5"),
        "min_predicted_class_fraction": _finite_fraction(
            args.min_predicted_class_fraction,
            name="min-predicted-class-fraction",
        ),
        "min_normalized_predicted_entropy": _finite_fraction(
            args.min_normalized_predicted_entropy,
            name="min-normalized-predicted-entropy",
        ),
        "max_top1_regression": _finite_fraction(
            args.max_top1_regression,
            name="max-top1-regression",
        ),
        "max_top5_regression": _finite_fraction(
            args.max_top5_regression,
            name="max-top5-regression",
        ),
    }
    cofitok = read_json_object(args.cofitok_report, name="CoFiTok class fidelity")
    dense = read_json_object(args.dense_report, name="dense class fidelity")
    report = build_qualification(
        cofitok=cofitok,
        dense=dense,
        cofitok_path=args.cofitok_report,
        dense_path=args.dense_report,
        stage=args.stage,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        thresholds=thresholds,
    )
    write_json_report(args.output, report)
    print(report["status"])
    if report["status"] != "pass" and not args.allow_hold:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
