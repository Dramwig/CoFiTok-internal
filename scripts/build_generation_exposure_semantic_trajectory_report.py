from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from cofitok.generation.exposure_semantic_trajectory import (
    AUTHORIZATION_ROLE,
    CHECKPOINT_STEPS,
    CLAIM_BOUNDARY,
    DESCRIPTIVE_TIMESTEPS,
    ELIGIBLE_TIMESTEPS,
    EVALUATION_REQUEST,
    EXECUTION_BOUNDARY,
    METHOD_RUN_DIRS,
    OUTPUT_ROOT,
    REPORT_ROLE,
    SCHEMA_VERSION,
    SCOPE,
    SIGNIFICANCE_LEVEL,
    STAGE,
    validate_preparation,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from scripts.evaluate_generation_conditioning_sensitivity import (
    MANIFEST_FILENAME,
    REPORT_ROLE as SENSITIVITY_REPORT_ROLE,
    REPORT_SCHEMA_VERSION as SENSITIVITY_SCHEMA_VERSION,
    validate_completed_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENSITIVITY_CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "generates_new_samples": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "replaces_formal_quality_gate": False,
}
COMPARISONS = ("versus_wrong", "versus_null")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical source-bound semantic exposure trajectory over "
            "the matched existing 5K checkpoints."
        )
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    for method in METHOD_RUN_DIRS:
        for step in CHECKPOINT_STEPS:
            option = method.replace("_", "-")
            parser.add_argument(
                f"--{option}-step-{step}-report",
                type=Path,
                required=True,
            )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _one_sided_binomial_pvalue(successes: int, trials: int) -> float:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial count")
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (
        2**trials
    )


def summarize_positive(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values]
    if not finite or any(not math.isfinite(value) for value in finite):
        raise ValueError("signed summary requires finite values")
    positive = sum(value > 0.0 for value in finite)
    return {
        "mean": _mean(finite),
        "median": float(statistics.median(finite)),
        "positive_count": positive,
        "positive_fraction": positive / len(finite),
        "one_sided_sign_test_pvalue": _one_sided_binomial_pvalue(
            positive,
            len(finite),
        ),
    }


def _row_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(row["sample_index"]),
        int(row["timestep"]),
        int(row["correct_label"]),
        int(row["wrong_label"]),
        int(row["noise_seed"]),
    )


def _sample_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["sample_index"]),
        int(row["correct_label"]),
        int(row["wrong_label"]),
        int(row["noise_seed"]),
    )


def _condition_mse(row: Mapping[str, Any], condition: str) -> float:
    try:
        value = float(row["conditions"][condition]["epsilon_mse_to_noise"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"sensitivity row lacks {condition} epsilon MSE") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"sensitivity {condition} epsilon MSE is invalid")
    return value


def _load_sensitivity(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = reject_symlink_chain(path, name=f"{label} report").resolve()
    report = read_json_object(report_path, name=f"{label} report")
    if (
        report.get("schema_version") != SENSITIVITY_SCHEMA_VERSION
        or report.get("role") != SENSITIVITY_REPORT_ROLE
        or report.get("status") != "completed"
    ):
        raise ValueError(f"{label} is not a completed sensitivity report")
    manifest_path = reject_symlink_chain(
        report_path.parent / MANIFEST_FILENAME,
        name=f"{label} manifest",
    ).resolve()
    manifest = read_json_object(manifest_path, name=f"{label} manifest")
    manifest_identity = file_identity(manifest_path)
    validate_completed_report(
        report,
        manifest=manifest,
        manifest_identity=manifest_identity,
    )
    return report, {
        "report": file_identity(report_path),
        "manifest": manifest_identity,
    }


def validate_sensitivity_reports(
    *,
    reports: Mapping[str, Mapping[int, Mapping[str, Any]]],
    preparation: Mapping[str, Any],
    evaluator_git: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_dataset: Mapping[str, Any] | None = None
    canonical_keys: list[tuple[int, int, int, int, int]] | None = None
    for method in METHOD_RUN_DIRS:
        if sorted(reports.get(method, {})) != CHECKPOINT_STEPS:
            raise ValueError(f"{method} sensitivity checkpoint set differs")
        prepared = preparation["methods"][method]
        prepared_by_step = {row["step"]: row for row in prepared["checkpoints"]}
        for step in CHECKPOINT_STEPS:
            report = reports[method][step]
            expected_checkpoint = prepared_by_step[step]["checkpoint"]
            runtime = report.get("runtime")
            checkpoint = report.get("checkpoint")
            model = report.get("model")
            rows = report.get("sample_rows")
            if (
                report.get("git") != evaluator_git
                or report.get("request") != EVALUATION_REQUEST
                or report.get("weights") != "ema"
                or report.get("claim_boundary") != SENSITIVITY_CLAIM_BOUNDARY
                or not isinstance(runtime, Mapping)
                or runtime.get("device") != "cpu"
                or int(runtime.get("threads", -1)) != EVALUATION_REQUEST["threads"]
                or not math.isfinite(float(runtime.get("elapsed_seconds", math.nan)))
                or float(runtime["elapsed_seconds"]) <= 0.0
                or not isinstance(checkpoint, Mapping)
                or checkpoint.get("path") != expected_checkpoint["path"]
                or int(checkpoint.get("bytes", -1)) != expected_checkpoint["bytes"]
                or checkpoint.get("sha256") != expected_checkpoint["sha256"]
                or int(checkpoint.get("step", -1)) != step
                or checkpoint.get("dataset_identity_sha256")
                != prepared["dataset_identity_sha256"]
                or checkpoint.get("runtime_environment_sha256")
                != prepared["runtime_environment_sha256"]
                or checkpoint.get("git", {}).get("revision")
                != prepared["training_revision"]
                or not isinstance(model, Mapping)
                or model.get("predictor_type") != "scalable_unet"
                or int(model.get("num_classes", -1)) != 1_000
                or int(model.get("parameter_count", -1))
                != prepared["parameter_count"]
                or not isinstance(rows, list)
                or len(rows)
                != EVALUATION_REQUEST["num_samples"]
                * len(EVALUATION_REQUEST["timesteps"])
            ):
                raise ValueError(f"{method} step {step} sensitivity contract differs")
            expected_synthesis = "fixed_basis" if method == "cofitok" else "dense_identity"
            expected_tokens = 8 if method == "cofitok" else 1
            if (
                model.get("synthesis_mode") != expected_synthesis
                or int(model.get("token_count", -1)) != expected_tokens
            ):
                raise ValueError(f"{method} step {step} model identity differs")
            row_keys = [_row_key(row) for row in rows]
            if len(set(row_keys)) != len(row_keys):
                raise ValueError(f"{method} step {step} contains duplicate rows")
            if canonical_keys is None:
                canonical_keys = row_keys
            elif row_keys != canonical_keys:
                raise ValueError("sensitivity row identities differ across exposure")
            dataset = report.get("dataset")
            if not isinstance(dataset, Mapping):
                raise ValueError("sensitivity dataset is malformed")
            if canonical_dataset is None:
                canonical_dataset = dataset
            elif dataset != canonical_dataset:
                raise ValueError("sensitivity datasets differ across exposure")
            for row in rows:
                correct = _condition_mse(row, "correct")
                wrong = _condition_mse(row, "wrong")
                null = _condition_mse(row, "null")
                reported = row.get("correct_relative_mse_improvement")
                better = row.get("correct_better")
                if not isinstance(reported, Mapping) or not isinstance(better, Mapping):
                    raise ValueError("sensitivity derived fields are missing")
                for comparison, reference in (
                    ("versus_wrong", wrong),
                    ("versus_null", null),
                ):
                    expected = (reference - correct) / reference
                    if not math.isclose(
                        float(reported.get(comparison, math.nan)),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError("sensitivity improvement differs from raw MSE")
                if (
                    better.get("than_wrong") is not (correct < wrong)
                    or better.get("than_null") is not (correct < null)
                ):
                    raise ValueError("sensitivity correct-better field differs")
    return {
        "request": EVALUATION_REQUEST,
        "dataset": dict(canonical_dataset or {}),
        "evaluator_git": dict(evaluator_git),
        "canonical_row_keys": [list(key) for key in canonical_keys or []],
    }


def aggregate_checkpoint(report: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int, int], list[Mapping[str, Any]]] = {}
    for row in report["sample_rows"]:
        if int(row["timestep"]) in ELIGIBLE_TIMESTEPS:
            grouped.setdefault(_sample_key(row), []).append(row)
    if len(grouped) != EVALUATION_REQUEST["num_samples"]:
        raise ValueError("eligible held-out image count differs")
    sample_rows = []
    for key, rows in sorted(grouped.items()):
        if sorted(int(row["timestep"]) for row in rows) != ELIGIBLE_TIMESTEPS:
            raise ValueError("held-out image lacks an eligible timestep")
        condition_mse = {
            condition: _mean([_condition_mse(row, condition) for row in rows])
            for condition in ("correct", "wrong", "null")
        }
        correct = condition_mse["correct"]
        semantic = {
            "versus_wrong": (condition_mse["wrong"] - correct)
            / condition_mse["wrong"],
            "versus_null": (condition_mse["null"] - correct)
            / condition_mse["null"],
        }
        sample_rows.append(
            {
                "sample_index": key[0],
                "correct_label": key[1],
                "wrong_label": key[2],
                "noise_seed": key[3],
                "eligible_timesteps": ELIGIBLE_TIMESTEPS,
                "condition_epsilon_mse": condition_mse,
                "semantic_advantage": semantic,
            }
        )
    return {
        "sample_count": len(sample_rows),
        "eligible_row_count": len(sample_rows) * len(ELIGIBLE_TIMESTEPS),
        "independent_unit": "held_out_validation_image",
        "timestep_rows_aggregated_before_sign_test": True,
        "correct_mse_mean": _mean(
            [row["condition_epsilon_mse"]["correct"] for row in sample_rows]
        ),
        "semantic_advantage": {
            comparison: summarize_positive(
                [row["semantic_advantage"][comparison] for row in sample_rows]
            )
            for comparison in COMPARISONS
        },
        "per_sample": sample_rows,
    }


def exposure_trajectory(checkpoints: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    by_step = {
        step: {row["sample_index"]: row for row in checkpoints[step]["per_sample"]}
        for step in CHECKPOINT_STEPS
    }
    sample_indices = sorted(by_step[CHECKPOINT_STEPS[0]])
    if any(sorted(rows) != sample_indices for rows in by_step.values()):
        raise ValueError("checkpoint aggregate sample identities differ")
    per_sample = []
    for index in sample_indices:
        rows = {step: by_step[step][index] for step in CHECKPOINT_STEPS}
        first = rows[CHECKPOINT_STEPS[0]]
        last = rows[CHECKPOINT_STEPS[-1]]
        correct_mse = {
            step: rows[step]["condition_epsilon_mse"]["correct"]
            for step in CHECKPOINT_STEPS
        }
        advantages = {
            comparison: {
                step: rows[step]["semantic_advantage"][comparison]
                for step in CHECKPOINT_STEPS
            }
            for comparison in COMPARISONS
        }
        per_sample.append(
            {
                "sample_index": index,
                "correct_label": first["correct_label"],
                "wrong_label": first["wrong_label"],
                "noise_seed": first["noise_seed"],
                "correct_mse_by_step": correct_mse,
                "correct_mse_reduction_1250_to_5000": (
                    correct_mse[1_250] - correct_mse[5_000]
                )
                / correct_mse[1_250],
                "correct_mse_strictly_improves_each_interval": (
                    correct_mse[1_250] > correct_mse[2_500] > correct_mse[5_000]
                ),
                "semantic_advantage_by_step": advantages,
                "semantic_advantage_change_1250_to_5000": {
                    comparison: advantages[comparison][5_000]
                    - advantages[comparison][1_250]
                    for comparison in COMPARISONS
                },
                "semantic_advantage_strictly_improves_each_interval": {
                    comparison: (
                        advantages[comparison][1_250]
                        < advantages[comparison][2_500]
                        < advantages[comparison][5_000]
                    )
                    for comparison in COMPARISONS
                },
            }
        )
    denoising = summarize_positive(
        [row["correct_mse_reduction_1250_to_5000"] for row in per_sample]
    )
    semantic_change = {
        comparison: summarize_positive(
            [
                row["semantic_advantage_change_1250_to_5000"][comparison]
                for row in per_sample
            ]
        )
        for comparison in COMPARISONS
    }
    final_absolute = checkpoints[5_000]["semantic_advantage"]
    gates = {
        "denoising_mse_improves_with_exposure": (
            denoising["mean"] > 0.0
            and denoising["one_sided_sign_test_pvalue"] < SIGNIFICANCE_LEVEL
        ),
        "final_absolute_semantic_direction": all(
            summary["mean"] > 0.0
            and summary["one_sided_sign_test_pvalue"] < SIGNIFICANCE_LEVEL
            for summary in final_absolute.values()
        ),
        "semantic_advantage_improves_with_exposure": all(
            summary["mean"] > 0.0
            and summary["one_sided_sign_test_pvalue"] < SIGNIFICANCE_LEVEL
            for summary in semantic_change.values()
        ),
    }
    return {
        "sample_count": len(per_sample),
        "steps": CHECKPOINT_STEPS,
        "denoising_correct_mse_reduction": denoising,
        "semantic_advantage_change": semantic_change,
        "correct_mse_final_to_initial_ratio": (
            checkpoints[5_000]["correct_mse_mean"]
            / checkpoints[1_250]["correct_mse_mean"]
        ),
        "strict_interval_improvement_counts": {
            "correct_mse": sum(
                row["correct_mse_strictly_improves_each_interval"]
                for row in per_sample
            ),
            "semantic_advantage": {
                comparison: sum(
                    row["semantic_advantage_strictly_improves_each_interval"][
                        comparison
                    ]
                    for row in per_sample
                )
                for comparison in COMPARISONS
            },
        },
        "gates": gates,
        "semantic_recovery": all(gates.values()),
        "per_sample": per_sample,
    }


def classify_decision(methods: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    semantic = {
        method: bool(result["trajectory"]["semantic_recovery"])
        for method, result in methods.items()
    }
    denoising = {
        method: bool(
            result["trajectory"]["gates"]["denoising_mse_improves_with_exposure"]
        )
        for method, result in methods.items()
    }
    shared_semantic = all(semantic.values())
    method_asymmetry = len(set(semantic.values())) > 1 or len(set(denoising.values())) > 1
    denoising_only = all(denoising.values()) and not any(semantic.values())
    if shared_semantic:
        category = "shared_exposure_semantic_recovery"
        action = "exposure_screen_passed_but_no_scale_or_training_authorization"
    elif method_asymmetry:
        category = "method_asymmetry"
        action = "inspect_method_specific_exposure_response_without_scale_authorization"
    elif denoising_only:
        category = "denoising_improves_but_semantic_alignment_does_not"
        action = "reject_exposure_as_sufficient_semantic_recovery_explanation"
    else:
        category = "no_exposure_recovery"
        action = "do_not_add_another_semantic_auxiliary_or_authorize_scale"
    return {
        "category": category,
        "shared_exposure_semantic_recovery_supported": shared_semantic,
        "denoising_improves_but_semantic_alignment_does_not": denoising_only,
        "method_asymmetry": method_asymmetry,
        "no_exposure_recovery": category == "no_exposure_recovery",
        "method_semantic_recovery": semantic,
        "method_denoising_improvement": denoising,
        "recommended_next_action": action,
    }


def build_report(
    *,
    preparation: Mapping[str, Any],
    authorization: Mapping[str, Any],
    sensitivity_reports: Mapping[str, Mapping[int, Mapping[str, Any]]],
    sources: Mapping[str, Any],
    git: Mapping[str, Any],
) -> dict[str, Any]:
    full_git = preparation["git"]
    validate_preparation(
        preparation,
        expected_revision=full_git["revision"],
        expected_tree=full_git["tree"],
        expected_branch=full_git["branch"],
    )
    expected_git = {
        "revision": full_git["revision"],
        "branch": full_git["branch"],
        "tracked_dirty": False,
    }
    if dict(git) != expected_git:
        raise ValueError("trajectory report builder Git differs from preparation")
    if (
        authorization.get("schema_version") != SCHEMA_VERSION
        or authorization.get("role") != AUTHORIZATION_ROLE
        or authorization.get("status") != "authorized"
        or authorization.get("authorized_git") != full_git
        or authorization.get("output_root") != OUTPUT_ROOT
        or authorization.get("execution_boundary") != EXECUTION_BOUNDARY
        or authorization.get("claim_boundary") != CLAIM_BOUNDARY
        or authorization.get("generation_advantage_proven") is not False
    ):
        raise ValueError("trajectory execution authorization differs")
    sensitivity_contract = validate_sensitivity_reports(
        reports=sensitivity_reports,
        preparation=preparation,
        evaluator_git=git,
    )
    methods = {}
    for method in METHOD_RUN_DIRS:
        checkpoints = {
            step: aggregate_checkpoint(sensitivity_reports[method][step])
            for step in CHECKPOINT_STEPS
        }
        methods[method] = {
            "training_source": preparation["methods"][method],
            "checkpoints": {str(step): checkpoints[step] for step in CHECKPOINT_STEPS},
            "trajectory": exposure_trajectory(checkpoints),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "scope": SCOPE,
        "stage": STAGE,
        "git": dict(git),
        "sources": dict(sources),
        "evaluation_contract": {
            **sensitivity_contract,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "eligible_timesteps": ELIGIBLE_TIMESTEPS,
            "descriptive_only_timesteps": DESCRIPTIVE_TIMESTEPS,
            "paired_unit": "held_out_validation_image",
            "timestep_rows_are_not_independent_units": True,
            "significance_level": SIGNIFICANCE_LEVEL,
            "same_images_labels_timesteps_and_noise_across_all_checkpoints": True,
            "no_new_diffusion_samples_generated": True,
        },
        "methods": methods,
        "decision": classify_decision(methods),
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_boundary": EXECUTION_BOUNDARY,
        "generation_advantage_proven": False,
    }


def main() -> None:
    args = parse_args()
    output = reject_symlink_chain(args.output, name="trajectory report output").resolve()
    with exclusive_output_lock(output, role=REPORT_ROLE):
        preparation_path = reject_symlink_chain(
            args.preparation,
            name="trajectory preparation",
        ).resolve()
        authorization_path = reject_symlink_chain(
            args.execution_authorization,
            name="trajectory execution authorization",
        ).resolve()
        preparation = read_json_object(preparation_path, name="trajectory preparation")
        authorization = read_json_object(
            authorization_path,
            name="trajectory execution authorization",
        )
        reports: dict[str, dict[int, dict[str, Any]]] = {}
        sources: dict[str, Any] = {
            "preparation": file_identity(preparation_path),
            "execution_authorization": file_identity(authorization_path),
            "sensitivity": {},
        }
        for method in METHOD_RUN_DIRS:
            reports[method] = {}
            sources["sensitivity"][method] = {}
            for step in CHECKPOINT_STEPS:
                attribute = f"{method}_step_{step}_report"
                report, identity = _load_sensitivity(
                    getattr(args, attribute),
                    label=f"{method} step {step}",
                )
                reports[method][step] = report
                sources["sensitivity"][method][str(step)] = identity
        expected = build_report(
            preparation=preparation,
            authorization=authorization,
            sensitivity_reports=reports,
            sources=sources,
            git=git_provenance(PROJECT_ROOT),
        )
        if output.exists():
            if not args.resume:
                raise FileExistsError(
                    "trajectory report exists; pass --resume to validate it"
                )
            existing = read_json_object(output, name="trajectory report")
            if existing != expected:
                raise ValueError("completed trajectory report differs from sources")
        else:
            write_json_report(output, expected)
        print(output.as_posix())


if __name__ == "__main__":
    main()
