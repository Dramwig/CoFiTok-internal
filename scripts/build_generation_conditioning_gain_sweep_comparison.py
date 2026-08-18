from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from scripts.evaluate_generation_conditioning_gain_sweep import (
    REPORT_ROLE as SOURCE_REPORT_ROLE,
    REPORT_SCHEMA_VERSION as SOURCE_REPORT_SCHEMA_VERSION,
    validate_completed_gain_sweep_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_conditioning_gain_sweep_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound matched comparison of CoFiTok and dense "
            "inference-only class-conditioning gain sweeps."
        )
    )
    parser.add_argument("--cofitok-report", required=True)
    parser.add_argument("--dense-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_source(
    path: str | Path,
    *,
    method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = reject_symlink_chain(
        path,
        name=f"{method} conditioning gain sweep report",
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"Gain sweep report is missing: {source_path}")
    report = read_json_object(source_path, name=f"{method} gain sweep report")
    if (
        report.get("schema_version") != SOURCE_REPORT_SCHEMA_VERSION
        or report.get("role") != SOURCE_REPORT_ROLE
        or report.get("status") != "completed"
    ):
        raise ValueError(f"{method} gain sweep is not completed evidence")
    validate_completed_gain_sweep_report(
        report,
        git=report["git"],
        checkpoint=report["checkpoint"],
        request=report["request"],
        dataset=report["dataset"],
    )
    return report, file_identity(source_path)


def _paired_contract(
    cofitok: Mapping[str, Any],
    dense: Mapping[str, Any],
) -> dict[str, Any]:
    for field in ("git", "request", "dataset"):
        if cofitok.get(field) != dense.get(field):
            raise ValueError(f"Gain sweep paired field differs: {field}")
    cofitok_model = cofitok.get("model")
    dense_model = dense.get("model")
    if not isinstance(cofitok_model, Mapping) or not isinstance(dense_model, Mapping):
        raise ValueError("Gain sweep model metadata is malformed")
    for field in ("predictor_type", "num_classes"):
        if cofitok_model.get(field) != dense_model.get(field):
            raise ValueError(f"Gain sweep shared model field differs: {field}")
    if cofitok["checkpoint"]["sha256"] == dense["checkpoint"]["sha256"]:
        raise ValueError("Gain sweep comparison requires distinct checkpoints")
    return {
        "git": cofitok["git"],
        "request": cofitok["request"],
        "dataset": cofitok["dataset"],
        "shared_model": {
            "predictor_type": cofitok_model["predictor_type"],
            "num_classes": cofitok_model["num_classes"],
        },
    }


def _one_sided_binomial_pvalue(successes: int, trials: int) -> float:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Invalid binomial count")
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (
        2**trials
    )


def _method_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for gain, summary in report["gain_summary"].items():
        trials = int(summary["row_count"])
        counts = {
            key: int(value)
            for key, value in summary["correct_better_count"].items()
        }
        result[gain] = {
            **summary,
            "correct_better_fraction": {
                key: value / trials for key, value in counts.items()
            },
            "one_sided_sign_test_pvalue": {
                key: _one_sided_binomial_pvalue(value, trials)
                for key, value in counts.items()
            },
        }
    return result


def _strict_semantic_support(summary: Mapping[str, Any]) -> bool:
    pvalues = summary["one_sided_sign_test_pvalue"]
    improvements = summary["correct_relative_mse_improvement_mean"]
    return (
        float(pvalues["than_wrong"]) < 0.05
        and float(pvalues["than_null"]) < 0.05
        and float(improvements["versus_wrong"]) > 0.0
        and float(improvements["versus_null"]) > 0.0
    )


def _interpretation(methods: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gain_keys = set(methods["cofitok_k8"])
    if gain_keys != set(methods["dense_identity"]) or "1" not in gain_keys:
        raise ValueError("Gain sweep comparison requires matched gains including 1")
    higher_gains = sorted(
        (gain for gain in gain_keys if float(gain) > 1.0),
        key=float,
    )
    baseline = {
        method: summaries["1"] for method, summaries in methods.items()
    }
    candidate_rows = []
    for gain in higher_gains:
        strict_shared_recovery = all(
            _strict_semantic_support(methods[method][gain])
            for method in methods
        )
        improves_wrong_mean = all(
            float(
                methods[method][gain]["correct_relative_mse_improvement_mean"]
                ["versus_wrong"]
            )
            > float(
                baseline[method]["correct_relative_mse_improvement_mean"]
                ["versus_wrong"]
            )
            for method in methods
        )
        improves_null_mean = all(
            float(
                methods[method][gain]["correct_relative_mse_improvement_mean"]
                ["versus_null"]
            )
            > float(
                baseline[method]["correct_relative_mse_improvement_mean"]
                ["versus_null"]
            )
            for method in methods
        )
        amplifies_output_delta = all(
            float(
                methods[method][gain]["relative_delta_to_correct_rms_mean"]
                ["correct_vs_wrong"]
            )
            > float(
                baseline[method]["relative_delta_to_correct_rms_mean"]
                ["correct_vs_wrong"]
            )
            for method in methods
        )
        candidate_rows.append(
            {
                "gain": float(gain),
                "strict_shared_semantic_recovery": strict_shared_recovery,
                "improves_both_methods_correct_vs_wrong_mean": improves_wrong_mean,
                "improves_both_methods_correct_vs_null_mean": improves_null_mean,
                "amplifies_both_methods_correct_vs_wrong_output_delta": (
                    amplifies_output_delta
                ),
            }
        )
    recovery = any(row["strict_shared_semantic_recovery"] for row in candidate_rows)
    amplified_without_recovery = (
        not recovery
        and any(
            row["amplifies_both_methods_correct_vs_wrong_output_delta"]
            for row in candidate_rows
        )
    )
    if recovery:
        next_action = "validate_shared_conditioning_gain_in_sampling"
    elif amplified_without_recovery:
        next_action = (
            "develop_matched_training_time_label_ranking_or_contrastive_"
            "denoising_loss"
        )
    else:
        next_action = "inspect_shared_conditioning_fusion_and_training_signal"
    return {
        "baseline_gain": 1.0,
        "higher_gain_rows": candidate_rows,
        "shared_inference_gain_recovery_supported": recovery,
        "gain_only_amplifies_without_semantic_recovery": amplified_without_recovery,
        "recommended_next_action": next_action,
    }


def build_comparison(
    *,
    cofitok: Mapping[str, Any],
    dense: Mapping[str, Any],
    sources: Mapping[str, Any],
    git: Mapping[str, Any],
) -> dict[str, Any]:
    paired = _paired_contract(cofitok, dense)
    methods = {
        "cofitok_k8": _method_summary(cofitok),
        "dense_identity": _method_summary(dense),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "git": dict(git),
        "sources": dict(sources),
        "paired_contract": paired,
        "methods": methods,
        "diagnostic_interpretation": _interpretation(methods),
        "claim_boundary": {
            "diagnostic_only": True,
            "authorizes_training": False,
            "authorizes_sampling": False,
            "authorizes_checkpoint_modification": False,
            "replaces_formal_quality_gate": False,
        },
    }


def _validate_completed_comparison(
    report: Mapping[str, Any],
    *,
    git: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> None:
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("git") != git
        or report.get("sources") != sources
    ):
        raise ValueError("Completed gain sweep comparison binding differs")
    interpretation = report.get("diagnostic_interpretation")
    if (
        not isinstance(interpretation, Mapping)
        or not isinstance(interpretation.get("higher_gain_rows"), list)
        or not interpretation["higher_gain_rows"]
    ):
        raise ValueError("Completed gain sweep comparison is malformed")


def main() -> None:
    args = parse_args()
    output = reject_symlink_chain(
        args.output,
        name="conditioning gain sweep comparison output",
    )
    if output.exists() and not output.is_file():
        raise ValueError(f"Comparison output is not a file: {output}")
    with exclusive_output_lock(output, role=REPORT_ROLE):
        cofitok, cofitok_identity = _load_source(
            args.cofitok_report,
            method="cofitok_k8",
        )
        dense, dense_identity = _load_source(
            args.dense_report,
            method="dense_identity",
        )
        git = git_provenance(PROJECT_ROOT)
        sources = {
            "cofitok_k8": cofitok_identity,
            "dense_identity": dense_identity,
        }
        if output.exists():
            if not args.resume:
                raise FileExistsError(
                    "Gain sweep comparison exists; pass --resume to validate it"
                )
            report = read_json_object(output, name="gain sweep comparison")
            _validate_completed_comparison(report, git=git, sources=sources)
            print(output.resolve().as_posix())
            return
        report = build_comparison(
            cofitok=cofitok,
            dense=dense,
            sources=sources,
            git=git,
        )
        write_json_report(output, report)
        _validate_completed_comparison(report, git=git, sources=sources)
        print(output.resolve().as_posix())


if __name__ == "__main__":
    main()
