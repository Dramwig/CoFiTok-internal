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
from scripts.evaluate_generation_conditioning_sensitivity import (
    MANIFEST_FILENAME,
    REPORT_ROLE as SOURCE_REPORT_ROLE,
    REPORT_SCHEMA_VERSION as SOURCE_REPORT_SCHEMA_VERSION,
    validate_completed_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_conditioning_sensitivity_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound matched comparison of CoFiTok and dense "
            "class-conditioning sensitivity diagnostics."
        )
    )
    parser.add_argument("--cofitok-report", required=True)
    parser.add_argument("--dense-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Revalidate and reuse an exactly bound completed comparison.",
    )
    return parser.parse_args()


def _load_source(path: str | Path, *, method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = reject_symlink_chain(path, name=f"{method} conditioning report")
    if not source_path.is_file():
        raise FileNotFoundError(f"Conditioning report is missing: {source_path}")
    report = read_json_object(source_path, name=f"{method} conditioning report")
    if (
        report.get("schema_version") != SOURCE_REPORT_SCHEMA_VERSION
        or report.get("role") != SOURCE_REPORT_ROLE
        or report.get("status") != "completed"
    ):
        raise ValueError(f"{method} conditioning report is not completed evidence")
    manifest_path = reject_symlink_chain(
        source_path.parent / MANIFEST_FILENAME,
        name=f"{method} conditioning manifest",
    )
    manifest = read_json_object(
        manifest_path,
        name=f"{method} conditioning manifest",
    )
    manifest_identity = file_identity(manifest_path)
    validate_completed_report(
        report,
        manifest=manifest,
        manifest_identity=manifest_identity,
    )
    return report, {
        "report": file_identity(source_path),
        "manifest": manifest_identity,
    }


def _paired_contract(
    cofitok: Mapping[str, Any],
    dense: Mapping[str, Any],
) -> dict[str, Any]:
    if cofitok.get("git") != dense.get("git"):
        raise ValueError("Conditioning diagnostics used different evaluator Git")
    if cofitok.get("request") != dense.get("request"):
        raise ValueError("Conditioning diagnostic requests differ")
    if cofitok.get("dataset") != dense.get("dataset"):
        raise ValueError("Conditioning diagnostic datasets or samples differ")
    cofitok_model = cofitok.get("model")
    dense_model = dense.get("model")
    if not isinstance(cofitok_model, Mapping) or not isinstance(dense_model, Mapping):
        raise ValueError("Conditioning diagnostic model metadata is malformed")
    for field in ("predictor_type", "num_classes"):
        if cofitok_model.get(field) != dense_model.get(field):
            raise ValueError(f"Conditioning diagnostic model {field} differs")
    if cofitok.get("weights") != dense.get("weights"):
        raise ValueError("Conditioning diagnostic weight selections differ")
    if cofitok["checkpoint"]["sha256"] == dense["checkpoint"]["sha256"]:
        raise ValueError("Conditioning comparison requires distinct checkpoints")
    return {
        "git": cofitok["git"],
        "request": cofitok["request"],
        "dataset": cofitok["dataset"],
        "weights": cofitok["weights"],
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


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return sum(values) / len(values)


def summarize_method(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report.get("sample_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Conditioning report has no sample rows")
    comparisons = ("correct_vs_wrong", "correct_vs_null", "wrong_vs_null")
    advantages = ("versus_wrong", "versus_null")
    counts = {
        "than_wrong": sum(bool(row["correct_better"]["than_wrong"]) for row in rows),
        "than_null": sum(bool(row["correct_better"]["than_null"]) for row in rows),
    }
    trials = len(rows)
    return {
        "checkpoint": report["checkpoint"],
        "model": report["model"],
        "row_count": trials,
        "relative_delta_to_correct_rms_mean": {
            comparison: _mean(
                [float(row["relative_delta_to_correct_rms"][comparison]) for row in rows]
            )
            for comparison in comparisons
        },
        "correct_relative_mse_improvement_mean": {
            comparison: _mean(
                [
                    float(row["correct_relative_mse_improvement"][comparison])
                    for row in rows
                ]
            )
            for comparison in advantages
        },
        "correct_better_count": counts,
        "correct_better_fraction": {
            key: value / trials for key, value in counts.items()
        },
        "one_sided_sign_test_pvalue": {
            key: _one_sided_binomial_pvalue(value, trials)
            for key, value in counts.items()
        },
        "ema_relative_parameter_update": {
            name: row["relative_update_rms"]
            for name, row in report["parameter_update_audit"]["ema"].items()
        },
    }


def _row_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(row["sample_index"]),
        int(row["timestep"]),
        int(row["correct_label"]),
        int(row["wrong_label"]),
        int(row["noise_seed"]),
    )


def paired_differences(
    cofitok: Mapping[str, Any],
    dense: Mapping[str, Any],
) -> dict[str, Any]:
    cofitok_rows = {_row_key(row): row for row in cofitok["sample_rows"]}
    dense_rows = {_row_key(row): row for row in dense["sample_rows"]}
    if len(cofitok_rows) != len(cofitok["sample_rows"]):
        raise ValueError("CoFiTok conditioning rows contain duplicate identities")
    if len(dense_rows) != len(dense["sample_rows"]):
        raise ValueError("Dense conditioning rows contain duplicate identities")
    if cofitok_rows.keys() != dense_rows.keys():
        raise ValueError("Conditioning row identities differ between methods")

    metrics: dict[str, list[float]] = {
        "correct_vs_wrong_relative_delta": [],
        "correct_vs_null_relative_delta": [],
        "correct_vs_wrong_mse_improvement": [],
        "correct_vs_null_mse_improvement": [],
    }
    for key in sorted(cofitok_rows):
        cofitok_row = cofitok_rows[key]
        dense_row = dense_rows[key]
        metrics["correct_vs_wrong_relative_delta"].append(
            float(
                cofitok_row["relative_delta_to_correct_rms"]["correct_vs_wrong"]
            )
            - float(dense_row["relative_delta_to_correct_rms"]["correct_vs_wrong"])
        )
        metrics["correct_vs_null_relative_delta"].append(
            float(cofitok_row["relative_delta_to_correct_rms"]["correct_vs_null"])
            - float(dense_row["relative_delta_to_correct_rms"]["correct_vs_null"])
        )
        metrics["correct_vs_wrong_mse_improvement"].append(
            float(
                cofitok_row["correct_relative_mse_improvement"]["versus_wrong"]
            )
            - float(
                dense_row["correct_relative_mse_improvement"]["versus_wrong"]
            )
        )
        metrics["correct_vs_null_mse_improvement"].append(
            float(cofitok_row["correct_relative_mse_improvement"]["versus_null"])
            - float(dense_row["correct_relative_mse_improvement"]["versus_null"])
        )
    return {
        "row_count": len(cofitok_rows),
        "cofitok_minus_dense_mean": {
            name: _mean(values) for name, values in metrics.items()
        },
    }


def build_comparison(
    *,
    cofitok: Mapping[str, Any],
    dense: Mapping[str, Any],
    sources: Mapping[str, Any],
    git: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _paired_contract(cofitok, dense)
    method_summaries = {
        "cofitok_k8": summarize_method(cofitok),
        "dense_identity": summarize_method(dense),
    }
    no_consistent_advantage = {
        method: {
            "over_wrong_not_significant_at_0_05": (
                summary["one_sided_sign_test_pvalue"]["than_wrong"] >= 0.05
            ),
            "over_null_not_significant_at_0_05": (
                summary["one_sided_sign_test_pvalue"]["than_null"] >= 0.05
            ),
        }
        for method, summary in method_summaries.items()
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "git": dict(git),
        "sources": dict(sources),
        "contract": contract,
        "methods": method_summaries,
        "paired_differences": paired_differences(cofitok, dense),
        "diagnostic_interpretation": {
            "correct_label_advantage_not_demonstrated": no_consistent_advantage,
            "shared_conditioning_weakness_supported": all(
                all(flags.values()) for flags in no_consistent_advantage.values()
            ),
            "scope": (
                "Internal checkpoint response diagnostic only; this does not replace "
                "formal generated-sample class fidelity or quality gates."
            ),
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "authorizes_training": False,
            "authorizes_sampling": False,
            "replaces_formal_quality_gate": False,
        },
    }


def _validate_completed_comparison(
    report: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    if report != expected:
        raise ValueError("Completed conditioning comparison differs from current sources")


def main() -> None:
    args = parse_args()
    output = reject_symlink_chain(args.output, name="conditioning comparison output")
    if output.exists() and not output.is_file():
        raise ValueError(f"Conditioning comparison output is not a file: {output}")
    with exclusive_output_lock(output, role=REPORT_ROLE):
        cofitok, cofitok_source = _load_source(
            args.cofitok_report,
            method="CoFiTok",
        )
        dense, dense_source = _load_source(
            args.dense_report,
            method="dense identity",
        )
        expected = build_comparison(
            cofitok=cofitok,
            dense=dense,
            sources={
                "cofitok_k8": cofitok_source,
                "dense_identity": dense_source,
            },
            git=git_provenance(PROJECT_ROOT),
        )
        if output.exists():
            if not args.resume:
                raise FileExistsError(
                    "Conditioning comparison already exists; pass --resume to validate it"
                )
            existing = read_json_object(output, name="conditioning comparison")
            _validate_completed_comparison(existing, expected=expected)
        else:
            write_json_report(output, expected)
        print(output.resolve().as_posix())


if __name__ == "__main__":
    main()
