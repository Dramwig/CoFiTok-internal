from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_posttraining_sampling import (
    CLAIM_BOUNDARY,
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_HELDOUT_DECISION,
    EXPECTED_OUTPUT_ROOT,
    METHOD_PREFIX_BUDGETS,
    RUN_NAMES,
    SAMPLE_RUN_NAME,
    SAMPLING_PROTOCOL,
    STAGE,
    build_sampling_execution_receipt,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from scripts.build_generation_conditioning_ranking_sampling_validation import (
    EXPECTED_REAL_SET,
    METHOD_ARMS,
    _clean_git,
    _identity,
    _load_json_source,
    method_sampling_decision,
    validate_generation_metrics_reports,
    validate_paired_class_reports,
    validate_sampling_evidence,
)
from scripts.prepare_generation_conditioning_ranking_posttraining_sampling_confirmation import (
    replay_heldout_evaluation,
    replay_sampling_preparation,
    validate_training_checkpoints,
)
from scripts.select_generation_conditioning_ranking_sampling_batch import (
    validate_completed_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = (
    "generation_conditioning_ranking_four_arm_posttraining_sampling_confirmation"
)
REPORT_FILENAME = "posttraining_sampling_confirmation.json"
ARM_METHOD = {
    "control_cofitok": "cofitok",
    "ranked_cofitok": "cofitok",
    "control_dense_identity": "dense_identity",
    "ranked_dense_identity": "dense_identity",
}


def _embedded_json(
    descriptor: object,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} identity is missing")
    report, identity = _load_json_source(
        str(descriptor.get("path", "")),
        label=label,
    )
    if identity != descriptor:
        raise ValueError(f"{label} identity differs")
    return report, identity


def replay_execution_receipt(
    *,
    output_root: Path,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    expected_git: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = output_root / "reports" / "execution_receipt.json"
    receipt, receipt_identity = _load_json_source(
        receipt_path,
        label="posttraining sampling execution receipt",
    )
    sources = receipt.get("source_reports")
    if not isinstance(sources, Mapping):
        raise ValueError("posttraining execution-receipt sources are missing")
    if sources.get("preparation") != preparation_identity:
        raise ValueError("posttraining execution receipt preparation differs")
    standing, standing_identity = _embedded_json(
        sources.get("standing_authorization"),
        label="standing experiment authorization",
    )
    idle, idle_identity = _embedded_json(
        sources.get("idle_gpu_evidence"),
        label="posttraining idle GPU evidence",
    )
    selection, selection_identity = _embedded_json(
        sources.get("batch_selection"),
        label="posttraining sampling batch selection",
    )
    validate_completed_selection(
        selection,
        preparation=preparation,
        preparation_identity=preparation_identity,
    )
    expected = build_sampling_execution_receipt(
        preparation=preparation,
        preparation_identity=preparation_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        idle_gpu_evidence=idle,
        idle_gpu_evidence_identity=idle_identity,
        batch_selection=selection,
        batch_selection_identity=selection_identity,
        receipt_git=expected_git,
        expected_revision=str(expected_git["revision"]),
        expected_branch=str(expected_git["branch"]),
        expected_output_root=output_root.as_posix(),
    )
    if receipt != expected:
        raise ValueError("posttraining sampling execution receipt replay differs")
    return receipt, receipt_identity


def build_posttraining_sampling_confirmation(
    *,
    heldout_evaluation: Mapping[str, Any],
    heldout_identity: Mapping[str, Any],
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    execution_receipt_identity: Mapping[str, Any],
    checkpoint_evidence: Mapping[str, Any],
    sampling_provenance: Mapping[str, Mapping[str, Any]],
    sampling_sources: Mapping[str, Any],
    generation_reports: Mapping[str, Mapping[str, Any]],
    generation_sources: Mapping[str, Any],
    paired_reports: Mapping[str, Mapping[str, Any]],
    paired_sources: Mapping[str, Any],
    git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    clean_git = _clean_git(git, label="posttraining sampling confirmation builder")
    if dict(heldout_evaluation.get("decision", {})) != EXPECTED_HELDOUT_DECISION:
        raise ValueError("posttraining sampling source decision differs")
    if preparation.get("source_reports", {}).get("heldout_evaluation") != heldout_identity:
        raise ValueError("posttraining preparation heldout identity differs")
    if (
        preparation.get("sampling_protocol") != SAMPLING_PROTOCOL
        or preparation.get("source_reports", {}).get("training_checkpoints")
        != checkpoint_evidence
        or execution_receipt.get("training_checkpoints") != checkpoint_evidence
    ):
        raise ValueError("posttraining sampling source contract differs")

    methods: dict[str, Any] = {}
    for method, arms in METHOD_ARMS.items():
        methods[method] = method_sampling_decision(
            method=method,
            paired_metrics=paired_reports[method]["metrics"],
            control_generation=generation_reports[arms[0]],
            ranked_generation=generation_reports[arms[1]],
        )
    method_passes = {method: result["pass"] for method, result in methods.items()}
    shared = all(method_passes.values())
    if shared:
        next_action = (
            "retain_shared_conditioning_ranking_recipe_for_separately_"
            "authorized_future_scaling"
        )
    elif any(method_passes.values()):
        next_action = "reject_shared_repair_due_posttraining_method_asymmetry"
    else:
        next_action = "revise_training_time_semantic_alignment_objective"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "stage": STAGE,
        "output_root": output_root,
        "git": clean_git,
        "sources": {
            "heldout_evaluation": _identity(
                heldout_identity,
                label="5K heldout evaluation",
            ),
            "preparation": _identity(
                preparation_identity,
                label="posttraining sampling preparation",
            ),
            "execution_receipt": _identity(
                execution_receipt_identity,
                label="posttraining sampling execution receipt",
            ),
            "training_checkpoints": dict(checkpoint_evidence),
            "sampling": dict(sampling_sources),
            "generation_metrics": dict(generation_sources),
            "paired_class_fidelity": dict(paired_sources),
        },
        "source_decision": dict(heldout_evaluation["decision"]),
        "sampling_contract": {
            "sample_count_per_arm": SAMPLING_PROTOCOL["num_samples_per_arm"],
            "sample_run_name": SAMPLE_RUN_NAME,
            "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
            "seed": SAMPLING_PROTOCOL["seed"],
            "start_index": SAMPLING_PROTOCOL["start_index"],
            "random_stream": SAMPLING_PROTOCOL["random_stream"],
            "methods": {
                method: {
                    "control": sampling_provenance[arms[0]],
                    "ranked": sampling_provenance[arms[1]],
                }
                for method, arms in METHOD_ARMS.items()
            },
        },
        "methods": methods,
        "difference_in_differences": {
            "mean_target_log_probability": (
                methods["cofitok"]["ranked_minus_control"][
                    "mean_target_log_probability"
                ]
                - methods["dense_identity"]["ranked_minus_control"][
                    "mean_target_log_probability"
                ]
            ),
            "fid_ratio": (
                methods["cofitok"]["ranked_minus_control"]["fid_ratio"]
                - methods["dense_identity"]["ranked_minus_control"]["fid_ratio"]
            ),
        },
        "difference_in_differences_is_descriptive_only": True,
        "decision": {
            "method_passes": method_passes,
            "shared_posttraining_generated_class_alignment_recovery_confirmed": shared,
            "cofitok_specific_advantage_claim_allowed": False,
            "recommended_next_action": next_action,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _rebuild(
    *,
    heldout_path: str | Path,
    output_root: Path,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    heldout_report, heldout_identity = replay_heldout_evaluation(heldout_path)
    preparation_path = output_root / "reports" / "preparation.json"
    preparation, preparation_identity = replay_sampling_preparation(preparation_path)
    if preparation["source_reports"]["heldout_evaluation"] != heldout_identity:
        raise ValueError("posttraining confirmation uses another heldout report")
    training_status, training_status_identity = _embedded_json(
        preparation["source_reports"]["training_status"],
        label="5K training status",
    )
    if training_status_identity != heldout_report["sources"]["training_status"]:
        raise ValueError("posttraining confirmation training status differs")
    checkpoint_evidence = validate_training_checkpoints(
        heldout_report,
        training_status,
    )
    execution_receipt, execution_receipt_identity = replay_execution_receipt(
        output_root=output_root,
        preparation=preparation,
        preparation_identity=preparation_identity,
        expected_git=git,
    )
    sampling_provenance, sampling_sources = validate_sampling_evidence(
        output_root=output_root,
        checkpoint_evidence=checkpoint_evidence,
        expected_git=git,
        expected_checkpoint_step=EXPECTED_CHECKPOINT_STEP,
        expected_num_samples=SAMPLING_PROTOCOL["num_samples_per_arm"],
        sample_run_name=SAMPLE_RUN_NAME,
        method_prefix_budgets=METHOD_PREFIX_BUDGETS,
        arm_method=ARM_METHOD,
    )
    generation_reports, generation_sources = validate_generation_metrics_reports(
        output_root=output_root,
        sampling_provenance=sampling_provenance,
        expected_git=git,
        expected_num_samples=SAMPLING_PROTOCOL["num_samples_per_arm"],
        sample_run_name=SAMPLE_RUN_NAME,
        expected_real_set=EXPECTED_REAL_SET,
    )
    paired_reports, paired_sources = validate_paired_class_reports(
        output_root=output_root,
        sampling_provenance=sampling_provenance,
        expected_git=git,
        expected_num_samples=SAMPLING_PROTOCOL["num_samples_per_arm"],
        expected_start_index=SAMPLING_PROTOCOL["start_index"],
        sampling_stage=STAGE,
    )
    return build_posttraining_sampling_confirmation(
        heldout_evaluation=heldout_report,
        heldout_identity=heldout_identity,
        preparation=preparation,
        preparation_identity=preparation_identity,
        execution_receipt=execution_receipt,
        execution_receipt_identity=execution_receipt_identity,
        checkpoint_evidence=checkpoint_evidence,
        sampling_provenance=sampling_provenance,
        sampling_sources=sampling_sources,
        generation_reports=generation_reports,
        generation_sources=generation_sources,
        paired_reports=paired_reports,
        paired_sources=paired_sources,
        git=git,
        output_root=output_root.as_posix(),
    )


def replay_posttraining_sampling_confirmation(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report, identity = _load_json_source(
        path,
        label="posttraining sampling confirmation",
    )
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != STAGE
        or report.get("output_root") != EXPECTED_OUTPUT_ROOT
    ):
        raise ValueError("posttraining sampling confirmation differs")
    sources = report.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("posttraining sampling confirmation sources are missing")
    heldout_descriptor = sources.get("heldout_evaluation")
    if not isinstance(heldout_descriptor, Mapping):
        raise ValueError("posttraining heldout source is missing")
    git = _clean_git(report.get("git", {}), label="posttraining replay Git")
    expected = _rebuild(
        heldout_path=str(heldout_descriptor.get("path", "")),
        output_root=Path(EXPECTED_OUTPUT_ROOT).resolve(),
        git=git,
    )
    if expected != report:
        raise ValueError("posttraining sampling confirmation replay differs")
    return report, identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-bound posttraining four-arm 5K sampling "
            "confirmation on the independent random stream."
        )
    )
    parser.add_argument("--heldout-evaluation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="posttraining sampling output root",
    ).resolve()
    if output_root.as_posix() != EXPECTED_OUTPUT_ROOT:
        raise ValueError("posttraining sampling output root differs")
    output = reject_symlink_chain(
        args.output,
        name="posttraining sampling confirmation output",
    ).resolve()
    expected_output = output_root / "reports" / REPORT_FILENAME
    if output != expected_output:
        raise ValueError("posttraining sampling confirmation output differs")
    current_git = git_provenance(PROJECT_ROOT)
    _clean_git(current_git, label="posttraining sampling confirmation builder")
    with exclusive_output_lock(output, role=REPORT_ROLE):
        expected = _rebuild(
            heldout_path=args.heldout_evaluation,
            output_root=output_root,
            git=current_git,
        )
        if output.exists():
            if not args.resume:
                raise FileExistsError(
                    "posttraining sampling confirmation exists; pass --resume"
                )
            existing = read_json_object(
                output,
                name="posttraining sampling confirmation",
            )
            if existing != expected:
                raise ValueError(
                    "completed posttraining sampling confirmation differs from sources"
                )
        else:
            write_json_report(output, expected)
        print(output.as_posix())


if __name__ == "__main__":
    main()
