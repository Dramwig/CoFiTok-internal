from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation_paths import generation_capacity_full_workspace_paths
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.audit_generation_stability_completion import (
        MILESTONE_STEPS,
        _check,
        aggregate_checks,
        checkpoint_evidence,
        formal_generation_evidence,
        gate_evidence,
        milestone_evidence,
        progress_audit_evidence,
        runtime_and_visual_evidence,
    )
    from scripts.audit_large_scale_generation_completion import (
        _comparison_evidence,
        _inference_export_evidence,
        _verify_checkpoint_file,
        _verify_formal_real_set_files,
        _verify_formal_sample_files,
        _verify_inference_artifact_file,
        _verify_inference_smoke_outputs,
    )
    from scripts.build_large_scale_generation_comparison import (
        verify_comparison_source_reports,
    )
    from scripts.run_generation_capacity_full_300k_posteval_supervisor import (
        SUPERVISOR_BOUNDARY as POSTEVAL_SUPERVISOR_BOUNDARY,
        validate_posteval_result,
        validate_training_completion,
        validate_training_supervisor_deployment,
    )
except ModuleNotFoundError:
    from audit_generation_stability_completion import (
        MILESTONE_STEPS,
        _check,
        aggregate_checks,
        checkpoint_evidence,
        formal_generation_evidence,
        gate_evidence,
        milestone_evidence,
        progress_audit_evidence,
        runtime_and_visual_evidence,
    )
    from audit_large_scale_generation_completion import (
        _comparison_evidence,
        _inference_export_evidence,
        _verify_checkpoint_file,
        _verify_formal_real_set_files,
        _verify_formal_sample_files,
        _verify_inference_artifact_file,
        _verify_inference_smoke_outputs,
    )
    from build_large_scale_generation_comparison import (
        verify_comparison_source_reports,
    )
    from run_generation_capacity_full_300k_posteval_supervisor import (
        SUPERVISOR_BOUNDARY as POSTEVAL_SUPERVISOR_BOUNDARY,
        validate_posteval_result,
        validate_training_completion,
        validate_training_supervisor_deployment,
    )


PROFILE = "capacity_full_generation_system_v1"
RELEASE_CHECK = "capacity_full_release_authorized_inference"
POSTEVAL_DEPLOYMENT_ROLE = "capacity_full_300k_posteval_supervisor_deployment"
POSTEVAL_SUPERVISOR_ROLE = "capacity_full_300k_posteval_supervisor"
SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
FORMAL_REAL_DIR = Path(
    "/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val"
)
OFFICIAL_RELATED = Path(
    "artifacts/reports/baselines/official_related_methods_2026-07-11_final/"
    "official_related_methods_table.json"
)


def _read_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"__load_error__": f"{path}: {error}"}
    if not isinstance(payload, dict):
        return {"__load_error__": f"{path}: expected a JSON object"}
    return payload


def _revision(value: str, *, name: str) -> str:
    if SHA1.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full lowercase Git SHA-1")
    return value


def _sha256(value: str, *, name: str) -> str:
    if SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def validate_posteval_supervisor_deployment(
    report: Mapping[str, Any],
    *,
    receipt_path: Path,
    expected_sha256: str,
    evaluation_project: Path,
    training_project: Path,
    full_output_root: Path,
    expected_evaluation_git: Mapping[str, Any],
    expected_training_git: Mapping[str, Any],
    training_supervisor_deployment_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = file_identity(receipt_path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("capacity-full posteval deployment SHA256 differs")
    checkout = report.get("checkout")
    training = report.get("training_checkout")
    if (
        report.get("schema_version") != 1
        or report.get("role") != POSTEVAL_DEPLOYMENT_ROLE
        or report.get("status") != "active"
        or report.get("git") != expected_evaluation_git
        or not isinstance(checkout, Mapping)
        or checkout.get("git") != expected_evaluation_git
        or Path(str(checkout.get("path", ""))).resolve()
        != evaluation_project.resolve()
        or not isinstance(training, Mapping)
        or training.get("git") != expected_training_git
        or Path(str(training.get("path", ""))).resolve()
        != training_project.resolve()
        or Path(str(report.get("full_output_root", ""))).resolve()
        != full_output_root.resolve()
        or report.get("training_supervisor_deployment")
        != dict(training_supervisor_deployment_identity)
        or report.get("authorization_boundary") != POSTEVAL_SUPERVISOR_BOUNDARY
    ):
        raise ValueError("capacity-full posteval deployment contract differs")
    sources = report.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("capacity-full posteval deployment sources are missing")
    for name, source in sources.items():
        if not isinstance(source, Mapping) or file_identity(source["path"]) != dict(source):
            raise ValueError(f"capacity-full posteval deployment source changed: {name}")
    return {"identity": identity, "git": dict(expected_evaluation_git)}


def validate_posteval_supervisor_status(
    report: Mapping[str, Any],
    *,
    status_path: Path,
    expected_evaluation_git: Mapping[str, Any],
    expected_training_git: Mapping[str, Any],
    full_output_root: Path,
    expected_result: Mapping[str, Any],
) -> dict[str, Any]:
    expected = report.get("expected")
    result_passed = expected_result.get("status") == "pass"
    expected_status = "pass" if result_passed else "hold"
    expected_detail = (
        "capacity_full_final_quality_gate_passed"
        if result_passed
        else "capacity_full_final_quality_gate_held"
    )
    if (
        report.get("schema_version") != 1
        or report.get("role") != POSTEVAL_SUPERVISOR_ROLE
        or report.get("status") != expected_status
        or report.get("detail") != expected_detail
        or report.get("final_quality_gate_passed") is not result_passed
        or report.get("inference_export_performed") is not False
        or report.get("release_receipt_built") is not False
        or report.get("formal_generation_completion_claimed") is not False
        or report.get("authorization_boundary") != POSTEVAL_SUPERVISOR_BOUNDARY
        or report.get("posteval_result") != dict(expected_result)
        or not isinstance(expected, Mapping)
        or expected.get("evaluation_git") != expected_evaluation_git
        or expected.get("training_git") != expected_training_git
        or Path(str(expected.get("full_output_root", ""))).resolve()
        != full_output_root.resolve()
    ):
        raise ValueError("capacity-full posteval supervisor pass state differs")
    return {"identity": file_identity(status_path), "result": dict(expected_result)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed completion audit for the source-bound experimental "
            "capacity-full matched 300K generation lineage."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/CoFiTok/checkpoints/generation"),
    )
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--formal-project", type=Path, required=True)
    parser.add_argument("--training-supervisor-status", type=Path, required=True)
    parser.add_argument("--training-supervisor-deployment", type=Path, required=True)
    parser.add_argument("--posteval-supervisor-status", type=Path, required=True)
    parser.add_argument("--posteval-supervisor-deployment", type=Path, required=True)
    parser.add_argument("--expected-training-supervisor-deployment-sha256", required=True)
    parser.add_argument("--expected-posteval-supervisor-deployment-sha256", required=True)
    parser.add_argument("--expected-training-launch-receipt-sha256", required=True)
    parser.add_argument("--expected-final-gate-sha256", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-evaluation-revision", required=True)
    parser.add_argument("--expected-evaluation-tree", required=True)
    parser.add_argument("--expected-evaluation-branch", required=True)
    parser.add_argument("--expected-export-revision", required=True)
    parser.add_argument("--expected-export-branch", required=True)
    parser.add_argument("--real-dir", type=Path, default=FORMAL_REAL_DIR)
    parser.add_argument("--official-related", type=Path, default=OFFICIAL_RELATED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    training_project = args.training_project.resolve()
    formal_project = args.formal_project.resolve()
    output_root = args.output_root.resolve()
    paths = generation_capacity_full_workspace_paths(output_root=output_root)
    full_root = paths["CAPACITY_FULL_ROOT"]
    report_root = paths["CAPACITY_FULL_REPORT_ROOT"]
    cofitok_run = paths["CAPACITY_FULL_COFITOK_RUN"]
    dense_run = paths["CAPACITY_FULL_DENSE_RUN"]
    export_root = paths["CAPACITY_FULL_EXPORT_ROOT"]
    expectations = {
        "training_revision": _revision(
            args.expected_training_revision,
            name="training revision",
        ),
        "training_tree": _revision(
            args.expected_training_tree,
            name="training tree",
        ),
        "training_branch": args.expected_training_branch,
        "evaluation_revision": _revision(
            args.expected_evaluation_revision,
            name="evaluation revision",
        ),
        "evaluation_tree": _revision(
            args.expected_evaluation_tree,
            name="evaluation tree",
        ),
        "evaluation_branch": args.expected_evaluation_branch,
        "export_revision": _revision(
            args.expected_export_revision,
            name="export revision",
        ),
        "export_branch": args.expected_export_branch,
        "training_supervisor_deployment_sha256": _sha256(
            args.expected_training_supervisor_deployment_sha256,
            name="training supervisor deployment SHA256",
        ),
        "posteval_supervisor_deployment_sha256": _sha256(
            args.expected_posteval_supervisor_deployment_sha256,
            name="posteval supervisor deployment SHA256",
        ),
        "training_launch_receipt_sha256": _sha256(
            args.expected_training_launch_receipt_sha256,
            name="training launch receipt SHA256",
        ),
        "final_gate_sha256": _sha256(
            args.expected_final_gate_sha256,
            name="final gate SHA256",
        ),
    }
    training_git = {
        "revision": expectations["training_revision"],
        "tree": expectations["training_tree"],
        "branch": expectations["training_branch"],
        "tracked_dirty": False,
    }
    evaluation_git = {
        "revision": expectations["evaluation_revision"],
        "tree": expectations["evaluation_tree"],
        "branch": expectations["evaluation_branch"],
        "tracked_dirty": False,
    }

    training_deployment = _read_optional(args.training_supervisor_deployment)
    training_status = _read_optional(args.training_supervisor_status)
    posteval_deployment = _read_optional(args.posteval_supervisor_deployment)
    posteval_status = _read_optional(args.posteval_supervisor_status)
    receipt_path = paths["CAPACITY_FULL_TRAINING_AUTHORIZATION"]
    receipt = _read_optional(receipt_path)
    pair_monitor = _read_optional(paths["CAPACITY_FULL_MONITOR"])
    training_reports = {
        "cofitok": _read_optional(cofitok_run / "training_report.json"),
        "dense_identity": _read_optional(dense_run / "training_report.json"),
    }
    training_audits = {
        "cofitok": _read_optional(report_root / "cofitok_training_audit.json"),
        "dense_identity": _read_optional(
            report_root / "dense_identity_training_audit.json"
        ),
    }
    checkpoint_files = {
        "cofitok": _verify_checkpoint_file(
            cofitok_run / "checkpoint_step_00300000.pt"
        ),
        "dense_identity": _verify_checkpoint_file(
            dense_run / "checkpoint_step_00300000.pt"
        ),
    }
    milestones = {
        step: _read_optional(report_root / "milestones" / f"step_{step:08d}.json")
        for step in MILESTONE_STEPS
    }
    milestone_checkpoint_files = {
        step: {
            "cofitok": _verify_checkpoint_file(
                cofitok_run / f"checkpoint_step_{step:08d}.pt"
            ),
            "dense_identity": _verify_checkpoint_file(
                dense_run / f"checkpoint_step_{step:08d}.pt"
            ),
        }
        for step in MILESTONE_STEPS
    }
    final_gate_path = paths["CAPACITY_FULL_GATE"]
    final_gate = _read_optional(final_gate_path)
    comparison_path = report_root / "comparison/large_scale_generation_comparison.json"
    comparison = _read_optional(comparison_path)
    generation_reports = {
        "cofitok": _read_optional(
            cofitok_run
            / "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "dense_identity": _read_optional(
            dense_run
            / "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
    }
    sampling_selection = _read_optional(report_root / "sampling_runtime_selection.json")
    visual_audit = _read_optional(report_root / "visual_audit/visual_audit_report.json")
    class_fidelity = _read_optional(
        report_root / "class_fidelity/qualification_report.json"
    )
    official_related_path = (
        args.official_related
        if args.official_related.is_absolute()
        else project / args.official_related
    ).resolve()
    official_related = _read_optional(official_related_path)
    export_reports = {
        "cofitok_export": _read_optional(report_root / "exports/cofitok_export_report.json"),
        "dense_identity_export": _read_optional(
            report_root / "exports/dense_export_report.json"
        ),
        "cofitok_preflight": _read_optional(
            report_root / "exports/cofitok_export_preflight.json"
        ),
        "dense_identity_preflight": _read_optional(
            report_root / "exports/dense_export_preflight.json"
        ),
        "cofitok_smoke": _read_optional(
            report_root / "exports/cofitok_export_inference_smoke.json"
        ),
        "dense_identity_smoke": _read_optional(
            report_root / "exports/dense_export_inference_smoke.json"
        ),
    }
    artifact_files = {
        "cofitok": _verify_inference_artifact_file(
            paths["CAPACITY_FULL_COFITOK_INFERENCE_ARTIFACT"]
        ),
        "dense_identity": _verify_inference_artifact_file(
            paths["CAPACITY_FULL_DENSE_INFERENCE_ARTIFACT"]
        ),
    }
    smoke_files = {
        "cofitok": _verify_inference_smoke_outputs(
            export_reports["cofitok_smoke"],
            expected_root=export_root / "smoke/cofitok",
        ),
        "dense_identity": _verify_inference_smoke_outputs(
            export_reports["dense_identity_smoke"],
            expected_root=export_root / "smoke/dense_identity",
        ),
    }

    training_deployment_evidence: dict[str, Any] | None = None
    training_completion_evidence: dict[str, Any] | None = None
    posteval_result_evidence: dict[str, Any] | None = None

    def training_deployment_check() -> dict[str, Any]:
        nonlocal training_deployment_evidence
        training_deployment_evidence = validate_training_supervisor_deployment(
            training_deployment,
            receipt_path=args.training_supervisor_deployment,
            expected_sha256=expectations["training_supervisor_deployment_sha256"],
            training_project=training_project,
            formal_project=formal_project,
            full_output_root=full_root,
            expected_training_git=training_git,
        )
        return training_deployment_evidence

    def training_completion_check() -> dict[str, Any]:
        nonlocal training_completion_evidence
        if file_sha256(receipt_path) != expectations["training_launch_receipt_sha256"]:
            raise ValueError("capacity-full training launch receipt SHA256 differs")
        training_completion_evidence = validate_training_completion(
            training_status,
            status_path=args.training_supervisor_status,
            training_receipt_path=receipt_path,
            cofitok_training_path=cofitok_run / "training_report.json",
            dense_training_path=dense_run / "training_report.json",
            pair_monitor_path=paths["CAPACITY_FULL_MONITOR"],
            expected_training_revision=expectations["training_revision"],
            expected_training_branch=expectations["training_branch"],
            expected_training_tree=expectations["training_tree"],
            full_output_root=full_root,
        )
        return training_completion_evidence

    def posteval_result_check() -> dict[str, Any]:
        nonlocal posteval_result_evidence
        posteval_result_evidence = validate_posteval_result(
            final_gate_path=final_gate_path,
            comparison_path=comparison_path,
            expected_training_revision=expectations["training_revision"],
            expected_training_branch=expectations["training_branch"],
            expected_evaluation_revision=expectations["evaluation_revision"],
            expected_evaluation_branch=expectations["evaluation_branch"],
        )
        if posteval_result_evidence["status"] != "pass":
            raise ValueError("capacity-full final quality gate did not pass")
        return posteval_result_evidence

    checks = [
        _check(
            "capacity_full_training_supervisor_deployment",
            [training_deployment],
            training_deployment_check,
        ),
        _check(
            "capacity_full_training_completion",
            [training_status, receipt, pair_monitor, *training_reports.values()],
            training_completion_check,
        ),
        _check(
            "capacity_full_training_integrity",
            [*training_reports.values(), *training_audits.values()],
            lambda: {
                "progress": progress_audit_evidence(
                    training_audits,
                    expected_steps=300_000,
                    required_checkpoint_steps=MILESTONE_STEPS,
                ),
                "checkpoints": checkpoint_evidence(
                    checkpoint_files,
                    training_reports,
                    expected_step=300_000,
                    expected_revision=expectations["training_revision"],
                    expected_branch=expectations["training_branch"],
                ),
            },
        ),
        _check(
            "capacity_full_milestones",
            list(milestones.values()),
            lambda: milestone_evidence(milestones, milestone_checkpoint_files),
        ),
        _check(
            "capacity_full_posteval_supervisor_deployment",
            [posteval_deployment, training_deployment],
            lambda: validate_posteval_supervisor_deployment(
                posteval_deployment,
                receipt_path=args.posteval_supervisor_deployment,
                expected_sha256=expectations[
                    "posteval_supervisor_deployment_sha256"
                ],
                evaluation_project=project,
                training_project=training_project,
                full_output_root=full_root,
                expected_evaluation_git=evaluation_git,
                expected_training_git=training_git,
                training_supervisor_deployment_identity=file_identity(
                    args.training_supervisor_deployment
                ),
            ),
        ),
        _check(
            "capacity_full_posteval_result",
            [final_gate, comparison],
            posteval_result_check,
        ),
        _check(
            "capacity_full_posteval_supervisor_status",
            [posteval_status, final_gate, comparison],
            lambda: validate_posteval_supervisor_status(
                posteval_status,
                status_path=args.posteval_supervisor_status,
                expected_evaluation_git=evaluation_git,
                expected_training_git=training_git,
                full_output_root=full_root,
                expected_result=validate_posteval_result(
                    final_gate_path=final_gate_path,
                    comparison_path=comparison_path,
                    expected_training_revision=expectations["training_revision"],
                    expected_training_branch=expectations["training_branch"],
                    expected_evaluation_revision=expectations[
                        "evaluation_revision"
                    ],
                    expected_evaluation_branch=expectations["evaluation_branch"],
                ),
            ),
        ),
        _check(
            "capacity_full_formal_generation",
            [*generation_reports.values(), *training_reports.values()],
            lambda: formal_generation_evidence(
                generation_reports,
                training_reports,
                {
                    "cofitok": _verify_formal_sample_files(
                        generation_reports["cofitok"],
                        expected_generated_dir=(
                            cofitok_run / "samples_50k_ddim250_cfg15/prefix_8"
                        ),
                    ),
                    "dense_identity": _verify_formal_sample_files(
                        generation_reports["dense_identity"],
                        expected_generated_dir=(
                            dense_run / "samples_50k_ddim250_cfg15/prefix_1"
                        ),
                    ),
                },
                _verify_formal_real_set_files(
                    generation_reports,
                    expected_real_dir=args.real_dir,
                ),
                expected_revision=expectations["evaluation_revision"],
                expected_branch=expectations["evaluation_branch"],
            ),
        ),
        _check(
            "capacity_full_runtime_and_visual",
            [sampling_selection, visual_audit, *generation_reports.values()],
            lambda: runtime_and_visual_evidence(
                sampling_selection,
                visual_audit,
                generation_reports,
                expected_revision=expectations["evaluation_revision"],
                expected_branch=expectations["evaluation_branch"],
                expected_visual_root=report_root / "visual_audit",
            ),
        ),
        _check(
            "capacity_full_final_gate",
            [final_gate],
            lambda: gate_evidence(
                final_gate,
                gate_path=final_gate_path,
                expected_sha256=expectations["final_gate_sha256"],
                stage="full",
                source_profile="capacity_full",
                training_revision=expectations["training_revision"],
                training_branch=expectations["training_branch"],
                evaluation_revision=expectations["evaluation_revision"],
                evaluation_branch=expectations["evaluation_branch"],
            ),
        ),
        _check(
            "capacity_full_strong_comparison",
            [
                comparison,
                official_related,
                class_fidelity,
                final_gate,
                pair_monitor,
                *generation_reports.values(),
                *training_reports.values(),
            ],
            lambda: _comparison_evidence(
                comparison,
                generation_reports,
                training_reports,
                pair_monitor,
                official_related,
                file_sha256(official_related_path),
                verify_comparison_source_reports(comparison),
                final_gate=final_gate,
                class_fidelity_qualification=class_fidelity,
            ),
        ),
        _check(
            RELEASE_CHECK,
            [
                *export_reports.values(),
                *generation_reports.values(),
                *training_reports.values(),
                final_gate,
            ],
            lambda: _inference_export_evidence(
                export_reports,
                artifact_files,
                smoke_files,
                generation_reports,
                training_reports,
                final_gate,
                expected_export_revision=expectations["export_revision"],
                expected_export_branch=expectations["export_branch"],
            ),
        ),
    ]
    audit = aggregate_checks(checks)
    audit["profile"] = PROFILE
    audit["expectations"] = expectations
    audit["claim_boundary"] = {
        "training_authorization_is_quality_promotion_gate": False,
        "final_gate_required_for_release": True,
        "release_receipt_requires_complete_audit": True,
    }
    write_json_report(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "pass" and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
