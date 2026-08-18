from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_posttraining_sampling import (
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_OUTPUT_ROOT,
    RUNBOOK_RELATIVE_PATH,
    RUN_NAMES,
    build_sampling_preparation,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance
from cofitok.training.checkpointing import verify_training_checkpoint
from scripts import (
    build_generation_conditioning_ranking_training_confirmation_posteval as heldout,
)
from scripts.build_generation_conditioning_ranking_probe_posteval import (
    _load_sensitivity_source,
)
from scripts.evaluate_generation_class_fidelity import classifier_identity


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bound_json(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def _embedded_identity(
    descriptor: object,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} identity is missing")
    source = reject_symlink_chain(
        Path(str(descriptor.get("path", ""))),
        name=label,
    ).resolve()
    identity = file_identity(source)
    if identity != descriptor:
        raise ValueError(f"{label} identity differs")
    return source, identity


def _embedded_json(
    descriptor: object,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source, identity = _embedded_identity(descriptor, label=label)
    return read_json_object(source, name=label), identity


def replay_heldout_evaluation(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name="5K heldout evaluation").resolve()
    report_identity = file_identity(source)
    report = read_json_object(source, name="5K heldout evaluation")
    if (
        report.get("schema_version") != heldout.SCHEMA_VERSION
        or report.get("role") != heldout.ROLE
        or report.get("status") != "completed"
        or report.get("stage") != heldout.STAGE
    ):
        raise ValueError("5K heldout evaluation is not completed evidence")
    sources = report.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("5K heldout evaluation sources are missing")

    preparation, preparation_identity = _embedded_json(
        sources.get("preparation"),
        label="5K training preparation",
    )
    training_status, training_status_identity = _embedded_json(
        sources.get("training_status"),
        label="5K training status",
    )
    configs: dict[str, dict[str, Any]] = {}
    config_identities: dict[str, dict[str, Any]] = {}
    training_reports: dict[str, dict[str, Any]] = {}
    training_report_identities: dict[str, dict[str, Any]] = {}
    training_audits: dict[str, dict[str, Any]] = {}
    training_audit_identities: dict[str, dict[str, Any]] = {}
    checkpoint_identities: dict[str, dict[str, Any]] = {}
    integrity_identities: dict[str, dict[str, Any]] = {}
    sensitivity_reports: dict[str, dict[str, Any]] = {}
    sensitivity_identities: dict[str, dict[str, Any]] = {}

    for run in RUN_NAMES:
        config_path, config_identity = _embedded_identity(
            sources.get("configs", {}).get(run),
            label=f"{run} config",
        )
        configs[run] = config_to_dict(load_config(config_path))
        config_identities[run] = config_identity
        training_reports[run], training_report_identities[run] = _embedded_json(
            sources.get("training_reports", {}).get(run),
            label=f"{run} training report",
        )
        training_audits[run], training_audit_identities[run] = _embedded_json(
            sources.get("training_audits", {}).get(run),
            label=f"{run} training audit",
        )
        _, checkpoint_identities[run] = _embedded_identity(
            sources.get("checkpoints", {}).get(run),
            label=f"{run} terminal checkpoint",
        )
        _, integrity_identities[run] = _embedded_identity(
            sources.get("checkpoint_integrity_manifests", {}).get(run),
            label=f"{run} checkpoint integrity manifest",
        )
        sensitivity_descriptor = sources.get("sensitivity", {}).get(run)
        if not isinstance(sensitivity_descriptor, Mapping):
            raise ValueError(f"{run} sensitivity source is missing")
        report_descriptor = sensitivity_descriptor.get("report")
        if not isinstance(report_descriptor, Mapping):
            raise ValueError(f"{run} sensitivity report identity is missing")
        sensitivity_report, sensitivity_identity = _load_sensitivity_source(
            str(report_descriptor.get("path", "")),
            run=run,
        )
        if sensitivity_identity != sensitivity_descriptor:
            raise ValueError(f"{run} sensitivity identities differ")
        sensitivity_reports[run] = sensitivity_report
        sensitivity_identities[run] = sensitivity_identity

    rebuilt_sources = {
        "preparation": preparation_identity,
        "training_status": training_status_identity,
        "configs": config_identities,
        "training_reports": training_report_identities,
        "training_audits": training_audit_identities,
        "checkpoints": checkpoint_identities,
        "checkpoint_integrity_manifests": integrity_identities,
        "sensitivity": sensitivity_identities,
    }
    rebuilt = heldout.build_heldout_evaluation(
        preparation=preparation,
        training_status=training_status,
        training_reports=training_reports,
        training_report_sources=training_report_identities,
        training_audits=training_audits,
        training_audit_sources=training_audit_identities,
        checkpoint_sources=checkpoint_identities,
        checkpoint_integrity_sources=integrity_identities,
        expected_configs=configs,
        config_identities=config_identities,
        sensitivity_reports=sensitivity_reports,
        sources=rebuilt_sources,
        evaluator_git=report.get("evaluator_git", {}),
    )
    if rebuilt != report:
        raise ValueError("5K heldout evaluation replay differs")
    return report, report_identity


def validate_training_checkpoints(
    heldout_report: Mapping[str, Any],
    training_status: Mapping[str, Any],
) -> dict[str, Any]:
    sources = heldout_report.get("sources")
    contract = heldout_report.get("training_contract")
    status_runs = training_status.get("runs")
    if (
        not isinstance(sources, Mapping)
        or not isinstance(contract, Mapping)
        or not isinstance(contract.get("runs"), Mapping)
        or not isinstance(status_runs, Mapping)
    ):
        raise ValueError("5K checkpoint source contract is missing")
    result: dict[str, Any] = {}
    for run in RUN_NAMES:
        checkpoint_path, checkpoint_identity = _embedded_identity(
            sources.get("checkpoints", {}).get(run),
            label=f"{run} terminal checkpoint",
        )
        integrity_path, integrity_identity = _embedded_identity(
            sources.get("checkpoint_integrity_manifests", {}).get(run),
            label=f"{run} checkpoint integrity manifest",
        )
        training_report_path, training_report_identity = _embedded_identity(
            sources.get("training_reports", {}).get(run),
            label=f"{run} training report",
        )
        verified = verify_training_checkpoint(checkpoint_path)
        run_contract = contract["runs"].get(run)
        status_run = status_runs.get(run)
        if not isinstance(run_contract, Mapping) or not isinstance(status_run, Mapping):
            raise ValueError(f"{run} 5K checkpoint contract is missing")
        if (
            int(verified.get("step", -1)) != EXPECTED_CHECKPOINT_STEP
            or int(verified.get("checkpoint_bytes", -1))
            != checkpoint_identity["bytes"]
            or verified.get("checkpoint_sha256") != checkpoint_identity["sha256"]
            or verified.get("git_revision") != heldout_report["training_git"]["revision"]
            or verified.get("git_branch") != heldout_report["training_git"]["branch"]
            or verified.get("git_dirty") is not False
            or integrity_path != checkpoint_path.with_name(
                f"{checkpoint_path.name}.integrity.json"
            )
            or checkpoint_identity != status_run.get("checkpoint")
            or integrity_identity != status_run.get("integrity_manifest")
            or training_report_identity != status_run.get("training_report")
            or checkpoint_identity["sha256"] != run_contract.get("checkpoint_sha256")
            or checkpoint_identity["bytes"] != run_contract.get("checkpoint_bytes")
            or integrity_identity
            != run_contract.get("checkpoint_integrity_manifest")
            or training_report_path
            != checkpoint_path.parent / "training_report.json"
        ):
            raise ValueError(f"{run} physical 5K checkpoint differs")
        result[run] = {
            "checkpoint": checkpoint_identity,
            "integrity_manifest": integrity_identity,
            "training_report": training_report_identity,
            "step": EXPECTED_CHECKPOINT_STEP,
        }
    return result


def replay_sampling_preparation(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(
        path,
        name="posttraining sampling preparation",
    ).resolve()
    preparation_identity = file_identity(source)
    preparation = read_json_object(source, name="posttraining sampling preparation")
    sources = preparation.get("source_reports")
    if not isinstance(sources, Mapping):
        raise ValueError("posttraining sampling preparation sources are missing")
    heldout_report, heldout_identity = replay_heldout_evaluation(
        str(sources.get("heldout_evaluation", {}).get("path", ""))
    )
    if heldout_identity != sources.get("heldout_evaluation"):
        raise ValueError("posttraining heldout evaluation identity differs")
    training_status, training_status_identity = _embedded_json(
        sources.get("training_status"),
        label="5K training status",
    )
    training_receipt, training_receipt_identity = _embedded_json(
        sources.get("training_execution_receipt"),
        label="5K training execution receipt",
    )
    standing, standing_identity = _embedded_json(
        sources.get("standing_authorization"),
        label="standing experiment authorization",
    )
    runbook_path, runbook_identity = _embedded_identity(
        sources.get("runbook"),
        label="posttraining sampling runbook",
    )
    if runbook_path != (PROJECT_ROOT / RUNBOOK_RELATIVE_PATH).resolve():
        raise ValueError("posttraining sampling runbook path differs")
    checkpoints = validate_training_checkpoints(heldout_report, training_status)
    if checkpoints != sources.get("training_checkpoints"):
        raise ValueError("posttraining checkpoint evidence differs")
    classifier = classifier_identity(preparation.get("classifier", {}).get("weights_path", ""))
    expected = build_sampling_preparation(
        heldout_evaluation=heldout_report,
        heldout_evaluation_identity=heldout_identity,
        training_status=training_status,
        training_status_identity=training_status_identity,
        training_execution_receipt=training_receipt,
        training_execution_receipt_identity=training_receipt_identity,
        checkpoint_evidence=checkpoints,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        classifier=classifier,
        runbook_identity=runbook_identity,
        builder_git=git_provenance(PROJECT_ROOT),
        expected_revision=preparation.get("git", {}).get("revision", ""),
        expected_branch=preparation.get("git", {}).get("branch", ""),
        expected_output_root=preparation.get("output_root", ""),
    )
    if expected != preparation:
        raise ValueError("posttraining sampling preparation replay differs")
    return preparation, preparation_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the source-bound independent-stream four-arm 5K sampling "
            "confirmation after the exact fresh-training heldout pass."
        )
    )
    parser.add_argument("--heldout-evaluation", type=Path, required=True)
    parser.add_argument("--expected-heldout-evaluation-sha256", required=True)
    parser.add_argument("--training-status", type=Path, required=True)
    parser.add_argument("--expected-training-status-sha256", required=True)
    parser.add_argument("--training-execution-receipt", type=Path, required=True)
    parser.add_argument("--expected-training-execution-receipt-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output-root", default=EXPECTED_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_report, heldout_identity = _bound_json(
        args.heldout_evaluation,
        expected_sha256=args.expected_heldout_evaluation_sha256,
        label="5K heldout evaluation",
    )
    replayed_heldout, replayed_identity = replay_heldout_evaluation(
        args.heldout_evaluation
    )
    if replayed_heldout != heldout_report or replayed_identity != heldout_identity:
        raise ValueError("5K heldout evaluation replay identity differs")
    training_status, training_status_identity = _bound_json(
        args.training_status,
        expected_sha256=args.expected_training_status_sha256,
        label="5K training status",
    )
    training_receipt, training_receipt_identity = _bound_json(
        args.training_execution_receipt,
        expected_sha256=args.expected_training_execution_receipt_sha256,
        label="5K training execution receipt",
    )
    standing, standing_identity = _bound_json(
        args.standing_authorization,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
    checkpoints = validate_training_checkpoints(heldout_report, training_status)
    runbook = reject_symlink_chain(
        args.runbook,
        name="posttraining sampling runbook",
    ).resolve()
    if runbook != (PROJECT_ROOT / RUNBOOK_RELATIVE_PATH).resolve():
        raise ValueError("posttraining sampling runbook path differs")
    classifier = classifier_identity(args.classifier_checkpoint)
    report = build_sampling_preparation(
        heldout_evaluation=heldout_report,
        heldout_evaluation_identity=heldout_identity,
        training_status=training_status,
        training_status_identity=training_status_identity,
        training_execution_receipt=training_receipt,
        training_execution_receipt_identity=training_receipt_identity,
        checkpoint_evidence=checkpoints,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        classifier=classifier,
        runbook_identity=file_identity(runbook),
        builder_git=git_provenance(PROJECT_ROOT),
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_output_root=args.output_root,
    )
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(identity["sha256"])


if __name__ == "__main__":
    main()
