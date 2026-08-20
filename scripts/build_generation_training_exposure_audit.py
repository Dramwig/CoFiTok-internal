from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.reporting import file_sha256, write_json_report
from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_DATASET,
    QUALITY_BRIDGE_RECIPE_STAGE,
    QUALITY_BRIDGE_RESULT_ROLE,
    QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
    QUALITY_BRIDGE_STEPS,
    RESULT_AUTHORIZATION_BOUNDARY,
    build_quality_bridge_result,
    validate_quality_bridge_preparation,
)
from cofitok.training_exposure import (
    TRAINING_EXPOSURE_SCHEMA_VERSION,
    compare_training_exposures,
    training_exposure_summary,
)
from scripts.build_generation_milestone_report import (
    validate_milestone_report,
    verify_milestone_source_reports,
)


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_training_exposure_audit"
TERMINAL_EXECUTION_DETAIL = (
    "quality bridge terminal evidence verified; no larger-training authorization "
    "was created"
)


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _normalized_git_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} lacks Git identity")
    revision = str(value.get("revision", ""))
    branch = str(value.get("branch", ""))
    dirty = value.get("dirty", value.get("tracked_dirty"))
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not branch
        or dirty is not False
    ):
        raise ValueError(f"{label} Git identity is invalid or dirty")
    return {"revision": revision, "branch": branch, "tracked_dirty": False}


def _identity_content(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} source identity is missing")
    path = str(value.get("path", ""))
    size = value.get("bytes")
    digest = str(value.get("sha256", ""))
    if (
        not (Path(path).is_absolute() or PurePosixPath(path).is_absolute())
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} source identity is invalid")
    return {"bytes": size, "sha256": digest}


def _read_terminal_source(
    source: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    path = Path(str(source["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _source_identity(path)
    if actual != source:
        raise ValueError(f"quality bridge terminal source changed: {label}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"quality bridge terminal source is not an object: {label}")
    return payload


def _replay_quality_bridge_terminal_result(
    terminal: dict[str, Any],
) -> dict[str, Any]:
    sources = terminal["source_reports"]
    payloads = {
        name: _read_terminal_source(source, label=name)
        for name, source in sources.items()
    }
    milestones = terminal.get("milestones")
    physical = terminal.get("terminal", {}).get("physical_evidence")
    if not isinstance(milestones, dict) or set(milestones) != {"50000", "100000"}:
        raise ValueError("quality bridge terminal milestone evidence differs")
    if not isinstance(physical, dict):
        raise ValueError("quality bridge terminal physical evidence is missing")
    git = _normalized_git_identity(
        terminal.get("git"), label="quality bridge terminal replay"
    )
    return build_quality_bridge_result(
        preparation=payloads["preparation"],
        launch_receipt=payloads["launch_receipt"],
        training_pair_validation=payloads["training_pair_validation"],
        cofitok_training_audit=payloads["cofitok_training_audit"],
        dense_training_audit=payloads["dense_training_audit"],
        milestone_evidence={
            50_000: milestones["50000"],
            100_000: milestones["100000"],
        },
        cofitok_sampling_preflight=payloads["cofitok_sampling_preflight"],
        dense_sampling_preflight=payloads["dense_sampling_preflight"],
        cofitok_generation=payloads["cofitok_generation"],
        dense_generation=payloads["dense_generation"],
        cofitok_checkpoint_eval=payloads["cofitok_checkpoint_eval"],
        dense_checkpoint_eval=payloads["dense_checkpoint_eval"],
        class_fidelity_qualification=payloads["class_fidelity_qualification"],
        physical_evidence=physical,
        source_identities=sources,
        expected_revision=git["revision"],
        expected_branch=git["branch"],
    )


def _bind_quality_bridge_terminal_result(
    report: dict[str, Any],
    *,
    training_reports: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    terminal_result: tuple[dict[str, Any], dict[str, Any]],
    execution_status: tuple[dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    if set(training_reports) != {"cofitok", "dense_identity"}:
        raise ValueError(
            "terminal-bound exposure requires cofitok and dense_identity labels"
        )
    milestone = report.get("milestone_binding")
    if (
        not isinstance(milestone, dict)
        or int(milestone.get("expected_step", -1)) != QUALITY_BRIDGE_STEPS
        or milestone.get("source_profile") != "quality_bridge"
        or milestone.get("matched_training_exposure_verified") is not True
    ):
        raise ValueError("terminal exposure lacks the exact 100K milestone binding")

    terminal, terminal_identity = terminal_result
    if (
        terminal.get("schema_version") != QUALITY_BRIDGE_RESULT_SCHEMA_VERSION
        or terminal.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or terminal.get("status") != "completed"
        or terminal.get("stage") != QUALITY_BRIDGE_RECIPE_STAGE
        or terminal.get("authorization_boundary") != RESULT_AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("quality bridge terminal result contract differs")
    terminal_git = _normalized_git_identity(
        terminal.get("git"), label="quality bridge terminal result"
    )
    if terminal_git != milestone.get("training_git"):
        raise ValueError("terminal result and training Git identities differ")

    pair = terminal.get("training", {}).get("pair_validation")
    if (
        not isinstance(pair, dict)
        or pair.get("status") != "pass"
        or int(pair.get("expected_steps", -1)) != QUALITY_BRIDGE_STEPS
        or pair.get("expected_dataset") != QUALITY_BRIDGE_DATASET
        or pair.get("expected_revision") != terminal_git["revision"]
        or pair.get("expected_branch") != terminal_git["branch"]
        or pair.get("training_recipe", {}).get("stage") != QUALITY_BRIDGE_RECIPE_STAGE
        or pair.get("training_recipe", {}).get("valid") is not True
    ):
        raise ValueError("terminal result matched training validation differs")

    sources = terminal.get("source_reports")
    methods = terminal.get("terminal", {}).get("methods")
    if not isinstance(sources, dict) or not isinstance(methods, dict):
        raise ValueError("quality bridge terminal result evidence is incomplete")
    expected_terminal_sources = {
        "preparation",
        "launch_receipt",
        "cofitok_training",
        "dense_training",
        "training_pair_validation",
        "cofitok_training_audit",
        "dense_training_audit",
        "milestone_50000",
        "milestone_100000",
        "cofitok_sampling_preflight",
        "dense_sampling_preflight",
        "cofitok_generation",
        "dense_generation",
        "cofitok_checkpoint_eval",
        "dense_checkpoint_eval",
        "class_fidelity_qualification",
        "cofitok_class_fidelity",
        "dense_class_fidelity",
    }
    if set(sources) != expected_terminal_sources:
        raise ValueError("quality bridge terminal result source set differs")
    for name, source in sources.items():
        _identity_content(source, label=f"terminal {name}")
    _identity_content(terminal_identity, label="snapshotted terminal result")
    if _replay_quality_bridge_terminal_result(terminal) != terminal:
        raise ValueError("quality bridge terminal result is not logically reproducible")
    training_source_binding: dict[str, Any] = {}
    checkpoint_binding: dict[str, Any] = {}
    milestone_checkpoints = milestone.get("training_checkpoint_binding", {})
    for method, source_name in (
        ("cofitok", "cofitok_training"),
        ("dense_identity", "dense_training"),
    ):
        training, snapshot_identity = training_reports[method]
        output_dir = str(training.get("output_dir", ""))
        expected_source_path = (
            PurePosixPath(output_dir) / "training_report.json" if output_dir else None
        )
        source = sources.get(source_name)
        if (
            expected_source_path is None
            or not isinstance(source, dict)
            or source.get("path") != expected_source_path.as_posix()
            or _identity_content(source, label=f"terminal {method} training")
            != _identity_content(
                snapshot_identity,
                label=f"snapshotted {method} training",
            )
        ):
            raise ValueError("terminal result uses another training report")
        method_row = methods.get(method)
        milestone_checkpoint = milestone_checkpoints.get(method)
        latest = training.get("latest_checkpoint")
        if not all(
            isinstance(value, dict)
            for value in (method_row, milestone_checkpoint, latest)
        ):
            raise ValueError("terminal checkpoint binding is incomplete")
        checkpoint_sha256 = str(latest.get("checkpoint_sha256", ""))
        if (
            int(latest.get("step", -1)) != QUALITY_BRIDGE_STEPS
            or int(method_row.get("checkpoint_step", -1)) != QUALITY_BRIDGE_STEPS
            or int(milestone_checkpoint.get("step", -1)) != QUALITY_BRIDGE_STEPS
            or method_row.get("checkpoint_sha256") != checkpoint_sha256
            or milestone_checkpoint.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise ValueError("terminal and training checkpoints differ")
        training_source_binding[method] = dict(source)
        checkpoint_binding[method] = {
            "step": QUALITY_BRIDGE_STEPS,
            "checkpoint_sha256": checkpoint_sha256,
        }

    milestone_source = sources.get("milestone_100000")
    preparation_source = sources.get("preparation")
    if _identity_content(
        milestone_source, label="terminal 100K milestone"
    ) != _identity_content(
        milestone.get("source"), label="snapshotted 100K milestone"
    ) or _identity_content(
        preparation_source, label="terminal preparation"
    ) != _identity_content(
        report.get("quality_bridge_plan", {}).get("source"),
        label="snapshotted preparation",
    ):
        raise ValueError("terminal result uses another milestone or preparation")

    execution, execution_identity = execution_status
    _identity_content(
        execution_identity,
        label="snapshotted quality bridge execution status",
    )
    if (
        execution.get("schema_version") != 1
        or execution.get("role") != "stability_full_data_quality_bridge_execution"
        or execution.get("status") != "completed"
        or execution.get("detail") != TERMINAL_EXECUTION_DETAIL
        or execution.get("exit_code") is not None
        or execution.get("quality_bridge_only") is not True
        or execution.get("full_training_launch_allowed") is not False
        or execution.get("full_300k_launch_allowed") is not False
        or execution.get("report_is_promotion_gate") is not False
        or _normalized_git_identity(
            execution.get("git"), label="quality bridge execution status"
        )
        != terminal_git
    ):
        raise ValueError("quality bridge verified execution status differs")

    quality_screen = terminal.get("quality_screen")
    if not isinstance(quality_screen, dict) or not quality_screen.get("status"):
        raise ValueError("quality bridge terminal result lacks a quality screen")
    return {
        "terminal_result": dict(terminal_identity),
        "verified_execution_status": dict(execution_identity),
        "training_source_binding": training_source_binding,
        "milestone_source_binding": dict(milestone_source),
        "preparation_source_binding": dict(preparation_source),
        "checkpoint_binding": checkpoint_binding,
        "training_git": terminal_git,
        "quality_screen": dict(quality_screen),
        "terminal_result_binding_verified": True,
        "active_runbook_verification_completed": True,
    }


def parse_training_spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError("training input must use LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


def build_report(
    training_reports: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    quality_bridge_preparation: tuple[dict[str, Any], dict[str, Any]] | None = None,
    milestone_report: tuple[dict[str, Any], dict[str, Any]] | None = None,
    expected_milestone_step: int | None = None,
    milestone_source_profile: str | None = None,
    quality_bridge_terminal_result: tuple[dict[str, Any], dict[str, Any]] | None = None,
    quality_bridge_execution_status: tuple[dict[str, Any], dict[str, Any]]
    | None = None,
) -> dict[str, Any]:
    if not training_reports:
        raise ValueError("at least one training report is required")
    rows: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for label, (report, identity) in training_reports.items():
        if not label or label in rows:
            raise ValueError("training report labels must be unique and non-empty")
        rows[label] = training_exposure_summary(report)
        sources[label] = dict(identity)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "exposure_schema_version": TRAINING_EXPOSURE_SCHEMA_VERSION,
        "status": "pass",
        "role": REPORT_ROLE,
        "sources": sources,
        "rows": rows,
        "comparison": compare_training_exposures(rows),
        "claim_boundary": {
            "training_scale_claim_allowed": True,
            "sample_quality_claim_allowed": False,
            "method_quality_ranking_allowed": False,
            "formal_gate_substitute": False,
        },
    }
    if quality_bridge_preparation is not None:
        preparation, identity = quality_bridge_preparation
        validated = validate_quality_bridge_preparation(preparation)
        report["quality_bridge_plan"] = {
            "source": dict(identity),
            "training_exposure": validated["training_exposure"],
        }
    if milestone_report is not None:
        if expected_milestone_step is None or milestone_source_profile is None:
            raise ValueError("milestone step and source profile are required")
        if set(training_reports) != {"cofitok", "dense_identity"}:
            raise ValueError(
                "milestone-bound exposure requires cofitok and dense_identity labels"
            )
        milestone, identity = milestone_report
        source_verification = verify_milestone_source_reports(
            milestone,
            source_profile=milestone_source_profile,
        )
        evidence, warnings = validate_milestone_report(
            milestone,
            expected_step=expected_milestone_step,
            source_verification=source_verification,
            expected_source_profile=milestone_source_profile,
        )
        checkpoint_binding: dict[str, Any] = {}
        training_git_identities: dict[str, dict[str, Any]] = {}
        target_steps: set[int] = set()
        completion_states: set[bool] = set()
        for method in ("cofitok", "dense_identity"):
            training = training_reports[method][0]
            exposure = rows[method]
            if int(exposure["completed_steps"]) != expected_milestone_step:
                raise ValueError("training exposure step differs from milestone")
            latest = training.get("latest_checkpoint")
            milestone_method = milestone.get("methods", {}).get(method)
            if not isinstance(latest, dict) or not isinstance(milestone_method, dict):
                raise ValueError("milestone checkpoint binding is incomplete")
            if (
                int(latest.get("step", -1)) != expected_milestone_step
                or int(milestone_method.get("checkpoint_step", -1))
                != expected_milestone_step
                or latest.get("checkpoint_sha256")
                != milestone_method.get("checkpoint_sha256")
            ):
                raise ValueError("training and milestone checkpoints differ")
            checkpoint_binding[method] = {
                "step": expected_milestone_step,
                "checkpoint_sha256": latest["checkpoint_sha256"],
                "training_report": dict(training_reports[method][1]),
            }
            training_git_identities[method] = _normalized_git_identity(
                exposure.get("git"),
                label=f"{method} training exposure",
            )
            target_steps.add(int(exposure["target_steps"]))
            completion_states.add(bool(exposure["training_complete"]))
        if (
            len({tuple(value.items()) for value in training_git_identities.values()})
            != 1
        ):
            raise ValueError("milestone training Git identities differ or are dirty")
        if len(target_steps) != 1 or len(completion_states) != 1:
            raise ValueError("milestone training horizons or completion states differ")
        required_comparison = {
            "same_dataset_identity",
            "same_effective_batch_size",
            "same_completed_steps",
            "same_images_seen",
            "same_dataset_normalized_exposure",
            "step_budget_directly_comparable",
            "image_budget_directly_comparable",
            "dataset_normalized_budget_directly_comparable",
        }
        if any(
            report["comparison"].get(name) is not True for name in required_comparison
        ):
            raise ValueError("milestone training exposure is not exactly matched")
        expected_git = training_git_identities["cofitok"]
        source_report_git: dict[str, Any] = {}
        for name, source in source_verification["source_reports"].items():
            source_payload = json.loads(
                Path(source["path"]).read_text(encoding="utf-8")
            )
            observed_git = _normalized_git_identity(
                source_payload.get("git"),
                label=f"milestone source report {name}",
            )
            if observed_git != expected_git:
                raise ValueError("milestone source and training Git identities differ")
            source_report_git[name] = observed_git
        report["milestone_binding"] = {
            "source": dict(identity),
            "expected_step": expected_milestone_step,
            "source_profile": milestone_source_profile,
            "verification": evidence,
            "warnings": warnings,
            "training_checkpoint_binding": checkpoint_binding,
            "training_git": expected_git,
            "source_report_git": source_report_git,
            "matched_training_exposure_verified": True,
        }
        report["claim_boundary"].update(
            {
                "milestone_quality_diagnostic_allowed": True,
                "formal_generation_claim_allowed": False,
            }
        )
    elif expected_milestone_step is not None or milestone_source_profile is not None:
        raise ValueError(
            "milestone step/profile were provided without a milestone report"
        )
    if (quality_bridge_terminal_result is None) != (
        quality_bridge_execution_status is None
    ):
        raise ValueError("terminal result and verified execution status must be paired")
    if quality_bridge_terminal_result is not None:
        report["terminal_binding"] = _bind_quality_bridge_terminal_result(
            report,
            training_reports=training_reports,
            terminal_result=quality_bridge_terminal_result,
            execution_status=quality_bridge_execution_status,
        )
        report["claim_boundary"].update(
            {
                "terminal_training_exposure_binding_allowed": True,
                "terminal_quality_result_context_allowed": True,
                "formal_generation_claim_allowed": False,
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable source-bound training report",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-bridge-preparation", type=Path)
    parser.add_argument("--milestone-report", type=Path)
    parser.add_argument("--expected-milestone-step", type=int)
    parser.add_argument(
        "--milestone-source-profile",
        choices=("full", "stability_full", "quality_bridge"),
    )
    parser.add_argument("--quality-bridge-terminal-result", type=Path)
    parser.add_argument("--quality-bridge-execution-status", type=Path)
    args = parser.parse_args()

    reports: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for raw in args.training:
        label, path = parse_training_spec(raw)
        if label in reports:
            raise ValueError(f"duplicate training label: {label}")
        if not path.is_file():
            raise FileNotFoundError(path)
        reports[label] = (
            json.loads(path.read_text(encoding="utf-8")),
            _source_identity(path),
        )
    preparation = None
    if args.quality_bridge_preparation is not None:
        path = args.quality_bridge_preparation
        if not path.is_file():
            raise FileNotFoundError(path)
        preparation = (
            json.loads(path.read_text(encoding="utf-8")),
            _source_identity(path),
        )
    milestone = None
    if args.milestone_report is not None:
        path = args.milestone_report
        if not path.is_file():
            raise FileNotFoundError(path)
        milestone = (
            json.loads(path.read_text(encoding="utf-8")),
            _source_identity(path),
        )
    terminal_result = None
    if args.quality_bridge_terminal_result is not None:
        path = args.quality_bridge_terminal_result
        if not path.is_file():
            raise FileNotFoundError(path)
        terminal_result = (
            json.loads(path.read_text(encoding="utf-8")),
            _source_identity(path),
        )
    execution_status = None
    if args.quality_bridge_execution_status is not None:
        path = args.quality_bridge_execution_status
        if not path.is_file():
            raise FileNotFoundError(path)
        execution_status = (
            json.loads(path.read_text(encoding="utf-8")),
            _source_identity(path),
        )
    write_json_report(
        args.output,
        build_report(
            reports,
            quality_bridge_preparation=preparation,
            milestone_report=milestone,
            expected_milestone_step=args.expected_milestone_step,
            milestone_source_profile=args.milestone_source_profile,
            quality_bridge_terminal_result=terminal_result,
            quality_bridge_execution_status=execution_status,
        ),
    )


if __name__ == "__main__":
    main()
