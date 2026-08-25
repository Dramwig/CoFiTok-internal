from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import OutputLockError, exclusive_output_lock

try:
    import build_generation_quality_bridge_statistical_claim_qualification as qualification_builder
    import build_generation_statistical_claim_language_guard as language_builder
    import build_generation_terminal_system_claim_guard as terminal_builder
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import (
        build_generation_quality_bridge_statistical_claim_qualification as qualification_builder,
    )
    from scripts import build_generation_statistical_claim_language_guard as language_builder
    from scripts import build_generation_terminal_system_claim_guard as terminal_builder


SCHEMA_VERSION = 1
ROLE = "generation_versioned_terminal_claim_chain"
QUALIFICATION_DIR_NAME = "statistical_claim_qualification_v2_authoritative"
LANGUAGE_DIR_NAME = "statistical_claim_language_guard_v2_authoritative"
TERMINAL_DIR_NAME = "terminal_system_claim_guard_v3_authoritative"
TERMINAL_WAITER_ROLE = "generation_terminal_system_claim_guard_waiter"
EXPOSURE_ROLE = "generation_training_exposure_audit"
FOLLOWUP_ROLE = "stability_quality_bridge_followup_experiment_decision"
RUNTIME_WAITER_ROLE = "generation_quality_bridge_runtime_claim_guard_waiter"

TERMINAL_WAITER_BOUNDARY = {
    "cpu_only_evidence_binding_allowed": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "promotion_or_release_allowed": False,
    "process_signals_allowed": False,
    "upstream_decisions_modified": False,
}
SCOPE = {
    "diagnostic_non_authorizing": True,
    "cpu_only_evidence_binding": True,
    "gpu_execution_allowed": False,
    "checkpoint_payload_loading_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "inference_export_authorization_allowed": False,
    "process_signals_allowed": False,
    "legacy_outputs_modified": False,
    "upstream_decisions_modified": False,
}
RUNTIME_POLICY = {
    "parent_pid": 1,
    "cuda_visible_devices": "-1",
    "omp_num_threads": "1",
    "mkl_num_threads": "1",
    "minimum_nice": 10,
    "ionice": "idle",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(project: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_identity(project: Path) -> dict[str, Any]:
    return {
        "revision": _git(project, "rev-parse", "HEAD"),
        "tree": _git(project, "rev-parse", "HEAD^{tree}"),
        "branch": _git(project, "branch", "--show-current"),
        "tracked_dirty": bool(
            _git(project, "status", "--porcelain", "--untracked-files=no")
        ),
    }


def _runtime_identity(*, require_detached: bool) -> dict[str, Any]:
    if not require_detached:
        return {
            "parent_pid": os.getppid(),
            "policy_enforced": False,
        }
    if os.name != "posix":
        raise ValueError("versioned terminal claim chain requires a Linux runtime")
    pid = os.getpid()
    parent_pid = os.getppid()
    proc = Path("/proc") / str(pid)
    stat_fields = proc.joinpath("stat").read_text(encoding="utf-8").split()
    ionice = subprocess.run(
        ["ionice", "-p", str(pid)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gpu_rows = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    gpu_pids = {
        int(row.split(",", 1)[0].strip())
        for row in gpu_rows
        if row.strip() and row.split(",", 1)[0].strip().isdigit()
    }
    runtime = {
        "pid": pid,
        "parent_pid": parent_pid,
        "start_ticks": int(stat_fields[21]),
        "hostname": socket.gethostname(),
        "cwd": os.readlink(proc / "cwd"),
        "cmdline": (
            proc.joinpath("cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", "replace")
            .strip()
        ),
        "python_executable": os.path.realpath(sys.executable),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "ionice": ionice,
        "gpu_processes": gpu_rows,
        "self_is_gpu_process": pid in gpu_pids,
        "policy_enforced": True,
    }
    if (
        parent_pid != RUNTIME_POLICY["parent_pid"]
        or runtime["cuda_visible_devices"] != RUNTIME_POLICY["cuda_visible_devices"]
        or runtime["omp_num_threads"] != RUNTIME_POLICY["omp_num_threads"]
        or runtime["mkl_num_threads"] != RUNTIME_POLICY["mkl_num_threads"]
        or runtime["nice"] < RUNTIME_POLICY["minimum_nice"]
        or ionice != RUNTIME_POLICY["ionice"]
        or runtime["gpu_processes"]
        or runtime["self_is_gpu_process"] is not False
    ):
        raise ValueError("versioned terminal claim chain runtime policy differs")
    return runtime


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    _require(identity["sha256"] == expected_sha256, f"{label} SHA256 differs")
    report = read_json_object(source, name=label)
    _require(file_identity(source) == identity, f"{label} changed during replay")
    return identity, report


def _source_identity(
    project: Path,
    name: str,
    expected_sha256: str,
) -> dict[str, Any]:
    identity = file_identity(project / "scripts" / name)
    _require(identity["sha256"] == expected_sha256, f"{name} SHA256 differs")
    return identity


def _validate_authoritative_sources(
    *,
    exposure_identity: Mapping[str, Any],
    exposure: Mapping[str, Any],
    followup_identity: Mapping[str, Any],
    followup: Mapping[str, Any],
    quality_identity: Mapping[str, Any],
    expected_training_git: Mapping[str, Any],
) -> None:
    terminal = exposure.get("terminal_binding")
    verification = (
        terminal.get("authoritative_terminal_verification")
        if isinstance(terminal, Mapping)
        else None
    )
    verified_result = (
        verification.get("verifier_output", {}).get("result")
        if isinstance(verification, Mapping)
        else None
    )
    _require(
        exposure.get("schema_version") == 1
        and exposure.get("role") == EXPOSURE_ROLE
        and exposure.get("status") == "pass"
        and isinstance(terminal, Mapping)
        and terminal.get("terminal_result_binding_verified") is True
        and terminal.get("training_git") == dict(expected_training_git)
        and isinstance(verification, Mapping)
        and verification.get("status") == "verified"
        and verification.get("quality_project") == dict(expected_training_git)
        and isinstance(verified_result, Mapping)
        and verified_result.get("sha256") == quality_identity["sha256"],
        "authoritative terminal exposure binding differs",
    )
    source_reports = followup.get("source_reports")
    exposure_source = (
        followup.get("training_exposure", {}).get("source_report")
        if isinstance(followup.get("training_exposure"), Mapping)
        else None
    )
    recommendation = followup.get("recommended_next_stage")
    boundary = followup.get("authorization_boundary")
    _require(
        followup.get("schema_version") == 2
        and followup.get("role") == FOLLOWUP_ROLE
        and followup.get("status") == "completed"
        and isinstance(source_reports, Mapping)
        and source_reports.get("terminal_training_exposure")
        == dict(exposure_identity)
        and exposure_source == dict(exposure_identity)
        and followup.get("quality_bridge_execution_git")
        == {
            "revision": expected_training_git["revision"],
            "branch": expected_training_git["branch"],
            "tracked_dirty": False,
        }
        and isinstance(recommendation, Mapping)
        and recommendation.get("execution_ready") is False
        and recommendation.get("gpu_execution_allowed") is False
        and recommendation.get("full_300k_launch_allowed") is False
        and isinstance(boundary, Mapping)
        and boundary.get("recommended_stage_execution_allowed") is False
        and boundary.get("full_training_launch_allowed") is False
        and boundary.get("full_300k_launch_allowed") is False
        and boundary.get("release_authorization_allowed") is False,
        "authoritative follow-up decision binding differs",
    )


def _validate_runtime_status(
    status: Mapping[str, Any],
    *,
    guard_identity: Mapping[str, Any],
) -> None:
    descriptor = status.get("runtime_claim_guard")
    _require(
        status.get("schema_version") == 1
        and status.get("role") == RUNTIME_WAITER_ROLE
        and status.get("status") == "pass"
        and status.get("phase") == "completed"
        and isinstance(descriptor, Mapping)
        and descriptor.get("identity") == dict(guard_identity),
        "runtime-strict waiter binding differs",
    )


def _prepare_exact(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return prepare_manifest(
        path,
        dict(payload),
        resume=path.is_file(),
        overwrite=False,
    )


def _output_paths(quality_root: Path) -> dict[str, Path]:
    reports = quality_root / "reports"
    qualification_dir = reports / QUALIFICATION_DIR_NAME
    language_dir = reports / LANGUAGE_DIR_NAME
    terminal_dir = reports / TERMINAL_DIR_NAME
    return {
        "qualification_dir": qualification_dir,
        "qualification": qualification_dir / "statistical_claim_qualification.json",
        "qualification_status": qualification_dir / "waiter_status.json",
        "language_dir": language_dir,
        "language": language_dir / "statistical_claim_language_guard.json",
        "language_status": language_dir / "waiter_status.json",
        "terminal_dir": terminal_dir,
        "terminal": terminal_dir / "terminal_system_claim_guard.json",
        "terminal_status": terminal_dir / "waiter_status.json",
        "deployment": terminal_dir / "deployment_receipt.json",
    }


def run_locked(args: argparse.Namespace, *, require_detached: bool = True) -> int:
    project = reject_symlink_chain(args.project, name="claim chain project").resolve()
    quality_root = reject_symlink_chain(
        args.quality_output_root,
        name="claim chain quality output root",
    ).resolve()
    terminal_uncertainty_root = reject_symlink_chain(
        args.terminal_uncertainty_output_root,
        name="claim chain terminal uncertainty root",
    ).resolve()
    quality_project = reject_symlink_chain(
        args.quality_project,
        name="claim chain authoritative quality project",
    ).resolve()
    runtime = _runtime_identity(require_detached=require_detached)
    git = _git_identity(project)
    expected_git = {
        "revision": args.expected_control_revision,
        "tree": args.expected_control_tree,
        "branch": args.expected_control_branch,
        "tracked_dirty": False,
    }
    _require(git == expected_git, "claim chain Git identity differs")
    expected_self_source = (project / "scripts" / Path(__file__).name).resolve()
    _require(
        Path(__file__).resolve() == expected_self_source,
        "claim chain executing source is outside the bound checkout",
    )
    sources = {
        "runner": _source_identity(
            project,
            Path(__file__).name,
            args.expected_runner_source_sha256,
        ),
        "qualification_builder": _source_identity(
            project,
            "build_generation_quality_bridge_statistical_claim_qualification.py",
            args.expected_qualification_builder_sha256,
        ),
        "language_builder": _source_identity(
            project,
            "build_generation_statistical_claim_language_guard.py",
            args.expected_language_builder_sha256,
        ),
        "terminal_builder": _source_identity(
            project,
            "build_generation_terminal_system_claim_guard.py",
            args.expected_terminal_builder_sha256,
        ),
    }
    paths = _output_paths(quality_root)
    for directory_key in ("qualification_dir", "language_dir", "terminal_dir"):
        directory = paths[directory_key]
        _require(not directory.is_symlink(), f"{directory_key} must not be a symlink")

    quality_identity, quality = _bound_json(
        args.quality_result,
        expected_sha256=args.expected_quality_result_sha256,
        label="authoritative terminal quality result",
    )
    exposure_identity, exposure = _bound_json(
        args.authoritative_exposure,
        expected_sha256=args.expected_authoritative_exposure_sha256,
        label="authoritative terminal exposure",
    )
    followup_identity, followup = _bound_json(
        args.authoritative_followup,
        expected_sha256=args.expected_authoritative_followup_sha256,
        label="authoritative follow-up decision",
    )
    execution_identity, _ = _bound_json(
        args.execution_manifest,
        expected_sha256=args.expected_execution_manifest_sha256,
        label="terminal uncertainty execution manifest",
    )
    uncertainty_identity = None
    if args.uncertainty_report is not None:
        uncertainty_identity, _ = _bound_json(
            args.uncertainty_report,
            expected_sha256=args.expected_uncertainty_report_sha256,
            label="terminal matched uncertainty report",
        )
    visual_identity, _ = _bound_json(
        args.visual_audit_waiter_status,
        expected_sha256=args.expected_visual_audit_waiter_status_sha256,
        label="schema-6 visual-audit waiter status",
    )
    runtime_guard_identity, _ = _bound_json(
        args.runtime_claim_guard,
        expected_sha256=args.expected_runtime_claim_guard_sha256,
        label="runtime-strict claim guard",
    )
    runtime_status_identity, runtime_status = _bound_json(
        args.runtime_waiter_status,
        expected_sha256=args.expected_runtime_waiter_status_sha256,
        label="runtime-strict waiter status",
    )
    expected_training_git = {
        "revision": args.expected_quality_revision,
        "tree": args.expected_quality_tree,
        "branch": args.expected_quality_branch,
        "tracked_dirty": False,
        "path": quality_project.as_posix(),
    }
    _require(
        _git_identity(quality_project)
        == {
            key: expected_training_git[key]
            for key in ("revision", "tree", "branch", "tracked_dirty")
        },
        "authoritative quality project Git identity differs",
    )
    _validate_authoritative_sources(
        exposure_identity=exposure_identity,
        exposure=exposure,
        followup_identity=followup_identity,
        followup=followup,
        quality_identity=quality_identity,
        expected_training_git=expected_training_git,
    )
    _validate_runtime_status(runtime_status, guard_identity=runtime_guard_identity)

    if uncertainty_identity is None:
        qualification = qualification_builder.build_fail_closed_qualification(
            quality_result_path=args.quality_result,
            expected_quality_result_sha256=quality_identity["sha256"],
            execution_manifest_path=args.execution_manifest,
            expected_execution_manifest_sha256=execution_identity["sha256"],
            expected_quality_revision=args.expected_quality_revision,
            expected_quality_branch=args.expected_quality_branch,
            output_root=terminal_uncertainty_root,
        )
    else:
        qualification = qualification_builder.build_qualification(
            quality_result_path=args.quality_result,
            expected_quality_result_sha256=quality_identity["sha256"],
            execution_manifest_path=args.execution_manifest,
            expected_execution_manifest_sha256=execution_identity["sha256"],
            uncertainty_report_path=args.uncertainty_report,
            expected_uncertainty_report_sha256=uncertainty_identity["sha256"],
            expected_quality_revision=args.expected_quality_revision,
            expected_quality_branch=args.expected_quality_branch,
            expected_evaluator_revision=args.expected_evaluator_revision,
            expected_evaluator_branch=args.expected_evaluator_branch,
            output_root=terminal_uncertainty_root,
        )
    qualification_identity = _prepare_exact(paths["qualification"], qualification)
    _prepare_exact(
        paths["qualification_status"],
        {
            "schema_version": 1,
            "role": "generation_quality_bridge_claim_qualification_waiter",
            "status": "pass",
            "phase": "completed",
            "detail": "authoritative_terminal_sources_replayed",
            "scientific_status": qualification["status"],
            "qualification": qualification_identity,
            "sources": {
                "quality_result": quality_identity,
                "execution_manifest": execution_identity,
                "uncertainty_report": uncertainty_identity,
                "authoritative_exposure": exposure_identity,
            },
            "scope": SCOPE,
        },
    )

    language = language_builder.build_guard(
        source_kind="quality_bridge_100k",
        source_report_path=paths["qualification"],
        expected_source_report_sha256=qualification_identity["sha256"],
    )
    language_identity = _prepare_exact(paths["language"], language)
    _prepare_exact(
        paths["language_status"],
        {
            "schema_version": 1,
            "role": "generation_quality_bridge_claim_language_guard_waiter",
            "status": "pass",
            "phase": "completed",
            "detail": "authoritative_statistical_claim_replayed",
            "scientific_status": language["status"],
            "guard": language_identity,
            "source": qualification_identity,
            "scope": SCOPE,
        },
    )

    terminal = terminal_builder.build_guard(
        quality_result_path=args.quality_result,
        expected_quality_result_sha256=quality_identity["sha256"],
        statistical_claim_guard_path=paths["language"],
        expected_statistical_claim_guard_sha256=language_identity["sha256"],
        visual_audit_waiter_status_path=args.visual_audit_waiter_status,
        expected_visual_audit_waiter_status_sha256=visual_identity["sha256"],
        runtime_claim_guard_path=args.runtime_claim_guard,
        expected_runtime_claim_guard_sha256=runtime_guard_identity["sha256"],
        quality_output_root=quality_root,
    )
    terminal_identity = _prepare_exact(paths["terminal"], terminal)
    terminal_status = {
        "schema_version": 1,
        "role": TERMINAL_WAITER_ROLE,
        "status": "completed",
        "detail": "terminal_system_claim_guard_source_revalidated",
        "updated_at": _utc_now(),
        "authorization_boundary": TERMINAL_WAITER_BOUNDARY,
        "expected": {
            "git": expected_git,
            "quality_output_root": quality_root.as_posix(),
            "output": paths["terminal"].as_posix(),
        },
        "git": git,
        "runtime": runtime,
        "guard": terminal_identity,
        "guard_status": terminal["status"],
        "guard_decision": terminal["decision"],
        "authoritative_sources": {
            "quality_result": quality_identity,
            "training_exposure": exposure_identity,
            "followup_decision": followup_identity,
            "schema6_visual_audit_waiter": visual_identity,
            "runtime_strict_guard": runtime_guard_identity,
            "runtime_strict_waiter": runtime_status_identity,
        },
    }
    terminal_status_identity = _prepare_exact(paths["terminal_status"], terminal_status)
    deployment = {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "pass",
        "created_at": _utc_now(),
        "control_git": git,
        "runtime": runtime,
        "sources": sources,
        "authoritative_inputs": terminal_status["authoritative_sources"],
        "outputs": {
            "qualification": qualification_identity,
            "language_guard": language_identity,
            "terminal_guard": terminal_identity,
            "terminal_waiter_status": terminal_status_identity,
        },
        "scientific_status": terminal["status"],
        "generation_advantage_proven": terminal["status"] == "pass",
        "scope": SCOPE,
    }
    _prepare_exact(paths["deployment"], deployment)
    print(
        json.dumps(
            {
                "status": "pass",
                "terminal_status": terminal["status"],
                "generation_advantage_proven": terminal["status"] == "pass",
                "terminal_guard": terminal_identity,
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay authoritative versioned terminal evidence into statistical, "
            "claim-language, and terminal-system guards without authorizing GPU work."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--terminal-uncertainty-output-root", type=Path, required=True)
    for name in (
        "quality-result",
        "authoritative-exposure",
        "authoritative-followup",
        "execution-manifest",
        "visual-audit-waiter-status",
        "runtime-claim-guard",
        "runtime-waiter-status",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--uncertainty-report", type=Path)
    parser.add_argument("--expected-uncertainty-report-sha256")
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-runner-source-sha256", required=True)
    parser.add_argument("--expected-qualification-builder-sha256", required=True)
    parser.add_argument("--expected-language-builder-sha256", required=True)
    parser.add_argument("--expected-terminal-builder-sha256", required=True)
    parser.add_argument("--expected-quality-revision", required=True)
    parser.add_argument("--expected-quality-tree", required=True)
    parser.add_argument("--expected-quality-branch", required=True)
    parser.add_argument("--expected-evaluator-revision", required=True)
    parser.add_argument("--expected-evaluator-branch", default="")
    args = parser.parse_args()
    if (args.uncertainty_report is None) is not (
        args.expected_uncertainty_report_sha256 is None
    ):
        parser.error(
            "--uncertainty-report and --expected-uncertainty-report-sha256 "
            "must be supplied together"
        )
    return args


def main() -> int:
    args = parse_args()
    terminal_dir = _output_paths(args.quality_output_root.resolve())["terminal_dir"]
    try:
        with exclusive_output_lock(terminal_dir, role=ROLE):
            return run_locked(args)
    except OutputLockError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
