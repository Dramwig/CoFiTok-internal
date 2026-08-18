from __future__ import annotations

import argparse
import copy
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_probe import (
    validate_standing_experiment_authorization,
)
from cofitok.generation.conditioning_ranking_sampling import (
    AUTHORIZATION_BOUNDARY,
    EXPECTED_OUTPUT_ROOT,
    EXPECTED_POSTEVALUATION_DECISION,
    IDLE_GPU_EVIDENCE_ROLE,
    STAGE,
    build_sampling_preparation,
    clean_git,
    validate_idle_gpu_evidence,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report
from scripts.build_generation_conditioning_ranking_sampling_validation import (
    replay_postevaluation,
    validate_training_checkpoints,
)
from scripts.evaluate_generation_class_fidelity import classifier_identity
from scripts.prepare_generation_conditioning_ranking_sampling_validation import (
    replay_sampling_preparation,
)


ROLE = "generation_conditioning_ranking_sampling_validation_supervisor"
LAUNCH_RECEIPT_ROLE = (
    "generation_conditioning_ranking_sampling_validation_runbook_launch_receipt"
)
FINAL_REPORT_RELATIVE_PATH = Path("reports") / "sampling_validation.json"
RUNBOOK_RELATIVE_PATH = (
    Path("artifacts")
    / "runbooks"
    / "generation_conditioning_ranking_four_arm_sampling5k_v1.sh"
)
NOT_SELECTED_ACTIONS = {
    "reject_shared_repair_due_method_asymmetry",
    "revise_training_time_semantic_alignment_objective",
}
SUPERVISOR_BOUNDARY = {
    **copy.deepcopy(AUTHORIZATION_BOUNDARY),
    "runbook_launch_count_maximum": 1,
    "automatic_runbook_relaunch_allowed": False,
    "asymmetric_or_failed_postevaluation_gpu_work_allowed": False,
}


def _bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def _full_git_status(project: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_gpu_process_pids(output: str) -> list[int]:
    pids: list[int] = []
    for line in output.splitlines():
        value = line.strip()
        if not value or value.lower().startswith("no running"):
            continue
        try:
            pids.append(int(value.split(",", 1)[0].strip()))
        except ValueError as error:
            raise ValueError(
                f"unparseable nvidia-smi compute process row: {value}"
            ) from error
    return sorted(set(pids))


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_process_pids(result.stdout)


def postevaluation_route(report: Mapping[str, Any]) -> str:
    if (
        report.get("schema_version") != 1
        or report.get("role")
        != "generation_conditioning_ranking_four_arm_postevaluation"
        or report.get("status") != "completed"
    ):
        return "invalid"
    decision = report.get("decision")
    if not isinstance(decision, Mapping):
        return "invalid"
    if dict(decision) == EXPECTED_POSTEVALUATION_DECISION:
        return "selected"
    passes = decision.get("method_passes")
    if (
        not isinstance(passes, Mapping)
        or set(passes) != {"cofitok", "dense_identity"}
        or any(type(passes[name]) is not bool for name in passes)
        or decision.get("cofitok_specific_advantage_claim_allowed") is not False
    ):
        return "invalid"
    shared = all(bool(passes[name]) for name in ("cofitok", "dense_identity"))
    if decision.get("shared_semantic_alignment_recovery_supported") is not shared:
        return "invalid"
    action = decision.get("recommended_next_action")
    if shared or action not in NOT_SELECTED_ACTIONS:
        return "invalid"
    if any(bool(passes[name]) for name in passes):
        return (
            "not_selected"
            if action == "reject_shared_repair_due_method_asymmetry"
            else "invalid"
        )
    return (
        "not_selected"
        if action == "revise_training_time_semantic_alignment_objective"
        else "invalid"
    )


def build_idle_gpu_evidence(
    *,
    git: Mapping[str, Any],
    output_root: str,
    observations: list[Mapping[str, Any]],
    required_polls: int,
) -> dict[str, Any]:
    if required_polls != 5:
        raise ValueError("conditioning-ranking sampling requires exactly five idle polls")
    report = {
        "schema_version": 1,
        "role": IDLE_GPU_EVIDENCE_ROLE,
        "status": "pass",
        "stage": STAGE,
        "output_root": output_root,
        "git": clean_git(git, label="idle GPU evidence builder"),
        "required_consecutive_idle_polls": required_polls,
        "hostname": socket.gethostname(),
        "observations": [copy.deepcopy(dict(row)) for row in observations],
    }
    validate_idle_gpu_evidence(
        report,
        expected_git=report["git"],
        expected_output_root=output_root,
        required_polls=required_polls,
    )
    return report


def build_launch_receipt(
    *,
    git: Mapping[str, Any],
    output_root: str,
    postevaluation_identity: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    standing_identity: Mapping[str, Any],
    idle_identity: Mapping[str, Any],
    runbook_identity: Mapping[str, Any],
    classifier: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": LAUNCH_RECEIPT_ROLE,
        "status": "launch_reserved",
        "stage": STAGE,
        "output_root": output_root,
        "git": clean_git(git, label="sampling launch receipt"),
        "sources": {
            "postevaluation": dict(postevaluation_identity),
            "preparation": dict(preparation_identity),
            "standing_authorization": dict(standing_identity),
            "idle_gpu_evidence": dict(idle_identity),
            "runbook": dict(runbook_identity),
        },
        "classifier": copy.deepcopy(dict(classifier)),
        "command": ["bash", str(runbook_identity["path"])],
        "authorization_boundary": copy.deepcopy(SUPERVISOR_BOUNDARY),
    }


def _status(
    *,
    status: str,
    detail: str,
    project: Path,
    output_root: Path,
    idle_polls: int,
    child_pid: int | None = None,
    sources: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "updated_at_unix": time.time(),
        "pid": os.getpid(),
        "child_pid": child_pid,
        "project": project.resolve().as_posix(),
        "output_root": output_root.resolve().as_posix(),
        "idle_gpu_polls": idle_polls,
        "sources": dict(sources or {}),
        "authorization_boundary": copy.deepcopy(SUPERVISOR_BOUNDARY),
    }
    if error is not None:
        payload["error"] = error
    return payload


def _write_status(path: Path, **kwargs: Any) -> None:
    write_json_report(path, _status(**kwargs))


def _assert_exact_clean_git(
    project: Path,
    *,
    expected_git: Mapping[str, Any],
) -> None:
    if git_provenance(project) != dict(expected_git) or _full_git_status(project):
        raise ValueError("sampling supervisor requires the exact fully clean checkout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact passing four-arm 1K postevaluation, record five "
            "idle-GPU observations, and launch the source-bound resumable 5K "
            "sampling validation once."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--postevaluation", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--idle-gpu-evidence", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> int:
    project = reject_symlink_chain(
        args.project,
        name="conditioning-ranking sampling supervisor project",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="conditioning-ranking 5K sampling output root",
    ).resolve()
    postevaluation_path = reject_symlink_chain(
        args.postevaluation,
        name="conditioning-ranking 1K postevaluation",
    ).resolve()
    preparation_path = reject_symlink_chain(
        args.preparation,
        name="conditioning-ranking 5K sampling preparation",
    ).resolve()
    idle_path = reject_symlink_chain(
        args.idle_gpu_evidence,
        name="conditioning-ranking idle GPU evidence",
    ).resolve()
    launch_receipt_path = reject_symlink_chain(
        args.launch_receipt,
        name="conditioning-ranking runbook launch receipt",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="conditioning-ranking sampling supervisor status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="conditioning-ranking sampling supervisor pid",
    ).resolve()
    runbook = reject_symlink_chain(
        args.runbook,
        name="conditioning-ranking sampling runbook",
    ).resolve()
    python = reject_symlink_chain(
        args.python,
        name="conditioning-ranking sampling Python",
    ).resolve()
    classifier_path = reject_symlink_chain(
        args.classifier_checkpoint,
        name="conditioning-ranking classifier checkpoint",
    ).resolve()
    standing_path = reject_symlink_chain(
        args.standing_authorization,
        name="standing experiment authorization",
    ).resolve()
    expected_git = {
        "revision": args.expected_revision,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if output_root.as_posix() != EXPECTED_OUTPUT_ROOT:
        raise ValueError("conditioning-ranking 5K sampling output root differs")
    if args.required_idle_polls != 5:
        raise ValueError("conditioning-ranking sampling requires exactly five idle polls")
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("sampling supervisor timing values must be positive")
    if runbook != (project / RUNBOOK_RELATIVE_PATH).resolve():
        raise ValueError("conditioning-ranking sampling runbook path differs")
    if not project.is_dir() or not runbook.is_file() or not python.is_file():
        raise FileNotFoundError("sampling supervisor checkout, runbook, or Python is missing")
    control_parent = preparation_path.parent
    if any(
        path.parent != control_parent
        for path in (idle_path, launch_receipt_path, status_output, pid_file)
    ):
        raise ValueError("sampling supervisor control artifacts must share one directory")
    _assert_exact_clean_git(project, expected_git=expected_git)
    standing, standing_identity = _bound_json(
        standing_path,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
    validate_standing_experiment_authorization(standing)
    classifier = classifier_identity(classifier_path)
    runbook_identity = file_identity(runbook)
    if launch_receipt_path.exists():
        raise FileExistsError(
            "sampling runbook launch was already reserved; automatic relaunch is forbidden"
        )
    if output_root.exists() or Path(f"{output_root}.lock").exists():
        raise FileExistsError("conditioning-ranking sampling output or lock already exists")
    write_json_report(
        pid_file,
        {
            "schema_version": 1,
            "role": f"{ROLE}_pid",
            "pid": os.getpid(),
            "project": project.as_posix(),
            "expected_revision": args.expected_revision,
            "expected_branch": args.expected_branch,
        },
    )
    sources: dict[str, Any] = {
        "standing_authorization": standing_identity,
        "runbook": runbook_identity,
    }
    started = time.monotonic()
    observations: list[dict[str, Any]] = []
    preparation_identity: dict[str, Any] | None = None
    postevaluation_identity: dict[str, Any] | None = None
    while True:
        if time.monotonic() - started > args.timeout_seconds:
            _write_status(
                status_output,
                status="failed",
                detail="timeout_before_sampling_launch",
                project=project,
                output_root=output_root,
                idle_polls=len(observations),
                sources=sources,
            )
            return 4
        if preparation_identity is None:
            if not postevaluation_path.is_file():
                observations = []
                _write_status(
                    status_output,
                    status="waiting",
                    detail="waiting_for_conditioning_ranking_1k_postevaluation",
                    project=project,
                    output_root=output_root,
                    idle_polls=0,
                    sources=sources,
                )
                time.sleep(args.poll_seconds)
                continue
            postevaluation, postevaluation_identity, training_reports = (
                replay_postevaluation(
                    postevaluation_path,
                    require_sampling_selected=False,
                )
            )
            sources["postevaluation"] = postevaluation_identity
            route = postevaluation_route(postevaluation)
            if route == "not_selected":
                _write_status(
                    status_output,
                    status="completed",
                    detail="sampling_not_selected_by_1k_postevaluation",
                    project=project,
                    output_root=output_root,
                    idle_polls=0,
                    sources=sources,
                )
                return 0
            if route != "selected":
                raise ValueError(
                    "conditioning-ranking 1K postevaluation route is malformed"
                )
            preparation = build_sampling_preparation(
                postevaluation=postevaluation,
                postevaluation_identity=postevaluation_identity,
                checkpoint_evidence=validate_training_checkpoints(
                    postevaluation,
                    training_reports,
                ),
                standing_authorization=standing,
                standing_authorization_identity=standing_identity,
                classifier=classifier,
                runbook_identity=runbook_identity,
                builder_git=git_provenance(project),
                expected_revision=args.expected_revision,
                expected_branch=args.expected_branch,
                expected_output_root=output_root.as_posix(),
            )
            preparation_identity = prepare_manifest(
                preparation_path,
                preparation,
                resume=preparation_path.is_file(),
                overwrite=False,
            )
            replayed, replayed_identity = replay_sampling_preparation(
                preparation_path
            )
            if replayed != preparation or replayed_identity != preparation_identity:
                raise ValueError(
                    "conditioning-ranking sampling preparation replay differs"
                )
            sources["preparation"] = preparation_identity
        if postevaluation_identity is None or preparation_identity is None:
            raise RuntimeError("sampling preparation state was not retained")
        if output_root.exists() or Path(f"{output_root}.lock").exists():
            raise FileExistsError("conditioning-ranking sampling output or lock appeared")
        pids = gpu_compute_pids()
        if pids:
            observations = []
            _write_status(
                status_output,
                status="waiting",
                detail="gpu_busy",
                project=project,
                output_root=output_root,
                idle_polls=0,
                sources={**sources, "gpu_compute_pids": pids},
            )
            time.sleep(args.poll_seconds)
            continue
        observed_at = time.time()
        if observations and observed_at <= observations[-1]["observed_at_unix"]:
            observed_at = float(observations[-1]["observed_at_unix"]) + 1e-6
        observations.append(
            {
                "poll_index": len(observations) + 1,
                "observed_at_unix": observed_at,
                "gpu_compute_pids": [],
            }
        )
        if len(observations) < args.required_idle_polls:
            _write_status(
                status_output,
                status="waiting",
                detail="waiting_for_consecutive_idle_gpu_polls",
                project=project,
                output_root=output_root,
                idle_polls=len(observations),
                sources=sources,
            )
            time.sleep(args.poll_seconds)
            continue
        final_gpu_pids = gpu_compute_pids()
        if final_gpu_pids:
            observations = []
            _write_status(
                status_output,
                status="waiting",
                detail="gpu_became_busy_before_launch_reservation",
                project=project,
                output_root=output_root,
                idle_polls=0,
                sources={**sources, "gpu_compute_pids": final_gpu_pids},
            )
            time.sleep(args.poll_seconds)
            continue
        idle_report = build_idle_gpu_evidence(
            git=expected_git,
            output_root=output_root.as_posix(),
            observations=observations,
            required_polls=args.required_idle_polls,
        )
        idle_identity = prepare_manifest(
            idle_path,
            idle_report,
            resume=idle_path.is_file(),
            overwrite=False,
        )
        sources["idle_gpu_evidence"] = idle_identity
        _assert_exact_clean_git(project, expected_git=expected_git)
        if file_identity(postevaluation_path) != postevaluation_identity:
            raise ValueError("conditioning-ranking postevaluation changed before launch")
        if file_identity(preparation_path) != preparation_identity:
            raise ValueError("conditioning-ranking sampling preparation changed before launch")
        launch_receipt = build_launch_receipt(
            git=expected_git,
            output_root=output_root.as_posix(),
            postevaluation_identity=postevaluation_identity,
            preparation_identity=preparation_identity,
            standing_identity=standing_identity,
            idle_identity=idle_identity,
            runbook_identity=runbook_identity,
            classifier=classifier,
        )
        launch_identity = prepare_manifest(
            launch_receipt_path,
            launch_receipt,
            resume=False,
            overwrite=False,
        )
        sources["runbook_launch_receipt"] = launch_identity
        environment = os.environ.copy()
        environment.update(
            {
                "PROJECT": project.as_posix(),
                "PYTHON": python.as_posix(),
                "EXPECTED_REVISION": args.expected_revision,
                "EXPECTED_BRANCH": args.expected_branch,
                "POSTEVALUATION": postevaluation_path.as_posix(),
                "EXPECTED_POSTEVALUATION_SHA256": postevaluation_identity["sha256"],
                "PREPARATION_REPORT": preparation_path.as_posix(),
                "EXPECTED_PREPARATION_SHA256": preparation_identity["sha256"],
                "STANDING_AUTHORIZATION": standing_path.as_posix(),
                "EXPECTED_STANDING_AUTHORIZATION_SHA256": standing_identity["sha256"],
                "IDLE_GPU_EVIDENCE": idle_path.as_posix(),
                "EXPECTED_IDLE_GPU_EVIDENCE_SHA256": idle_identity["sha256"],
                "CLASSIFIER_CHECKPOINT": classifier_path.as_posix(),
                "OUTPUT_ROOT": output_root.as_posix(),
            }
        )
        child = subprocess.Popen(
            ["bash", runbook.as_posix()],
            cwd=project,
            env=environment,
        )
        while child.poll() is None:
            _write_status(
                status_output,
                status="running",
                detail="conditioning_ranking_four_arm_sampling5k_running",
                project=project,
                output_root=output_root,
                idle_polls=len(observations),
                child_pid=child.pid,
                sources=sources,
            )
            time.sleep(min(args.poll_seconds, 60.0))
        if child.returncode != 0:
            _write_status(
                status_output,
                status="failed",
                detail="conditioning_ranking_sampling_runbook_failed",
                project=project,
                output_root=output_root,
                idle_polls=len(observations),
                child_pid=child.pid,
                sources=sources,
                error=f"runbook exit code {child.returncode}",
            )
            return int(child.returncode or 1)
        final_report = output_root / FINAL_REPORT_RELATIVE_PATH
        if not final_report.is_file():
            raise FileNotFoundError("conditioning-ranking sampling validation is missing")
        sources["sampling_validation"] = file_identity(final_report)
        _write_status(
            status_output,
            status="completed",
            detail="conditioning_ranking_four_arm_sampling5k_completed",
            project=project,
            output_root=output_root,
            idle_polls=len(observations),
            child_pid=child.pid,
            sources=sources,
        )
        return 0


def main() -> int:
    args = parse_args()
    try:
        return _run(args)
    except Exception as error:
        try:
            project = Path(args.project).resolve()
            output_root = Path(args.output_root).resolve()
            status_output = Path(args.status_output).resolve()
            _write_status(
                status_output,
                status="failed",
                detail="sampling_supervisor_exception",
                project=project,
                output_root=output_root,
                idle_polls=0,
                sources={},
                error=f"{type(error).__name__}: {error}",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
