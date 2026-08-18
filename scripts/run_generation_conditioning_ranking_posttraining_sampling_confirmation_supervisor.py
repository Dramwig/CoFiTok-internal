from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_posttraining_sampling import (
    EXPECTED_HELDOUT_DECISION,
    EXPECTED_OUTPUT_ROOT,
    IDLE_GPU_EVIDENCE_ROLE,
    RUNBOOK_RELATIVE_PATH,
    STAGE,
    build_sampling_preparation,
    validate_idle_gpu_evidence,
)
from cofitok.generation.conditioning_ranking_probe import (
    validate_standing_experiment_authorization,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report
from scripts.evaluate_generation_class_fidelity import classifier_identity
from scripts.prepare_generation_conditioning_ranking_posttraining_sampling_confirmation import (
    replay_heldout_evaluation,
    replay_sampling_preparation,
    validate_training_checkpoints,
)


ROLE = (
    "generation_conditioning_ranking_posttraining_sampling_confirmation_supervisor"
)
LAUNCH_RECEIPT_ROLE = (
    "generation_conditioning_ranking_posttraining_sampling_confirmation_"
    "runbook_launch_receipt"
)
FINAL_REPORT_RELATIVE_PATH = (
    Path("reports") / "posttraining_sampling_confirmation.json"
)
SUPERVISOR_BOUNDARY = {
    "runbook_launch_count_maximum": 1,
    "automatic_runbook_relaunch_allowed": False,
    "training_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_100k_or_300k_launch_allowed": False,
    "release_authorization_allowed": False,
    "unrelated_process_signaling_allowed": False,
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


def _embedded_json(
    descriptor: object,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} identity is missing")
    source = reject_symlink_chain(
        str(descriptor.get("path", "")),
        name=label,
    ).resolve()
    identity = file_identity(source)
    if identity != descriptor:
        raise ValueError(f"{label} identity differs")
    return read_json_object(source, name=label), identity


def _full_git_status(project: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def parse_gpu_process_pids(output: str) -> list[int]:
    pids: list[int] = []
    for raw in output.splitlines():
        value = raw.strip()
        if not value:
            continue
        if not value.isdigit():
            raise ValueError("nvidia-smi returned a malformed compute PID")
        pids.append(int(value))
    return sorted(set(pids))


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return parse_gpu_process_pids(result.stdout)


def heldout_route(report: Mapping[str, Any]) -> str:
    decision = report.get("decision")
    if not isinstance(decision, Mapping):
        return "malformed"
    if dict(decision) == EXPECTED_HELDOUT_DECISION:
        return "selected"
    action = decision.get("recommended_next_action")
    if action in {
        "reject_shared_repair_due_method_asymmetry",
        "revise_training_time_semantic_alignment_objective",
    }:
        return "not_selected"
    return "malformed"


def build_idle_gpu_evidence(
    *,
    git: Mapping[str, Any],
    output_root: str,
    observations: list[dict[str, Any]],
    required_polls: int,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "role": IDLE_GPU_EVIDENCE_ROLE,
        "status": "pass",
        "stage": STAGE,
        "output_root": output_root,
        "git": dict(git),
        "required_consecutive_idle_polls": required_polls,
        "observations": observations,
    }
    validate_idle_gpu_evidence(
        report,
        expected_git=git,
        expected_output_root=output_root,
        required_polls=required_polls,
    )
    return report


def build_launch_receipt(
    *,
    git: Mapping[str, Any],
    output_root: str,
    heldout_identity: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    standing_identity: Mapping[str, Any],
    idle_identity: Mapping[str, Any],
    runbook_identity: Mapping[str, Any],
    classifier: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": LAUNCH_RECEIPT_ROLE,
        "status": "reserved",
        "stage": STAGE,
        "output_root": output_root,
        "git": dict(git),
        "sources": {
            "heldout_evaluation": dict(heldout_identity),
            "preparation": dict(preparation_identity),
            "standing_authorization": dict(standing_identity),
            "idle_gpu_evidence": dict(idle_identity),
            "runbook": dict(runbook_identity),
        },
        "classifier": dict(classifier),
        "command": ["bash", str(runbook_identity["path"])],
        "boundary": dict(SUPERVISOR_BOUNDARY),
    }


def _status(
    *,
    status: str,
    detail: str,
    project: Path,
    output_root: Path,
    idle_polls: int,
    sources: Mapping[str, Any],
    child_pid: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "stage": STAGE,
        "project": project.as_posix(),
        "output_root": output_root.as_posix(),
        "idle_gpu_polls": idle_polls,
        "child_pid": child_pid,
        "sources": dict(sources),
        "error": error,
        "boundary": dict(SUPERVISOR_BOUNDARY),
        "updated_at_unix": time.time(),
    }


def _write_status(path: Path, **kwargs: Any) -> None:
    write_json_report(path, _status(**kwargs))


def _assert_exact_clean_git(
    project: Path,
    *,
    expected_git: Mapping[str, Any],
) -> None:
    actual = git_provenance(project)
    if actual != expected_git or _full_git_status(project):
        raise ValueError("posttraining sampling supervisor checkout is not exact and clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact fresh-5K heldout shared pass, collect five idle "
            "GPU polls, and launch the posttraining sampling confirmation once."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--heldout-evaluation", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--idle-gpu-evidence", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=31_536_000.0)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> int:
    project = reject_symlink_chain(args.project, name="posttraining project").resolve()
    heldout_path = reject_symlink_chain(
        args.heldout_evaluation,
        name="5K heldout evaluation",
    ).resolve()
    standing_path = reject_symlink_chain(
        args.standing_authorization,
        name="standing experiment authorization",
    ).resolve()
    classifier_path = reject_symlink_chain(
        args.classifier_checkpoint,
        name="conditioning classifier checkpoint",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="posttraining sampling output root",
    ).resolve()
    preparation_path = reject_symlink_chain(
        args.preparation,
        name="posttraining sampling preparation",
    ).resolve()
    idle_path = reject_symlink_chain(
        args.idle_gpu_evidence,
        name="posttraining idle GPU evidence",
    ).resolve()
    launch_path = reject_symlink_chain(
        args.launch_receipt,
        name="posttraining sampling launch receipt",
    ).resolve()
    runbook = reject_symlink_chain(args.runbook, name="posttraining runbook").resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="posttraining supervisor status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="posttraining supervisor pid",
    ).resolve()
    python = reject_symlink_chain(args.python, name="posttraining Python").resolve()
    expected_git = {
        "revision": args.expected_revision,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if output_root.as_posix() != EXPECTED_OUTPUT_ROOT:
        raise ValueError("posttraining sampling output root differs")
    if args.required_idle_polls != 5:
        raise ValueError("posttraining sampling requires exactly five idle polls")
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("posttraining supervisor timing values must be positive")
    if runbook != (project / RUNBOOK_RELATIVE_PATH).resolve():
        raise ValueError("posttraining sampling runbook path differs")
    if not project.is_dir() or not runbook.is_file() or not python.is_file():
        raise FileNotFoundError("posttraining checkout, runbook, or Python is missing")
    control_parent = preparation_path.parent
    if any(
        path.parent != control_parent
        for path in (idle_path, launch_path, status_output, pid_file)
    ):
        raise ValueError("posttraining control artifacts must share one directory")
    _assert_exact_clean_git(project, expected_git=expected_git)
    standing, standing_identity = _bound_json(
        standing_path,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
    validate_standing_experiment_authorization(standing)
    classifier = classifier_identity(classifier_path)
    runbook_identity = file_identity(runbook)
    if launch_path.exists():
        raise FileExistsError(
            "posttraining sampling launch was already reserved; relaunch is forbidden"
        )
    if output_root.exists() or Path(f"{output_root}.lock").exists():
        raise FileExistsError("posttraining sampling output or lock already exists")
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
    heldout_identity: dict[str, Any] | None = None
    training_status_identity: dict[str, Any] | None = None
    training_receipt_identity: dict[str, Any] | None = None
    while True:
        if time.monotonic() - started > args.timeout_seconds:
            _write_status(
                status_output,
                status="failed",
                detail="timeout_before_posttraining_sampling_launch",
                project=project,
                output_root=output_root,
                idle_polls=len(observations),
                sources=sources,
            )
            return 4
        if preparation_identity is None:
            if not heldout_path.is_file():
                observations = []
                _write_status(
                    status_output,
                    status="waiting",
                    detail="waiting_for_exact_5k_heldout_evaluation",
                    project=project,
                    output_root=output_root,
                    idle_polls=0,
                    sources=sources,
                )
                time.sleep(args.poll_seconds)
                continue
            heldout_report, heldout_identity = replay_heldout_evaluation(heldout_path)
            sources["heldout_evaluation"] = heldout_identity
            route = heldout_route(heldout_report)
            if route == "not_selected":
                _write_status(
                    status_output,
                    status="completed",
                    detail="posttraining_sampling_not_selected_by_heldout_evaluation",
                    project=project,
                    output_root=output_root,
                    idle_polls=0,
                    sources=sources,
                )
                return 0
            if route != "selected":
                raise ValueError("posttraining heldout evaluation route is malformed")
            training_status, training_status_identity = _embedded_json(
                heldout_report["sources"]["training_status"],
                label="5K training status",
            )
            training_receipt_path = reject_symlink_chain(
                Path(str(training_status["output_root"]))
                / "reports"
                / "execution_receipt.json",
                name="5K training execution receipt",
            ).resolve()
            training_receipt = read_json_object(
                training_receipt_path,
                name="5K training execution receipt",
            )
            training_receipt_identity = file_identity(training_receipt_path)
            checkpoints = validate_training_checkpoints(
                heldout_report,
                training_status,
            )
            preparation = build_sampling_preparation(
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
            replayed, replayed_identity = replay_sampling_preparation(preparation_path)
            if replayed != preparation or replayed_identity != preparation_identity:
                raise ValueError("posttraining sampling preparation replay differs")
            sources.update(
                {
                    "training_status": training_status_identity,
                    "training_execution_receipt": training_receipt_identity,
                    "preparation": preparation_identity,
                }
            )
        if not all(
            value is not None
            for value in (
                heldout_identity,
                training_status_identity,
                training_receipt_identity,
                preparation_identity,
            )
        ):
            raise RuntimeError("posttraining preparation state was not retained")
        if output_root.exists() or Path(f"{output_root}.lock").exists():
            raise FileExistsError("posttraining sampling output or lock appeared")
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
        if gpu_compute_pids():
            observations = []
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
        assert heldout_identity is not None
        assert preparation_identity is not None
        if file_identity(heldout_path) != heldout_identity:
            raise ValueError("5K heldout evaluation changed before launch")
        if file_identity(preparation_path) != preparation_identity:
            raise ValueError("posttraining sampling preparation changed before launch")
        launch_receipt = build_launch_receipt(
            git=expected_git,
            output_root=output_root.as_posix(),
            heldout_identity=heldout_identity,
            preparation_identity=preparation_identity,
            standing_identity=standing_identity,
            idle_identity=idle_identity,
            runbook_identity=runbook_identity,
            classifier=classifier,
        )
        launch_identity = prepare_manifest(
            launch_path,
            launch_receipt,
            resume=False,
            overwrite=False,
        )
        sources["runbook_launch_receipt"] = launch_identity
        assert training_status_identity is not None
        assert training_receipt_identity is not None
        environment = os.environ.copy()
        environment.update(
            {
                "PROJECT": project.as_posix(),
                "PYTHON": python.as_posix(),
                "EXPECTED_REVISION": args.expected_revision,
                "EXPECTED_BRANCH": args.expected_branch,
                "HELDOUT_EVALUATION": heldout_path.as_posix(),
                "EXPECTED_HELDOUT_EVALUATION_SHA256": heldout_identity["sha256"],
                "TRAINING_STATUS": training_status_identity["path"],
                "EXPECTED_TRAINING_STATUS_SHA256": training_status_identity["sha256"],
                "TRAINING_EXECUTION_RECEIPT": training_receipt_identity["path"],
                "EXPECTED_TRAINING_EXECUTION_RECEIPT_SHA256": training_receipt_identity[
                    "sha256"
                ],
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
                detail="posttraining_four_arm_sampling_confirmation_running",
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
                detail="posttraining_sampling_runbook_failed",
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
            raise FileNotFoundError("posttraining sampling confirmation is missing")
        sources["posttraining_sampling_confirmation"] = file_identity(final_report)
        _write_status(
            status_output,
            status="completed",
            detail="posttraining_four_arm_sampling_confirmation_completed",
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
            _write_status(
                reject_symlink_chain(
                    args.status_output,
                    name="posttraining supervisor status",
                ).resolve(),
                status="failed",
                detail="posttraining_sampling_supervisor_failed",
                project=Path(args.project).resolve(),
                output_root=Path(args.output_root).resolve(),
                idle_polls=0,
                sources={},
                error=f"{type(error).__name__}: {error}",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
