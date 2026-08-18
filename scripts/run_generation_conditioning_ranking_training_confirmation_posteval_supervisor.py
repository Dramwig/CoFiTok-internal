from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import copy
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_training_confirmation import (
    CONFIG_RELATIVE_PATHS,
    EXECUTION_RECEIPT_ROLE,
    RUNBOOK_RELATIVE_PATH as TRAINING_RUNBOOK_RELATIVE_PATH,
    RUN_NAMES,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report
from scripts.build_generation_conditioning_ranking_training_confirmation_posteval import (
    CLAIM_BOUNDARY as HELDOUT_CLAIM_BOUNDARY,
    EXPECTED_OUTPUT_ROOT,
    EXPECTED_STEPS,
    ROLE as HELDOUT_ROLE,
    STAGE as HELDOUT_STAGE,
    TRAINING_GIT,
    TRAINING_STAGE,
    TRAINING_STATUS_CLAIM_BOUNDARY,
    TRAINING_STATUS_EXECUTION_BOUNDARY,
    TRAINING_STATUS_ROLE,
)


SCHEMA_VERSION = 1
ROLE = "generation_conditioning_ranking_train5k_heldout_evaluation_supervisor"
REPLAY_ROLE = "generation_conditioning_ranking_train5k_execution_receipt_replay"
LAUNCH_RECEIPT_ROLE = (
    "generation_conditioning_ranking_train5k_heldout_evaluation_launch_receipt"
)
RUNBOOK_RELATIVE_PATH = Path("artifacts/runbooks") / (
    "generation_conditioning_ranking_four_arm_train5k_heldout_evaluation_v1.sh"
)
POSTEVAL_RELATIVE_ROOT = Path("reports") / (
    "conditioning_ranking_train5k_heldout_evaluation_v1"
)
FINAL_REPORT_NAME = "heldout_evaluation.json"
SUPERVISOR_BOUNDARY = {
    "cpu_only": True,
    "cuda_visible_devices": "-1",
    "waits_for_exact_completed_training_status": True,
    "waits_for_exact_training_process_exit": True,
    "creates_postevaluation_root_while_waiting": False,
    "signals_training_or_unrelated_processes": False,
    "automatic_runbook_relaunch_allowed": False,
    "runbook_launch_count_maximum": 1,
    "generates_samples": False,
    "training_allowed": False,
    "sampling_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_100k_or_300k_launch_allowed": False,
    "release_authorization_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact four-arm 5K training completion, replay its "
            "execution receipt from the exact training checkout, and launch "
            "the CPU-only held-out evaluation at most once."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-runbook-sha256", required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--training-execution-receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-training-execution-receipt-sha256",
    )
    parser.add_argument(
        "--training-status",
        type=Path,
        default=Path(EXPECTED_OUTPUT_ROOT) / "reports" / "training_status.json",
    )
    parser.add_argument("--training-replay-output", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(EXPECTED_OUTPUT_ROOT))
    parser.add_argument("--posteval-root", type=Path)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10"),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    return parser.parse_args()


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _exact_clean_git(
    project: Path,
    *,
    expected_revision: str,
    expected_branch: str,
    label: str,
) -> dict[str, Any]:
    project = reject_symlink_chain(project, name=label).resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"{label} is missing: {project}")
    git = git_provenance(project)
    if git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError(f"{label} Git differs")
    full_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if full_status.strip():
        raise ValueError(f"{label} contains tracked or untracked changes")
    return dict(git)


def _matching_training_processes(
    process_rows: str,
    *,
    output_root: Path,
) -> list[dict[str, Any]]:
    marker = output_root.resolve().as_posix().rstrip("/") + "/"
    matches = []
    for raw in process_rows.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        command = fields[1]
        if "scripts/train_generation.py" in command and marker in command:
            matches.append({"pid": int(fields[0]), "command": command})
    return matches


def _active_training_processes(output_root: Path) -> list[dict[str, Any]]:
    process_rows = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return _matching_training_processes(process_rows, output_root=output_root)


def _validate_training_status(
    status: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    expected_root = output_root.resolve().as_posix()
    runs = status.get("runs")
    if (
        status.get("schema_version") != SCHEMA_VERSION
        or status.get("role") != TRAINING_STATUS_ROLE
        or status.get("status") != "completed"
        or status.get("stage") != TRAINING_STAGE
        or status.get("revision") != TRAINING_GIT["revision"]
        or status.get("branch") != TRAINING_GIT["branch"]
        or status.get("output_root") != expected_root
        or status.get("execution_boundary")
        != TRAINING_STATUS_EXECUTION_BOUNDARY
        or status.get("claim_boundary") != TRAINING_STATUS_CLAIM_BOUNDARY
        or not isinstance(runs, Mapping)
        or set(runs) != set(RUN_NAMES)
    ):
        raise ValueError("5K training status is not exact completed evidence")
    validated_runs = {}
    for run in RUN_NAMES:
        value = runs[run]
        if not isinstance(value, Mapping):
            raise ValueError(f"5K training status run is malformed: {run}")
        run_dir = f"{expected_root}/{run}"
        report = _identity(value.get("training_report", {}), label=f"{run} report")
        audit = _identity(value.get("training_audit", {}), label=f"{run} audit")
        checkpoint = _identity(
            value.get("checkpoint", {}), label=f"{run} checkpoint"
        )
        integrity = _identity(
            value.get("integrity_manifest", {}), label=f"{run} integrity"
        )
        final_metrics = value.get("final_metrics")
        if (
            value.get("run_dir") != run_dir
            or report["path"] != f"{run_dir}/training_report.json"
            or audit["path"]
            != f"{expected_root}/reports/{run}_training_audit.json"
            or checkpoint["path"]
            != f"{run_dir}/checkpoint_step_00005000.pt"
            or integrity["path"]
            != f"{run_dir}/checkpoint_step_00005000.pt.integrity.json"
            or not isinstance(final_metrics, Mapping)
            or int(final_metrics.get("step", -1)) != EXPECTED_STEPS
            or int(final_metrics.get("samples_seen", -1)) != 320_000
        ):
            raise ValueError(f"5K training status run differs: {run}")
        validated_runs[run] = {
            "run_dir": run_dir,
            "training_report": report,
            "training_audit": audit,
            "checkpoint": checkpoint,
            "integrity_manifest": integrity,
        }
    return {"output_root": expected_root, "runs": validated_runs}


def _source_descriptor(receipt: Mapping[str, Any], name: str) -> dict[str, Any]:
    sources = receipt.get("source_reports")
    value = sources.get(name) if isinstance(sources, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"training execution receipt lacks source: {name}")
    return _identity(value, label=f"training execution receipt {name}")


def _replay_training_execution_receipt(
    *,
    training_project: Path,
    python: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
    replay_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training_git = _exact_clean_git(
        training_project,
        expected_revision=TRAINING_GIT["revision"],
        expected_branch=TRAINING_GIT["branch"],
        label="exact 5K training checkout",
    )
    receipt_path = reject_symlink_chain(
        receipt_path,
        name="5K training execution receipt",
    ).resolve()
    receipt_identity = file_identity(receipt_path)
    if receipt_identity["sha256"] != expected_receipt_sha256:
        raise ValueError("5K training execution receipt SHA256 differs")
    receipt = read_json_object(receipt_path, name="5K training execution receipt")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("role") != EXECUTION_RECEIPT_ROLE
        or receipt.get("status") != "authorized"
        or receipt.get("stage") != TRAINING_STAGE
        or receipt.get("authorized_revision") != TRAINING_GIT["revision"]
        or receipt.get("authorized_branch") != TRAINING_GIT["branch"]
        or receipt.get("output_root") != EXPECTED_OUTPUT_ROOT
        or receipt.get("git") != TRAINING_GIT
    ):
        raise ValueError("5K training execution receipt contract differs")

    preparation = _source_descriptor(receipt, "preparation")
    sampling = _source_descriptor(receipt, "sampling_validation")
    standing = _source_descriptor(receipt, "standing_authorization")
    idle = _source_descriptor(receipt, "idle_gpu_evidence")
    training_runbook = _source_descriptor(receipt, "runbook")
    config_descriptors = receipt.get("source_reports", {}).get("configs")
    if not isinstance(config_descriptors, Mapping) or set(config_descriptors) != set(
        RUN_NAMES
    ):
        raise ValueError("5K training execution receipt config set differs")

    verifier = (
        training_project
        / "scripts"
        / "verify_generation_conditioning_ranking_training_confirmation_execution_receipt.py"
    ).resolve()
    command = [
        str(python),
        str(verifier),
        "--receipt",
        receipt_identity["path"],
        "--expected-receipt-sha256",
        expected_receipt_sha256,
        "--preparation",
        preparation["path"],
        "--expected-preparation-sha256",
        preparation["sha256"],
        "--sampling-validation",
        sampling["path"],
        "--expected-sampling-validation-sha256",
        sampling["sha256"],
        "--standing-authorization",
        standing["path"],
        "--expected-standing-authorization-sha256",
        standing["sha256"],
        "--idle-gpu-evidence",
        idle["path"],
        "--expected-idle-gpu-evidence-sha256",
        idle["sha256"],
        "--control-cofitok",
        str((training_project / CONFIG_RELATIVE_PATHS["control_cofitok"]).resolve()),
        "--control-dense",
        str(
            (
                training_project
                / CONFIG_RELATIVE_PATHS["control_dense_identity"]
            ).resolve()
        ),
        "--ranked-cofitok",
        str((training_project / CONFIG_RELATIVE_PATHS["ranked_cofitok"]).resolve()),
        "--ranked-dense",
        str(
            (
                training_project
                / CONFIG_RELATIVE_PATHS["ranked_dense_identity"]
            ).resolve()
        ),
        "--runbook",
        str((training_project / TRAINING_RUNBOOK_RELATIVE_PATH).resolve()),
        "--expected-runbook-sha256",
        training_runbook["sha256"],
        "--expected-revision",
        TRAINING_GIT["revision"],
        "--expected-branch",
        TRAINING_GIT["branch"],
        "--expected-output-root",
        EXPECTED_OUTPUT_ROOT,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONPATH": ".:src",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=training_project,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip().splitlines()[-1] != expected_receipt_sha256:
        raise ValueError("5K training execution receipt replay output differs")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "role": REPLAY_ROLE,
        "status": "pass",
        "stage": HELDOUT_STAGE,
        "training_git": training_git,
        "execution_receipt": receipt_identity,
        "verifier": file_identity(verifier),
        "source_reports": {
            "preparation": preparation,
            "sampling_validation": sampling,
            "standing_authorization": standing,
            "idle_gpu_evidence": idle,
            "runbook": training_runbook,
            "configs": {
                run: _identity(
                    config_descriptors[run],
                    label=f"training execution receipt config {run}",
                )
                for run in RUN_NAMES
            },
        },
        "replayed_receipt_sha256": expected_receipt_sha256,
        "authorization_boundary": copy.deepcopy(SUPERVISOR_BOUNDARY),
    }
    replay_output = reject_symlink_chain(
        replay_output,
        name="training execution replay output",
    )
    if replay_output.exists():
        if read_json_object(replay_output, name="training execution replay") != payload:
            raise ValueError("Existing training execution replay differs")
    else:
        write_json_report(replay_output, payload)
    return payload, file_identity(replay_output)


def _build_launch_receipt(
    *,
    evaluator_git: Mapping[str, Any],
    runbook_identity: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    training_status_identity: Mapping[str, Any],
    training_execution_receipt_identity: Mapping[str, Any],
    training_replay_identity: Mapping[str, Any],
    output_root: Path,
    posteval_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": LAUNCH_RECEIPT_ROLE,
        "status": "launch_reserved",
        "stage": HELDOUT_STAGE,
        "evaluator_git": dict(evaluator_git),
        "training_git": copy.deepcopy(TRAINING_GIT),
        "output_root": output_root.resolve().as_posix(),
        "posteval_root": posteval_root.resolve().as_posix(),
        "sources": {
            "runbook": _identity(runbook_identity, label="held-out runbook"),
            "preparation": _identity(
                preparation_identity, label="held-out preparation"
            ),
            "training_status": _identity(
                training_status_identity, label="held-out training status"
            ),
            "training_execution_receipt": _identity(
                training_execution_receipt_identity,
                label="held-out training execution receipt",
            ),
            "training_execution_replay": _identity(
                training_replay_identity,
                label="held-out training execution replay",
            ),
        },
        "command": ["bash", str(runbook_identity["path"])],
        "authorization_boundary": copy.deepcopy(SUPERVISOR_BOUNDARY),
    }


def _status_payload(
    *,
    status: str,
    detail: str,
    evaluator_project: Path,
    training_project: Path,
    output_root: Path,
    posteval_root: Path,
    child_pid: int | None = None,
    sources: Mapping[str, Any] | None = None,
    active_training_processes: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "updated_at_unix": time.time(),
        "pid": os.getpid(),
        "child_pid": child_pid,
        "evaluator_project": evaluator_project.resolve().as_posix(),
        "training_project": training_project.resolve().as_posix(),
        "output_root": output_root.resolve().as_posix(),
        "posteval_root": posteval_root.resolve().as_posix(),
        "sources": copy.deepcopy(dict(sources or {})),
        "active_training_processes": copy.deepcopy(
            active_training_processes or []
        ),
        "authorization_boundary": copy.deepcopy(SUPERVISOR_BOUNDARY),
    }
    if error is not None:
        payload["error"] = error
    return payload


def _validate_completed_report(
    report: Mapping[str, Any],
    *,
    evaluator_git: Mapping[str, Any],
    output_root: Path,
) -> None:
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("role") != HELDOUT_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != HELDOUT_STAGE
        or report.get("training_git") != TRAINING_GIT
        or report.get("evaluator_git") != evaluator_git
        or report.get("output_root") != output_root.resolve().as_posix()
        or report.get("claim_boundary") != HELDOUT_CLAIM_BOUNDARY
    ):
        raise ValueError("Completed held-out evaluation contract differs")


def main() -> None:
    args = parse_args()
    evaluator_project = reject_symlink_chain(
        args.project, name="held-out evaluator checkout"
    ).resolve()
    training_project = reject_symlink_chain(
        args.training_project, name="exact 5K training checkout"
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root, name="5K training output root"
    ).resolve()
    if output_root.as_posix() != EXPECTED_OUTPUT_ROOT:
        raise ValueError("5K training output root differs")
    posteval_root = reject_symlink_chain(
        args.posteval_root or output_root / POSTEVAL_RELATIVE_ROOT,
        name="5K held-out evaluation root",
    ).resolve()
    expected_posteval_root = (output_root / POSTEVAL_RELATIVE_ROOT).resolve()
    if posteval_root != expected_posteval_root:
        raise ValueError("5K held-out evaluation root differs")
    status_output = reject_symlink_chain(
        args.status_output, name="held-out supervisor status"
    )

    sources: dict[str, Any] = {}
    child_pid: int | None = None
    try:
        evaluator_git = _exact_clean_git(
            evaluator_project,
            expected_revision=args.expected_revision,
            expected_branch=args.expected_branch,
            label="held-out evaluator checkout",
        )
        if (
            evaluator_git["revision"] == TRAINING_GIT["revision"]
            or evaluator_git["branch"] == TRAINING_GIT["branch"]
        ):
            raise ValueError("Held-out evaluator checkout is not separate")
        python = reject_symlink_chain(args.python, name="held-out Python").resolve()
        if not python.is_file():
            raise FileNotFoundError(f"Held-out Python is missing: {python}")
        runbook = reject_symlink_chain(
            evaluator_project / RUNBOOK_RELATIVE_PATH,
            name="held-out evaluation runbook",
        ).resolve()
        runbook_identity = file_identity(runbook)
        if runbook_identity["sha256"] != args.expected_runbook_sha256:
            raise ValueError("Held-out evaluation runbook SHA256 differs")
        training_execution_receipt = reject_symlink_chain(
            args.training_execution_receipt,
            name="5K training execution receipt",
        ).resolve()
        training_execution_receipt_identity: dict[str, Any] | None = None
        sources["runbook"] = runbook_identity

        launch_receipt = reject_symlink_chain(
            args.launch_receipt, name="held-out launch receipt"
        )
        final_report = posteval_root / FINAL_REPORT_NAME
        if launch_receipt.exists():
            if not final_report.is_file():
                raise FileExistsError(
                    "Held-out launch is already reserved; automatic relaunch is forbidden"
                )
            completed = read_json_object(final_report, name="held-out evaluation")
            _validate_completed_report(
                completed,
                evaluator_git=evaluator_git,
                output_root=output_root,
            )
            sources["final_report"] = file_identity(final_report)
            write_json_report(
                status_output,
                _status_payload(
                    status="completed",
                    detail="heldout_evaluation_already_completed",
                    evaluator_project=evaluator_project,
                    training_project=training_project,
                    output_root=output_root,
                    posteval_root=posteval_root,
                    sources=sources,
                ),
            )
            return
        if posteval_root.exists():
            raise FileExistsError(
                "Held-out evaluation root exists without a launch receipt"
            )

        training_status_path = reject_symlink_chain(
            args.training_status, name="5K training status"
        ).resolve()
        deadline = time.monotonic() + args.timeout_seconds
        training_status_identity: dict[str, Any] | None = None
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for exact 5K training completion")
            if not training_status_path.is_file():
                write_json_report(
                    status_output,
                    _status_payload(
                        status="waiting",
                        detail="waiting_for_exact_5k_training_status",
                        evaluator_project=evaluator_project,
                        training_project=training_project,
                        output_root=output_root,
                        posteval_root=posteval_root,
                        sources=sources,
                    ),
                )
                time.sleep(args.poll_seconds)
                continue
            training_status = read_json_object(
                training_status_path, name="5K training status"
            )
            _validate_training_status(training_status, output_root=output_root)
            training_status_identity = file_identity(training_status_path)
            active = _active_training_processes(output_root)
            if active:
                write_json_report(
                    status_output,
                    _status_payload(
                        status="waiting",
                        detail="waiting_for_exact_5k_training_process_exit",
                        evaluator_project=evaluator_project,
                        training_project=training_project,
                        output_root=output_root,
                        posteval_root=posteval_root,
                        sources={
                            **sources,
                            "training_status": training_status_identity,
                        },
                        active_training_processes=active,
                    ),
                )
                time.sleep(args.poll_seconds)
                continue
            if not training_execution_receipt.is_file():
                write_json_report(
                    status_output,
                    _status_payload(
                        status="waiting",
                        detail="waiting_for_exact_5k_training_execution_receipt",
                        evaluator_project=evaluator_project,
                        training_project=training_project,
                        output_root=output_root,
                        posteval_root=posteval_root,
                        sources={
                            **sources,
                            "training_status": training_status_identity,
                        },
                    ),
                )
                time.sleep(args.poll_seconds)
                continue
            training_execution_receipt_identity = file_identity(
                training_execution_receipt
            )
            if (
                args.expected_training_execution_receipt_sha256 is not None
                and training_execution_receipt_identity["sha256"]
                != args.expected_training_execution_receipt_sha256
            ):
                raise ValueError("5K training execution receipt SHA256 differs")
            break

        assert training_status_identity is not None
        assert training_execution_receipt_identity is not None
        preparation_path = output_root / "reports" / "preparation.json"
        preparation_identity = file_identity(preparation_path)
        receipt = read_json_object(
            training_execution_receipt,
            name="5K training execution receipt",
        )
        original_preparation = _source_descriptor(receipt, "preparation")
        if any(
            preparation_identity[field] != original_preparation[field]
            for field in ("bytes", "sha256")
        ):
            raise ValueError("Copied 5K preparation differs from execution receipt")

        _, replay_identity = _replay_training_execution_receipt(
            training_project=training_project,
            python=python,
            receipt_path=training_execution_receipt,
            expected_receipt_sha256=training_execution_receipt_identity["sha256"],
            replay_output=args.training_replay_output,
        )
        sources.update(
            {
                "preparation": preparation_identity,
                "training_status": training_status_identity,
                "training_execution_receipt": training_execution_receipt_identity,
                "training_execution_replay": replay_identity,
            }
        )
        launch_payload = _build_launch_receipt(
            evaluator_git=evaluator_git,
            runbook_identity=runbook_identity,
            preparation_identity=preparation_identity,
            training_status_identity=training_status_identity,
            training_execution_receipt_identity=training_execution_receipt_identity,
            training_replay_identity=replay_identity,
            output_root=output_root,
            posteval_root=posteval_root,
        )
        write_json_report(launch_receipt, launch_payload)
        sources["launch_receipt"] = file_identity(launch_receipt)

        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "-1",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PROJECT": evaluator_project.as_posix(),
                "PYTHON": python.as_posix(),
                "EXPECTED_REVISION": args.expected_revision,
                "EXPECTED_BRANCH": args.expected_branch,
                "OUTPUT_ROOT": output_root.as_posix(),
                "PREPARATION_REPORT": preparation_path.as_posix(),
                "EXPECTED_PREPARATION_SHA256": preparation_identity["sha256"],
                "TRAINING_STATUS": training_status_path.as_posix(),
                "EXPECTED_TRAINING_STATUS_SHA256": training_status_identity[
                    "sha256"
                ],
                "TRAINING_EXECUTION_REPLAY": Path(
                    replay_identity["path"]
                ).as_posix(),
                "EXPECTED_TRAINING_EXECUTION_REPLAY_SHA256": replay_identity[
                    "sha256"
                ],
                "POSTEVAL_ROOT": posteval_root.as_posix(),
            }
        )
        child = subprocess.Popen(
            ["bash", runbook.as_posix()],
            cwd=evaluator_project,
            env=environment,
        )
        child_pid = child.pid
        write_json_report(
            status_output,
            _status_payload(
                status="running",
                detail="heldout_evaluation_runbook_active",
                evaluator_project=evaluator_project,
                training_project=training_project,
                output_root=output_root,
                posteval_root=posteval_root,
                child_pid=child_pid,
                sources=sources,
            ),
        )
        return_code = child.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Held-out evaluation runbook exited with code {return_code}"
            )
        completed = read_json_object(final_report, name="held-out evaluation")
        _validate_completed_report(
            completed,
            evaluator_git=evaluator_git,
            output_root=output_root,
        )
        sources["final_report"] = file_identity(final_report)
        write_json_report(
            status_output,
            _status_payload(
                status="completed",
                detail="heldout_evaluation_completed",
                evaluator_project=evaluator_project,
                training_project=training_project,
                output_root=output_root,
                posteval_root=posteval_root,
                child_pid=child_pid,
                sources=sources,
            ),
        )
    except Exception as error:
        write_json_report(
            status_output,
            _status_payload(
                status="failed",
                detail="heldout_evaluation_supervisor_failed",
                evaluator_project=evaluator_project,
                training_project=training_project,
                output_root=output_root,
                posteval_root=posteval_root,
                child_pid=child_pid,
                sources=sources,
                error=f"{type(error).__name__}: {error}",
            ),
        )
        raise


if __name__ == "__main__":
    main()
