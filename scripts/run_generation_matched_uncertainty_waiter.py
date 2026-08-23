from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import OutputLockError, exclusive_output_lock
from cofitok.reporting import file_sha256, write_json_report

try:
    import build_generation_matched_uncertainty_summary as uncertainty_summary
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import (
        build_generation_matched_uncertainty_summary as uncertainty_summary,
    )


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_matched_uncertainty_waiter"
CACHE_RECEIPT_SCHEMA_VERSION = 1
CACHE_RECEIPT_ROLE = "generation_matched_uncertainty_real_feature_cache"
QUALITY_EXECUTION_ROLE = "stability_full_data_quality_bridge_execution"
QUALITY_EXECUTION_COMPLETED_DETAIL = (
    "quality bridge terminal evidence verified; no larger-training authorization "
    "was created"
)
EXECUTION_MANIFEST_ROLE = (
    "non_authorizing_matched_generation_uncertainty_execution_manifest"
)
FEATURE_CACHE_SUFFIX = "-inception-v3-compat-features-2048.pt"
REAL_FEATURE_CACHE_BYTES = 409_601_577
REAL_FEATURE_CACHE_SHA256 = (
    "20103588dca9ce47bfceef6b68b473fdf4be720f149d1b8bdd96341d27c10dcd"
)
LOCK_CONTENDED_EXIT_CODE = 75
CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "replaces_fid_point_estimates": False,
    "replaces_promotion_gate": False,
    "broad_generation_superiority_claim_allowed": False,
    "training_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "release_allowed": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_identity(
    project: Path,
    *,
    include_untracked: bool = False,
) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status_arguments = ["status", "--porcelain"]
    if not include_untracked:
        status_arguments.append("--untracked-files=no")
    return {
        "revision": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run(*status_arguments)),
    }


def verify_checkout(
    project: Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    label: str,
) -> dict[str, Any]:
    expected = {
        "revision": expected_revision,
        "tree": expected_tree,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    actual = _git_identity(project)
    if actual != expected:
        raise ValueError(f"{label} checkout identity differs")
    return actual


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _source(path: Path) -> dict[str, Any]:
    return file_identity(path)


def _expected_file(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    identity = _source(path)
    if (
        int(identity["bytes"]) != expected_bytes
        or identity["sha256"] != expected_sha256
    ):
        raise ValueError(f"{label} identity differs")
    return identity


def _quality_paths(output_root: Path) -> dict[str, Path]:
    reports = output_root / "reports"
    cofitok = output_root / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
    dense = output_root / "dense_rollout_x0_u2_ema_teacher"
    return {
        "execution_status": reports / "execution_status.json",
        "result": reports / "quality_bridge_result.json",
        "preparation": reports / "preparation.json",
        "launch_receipt": reports / "launch_receipt.json",
        "cofitok_training": cofitok / "training_report.json",
        "dense_training": dense / "training_report.json",
        "training_pair_validation": reports / "training_pair_validation.json",
        "cofitok_training_audit": reports / "cofitok_training_audit.json",
        "dense_training_audit": reports / "dense_training_audit.json",
        "milestone_50000": reports / "milestones" / "step_00050000.json",
        "milestone_100000": reports / "milestones" / "step_00100000.json",
        "cofitok_sampling_preflight": (
            cofitok / "terminal_100k" / "sampling_preflight.json"
        ),
        "dense_sampling_preflight": (
            dense / "terminal_100k" / "sampling_preflight.json"
        ),
        "cofitok_generation": (
            cofitok
            / "terminal_100k"
            / "samples_10000_ddim100_cfg15"
            / "metrics"
            / "generation_metrics_report.json"
        ),
        "dense_generation": (
            dense
            / "terminal_100k"
            / "samples_10000_ddim100_cfg15"
            / "metrics"
            / "generation_metrics_report.json"
        ),
        "cofitok_checkpoint_eval": (
            cofitok
            / "terminal_100k"
            / "checkpoint_eval"
            / "checkpoint_evaluation_report.json"
        ),
        "dense_checkpoint_eval": (
            dense
            / "terminal_100k"
            / "checkpoint_eval"
            / "checkpoint_evaluation_report.json"
        ),
        "class_fidelity_qualification": (
            reports / "class_fidelity" / "qualification_report.json"
        ),
        "cofitok_class_fidelity": (
            cofitok
            / "terminal_100k"
            / "samples_10000_ddim100_cfg15"
            / "class_fidelity"
            / "class_fidelity_report.json"
        ),
        "dense_class_fidelity": (
            dense
            / "terminal_100k"
            / "samples_10000_ddim100_cfg15"
            / "class_fidelity"
            / "class_fidelity_report.json"
        ),
    }


def validate_quality_execution_status(
    report: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    status = str(report.get("status", ""))
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("role") != QUALITY_EXECUTION_ROLE
        or status not in {"running", "failed", "completed"}
        or report.get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or report.get("quality_bridge_only") is not True
        or report.get("full_training_launch_allowed") is not False
        or report.get("full_300k_launch_allowed") is not False
        or report.get("report_is_promotion_gate") is not False
    ):
        raise ValueError("quality-bridge execution status contract differs")
    if (
        status == "completed"
        and report.get("detail") != QUALITY_EXECUTION_COMPLETED_DETAIL
    ):
        raise ValueError("quality-bridge completion detail differs")
    return {
        "status": status,
        "detail": report.get("detail"),
        "updated_at": report.get("updated_at"),
        "exit_code": report.get("exit_code"),
    }


def verify_quality_bridge_result(
    *,
    python_executable: Path,
    quality_project: Path,
    quality_output_root: Path,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    paths = _quality_paths(quality_output_root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "quality-bridge terminal replay inputs are missing: "
            + ", ".join(sorted(missing))
        )
    execution = validate_quality_execution_status(
        read_json_object(
            paths["execution_status"],
            name="quality-bridge execution status",
        ),
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    if execution["status"] != "completed":
        raise ValueError("quality-bridge execution is not completed")

    command = [
        str(python_executable),
        str(quality_project / "scripts" / "verify_generation_quality_bridge_result.py"),
        "--preparation",
        str(paths["preparation"]),
        "--expected-preparation-sha256",
        file_sha256(paths["preparation"]),
        "--launch-receipt",
        str(paths["launch_receipt"]),
        "--expected-launch-receipt-sha256",
        file_sha256(paths["launch_receipt"]),
        "--cofitok-training",
        str(paths["cofitok_training"]),
        "--dense-training",
        str(paths["dense_training"]),
        "--training-pair-validation",
        str(paths["training_pair_validation"]),
        "--cofitok-training-audit",
        str(paths["cofitok_training_audit"]),
        "--dense-training-audit",
        str(paths["dense_training_audit"]),
        "--milestone-50000",
        str(paths["milestone_50000"]),
        "--milestone-100000",
        str(paths["milestone_100000"]),
        "--cofitok-sampling-preflight",
        str(paths["cofitok_sampling_preflight"]),
        "--dense-sampling-preflight",
        str(paths["dense_sampling_preflight"]),
        "--cofitok-generation",
        str(paths["cofitok_generation"]),
        "--dense-generation",
        str(paths["dense_generation"]),
        "--cofitok-checkpoint-eval",
        str(paths["cofitok_checkpoint_eval"]),
        "--dense-checkpoint-eval",
        str(paths["dense_checkpoint_eval"]),
        "--class-fidelity-qualification",
        str(paths["class_fidelity_qualification"]),
        "--cofitok-class-fidelity",
        str(paths["cofitok_class_fidelity"]),
        "--dense-class-fidelity",
        str(paths["dense_class_fidelity"]),
        "--expected-revision",
        expected_revision,
        "--expected-branch",
        expected_branch,
        "--result",
        str(paths["result"]),
        "--expected-result-sha256",
        file_sha256(paths["result"]),
    ]
    environment = os.environ.copy()
    python_roots = [str(quality_project), str(quality_project / "src")]
    inherited_pythonpath = environment.get("PYTHONPATH")
    if inherited_pythonpath:
        python_roots.append(inherited_pythonpath)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_roots),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=quality_project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(
            "quality-bridge terminal result replay failed: " + detail.strip()
        )
    try:
        replay = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError(
            "quality-bridge result verifier output is malformed"
        ) from error
    boundary = replay.get("authorization_boundary")
    if (
        replay.get("status") != "verified"
        or replay.get("result") != _source(paths["result"])
        or not isinstance(boundary, Mapping)
        or boundary.get("full_training_launch_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
        or boundary.get("report_is_promotion_gate") is not False
    ):
        raise ValueError("quality-bridge result replay boundary differs")
    return {
        "status": "verified",
        "execution_status": _source(paths["execution_status"]),
        "result": _source(paths["result"]),
        "source_files": {
            name: _source(path)
            for name, path in paths.items()
            if name not in {"execution_status", "result"}
        },
        "quality_screen": replay.get("quality_screen"),
        "authorization_boundary": dict(boundary),
    }


def _process_rows() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,etimes=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4 or not all(value.isdigit() for value in fields[:3]):
            continue
        pid = int(fields[0])
        cwd = ""
        try:
            cwd = Path(os.readlink(f"/proc/{pid}/cwd")).as_posix()
        except OSError:
            pass
        rows.append(
            {
                "pid": pid,
                "ppid": int(fields[1]),
                "elapsed_seconds": int(fields[2]),
                "command": fields[3],
                "cwd": cwd,
            }
        )
    return rows


def quality_processes(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality_project: Path,
    quality_output_root: Path,
) -> list[dict[str, Any]]:
    project_text = quality_project.as_posix()
    output_text = quality_output_root.as_posix()
    runbook_name = "generation_stability_full_data_quality_bridge_100k_execute.sh"
    markers = (
        runbook_name,
        "scripts/train_generation.py",
        "scripts/run_generation_training_watchdog.py",
        "scripts/monitor_generation_pair.py",
        "scripts/select_generation_training_runtime.py",
        "scripts/preflight_generation_sampling.py",
        "scripts/generate_samples.py",
        "scripts/evaluate_generation_metrics.py",
        "scripts/evaluate_generation_checkpoint.py",
        "scripts/evaluate_generation_class_fidelity.py",
        "scripts/build_generation_quality_bridge_result.py",
        "scripts/verify_generation_quality_bridge_result.py",
    )
    matches = []
    for raw in rows:
        pid = int(raw.get("pid", -1))
        command = str(raw.get("command", ""))
        cwd = str(raw.get("cwd", ""))
        if pid == os.getpid():
            continue
        relevant = cwd == project_text or (
            output_text in command and any(marker in command for marker in markers)
        )
        if relevant:
            matches.append(dict(raw))
    return matches


def uncertainty_processes(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
) -> list[dict[str, Any]]:
    output_text = output_root.as_posix()
    markers = (
        "audit_generation_matched_uncertainty.py",
        "build_generation_matched_uncertainty_summary.py",
        "run_generation_stage_once.py",
    )
    return [
        dict(row)
        for row in rows
        if int(row.get("pid", -1)) != os.getpid()
        and output_text in str(row.get("command", ""))
        and any(marker in str(row.get("command", "")) for marker in markers)
    ]


def _gpu_compute_rows() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        rows.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_memory_mib": (
                    int(fields[2]) if fields[2].isdigit() else fields[2]
                ),
            }
        )
    return rows


def next_idle_count(
    current: int,
    *,
    gpu_rows: Sequence[Mapping[str, Any]],
    competing_processes: Sequence[Mapping[str, Any]],
) -> int:
    return 0 if gpu_rows or competing_processes else current + 1


def _stat_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        **_source(path),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _validate_cache_receipt(
    receipt: Mapping[str, Any],
    *,
    source: Path,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    source_identity = _stat_identity(source)
    destination_identity = _stat_identity(destination)
    if (
        int(receipt.get("schema_version", -1)) != CACHE_RECEIPT_SCHEMA_VERSION
        or receipt.get("role") != CACHE_RECEIPT_ROLE
        or receipt.get("status") != "verified"
        or receipt.get("method")
        not in {"hardlink", "cross_device_copy", "recovered_existing"}
        or receipt.get("source") != source_identity
        or receipt.get("destination") != destination_identity
        or receipt.get("expected")
        != {"bytes": expected_bytes, "sha256": expected_sha256}
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("matched uncertainty real-feature cache receipt differs")
    return dict(receipt)


def prepare_real_feature_cache(
    *,
    source: Path,
    destination: Path,
    receipt_path: Path,
    expected_bytes: int = REAL_FEATURE_CACHE_BYTES,
    expected_sha256: str = REAL_FEATURE_CACHE_SHA256,
) -> dict[str, Any]:
    source = reject_symlink_chain(source, name="real feature cache source")
    destination = reject_symlink_chain(
        destination,
        name="real feature cache destination",
    )
    receipt_path = reject_symlink_chain(
        receipt_path,
        name="real feature cache receipt",
    )
    _expected_file(
        source,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        label="real feature cache source",
    )
    if receipt_path.is_file():
        return _validate_cache_receipt(
            read_json_object(receipt_path, name="real feature cache receipt"),
            source=source,
            destination=destination,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    method = "recovered_existing"
    if not destination.exists():
        try:
            os.link(source, destination)
            method = "hardlink"
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            temporary = destination.with_name(f".{destination.name}.copy.{os.getpid()}")
            if temporary.exists() or temporary.is_symlink():
                raise FileExistsError(temporary)
            try:
                with (
                    source.open("rb") as source_handle,
                    temporary.open("xb") as destination_handle,
                ):
                    shutil.copyfileobj(
                        source_handle,
                        destination_handle,
                        length=8 * 1024 * 1024,
                    )
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
                _expected_file(
                    temporary,
                    expected_bytes=expected_bytes,
                    expected_sha256=expected_sha256,
                    label="copied real feature cache",
                )
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
            method = "cross_device_copy"

    _expected_file(
        destination,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        label="real feature cache destination",
    )
    receipt = {
        "schema_version": CACHE_RECEIPT_SCHEMA_VERSION,
        "role": CACHE_RECEIPT_ROLE,
        "status": "verified",
        "method": method,
        "source": _stat_identity(source),
        "destination": _stat_identity(destination),
        "same_inode": (
            source.stat().st_dev == destination.stat().st_dev
            and source.stat().st_ino == destination.stat().st_ino
        ),
        "expected": {"bytes": expected_bytes, "sha256": expected_sha256},
        "claim_boundary": CLAIM_BOUNDARY,
        "created_at": _utc_now(),
    }
    write_json_report(receipt_path, receipt)
    return _validate_cache_receipt(
        receipt,
        source=source,
        destination=destination,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )


def _feature_cache_path(cache_root: Path, cache_name: str) -> Path:
    return cache_root / f"{cache_name}{FEATURE_CACHE_SUFFIX}"


def validate_execution_manifest(
    path: Path,
    *,
    expected_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    identity = _expected_file(
        path,
        expected_bytes=path.stat().st_size,
        expected_sha256=expected_sha256,
        label="matched uncertainty execution manifest",
    )
    report = read_json_object(path, name="matched uncertainty execution manifest")
    arguments = report.get("arguments")
    expected = report.get("expected")
    sources = report.get("source_files")
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("role") != EXECUTION_MANIFEST_ROLE
        or report.get("claim_boundary")
        != {
            key: CLAIM_BOUNDARY[key]
            for key in (
                "diagnostic_non_authorizing",
                "replaces_fid_point_estimates",
                "replaces_promotion_gate",
                "broad_generation_superiority_claim_allowed",
                "full_training_launch_allowed",
                "full_300k_launch_allowed",
            )
        }
        or not isinstance(arguments, Mapping)
        or not isinstance(expected, Mapping)
        or not isinstance(sources, Mapping)
        or arguments.get("cpu") is not False
    ):
        raise ValueError("matched uncertainty execution manifest contract differs")
    output = Path(str(arguments.get("output", ""))).resolve()
    cache_root = Path(str(arguments.get("cache_root", ""))).resolve()
    if (
        not _is_within(output, output_root)
        or cache_root != (output_root / "feature_cache").resolve()
    ):
        raise ValueError("matched uncertainty manifest output scope differs")
    bound_sources: list[dict[str, Any]] = []
    for name, record in sources.items():
        if not isinstance(record, Mapping):
            raise TypeError(f"matched uncertainty source record is malformed: {name}")
        source_path = Path(str(record.get("path", ""))).resolve()
        actual = _source(source_path)
        if actual != dict(record):
            raise ValueError(f"matched uncertainty source identity differs: {name}")
        bound_sources.append(actual)
    matched = expected.get("matched_sampling")
    real_set = expected.get("real_set")
    sample_sets = expected.get("sample_sets")
    matched_sample_count = (
        int(matched.get("sample_count", -1))
        if isinstance(matched, Mapping)
        else -1
    )
    min_samples = arguments.get("min_samples", matched_sample_count)
    block_size = arguments.get("block_size", 500)
    real_fold_count = arguments.get("real_fold_count", 1)
    real_image_count = (
        real_set.get("image_count") if isinstance(real_set, Mapping) else None
    )
    if (
        not isinstance(matched, Mapping)
        or type(min_samples) is not int
        or min_samples < 1
        or type(block_size) is not int
        or block_size < 2
        or type(real_fold_count) is not int
        or real_fold_count < 1
        or int(matched.get("sample_count", -1)) != min_samples
        or int(matched.get("end_index_exclusive", -1))
        - int(matched.get("start_index", -1))
        != min_samples
        or min_samples % block_size != 0
        or min_samples // block_size < 8
        or not isinstance(real_set, Mapping)
        or (
            real_image_count is not None
            and int(real_image_count) < min_samples * real_fold_count
        )
        or not isinstance(sample_sets, Mapping)
        or not isinstance(sample_sets.get("cofitok"), Mapping)
        or not isinstance(sample_sets.get("dense_identity"), Mapping)
    ):
        raise ValueError("matched uncertainty manifest scientific scope differs")
    real_cache_name = (
        str(arguments["real_cache_name"]) + "__cofitok_" + str(real_set["sha256"])[:16]
    )
    cache_names = {
        "real": real_cache_name,
        "cofitok": "matched_uncertainty_cofitok_"
        + str(sample_sets["cofitok"]["sha256"])[:16],
        "dense_identity": "matched_uncertainty_dense_"
        + str(sample_sets["dense_identity"]["sha256"])[:16],
    }
    return {
        "identity": identity,
        "stream_id": report.get("stream_id"),
        "report": report,
        "output": output,
        "cache_root": cache_root,
        "cache_names": cache_names,
        "bound_sources": sorted(bound_sources, key=lambda row: row["path"]),
        "start_index": int(matched["start_index"]),
        "end_index_exclusive": int(matched["end_index_exclusive"]),
        "cofitok_checkpoint_sha256": sample_sets["cofitok"]["checkpoint_sha256"],
        "dense_checkpoint_sha256": sample_sets["dense_identity"]["checkpoint_sha256"],
        "checkpoint_step": int(sample_sets["cofitok"]["checkpoint_step"]),
        "real_set_sha256": real_set["sha256"],
    }


def validate_manifest_pair(
    formal: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> None:
    if (
        formal["stream_id"] == confirmation["stream_id"]
        or formal["end_index_exclusive"] > confirmation["start_index"]
        or formal["cofitok_checkpoint_sha256"]
        != confirmation["cofitok_checkpoint_sha256"]
        or formal["dense_checkpoint_sha256"] != confirmation["dense_checkpoint_sha256"]
        or formal["checkpoint_step"] != confirmation["checkpoint_step"]
        or formal["real_set_sha256"] != confirmation["real_set_sha256"]
        or formal["cache_root"] != confirmation["cache_root"]
        or formal["output"] == confirmation["output"]
    ):
        raise ValueError(
            "matched uncertainty execution manifests are not disjoint matched streams"
        )


@dataclass(frozen=True)
class StageSpec:
    name: str
    state: Path
    command: tuple[str, ...]
    input_files: tuple[Path, ...]
    output_files: tuple[Path, ...]
    gpu_required: bool


def audit_stage_spec(
    *,
    name: str,
    evaluator_project: Path,
    python_executable: Path,
    manifest: Mapping[str, Any],
    real_cache: Path,
    stage_root: Path,
) -> StageSpec:
    cache_root = Path(manifest["cache_root"])
    output_files = (
        Path(manifest["output"]),
        _feature_cache_path(cache_root, manifest["cache_names"]["cofitok"]),
        _feature_cache_path(
            cache_root,
            manifest["cache_names"]["dense_identity"],
        ),
    )
    return StageSpec(
        name=name,
        state=stage_root / f"{name}.stage.json",
        command=(
            str(python_executable),
            str(
                evaluator_project
                / "scripts"
                / "audit_generation_matched_uncertainty.py"
            ),
            "--execution-manifest",
            str(manifest["identity"]["path"]),
        ),
        input_files=tuple(
            [Path(manifest["identity"]["path"]), real_cache]
            + [Path(row["path"]) for row in manifest["bound_sources"]]
        ),
        output_files=output_files,
        gpu_required=True,
    )


def summary_stage_spec(
    *,
    evaluator_project: Path,
    python_executable: Path,
    formal_output: Path,
    confirmation_output: Path,
    summary_output: Path,
    stage_root: Path,
) -> StageSpec:
    return StageSpec(
        name="summary",
        state=stage_root / "summary.stage.json",
        command=(
            str(python_executable),
            str(
                evaluator_project
                / "scripts"
                / "build_generation_matched_uncertainty_summary.py"
            ),
            "--audit",
            str(formal_output),
            "--audit",
            str(confirmation_output),
            "--output",
            str(summary_output),
        ),
        input_files=(formal_output, confirmation_output),
        output_files=(summary_output,),
        gpu_required=False,
    )


def run_stage(
    spec: StageSpec,
    *,
    evaluator_project: Path,
    python_executable: Path,
) -> dict[str, Any]:
    command = [
        str(python_executable),
        str(evaluator_project / "scripts" / "run_generation_stage_once.py"),
        "--state",
        str(spec.state),
        "--project",
        str(evaluator_project),
        "--cwd",
        str(evaluator_project),
    ]
    for path in spec.input_files:
        command.extend(("--input-file", str(path)))
    for path in spec.output_files:
        command.extend(("--output-file", str(path)))
    command.extend(("--", *spec.command))
    environment = os.environ.copy()
    python_roots = [str(evaluator_project), str(evaluator_project / "src")]
    inherited_pythonpath = environment.get("PYTHONPATH")
    if inherited_pythonpath:
        python_roots.append(inherited_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_roots)
    completed = subprocess.run(
        command,
        cwd=evaluator_project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(f"matched uncertainty {spec.name} stage failed: {detail}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError(
            f"matched uncertainty {spec.name} stage output is malformed"
        ) from error
    if result.get("status") != "completed":
        raise ValueError(f"matched uncertainty {spec.name} stage did not complete")
    return result


def validate_audit_output(
    path: Path,
    *,
    manifest_identity: Mapping[str, Any],
    evaluator_git: Mapping[str, Any],
) -> dict[str, Any]:
    report = read_json_object(path, name="matched uncertainty audit")
    row = uncertainty_summary._validate_audit(report, path)
    if row["execution_manifest"]["source"] != dict(manifest_identity) or report.get(
        "git"
    ) != {
        "revision": evaluator_git["revision"],
        "branch": evaluator_git["branch"],
        "tracked_dirty": False,
    }:
        raise ValueError("matched uncertainty audit source or evaluator differs")
    return {
        "status": report["status"],
        "decision": report["decision"],
        "advantage_supported": report["advantage_supported"],
        "source": _source(path),
        "stream": row,
        "claim_boundary": report["claim_boundary"],
    }


def validate_summary_output(
    path: Path,
    *,
    evaluator_git: Mapping[str, Any],
) -> dict[str, Any]:
    report = read_json_object(path, name="matched uncertainty summary")
    if (
        int(report.get("schema_version", -1))
        != uncertainty_summary.REPORT_SCHEMA_VERSION
        or report.get("role") != uncertainty_summary.REPORT_ROLE
        or report.get("status") not in {"pass", "hold"}
        or report.get("claim_boundary") != uncertainty_summary.CLAIM_BOUNDARY
        or report.get("git")
        != {
            "revision": evaluator_git["revision"],
            "branch": evaluator_git["branch"],
            "tracked_dirty": False,
        }
        or (report.get("status") == "pass")
        != (report.get("repeated_advantage_supported") is True)
    ):
        raise ValueError("matched uncertainty repeated summary contract differs")
    return {
        "status": report["status"],
        "decision": report["decision"],
        "repeated_advantage_supported": report["repeated_advantage_supported"],
        "checks": report.get("checks"),
        "source": _source(path),
        "claim_boundary": report["claim_boundary"],
    }


def _status(
    *,
    status: str,
    detail: str,
    expected: Mapping[str, Any],
    phase: str,
    quality: Mapping[str, Any] | None = None,
    quality_process_rows: Sequence[Mapping[str, Any]] | None = None,
    gpu_rows: Sequence[Mapping[str, Any]] | None = None,
    competing_process_rows: Sequence[Mapping[str, Any]] | None = None,
    idle_polls: int = 0,
    cache: Mapping[str, Any] | None = None,
    stages: Mapping[str, Any] | None = None,
    audits: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": WAITER_SCHEMA_VERSION,
        "role": WAITER_ROLE,
        "status": status,
        "detail": detail,
        "phase": phase,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "expected": dict(expected),
        "quality_bridge": dict(quality) if quality is not None else None,
        "quality_processes": [dict(row) for row in quality_process_rows or []],
        "gpu_compute_rows": [dict(row) for row in gpu_rows or []],
        "competing_uncertainty_processes": [
            dict(row) for row in competing_process_rows or []
        ],
        "idle_polls": idle_polls,
        "required_idle_polls": int(expected["required_idle_polls"]),
        "real_feature_cache": dict(cache) if cache is not None else None,
        "stages": dict(stages or {}),
        "audits": dict(audits or {}),
        "summary": dict(summary) if summary is not None else None,
        "claim_boundary": CLAIM_BOUNDARY,
        "updated_at": _utc_now(),
    }


def _write_pid(path: Path, *, expected: Mapping[str, Any]) -> None:
    write_json_report(
        path,
        {
            "schema_version": 1,
            "role": WAITER_ROLE,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "expected_control_revision": expected["control_git"]["revision"],
            "started_at": _utc_now(),
        },
    )


def _remove_pid(path: Path) -> None:
    if not path.is_file():
        return
    try:
        payload = read_json_object(path, name="matched uncertainty waiter PID")
    except ValueError:
        return
    if int(payload.get("pid", -1)) == os.getpid():
        path.unlink()


def _sleep_until_next_poll(
    *,
    deadline: float,
    poll_seconds: float,
) -> bool:
    if time.monotonic() >= deadline:
        return False
    time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    return time.monotonic() < deadline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the full-data 100K quality bridge and a verified idle GPU "
            "slot, then run only the non-authorizing matched uncertainty audits."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--evaluator-project", type=Path, required=True)
    parser.add_argument("--quality-project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--expected-formal-manifest-sha256", required=True)
    parser.add_argument("--expected-confirmation-manifest-sha256", required=True)
    parser.add_argument("--real-feature-cache-source", type=Path, required=True)
    parser.add_argument(
        "--expected-real-feature-cache-bytes",
        type=int,
        default=REAL_FEATURE_CACHE_BYTES,
    )
    parser.add_argument(
        "--expected-real-feature-cache-sha256",
        default=REAL_FEATURE_CACHE_SHA256,
    )
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-evaluator-revision", required=True)
    parser.add_argument("--expected-evaluator-tree", required=True)
    parser.add_argument("--expected-evaluator-branch", default="")
    parser.add_argument("--expected-quality-revision", required=True)
    parser.add_argument("--expected-quality-tree", required=True)
    parser.add_argument("--expected-quality-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    args = parser.parse_args()
    if (
        args.poll_seconds <= 0
        or args.required_idle_polls < 1
        or args.timeout_seconds <= 0
        or args.expected_real_feature_cache_bytes < 1
    ):
        parser.error("matched uncertainty waiter timing or cache arguments are invalid")
    return args


def _run_locked(args: argparse.Namespace) -> int:
    project = reject_symlink_chain(args.project, name="control checkout").resolve()
    evaluator_project = reject_symlink_chain(
        args.evaluator_project,
        name="uncertainty evaluator checkout",
    ).resolve()
    quality_project = reject_symlink_chain(
        args.quality_project,
        name="quality-bridge checkout",
    ).resolve()
    quality_output_root = reject_symlink_chain(
        args.quality_output_root,
        name="quality-bridge output root",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="uncertainty output root",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="uncertainty waiter status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="uncertainty waiter PID file",
    ).resolve()
    python_executable = reject_symlink_chain(
        args.python_executable,
        name="uncertainty Python executable",
    ).resolve()
    if (
        not _is_within(status_output, output_root)
        or not _is_within(pid_file, output_root)
        or not python_executable.is_file()
    ):
        raise ValueError("matched uncertainty waiter output or Python scope differs")

    def verify_checkouts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return (
            verify_checkout(
                project,
                expected_revision=args.expected_control_revision,
                expected_tree=args.expected_control_tree,
                expected_branch=args.expected_control_branch,
                label="matched uncertainty control",
            ),
            verify_checkout(
                evaluator_project,
                expected_revision=args.expected_evaluator_revision,
                expected_tree=args.expected_evaluator_tree,
                expected_branch=args.expected_evaluator_branch,
                label="matched uncertainty evaluator",
            ),
            verify_checkout(
                quality_project,
                expected_revision=args.expected_quality_revision,
                expected_tree=args.expected_quality_tree,
                expected_branch=args.expected_quality_branch,
                label="quality bridge",
            ),
        )

    control_git, evaluator_git, quality_git = verify_checkouts()
    formal = validate_execution_manifest(
        args.formal_manifest.resolve(),
        expected_sha256=args.expected_formal_manifest_sha256,
        output_root=output_root,
    )
    confirmation = validate_execution_manifest(
        args.confirmation_manifest.resolve(),
        expected_sha256=args.expected_confirmation_manifest_sha256,
        output_root=output_root,
    )
    validate_manifest_pair(formal, confirmation)
    real_cache = _feature_cache_path(
        formal["cache_root"],
        formal["cache_names"]["real"],
    )
    expected = {
        "control_git": control_git,
        "evaluator_git": evaluator_git,
        "quality_git": quality_git,
        "quality_output_root": quality_output_root.as_posix(),
        "output_root": output_root.as_posix(),
        "formal_manifest": formal["identity"],
        "confirmation_manifest": confirmation["identity"],
        "real_feature_cache_source": {
            "path": args.real_feature_cache_source.resolve().as_posix(),
            "bytes": args.expected_real_feature_cache_bytes,
            "sha256": args.expected_real_feature_cache_sha256,
        },
        "required_idle_polls": args.required_idle_polls,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_pid(pid_file, expected=expected)
    deadline = time.monotonic() + args.timeout_seconds
    quality_verification: dict[str, Any] | None = None
    cache_receipt: dict[str, Any] | None = None
    stage_results: dict[str, Any] = {}
    audit_results: dict[str, Any] = {}
    summary_result: dict[str, Any] | None = None

    def publish(
        *,
        status: str,
        detail: str,
        phase: str,
        quality_process_rows: Sequence[Mapping[str, Any]] | None = None,
        gpu_rows: Sequence[Mapping[str, Any]] | None = None,
        competing_process_rows: Sequence[Mapping[str, Any]] | None = None,
        idle_polls: int = 0,
    ) -> None:
        write_json_report(
            status_output,
            _status(
                status=status,
                detail=detail,
                phase=phase,
                expected=expected,
                quality=quality_verification,
                quality_process_rows=quality_process_rows,
                gpu_rows=gpu_rows,
                competing_process_rows=competing_process_rows,
                idle_polls=idle_polls,
                cache=cache_receipt,
                stages=stage_results,
                audits=audit_results,
                summary=summary_result,
            ),
        )

    quality_paths = _quality_paths(quality_output_root)
    while quality_verification is None:
        verify_checkouts()
        if not quality_paths["execution_status"].is_file():
            publish(
                status="waiting",
                detail="waiting_for_quality_bridge_execution_status",
                phase="quality_bridge",
            )
        else:
            execution = validate_quality_execution_status(
                read_json_object(
                    quality_paths["execution_status"],
                    name="quality-bridge execution status",
                ),
                expected_revision=args.expected_quality_revision,
                expected_branch=args.expected_quality_branch,
            )
            if (
                execution["status"] != "completed"
                or not quality_paths["result"].is_file()
            ):
                publish(
                    status="waiting",
                    detail="waiting_for_quality_bridge_terminal_result",
                    phase="quality_bridge",
                )
            else:
                active = quality_processes(
                    _process_rows(),
                    quality_project=quality_project,
                    quality_output_root=quality_output_root,
                )
                if active:
                    publish(
                        status="waiting",
                        detail="waiting_for_quality_bridge_process_exit",
                        phase="quality_bridge",
                        quality_process_rows=active,
                    )
                else:
                    quality_verification = verify_quality_bridge_result(
                        python_executable=python_executable,
                        quality_project=quality_project,
                        quality_output_root=quality_output_root,
                        expected_revision=args.expected_quality_revision,
                        expected_branch=args.expected_quality_branch,
                    )
                    publish(
                        status="waiting",
                        detail="quality_bridge_terminal_result_verified",
                        phase="cache_preparation",
                    )
                    break
        if not _sleep_until_next_poll(
            deadline=deadline,
            poll_seconds=args.poll_seconds,
        ):
            publish(
                status="failed",
                detail="matched_uncertainty_waiter_timeout",
                phase="quality_bridge",
            )
            return 1

    cache_receipt = prepare_real_feature_cache(
        source=args.real_feature_cache_source.resolve(),
        destination=real_cache,
        receipt_path=output_root / "reports" / "real_feature_cache_receipt.json",
        expected_bytes=args.expected_real_feature_cache_bytes,
        expected_sha256=args.expected_real_feature_cache_sha256,
    )
    stage_root = output_root / "stage_receipts"
    formal_stage = audit_stage_spec(
        name="formal",
        evaluator_project=evaluator_project,
        python_executable=python_executable,
        manifest=formal,
        real_cache=real_cache,
        stage_root=stage_root,
    )
    confirmation_stage = audit_stage_spec(
        name="confirmation",
        evaluator_project=evaluator_project,
        python_executable=python_executable,
        manifest=confirmation,
        real_cache=real_cache,
        stage_root=stage_root,
    )
    summary_output = output_root / "matched_uncertainty_summary.json"
    summary_stage = summary_stage_spec(
        evaluator_project=evaluator_project,
        python_executable=python_executable,
        formal_output=formal["output"],
        confirmation_output=confirmation["output"],
        summary_output=summary_output,
        stage_root=stage_root,
    )

    def assert_quality_sources_unchanged() -> None:
        if quality_verification is None:
            raise RuntimeError("quality-bridge verification is absent")
        if (
            _source(quality_paths["execution_status"])
            != quality_verification["execution_status"]
            or _source(quality_paths["result"]) != quality_verification["result"]
        ):
            raise ValueError("quality-bridge terminal sources changed after replay")

    for spec, manifest in (
        (formal_stage, formal),
        (confirmation_stage, confirmation),
    ):
        idle_polls = 0
        while True:
            verify_checkouts()
            assert_quality_sources_unchanged()
            processes = _process_rows()
            competing = uncertainty_processes(processes, output_root=output_root)
            gpu_rows = _gpu_compute_rows()
            idle_polls = next_idle_count(
                idle_polls,
                gpu_rows=gpu_rows,
                competing_processes=competing,
            )
            if idle_polls < args.required_idle_polls:
                publish(
                    status="waiting",
                    detail=(
                        "waiting_for_gpu_idle"
                        if gpu_rows or competing
                        else "confirming_gpu_idle"
                    ),
                    phase=f"{spec.name}_gpu_slot",
                    gpu_rows=gpu_rows,
                    competing_process_rows=competing,
                    idle_polls=idle_polls,
                )
            else:
                final_gpu_rows = _gpu_compute_rows()
                final_processes = uncertainty_processes(
                    _process_rows(),
                    output_root=output_root,
                )
                if final_gpu_rows or final_processes:
                    idle_polls = 0
                    publish(
                        status="waiting",
                        detail="gpu_slot_changed_during_final_recheck",
                        phase=f"{spec.name}_gpu_slot",
                        gpu_rows=final_gpu_rows,
                        competing_process_rows=final_processes,
                    )
                else:
                    publish(
                        status="running",
                        detail=f"running_{spec.name}_matched_uncertainty_audit",
                        phase=spec.name,
                        idle_polls=idle_polls,
                    )
                    stage_results[spec.name] = run_stage(
                        spec,
                        evaluator_project=evaluator_project,
                        python_executable=python_executable,
                    )
                    audit_results[spec.name] = validate_audit_output(
                        Path(manifest["output"]),
                        manifest_identity=manifest["identity"],
                        evaluator_git=evaluator_git,
                    )
                    publish(
                        status="waiting",
                        detail=f"{spec.name}_matched_uncertainty_audit_completed",
                        phase=(
                            "confirmation_gpu_slot"
                            if spec.name == "formal"
                            else "summary"
                        ),
                    )
                    break
            if not _sleep_until_next_poll(
                deadline=deadline,
                poll_seconds=args.poll_seconds,
            ):
                publish(
                    status="failed",
                    detail="matched_uncertainty_waiter_timeout",
                    phase=f"{spec.name}_gpu_slot",
                    idle_polls=idle_polls,
                )
                return 1

    verify_checkouts()
    assert_quality_sources_unchanged()
    publish(
        status="running",
        detail="building_repeated_matched_uncertainty_summary",
        phase="summary",
    )
    stage_results["summary"] = run_stage(
        summary_stage,
        evaluator_project=evaluator_project,
        python_executable=python_executable,
    )
    summary_result = validate_summary_output(
        summary_output,
        evaluator_git=evaluator_git,
    )
    terminal_status = str(summary_result["status"])
    publish(
        status=terminal_status,
        detail=str(summary_result["decision"]),
        phase="completed",
    )
    return 0


def run_waiter(args: argparse.Namespace) -> int:
    output_root = reject_symlink_chain(
        args.output_root,
        name="uncertainty output root",
    ).resolve()
    with exclusive_output_lock(output_root, role=WAITER_ROLE):
        try:
            return _run_locked(args)
        except Exception as error:
            status_output = reject_symlink_chain(
                args.status_output,
                name="uncertainty waiter status",
            ).resolve()
            if _is_within(status_output, output_root):
                write_json_report(
                    status_output,
                    {
                        "schema_version": WAITER_SCHEMA_VERSION,
                        "role": WAITER_ROLE,
                        "status": "failed",
                        "detail": f"{type(error).__name__}: {error}",
                        "phase": "failed",
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                        "claim_boundary": CLAIM_BOUNDARY,
                        "updated_at": _utc_now(),
                    },
                )
            raise
        finally:
            _remove_pid(args.pid_file.resolve())


def main() -> int:
    args = _parse_args()
    try:
        return run_waiter(args)
    except OutputLockError as error:
        print(str(error), file=sys.stderr)
        return LOCK_CONTENDED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
