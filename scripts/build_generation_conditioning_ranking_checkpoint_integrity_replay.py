from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report
from scripts.audit_generation_training_progress import audit_progress


RUNS = (
    "control_cofitok",
    "ranked_cofitok",
    "control_dense_identity",
    "ranked_dense_identity",
)
EXPECTED_STEPS = 1_000
CHECKPOINT_INTERVAL = 250
EVALUATION_INTERVAL = 250
REQUIRED_CHECKPOINT_STEPS = (500, 750, 1_000)
SAMPLES_PER_STEP = 64
SOURCE_PATHS = {
    "preparation": "reports/preparation.json",
    "execution_authorization": "reports/execution_authorization.json",
    "training_status": "reports/training_status.json",
    "postevaluation": "reports/conditioning_ranking_posteval_v1/postevaluation.json",
}


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Evidence source must be a regular file: {path}")
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _stat_identity(path: Path) -> dict[str, int]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Checkpoint payload must be a regular file: {path}")
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
    }


def _git(project: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _code_identity(project: Path) -> dict[str, Any]:
    return {
        "revision": _git(project, "rev-parse", "HEAD"),
        "tree": _git(project, "rev-parse", "HEAD^{tree}"),
        "branch": _git(project, "branch", "--show-current"),
        "tracked_dirty": bool(
            _git(project, "status", "--porcelain", "--untracked-files=no")
        ),
    }


def _source_inventory(root: Path) -> dict[str, dict[str, Any]]:
    paths: dict[str, Path] = {
        name: root / relative for name, relative in SOURCE_PATHS.items()
    }
    for run in RUNS:
        run_dir = root / run
        paths[f"{run}.metrics"] = run_dir / "train_metrics.jsonl"
        paths[f"{run}.latest"] = run_dir / "latest.json"
        paths[f"{run}.training_report"] = run_dir / "training_report.json"
        paths[f"{run}.training_audit"] = root / "reports" / f"{run}_training_audit.json"
        for step in REQUIRED_CHECKPOINT_STEPS:
            checkpoint = run_dir / f"checkpoint_step_{step:08d}.pt"
            paths[f"{run}.sidecar.{step}"] = checkpoint.with_name(
                f"{checkpoint.name}.integrity.json"
            )
    return {name: _identity(path) for name, path in sorted(paths.items())}


def _payload_inventory(root: Path) -> dict[str, dict[str, int]]:
    return {
        f"{run}.{step}": _stat_identity(
            root / run / f"checkpoint_step_{step:08d}.pt"
        )
        for run in RUNS
        for step in REQUIRED_CHECKPOINT_STEPS
    }


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"metrics row {line_number} is not an object: {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"metrics are empty: {path}")
    return rows


def _validate_source_reports(
    root: Path,
    *,
    expected_sha256: Mapping[str, str],
    training_revision: str,
    training_tree: str,
    training_branch: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reports = {
        name: _read_json(root / relative, label=name)
        for name, relative in SOURCE_PATHS.items()
    }
    identities = {
        name: _identity(root / SOURCE_PATHS[name]) for name in SOURCE_PATHS
    }
    if set(expected_sha256) != set(SOURCE_PATHS):
        raise ValueError("Expected source hashes must name every canonical source report")
    for name, expected in expected_sha256.items():
        if identities[name]["sha256"] != expected:
            raise ValueError(f"{name} SHA256 differs from the expected completed screen")

    preparation = reports["preparation"]
    authorization = reports["execution_authorization"]
    training_status = reports["training_status"]
    postevaluation = reports["postevaluation"]
    expected_git = {
        "revision": training_revision,
        "branch": training_branch,
        "tracked_dirty": False,
    }
    if (
        preparation.get("status") != "pass"
        or preparation.get("git") != expected_git
        or preparation.get("tree") != training_tree
    ):
        raise ValueError("Preparation does not bind the expected clean training source")
    if (
        authorization.get("status") != "authorized"
        or authorization.get("generation_advantage_proven") is not False
        or authorization.get("claim_boundary", {}).get("diagnostic_non_authorizing")
        is not True
        or authorization.get("claim_boundary", {}).get("followup_training_allowed")
        is not False
    ):
        raise ValueError("Execution authorization boundary differs")
    if (
        training_status.get("status") != "completed"
        or training_status.get("revision") != training_revision
        or training_status.get("generation_advantage_proven") is not False
    ):
        raise ValueError("Training status does not describe the completed four-arm screen")
    decision = postevaluation.get("decision")
    if (
        postevaluation.get("status") != "completed"
        or postevaluation.get("git") != expected_git
        or not isinstance(decision, Mapping)
        or decision.get("shared_semantic_alignment_recovery_supported") is not False
        or decision.get("cofitok_specific_advantage_claim_allowed") is not False
    ):
        raise ValueError("Postevaluation decision differs from the held result")
    return identities, dict(decision)


def _validate_run(
    run_dir: Path,
    *,
    training_revision: str,
    training_branch: str,
) -> dict[str, Any]:
    audit = audit_progress(
        run_dir,
        expected_steps=EXPECTED_STEPS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        evaluation_interval=EVALUATION_INTERVAL,
        required_checkpoint_steps=REQUIRED_CHECKPOINT_STEPS,
        integrity_policy="required",
    )
    required = audit.get("checkpoint", {}).get("required_integrity", {})
    checkpoints = required.get("checkpoints", [])
    if (
        audit.get("status") != "complete"
        or audit.get("issues") != []
        or audit.get("warnings") != []
        or required.get("status") != "verified"
        or required.get("requested_steps") != list(REQUIRED_CHECKPOINT_STEPS)
        or required.get("reached_steps") != list(REQUIRED_CHECKPOINT_STEPS)
        or [item.get("step") for item in checkpoints]
        != list(REQUIRED_CHECKPOINT_STEPS)
        or any(item.get("status") != "verified" for item in checkpoints)
    ):
        raise ValueError(f"Required checkpoint audit failed for {run_dir.name}")

    latest_path = run_dir / "latest.json"
    training_report_path = run_dir / "training_report.json"
    metrics_path = run_dir / "train_metrics.jsonl"
    latest = _read_json(latest_path, label=f"{run_dir.name} latest")
    training_report = _read_json(
        training_report_path, label=f"{run_dir.name} training report"
    )
    rows = _read_metrics(metrics_path)
    steps = [int(row.get("step", -1)) for row in rows]
    samples_seen = [int(row.get("samples_seen", -1)) for row in rows]
    if (
        steps[-1] != EXPECTED_STEPS
        or any(current <= previous for previous, current in zip(steps, steps[1:]))
        or any(
            samples != step * SAMPLES_PER_STEP
            for step, samples in zip(steps, samples_seen)
        )
        or any(
            current <= previous
            for previous, current in zip(samples_seen, samples_seen[1:])
        )
    ):
        raise ValueError(f"Metrics continuity or sample accounting failed for {run_dir.name}")

    expected_git = {
        "revision": training_revision,
        "branch": training_branch,
        "dirty": False,
    }
    dataset_sha = latest.get("dataset_identity_sha256")
    runtime_sha = latest.get("runtime_environment_sha256")
    if (
        training_report.get("training_complete") is not True
        or int(training_report.get("completed_steps", -1)) != EXPECTED_STEPS
        or int(training_report.get("target_steps", -1)) != EXPECTED_STEPS
        or training_report.get("git") != expected_git
        or training_report.get("runtime_environment_sha256") != runtime_sha
        or training_report.get("dataset_provenance", {}).get("identity_sha256")
        != dataset_sha
        or training_report.get("final_metrics") != rows[-1]
        or training_report.get("latest_checkpoint") != latest
    ):
        raise ValueError(f"Training report lineage differs for {run_dir.name}")

    for item in checkpoints:
        if (
            item.get("git_revision") != training_revision
            or item.get("git_branch") != training_branch
            or item.get("git_dirty") is not False
            or item.get("dataset_identity_sha256") != dataset_sha
            or item.get("runtime_environment_sha256") != runtime_sha
            or int(item.get("checkpoint_bytes", -1)) < 1
            or len(str(item.get("checkpoint_sha256", ""))) != 64
        ):
            raise ValueError(f"Checkpoint lineage differs for {run_dir.name}")

    return {
        "status": "verified",
        "run_dir": run_dir.resolve().as_posix(),
        "metrics": {
            **_identity(metrics_path),
            "row_count": len(rows),
            "first_step": steps[0],
            "last_step": steps[-1],
            "last_samples_seen": samples_seen[-1],
            "strictly_increasing": True,
            "samples_seen_equals_step_times_64": True,
        },
        "latest": _identity(latest_path),
        "training_report": _identity(training_report_path),
        "dataset_identity_sha256": dataset_sha,
        "runtime_environment_sha256": runtime_sha,
        "required_checkpoint_integrity": required,
    }


def build_replay(
    *,
    project: Path,
    output_root: Path,
    code_identity: Mapping[str, Any],
    training_revision: str,
    training_tree: str,
    training_branch: str,
    expected_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    root = output_root.resolve()
    if not root.is_dir() or output_root.is_symlink():
        raise ValueError("Output root must be a real directory")
    source_inventory_before = _source_inventory(root)
    payload_inventory_before = _payload_inventory(root)
    source_reports, decision = _validate_source_reports(
        root,
        expected_sha256=expected_source_sha256,
        training_revision=training_revision,
        training_tree=training_tree,
        training_branch=training_branch,
    )
    runs = {
        run: _validate_run(
            root / run,
            training_revision=training_revision,
            training_branch=training_branch,
        )
        for run in RUNS
    }
    if source_inventory_before != _source_inventory(root):
        raise ValueError("Small source evidence changed during physical replay")
    if payload_inventory_before != _payload_inventory(root):
        raise ValueError("Checkpoint payload identity changed during physical replay")

    builder = Path(__file__).resolve()
    auditor = project.resolve() / "scripts" / "audit_generation_training_progress.py"
    return {
        "schema_version": 1,
        "role": "conditioning_ranking_four_arm_checkpoint_integrity_replay",
        "status": "pass",
        "scope": "completed_four_arm_probe1k_physical_integrity_only",
        "code": {
            **dict(code_identity),
            "builder": _identity(builder),
            "auditor": _identity(auditor),
        },
        "training_source": {
            "revision": training_revision,
            "tree": training_tree,
            "branch": training_branch,
            "tracked_dirty": False,
        },
        "source_reports": source_reports,
        "required_checkpoint_steps": list(REQUIRED_CHECKPOINT_STEPS),
        "physical_checkpoint_count": sum(
            len(run["required_checkpoint_integrity"]["checkpoints"])
            for run in runs.values()
        ),
        "runs": runs,
        "scientific_decision_preserved": decision,
        "generation_advantage_proven": False,
        "claim_boundary": {
            "diagnostic_non_authorizing": True,
            "training_allowed": False,
            "sampling_allowed": False,
            "followup_training_allowed": False,
            "checkpoint_promotion_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "export_allowed": False,
            "release_allowed": False,
            "process_signal_allowed": False,
        },
    }


def _write_immutable(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path, label="existing replay report")
        if existing != report:
            raise ValueError("Refusing to overwrite a different checkpoint replay report")
        return
    write_json_report(path, report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physically replay all checkpoints from the completed four-arm screen."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--expected-code-tree", required=True)
    parser.add_argument("--expected-code-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    for name in SOURCE_PATHS:
        parser.add_argument(f"--expected-{name.replace('_', '-')}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    code = _code_identity(project)
    expected_code = {
        "revision": args.expected_code_revision,
        "tree": args.expected_code_tree,
        "branch": args.expected_code_branch,
        "tracked_dirty": False,
    }
    if code != expected_code:
        raise ValueError("Replay checkout does not match the expected clean code identity")
    source_hashes = {
        name: getattr(args, f"expected_{name}_sha256") for name in SOURCE_PATHS
    }
    report = build_replay(
        project=project,
        output_root=args.output_root,
        code_identity=code,
        training_revision=args.expected_training_revision,
        training_tree=args.expected_training_tree,
        training_branch=args.expected_training_branch,
        expected_source_sha256=source_hashes,
    )
    _write_immutable(args.output, report)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
