from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, git_provenance, write_json_report

try:
    from scripts.build_generation_full_readiness import (
        _read_json,
        _current_runtime_environment_sha,
        source_identity,
        verify_readiness_report,
    )
    from scripts.build_generation_large_capacity_deployment_receipt import (
        verify_deployment_receipt,
    )
except ModuleNotFoundError:
    from build_generation_full_readiness import (
        _read_json,
        _current_runtime_environment_sha,
        source_identity,
        verify_readiness_report,
    )
    from build_generation_large_capacity_deployment_receipt import (
        verify_deployment_receipt,
    )


SCHEMA_VERSION = 1
ROLE = "stability_full_readiness_revision_bridge"
FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
SOURCE_REPORT_NAMES = {
    "deployment_receipt",
    "promotion_gate",
    "cofitok_config",
    "dense_config",
    "config_validation",
    "storage_capacity",
    "runtime_selection",
}
FULL_RUNBOOK = (
    "artifacts/runbooks/"
    "generation_stability_ema_teacher_full_matched_300k_after_gate.sh"
)
SOURCE_SAMPLE_RESERVE = b"--sample-count 16384 \\\n"
TARGET_SAMPLE_RESERVE = b"--sample-count 116640 \\\n"
TRAINING_EXECUTION_MARKER = b"monitor_report_passes() {\n"
TARGET_AUTHORIZATION_REQUIREMENTS = (
    b"EXPECTED_READINESS_BRIDGE_SHA256=",
    b"EXPECTED_STABILITY_SUPPLEMENTAL_SHA256=",
    b"validate_generation_full_readiness_bridge.py",
    b'--expected-bridge-sha256 "$EXPECTED_READINESS_BRIDGE_SHA256"',
    b'--readiness-bridge "$READINESS_BRIDGE"',
    b'--stability-supplemental "$STABILITY_SUPPLEMENTAL"',
    b'--expected-stability-supplemental-sha256 "$EXPECTED_STABILITY_SUPPLEMENTAL_SHA256"',
    TARGET_SAMPLE_RESERVE,
)
BRIDGE_ONLY_PREAMBLE_LINES = (
    b"EXPECTED_STABILITY_SUPPLEMENTAL_SHA256=${EXPECTED_STABILITY_SUPPLEMENTAL_SHA256:?set the passing frozen stability supplemental SHA256}\n",
    b"EXPECTED_READINESS_BRIDGE_SHA256=${EXPECTED_READINESS_BRIDGE_SHA256:?set the immutable readiness revision bridge SHA256}\n",
    b'STABILITY_SUPPLEMENTAL="$SCALING_ROOT/reports/frozen_posteval_supplemental/supplemental_qualification.json"\n',
    b'READINESS_BRIDGE="$REPORT_ROOT/full_training_readiness_bridge.json"\n',
    b'[[ -f "$STABILITY_SUPPLEMENTAL" ]]\n',
    b'[[ -f "$READINESS_BRIDGE" ]]\n',
    b"[[ \"$(sha256sum \"$STABILITY_SUPPLEMENTAL\" | awk '{print $1}')\" == \"$EXPECTED_STABILITY_SUPPLEMENTAL_SHA256\" ]]\n",
    b"[[ \"$(sha256sum \"$READINESS_BRIDGE\" | awk '{print $1}')\" == \"$EXPECTED_READINESS_BRIDGE_SHA256\" ]]\n",
    b'  --stability-supplemental "$STABILITY_SUPPLEMENTAL"\n',
    b'  --expected-stability-supplemental-sha256 "$EXPECTED_STABILITY_SUPPLEMENTAL_SHA256"\n',
    b'  --readiness-bridge "$READINESS_BRIDGE"\n',
    b'    --stability-supplemental "$STABILITY_SUPPLEMENTAL" \\\n',
    b'    --expected-stability-supplemental-sha256 "$EXPECTED_STABILITY_SUPPLEMENTAL_SHA256" \\\n',
    b'    --readiness-bridge "$READINESS_BRIDGE" \\\n',
)
SOURCE_RUNTIME_VALIDATOR_START = (
    b'runtime_selected="$("$PYTHON" scripts/validate_generation_full_readiness.py \\\n'
)
TARGET_RUNTIME_VALIDATION_BLOCK = (
    b'runtime_selected="$("$PYTHON" scripts/validate_generation_full_readiness_bridge.py \\\n'
    b'  --project-root "$PROJECT" \\\n'
    b'  --bridge "$READINESS_BRIDGE" \\\n'
    b'  --expected-bridge-sha256 "$EXPECTED_READINESS_BRIDGE_SHA256" \\\n'
    b'  --readiness "$READINESS" \\\n'
    b'  --expected-readiness-sha256 "$EXPECTED_READINESS_SHA256" \\\n'
    b'  --source-deployment-receipt "$("$PYTHON" -c \'import json,sys; print(json.load(open(sys.argv[1]))["source_deployment_receipt"]["path"])\' "$READINESS_BRIDGE")" \\\n'
    b'  --expected-source-deployment-receipt-sha256 "$("$PYTHON" -c \'import json,sys; print(json.load(open(sys.argv[1]))["source_deployment_receipt"]["sha256"])\' "$READINESS_BRIDGE")" \\\n'
    b'  --target-deployment-receipt "$DEPLOYMENT_RECEIPT" \\\n'
    b'  --expected-target-deployment-receipt-sha256 "$EXPECTED_DEPLOYMENT_RECEIPT_SHA256" \\\n'
    b'  --expected-source-revision "$("$PYTHON" -c \'import json,sys; print(json.load(open(sys.argv[1]))["source_git"]["revision"])\' "$READINESS_BRIDGE")" \\\n'
    b'  --expected-source-branch "$("$PYTHON" -c \'import json,sys; print(json.load(open(sys.argv[1]))["source_git"]["branch"])\' "$READINESS_BRIDGE")" \\\n'
    b'  --expected-target-revision "$EXPECTED_TARGET_REVISION" \\\n'
    b'  --expected-target-branch "$EXPECTED_TARGET_BRANCH" \\\n'
    b'  --print-selected-runtime)"\n'
)
TARGET_RUNTIME_VALIDATOR_START = TARGET_RUNTIME_VALIDATION_BLOCK.splitlines(
    keepends=True
)[0]
RUNTIME_VALIDATOR_END = b'  --print-selected-runtime)"\n'
TRAINING_CRITICAL_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "configs/generation/imagenet256_stability_rgbtail3_rollout_x0_u2_"
    "ema_teacher_k8_300k.json",
    "configs/generation/imagenet256_stability_rollout_x0_u2_ema_teacher_"
    "dense_300k.json",
    "src/cofitok",
    "scripts/train_generation.py",
    "scripts/select_generation_training_runtime.py",
    "scripts/run_generation_training_watchdog.py",
    "scripts/monitor_generation_pair.py",
    "scripts/audit_generation_training_progress.py",
    "scripts/validate_generation_training_completion.py",
    "scripts/validate_generation_training_pair.py",
    "scripts/build_generation_milestone_report.py",
    "scripts/validate_generation_milestone_report.py",
    "artifacts/runbooks/generation_full_milestone_eval.sh",
)


def _git(project: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(project), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _git_bytes(project: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(project), "show", f"{revision}:{path}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _critical_manifest(project: Path, revision: str) -> list[dict[str, str]]:
    result = _git(
        project,
        "ls-tree",
        "-r",
        "--full-tree",
        revision,
        "--",
        *TRAINING_CRITICAL_PATHS,
    )
    rows = []
    for line in result.stdout.splitlines():
        metadata, path = line.split("\t", 1)
        _mode, object_type, object_id = metadata.split(" ", 2)
        if object_type != "blob":
            raise ValueError(f"training-critical Git entry is not a blob: {path}")
        rows.append({"path": path, "git_blob": object_id})
    if not rows:
        raise ValueError("training-critical Git manifest is empty")
    return rows

def _runtime_validation_block(preamble: bytes, start: bytes) -> bytes:
    if preamble.count(start) != 1:
        raise ValueError("full runbook runtime validator is missing or duplicated")
    begin = preamble.index(start)
    end = preamble.find(RUNTIME_VALIDATOR_END, begin)
    if end < 0:
        raise ValueError("full runbook runtime validator end is missing")
    end += len(RUNTIME_VALIDATOR_END)
    return preamble[begin:end]


def _normalize_target_preamble(
    source_preamble: bytes,
    target_preamble: bytes,
) -> bytes:
    source_runtime = _runtime_validation_block(
        source_preamble,
        SOURCE_RUNTIME_VALIDATOR_START,
    )
    target_runtime = _runtime_validation_block(
        target_preamble,
        TARGET_RUNTIME_VALIDATOR_START,
    )
    if target_runtime != TARGET_RUNTIME_VALIDATION_BLOCK:
        raise ValueError("target full runbook bridge validator command differs")
    normalized = target_preamble.replace(target_runtime, source_runtime, 1)
    for line in BRIDGE_ONLY_PREAMBLE_LINES:
        if normalized.count(line) != 1:
            raise ValueError("target full runbook bridge authorization line differs")
        normalized = normalized.replace(line, b"", 1)
    if normalized.count(TARGET_SAMPLE_RESERVE) != 1:
        raise ValueError("target full runbook completion reserve is unexpected")
    normalized = normalized.replace(
        TARGET_SAMPLE_RESERVE,
        SOURCE_SAMPLE_RESERVE,
        1,
    )
    return normalized



def _verify_runbook_change(
    project: Path,
    *,
    source_revision: str,
    target_revision: str,
) -> dict[str, Any]:
    source = _git_bytes(project, source_revision, FULL_RUNBOOK)
    target = _git_bytes(project, target_revision, FULL_RUNBOOK)
    if source.count(SOURCE_SAMPLE_RESERVE) != 1:
        raise ValueError("source full runbook sample reserve is unexpected")
    if target.count(TARGET_SAMPLE_RESERVE) != 1:
        raise ValueError("target full runbook completion reserve is unexpected")
    for requirement in TARGET_AUTHORIZATION_REQUIREMENTS:
        if requirement not in target:
            raise ValueError("target full runbook authorization bridge is incomplete")
    if source.count(TRAINING_EXECUTION_MARKER) != 1 or target.count(
        TRAINING_EXECUTION_MARKER
    ) != 1:
        raise ValueError("full training execution marker is missing or duplicated")
    source_execution = source[source.index(TRAINING_EXECUTION_MARKER) :]
    source_preamble = source[: source.index(TRAINING_EXECUTION_MARKER)]
    target_preamble = target[: target.index(TRAINING_EXECUTION_MARKER)]
    normalized_target_preamble = _normalize_target_preamble(
        source_preamble,
        target_preamble,
    )
    if normalized_target_preamble != source_preamble:
        raise ValueError(
            "full training authorization preamble changed beyond the controlled bridge upgrade"
        )
    preamble_sha256 = hashlib.sha256(source_preamble).hexdigest()
    normalized_preamble_sha256 = hashlib.sha256(normalized_target_preamble).hexdigest()
    target_execution = target[target.index(TRAINING_EXECUTION_MARKER) :]
    if source_execution != target_execution:
        raise ValueError("full training execution changed across readiness bridge")
    return {
        "path": FULL_RUNBOOK,
        "source_git_blob": _git(
            project, "rev-parse", f"{source_revision}:{FULL_RUNBOOK}"
        ).stdout.strip(),
        "target_git_blob": _git(
            project, "rev-parse", f"{target_revision}:{FULL_RUNBOOK}"
        ).stdout.strip(),
        "authorization_upgrade": {
            "readiness_bridge_required": True,
            "frozen_stability_supplemental_required": True,
            "launch_receipt_schema_version": 3,
            "sample_count": {
                "source": 16_384,
                "target": 116_640,
            },
        },
        "training_execution_marker": TRAINING_EXECUTION_MARKER.decode(
            "ascii"
        ).strip(),
        "training_execution_sha256": hashlib.sha256(
            source_execution
        ).hexdigest(),
        "training_execution_identical": True,
        "controlled_preamble_upgrade": True,
        "source_preamble_sha256": preamble_sha256,
        "normalized_target_preamble_sha256": normalized_preamble_sha256,
    }


def _source_paths_from_readiness(report: dict[str, Any]) -> dict[str, Path]:
    sources = report.get("source_reports")
    if not isinstance(sources, dict) or set(sources) != SOURCE_REPORT_NAMES:
        raise ValueError("readiness source report set differs")
    resolved = {}
    for name, identity in sources.items():
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise ValueError(f"readiness source identity is malformed: {name}")
        path = Path(identity["path"])
        if source_identity(path) != identity:
            raise ValueError(f"readiness source changed: {name}")
        resolved[name] = path
    return resolved


def build_readiness_bridge(
    *,
    project_root: Path,
    readiness_path: Path,
    expected_readiness_sha256: str,
    source_deployment_receipt: Path,
    expected_source_deployment_receipt_sha256: str,
    target_deployment_receipt: Path,
    expected_target_deployment_receipt_sha256: str,
    expected_source_revision: str,
    expected_source_branch: str,
    expected_target_revision: str,
    expected_target_branch: str,
    require_current_target_git: bool = True,
) -> dict[str, Any]:
    for revision in (expected_source_revision, expected_target_revision):
        if not FULL_REVISION.fullmatch(revision):
            raise ValueError("readiness bridge revisions must be full Git SHA-1 values")
    if expected_source_revision == expected_target_revision:
        raise ValueError("readiness bridge requires distinct source and target revisions")
    project_root = project_root.resolve()
    target_git = {
        "revision": expected_target_revision,
        "branch": expected_target_branch,
        "tracked_dirty": False,
    }
    if require_current_target_git and git_provenance(project_root) != target_git:
        raise ValueError("readiness bridge target Git identity differs")
    if file_sha256(readiness_path) != expected_readiness_sha256:
        raise ValueError("stability full readiness SHA256 differs")

    source_deployment = verify_deployment_receipt(
        _read_json(source_deployment_receipt),
        receipt_path=source_deployment_receipt,
        expected_receipt_sha256=expected_source_deployment_receipt_sha256,
        require_current_formal_repository=False,
    )
    target_deployment = verify_deployment_receipt(
        _read_json(target_deployment_receipt),
        receipt_path=target_deployment_receipt,
        expected_receipt_sha256=expected_target_deployment_receipt_sha256,
        require_current_formal_repository=False,
    )
    source_git = {
        "revision": expected_source_revision,
        "branch": expected_source_branch,
        "tracked_dirty": False,
    }
    if source_deployment.get("checkout", {}).get("git") != source_git:
        raise ValueError("readiness bridge source deployment identity differs")
    if target_deployment.get("checkout", {}).get("git") != target_git:
        raise ValueError("readiness bridge target deployment identity differs")
    if Path(target_deployment["checkout"]["path"]).resolve() != project_root:
        raise ValueError("readiness bridge is not running in the target checkout")
    ancestor = _git(
        project_root,
        "merge-base",
        "--is-ancestor",
        expected_source_revision,
        expected_target_revision,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("readiness source revision is not a target ancestor")

    readiness = _read_json(readiness_path)
    source_paths = _source_paths_from_readiness(readiness)
    if source_identity(source_deployment_receipt) != readiness["source_reports"][
        "deployment_receipt"
    ]:
        raise ValueError("readiness source deployment binding differs")
    storage_report = _read_json(source_paths["storage_capacity"])
    verified_readiness = verify_readiness_report(
        readiness,
        source_paths=source_paths,
        training_run_dirs=[Path(path) for path in readiness["training_run_dirs"]],
        benchmark_root=Path(readiness["benchmark_root"]),
        storage_path=Path(storage_report["filesystem"]["path"]),
        project_root=Path(source_deployment["checkout"]["path"]),
        expected_revision=expected_source_revision,
        expected_branch=expected_source_branch,
        require_current_runtime_environment=False,
        require_current_formal_repository=False,
        require_current_git=False,
        require_training_state_absent=False,
    )
    current_runtime_environment_sha256 = _current_runtime_environment_sha(
        source_paths["cofitok_config"],
        project_root=project_root,
    )
    if (
        verified_readiness["runtime_selection"][
            "runtime_environment_sha256"
        ]
        != current_runtime_environment_sha256
    ):
        raise ValueError(
            "target runtime environment differs from source readiness"
        )

    source_manifest = _critical_manifest(project_root, expected_source_revision)
    target_manifest = _critical_manifest(project_root, expected_target_revision)
    if source_manifest != target_manifest:
        raise ValueError("training-critical Git blobs differ across readiness bridge")
    runbook = _verify_runbook_change(
        project_root,
        source_revision=expected_source_revision,
        target_revision=expected_target_revision,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "role": ROLE,
        "source_git": source_git,
        "target_git": target_git,
        "source_readiness": source_identity(readiness_path),
        "source_deployment_receipt": source_identity(source_deployment_receipt),
        "target_deployment_receipt": source_identity(target_deployment_receipt),
        "training_critical_paths": list(TRAINING_CRITICAL_PATHS),
        "training_critical_manifest": source_manifest,
        "training_semantics_identical": True,
        "full_training_runbook": runbook,
        "runtime_selection": verified_readiness["runtime_selection"],
        "target_runtime_environment_sha256": (
            current_runtime_environment_sha256
        ),
        "config_contract": verified_readiness["config_contract"],
        "storage_capacity": verified_readiness["storage_capacity"],
        "promotion_authorization": verified_readiness[
            "promotion_authorization"
        ],
        "training_run_dirs": verified_readiness["training_run_dirs"],
        "benchmark_root": verified_readiness["benchmark_root"],
        "full_training_launch_allowed": True,
        "formal_generation_completion_claimed": False,
    }


def verify_readiness_bridge(
    report: dict[str, Any],
    *,
    bridge_path: Path,
    expected_bridge_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    if file_sha256(bridge_path) != expected_bridge_sha256:
        raise ValueError("stability full readiness bridge SHA256 differs")
    expected = build_readiness_bridge(**kwargs)
    if report != expected:
        raise ValueError("stability full readiness bridge is not reproducible")
    return expected


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--expected-readiness-sha256", required=True)
    parser.add_argument("--source-deployment-receipt", type=Path, required=True)
    parser.add_argument("--expected-source-deployment-receipt-sha256", required=True)
    parser.add_argument("--target-deployment-receipt", type=Path, required=True)
    parser.add_argument("--expected-target-deployment-receipt-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-source-branch", required=True)
    parser.add_argument("--expected-target-revision", required=True)
    parser.add_argument("--expected-target-branch", required=True)


def kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "project_root": args.project_root,
        "readiness_path": args.readiness,
        "expected_readiness_sha256": args.expected_readiness_sha256,
        "source_deployment_receipt": args.source_deployment_receipt,
        "expected_source_deployment_receipt_sha256": (
            args.expected_source_deployment_receipt_sha256
        ),
        "target_deployment_receipt": args.target_deployment_receipt,
        "expected_target_deployment_receipt_sha256": (
            args.expected_target_deployment_receipt_sha256
        ),
        "expected_source_revision": args.expected_source_revision,
        "expected_source_branch": args.expected_source_branch,
        "expected_target_revision": args.expected_target_revision,
        "expected_target_branch": args.expected_target_branch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a source-bound readiness-to-training revision bridge."
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"readiness bridge already exists: {args.output}")
    report = build_readiness_bridge(**kwargs_from_args(args))
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
