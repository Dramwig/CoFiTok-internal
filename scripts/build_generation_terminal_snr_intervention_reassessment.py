"""Build or validate a fail-closed terminal-SNR intervention reassessment.

This script is deliberately standard-library-only.  It consumes an exact set of
content-addressed, immutable scientific sources; re-reads every source with a
TOCTOU guard; and emits a deterministic decision or an adjacent deterministic
validation receipt.  It never launches training, sampling, evaluation, or GPU
work.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "cofitok_generation_terminal_snr_intervention_reassessment_v1"
ROLE = "source_bound_terminal_snr_intervention_reassessment"
VALIDATION_SCHEMA = (
    "cofitok_generation_terminal_snr_intervention_reassessment_validation_v1"
)
VALIDATION_ROLE = (
    "content_addressed_terminal_snr_intervention_reassessment_validation"
)
DECISION = "no_defensible_shared_intervention_selected"

EXECUTION_REVISION = "89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430"
EXECUTION_TREE = "46efd20cff489bccd799bb13c4155a0cc79e7649"
EXECUTION_BRANCH = "terminal-snr-execution-89bcd9a"

ARM_KEYS = {
    "arm_control_cofitok": "control_cofitok",
    "arm_control_dense_identity": "control_dense_identity",
    "arm_endpoint0975_cofitok": "endpoint0975_cofitok",
    "arm_endpoint0975_dense_identity": "endpoint0975_dense_identity",
}

REQUIRED_SOURCES = {
    "terminal_result",
    "terminal_result_validation",
    "controller_status",
    "controller_log",
    "preparation",
    "execution_authorization",
    "launch_receipt",
    "runtime_selection",
    "live_snapshot",
    "storage_capacity",
    *ARM_KEYS,
    "capacity_result",
    "capacity_result_validation",
    "capacity_hold_decision",
    "capacity_hold_validation",
    "prior_terminal_reassessment",
    "prior_terminal_reassessment_validation",
    "exposure_result",
    "exposure_result_validation",
    "exposure_decision",
    "exposure_decision_validation",
    "exposure_controller_status",
    "exposure_controller_log",
    "sampling_recovery_result",
    "min_snr_result",
    "min_snr_physical_guard",
    "semantic_residual_postevaluation",
    "conditioning_ranking_postevaluation",
    "exposure_semantic_trajectory",
    "exposure_semantic_trajectory_replay_audit",
}

JSON_SOURCES = REQUIRED_SOURCES - {
    "controller_log",
    "exposure_controller_log",
}

AUTHORIZATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "gpu_execution_allowed": False,
    "process_signals_allowed": False,
    "frozen_confirmation_preparation_allowed": False,
    "frozen_confirmation_launch_allowed": False,
    "large_capacity_readiness_preparation_allowed": False,
    "full_training_preparation_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "paper_integration_allowed": False,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return list(value)


def _hex(value: Any, *, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not a lowercase {length}-character hex digest")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _reject_symlink_chain(path: Path, *, allow_missing_leaf: bool = False) -> None:
    absolute = path if path.is_absolute() else path.absolute()
    parts = absolute.parts
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        leaf = index == len(parts) - 1
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing_leaf and leaf:
                return
            raise
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink path component is forbidden: {current}")


def _stable_read(path: Path) -> tuple[bytes, dict[str, Any]]:
    if not path.is_absolute():
        raise ValueError(f"source path must be absolute: {path}")
    _reject_symlink_chain(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"source is not a regular file: {path}")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    bytes_read = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            chunks.append(block)
            digest.update(block)
            bytes_read += len(block)
    after = path.stat()
    if (
        bytes_read != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ino != before.st_ino
        or after.st_dev != before.st_dev
    ):
        raise RuntimeError(f"source changed while being read: {path}")
    return b"".join(chunks), {
        "path": str(path),
        "bytes": bytes_read,
        "sha256": digest.hexdigest(),
    }


def _read_json(data: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    return _object(payload, name)


def _identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    path = row["path"]
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"{name} path must be absolute")
    if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int):
        raise ValueError(f"{name} bytes must be an integer")
    if row["bytes"] < 1:
        raise ValueError(f"{name} bytes must be positive")
    _hex(row["sha256"], length=64, name=f"{name} SHA256")
    return row


def _checkout_identity(project_root: Path) -> dict[str, Any]:
    _reject_symlink_chain(project_root)
    if not project_root.is_dir():
        raise ValueError(f"project root is not a directory: {project_root}")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError(f"project root must be completely clean: {project_root}")
    branch = git("branch", "--show-current")
    if not branch:
        raise ValueError(f"project root must be on a named branch: {project_root}")
    return {
        "branch": branch,
        "revision": _hex(git("rev-parse", "HEAD^{commit}"), length=40, name="revision"),
        "tracked_dirty": False,
        "tree": _hex(git("rev-parse", "HEAD^{tree}"), length=40, name="tree"),
    }


def _git_identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"branch", "revision", "tracked_dirty", "tree"}:
        raise ValueError(f"{name} fields differ")
    if not isinstance(row["branch"], str) or not row["branch"]:
        raise ValueError(f"{name} branch is invalid")
    _hex(row["revision"], length=40, name=f"{name} revision")
    _hex(row["tree"], length=40, name=f"{name} tree")
    if row["tracked_dirty"] is not False:
        raise ValueError(f"{name} must be tracked-clean")
    return row


def _load_sources(
    rows: Sequence[Sequence[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, bytes]]:
    paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    for row in rows:
        if len(row) != 3:
            raise ValueError("each --source requires KEY PATH EXPECTED_SHA256")
        key, raw_path, digest = row
        if key in paths:
            raise ValueError(f"duplicate source key: {key}")
        paths[key] = Path(raw_path)
        expected[key] = _hex(digest, length=64, name=f"{key} expected SHA256")
    if set(paths) != REQUIRED_SOURCES:
        missing = sorted(REQUIRED_SOURCES - set(paths))
        extra = sorted(set(paths) - REQUIRED_SOURCES)
        raise ValueError(f"source key set differs; missing={missing}, extra={extra}")

    payloads: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    raw: dict[str, bytes] = {}
    for key in sorted(paths):
        data, descriptor = _stable_read(paths[key])
        if descriptor["sha256"] != expected[key]:
            raise ValueError(
                f"{key} SHA256 differs: expected {expected[key]}, "
                f"observed {descriptor['sha256']}"
            )
        raw[key] = data
        identities[key] = descriptor
        if key in JSON_SOURCES:
            payloads[key] = _read_json(data, key)
    return payloads, identities, raw


def _require_false_fields(row: Mapping[str, Any], fields: Sequence[str], name: str) -> None:
    for field in fields:
        if row.get(field) is not False:
            raise ValueError(f"{name}.{field} must be false")


def _validate_active_screen(
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    raw: Mapping[str, bytes],
) -> dict[str, Any]:
    result = _object(payloads["terminal_result"], "terminal result")
    if (
        result.get("schema_version")
        != "cofitok_generation_terminal_snr_screen_result_v1"
        or result.get("role") != "source_bound_terminal_snr_screen_scientific_result"
        or result.get("status") != "completed"
        or result.get("operational_status") != "pass"
        or result.get("scientific_status") != "hold"
        or result.get("terminal_status") != "hold"
        or result.get("screen_pass") is not False
        or result.get("generation_advantage_proven") is not False
        or result.get("decision") != "hold_terminal_snr_intervention"
    ):
        raise ValueError("active terminal-SNR result state differs")
    failed = sorted(_sequence(result.get("failed_checks"), "terminal failed checks"))
    expected_failed = [
        "cofitok.relative_fid_improvement",
        "dense_identity.relative_fid_improvement",
    ]
    if failed != expected_failed:
        raise ValueError("active terminal-SNR failed-check set differs")

    checks = _sequence(result.get("checks"), "terminal checks")
    observed_failed = sorted(
        str(_object(check, "terminal check").get("name"))
        for check in checks
        if _object(check, "terminal check").get("pass") is False
    )
    if observed_failed != expected_failed:
        raise ValueError("terminal-SNR check dispositions differ")
    if any(_object(check, "terminal check").get("pass") not in (True, False) for check in checks):
        raise ValueError("terminal-SNR checks contain an invalid pass value")

    comparisons = _object(result.get("comparisons"), "terminal comparisons")
    cofitok = _object(comparisons.get("cofitok"), "CoFiTok terminal comparison")
    dense = _object(comparisons.get("dense_identity"), "dense terminal comparison")
    cofitok_fid = _finite(
        cofitok.get("relative_fid_improvement"), "CoFiTok relative FID improvement"
    )
    dense_fid = _finite(
        dense.get("relative_fid_improvement"), "dense relative FID improvement"
    )
    cofitok_clip = _finite(
        cofitok.get("terminal_raw_x0_clip_fraction_reduction"),
        "CoFiTok clipping reduction",
    )
    dense_clip = _finite(
        dense.get("terminal_raw_x0_clip_fraction_reduction"),
        "dense clipping reduction",
    )
    if cofitok_fid >= 0.0 or dense_fid >= 0.0 or cofitok_clip <= 0.0 or dense_clip <= 0.0:
        raise ValueError("terminal endpoint effect no longer matches hold interpretation")

    thresholds = _object(result.get("thresholds"), "terminal thresholds")
    if thresholds.get("both_methods_min_relative_fid_improvement") != 0.05:
        raise ValueError("terminal relative-FID threshold differs")

    arm_summaries = _object(result.get("arm_summaries"), "terminal arm summaries")
    source_evidence = _object(result.get("source_evidence"), "terminal source evidence")
    result_arms = _object(source_evidence.get("arm_validations"), "result arm validations")
    if set(arm_summaries) != set(ARM_KEYS.values()) or set(result_arms) != set(ARM_KEYS.values()):
        raise ValueError("terminal result arm set differs")
    for source_key, arm in ARM_KEYS.items():
        validation = _object(payloads[source_key], f"{arm} validation")
        if (
            validation.get("schema_version")
            != "cofitok_generation_terminal_snr_screen_arm_validation_v1"
            or validation.get("role") != "physical_terminal_snr_screen_arm_validation"
            or validation.get("status") != "pass"
            or validation.get("arm") != arm
        ):
            raise ValueError(f"{arm} validation state differs")
        execution_git = _object(validation.get("execution_git"), f"{arm} execution Git")
        if execution_git != {
            "branch": EXECUTION_BRANCH,
            "revision": EXECUTION_REVISION,
            "tracked_dirty": False,
            "tree": EXECUTION_TREE,
        }:
            raise ValueError(f"{arm} execution Git differs")
        if _identity(result_arms.get(arm), f"{arm} result identity") != identities[source_key]:
            raise ValueError(f"terminal result does not bind the current {arm} validation")

    validation = _object(payloads["terminal_result_validation"], "terminal result validation")
    if (
        validation.get("schema_version")
        != "cofitok_generation_terminal_snr_screen_result_validation_v1"
        or validation.get("role")
        != "content_addressed_terminal_snr_screen_result_validation"
        or validation.get("status") != "pass"
        or validation.get("scientific_status") != "hold"
        or validation.get("screen_pass") is not False
        or validation.get("generation_advantage_proven") is not False
        or _identity(validation.get("result"), "validated terminal result")
        != identities["terminal_result"]
    ):
        raise ValueError("terminal result validation state differs")

    controller = _object(payloads["controller_status"], "terminal controller status")
    if (
        controller.get("schema_version") != 1
        or controller.get("role") != "generation_terminal_snr_screen_controller"
        or controller.get("status") != "completed"
        or controller.get("stage") != "complete"
        or controller.get("terminal_status") != "hold"
        or controller.get("generation_advantage_proven") is not False
    ):
        raise ValueError("terminal controller state differs")
    _require_false_fields(
        controller,
        [
            "frozen_confirmation_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_allowed",
            "export_allowed",
            "release_allowed",
            "process_signals_allowed",
        ],
        "terminal controller",
    )
    controller_log = raw["controller_log"].decode("utf-8", errors="strict")
    if controller_log.count("generated 1000/1000") != 4:
        raise ValueError("terminal controller log does not contain four completed samplings")
    if controller_log.count("scripts/train_generation.py") != 4:
        raise ValueError("terminal controller log does not contain four training arms")

    preparation = _object(payloads["preparation"], "terminal preparation")
    if (
        preparation.get("schema_version")
        != "cofitok_generation_terminal_snr_screen_preparation_v1"
        or preparation.get("status") != "prepared"
    ):
        raise ValueError("terminal preparation state differs")
    authorization = _object(payloads["execution_authorization"], "terminal authorization")
    if (
        authorization.get("schema_version")
        != "cofitok_generation_terminal_snr_screen_execution_authorization_v1"
        or authorization.get("status") != "authorized"
    ):
        raise ValueError("terminal execution authorization state differs")
    launch = _object(payloads["launch_receipt"], "terminal launch receipt")
    if (
        launch.get("schema_version")
        != "cofitok_generation_terminal_snr_screen_launch_receipt_v1"
        or launch.get("status") != "pass"
    ):
        raise ValueError("terminal launch receipt state differs")
    launch_sources = _object(launch.get("source_evidence"), "launch sources")
    for key in (
        "preparation",
        "execution_authorization",
        "runtime_selection",
        "live_snapshot",
        "storage_capacity",
    ):
        if _identity(launch_sources.get(key), f"launch {key}") != identities[key]:
            raise ValueError(f"launch receipt no longer binds {key}")

    endpoint = _object(arm_summaries["endpoint0975_cofitok"], "endpoint CoFiTok arm")
    diagnostics = _object(endpoint.get("cofitok_diagnostics"), "CoFiTok diagnostics")
    mechanism = {
        "ordered_rank_by_path_auc": diagnostics.get("ordered_rank_by_path_auc"),
        "coarse_token_energy_ratio": _finite(
            diagnostics.get("coarse_token_energy_ratio"), "coarse token energy ratio"
        ),
        "tail_two_energy_ratio": _finite(
            diagnostics.get("tail_two_energy_ratio"), "tail-two energy ratio"
        ),
        "zero_token_max_abs": _finite(
            diagnostics.get("zero_token_max_abs"), "zero-token maximum"
        ),
        "shuffled_to_ordered_endpoint_ratio": _finite(
            diagnostics.get("shuffled_to_ordered_endpoint_ratio"),
            "shuffled-to-ordered endpoint ratio",
        ),
    }
    if (
        mechanism["ordered_rank_by_path_auc"] != 1
        or mechanism["coarse_token_energy_ratio"] < 0.1
        or mechanism["tail_two_energy_ratio"] > 0.65
        or mechanism["zero_token_max_abs"] > 1e-8
        or mechanism["shuffled_to_ordered_endpoint_ratio"] < 2.0
    ):
        raise ValueError("CoFiTok mechanism checks no longer pass")

    return {
        "failed_checks": expected_failed,
        "cofitok_relative_fid_improvement": cofitok_fid,
        "dense_identity_relative_fid_improvement": dense_fid,
        "cofitok_terminal_raw_x0_clip_fraction_reduction": cofitok_clip,
        "dense_identity_terminal_raw_x0_clip_fraction_reduction": dense_clip,
        "mechanism_checks": mechanism,
        "screen_result": copy.deepcopy(identities["terminal_result"]),
        "screen_result_validation": copy.deepcopy(
            identities["terminal_result_validation"]
        ),
        "controller_status": copy.deepcopy(identities["controller_status"]),
        "controller_log": copy.deepcopy(identities["controller_log"]),
        "arm_validations": {
            arm: copy.deepcopy(identities[key]) for key, arm in sorted(ARM_KEYS.items())
        },
        "prelaunch_evidence": {
            key: copy.deepcopy(identities[key])
            for key in (
                "preparation",
                "execution_authorization",
                "launch_receipt",
                "runtime_selection",
                "live_snapshot",
                "storage_capacity",
            )
        },
    }


def _validate_prior_evidence(
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    raw: Mapping[str, bytes],
) -> dict[str, Any]:
    capacity = _object(payloads["capacity_result"], "capacity result")
    if (
        capacity.get("status") != "completed"
        or capacity.get("scientific_status") != "hold"
        or capacity.get("terminal_status") != "hold"
        or capacity.get("generation_advantage_proven") is not False
    ):
        raise ValueError("capacity result state differs")
    capacity_validation = _object(
        payloads["capacity_result_validation"], "capacity result validation"
    )
    if (
        capacity_validation.get("status") != "pass"
        or capacity_validation.get("scientific_status") != "hold"
        or capacity_validation.get("generation_advantage_proven") is not False
    ):
        raise ValueError("capacity result validation state differs")
    hold = _object(payloads["capacity_hold_decision"], "capacity hold decision")
    if (
        hold.get("decision") != "hold_for_source_bound_training_objective_reassessment"
        or hold.get("scientific_status") != "hold"
        or hold.get("terminal_status") != "hold"
        or hold.get("generation_advantage_proven") is not False
    ):
        raise ValueError("capacity hold decision state differs")
    hold_validation = _object(payloads["capacity_hold_validation"], "capacity hold validation")
    if (
        hold_validation.get("status") != "pass"
        or _identity(hold_validation.get("decision"), "validated capacity hold")
        != identities["capacity_hold_decision"]
    ):
        raise ValueError("capacity hold validation state differs")

    prior = _object(payloads["prior_terminal_reassessment"], "prior terminal reassessment")
    if (
        prior.get("decision")
        != "select_one_matched_terminal_snr_intervention_for_preparation"
        or prior.get("scientific_status") != "bounded_screen_selected_not_executed"
        or _object(prior.get("selected_intervention"), "prior intervention").get("id")
        != "matched_cosine_endpoint_fraction_0p975"
        or prior.get("generation_advantage_proven") is not False
    ):
        raise ValueError("prior terminal reassessment state differs")
    prior_validation = _object(
        payloads["prior_terminal_reassessment_validation"],
        "prior terminal reassessment validation",
    )
    if (
        prior_validation.get("status") != "pass"
        or _identity(prior_validation.get("decision"), "validated prior reassessment")
        != identities["prior_terminal_reassessment"]
    ):
        raise ValueError("prior terminal reassessment validation state differs")

    exposure = _object(payloads["exposure_result"], "exposure result")
    if (
        exposure.get("decision")
        != "bounded_exposure_continuation_completed_without_claim_upgrade"
        or exposure.get("scientific_status") != "hold"
        or exposure.get("terminal_status") != "hold"
        or exposure.get("generation_advantage_proven") is not False
    ):
        raise ValueError("exposure result state differs")
    exposure_validation = _object(
        payloads["exposure_result_validation"], "exposure result validation"
    )
    if exposure_validation.get("status") != "pass":
        raise ValueError("exposure result validation state differs")
    exposure_decision = _object(payloads["exposure_decision"], "exposure decision")
    if (
        exposure_decision.get("decision") != "capacity_screen"
        or exposure_decision.get("terminal_status") != "hold"
        or exposure_decision.get("generation_advantage_proven") is not False
    ):
        raise ValueError("exposure decision state differs")
    exposure_decision_validation = _object(
        payloads["exposure_decision_validation"], "exposure decision validation"
    )
    if (
        exposure_decision_validation.get("status") != "pass"
        or _identity(
            exposure_decision_validation.get("decision"),
            "validated exposure decision",
        )
        != identities["exposure_decision"]
    ):
        raise ValueError("exposure decision validation state differs")

    historical_controller = _object(
        payloads["exposure_controller_status"], "historical exposure controller"
    )
    if (
        historical_controller.get("role")
        != "generation_exposure_capacity_continuation_controller"
        or historical_controller.get("status") != "failed"
        or historical_controller.get("stage") != "failed"
        or historical_controller.get("terminal_status") != "hold"
        or historical_controller.get("generation_advantage_proven") is not False
    ):
        raise ValueError("historical exposure controller state differs")
    historical_log = raw["exposure_controller_log"].decode("utf-8", errors="strict")
    if "ValueError: cofitok runtime environment differs from authorization" not in historical_log:
        raise ValueError("historical exposure controller failure evidence differs")

    sampling = _object(payloads["sampling_recovery_result"], "sampling recovery result")
    if (
        sampling.get("status") != "pass"
        or sampling.get("selection_status") != "no_shared_sampling_recovery_candidate"
        or sampling.get("generation_advantage_proven") is not False
    ):
        raise ValueError("sampling recovery result state differs")

    min_snr = _object(payloads["min_snr_result"], "Min-SNR result")
    if (
        min_snr.get("status") != "completed"
        or min_snr.get("selection_status") != "no_shared_min_snr_candidate_at_50k"
        or min_snr.get("terminal_status") != "hold"
        or min_snr.get("generation_advantage_proven") is not False
    ):
        raise ValueError("Min-SNR result state differs")
    guard = _object(payloads["min_snr_physical_guard"], "Min-SNR physical guard")
    if (
        guard.get("status") != "pass"
        or guard.get("selection_status") != "no_shared_min_snr_candidate_at_50k"
        or guard.get("terminal_status") != "hold"
        or guard.get("generation_advantage_proven") is not False
    ):
        raise ValueError("Min-SNR physical guard state differs")

    semantic = _object(
        payloads["semantic_residual_postevaluation"], "semantic residual post-evaluation"
    )
    semantic_decision = _object(semantic.get("decision"), "semantic residual decision")
    if (
        semantic.get("status") != "completed"
        or semantic_decision.get("shared_semantic_alignment_recovery_supported") is not False
        or semantic_decision.get("method_passes")
        != {"cofitok": False, "dense_identity": False}
        or semantic_decision.get("recommended_next_action")
        != "reject_residual_alignment_candidate"
    ):
        raise ValueError("semantic residual post-evaluation state differs")

    conditioning = _object(
        payloads["conditioning_ranking_postevaluation"],
        "conditioning-ranking post-evaluation",
    )
    conditioning_decision = _object(
        conditioning.get("decision"), "conditioning-ranking decision"
    )
    if (
        conditioning.get("status") != "completed"
        or conditioning_decision.get("shared_semantic_alignment_recovery_supported")
        is not False
        or conditioning_decision.get("method_passes")
        != {"cofitok": False, "dense_identity": False}
        or conditioning_decision.get("recommended_next_action")
        != "revise_training_time_semantic_alignment_objective"
    ):
        raise ValueError("conditioning-ranking post-evaluation state differs")

    trajectory = _object(
        payloads["exposure_semantic_trajectory"], "exposure semantic trajectory"
    )
    trajectory_decision = _object(
        trajectory.get("decision"), "exposure semantic trajectory decision"
    )
    if (
        trajectory.get("status") != "completed"
        or trajectory_decision.get("shared_exposure_semantic_recovery_supported")
        is not False
        or trajectory_decision.get("recommended_next_action")
        != "reject_exposure_as_sufficient_semantic_recovery_explanation"
    ):
        raise ValueError("exposure semantic trajectory state differs")
    trajectory_audit = _object(
        payloads["exposure_semantic_trajectory_replay_audit"],
        "exposure semantic trajectory replay audit",
    )
    if (
        trajectory_audit.get("status") != "pass"
        or trajectory_audit.get("canonical_byte_exact_replay") is not True
        or trajectory_audit.get("generation_advantage_proven") is not False
        or trajectory_audit.get("scientific_decision_preserved")
        != trajectory_decision
    ):
        raise ValueError("exposure semantic trajectory replay state differs")

    groups = {
        "capacity_and_prior_selection": [
            "capacity_result",
            "capacity_result_validation",
            "capacity_hold_decision",
            "capacity_hold_validation",
            "prior_terminal_reassessment",
            "prior_terminal_reassessment_validation",
        ],
        "exposure_history": [
            "exposure_result",
            "exposure_result_validation",
            "exposure_decision",
            "exposure_decision_validation",
            "exposure_controller_status",
            "exposure_controller_log",
        ],
        "candidate_diagnostics": [
            "sampling_recovery_result",
            "min_snr_result",
            "min_snr_physical_guard",
            "semantic_residual_postevaluation",
            "conditioning_ranking_postevaluation",
            "exposure_semantic_trajectory",
            "exposure_semantic_trajectory_replay_audit",
        ],
    }
    return {
        group: {key: copy.deepcopy(identities[key]) for key in keys}
        for group, keys in groups.items()
    }


def _candidate_disposition(active: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "matched_cosine_endpoint_fraction_0p975",
            "status": "rejected_by_fresh_four_arm_screen",
            "reason": (
                "The endpoint reduced terminal raw-x0 clipping for both methods, "
                "but relative FID improvement was negative for both and therefore "
                "failed the predeclared >=0.05 scientific threshold."
            ),
            "observed": {
                "cofitok_relative_fid_improvement": active[
                    "cofitok_relative_fid_improvement"
                ],
                "dense_identity_relative_fid_improvement": active[
                    "dense_identity_relative_fid_improvement"
                ],
                "cofitok_terminal_raw_x0_clip_fraction_reduction": active[
                    "cofitok_terminal_raw_x0_clip_fraction_reduction"
                ],
                "dense_identity_terminal_raw_x0_clip_fraction_reduction": active[
                    "dense_identity_terminal_raw_x0_clip_fraction_reduction"
                ],
            },
        },
        {
            "id": "exposure_or_capacity",
            "status": "rejected_or_held_by_completed_screens",
            "reason": (
                "The completed exposure continuation and four-arm capacity screen "
                "both preserve a terminal scientific hold and prove no generation advantage."
            ),
        },
        {
            "id": "sampling_only_recovery",
            "status": "rejected_no_shared_candidate",
            "reason": (
                "The immutable matched sampling-recovery screen selected no shared "
                "sampling-only candidate."
            ),
        },
        {
            "id": "min_snr_or_scalar_timestep_weighting",
            "status": "rejected_or_deferred_no_shared_candidate",
            "reason": (
                "The matched Min-SNR gamma-5 pilot and its physical guard both report "
                "no shared candidate at 50K; another scalar weighting lacks a new discriminator."
            ),
        },
        {
            "id": "semantic_residual_alignment",
            "status": "rejected_by_postevaluation",
            "reason": (
                "Both methods fail the post-evaluation and the recorded recommendation "
                "is to reject the residual-alignment candidate."
            ),
        },
        {
            "id": "conditioning_ranking",
            "status": "insufficiently_supported_and_confounded",
            "reason": (
                "Both methods fail, shared semantic recovery is false, and the report's "
                "objective-revision recommendation is not evidence of a successful shared intervention."
            ),
        },
        {
            "id": "velocity_or_x0_prediction",
            "status": "rejected_as_incompatible_with_research_boundary",
            "reason": (
                "Velocity recovery would require an input-dependent x_t bypass and x0 "
                "prediction changes the target; neither preserves direct dense-epsilon-component semantics."
            ),
        },
        {
            "id": "architecture_or_token_layout_change",
            "status": "deferred_without_discriminating_evidence",
            "reason": (
                "The fresh screen's CoFiTok ordering, utilization, zero-token, and shuffle "
                "mechanism checks pass, so current evidence does not discriminate an architecture/layout repair."
            ),
        },
    ]


def build_reassessment(
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    raw: Mapping[str, bytes],
    builder_git: Mapping[str, Any],
    builder_script: Mapping[str, Any],
) -> dict[str, Any]:
    active = _validate_active_screen(payloads, identities, raw)
    prior = _validate_prior_evidence(payloads, identities, raw)
    report = {
        "schema_version": SCHEMA,
        "role": ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "hold",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "decision": DECISION,
        "builder_git": _git_identity(builder_git, "builder Git"),
        "builder_script": _identity(builder_script, "builder script"),
        "source_evidence": {
            "active_terminal_screen": active,
            "prior_and_candidate_evidence": prior,
        },
        "scientific_judgment": {
            "endpoint_intervention_reduced_terminal_clipping": True,
            "endpoint_intervention_improved_fid_for_cofitok": False,
            "endpoint_intervention_improved_fid_for_dense_identity": False,
            "endpoint_intervention_is_a_successful_shared_repair": False,
            "cofitok_mechanism_checks_pass": True,
            "one_source_compatible_shared_intervention_is_selected": False,
            "interpretation": (
                "The endpoint intervention repaired the measured terminal clipping symptom "
                "but worsened FID in both matched methods. Together with the completed negative "
                "or held exposure, capacity, sampling, Min-SNR, semantic-residual, conditioning-ranking, "
                "and trajectory evidence, this leaves no defensible shared intervention selected."
            ),
        },
        "candidate_disposition": _candidate_disposition(active),
        "historical_controller_policy": {
            "historical_exposure_controller_is_active_screen": False,
            "historical_exposure_controller_failure_may_be_rewritten": False,
            "active_screen": identities["terminal_result"]["path"],
            "interpretation": (
                "The failed exposure builder controller is immutable historical evidence; "
                "the completed terminal-SNR screen and its validated result are authoritative."
            ),
        },
        "next_stage": {
            "route": "hold",
            "selected_intervention": None,
            "frozen_confirmation_preparation_allowed": False,
            "frozen_confirmation_launch_allowed": False,
            "large_capacity_readiness_preparation_allowed": False,
            "full_training_preparation_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "required_next_evidence": (
                "A separate source-bound scientific reassessment must identify one "
                "predeclared, source-compatible shared intervention using new discriminating evidence."
            ),
        },
        "claim_policy": {
            "terminal_clipping_is_proven_root_cause": False,
            "shared_generation_recovery_claim_allowed": False,
            "cofitok_generation_advantage_claim_allowed": False,
            "formal_generation_claim_allowed": False,
            "generation_advantage_proven": False,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }
    validate_reassessment(report)
    return report


def validate_reassessment(report: Mapping[str, Any]) -> dict[str, Any]:
    row = _object(report, "terminal-SNR intervention reassessment")
    judgment = _object(row.get("scientific_judgment"), "scientific judgment")
    next_stage = _object(row.get("next_stage"), "next stage")
    claims = _object(row.get("claim_policy"), "claim policy")
    historical = _object(row.get("historical_controller_policy"), "historical policy")
    dispositions = _sequence(row.get("candidate_disposition"), "candidate dispositions")
    if (
        row.get("schema_version") != SCHEMA
        or row.get("role") != ROLE
        or row.get("status") != "completed"
        or row.get("operational_status") != "pass"
        or row.get("scientific_status") != "hold"
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("decision") != DECISION
        or row.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
        or judgment.get("endpoint_intervention_reduced_terminal_clipping") is not True
        or judgment.get("endpoint_intervention_improved_fid_for_cofitok") is not False
        or judgment.get("endpoint_intervention_improved_fid_for_dense_identity") is not False
        or judgment.get("endpoint_intervention_is_a_successful_shared_repair") is not False
        or judgment.get("cofitok_mechanism_checks_pass") is not True
        or judgment.get("one_source_compatible_shared_intervention_is_selected") is not False
        or next_stage.get("route") != "hold"
        or next_stage.get("selected_intervention") is not None
        or claims.get("terminal_clipping_is_proven_root_cause") is not False
        or claims.get("formal_generation_claim_allowed") is not False
        or claims.get("generation_advantage_proven") is not False
        or historical.get("historical_exposure_controller_is_active_screen") is not False
        or historical.get("historical_exposure_controller_failure_may_be_rewritten") is not False
    ):
        raise ValueError("terminal-SNR intervention reassessment contract differs")
    _git_identity(row.get("builder_git"), "builder Git")
    _identity(row.get("builder_script"), "builder script")
    if len(dispositions) != 8:
        raise ValueError("candidate disposition count differs")
    expected_ids = [
        "matched_cosine_endpoint_fraction_0p975",
        "exposure_or_capacity",
        "sampling_only_recovery",
        "min_snr_or_scalar_timestep_weighting",
        "semantic_residual_alignment",
        "conditioning_ranking",
        "velocity_or_x0_prediction",
        "architecture_or_token_layout_change",
    ]
    if [str(_object(item, "candidate disposition").get("id")) for item in dispositions] != expected_ids:
        raise ValueError("candidate disposition ordering differs")
    source_evidence = _object(row.get("source_evidence"), "source evidence")
    active = _object(source_evidence.get("active_terminal_screen"), "active screen evidence")
    if active.get("failed_checks") != [
        "cofitok.relative_fid_improvement",
        "dense_identity.relative_fid_improvement",
    ]:
        raise ValueError("active failed-check evidence differs")
    if (
        _finite(
            active.get("cofitok_relative_fid_improvement"),
            "CoFiTok relative FID improvement",
        )
        >= 0.0
        or _finite(
            active.get("dense_identity_relative_fid_improvement"),
            "dense relative FID improvement",
        )
        >= 0.0
    ):
        raise ValueError("reassessment does not preserve the negative FID evidence")
    return copy.deepcopy(row)


def build_validation(
    *,
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
    validator_script: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_reassessment(decision)
    decision_id = _identity(decision_identity, "decision")
    validator = _git_identity(validator_git, "validator Git")
    script = _identity(validator_script, "validator script")
    basis = {
        "decision": decision_id,
        "builder_git": validated["builder_git"],
        "builder_script": validated["builder_script"],
        "validator_git": validator,
        "validator_script": script,
        "source_evidence": validated["source_evidence"],
        "scientific_judgment": validated["scientific_judgment"],
        "candidate_disposition": validated["candidate_disposition"],
    }
    return {
        "schema_version": VALIDATION_SCHEMA,
        "role": VALIDATION_ROLE,
        "status": "pass",
        "scientific_status": "hold",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "decision": decision_id,
        "builder_git": copy.deepcopy(validated["builder_git"]),
        "builder_script": copy.deepcopy(validated["builder_script"]),
        "validator_git": validator,
        "validator_script": script,
        "source_evidence": copy.deepcopy(validated["source_evidence"]),
        "scientific_judgment": copy.deepcopy(validated["scientific_judgment"]),
        "candidate_disposition": copy.deepcopy(validated["candidate_disposition"]),
        "validation_basis_sha256": _canonical_sha256(basis),
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def validate_validation(
    receipt: Mapping[str, Any],
    *,
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(receipt, "terminal-SNR reassessment validation")
    expected = build_validation(
        decision=decision,
        decision_identity=decision_identity,
        validator_git=_git_identity(row.get("validator_git"), "validator Git"),
        validator_script=_identity(row.get("validator_script"), "validator script"),
    )
    if row != expected:
        raise ValueError("terminal-SNR reassessment validation is not reproducible")
    return expected


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("output path must be absolute")
    _reject_symlink_chain(path, allow_missing_leaf=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    _, identity = _stable_read(path)
    return identity


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        nargs=3,
        metavar=("KEY", "PATH", "EXPECTED_SHA256"),
        action="append",
        required=True,
    )
    parser.add_argument("--builder-project-root", type=Path, required=True)
    parser.add_argument("--builder-script", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a new immutable reassessment")
    build.add_argument("--decision", type=Path, required=True)
    _common_arguments(build)
    validate = subparsers.add_parser(
        "validate", help="physically replay a reassessment and write/verify its receipt"
    )
    validate.add_argument("--decision", type=Path, required=True)
    validate.add_argument("--expected-decision-sha256", required=True)
    validate.add_argument("--validation-receipt", type=Path, required=True)
    validate.add_argument("--validator-project-root", type=Path, required=True)
    _common_arguments(validate)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payloads, identities, raw = _load_sources(args.source)
    builder_git = _checkout_identity(args.builder_project_root)
    builder_script_bytes, builder_script_identity = _stable_read(args.builder_script)
    if builder_script_bytes != Path(__file__).read_bytes():
        raise ValueError("running script differs from the declared builder script")
    decision = build_reassessment(
        payloads=payloads,
        identities=identities,
        raw=raw,
        builder_git=builder_git,
        builder_script=builder_script_identity,
    )

    if args.command == "build":
        decision_path = args.decision
        if decision_path.parent.exists():
            raise FileExistsError(
                f"refusing to use an existing reassessment directory: {decision_path.parent}"
            )
        _reject_symlink_chain(decision_path.parent, allow_missing_leaf=True)
        decision_path.parent.mkdir(mode=0o755)
        decision_identity = _write_exclusive(decision_path, decision)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "scientific_status": "hold",
                    "decision": DECISION,
                    "decision_identity": decision_identity,
                },
                sort_keys=True,
            )
        )
        return

    expected_sha = _hex(
        args.expected_decision_sha256,
        length=64,
        name="expected decision SHA256",
    )
    decision_bytes, decision_identity = _stable_read(args.decision)
    if decision_identity["sha256"] != expected_sha:
        raise ValueError("terminal-SNR intervention reassessment SHA256 differs")
    actual = _read_json(decision_bytes, "terminal-SNR intervention reassessment")
    if actual != decision:
        raise ValueError("terminal-SNR intervention reassessment differs from physical replay")
    validate_reassessment(actual)

    validator_git = _checkout_identity(args.validator_project_root)
    validator_bytes, validator_script_identity = _stable_read(Path(__file__))
    if validator_bytes != builder_script_bytes:
        raise ValueError("validator script bytes differ from builder script bytes")
    receipt = build_validation(
        decision=actual,
        decision_identity=decision_identity,
        validator_git=validator_git,
        validator_script=validator_script_identity,
    )
    if args.validation_receipt.exists():
        receipt_bytes, receipt_identity = _stable_read(args.validation_receipt)
        existing = _read_json(receipt_bytes, "terminal-SNR reassessment validation")
        validate_validation(
            existing,
            decision=actual,
            decision_identity=decision_identity,
        )
        if existing != receipt:
            raise ValueError("existing validation receipt differs from physical replay")
    else:
        if args.validation_receipt.parent != args.decision.parent:
            raise ValueError("validation receipt must be adjacent to the decision")
        receipt_identity = _write_exclusive(args.validation_receipt, receipt)
        receipt_bytes, reread_identity = _stable_read(args.validation_receipt)
        if reread_identity != receipt_identity:
            raise RuntimeError("validation receipt identity changed after creation")
        validate_validation(
            _read_json(receipt_bytes, "terminal-SNR reassessment validation"),
            decision=actual,
            decision_identity=decision_identity,
        )
    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": "hold",
                "decision": DECISION,
                "decision_identity": decision_identity,
                "validation_receipt": receipt_identity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
