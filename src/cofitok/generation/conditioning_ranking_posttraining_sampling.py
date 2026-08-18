from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_probe import (
    validate_standing_experiment_authorization,
)
from cofitok.generation.conditioning_ranking_training_confirmation import (
    CLAIM_BOUNDARY as TRAINING_EXECUTION_CLAIM_BOUNDARY,
    EXECUTION_AUTHORIZATION_BOUNDARY as TRAINING_EXECUTION_BOUNDARY,
    EXECUTION_RECEIPT_ROLE as TRAINING_EXECUTION_RECEIPT_ROLE,
    EXPECTED_OUTPUT_ROOT as TRAINING_OUTPUT_ROOT,
    STAGE as TRAINING_STAGE,
)
from cofitok.generation_class_fidelity import (
    CLASS_FIDELITY_CLASSIFIER_BYTES,
    CLASS_FIDELITY_CLASSIFIER_NAME,
    CLASS_FIDELITY_CLASSIFIER_SHA256,
)


SCHEMA_VERSION = 1
STAGE = "conditioning_ranking_four_arm_posttraining_sampling5k_confirmation_v1"
SCOPE = (
    "imagenet256_four_arm_conditioning_ranking_posttraining_"
    "sampling5k_confirmation_only"
)
EXPECTED_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "conditioning_ranking_four_arm_posttraining_sampling5k_confirmation_v1"
)
PREPARATION_ROLE = (
    "generation_conditioning_ranking_four_arm_posttraining_sampling_"
    "confirmation_preparation"
)
EXECUTION_RECEIPT_ROLE = (
    "generation_conditioning_ranking_four_arm_posttraining_sampling_"
    "confirmation_execution_receipt"
)
IDLE_GPU_EVIDENCE_ROLE = (
    "generation_conditioning_ranking_four_arm_posttraining_sampling_"
    "confirmation_idle_gpu_evidence"
)
HELDOUT_ROLE = (
    "generation_conditioning_ranking_four_arm_train5k_heldout_evaluation"
)
HELDOUT_STAGE = "conditioning_ranking_four_arm_train5k_heldout_evaluation_v1"
HELDOUT_EVALUATOR_GIT = {
    "revision": "90f7206dd6d65b398a05222821cd6bc4cbf95efd",
    "branch": "scale/generation-label-ranking-5k-heldout-evaluation-v1",
    "tracked_dirty": False,
}
TRAINING_GIT = {
    "revision": "ed9463eb85ffb97545b5506e264123e46ff2fcfc",
    "branch": "scale/generation-label-ranking-5k-training-confirmation-v1",
    "tracked_dirty": False,
}
TRAINING_STATUS_ROLE = (
    "generation_conditioning_ranking_four_arm_train5k_confirmation_status"
)
RUN_NAMES = (
    "control_cofitok",
    "ranked_cofitok",
    "control_dense_identity",
    "ranked_dense_identity",
)
RUN_METHODS = {
    "control_cofitok": "cofitok",
    "ranked_cofitok": "cofitok",
    "control_dense_identity": "dense_identity",
    "ranked_dense_identity": "dense_identity",
}
METHOD_PREFIX_BUDGETS = {"cofitok": 8, "dense_identity": 1}
EXPECTED_CHECKPOINT_STEP = 5_000
EXPECTED_CHECKPOINT_FILENAME = "checkpoint_step_00005000.pt"
SAMPLE_RUN_NAME = "samples_5000_ddim50_cfg15_independent_stream_v1"
RUNBOOK_RELATIVE_PATH = (
    "artifacts/runbooks/"
    "generation_conditioning_ranking_four_arm_"
    "posttraining_sampling5k_confirmation_v1.sh"
)
EXPECTED_HELDOUT_DECISION = {
    "method_passes": {"cofitok": True, "dense_identity": True},
    "shared_semantic_alignment_recovery_supported": True,
    "cofitok_specific_advantage_claim_allowed": False,
    "recommended_next_action": (
        "prepare_separately_source_bound_posttraining_5k_sampling_confirmation"
    ),
}


def _stream_interval(*, seed: int, start_index: int, count: int) -> list[int]:
    return [seed + start_index, seed + start_index + count - 1]


EXCLUDED_SAMPLING_STREAM = {
    "stage": "conditioning_ranking_four_arm_sampling5k_v1",
    "seed": 406_020,
    "start_index": 0,
    "num_samples_per_arm": 5_000,
    "global_index_interval_inclusive": [0, 4_999],
    "per_sample_seed_interval_inclusive": _stream_interval(
        seed=406_020,
        start_index=0,
        count=5_000,
    ),
}
SAMPLING_PROTOCOL = {
    "num_samples_per_arm": 5_000,
    "start_index": 5_000,
    "sample_steps": 50,
    "num_train_timesteps": 1_000,
    "guidance_scale": 1.5,
    "guidance_rescale": 0.0,
    "cfg_batch_mode": "batched",
    "eta": 0.0,
    "seed": 506_020,
    "precision": "bf16",
    "weights": "ema",
    "class_schedule": "balanced_modulo",
    "sampler": "ddim",
    "clip_x0": True,
    "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
    "candidate_batch_sizes": [16, 32, 64],
    "baseline_batch_size": 16,
    "maximum_memory_fraction": 0.90,
    "warmup_forwards": 2,
    "measured_forwards": 5,
    "prefix_budgets": copy.deepcopy(METHOD_PREFIX_BUDGETS),
    "random_stream": {
        "scope": "per_global_sample_index",
        "seed_formula": "(seed + global_index) mod 2^63",
        "batch_size_invariant": True,
        "resume_index_invariant": True,
        "prefix_budgets_share_stream": True,
        "global_index_interval_inclusive": [5_000, 9_999],
        "per_sample_seed_interval_inclusive": _stream_interval(
            seed=506_020,
            start_index=5_000,
            count=5_000,
        ),
        "balanced_class_count_per_class": 5,
        "excluded_prior_stream": copy.deepcopy(EXCLUDED_SAMPLING_STREAM),
        "global_index_overlap_count": 0,
        "per_sample_seed_overlap_count": 0,
    },
}
AUTHORIZATION_BOUNDARY = {
    "standing_authorization_required": True,
    "exact_heldout_shared_pass_required": True,
    "exact_training_status_and_execution_receipt_required": True,
    "exact_step_5000_checkpoint_and_sidecar_required": True,
    "exact_revision_stage_and_output_binding_required": True,
    "independent_clean_checkout_required": True,
    "independent_random_stream_required": True,
    "five_consecutive_idle_gpu_polls_required": True,
    "shared_four_arm_preflight_batch_required": True,
    "sampling_preflight_allowed": True,
    "sampling_allowed": True,
    "samples_per_arm": 5_000,
    "training_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "full_100k_or_300k_launch_allowed": False,
    "release_authorization_allowed": False,
    "unrelated_process_signaling_allowed": False,
}
CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "formal_quality_gate": False,
    "replaces_active_quality_bridge": False,
    "replaces_prior_sampling_validation": False,
    "broad_generation_superiority_claim_allowed": False,
    "cofitok_specific_advantage_claim_allowed": False,
    "authorizes_training": False,
    "authorizes_checkpoint_promotion": False,
    "authorizes_full_100k_or_300k": False,
    "authorizes_release": False,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not _is_sha256(digest)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def clean_git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or not all(character in "0123456789abcdef" for character in revision)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    return {"revision": revision, "branch": branch, "tracked_dirty": False}


def validate_stream_independence(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if dict(protocol) != SAMPLING_PROTOCOL:
        raise ValueError("posttraining sampling protocol differs")
    random_stream = protocol.get("random_stream")
    if not isinstance(random_stream, Mapping):
        raise ValueError("posttraining random-stream contract is missing")
    current_indices = random_stream.get("global_index_interval_inclusive")
    current_seeds = random_stream.get("per_sample_seed_interval_inclusive")
    excluded = random_stream.get("excluded_prior_stream")
    if not isinstance(excluded, Mapping):
        raise ValueError("excluded prior sampling stream is missing")
    old_indices = excluded.get("global_index_interval_inclusive")
    old_seeds = excluded.get("per_sample_seed_interval_inclusive")
    if not all(
        isinstance(value, list)
        and len(value) == 2
        and all(type(item) is int for item in value)
        for value in (current_indices, current_seeds, old_indices, old_seeds)
    ):
        raise ValueError("sampling stream intervals are malformed")
    assert isinstance(current_indices, list)
    assert isinstance(current_seeds, list)
    assert isinstance(old_indices, list)
    assert isinstance(old_seeds, list)
    index_overlap = max(
        0,
        min(current_indices[1], old_indices[1])
        - max(current_indices[0], old_indices[0])
        + 1,
    )
    seed_overlap = max(
        0,
        min(current_seeds[1], old_seeds[1])
        - max(current_seeds[0], old_seeds[0])
        + 1,
    )
    if (
        index_overlap != 0
        or seed_overlap != 0
        or random_stream.get("global_index_overlap_count") != 0
        or random_stream.get("per_sample_seed_overlap_count") != 0
        or random_stream.get("balanced_class_count_per_class") != 5
    ):
        raise ValueError("posttraining sampling stream is not independent")
    return copy.deepcopy(dict(random_stream))


def _validate_classifier(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        value.get("name") != CLASS_FIDELITY_CLASSIFIER_NAME
        or value.get("weights_bytes") != CLASS_FIDELITY_CLASSIFIER_BYTES
        or value.get("weights_sha256") != CLASS_FIDELITY_CLASSIFIER_SHA256
        or value.get("num_classes") != 1_000
        or not isinstance(value.get("weights_path"), str)
        or not value.get("weights_path")
        or not _is_sha256(value.get("categories_sha256"))
        or not isinstance(value.get("preprocessing"), Mapping)
    ):
        raise ValueError("posttraining sampling classifier contract differs")
    return copy.deepcopy(dict(value))


def _validate_training_status(
    report: Mapping[str, Any],
    *,
    heldout_sources: Mapping[str, Any],
) -> dict[str, Any]:
    runs = report.get("runs")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("role") != TRAINING_STATUS_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != TRAINING_STAGE
        or report.get("revision") != TRAINING_GIT["revision"]
        or report.get("branch") != TRAINING_GIT["branch"]
        or report.get("output_root") != TRAINING_OUTPUT_ROOT
        or not isinstance(runs, Mapping)
        or set(runs) != set(RUN_NAMES)
    ):
        raise ValueError("posttraining source training status differs")
    if heldout_sources.get("training_status") is None:
        raise ValueError("heldout report lacks its training-status source")
    return copy.deepcopy(dict(report))


def _validate_training_execution_receipt(
    receipt: Mapping[str, Any],
    *,
    standing_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("role") != TRAINING_EXECUTION_RECEIPT_ROLE
        or receipt.get("status") != "authorized"
        or receipt.get("authorization_mode")
        != "active_standing_experiment_authorization"
        or receipt.get("stage") != TRAINING_STAGE
        or receipt.get("authorized_revision") != TRAINING_GIT["revision"]
        or receipt.get("authorized_branch") != TRAINING_GIT["branch"]
        or receipt.get("output_root") != TRAINING_OUTPUT_ROOT
        or receipt.get("authorization_boundary") != TRAINING_EXECUTION_BOUNDARY
        or receipt.get("claim_boundary") != TRAINING_EXECUTION_CLAIM_BOUNDARY
        or receipt.get("source_reports", {}).get("standing_authorization")
        != identity(standing_identity, label="standing experiment authorization")
    ):
        raise ValueError("posttraining source execution receipt differs")
    return copy.deepcopy(dict(receipt))


def _validate_heldout(
    report: Mapping[str, Any],
    *,
    training_status_identity: Mapping[str, Any],
) -> dict[str, Any]:
    sources = report.get("sources")
    contract = report.get("training_contract")
    decision = report.get("decision")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("role") != HELDOUT_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != HELDOUT_STAGE
        or report.get("training_git") != TRAINING_GIT
        or report.get("evaluator_git") != HELDOUT_EVALUATOR_GIT
        or report.get("output_root") != TRAINING_OUTPUT_ROOT
        or not isinstance(sources, Mapping)
        or not isinstance(contract, Mapping)
        or not isinstance(decision, Mapping)
        or dict(decision) != EXPECTED_HELDOUT_DECISION
        or sources.get("training_status")
        != identity(training_status_identity, label="5K training status")
    ):
        raise ValueError("posttraining heldout shared-pass evidence differs")
    runs = contract.get("runs")
    if (
        contract.get("revision") != TRAINING_GIT["revision"]
        or contract.get("branch") != TRAINING_GIT["branch"]
        or contract.get("output_root") != TRAINING_OUTPUT_ROOT
        or not isinstance(runs, Mapping)
        or set(runs) != set(RUN_NAMES)
    ):
        raise ValueError("posttraining heldout training contract differs")
    for method in ("cofitok", "dense_identity"):
        method_report = report.get("methods", {}).get(method)
        if not isinstance(method_report, Mapping) or method_report.get("pass") is not True:
            raise ValueError(f"posttraining heldout method did not pass: {method}")
    return copy.deepcopy(dict(contract))


def _validate_checkpoints(
    evidence: Mapping[str, Any],
    *,
    training_contract: Mapping[str, Any],
    heldout_sources: Mapping[str, Any],
    training_status: Mapping[str, Any],
) -> dict[str, Any]:
    if set(evidence) != set(RUN_NAMES):
        raise ValueError("posttraining checkpoint evidence set differs")
    contract_runs = training_contract["runs"]
    status_runs = training_status["runs"]
    result: dict[str, Any] = {}
    for run in RUN_NAMES:
        row = evidence.get(run)
        contract = contract_runs.get(run)
        status = status_runs.get(run)
        if not all(isinstance(value, Mapping) for value in (row, contract, status)):
            raise ValueError(f"{run} posttraining checkpoint evidence is missing")
        assert isinstance(row, Mapping)
        assert isinstance(contract, Mapping)
        assert isinstance(status, Mapping)
        checkpoint = identity(row.get("checkpoint", {}), label=f"{run} checkpoint")
        integrity_manifest = identity(
            row.get("integrity_manifest", {}),
            label=f"{run} checkpoint integrity manifest",
        )
        training_report = identity(
            row.get("training_report", {}),
            label=f"{run} training report",
        )
        expected_checkpoint = (
            f'{contract.get("run_dir")}/{EXPECTED_CHECKPOINT_FILENAME}'
        )
        heldout_checkpoint = heldout_sources.get("checkpoints", {}).get(run)
        heldout_integrity = heldout_sources.get(
            "checkpoint_integrity_manifests", {}
        ).get(run)
        heldout_report = heldout_sources.get("training_reports", {}).get(run)
        if (
            row.get("step") != EXPECTED_CHECKPOINT_STEP
            or checkpoint["path"] != expected_checkpoint
            or checkpoint["bytes"] != contract.get("checkpoint_bytes")
            or checkpoint["sha256"] != contract.get("checkpoint_sha256")
            or checkpoint != heldout_checkpoint
            or checkpoint != status.get("checkpoint")
            or integrity_manifest["path"] != f"{expected_checkpoint}.integrity.json"
            or integrity_manifest != contract.get("checkpoint_integrity_manifest")
            or integrity_manifest != heldout_integrity
            or integrity_manifest != status.get("integrity_manifest")
            or training_report["path"]
            != f'{contract.get("run_dir")}/training_report.json'
            or training_report != heldout_report
            or training_report != status.get("training_report")
            or int(contract.get("final_metrics", {}).get("step", -1))
            != EXPECTED_CHECKPOINT_STEP
        ):
            raise ValueError(f"{run} posttraining checkpoint evidence differs")
        result[run] = {
            "checkpoint": checkpoint,
            "integrity_manifest": integrity_manifest,
            "training_report": training_report,
            "step": EXPECTED_CHECKPOINT_STEP,
        }
    return result


def build_sampling_preparation(
    *,
    heldout_evaluation: Mapping[str, Any],
    heldout_evaluation_identity: Mapping[str, Any],
    training_status: Mapping[str, Any],
    training_status_identity: Mapping[str, Any],
    training_execution_receipt: Mapping[str, Any],
    training_execution_receipt_identity: Mapping[str, Any],
    checkpoint_evidence: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    classifier: Mapping[str, Any],
    runbook_identity: Mapping[str, Any],
    builder_git: Mapping[str, Any],
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    git = clean_git(builder_git, label="posttraining sampling preparation builder")
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if git != expected_git:
        raise ValueError("posttraining sampling preparation builder Git differs")
    if expected_output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError("posttraining sampling output root differs")
    heldout_sources = heldout_evaluation.get("sources")
    if not isinstance(heldout_sources, Mapping):
        raise ValueError("posttraining heldout sources are missing")
    status = _validate_training_status(
        training_status,
        heldout_sources=heldout_sources,
    )
    training_contract = _validate_heldout(
        heldout_evaluation,
        training_status_identity=training_status_identity,
    )
    standing = validate_standing_experiment_authorization(standing_authorization)
    standing_id = identity(
        standing_authorization_identity,
        label="standing experiment authorization",
    )
    execution_receipt = _validate_training_execution_receipt(
        training_execution_receipt,
        standing_identity=standing_id,
    )
    checkpoints = _validate_checkpoints(
        checkpoint_evidence,
        training_contract=training_contract,
        heldout_sources=heldout_sources,
        training_status=status,
    )
    validate_stream_independence(SAMPLING_PROTOCOL)
    return {
        "schema_version": SCHEMA_VERSION,
        "role": PREPARATION_ROLE,
        "status": "pass",
        "valid": True,
        "stage": STAGE,
        "scope": SCOPE,
        "output_root": expected_output_root,
        "git": git,
        "source_reports": {
            "heldout_evaluation": identity(
                heldout_evaluation_identity,
                label="5K heldout evaluation",
            ),
            "training_status": identity(
                training_status_identity,
                label="5K training status",
            ),
            "training_execution_receipt": identity(
                training_execution_receipt_identity,
                label="5K training execution receipt",
            ),
            "standing_authorization": standing_id,
            "runbook": identity(runbook_identity, label="posttraining sampling runbook"),
            "training_checkpoints": checkpoints,
        },
        "source_decision": copy.deepcopy(EXPECTED_HELDOUT_DECISION),
        "source_training_contract": training_contract,
        "source_training_execution_receipt": execution_receipt,
        "standing_authorization": standing,
        "classifier": _validate_classifier(classifier),
        "sampling_protocol": copy.deepcopy(SAMPLING_PROTOCOL),
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
    }


def validate_idle_gpu_evidence(
    report: Mapping[str, Any],
    *,
    expected_git: Mapping[str, Any],
    expected_output_root: str,
    required_polls: int = 5,
) -> dict[str, Any]:
    observations = report.get("observations")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("role") != IDLE_GPU_EVIDENCE_ROLE
        or report.get("status") != "pass"
        or report.get("stage") != STAGE
        or report.get("output_root") != expected_output_root
        or report.get("required_consecutive_idle_polls") != required_polls
        or clean_git(report.get("git", {}), label="idle GPU evidence")
        != clean_git(expected_git, label="expected posttraining sampling Git")
        or not isinstance(observations, list)
        or len(observations) != required_polls
    ):
        raise ValueError("posttraining sampling idle GPU evidence differs")
    previous = -math.inf
    for expected_poll_index, observation in enumerate(observations, start=1):
        if not isinstance(observation, Mapping):
            raise ValueError("posttraining idle GPU observation is malformed")
        timestamp = observation.get("observed_at_unix")
        if (
            observation.get("poll_index") != expected_poll_index
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or float(timestamp) <= previous
            or observation.get("gpu_compute_pids") != []
        ):
            raise ValueError("posttraining idle GPU observation differs")
        previous = float(timestamp)
    return copy.deepcopy(dict(report))


def build_sampling_execution_receipt(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    idle_gpu_evidence: Mapping[str, Any],
    idle_gpu_evidence_identity: Mapping[str, Any],
    batch_selection: Mapping[str, Any],
    batch_selection_identity: Mapping[str, Any],
    receipt_git: Mapping[str, Any],
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    git = clean_git(receipt_git, label="posttraining sampling receipt builder")
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if git != expected_git:
        raise ValueError("posttraining sampling receipt builder Git differs")
    if (
        preparation.get("schema_version") != SCHEMA_VERSION
        or preparation.get("role") != PREPARATION_ROLE
        or preparation.get("status") != "pass"
        or preparation.get("valid") is not True
        or preparation.get("stage") != STAGE
        or preparation.get("scope") != SCOPE
        or preparation.get("output_root") != expected_output_root
        or preparation.get("git") != expected_git
        or preparation.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
        or preparation.get("claim_boundary") != CLAIM_BOUNDARY
        or preparation.get("sampling_protocol") != SAMPLING_PROTOCOL
    ):
        raise ValueError("posttraining sampling preparation differs")
    standing = validate_standing_experiment_authorization(standing_authorization)
    standing_id = identity(
        standing_authorization_identity,
        label="standing experiment authorization",
    )
    if preparation.get("source_reports", {}).get("standing_authorization") != standing_id:
        raise ValueError("posttraining preparation standing authorization differs")
    idle = validate_idle_gpu_evidence(
        idle_gpu_evidence,
        expected_git=expected_git,
        expected_output_root=expected_output_root,
    )
    selected = batch_selection.get("selected")
    if (
        batch_selection.get("schema_version") != SCHEMA_VERSION
        or batch_selection.get("role")
        != "generation_conditioning_ranking_four_arm_sampling_batch_selection"
        or batch_selection.get("status") != "selected"
        or batch_selection.get("stage") != STAGE
        or batch_selection.get("output_root") != expected_output_root
        or batch_selection.get("git") != expected_git
        or batch_selection.get("sampling_protocol") != SAMPLING_PROTOCOL
        or batch_selection.get("checkpoints")
        != preparation.get("source_reports", {}).get("training_checkpoints")
        or not isinstance(selected, Mapping)
        or selected.get("batch_size") not in SAMPLING_PROTOCOL["candidate_batch_sizes"]
    ):
        raise ValueError("posttraining sampling batch selection differs")
    validate_stream_independence(SAMPLING_PROTOCOL)
    return {
        "schema_version": SCHEMA_VERSION,
        "role": EXECUTION_RECEIPT_ROLE,
        "status": "authorized",
        "authorization_mode": "active_standing_experiment_authorization",
        "stage": STAGE,
        "scope": SCOPE,
        "authorized_revision": expected_revision,
        "authorized_branch": expected_branch,
        "output_root": expected_output_root,
        "git": git,
        "source_reports": {
            "preparation": identity(
                preparation_identity,
                label="posttraining sampling preparation",
            ),
            "standing_authorization": standing_id,
            "idle_gpu_evidence": identity(
                idle_gpu_evidence_identity,
                label="idle GPU evidence",
            ),
            "batch_selection": identity(
                batch_selection_identity,
                label="four-arm posttraining sampling batch selection",
            ),
        },
        "standing_authorization": standing,
        "idle_gpu_evidence": idle,
        "selected_batch": copy.deepcopy(dict(selected)),
        "sampling_protocol": copy.deepcopy(SAMPLING_PROTOCOL),
        "training_checkpoints": copy.deepcopy(batch_selection["checkpoints"]),
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
    }
