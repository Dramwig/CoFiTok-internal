from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_probe import (
    validate_standing_experiment_authorization,
)
from cofitok.generation_class_fidelity import (
    CLASS_FIDELITY_CLASSIFIER_BYTES,
    CLASS_FIDELITY_CLASSIFIER_NAME,
    CLASS_FIDELITY_CLASSIFIER_SHA256,
)


SCHEMA_VERSION = 1
STAGE = "conditioning_ranking_four_arm_sampling5k_v1"
SCOPE = "imagenet256_four_arm_conditioning_ranking_sampling5k_only"
EXPECTED_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "conditioning_ranking_four_arm_sampling5k_v1"
)
PREPARATION_ROLE = (
    "generation_conditioning_ranking_four_arm_sampling_preparation"
)
EXECUTION_RECEIPT_ROLE = (
    "generation_conditioning_ranking_four_arm_sampling_execution_receipt"
)
IDLE_GPU_EVIDENCE_ROLE = (
    "generation_conditioning_ranking_four_arm_sampling_idle_gpu_evidence"
)
POSTEVALUATION_ROLE = "generation_conditioning_ranking_four_arm_postevaluation"
RUN_NAMES = (
    "control_cofitok",
    "ranked_cofitok",
    "control_dense_identity",
    "ranked_dense_identity",
)
METHOD_PREFIX_BUDGETS = {"cofitok": 8, "dense_identity": 1}
RUN_METHODS = {
    "control_cofitok": "cofitok",
    "ranked_cofitok": "cofitok",
    "control_dense_identity": "dense_identity",
    "ranked_dense_identity": "dense_identity",
}
EXPECTED_POSTEVALUATION_DECISION = {
    "method_passes": {"cofitok": True, "dense_identity": True},
    "shared_semantic_alignment_recovery_supported": True,
    "cofitok_specific_advantage_claim_allowed": False,
    "recommended_next_action": (
        "consider_separately_authorized_matched_sampling_validation"
    ),
}
SAMPLING_PROTOCOL = {
    "num_samples_per_arm": 5_000,
    "start_index": 0,
    "sample_steps": 50,
    "num_train_timesteps": 1_000,
    "guidance_scale": 1.5,
    "guidance_rescale": 0.0,
    "cfg_batch_mode": "batched",
    "eta": 0.0,
    "seed": 406_020,
    "precision": "bf16",
    "weights": "ema",
    "class_schedule": "balanced_modulo",
    "sampler": "ddim",
    "clip_x0": True,
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
    },
}
AUTHORIZATION_BOUNDARY = {
    "standing_authorization_required": True,
    "exact_postevaluation_pass_required": True,
    "exact_revision_stage_and_output_binding_required": True,
    "independent_clean_checkout_required": True,
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
    "broad_generation_superiority_claim_allowed": False,
    "cofitok_specific_advantage_claim_allowed": False,
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
    return {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }


def _validate_postevaluation(report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        report.get("schema_version") != 1
        or report.get("role") != POSTEVALUATION_ROLE
        or report.get("status") != "completed"
        or report.get("decision") != EXPECTED_POSTEVALUATION_DECISION
    ):
        raise ValueError("conditioning-ranking postevaluation did not authorize sampling")
    contract = report.get("training_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("conditioning-ranking training contract is missing")
    revision = contract.get("revision")
    branch = contract.get("branch")
    runs = contract.get("runs")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or not isinstance(branch, str)
        or not branch
        or not isinstance(runs, Mapping)
        or set(runs) != set(RUN_NAMES)
    ):
        raise ValueError("conditioning-ranking training contract differs")
    return copy.deepcopy(dict(contract))


def _validate_checkpoints(
    evidence: Mapping[str, Any],
    *,
    training_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if set(evidence) != set(RUN_NAMES):
        raise ValueError("conditioning-ranking checkpoint evidence set differs")
    contract_runs = training_contract["runs"]
    result: dict[str, Any] = {}
    for run in RUN_NAMES:
        row = evidence.get(run)
        contract = contract_runs.get(run)
        if not isinstance(row, Mapping) or not isinstance(contract, Mapping):
            raise ValueError(f"{run} checkpoint evidence is missing")
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
            f'{contract.get("run_dir")}/checkpoint_step_00001000.pt'
        )
        if (
            row.get("step") != 1_000
            or checkpoint["path"] != expected_checkpoint
            or checkpoint["bytes"] != contract.get("checkpoint_bytes")
            or checkpoint["sha256"] != contract.get("checkpoint_sha256")
            or integrity_manifest["path"] != f"{expected_checkpoint}.integrity.json"
            or training_report["path"]
            != f'{contract.get("run_dir")}/training_report.json'
        ):
            raise ValueError(f"{run} checkpoint evidence differs from postevaluation")
        result[run] = {
            "checkpoint": checkpoint,
            "integrity_manifest": integrity_manifest,
            "training_report": training_report,
            "step": 1_000,
        }
    return result


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
        raise ValueError("conditioning-ranking classifier contract differs")
    return copy.deepcopy(dict(value))


def build_sampling_preparation(
    *,
    postevaluation: Mapping[str, Any],
    postevaluation_identity: Mapping[str, Any],
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
    git = clean_git(builder_git, label="sampling preparation builder")
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if git != expected_git:
        raise ValueError("sampling preparation builder Git differs")
    if expected_output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError("conditioning-ranking sampling output root differs")
    training_contract = _validate_postevaluation(postevaluation)
    checkpoints = _validate_checkpoints(
        checkpoint_evidence,
        training_contract=training_contract,
    )
    validated_standing = validate_standing_experiment_authorization(
        standing_authorization
    )
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
            "postevaluation": identity(
                postevaluation_identity,
                label="conditioning-ranking postevaluation",
            ),
            "standing_authorization": identity(
                standing_authorization_identity,
                label="standing experiment authorization",
            ),
            "runbook": identity(runbook_identity, label="sampling runbook"),
            "training_checkpoints": checkpoints,
        },
        "source_decision": copy.deepcopy(EXPECTED_POSTEVALUATION_DECISION),
        "source_training_contract": training_contract,
        "standing_authorization": validated_standing,
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
        != clean_git(expected_git, label="expected sampling Git")
        or not isinstance(observations, list)
        or len(observations) != required_polls
    ):
        raise ValueError("conditioning-ranking idle GPU evidence differs")
    previous = -math.inf
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise ValueError("conditioning-ranking idle GPU observation is malformed")
        timestamp = observation.get("observed_at_unix")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or float(timestamp) <= previous
            or observation.get("gpu_compute_pids") != []
        ):
            raise ValueError("conditioning-ranking idle GPU observation differs")
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
    git = clean_git(receipt_git, label="sampling execution receipt builder")
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if git != expected_git:
        raise ValueError("sampling execution receipt builder Git differs")
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
        raise ValueError("conditioning-ranking sampling preparation differs")
    standing = validate_standing_experiment_authorization(standing_authorization)
    standing_id = identity(
        standing_authorization_identity,
        label="standing experiment authorization",
    )
    if preparation.get("source_reports", {}).get("standing_authorization") != standing_id:
        raise ValueError("sampling preparation standing authorization differs")
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
        raise ValueError("conditioning-ranking sampling batch selection differs")
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
                label="sampling preparation",
            ),
            "standing_authorization": standing_id,
            "idle_gpu_evidence": identity(
                idle_gpu_evidence_identity,
                label="idle GPU evidence",
            ),
            "batch_selection": identity(
                batch_selection_identity,
                label="four-arm sampling batch selection",
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
