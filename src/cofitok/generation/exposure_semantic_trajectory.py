from __future__ import annotations

import copy
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_probe import (
    validate_standing_experiment_authorization,
)


SCHEMA_VERSION = 1
PREPARATION_ROLE = "generation_exposure_semantic_trajectory_preparation"
AUTHORIZATION_ROLE = "generation_exposure_semantic_trajectory_execution_authorization"
REPORT_ROLE = "generation_exposure_semantic_trajectory_report"
SUPERVISOR_ROLE = "generation_exposure_semantic_trajectory_supervisor"
SCOPE = "matched_5k_existing_checkpoint_cpu_exposure_semantic_trajectory_only"
STAGE = "exposure_semantic_trajectory_pair5k_v1"

OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "exposure_semantic_trajectory_pair5k_v1"
)
PAIR_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher"
)
REJECTED_POSTEVALUATION_PATH = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "semantic_residual_alignment_four_arm_probe1k_v1/reports/"
    "semantic_residual_alignment_posteval_v1/postevaluation.json"
)
STANDING_AUTHORIZATION_PATH = (
    "/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json"
)

TRAINING_REVISION = "59db142fc45d69dc92bb0333be5ac2d0162d9dc4"
DATASET_IDENTITY_SHA256 = (
    "97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741"
)
RUNTIME_ENVIRONMENT_SHA256 = (
    "d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e"
)
REJECTED_POSTEVALUATION_SHA256 = (
    "3e63f7ed934cbc017e406dd71ccc598eb7e23e6d0ddbf998e98a8108d0d5675a"
)
STANDING_AUTHORIZATION_SHA256 = (
    "5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df"
)

METHOD_RUN_DIRS = {
    "cofitok": "cofitok_rgbtail3_rollout_x0_u2_ema_teacher",
    "dense_identity": "dense_rollout_x0_u2_ema_teacher",
}
CHECKPOINT_STEPS = [1_250, 2_500, 5_000]
EXPECTED_PARAMETER_COUNTS = {
    "cofitok": 62_834_083,
    "dense_identity": 62_824_707,
}
EXPECTED_RUN_FILES = {
    "cofitok": {
        "training_report.json": "3de577d830d2ba07fc399296e98ebefd66d1ea9f3ccfe3552f98ccc4243ad4c7",
        "train_metrics.jsonl": "14c065bbf9d9ff2748e18bd6662e155024461e39716c9405a1599e92374b0a00",
        "run_manifest.json": "6435685aa62bdb71679ae1d3daece439f883e142ffa2e7c934244434dbff0db7",
        "latest.json": "97575ff92c0e20ffd1a80d2b64448e0588aa348aa7e7a05171c23182ba35d2c3",
    },
    "dense_identity": {
        "training_report.json": "d884cced7b58d03cb88402bc87b90bf7c7f8cd29186eceb4b4e1f5bf17d00474",
        "train_metrics.jsonl": "1c90167c7bd8782295d855b9d85bcd6a42779c7b99052991b41a424f272b1efc",
        "run_manifest.json": "6bede166ec6177fae3eb75edce5bc2cc020256934f250f8852cae033f50b4e17",
        "latest.json": "365f2f980465055385fb3ea90b9f8908aca208baca89e1cf3e1126ceb357505b",
    },
}
EXPECTED_CHECKPOINTS = {
    "cofitok": {
        1_250: {
            "bytes": 1_006_321_770,
            "sha256": "f234c142aaa39e21520bcf8ec2d3458feeef85c26c8431209dfab62d4a9610c9",
            "sidecar_sha256": "1856a42f2bd24840ea9eb78d97712ca444a5d01aac7e14615242796a6943e5dc",
        },
        2_500: {
            "bytes": 1_006_321_770,
            "sha256": "bc9e7942f9fac143064a901a78f33ddf7bd638c8df4d2409481e1305a794d6df",
            "sidecar_sha256": "8ba61c2e9c4e56f36bd1122375bae665c748122d6f230d878cb03312f85b5970",
        },
        5_000: {
            "bytes": 1_006_321_770,
            "sha256": "cb432c75ccbc4eba00dab878e43dd0e45740ebdd9a6b95e0cd014ce97b6d6450",
            "sidecar_sha256": "c5f1c6906096bff6402a2ab83c2d3031c20916e5cc17669bb4f6407a2711fbb0",
        },
    },
    "dense_identity": {
        1_250: {
            "bytes": 1_006_120_150,
            "sha256": "227ca21455f1bc27295cd9ff8f325b6960932da373122192e10053cb17d1463a",
            "sidecar_sha256": "f26f414263b6ca408d4435f074e89364cdaabcee4b40575601de56d89b56fce3",
        },
        2_500: {
            "bytes": 1_006_120_150,
            "sha256": "16711ab7b400bd36ab758b0ece897c240f27739412993c82bd889699928ebcdc",
            "sidecar_sha256": "31497fca63aaf715f92fc212df2fa01b2746c7df3e048599f130736c02aea6a6",
        },
        5_000: {
            "bytes": 1_006_120_150,
            "sha256": "92bc9aa315eec1e06f201961dbc7d5638cdadf471bc4dbdab0a781252254ecab",
            "sidecar_sha256": "1a77ed4a0ad1f4463bcdd21e552d57d0a58cb1cfb557da60f4793f64825c39be",
        },
    },
}

EVALUATION_REQUEST = {
    "weights": "ema",
    "num_samples": 16,
    "start_label": 128,
    "wrong_label_offset": 250,
    "timesteps": [100, 500, 700, 900],
    "noise_seed": 314159,
    "threads": 2,
}
ELIGIBLE_TIMESTEPS = [500, 700, 900]
DESCRIPTIVE_TIMESTEPS = [100]
SIGNIFICANCE_LEVEL = 0.05

EXECUTION_BOUNDARY = {
    "cpu_checkpoint_evaluation_allowed": True,
    "checkpoint_payload_read_allowed": True,
    "source_physical_hash_replay_allowed": True,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "generation_sampling_allowed": False,
    "gpu_execution_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "non_authorizing": True,
    "generates_new_samples": False,
    "replaces_formal_quality_gate": False,
    "generation_advantage_proven": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "authorizes_checkpoint_promotion": False,
    "authorizes_followup_training": False,
    "authorizes_full_training": False,
    "authorizes_300k_training": False,
    "authorizes_export": False,
    "authorizes_release": False,
    "authorizes_process_signals": False,
}


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not is_sha256(digest)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def validate_clean_git(
    value: Mapping[str, Any],
    *,
    label: str,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
) -> dict[str, Any]:
    expected = {
        "revision": expected_revision,
        "tree": expected_tree,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if dict(value) != expected:
        raise ValueError(f"{label} Git identity differs")
    return expected


def validate_rejected_postevaluation(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision")
    boundary = report.get("claim_boundary")
    if (
        report.get("schema_version") != 1
        or report.get("role")
        != "generation_semantic_residual_alignment_four_arm_postevaluation"
        or report.get("status") != "completed"
        or report.get("generation_advantage_proven") is not False
        or not isinstance(decision, Mapping)
        or decision.get("method_passes")
        != {"cofitok": False, "dense_identity": False}
        or decision.get("shared_semantic_alignment_recovery_supported") is not False
        or decision.get("recommended_next_action")
        != "reject_residual_alignment_candidate"
        or not isinstance(boundary, Mapping)
        or boundary.get("diagnostic_only") is not True
        or any(
            boundary.get(field) is not False
            for field in (
                "authorizes_training",
                "authorizes_sampling",
                "authorizes_checkpoint_promotion",
                "authorizes_followup_training",
                "authorizes_full_training",
                "authorizes_300k_training",
                "authorizes_export",
                "authorizes_release",
                "authorizes_process_signals",
            )
        )
    ):
        raise ValueError("rejected residual-alignment postevaluation differs")
    return copy.deepcopy(dict(report))


def validate_preparation(
    preparation: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: str = OUTPUT_ROOT,
) -> dict[str, Any]:
    validate_clean_git(
        preparation.get("git", {}),
        label="exposure-trajectory preparation",
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_branch=expected_branch,
    )
    sources = preparation.get("sources")
    methods = preparation.get("methods")
    if (
        preparation.get("schema_version") != SCHEMA_VERSION
        or preparation.get("role") != PREPARATION_ROLE
        or preparation.get("status") != "pass"
        or preparation.get("scope") != SCOPE
        or preparation.get("stage") != STAGE
        or preparation.get("output_root") != expected_output_root
        or preparation.get("pair_root") != PAIR_ROOT
        or preparation.get("evaluation_request") != EVALUATION_REQUEST
        or preparation.get("eligible_timesteps") != ELIGIBLE_TIMESTEPS
        or preparation.get("descriptive_timesteps") != DESCRIPTIVE_TIMESTEPS
        or preparation.get("checkpoint_steps") != CHECKPOINT_STEPS
        or preparation.get("execution_boundary") != EXECUTION_BOUNDARY
        or preparation.get("claim_boundary") != CLAIM_BOUNDARY
        or preparation.get("generation_advantage_proven") is not False
        or not isinstance(sources, Mapping)
        or set(sources)
        != {
            "standing_authorization",
            "rejected_postevaluation",
            "evaluator",
            "report_builder",
            "preparation_builder",
            "runbook",
        }
        or not isinstance(methods, Mapping)
        or set(methods) != set(METHOD_RUN_DIRS)
    ):
        raise ValueError("exposure-trajectory preparation contract differs")
    rejected = validate_identity(
        sources.get("rejected_postevaluation", {}),
        label="rejected postevaluation",
    )
    standing = validate_identity(
        sources.get("standing_authorization", {}),
        label="standing authorization",
    )
    if (
        rejected["path"] != REJECTED_POSTEVALUATION_PATH
        or rejected["sha256"] != REJECTED_POSTEVALUATION_SHA256
        or standing["path"] != STANDING_AUTHORIZATION_PATH
        or standing["sha256"] != STANDING_AUTHORIZATION_SHA256
    ):
        raise ValueError("exposure-trajectory upstream identity differs")
    code_suffixes = {
        "evaluator": "/scripts/evaluate_generation_conditioning_sensitivity.py",
        "report_builder": "/scripts/build_generation_exposure_semantic_trajectory_report.py",
        "preparation_builder": "/scripts/prepare_generation_exposure_semantic_trajectory.py",
        "runbook": "/artifacts/runbooks/generation_exposure_semantic_trajectory_pair5k_v1.sh",
    }
    for name, suffix in code_suffixes.items():
        identity = validate_identity(sources.get(name, {}), label=name)
        if not identity["path"].endswith(suffix):
            raise ValueError(f"exposure-trajectory code source path differs: {name}")
    pair_contract = preparation.get("pair_contract")
    if not isinstance(pair_contract, Mapping) or pair_contract.get("issues") != []:
        raise ValueError("exposure-trajectory matched pair contract did not pass")
    for method, expected_run in METHOD_RUN_DIRS.items():
        row = methods.get(method)
        expected_run_dir = f"{PAIR_ROOT}/{expected_run}"
        if (
            not isinstance(row, Mapping)
            or row.get("run_dir") != expected_run_dir
            or row.get("training_revision") != TRAINING_REVISION
            or row.get("dataset_identity_sha256") != DATASET_IDENTITY_SHA256
            or row.get("runtime_environment_sha256")
            != RUNTIME_ENVIRONMENT_SHA256
            or row.get("parameter_count") != EXPECTED_PARAMETER_COUNTS[method]
            or row.get("training_complete") is not True
            or row.get("completed_steps") != 5_000
            or row.get("samples_seen") != 320_000
            or row.get("metrics_strictly_increasing") is not True
            or row.get("metrics_exposure_bound") is not True
            or row.get("semantic_auxiliary_weights_zero") is not True
        ):
            raise ValueError(f"{method} preparation evidence differs")
        source_files = row.get("source_files")
        metrics_summary = row.get("metrics_summary")
        if (
            not isinstance(source_files, Mapping)
            or set(source_files) != set(EXPECTED_RUN_FILES[method])
            or not isinstance(metrics_summary, Mapping)
        ):
            raise ValueError(f"{method} source file inventory differs")
        for filename, expected_sha in EXPECTED_RUN_FILES[method].items():
            identity = validate_identity(
                source_files.get(filename, {}),
                label=f"{method} {filename}",
            )
            if (
                identity["path"] != f"{expected_run_dir}/{filename}"
                or identity["sha256"] != expected_sha
            ):
                raise ValueError(f"{method} {filename} identity differs")
        if (
            metrics_summary.get("identity") != source_files["train_metrics.jsonl"]
            or metrics_summary.get("row_count") != 201
            or metrics_summary.get("first_step") != 1
            or metrics_summary.get("last_step") != 5_000
            or metrics_summary.get("strictly_increasing") is not True
            or metrics_summary.get("samples_seen_equals_step_times_64") is not True
            or metrics_summary.get("all_float_values_finite") is not True
        ):
            raise ValueError(f"{method} metrics summary differs")
        checkpoints = row.get("checkpoints")
        if not isinstance(checkpoints, list) or [
            item.get("step") for item in checkpoints
        ] != CHECKPOINT_STEPS:
            raise ValueError(f"{method} checkpoint inventory differs")
        for item in checkpoints:
            step = int(item["step"])
            expected = EXPECTED_CHECKPOINTS[method][step]
            checkpoint = validate_identity(
                item.get("checkpoint", {}),
                label=f"{method} checkpoint {step}",
            )
            sidecar = validate_identity(
                item.get("sidecar", {}),
                label=f"{method} sidecar {step}",
            )
            if (
                checkpoint["path"]
                != f"{expected_run_dir}/checkpoint_step_{step:08d}.pt"
                or checkpoint["bytes"] != expected["bytes"]
                or checkpoint["sha256"] != expected["sha256"]
                or sidecar["path"]
                != f"{expected_run_dir}/checkpoint_step_{step:08d}.pt.integrity.json"
                or sidecar["sha256"] != expected["sidecar_sha256"]
                or item.get("physical_sha256_verified") is not True
                or item.get("sidecar_verified") is not True
                or item.get("samples_seen") != step * 64
            ):
                raise ValueError(f"{method} checkpoint {step} binding differs")
    return copy.deepcopy(dict(preparation))


def build_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    rejected_postevaluation: Mapping[str, Any],
    rejected_postevaluation_identity: Mapping[str, Any],
    runbook_identity: Mapping[str, Any],
    authorization_git: Mapping[str, Any],
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: str = OUTPUT_ROOT,
) -> dict[str, Any]:
    preparation_id = validate_identity(
        preparation_identity,
        label="exposure-trajectory preparation",
    )
    standing_id = validate_identity(
        standing_authorization_identity,
        label="standing authorization",
    )
    rejected_id = validate_identity(
        rejected_postevaluation_identity,
        label="rejected postevaluation",
    )
    runbook_id = validate_identity(runbook_identity, label="exposure runbook")
    git = validate_clean_git(
        authorization_git,
        label="exposure-trajectory authorization",
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_branch=expected_branch,
    )
    validate_preparation(
        preparation,
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_branch=expected_branch,
        expected_output_root=expected_output_root,
    )
    validate_standing_experiment_authorization(standing_authorization)
    validate_rejected_postevaluation(rejected_postevaluation)
    if (
        standing_id["path"] != STANDING_AUTHORIZATION_PATH
        or standing_id["sha256"] != STANDING_AUTHORIZATION_SHA256
        or rejected_id["path"] != REJECTED_POSTEVALUATION_PATH
        or rejected_id["sha256"] != REJECTED_POSTEVALUATION_SHA256
        or preparation["sources"]["standing_authorization"] != standing_id
        or preparation["sources"]["rejected_postevaluation"] != rejected_id
    ):
        raise ValueError("exposure-trajectory authorization sources differ")
    return {
        "schema_version": SCHEMA_VERSION,
        "role": AUTHORIZATION_ROLE,
        "status": "authorized",
        "authorization_mode": "active_standing_experiment_authorization",
        "scope": SCOPE,
        "stage": STAGE,
        "authorized_git": git,
        "output_root": expected_output_root,
        "source_reports": {
            "preparation": preparation_id,
            "standing_authorization": standing_id,
            "rejected_postevaluation": rejected_id,
            "runbook": runbook_id,
        },
        "scientific_route": {
            "rejected_candidate": "shared_low_frequency_x0_residual_alignment",
            "rejected_method_passes": {
                "cofitok": False,
                "dense_identity": False,
            },
            "next_question": "does_existing_training_exposure_improve_semantic_alignment",
            "generation_advantage_proven": False,
        },
        "execution_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
        "generation_advantage_proven": False,
    }


def validate_execution_authorization(
    authorization: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    expected = build_execution_authorization(**kwargs)
    if dict(authorization) != expected:
        raise ValueError("exposure-trajectory execution authorization differs")
    return expected
