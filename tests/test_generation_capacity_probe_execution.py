from __future__ import annotations

import copy

import pytest

from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_PREPARATION_BOUNDARY,
    CAPACITY_PROBE_PREPARATION_ROLE,
)
from cofitok.generation.capacity_probe_execution import (
    CAPACITY_PROBE_EXECUTION_SCOPE,
    CAPACITY_PROBE_LAUNCH_RECEIPT_ROLE,
    CAPACITY_PROBE_STORAGE_STAGE,
    EXECUTION_AUTHORIZATION_BOUNDARY,
    LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY,
    STANDING_AUTHORIZATION_EXACT_TEXT,
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    build_capacity_probe_execution_authorization,
    build_capacity_probe_launch_receipt,
    validate_capacity_probe_execution_authorization,
    validate_capacity_probe_launch_receipt_contract,
)


REVISION = "a" * 40
BRANCH = "scale/generation-capacity-probe-v1"
OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_capacity_probe_250m_10k_v1"
)
STORAGE_PATH = "/root/autodl-tmp/CoFiTok/checkpoints/generation"
RUN_DIRS = [
    f"{OUTPUT_ROOT}/base256_cofitok",
    f"{OUTPUT_ROOT}/base256_dense_identity",
]
BENCHMARK_ROOT = f"{OUTPUT_ROOT}/runtime_preflight/training"


def _identity(path: str, character: str = "b") -> dict[str, object]:
    return {"path": path, "bytes": 123, "sha256": character * 64}


def _git() -> dict[str, object]:
    return {
        "revision": REVISION,
        "branch": BRANCH,
        "tracked_dirty": False,
    }


def _standing() -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": STANDING_AUTHORIZATION_ROLE,
        "status": "active",
        "instruction": {
            "language": "zh-CN",
            "exact_text": STANDING_AUTHORIZATION_EXACT_TEXT,
            "received_at": "2026-08-13T00:11:00+08:00",
            "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
        },
        "preserved_safety_boundaries": copy.deepcopy(
            STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
        ),
    }


def _matched_config() -> dict[str, object]:
    return {
        "status": "pass",
        "recipe_stage": "stability_capacity_probe",
        "parameter_counts": {
            "cofitok": 250_153_763,
            "dense_identity": 250_135_043,
        },
        "relative_parameter_gap": 0.0000748395737577641,
        "effective_batches": {
            "cofitok": {
                "micro_batch_size": 16,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 64,
            },
            "dense_identity": {
                "micro_batch_size": 16,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 64,
            },
        },
        "recipe_schema": "cofitok_generation_training_recipe_v4",
    }


def _config_validation() -> dict[str, object]:
    matched = _matched_config()
    return {
        "status": "pass",
        "mismatches": [],
        "cofitok": {"parameter_count": 250_153_763},
        "dense": {"parameter_count": 250_135_043},
        "relative_parameter_gap": matched["relative_parameter_gap"],
        "training_recipe": {
            "stage": "stability_capacity_probe",
            "valid": True,
            "schema": matched["recipe_schema"],
            "effective_batches": copy.deepcopy(matched["effective_batches"]),
        },
    }


def _preparation() -> dict[str, object]:
    capacity_configs = {
        "cofitok": _identity("/configs/capacity_cofitok.json", "c"),
        "dense_identity": _identity("/configs/capacity_dense.json", "d"),
    }
    return {
        "schema_version": 1,
        "status": "prepared",
        "role": CAPACITY_PROBE_PREPARATION_ROLE,
        "selection": {
            "dataset": "imagenet_256",
            "configured_training_horizon": 100_000,
            "stop_after_step": 10_000,
            "effective_batch_size": 64,
            "images_seen_per_training_arm": 640_000,
            "base_channels": [128, 256],
            "output_root": OUTPUT_ROOT,
            "fresh_training_arms": [
                "base256_cofitok",
                "base256_dense_identity",
            ],
            "preserved_reference_arms": [
                "base128_cofitok",
                "base128_dense_identity",
            ],
            "training_intervention": ["model.base_channels"],
            "logistical_differences": [
                "name",
                "runtime.protected_checkpoint_steps",
            ],
        },
        "matched_training_contract": {
            "base256": _matched_config(),
            "capacity_configs": capacity_configs,
            "configured_100k_schedule_preserved_at_step_10k": True,
            "fresh_initialization_required": True,
            "resume_from_base128_forbidden": True,
        },
        "checkpoint_references": {
            "cofitok": {},
            "dense_identity": {},
        },
        "evaluation_contract": {
            "arms": [
                "base128_cofitok",
                "base128_dense_identity",
                "base256_cofitok",
                "base256_dense_identity",
            ],
            "checkpoint_step": 10_000,
            "weights": "ema",
            "sampler": "ddim",
            "sample_steps": 50,
            "guidance_scale": 1.5,
            "samples_per_arm": 2_048,
            "fixed_random_stream_across_arms": True,
            "role": "non_claim_capacity_causal_diagnostic",
        },
        "decision_contract": {
            "result_can_authorize_full_300k": False,
            "new_source_compatible_decision_required": True,
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_PROBE_PREPARATION_BOUNDARY
        ),
    }


def _authorization() -> tuple[dict[str, object], dict[str, object]]:
    preparation_identity = _identity("/evidence/preparation.json")
    authorization = build_capacity_probe_execution_authorization(
        preparation=_preparation(),
        preparation_identity=preparation_identity,
        standing_authorization=_standing(),
        standing_authorization_identity=_identity(
            "/evidence/standing_authorization.json",
            "e",
        ),
        execution_git=_git(),
        output_root=OUTPUT_ROOT,
    )
    return authorization, preparation_identity


def _launch_receipt() -> dict[str, object]:
    preparation = _preparation()
    authorization, preparation_identity = _authorization()
    sources = {
        "preparation": preparation_identity,
        "execution_authorization": _identity(
            "/evidence/execution_authorization.json",
            "f",
        ),
        "cofitok_config": {
            **preparation["matched_training_contract"]["capacity_configs"]["cofitok"],
            "path": "/execution/configs/capacity_cofitok.json",
        },
        "dense_config": {
            **preparation["matched_training_contract"]["capacity_configs"][
                "dense_identity"
            ],
            "path": "/execution/configs/capacity_dense.json",
        },
        "config_validation": _identity("/evidence/config_validation.json", "1"),
        "storage_capacity": _identity("/evidence/storage_capacity.json", "2"),
        "runtime_selection": _identity("/evidence/runtime_selection.json", "3"),
    }
    storage = {
        "schema_version": 2,
        "role": "generation_storage_capacity_preflight",
        "stage": CAPACITY_PROBE_STORAGE_STAGE,
        "status": "pass",
        "git": _git(),
        "filesystem": {"path": STORAGE_PATH, "free_bytes": 200 * 1024**3},
        "plan": {
            "sample_count": 8_192,
            "checkpoint_count": 4,
            "checkpoint_size_multiplier": 4.0,
            "required_free_bytes": 100 * 1024**3,
        },
        "headroom_bytes": 100 * 1024**3,
    }
    config_sha = {
        "cofitok": sources["cofitok_config"]["sha256"],
        "dense_identity": sources["dense_config"]["sha256"],
    }
    runtime = {
        "status": "selected",
        "git_revision": REVISION,
        "config_sha256": config_sha,
        "benchmark_root": BENCHMARK_ROOT,
        "runtime_environment_sha256": "4" * 64,
        "selected": {
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 32,
        },
        "selection_lock": {
            "training_run_dirs": RUN_DIRS,
            "training_target_steps": 100_000,
            "expected_effective_batch_size": 64,
            "git": _git(),
            "config_sha256": config_sha,
            "benchmark_root": BENCHMARK_ROOT,
        },
    }
    return build_capacity_probe_launch_receipt(
        preparation=preparation,
        execution_authorization=authorization,
        config_validation=_config_validation(),
        storage_capacity=storage,
        runtime_selection=runtime,
        source_identities=sources,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        output_root=OUTPUT_ROOT,
        storage_path=STORAGE_PATH,
        training_run_dirs=RUN_DIRS,
        benchmark_root=BENCHMARK_ROOT,
        training_state_absent_at_launch=True,
    )


def test_standing_instruction_authorizes_only_exact_10k_probe() -> None:
    authorization, preparation_identity = _authorization()
    evidence = validate_capacity_probe_execution_authorization(
        authorization,
        preparation_identity=preparation_identity,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        expected_output_root=OUTPUT_ROOT,
    )
    assert authorization["scope"] == CAPACITY_PROBE_EXECUTION_SCOPE
    assert evidence["authorization_boundary"] == EXECUTION_AUTHORIZATION_BOUNDARY
    assert evidence["authorization_boundary"]["intentional_stop_step_required"] == 10_000
    assert evidence["authorization_boundary"]["configured_100k_completion_allowed"] is False
    assert evidence["authorization_boundary"]["full_300k_launch_allowed"] is False


def test_authorization_rejects_standing_instruction_or_boundary_drift() -> None:
    authorization, preparation_identity = _authorization()
    drifted = copy.deepcopy(authorization)
    drifted["standing_authorization"]["validated_record"]["instruction"][
        "exact_text"
    ] = "run everything"
    with pytest.raises(ValueError, match="standing experiment authorization"):
        validate_capacity_probe_execution_authorization(
            drifted,
            preparation_identity=preparation_identity,
            expected_revision=REVISION,
            expected_branch=BRANCH,
            expected_output_root=OUTPUT_ROOT,
        )

    drifted = copy.deepcopy(authorization)
    drifted["authorization_boundary"]["configured_100k_completion_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_capacity_probe_execution_authorization(
            drifted,
            preparation_identity=preparation_identity,
            expected_revision=REVISION,
            expected_branch=BRANCH,
            expected_output_root=OUTPUT_ROOT,
        )


def test_launch_receipt_is_replayable_and_non_authorizing() -> None:
    receipt = _launch_receipt()
    evidence = validate_capacity_probe_launch_receipt_contract(
        receipt,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )
    assert receipt["role"] == CAPACITY_PROBE_LAUNCH_RECEIPT_ROLE
    assert receipt["authorization_boundary"] == LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY
    assert evidence["stop_after_step"] == 10_000
    assert evidence["authorization_boundary"]["full_300k_launch_allowed"] is False

    drifted = copy.deepcopy(receipt)
    drifted["authorization_boundary"]["scaling_authorization_created"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_capacity_probe_launch_receipt_contract(
            drifted,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_launch_receipt_accepts_relocated_byte_identical_configs() -> None:
    receipt = _launch_receipt()
    assert receipt["status"] == "pass"
    assert receipt["source_reports"]["cofitok_config"]["path"].startswith(
        "/execution/"
    )


@pytest.mark.parametrize(
    "entrypoint",
    [
        "scripts.build_generation_capacity_probe_execution_authorization",
        "scripts.verify_generation_capacity_probe_execution_authorization",
        "scripts.build_generation_capacity_probe_launch_receipt",
        "scripts.verify_generation_capacity_probe_launch_receipt",
    ],
)
def test_capacity_probe_execution_entrypoints_import(entrypoint: str) -> None:
    __import__(entrypoint)
