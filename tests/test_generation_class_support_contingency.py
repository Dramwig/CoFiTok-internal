from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import pytest
from PIL import Image

import cofitok.generation_class_support_contingency as contingency
from cofitok.diffusion import select_sampling_timesteps


def _identity(path: str, digest: str, *, size: int = 100) -> dict[str, object]:
    return {"path": path, "bytes": size, "sha256": digest}


def _git(character: str = "a") -> dict[str, object]:
    return {
        "revision": character * 40,
        "tree": character * 40,
        "branch": "analysis/class-support-test",
        "tracked_dirty": False,
    }


def _sampling(prefix: int) -> dict[str, object]:
    return {
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "inference_api": {
            "name": "cofitok.generation.GenerationSession",
            "version": 1,
        },
        "sampler": "ddim",
        "num_samples": 10_000,
        "start_index": 0,
        "batch_size": 32,
        "sample_steps": 100,
        "num_train_timesteps": 1000,
        "actual_timesteps": select_sampling_timesteps(1000, 100),
        "image_shape": [3, 256, 256],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "class_schedule": "balanced_modulo",
        "prefix_budgets": [prefix],
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
        "random_stream": {
            "scope": "per_global_sample_index",
            "seed_formula": "(seed + global_index) mod 2^63",
            "prefix_budgets_share_stream": True,
            "batch_size_invariant": True,
            "resume_index_invariant": True,
        },
        "sample_set_digest": {
            "algorithm": "sha256",
            "framing": "filename_utf8_nul_file_bytes_nul",
        },
    }


def _frozen_source(method: str) -> dict[str, object]:
    spec = contingency.FROZEN_SOURCE_SPECS[method]
    prefix = int(spec["prefix_budget"])
    return {
        "identities": {
            name: _identity(f"/evidence/{method}/{name}.json", spec[f"{name}_sha256"])
            for name in (
                "sampling_report",
                "sampling_manifest",
                "sampling_progress",
                "class_fidelity_report",
                "generation_metrics_report",
            )
        },
        "sample_tree": {
            "path": f"/frozen/{method}/prefix_{prefix}",
            "count": 10_000,
            "total_bytes": 1_000_000,
            "sample_set_sha256": spec["sample_set_sha256"],
            "filename_contract": "zero_based_six_digit_png",
            "image_shape": [3, 256, 256],
            "physical_validation_performed": True,
        },
        "sampling": _sampling(prefix),
        "sampling_git": {
            "revision": contingency.SOURCE_SAMPLING_REVISION,
            "branch": contingency.SOURCE_SAMPLING_BRANCH,
            "tracked_dirty": False,
        },
        "checkpoint_step": 100_000,
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "prefix_budget": prefix,
        "weights": "ema",
        "existing_class_fidelity": {
            "sample_count": 10_000,
            "num_classes": 1_000,
            "requested_class_count": 1_000,
            "requested_count_min": 10,
            "requested_count_max": 10,
            "top1_accuracy": 0.002,
            "top5_accuracy": 0.01,
            "predicted_class_fraction": 0.7,
            "normalized_predicted_class_entropy": 0.75,
        },
        "existing_generation_metrics": {
            "frechet_inception_distance": 120.0,
            "precision": 0.7,
            "recall": 0.01,
        },
    }


def _preparation() -> tuple[dict, dict]:
    decision_id = _identity(
        "/evidence/decision.json", contingency.CAUSAL_SOURCE_SHA256["decision"]
    )
    validation_id = _identity(
        "/evidence/decision.validation.json",
        contingency.CAUSAL_SOURCE_SHA256["validation"],
    )
    decision = {
        "status": "completed",
        "scientific_status": "hold",
        "terminal_status": "hold",
        "selected_candidate": None,
        "decision": contingency.CAUSAL_DECISION,
        "generation_advantage_proven": False,
    }
    validation = {
        "status": "pass",
        "scientific_status": "hold",
        "decision": decision_id,
    }
    classifier_id = _identity(
        "/checkpoints/resnet50-11ad3fa6.pth",
        contingency.CLASSIFIER["weights_sha256"],
        size=int(contingency.CLASSIFIER["weights_bytes"]),
    )
    report = contingency.build_preparation(
        causal_discriminator=decision,
        causal_discriminator_identity=decision_id,
        causal_validation=validation,
        causal_validation_identity=validation_id,
        classifier_identity=classifier_id,
        sources={method: _frozen_source(method) for method in contingency.METHODS},
        preparation_git=_git(),
        output_root="/outputs/class_support_contingency_v1",
    )
    return report, _identity(
        "/evidence/preparation.json", contingency.canonical_sha256(report), size=1000
    )


def _stage_approval(preparation_id: dict, evaluator_git: dict) -> dict:
    return {
        "schema_version": contingency.STAGE_APPROVAL_SCHEMA,
        "role": contingency.STAGE_APPROVAL_ROLE,
        "status": "approved",
        "scope": contingency.STAGE_APPROVAL_SCOPE,
        "decision": "authorize_frozen_sample_class_support_contingency_evaluation_only",
        "user_authorized": True,
        "preparation": preparation_id,
        "evaluator_git": evaluator_git,
        "training_launch_allowed": False,
        "sampling_launch_allowed": False,
        "retraining_allowed": False,
        "resampling_allowed": False,
        "confirmation_preparation_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_allowed": False,
        "export_allowed": False,
        "release_allowed": False,
        "paper_integration_allowed": False,
        "process_signals_allowed": False,
    }


def _authorization(preparation: dict, preparation_id: dict) -> tuple[dict, dict, dict]:
    evaluator_git = _git("b")
    approval = _stage_approval(preparation_id, evaluator_git)
    approval_id = _identity(
        "/evidence/stage_approval.json", contingency.canonical_sha256(approval)
    )
    authorization = contingency.build_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_id,
        stage_approval=approval,
        stage_approval_identity=approval_id,
        evaluator_git=evaluator_git,
        authorization_git=_git("c"),
    )
    authorization_id = _identity(
        "/evidence/execution_authorization.json",
        contingency.canonical_sha256(authorization),
    )
    return authorization, authorization_id, evaluator_git


def _runtime() -> dict[str, object]:
    return {
        "elapsed_seconds": 1.0,
        "device": "cpu",
        "batch_size": 8,
        "num_workers": 0,
        "torch_version": "test",
        "torchvision_version": "test",
        "scipy_version": "test",
    }


def _rows(
    *,
    mode: str,
    method: str,
    num_classes: int = 5,
    occurrences: int = 4,
) -> list[dict[str, object]]:
    rows = []
    for occurrence in range(occurrences):
        for requested in range(num_classes):
            index = occurrence * num_classes + requested
            if mode == "offset":
                predicted = (requested + 1) % num_classes
            elif mode == "constant":
                predicted = 0
            elif mode == "grouped":
                predicted = requested // 2
            else:
                raise ValueError(mode)
            top5 = [predicted] + [
                value for value in range(num_classes) if value != predicted
            ][:4]
            top5_probabilities = [0.5, 0.2, 0.15, 0.1, 0.05]
            digest_source = (
                f"shared-{index}"
                if index == 0
                else f"{method}-duplicate-1-2"
                if index in (1, 2)
                else f"{method}-{index}"
            )
            rows.append(
                {
                    "sample_index": index,
                    "filename": f"{index:06d}.png",
                    "requested_class": requested,
                    "requested_probability": (
                        top5_probabilities[top5.index(requested)]
                        if requested in top5
                        else 0.01
                    ),
                    "predicted_top1_class": predicted,
                    "predicted_top1_probability": 0.5,
                    "predicted_top5_classes": top5,
                    "predicted_top5_probabilities": top5_probabilities,
                    "image_sha256": hashlib.sha256(digest_source.encode()).hexdigest(),
                }
            )
    return rows


def test_physical_sample_tree_identity_binds_names_shape_and_bytes(
    tmp_path: Path,
) -> None:
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        Image.new("RGB", (8, 8), color).save(tmp_path / f"{index:06d}.png")
    identity = contingency.physical_sample_tree_identity(
        tmp_path,
        expected_count=3,
        expected_shape=[3, 8, 8],
        expected_image_sha256=[
            hashlib.sha256((tmp_path / f"{index:06d}.png").read_bytes()).hexdigest()
            for index in range(3)
        ],
    )
    assert identity["count"] == 3
    assert len(identity["sample_set_sha256"]) == 64

    (tmp_path / "000002.png").rename(tmp_path / "000003.png")
    with pytest.raises(ValueError, match="zero-based"):
        contingency.physical_sample_tree_identity(
            tmp_path,
            expected_count=3,
            expected_shape=[3, 8, 8],
        )


def test_physical_sample_tree_identity_replays_per_image_hashes(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8), (1, 2, 3)).save(tmp_path / "000000.png")
    with pytest.raises(ValueError, match="image SHA256 differs"):
        contingency.physical_sample_tree_identity(
            tmp_path,
            expected_count=1,
            expected_shape=[3, 8, 8],
            expected_image_sha256=["0" * 64],
        )


def test_exact_binomial_survival_matches_direct_small_sum() -> None:
    observed = contingency.exact_binomial_survival(3, 8, 0.2)
    expected = sum(
        math.comb(8, value) * 0.2**value * 0.8 ** (8 - value) for value in range(3, 9)
    )
    assert observed == pytest.approx(expected, abs=1e-14)


def test_exact_binomial_survival_handles_degenerate_probability() -> None:
    assert contingency.exact_binomial_survival(8, 8, 1.0) == 1.0
    assert contingency.exact_binomial_survival(1, 8, 0.0) == 0.0
    with pytest.raises(ValueError, match="must be an integer"):
        contingency.exact_binomial_survival(1.0, 8, 0.2)


def test_adjusted_mutual_information_is_one_for_identical_partition() -> None:
    report = contingency.adjusted_mutual_information([[5, 0], [0, 5]])
    assert report["adjusted_mutual_information"] == pytest.approx(1.0)
    assert report["normalization"] == "arithmetic_mean_entropy"
    with pytest.raises(ValueError, match="must be an integer"):
        contingency.adjusted_mutual_information([[5.0, 0], [0, 5]])


def test_analysis_detects_common_cyclic_label_offset() -> None:
    analysis = contingency.build_analysis(
        {
            "cofitok": _rows(mode="offset", method="cofitok"),
            "dense_identity": _rows(mode="offset", method="dense"),
        },
        existing_recall={"cofitok": 0.01, "dense_identity": 0.01},
        num_classes=5,
        null_seed=7,
        null_replicates=32,
    )
    assert (
        analysis["interpretation"]["primary_interpretation"]
        == "permuted_or_offset_conditioning"
    )
    assert (
        analysis["methods"]["cofitok"]["best_cyclic_offset_alignment"]["best_offset"]
        == 1
    )
    assert analysis["methods"]["cofitok"]["best_one_to_one_mapping"][
        "cross_validated_accuracy"
    ] == pytest.approx(1.0)
    assert (
        analysis["methods"]["cofitok"]["best_one_to_one_mapping"][
            "cross_validated_direct_count"
        ]
        == 0
    )


def test_analysis_detects_shared_unconditional_support_and_exact_duplicates() -> None:
    first = contingency.build_analysis(
        {
            "cofitok": _rows(mode="constant", method="cofitok"),
            "dense_identity": _rows(mode="constant", method="dense"),
        },
        existing_recall={"cofitok": 0.008, "dense_identity": 0.01},
        num_classes=5,
        null_seed=11,
        null_replicates=24,
    )
    second = contingency.build_analysis(
        {
            "cofitok": _rows(mode="constant", method="cofitok"),
            "dense_identity": _rows(mode="constant", method="dense"),
        },
        existing_recall={"cofitok": 0.008, "dense_identity": 0.01},
        num_classes=5,
        null_seed=11,
        null_replicates=24,
    )
    assert first == second
    assert (
        first["interpretation"]["primary_interpretation"]
        == "shared_unconditional_support_collapse"
    )
    assert (
        first["cross_method"]["exact_duplicates"]["same_index_exact_duplicate_count"]
        == 1
    )
    assert (
        first["methods"]["cofitok"]["exact_duplicates"]["duplicate_excess_image_count"]
        == 1
    )


def test_prediction_rows_reject_inconsistent_top1_probability() -> None:
    rows = _rows(mode="offset", method="cofitok")
    rows[0]["predicted_top1_probability"] = 0.4
    with pytest.raises(ValueError, match="top1 probability differs"):
        contingency.validate_prediction_rows(
            rows,
            sample_count=len(rows),
            num_classes=5,
        )


def test_prediction_rows_reject_coerced_labels_and_probability_mismatch() -> None:
    rows = _rows(mode="offset", method="cofitok")
    rows[0]["sample_index"] = 0.0
    with pytest.raises(ValueError, match="must be an integer"):
        contingency.validate_prediction_rows(
            rows,
            sample_count=len(rows),
            num_classes=5,
        )

    rows = _rows(mode="constant", method="cofitok", num_classes=6)
    rows[5]["requested_probability"] = 0.2
    with pytest.raises(ValueError, match="exceeds top5 boundary"):
        contingency.validate_prediction_rows(
            rows,
            sample_count=len(rows),
            num_classes=6,
        )

    rows = _rows(mode="offset", method="cofitok")
    rows[0]["requested_probability"] = 0.01
    with pytest.raises(ValueError, match="requested probability differs"):
        contingency.validate_prediction_rows(
            rows,
            sample_count=len(rows),
            num_classes=5,
        )


def test_preparation_is_non_authorizing_and_requires_locked_sources() -> None:
    report, _ = _preparation()
    assert contingency.validate_preparation(report) == report
    assert report["authorization_boundary"]["classifier_inference_allowed"] is False
    assert (
        report["execution_contract"]["separate_exact_execution_authorization_required"]
        is True
    )

    tampered = copy.deepcopy(report)
    tampered["source_evidence"]["frozen_methods"]["cofitok"]["sample_tree"][
        "sample_set_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="sample tree contract differs"):
        contingency.validate_preparation(tampered)

    tampered = copy.deepcopy(report)
    tampered["unexpected"] = False
    with pytest.raises(ValueError, match="preparation fields differ"):
        contingency.validate_preparation(tampered)

    sources = {method: _frozen_source(method) for method in contingency.METHODS}
    sources["cofitok"]["sample_tree"]["count"] = 10_000.0
    with pytest.raises(ValueError, match="must be an integer"):
        contingency.build_preparation(
            causal_discriminator={
                "status": "completed",
                "scientific_status": "hold",
                "terminal_status": "hold",
                "selected_candidate": None,
                "decision": contingency.CAUSAL_DECISION,
                "generation_advantage_proven": False,
            },
            causal_discriminator_identity=_identity(
                "/evidence/decision.json",
                contingency.CAUSAL_SOURCE_SHA256["decision"],
            ),
            causal_validation={
                "status": "pass",
                "scientific_status": "hold",
                "decision": _identity(
                    "/evidence/decision.json",
                    contingency.CAUSAL_SOURCE_SHA256["decision"],
                ),
            },
            causal_validation_identity=_identity(
                "/evidence/decision.validation.json",
                contingency.CAUSAL_SOURCE_SHA256["validation"],
            ),
            classifier_identity=_identity(
                "/checkpoints/resnet50-11ad3fa6.pth",
                contingency.CLASSIFIER["weights_sha256"],
                size=contingency.CLASSIFIER["weights_bytes"],
            ),
            sources=sources,
            preparation_git=_git(),
            output_root="/outputs/class_support_contingency_v1",
        )


def test_execution_authorization_requires_separate_user_stage_approval() -> None:
    preparation, preparation_id = _preparation()
    evaluator_git = _git("b")
    approval = _stage_approval(preparation_id, evaluator_git)
    approval_id = _identity(
        "/evidence/stage_approval.json", contingency.canonical_sha256(approval)
    )
    authorization = contingency.build_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_id,
        stage_approval=approval,
        stage_approval_identity=approval_id,
        evaluator_git=evaluator_git,
        authorization_git=_git("c"),
    )
    assert authorization["authorization_boundary"]["classifier_inference_allowed"]
    assert authorization["authorization_boundary"]["training_launch_allowed"] is False
    assert authorization["authorization_boundary"]["sampling_launch_allowed"] is False

    forged = copy.deepcopy(approval)
    forged["user_authorized"] = False
    with pytest.raises(ValueError, match="stage approval contract differs"):
        contingency.build_execution_authorization(
            preparation=preparation,
            preparation_identity=preparation_id,
            stage_approval=forged,
            stage_approval_identity=approval_id,
            evaluator_git=evaluator_git,
            authorization_git=_git("c"),
        )


def test_result_builder_rejects_nonproduction_prediction_count() -> None:
    preparation, preparation_id = _preparation()
    authorization, authorization_id, evaluator_git = _authorization(
        preparation, preparation_id
    )
    prediction_identities = {
        method: _identity(
            f"/outputs/class_support_contingency_v1/{method}_predictions.jsonl",
            character * 64,
        )
        for method, character in zip(contingency.METHODS, ("d", "e"))
    }
    with pytest.raises(ValueError, match="exact frozen sample count"):
        contingency.build_result(
            preparation=preparation,
            preparation_identity=preparation_id,
            execution_authorization=authorization,
            execution_authorization_identity=authorization_id,
            evaluator_git=evaluator_git,
            prediction_identities=prediction_identities,
            rows_by_method={
                "cofitok": _rows(mode="offset", method="cofitok"),
                "dense_identity": _rows(mode="offset", method="dense"),
            },
            existing_recall={"cofitok": 0.01, "dense_identity": 0.01},
            runtime=_runtime(),
        )


def test_result_builder_stamps_elapsed_after_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation, preparation_id = _preparation()
    authorization, authorization_id, evaluator_git = _authorization(
        preparation, preparation_id
    )
    prediction_identities = {
        method: _identity(
            f"/outputs/class_support_contingency_v1/{method}_predictions.jsonl",
            character * 64,
        )
        for method, character in zip(contingency.METHODS, ("d", "e"))
    }
    state = {"analysis_complete": False}

    def fake_analysis(*args: object, **kwargs: object) -> dict[str, object]:
        state["analysis_complete"] = True
        return {"analysis": "sentinel"}

    def fake_clock() -> float:
        assert state["analysis_complete"]
        return 101.25

    monkeypatch.setattr(contingency, "build_analysis", fake_analysis)
    monkeypatch.setattr(contingency.time, "perf_counter", fake_clock)
    monkeypatch.setattr(
        contingency,
        "validate_result",
        lambda result, **kwargs: result,
    )
    runtime = _runtime()
    runtime.pop("elapsed_seconds")
    result = contingency.build_result(
        preparation=preparation,
        preparation_identity=preparation_id,
        execution_authorization=authorization,
        execution_authorization_identity=authorization_id,
        evaluator_git=evaluator_git,
        prediction_identities=prediction_identities,
        rows_by_method={method: [{}] * 10_000 for method in contingency.METHODS},
        existing_recall={"cofitok": 0.01, "dense_identity": 0.01},
        runtime=runtime,
        elapsed_started_at=100.0,
    )
    assert result["runtime"]["elapsed_seconds"] == pytest.approx(1.25)
