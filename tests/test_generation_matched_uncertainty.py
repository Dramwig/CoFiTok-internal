from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cofitok.environment import runtime_environment_sha256
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA
from scripts import audit_generation_matched_uncertainty as audit
from scripts import build_generation_matched_uncertainty_summary as summary


def _sampling(*, prefix_budget: int, guidance_scale: float = 1.5) -> dict:
    return {
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "inference_api": {
            "name": "cofitok.generation.GenerationSession",
            "version": 1,
        },
        "sampler": "ddim",
        "start_index": 10_000,
        "num_samples": 80,
        "sample_steps": 100,
        "num_train_timesteps": 1000,
        "actual_timesteps": [999, 0],
        "guidance_scale": guidance_scale,
        "guidance_rescale": 1.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "precision": "bf16",
        "image_shape": [3, 256, 256],
        "prefix_budgets": [prefix_budget],
        "batch_size": 32,
        "num_classes": 1000,
        "class_schedule": "balanced_modulo",
        "seed": 0,
        "random_stream": {
            "batch_size_invariant": True,
            "prefix_budgets_share_stream": True,
            "resume_index_invariant": True,
            "scope": "per_global_sample_index",
            "seed_formula": "(seed + global_index) mod 2^63",
        },
    }


def _provenance(*, prefix_budget: int) -> dict:
    return {
        "sampling": _sampling(prefix_budget=prefix_budget),
        "sample_set_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_step": 50_000,
    }


def _features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    real = rng.normal(size=(160, 8)).astype(np.float32)
    cofitok = rng.normal(size=(80, 8)).astype(np.float32)
    dense = (rng.normal(size=(80, 8)) + 2.5).astype(np.float32)
    return cofitok, dense, real


def _build_inputs() -> dict:
    cofitok, dense, real = _features()
    runtime = {"schema_version": 1, "device": {"type": "cpu"}}
    return {
        "cofitok_features": cofitok,
        "dense_features": dense,
        "real_features": real,
        "cofitok_metrics": {"fid": 10.0},
        "dense_metrics": {"fid": 12.0},
        "matched_sampling": {
            "signature": audit.matched_sampling_signature(_sampling(prefix_budget=8)),
            "start_index": 10_000,
            "end_index_exclusive": 10_080,
            "sample_count": 80,
            "cofitok_prefix_budgets": [8],
            "dense_prefix_budgets": [1],
        },
        "sources": {
            "real_set": {
                "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
                "root": "/real",
                "image_count": 160,
                "sha256": "c" * 64,
            },
            "sample_sets": {
                "cofitok": {
                    "sha256": "1" * 64,
                    "checkpoint_sha256": "2" * 64,
                    "checkpoint_step": 50_000,
                },
                "dense_identity": {
                    "sha256": "3" * 64,
                    "checkpoint_sha256": "4" * 64,
                    "checkpoint_step": 50_000,
                },
            },
        },
        "parameters": {
            "block_size": 10,
            "real_fold_count": 2,
            "bootstrap_repetitions": 1000,
            "seed": 7,
        },
        "implementation": {
            "torch_fidelity_version": "0.4.0",
            "feature_extractor": audit.FEATURE_EXTRACTOR,
            "feature_layer": audit.FEATURE_LAYER,
            "kid_kernel": audit.KID_KERNEL,
        },
        "evaluator_git": {
            "revision": "d" * 40,
            "branch": "analysis/generation-matched-uncertainty-v1",
            "tracked_dirty": False,
        },
        "runtime_environment": runtime,
    }


def _execution_manifest_fixture(tmp_path: Path) -> tuple[SimpleNamespace, dict]:
    directories = {
        "real_dir": tmp_path / "real",
        "cofitok_generated_dir": tmp_path / "cofitok",
        "dense_generated_dir": tmp_path / "dense",
        "cache_root": tmp_path / "cache",
    }
    for path in directories.values():
        path.mkdir()
    report_keys = (
        "cofitok_sampling_report",
        "dense_sampling_report",
        "cofitok_metrics_report",
        "dense_metrics_report",
    )
    report_paths = {}
    for key in report_keys:
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({"source": key}), encoding="utf-8")
        report_paths[key] = path
    signature = {"protocol": "matched", "start_index": 10_000}
    real_set = {
        "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
        "root": directories["real_dir"].resolve().as_posix(),
        "image_count": 160,
        "sha256": "c" * 64,
    }
    sample_sets = {
        "cofitok": {
            "sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "checkpoint_step": 50_000,
        },
        "dense_identity": {
            "sha256": "3" * 64,
            "checkpoint_sha256": "4" * 64,
            "checkpoint_step": 50_000,
        },
    }
    arguments = {
        **{key: path.resolve().as_posix() for key, path in directories.items()},
        **{key: path.resolve().as_posix() for key, path in report_paths.items()},
        "output": (tmp_path / "report.json").resolve().as_posix(),
        "real_cache_name": "real-cache",
        "batch_size": 16,
        "min_samples": 80,
        "block_size": 10,
        "real_fold_count": 2,
        "bootstrap_repetitions": 1000,
        "seed": 7,
        "cpu": True,
    }
    manifest = {
        "schema_version": audit.EXECUTION_MANIFEST_SCHEMA_VERSION,
        "role": audit.EXECUTION_MANIFEST_ROLE,
        "stream_id": "test_00010000_00010080",
        "claim_boundary": audit.CLAIM_BOUNDARY,
        "arguments": arguments,
        "source_files": {
            key: audit._source(path) for key, path in report_paths.items()
        },
        "expected": {
            "matched_sampling": {
                "start_index": 10_000,
                "end_index_exclusive": 10_080,
                "sample_count": 80,
                "cofitok_prefix_budgets": [8],
                "dense_prefix_budgets": [1],
                "signature_sha256": audit._canonical_sha256(signature),
            },
            "real_set": real_set,
            "sample_sets": sample_sets,
            "fid_point_estimates": {
                "cofitok": 10.0,
                "dense_identity": 12.0,
            },
            "metrics_runtime_environment_sha256": "e" * 64,
        },
    }
    manifest_path = tmp_path / "execution_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(
        **{key: Path(value) for key, value in arguments.items() if key in directories},
        **{key: path for key, path in report_paths.items()},
        output=Path(arguments["output"]),
        real_cache_name=arguments["real_cache_name"],
        batch_size=arguments["batch_size"],
        min_samples=arguments["min_samples"],
        block_size=arguments["block_size"],
        real_fold_count=arguments["real_fold_count"],
        bootstrap_repetitions=arguments["bootstrap_repetitions"],
        seed=arguments["seed"],
        cpu=arguments["cpu"],
        execution_manifest=manifest_path,
    )
    observations = {
        "matched_sampling": {
            "signature": signature,
            "start_index": 10_000,
            "end_index_exclusive": 10_080,
            "sample_count": 80,
            "cofitok_prefix_budgets": [8],
            "dense_prefix_budgets": [1],
        },
        "real_set": real_set,
        "sample_provenance": {
            method: {
                "sample_set_sha256": row["sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "checkpoint_step": row["checkpoint_step"],
            }
            for method, row in sample_sets.items()
        },
        "metrics": {
            "cofitok": {
                "fid": 10.0,
                "runtime_environment_sha256": "e" * 64,
            },
            "dense_identity": {
                "fid": 12.0,
                "runtime_environment_sha256": "e" * 64,
            },
        },
    }
    return args, observations


def test_execution_manifest_binds_sources_arguments_and_observations(tmp_path) -> None:
    args, observations = _execution_manifest_fixture(tmp_path)

    binding = audit.validate_execution_manifest(args)
    verification = audit.validate_execution_manifest_observations(
        binding,
        **observations,
    )

    assert binding is not None
    assert verification is not None
    assert verification["stream_id"] == "test_00010000_00010080"
    assert verification["status"] == "verified"


def test_execution_manifest_can_hydrate_manifest_only_cli_arguments(tmp_path) -> None:
    args, _ = _execution_manifest_fixture(tmp_path)
    payload = json.loads(args.execution_manifest.read_text(encoding="utf-8"))
    manifest_only = SimpleNamespace(
        **{key: None for key in payload["arguments"]},
        execution_manifest=args.execution_manifest,
        resume=False,
        require_advantage=False,
    )

    hydrated = audit.hydrate_execution_arguments(manifest_only)

    assert hydrated.real_dir == args.real_dir
    assert hydrated.output == args.output
    assert hydrated.batch_size == 16
    assert hydrated.cpu is True
    assert audit.validate_execution_manifest(hydrated) is not None


def test_execution_manifest_rejects_argument_or_source_drift(tmp_path) -> None:
    args, _ = _execution_manifest_fixture(tmp_path)
    args.batch_size = 32
    with pytest.raises(ValueError, match="argument mismatch: batch_size"):
        audit.validate_execution_manifest(args)

    args.batch_size = 16
    args.cofitok_sampling_report.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source identity mismatch"):
        audit.validate_execution_manifest(args)


def test_execution_manifest_rejects_observed_identity_drift(tmp_path) -> None:
    args, observations = _execution_manifest_fixture(tmp_path)
    binding = audit.validate_execution_manifest(args)
    observations["metrics"]["cofitok"]["fid"] = 10.5

    with pytest.raises(ValueError, match="observation mismatch"):
        audit.validate_execution_manifest_observations(binding, **observations)


def test_execution_manifest_rechecks_sources_before_feature_extraction(
    tmp_path,
) -> None:
    args, observations = _execution_manifest_fixture(tmp_path)
    binding = audit.validate_execution_manifest(args)
    args.dense_metrics_report.write_text("changed after preflight", encoding="utf-8")

    with pytest.raises(ValueError, match="source identity mismatch"):
        audit.validate_execution_manifest_observations(binding, **observations)


def test_validate_matched_sampling_allows_only_prefix_budget_difference() -> None:
    result = audit.validate_matched_sampling(
        _provenance(prefix_budget=8),
        _provenance(prefix_budget=1),
    )

    assert result["sample_count"] == 80
    assert result["start_index"] == 10_000
    assert result["end_index_exclusive"] == 10_080
    assert result["cofitok_prefix_budgets"] == [8]
    assert result["dense_prefix_budgets"] == [1]


def test_validate_matched_sampling_rejects_protocol_drift() -> None:
    dense = _provenance(prefix_budget=1)
    dense["sampling"]["guidance_scale"] = 1.25

    with pytest.raises(ValueError, match="not matched"):
        audit.validate_matched_sampling(_provenance(prefix_budget=8), dense)


def test_polynomial_mmd2_matches_explicit_formula() -> None:
    candidate = np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    reference = np.asarray([[2.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    gamma = 0.5
    kcc = (gamma * candidate.dot(candidate.T) + 1.0) ** 3
    krr = (gamma * reference.dot(reference.T) + 1.0) ** 3
    kcr = (gamma * candidate.dot(reference.T) + 1.0) ** 3
    expected = (
        (kcc.sum() - np.trace(kcc)) / 6
        + (krr.sum() - np.trace(krr)) / 6
        - 2.0 * kcr.mean()
    )

    assert audit.polynomial_mmd2_unbiased(candidate, reference) == pytest.approx(
        expected
    )


def test_polynomial_mmd2_matches_torch_fidelity_reference() -> None:
    metric_kid = pytest.importorskip("torch_fidelity.metric_kid")
    rng = np.random.default_rng(19)
    candidate = rng.normal(size=(7, 5)).astype(np.float32)
    reference = rng.normal(size=(7, 5)).astype(np.float32)

    expected = metric_kid.kernel_mmd(
        candidate,
        reference,
        kid_kernel="poly",
        kid_kernel_poly_degree=3,
        kid_kernel_poly_gamma=None,
        kid_kernel_poly_coef0=1.0,
    )

    assert audit.polynomial_mmd2_unbiased(candidate, reference) == pytest.approx(
        expected,
        abs=5e-7,
    )


def test_shared_reference_contrast_cancels_real_only_kid_term() -> None:
    rng = np.random.default_rng(23)
    cofitok = rng.normal(size=(9, 6)).astype(np.float32)
    dense = rng.normal(size=(9, 6)).astype(np.float32)
    reference = rng.normal(size=(9, 6)).astype(np.float32)

    contrast = audit.polynomial_mmd2_difference_shared_reference(
        cofitok,
        dense,
        reference,
    )
    explicit = audit.polynomial_mmd2_unbiased(
        cofitok,
        reference,
    ) - audit.polynomial_mmd2_unbiased(dense, reference)

    assert contrast["cofitok_minus_dense"] == pytest.approx(explicit, abs=1e-12)


def test_paired_block_kid_detects_large_matched_advantage() -> None:
    cofitok, dense, real = _features()

    result = audit.paired_block_kid(
        cofitok_features=cofitok,
        dense_features=dense,
        real_features=real,
        block_size=10,
        real_fold_count=2,
        bootstrap_repetitions=1000,
        seed=7,
    )

    assert result["block_count"] == 8
    assert result["real_fold_count"] == 2
    assert result["real_samples_used"] == 160
    assert result["cofitok_better_blocks"] == 8
    assert result["one_sided_exact_sign_test_p"] == pytest.approx(1 / 256)
    assert result["paired_block_bootstrap"]["ci_high"] < 0.0
    assert result["uncertainty_supports_cofitok_advantage"] is True


def test_exact_sign_test_all_ties_is_conservative() -> None:
    assert audit.exact_one_sided_sign_test(wins=0, losses=0) == 1.0


def test_paired_block_kid_does_not_pass_identical_methods() -> None:
    cofitok, _, real = _features()

    result = audit.paired_block_kid(
        cofitok_features=cofitok,
        dense_features=cofitok.copy(),
        real_features=real,
        block_size=10,
        real_fold_count=2,
        bootstrap_repetitions=1000,
        seed=7,
    )

    assert result["cofitok_better_blocks"] == 0
    assert result["dense_better_blocks"] == 0
    assert result["tied_blocks"] == 8
    assert result["one_sided_exact_sign_test_p"] == 1.0
    assert result["uncertainty_supports_cofitok_advantage"] is False


def test_build_report_supports_only_scoped_relative_advantage() -> None:
    inputs = _build_inputs()

    report = audit.build_report(**inputs)

    assert report["status"] == "pass"
    assert report["decision"] == "matched_relative_generation_advantage_supported"
    assert report["advantage_supported"] is True
    assert report["claim_boundary"] == audit.CLAIM_BOUNDARY
    assert report["claim_boundary"]["full_training_launch_allowed"] is False
    assert report["claim_boundary"]["full_300k_launch_allowed"] is False
    assert (
        report["claim_boundary"]["broad_generation_superiority_claim_allowed"] is False
    )
    assert report["runtime_environment_sha256"] == runtime_environment_sha256(
        inputs["runtime_environment"]
    )
    assert len(report["feature_matrices"]["cofitok"]["sha256"]) == 64


def _write_audit(
    tmp_path,
    *,
    name: str,
    start_index: int,
    guidance_rescale: float = 1.0,
    cofitok_fid: float = 10.0,
) -> tuple[dict, object]:
    inputs = _build_inputs()
    signature = inputs["matched_sampling"]["signature"]
    signature["start_index"] = start_index
    signature["guidance_rescale"] = guidance_rescale
    inputs["matched_sampling"]["start_index"] = start_index
    inputs["matched_sampling"]["end_index_exclusive"] = start_index + 80
    inputs["cofitok_metrics"] = {"fid": cofitok_fid}
    inputs["sources"]["sample_sets"]["cofitok"]["sha256"] = f"{start_index:064x}"[-64:]
    inputs["sources"]["sample_sets"]["dense_identity"]["sha256"] = (
        f"{start_index + 1:064x}"[-64:]
    )
    report = audit.build_report(**inputs)
    manifest_path = tmp_path / f"{name}_execution_manifest.json"
    manifest_path.write_text(
        json.dumps({"name": name, "start_index": start_index}),
        encoding="utf-8",
    )
    report["sources"]["execution_manifest"] = {
        "source": audit._source(manifest_path),
        "stream_id": f"{name}_{start_index:08d}",
        "status": "verified",
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return report, path


def test_repeated_summary_accepts_disjoint_exact_protocol_streams(tmp_path) -> None:
    first = _write_audit(tmp_path, name="first", start_index=0)
    second = _write_audit(tmp_path, name="second", start_index=80)

    report = summary.build_summary(
        [first, second],
        builder_git={
            "revision": "f" * 40,
            "branch": "analysis/generation-matched-uncertainty-v1",
            "tracked_dirty": False,
        },
    )

    assert report["status"] == "pass"
    assert report["decision"] == "repeated_exact_protocol_relative_advantage_supported"
    assert report["checks"]["source_bound_execution_manifests"] is True
    assert report["checks"]["disjoint_global_index_windows"] is True
    assert report["checks"]["exact_protocol_replication"] is True
    assert report["claim_boundary"]["full_training_launch_allowed"] is False


def test_repeated_summary_labels_cross_protocol_support(tmp_path) -> None:
    first = _write_audit(
        tmp_path,
        name="formal",
        start_index=0,
        guidance_rescale=0.0,
    )
    second = _write_audit(
        tmp_path,
        name="confirmation",
        start_index=80,
        guidance_rescale=1.0,
    )

    report = summary.build_summary([first, second], builder_git={})

    assert report["status"] == "pass"
    assert report["decision"] == "repeated_cross_protocol_relative_advantage_supported"
    assert report["checks"]["exact_protocol_replication"] is False


def test_repeated_summary_holds_for_overlapping_windows(tmp_path) -> None:
    first = _write_audit(tmp_path, name="first", start_index=0)
    second = _write_audit(tmp_path, name="second", start_index=40)

    report = summary.build_summary([first, second], builder_git={})

    assert report["status"] == "hold"
    assert report["checks"]["disjoint_global_index_windows"] is False
    assert report["repeated_advantage_supported"] is False


def test_repeated_summary_rejects_unbound_stream_report(tmp_path) -> None:
    first_report, first_path = _write_audit(
        tmp_path,
        name="first",
        start_index=0,
    )
    second = _write_audit(tmp_path, name="second", start_index=80)
    del first_report["sources"]["execution_manifest"]
    first_path.write_text(json.dumps(first_report), encoding="utf-8")

    with pytest.raises(TypeError, match="execution manifest is missing"):
        summary.build_summary([(first_report, first_path), second], builder_git={})


def test_build_report_holds_when_fid_direction_disagrees() -> None:
    inputs = _build_inputs()
    inputs["cofitok_metrics"] = {"fid": 13.0}

    report = audit.build_report(**inputs)

    assert report["paired_block_kid"]["uncertainty_supports_cofitok_advantage"] is True
    assert (
        report["fid_point_estimates"]["direction_supports_cofitok_advantage"] is False
    )
    assert report["advantage_supported"] is False
    assert report["status"] == "hold"


def test_validate_metrics_report_binds_physical_paths_and_provenance(tmp_path) -> None:
    real = tmp_path / "real"
    generated = tmp_path / "generated"
    real.mkdir()
    generated.mkdir()
    provenance = _provenance(prefix_budget=8)
    report_path = tmp_path / "metrics.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "role": "generation_directory_metrics_report",
                "protocol": "torch_fidelity_directory_metrics",
                "status": "completed",
                "paths": {
                    "real_dir": real.resolve().as_posix(),
                    "generated_dir": generated.resolve().as_posix(),
                },
                "counts": {
                    "real_image_count": 160,
                    "generated_image_count": 80,
                },
                "real_set": {
                    "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
                    "image_count": 160,
                    "sha256": "c" * 64,
                },
                "parameters": {
                    "samples_find_deep": True,
                    "samples_shuffle": False,
                },
                "sample_provenance": provenance,
                "metrics": {"frechet_inception_distance": 10.0},
                "implementation": {"torch_fidelity": "0.4.0"},
                "runtime_environment_sha256": "e" * 64,
            }
        ),
        encoding="utf-8",
    )

    result = audit.validate_metrics_report(
        report_path,
        generated_dir=generated,
        real_dir=real,
        sample_provenance=provenance,
        generated_count=80,
        real_count=160,
        real_set_sha256="c" * 64,
    )

    assert result["fid"] == 10.0
    assert result["sample_set_sha256"] == "a" * 64
    assert result["checkpoint_sha256"] == "b" * 64

    changed = json.loads(report_path.read_text(encoding="utf-8"))
    changed["sample_provenance"] = deepcopy(provenance)
    changed["sample_provenance"]["checkpoint_sha256"] = "f" * 64
    report_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="contract mismatch"):
        audit.validate_metrics_report(
            report_path,
            generated_dir=generated,
            real_dir=real,
            sample_provenance=provenance,
            generated_count=80,
            real_count=160,
            real_set_sha256="c" * 64,
        )


def test_extract_feature_triplet_uses_content_bound_cache_names(
    tmp_path,
    monkeypatch,
) -> None:
    real = tmp_path / "real"
    cofitok = tmp_path / "cofitok"
    dense = tmp_path / "dense"
    cache = tmp_path / "cache"
    for path in (real, cofitok, dense, cache):
        path.mkdir()
    calls = []

    class FakeExtractor:
        pass

    def create_feature_extractor(name, layers, **kwargs):
        calls.append(("create", name, layers, kwargs))
        return FakeExtractor()

    feature_rows = {
        "real-cache": torch.ones(4, 3),
        "cofitok-cache": torch.ones(2, 3) * 2,
        "dense-cache": torch.ones(2, 3) * 3,
    }

    def extract_featuresdict_from_input_id_cached(input_id, extractor, **kwargs):
        calls.append(("extract", input_id, extractor, kwargs))
        return {audit.FEATURE_LAYER: feature_rows[kwargs["input1_cache_name"]]}

    modules = {
        "torch_fidelity": SimpleNamespace(__version__="0.4.0"),
        "torch_fidelity.utils": SimpleNamespace(
            create_feature_extractor=create_feature_extractor,
            extract_featuresdict_from_input_id_cached=(
                extract_featuresdict_from_input_id_cached
            ),
        ),
    }
    monkeypatch.setattr(
        audit.importlib,
        "import_module",
        lambda name: modules[name],
    )

    cofitok_features, dense_features, real_features, version = (
        audit.extract_feature_triplet(
            real_dir=real,
            cofitok_dir=cofitok,
            dense_dir=dense,
            real_cache_name="real-cache",
            cofitok_cache_name="cofitok-cache",
            dense_cache_name="dense-cache",
            cache_root=cache,
            batch_size=16,
            cuda=False,
        )
    )

    assert version == "0.4.0"
    assert real_features.shape == (4, 3)
    assert np.all(cofitok_features == 2)
    assert np.all(dense_features == 3)
    extract_calls = [call for call in calls if call[0] == "extract"]
    assert [call[3]["input1_cache_name"] for call in extract_calls] == [
        "real-cache",
        "cofitok-cache",
        "dense-cache",
    ]
    assert all(call[3]["samples_shuffle"] is False for call in extract_calls)
    assert all(
        call[3]["cache_root"] == cache.resolve().as_posix() for call in extract_calls
    )
