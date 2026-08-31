"""Fail-closed physical audit for the non-authorizing sampling recovery result."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256


RESULT_SCHEMA = "cofitok_matched_epsilon_stability_sampling_result_v1"
OBSERVATION_SCHEMA = "cofitok_matched_epsilon_stability_case_observation_v1"
REAL_ARTIFACT_SCHEMA = "cofitok_epsilon_stability_real_artifact_reference_v1"
PROJECT_OUTPUT_PREFIX = "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
METHODS = ("cofitok", "dense_identity")
PREFIX_BUDGETS = {"cofitok": 8, "dense_identity": 1}
ARTIFACT_METRIC_KEYS = (
    "channel_saturation_fraction",
    "median_filter_residual_fraction",
    "total_variation",
)
CASE_IDS = frozenset(
    {
        "legacy_terminal_hard_clip",
        "start975_unit_hard_clip",
        "start975_sigma_hard_clip",
        "terminal_dynamic_threshold",
        "terminal_dynamic_threshold_recompute",
        "terminal_hard_clip_recompute",
        "start975_sigma_dynamic_threshold",
        "start975_sigma_dynamic_threshold_recompute",
    }
)
REFERENCE_CASE = "legacy_terminal_hard_clip"
CANDIDATE_CASE_IDS = CASE_IDS - {REFERENCE_CASE}
SOURCE_BINDING_NAMES = frozenset(
    {
        "design",
        "execution_authorization",
        "observation_manifest",
        "preparation",
        "real_artifact_reference",
        "separate_execution_authorization",
        "terminal_route_receipt",
    }
)
SOURCE_REPORT_NAMES = frozenset(
    {"artifact_report", "class_fidelity_report", "metrics_report", "sampling_report"}
)
IDENTITY_KEYS = frozenset({"path", "bytes", "sha256"})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")

AUTHORIZATION_BOUNDARY = {
    "associated_non_formal_evaluation_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "full_training_launch_allowed": False,
    "independent_matched_10000_confirmation_launch_allowed": False,
    "inference_export_allowed": False,
    "matched_1000_sample_sampling_launch_allowed": False,
    "process_signals_allowed": False,
    "promotion_allowed": False,
    "release_allowed": False,
    "training_launch_allowed": False,
}

AUTHORIZED_ACTIONS = {
    "associated_non_formal_evaluation": True,
    "full_300k": False,
    "independent_10000_confirmation": False,
    "inference_export": False,
    "matched_1000_sample_sampling": True,
    "process_signals": False,
    "promotion": False,
    "release": False,
    "training": False,
}

SELECTION_CONTRACT = {
    "candidate_ranking": "descending worst-method relative FID improvement, then case_id",
    "inception_score_role": "descriptive_only_at_1000_samples",
    "reference_case": REFERENCE_CASE,
    "required_methods": list(METHODS),
    "required_per_method_checks": [
        "FID strictly lower than the matched legacy case",
        "class top-1 and top-5 do not regress",
        "all artifact-metric distances to the same real reference do not regress",
    ],
    "single_factor_attribution_requires_parent_comparison": True,
}

CLAIM_BOUNDARY = {
    "cofitok_generation_advantage_claim_allowed": False,
    "independent_matched_10000_confirmation_required": True,
    "min_snr_training_tested": False,
    "one_thousand_sample_screening_only": True,
    "selected_case_is_not_confirmed": False,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _hex(value: Any, *, length: int, name: str) -> str:
    if not isinstance(value, str) or (length == 64 and HEX64.fullmatch(value) is None) or (
        length == 40 and HEX40.fullmatch(value) is None
    ):
        raise ValueError(f"{name} is not a lowercase hexadecimal digest")
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} is below its minimum")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} is above its maximum")
    return number


def _descriptor(value: Any, name: str) -> dict[str, Any]:
    descriptor = _object(value, name)
    if set(descriptor) != IDENTITY_KEYS:
        raise ValueError(f"{name} identity fields differ")
    path = descriptor.get("path")
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ValueError(f"{name} identity path is not absolute")
    byte_count = descriptor.get("bytes")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 1:
        raise ValueError(f"{name} identity byte count is invalid")
    _hex(descriptor.get("sha256"), length=64, name=f"{name} identity SHA256")
    return {"path": path, "bytes": byte_count, "sha256": descriptor["sha256"]}


def _physical_identity(value: Any, *, name: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    descriptor = _descriptor(value, name)
    source = reject_symlink_chain(descriptor["path"], name=name)
    canonical = source.resolve().as_posix()
    if canonical != descriptor["path"]:
        raise ValueError(f"{name} identity path is not canonical")
    if canonical in cache:
        if cache[canonical] != descriptor:
            raise ValueError(f"{name} identity disagrees with another binding")
        return copy.deepcopy(cache[canonical])
    if not source.is_file():
        raise FileNotFoundError(f"{name} source is missing: {source}")
    actual = {
        "path": canonical,
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }
    if actual != descriptor:
        raise ValueError(f"{name} source bytes or SHA256 changed")
    cache[canonical] = actual
    return copy.deepcopy(actual)


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=name)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable") from error
    return _object(value, name)


def _git(value: Any, *, name: str, allow_tree: bool = False) -> dict[str, Any]:
    git = _object(value, name)
    expected_keys = {"branch", "revision", "tracked_dirty"}
    if allow_tree:
        expected_keys.add("tree")
    if set(git) != expected_keys:
        raise ValueError(f"{name} Git fields differ")
    _hex(git.get("revision"), length=40, name=f"{name} revision")
    if allow_tree:
        _hex(git.get("tree"), length=40, name=f"{name} tree")
    if not isinstance(git.get("branch"), str) or not git["branch"]:
        raise ValueError(f"{name} branch is missing")
    if git.get("tracked_dirty") is not False:
        raise ValueError(f"{name} checkout is dirty")
    return git


def _timesteps(start_timestep: int = 999) -> list[int]:
    return sorted({round(index * start_timestep / 99) for index in range(100)}, reverse=True)


def _validate_real_set(value: Any, *, name: str) -> dict[str, Any]:
    real_set = _object(value, name)
    if set(real_set) != {"digest_schema", "image_count", "root", "sha256"}:
        raise ValueError(f"{name} fields differ")
    if real_set.get("digest_schema") != "cofitok_image_tree_sha256_v1":
        raise ValueError(f"{name} digest schema differs")
    if real_set.get("image_count") != 50_000:
        raise ValueError(f"{name} image count differs")
    root = real_set.get("root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError(f"{name} root is invalid")
    if not Path(root).is_dir():
        raise FileNotFoundError(f"{name} root is missing: {root}")
    _hex(real_set.get("sha256"), length=64, name=f"{name} tree SHA256")
    return real_set


def _validate_sampling(value: Any, *, method: str, case_id: str, expected_seed: int | None) -> dict[str, Any]:
    sampling = _object(value, f"{case_id}/{method} sampling")
    expected_keys = {
        "actual_timesteps", "batch_size", "cfg_batch_mode", "class_schedule", "clip_x0",
        "dynamic_threshold_percentile", "eta", "guidance_rescale", "guidance_scale",
        "image_shape", "inference_api", "initial_noise_scale", "num_samples",
        "num_train_timesteps", "precision", "prefix_budgets", "protocol_schema",
        "random_stream", "recompute_epsilon_after_x0_constraint", "requested_start_timestep",
        "sample_set_digest", "sample_steps", "sampler", "scale_initial_noise_by_sigma",
        "seed", "start_index", "start_timestep", "x0_constraint",
    }
    if set(sampling) != expected_keys:
        raise ValueError(f"{case_id}/{method} sampling fields differ")
    start975 = case_id.startswith("start975_")
    recompute_after_constraint = case_id.endswith("_recompute")
    sigma_scaled = case_id in {
        "start975_sigma_hard_clip",
        "start975_sigma_dynamic_threshold",
        "start975_sigma_dynamic_threshold_recompute",
    }
    scalar_expected = {
        "batch_size": 32,
        "cfg_batch_mode": "batched",
        "class_schedule": "balanced_modulo",
        "clip_x0": True,
        "eta": 0.0,
        "guidance_rescale": 0.0,
        "guidance_scale": 1.5,
        "image_shape": [3, 256, 256],
        "initial_noise_scale": "schedule_sigma" if sigma_scaled else "unit",
        "num_samples": 1000,
        "num_train_timesteps": 1000,
        "precision": "bf16",
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "recompute_epsilon_after_x0_constraint": recompute_after_constraint,
        "requested_start_timestep": 975 if start975 else None,
        "sample_steps": 100,
        "sampler": "ddim",
        "scale_initial_noise_by_sigma": sigma_scaled,
        "start_index": 0,
        "start_timestep": 975 if start975 else 999,
        "x0_constraint": "dynamic_threshold"
        if case_id in {
            "terminal_dynamic_threshold",
            "terminal_dynamic_threshold_recompute",
            "start975_sigma_dynamic_threshold",
            "start975_sigma_dynamic_threshold_recompute",
        }
        else "clip",
    }
    for key, expected in scalar_expected.items():
        if sampling.get(key) != expected:
            raise ValueError(f"{case_id}/{method} sampling {key} differs")
    expected_dynamic = 0.0 if case_id in {REFERENCE_CASE, "start975_unit_hard_clip", "start975_sigma_hard_clip", "terminal_hard_clip_recompute"} else 0.995
    if sampling.get("dynamic_threshold_percentile") != expected_dynamic:
        raise ValueError(f"{case_id}/{method} dynamic threshold differs")
    if sampling.get("prefix_budgets") != [PREFIX_BUDGETS[method]]:
        raise ValueError(f"{case_id}/{method} prefix budget differs")
    if sampling.get("actual_timesteps") != _timesteps(975 if start975 else 999):
        raise ValueError(f"{case_id}/{method} timestep schedule differs")
    seed = sampling.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError(f"{case_id}/{method} sampling seed is invalid")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(f"{case_id}/{method} sampling seed differs across the pair")
    stream = _object(sampling.get("random_stream"), f"{case_id}/{method} random stream")
    if stream != {
        "batch_size_invariant": True,
        "prefix_budgets_share_stream": True,
        "resume_index_invariant": True,
        "scope": "per_global_sample_index",
        "seed_formula": "(seed + global_index) mod 2^63",
    }:
        raise ValueError(f"{case_id}/{method} random-stream contract differs")
    digest = _object(sampling.get("sample_set_digest"), f"{case_id}/{method} sample digest")
    if digest != {"algorithm": "sha256", "framing": "filename_utf8_nul_file_bytes_nul"}:
        raise ValueError(f"{case_id}/{method} sample digest contract differs")
    api = _object(sampling.get("inference_api"), f"{case_id}/{method} inference API")
    if api != {"name": "cofitok.generation.GenerationSession", "version": 1}:
        raise ValueError(f"{case_id}/{method} inference API differs")
    return sampling


def _validate_metrics(value: Any, *, name: str) -> dict[str, float]:
    metrics = _object(value, name)
    expected = {
        "channel_saturation_fraction",
        "class_top1",
        "class_top5",
        "fid",
        "inception_score",
        "median_filter_residual_fraction",
        "total_variation",
    }
    if set(metrics) != expected:
        raise ValueError(f"{name} fields differ")
    normalized = {
        key: _finite(metrics[key], f"{name} {key}", minimum=0.0)
        for key in expected
    }
    for key in ("channel_saturation_fraction", "class_top1", "class_top5", "median_filter_residual_fraction"):
        if normalized[key] > 1.0:
            raise ValueError(f"{name} {key} is outside [0, 1]")
    if normalized["class_top1"] > normalized["class_top5"]:
        raise ValueError(f"{name} top-1 exceeds top-5")
    if normalized["inception_score"] <= 0.0:
        raise ValueError(f"{name} inception score is not positive")
    return normalized


def _validate_artifact_metrics(value: Any, *, name: str) -> dict[str, float]:
    metrics = _object(value, name)
    if set(metrics) != set(ARTIFACT_METRIC_KEYS):
        raise ValueError(f"{name} fields differ")
    return {
        key: _finite(metrics[key], f"{name} {key}", minimum=0.0)
        for key in ARTIFACT_METRIC_KEYS
    }


def _validate_candidate_comparison(
    value: Any,
    *,
    candidate_metrics: dict[str, float],
    reference_metrics: dict[str, float],
    real_artifact_metrics: dict[str, float],
    name: str,
) -> dict[str, Any]:
    comparison = _object(value, name)
    expected_keys = {
        "artifact_distance_to_real",
        "checks",
        "fid_relative_improvement",
        "metrics",
        "passes",
        "reference_artifact_distance_to_real",
        "reference_metrics",
    }
    if set(comparison) != expected_keys:
        raise ValueError(f"{name} fields differ")
    stored_metrics = _validate_metrics(comparison["metrics"], name=f"{name} metrics")
    if stored_metrics != candidate_metrics:
        raise ValueError(f"{name} candidate metrics differ")
    stored_reference_metrics = _validate_metrics(
        comparison["reference_metrics"],
        name=f"{name} reference metrics",
    )
    if stored_reference_metrics != reference_metrics:
        raise ValueError(f"{name} reference metrics differ")
    if reference_metrics["fid"] <= 0.0:
        raise ValueError(f"{name} reference FID is not positive")

    def distances(metrics: dict[str, float]) -> dict[str, float]:
        return {
            key: abs(metrics[key] - real_artifact_metrics[key])
            for key in ARTIFACT_METRIC_KEYS
        }

    expected_distance = distances(candidate_metrics)
    expected_reference_distance = distances(reference_metrics)
    for field, expected in (
        ("artifact_distance_to_real", expected_distance),
        ("reference_artifact_distance_to_real", expected_reference_distance),
    ):
        stored = _object(comparison[field], f"{name} {field}")
        if set(stored) != set(ARTIFACT_METRIC_KEYS):
            raise ValueError(f"{name} {field} fields differ")
        for key in ARTIFACT_METRIC_KEYS:
            actual = _finite(stored[key], f"{name} {field} {key}", minimum=0.0)
            if not math.isclose(actual, expected[key], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{name} {field} differs")
    expected_checks = {
        "channel_saturation_fraction_real_distance_non_regression": expected_distance["channel_saturation_fraction"] <= expected_reference_distance["channel_saturation_fraction"],
        "class_top1_non_regression": candidate_metrics["class_top1"] >= reference_metrics["class_top1"],
        "class_top5_non_regression": candidate_metrics["class_top5"] >= reference_metrics["class_top5"],
        "fid_strictly_lower": candidate_metrics["fid"] < reference_metrics["fid"],
        "median_filter_residual_fraction_real_distance_non_regression": expected_distance["median_filter_residual_fraction"] <= expected_reference_distance["median_filter_residual_fraction"],
        "total_variation_real_distance_non_regression": expected_distance["total_variation"] <= expected_reference_distance["total_variation"],
    }
    checks = _object(comparison["checks"], f"{name} checks")
    if set(checks) != set(expected_checks) or any(
        not isinstance(checks[key], bool) or checks[key] is not expected
        for key, expected in expected_checks.items()
    ):
        raise ValueError(f"{name} checks differ")
    expected_passes = all(expected_checks.values())
    if comparison.get("passes") is not expected_passes:
        raise ValueError(f"{name} pass status differs")
    expected_improvement = (
        reference_metrics["fid"] - candidate_metrics["fid"]
    ) / reference_metrics["fid"]
    actual_improvement = _finite(
        comparison["fid_relative_improvement"],
        f"{name} FID improvement",
    )
    if not math.isclose(actual_improvement, expected_improvement, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} FID improvement differs")
    return {"passes": expected_passes, "fid_relative_improvement": expected_improvement}


def _validate_source_reports(
    observation: dict[str, Any],
    *,
    method: str,
    cache: dict[str, dict[str, Any]],
) -> None:
    reports = _object(observation.get("source_reports"), f"{method} source reports")
    if set(reports) != SOURCE_REPORT_NAMES:
        raise ValueError(f"{method} source report set differs")
    identities = {
        name: _physical_identity(reports[name], name=f"{method} {name}", cache=cache)
        for name in SOURCE_REPORT_NAMES
    }
    del identities
    artifact = _read_json(reports["artifact_report"]["path"], name=f"{method} artifact report")
    if (
        artifact.get("status") != "completed"
        or artifact.get("schema_version") != 1
        or artifact.get("role") != "generation_epsilon_stability_artifact_statistics_report"
        or artifact.get("sample_count") != 1000
        or artifact.get("sample_set_sha256") != observation["sample_set_sha256"]
        or artifact.get("metrics") != {
            key: observation["metrics"][key]
            for key in ("channel_saturation_fraction", "median_filter_residual_fraction", "total_variation")
        }
    ):
        raise ValueError(f"{method} artifact report is not bound to the observation")
    metrics = _read_json(reports["metrics_report"]["path"], name=f"{method} metrics report")
    metric_values = _object(metrics.get("metrics"), f"{method} metrics report metrics")
    counts = _object(metrics.get("counts"), f"{method} metrics report counts")
    provenance = _object(metrics.get("sample_provenance"), f"{method} metrics report provenance")
    class_report = _read_json(reports["class_fidelity_report"]["path"], name=f"{method} class-fidelity report")
    class_metrics = _object(class_report.get("metrics"), f"{method} class-fidelity metrics")
    class_provenance = _object(class_report.get("sample_provenance"), f"{method} class-fidelity provenance")
    try:
        metric_fid = _finite(metric_values.get("frechet_inception_distance"), f"{method} metrics report FID", minimum=0.0)
        metric_is = _finite(metric_values.get("inception_score_mean"), f"{method} metrics report inception score", minimum=0.0)
        class_top1 = _finite(class_metrics.get("top1_accuracy"), f"{method} class-fidelity top-1", minimum=0.0, maximum=1.0)
        class_top5 = _finite(class_metrics.get("top5_accuracy"), f"{method} class-fidelity top-5", minimum=0.0, maximum=1.0)
    except ValueError as error:
        raise ValueError(f"{method} source metric payload is malformed") from error
    if (
        metrics.get("status") != "completed"
        or metrics.get("protocol") != "torch_fidelity_directory_metrics"
        or counts.get("generated_image_count") != 1000
        or not math.isclose(metric_fid, observation["metrics"]["fid"], rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(metric_is, observation["metrics"]["inception_score"], rel_tol=0.0, abs_tol=1e-12)
        or provenance.get("checkpoint_sha256") != observation["checkpoint"]["sha256"]
        or provenance.get("checkpoint_step") != 100000
        or provenance.get("sample_set_sha256") != observation["sample_set_sha256"]
    ):
        raise ValueError(f"{method} metrics report is not bound to the observation")
    if (
        class_report.get("status") != "completed"
        or class_report.get("role") != "generation_class_fidelity_report"
        or class_metrics.get("sample_count") != 1000
        or not math.isclose(class_top1, observation["metrics"]["class_top1"], rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(class_top5, observation["metrics"]["class_top5"], rel_tol=0.0, abs_tol=1e-12)
        or class_provenance.get("checkpoint_sha256") != observation["checkpoint"]["sha256"]
        or class_provenance.get("checkpoint_step") != 100000
        or class_provenance.get("sample_set_sha256") != observation["sample_set_sha256"]
    ):
        raise ValueError(f"{method} class-fidelity report is not bound to the observation")
    sampling_report = _read_json(reports["sampling_report"]["path"], name=f"{method} sampling report")
    sampling = _object(sampling_report.get("sampling"), f"{method} sampling report protocol")
    if (
        sampling_report.get("status") != "completed"
        or sampling_report.get("checkpoint_step") != 100000
        or sampling_report.get("checkpoint_sha256") != observation["checkpoint"]["sha256"]
        or sampling != observation["sampling"]
    ):
        raise ValueError(f"{method} sampling report is not bound to the observation")


def _validate_checkpoint_pair(
    observation: dict[str, Any],
    *,
    method: str,
    expected_execution_method: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> None:
    checkpoint = _physical_identity(observation["checkpoint"], name=f"{method} observation checkpoint", cache=cache)
    sidecar = _physical_identity(observation["integrity_sidecar"], name=f"{method} observation sidecar", cache=cache)
    checkpoint_path = Path(checkpoint["path"])
    sidecar_path = Path(sidecar["path"])
    if checkpoint_path.name != "checkpoint_step_00100000.pt" or sidecar_path != checkpoint_path.with_name(f"{checkpoint_path.name}.integrity.json"):
        raise ValueError(f"{method} checkpoint/sidecar filename binding differs")
    sidecar_payload = _read_json(sidecar_path, name=f"{method} checkpoint sidecar")
    if (
        sidecar_payload.get("step") != 100000
        or sidecar_payload.get("checkpoint") != checkpoint_path.name
        or sidecar_payload.get("checkpoint_bytes") != checkpoint["bytes"]
        or sidecar_payload.get("checkpoint_sha256") != checkpoint["sha256"]
    ):
        raise ValueError(f"{method} checkpoint sidecar payload differs")
    expected_checkpoint = expected_execution_method["checkpoint"]
    if checkpoint != expected_checkpoint:
        raise ValueError(f"{method} observation checkpoint differs from execution binding")
    if sidecar != expected_execution_method["integrity_sidecar"]:
        raise ValueError(f"{method} observation sidecar differs from execution binding")


def _validate_observation(
    observation: dict[str, Any],
    *,
    case_id: str,
    method: str,
    expected_real_set: dict[str, Any],
    execution: dict[str, Any],
    expected_execution_authorization: dict[str, Any],
    expected_seed: int | None,
    cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    expected_keys = {
        "authorization_boundary", "case_id", "checkpoint", "design_identity",
        "execution_authorization_identity", "identity", "integrity_sidecar", "method",
        "metrics", "real_set", "sample_count", "sample_set_sha256", "sampling", "schema",
        "source_reports", "status",
    }
    if set(observation) != expected_keys:
        raise ValueError(f"{case_id}/{method} observation fields differ")
    if observation.get("schema") != OBSERVATION_SCHEMA or observation.get("status") != "pass":
        raise ValueError(f"{case_id}/{method} observation status/schema differs")
    if observation.get("case_id") != case_id or observation.get("method") != method:
        raise ValueError(f"{case_id}/{method} observation identity differs")
    if observation.get("sample_count") != 1000:
        raise ValueError(f"{case_id}/{method} sample count differs")
    _hex(observation.get("sample_set_sha256"), length=64, name=f"{case_id}/{method} sample set SHA256")
    if observation.get("authorization_boundary") != AUTHORIZATION_BOUNDARY:
        raise ValueError(f"{case_id}/{method} authorization boundary differs")
    if observation.get("real_set") != expected_real_set:
        raise ValueError(f"{case_id}/{method} real-set binding differs")
    design = _physical_identity(observation["design_identity"], name=f"{case_id}/{method} design", cache=cache)
    execution_design = _physical_identity(execution["design_identity"], name="execution design", cache=cache)
    if design != execution_design:
        raise ValueError(f"{case_id}/{method} design identity differs from execution")
    separate_auth = _physical_identity(observation["execution_authorization_identity"], name=f"{case_id}/{method} execution authorization", cache=cache)
    if separate_auth != expected_execution_authorization:
        raise ValueError(f"{case_id}/{method} authorization identity differs from execution")
    metrics = _validate_metrics(observation["metrics"], name=f"{case_id}/{method} metrics")
    sampling = _validate_sampling(observation["sampling"], method=method, case_id=case_id, expected_seed=expected_seed)
    _validate_checkpoint_pair(
        observation,
        method=method,
        expected_execution_method=_object(execution["methods"], "execution methods")[method],
        cache=cache,
    )
    _validate_source_reports(observation, method=method, cache=cache)
    identity_descriptor = _physical_identity(observation["identity"], name=f"{case_id}/{method} observation", cache=cache)
    stored = _read_json(identity_descriptor["path"], name=f"{case_id}/{method} observation")
    expected_stored = dict(observation)
    expected_stored.pop("identity")
    if stored != expected_stored:
        raise ValueError(f"{case_id}/{method} observation file payload differs")
    return metrics, int(sampling["seed"])


def _validate_execution(
    execution: Any,
    *,
    cache: dict[str, dict[str, Any]],
    allowed_output_prefix: str,
) -> dict[str, Any]:
    value = _object(execution, "execution")
    expected_keys = {
        "authorized_actions", "design_identity", "evaluator_git", "git", "methods", "output_root",
        "output_root_non_overlapping", "preparation_identity", "random_stream", "real_set",
        "runtime_environment_sha256s", "schema", "scope", "separate_execution_authorization_identity",
        "source_bindings", "status", "terminal_route_receipt_identity",
    }
    if set(value) != expected_keys:
        raise ValueError("sampling recovery execution fields differ")
    if value.get("schema") != "cofitok_matched_epsilon_stability_execution_authorization_v1" or value.get("scope") != "matched_1000_sample_epsilon_stability_sampling_diagnostic_only" or value.get("status") != "approved":
        raise ValueError("sampling recovery execution scope differs")
    if value.get("authorized_actions") != AUTHORIZED_ACTIONS or value.get("output_root_non_overlapping") is not True:
        raise ValueError("sampling recovery execution authorization boundary differs")
    output_root = value.get("output_root")
    if not isinstance(output_root, str) or not output_root.startswith(allowed_output_prefix.rstrip("/") + "/") or not Path(output_root).is_dir():
        raise ValueError("sampling recovery execution output root is invalid")
    git = _git(value["git"], name="sampling recovery execution Git", allow_tree=True)
    evaluator_git = _git(value["evaluator_git"], name="sampling recovery evaluator Git")
    if any(git[key] != evaluator_git[key] for key in ("branch", "revision", "tracked_dirty")):
        raise ValueError("sampling recovery evaluator Git differs from execution Git")
    random_stream = _object(value["random_stream"], "sampling recovery execution random stream")
    seed = random_stream.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63 or random_stream.get("fresh") is not True or random_stream.get("start_index") != 0 or not isinstance(random_stream.get("namespace"), str) or not random_stream["namespace"]:
        raise ValueError("sampling recovery execution random stream differs")
    runtimes = _object(value["runtime_environment_sha256s"], "sampling recovery runtime identities")
    if set(runtimes) != {"artifact", "class_fidelity", "metrics", "sampling"}:
        raise ValueError("sampling recovery runtime identity fields differ")
    for name, digest in runtimes.items():
        _hex(digest, length=64, name=f"sampling recovery {name} runtime SHA256")
    real_set = _validate_real_set(value["real_set"], name="execution real set")
    methods = _object(value["methods"], "sampling recovery execution methods")
    if set(methods) != set(METHODS):
        raise ValueError("sampling recovery execution methods differ")
    training_gits: list[tuple[str, str]] = []
    for method in METHODS:
        row = _object(methods[method], f"execution {method}")
        if set(row) != {"checkpoint", "checkpoint_step", "integrity_sidecar", "latest", "prefix_budget", "training_report"}:
            raise ValueError(f"execution {method} fields differ")
        if row.get("checkpoint_step") != 100000 or row.get("prefix_budget") != PREFIX_BUDGETS[method]:
            raise ValueError(f"execution {method} step/prefix differs")
        checkpoint = _physical_identity(row["checkpoint"], name=f"execution {method} checkpoint", cache=cache)
        sidecar = _physical_identity(row["integrity_sidecar"], name=f"execution {method} sidecar", cache=cache)
        latest = _physical_identity(row["latest"], name=f"execution {method} latest", cache=cache)
        report_identity = _physical_identity(row["training_report"], name=f"execution {method} training report", cache=cache)
        checkpoint_path = Path(checkpoint["path"])
        if sidecar["path"] != checkpoint_path.with_name(f"{checkpoint_path.name}.integrity.json").as_posix() or Path(latest["path"]).name != "latest.json" or Path(latest["path"]).parent != checkpoint_path.parent:
            raise ValueError(f"execution {method} checkpoint pointer layout differs")
        sidecar_payload = _read_json(sidecar["path"], name=f"execution {method} sidecar")
        if sidecar_payload.get("step") != 100000 or sidecar_payload.get("checkpoint_bytes") != checkpoint["bytes"] or sidecar_payload.get("checkpoint_sha256") != checkpoint["sha256"]:
            raise ValueError(f"execution {method} sidecar is not bound")
        latest_payload = _read_json(latest["path"], name=f"execution {method} latest")
        if any(latest_payload.get(key) != expected for key, expected in {
            "checkpoint": checkpoint_path.name,
            "step": 100000,
            "checkpoint_bytes": checkpoint["bytes"],
            "checkpoint_sha256": checkpoint["sha256"],
            "integrity_manifest": Path(sidecar["path"]).name,
        }.items()):
            raise ValueError(f"execution {method} latest pointer is not bound")
        report = _read_json(report_identity["path"], name=f"execution {method} training report")
        report_git = _object(report.get("git"), f"execution {method} training Git")
        if (
            report.get("training_complete") is not True
            or report.get("completed_steps") != 100000
            or report.get("target_steps") != 100000
            or report.get("output_dir") != checkpoint_path.parent.as_posix()
            or report_git.get("dirty") is not False
        ):
            raise ValueError(f"execution {method} training report is not source-bound")
        revision = report_git.get("revision")
        branch = report_git.get("branch")
        if not isinstance(revision, str) or HEX40.fullmatch(revision) is None or not isinstance(branch, str) or not branch:
            raise ValueError(f"execution {method} training Git is malformed")
        training_gits.append((revision, branch))
    if len(set(training_gits)) != 1:
        raise ValueError("execution methods do not share one source training Git identity")
    execution_sources = _object(value["source_bindings"], "sampling recovery execution sources")
    expected_source_names = {
        "classifier", "classifier_report", "cross_protocol_reconciliation", "dataset", "evaluator",
        "post_reconciliation_decision", "post_reconciliation_verification", "quality_bridge_result",
        "real_set", "runtime_environment", "training_pair_report",
    }
    if set(execution_sources) != expected_source_names:
        raise ValueError("sampling recovery execution source set differs")
    for name, descriptor in execution_sources.items():
        _physical_identity(descriptor, name=f"execution source {name}", cache=cache)
    return {
        "git": git,
        "real_set": real_set,
        "methods": methods,
        "seed": seed,
        "sources": execution_sources,
    }


def _validate_source_bindings(
    value: Any,
    *,
    execution: dict[str, Any],
    real_artifact_reference: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    bindings = _object(value, "sampling recovery source bindings")
    if set(bindings) != SOURCE_BINDING_NAMES:
        raise ValueError("sampling recovery source binding set differs")
    physical = {
        name: _physical_identity(bindings[name], name=f"source binding {name}", cache=cache)
        for name in SOURCE_BINDING_NAMES
    }
    if physical["design"] != _physical_identity(execution["design_identity"], name="execution design", cache=cache):
        raise ValueError("source design binding differs from execution")
    if physical["separate_execution_authorization"] != _physical_identity(execution["separate_execution_authorization_identity"], name="execution separate authorization", cache=cache):
        raise ValueError("source separate authorization binding differs from execution")
    if physical["preparation"] != _physical_identity(execution["preparation_identity"], name="execution preparation", cache=cache):
        raise ValueError("source preparation binding differs from execution")
    real_artifact_source = _read_json(
        physical["real_artifact_reference"]["path"],
        name="source real-artifact reference",
    )
    expected_real_artifact_source_keys = {
        "authorization_boundary",
        "design_identity",
        "execution_authorization_identity",
        "metrics",
        "real_set",
        "real_set_identity",
        "sample_count",
        "sample_set_sha256",
        "schema",
        "source_reports",
        "status",
    }
    if (
        set(real_artifact_source) != expected_real_artifact_source_keys
        or real_artifact_source.get("schema") != REAL_ARTIFACT_SCHEMA
        or real_artifact_source.get("status") != "pass"
        or real_artifact_source.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("source real-artifact reference schema/boundary differs")
    for key in (
        "metrics",
        "real_set",
        "real_set_identity",
        "sample_count",
        "sample_set_sha256",
        "source_reports",
    ):
        if real_artifact_source[key] != real_artifact_reference[key]:
            raise ValueError("source real-artifact reference payload differs")
    if _physical_identity(
        real_artifact_source["design_identity"],
        name="source real-artifact design",
        cache=cache,
    ) != physical["design"]:
        raise ValueError("source real-artifact design binding differs")
    if _physical_identity(
        real_artifact_source["execution_authorization_identity"],
        name="source real-artifact execution authorization",
        cache=cache,
    ) != physical["execution_authorization"]:
        raise ValueError("source real-artifact authorization binding differs")
    if _physical_identity(
        real_artifact_source["real_set_identity"],
        name="source real-artifact real-set identity",
        cache=cache,
    ) != _physical_identity(
        real_artifact_reference["real_set_identity"],
        name="real artifact real-set identity",
        cache=cache,
    ):
        raise ValueError("source real-artifact real-set binding differs")
    if physical["terminal_route_receipt"] != _physical_identity(
        execution["terminal_route_receipt_identity"],
        name="execution terminal route receipt",
        cache=cache,
    ):
        raise ValueError("source terminal-route binding differs from execution")
    execution_sources = _object(execution["source_bindings"], "sampling recovery execution sources")
    if physical["terminal_route_receipt"] != _physical_identity(
        execution_sources["post_reconciliation_decision"],
        name="execution post-reconciliation decision",
        cache=cache,
    ):
        raise ValueError("source terminal-route binding differs from post-reconciliation decision")
    return physical


def validate_result(
    report: Mapping[str, Any],
    *,
    result_path: str | Path | None = None,
    expected_result_sha256: str | None = None,
    allowed_output_prefix: str = PROJECT_OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Revalidate a completed recovery result and every declared source identity."""

    result = _object(report, "sampling recovery result")
    expected_keys = {
        "authorization_boundary", "candidates", "claim_boundary", "execution", "generation_advantage_proven",
        "observations", "real_artifact_reference", "schema", "scientific_status", "selected_case_id",
        "selection_contract", "selection_status", "source_bindings", "status",
    }
    if set(result) != expected_keys or result.get("schema") != RESULT_SCHEMA or result.get("status") != "pass":
        raise ValueError("sampling recovery result schema/status differs")
    if result.get("scientific_status") != "screening_only" or result.get("selection_status") != "no_shared_sampling_recovery_candidate" or result.get("selected_case_id") is not None or result.get("generation_advantage_proven") is not False:
        raise ValueError("sampling recovery result weakens the no-candidate boundary")
    if result.get("authorization_boundary") != AUTHORIZATION_BOUNDARY or result.get("claim_boundary") != CLAIM_BOUNDARY or result.get("selection_contract") != SELECTION_CONTRACT:
        raise ValueError("sampling recovery result boundary/selection contract differs")
    if result_path is not None:
        source = reject_symlink_chain(result_path, name="sampling recovery result")
        if expected_result_sha256 is not None:
            _hex(expected_result_sha256, length=64, name="expected sampling recovery result SHA256")
            if file_sha256(source) != expected_result_sha256:
                raise ValueError("sampling recovery result SHA256 differs")
    cache: dict[str, dict[str, Any]] = {}
    execution = _object(result["execution"], "execution")
    execution_summary = _validate_execution(
        execution,
        cache=cache,
        allowed_output_prefix=allowed_output_prefix,
    )
    real_artifact = _object(result["real_artifact_reference"], "real artifact reference")
    if set(real_artifact) != {"metrics", "real_set", "real_set_identity", "sample_count", "sample_set_sha256", "source_reports"}:
        raise ValueError("real artifact reference fields differ")
    real_set = _validate_real_set(real_artifact["real_set"], name="real artifact real set")
    if real_set != execution_summary["real_set"] or real_artifact.get("sample_count") != 1000:
        raise ValueError("real artifact reference real set/count differs")
    artifact_metrics = _validate_artifact_metrics(
        real_artifact["metrics"],
        name="real artifact metrics",
    )
    _hex(real_artifact.get("sample_set_sha256"), length=64, name="real artifact sample set SHA256")
    _physical_identity(real_artifact["real_set_identity"], name="real artifact real-set identity", cache=cache)
    artifact_reports = _object(real_artifact["source_reports"], "real artifact source reports")
    if set(artifact_reports) != {"artifact_report", "subset_manifest"}:
        raise ValueError("real artifact source report set differs")
    for name, descriptor in artifact_reports.items():
        _physical_identity(descriptor, name=f"real artifact {name}", cache=cache)
    source_binding_identities = _validate_source_bindings(
        result["source_bindings"],
        execution=execution,
        real_artifact_reference=real_artifact,
        cache=cache,
    )
    observations = result.get("observations")
    if not isinstance(observations, list) or len(observations) != len(CASE_IDS) * len(METHODS):
        raise ValueError("sampling recovery observation count differs")
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    seed: int | None = None
    for raw in observations:
        observation = _object(raw, "sampling recovery observation")
        case_id = observation.get("case_id")
        method = observation.get("method")
        if case_id not in CASE_IDS or method not in METHODS:
            raise ValueError("sampling recovery observation case/method differs")
        case = by_case.setdefault(case_id, {})
        if method in case:
            raise ValueError("sampling recovery contains a duplicate observation")
        _validate_observation(
            observation,
            case_id=case_id,
            method=method,
            expected_real_set=real_set,
            execution=execution,
            expected_execution_authorization=source_binding_identities["execution_authorization"],
            expected_seed=seed,
            cache=cache,
        )
        current_seed = int(observation["sampling"]["seed"])
        seed = current_seed if seed is None else seed
        case[method] = observation
    if set(by_case) != CASE_IDS or any(set(rows) != set(METHODS) for rows in by_case.values()):
        raise ValueError("sampling recovery does not contain one complete pair per case")
    if execution_summary["seed"] != seed:
        raise ValueError("sampling recovery execution seed differs from observations")
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(CANDIDATE_CASE_IDS):
        raise ValueError("sampling recovery candidate count differs")
    seen_candidates: set[str] = set()
    reference = by_case[REFERENCE_CASE]
    for raw in candidates:
        candidate = _object(raw, "sampling recovery candidate")
        if set(candidate) != {"attribution_role", "case_id", "incremental_control", "methods", "parent_case_id", "passes_shared_recovery_screen", "worst_method_fid_relative_improvement"}:
            raise ValueError("sampling recovery candidate fields differ")
        case_id = candidate.get("case_id")
        if case_id not in CANDIDATE_CASE_IDS or case_id in seen_candidates:
            raise ValueError("sampling recovery candidate case set differs")
        seen_candidates.add(case_id)
        parent = candidate.get("parent_case_id")
        if not isinstance(parent, str) or parent not in CASE_IDS or parent == case_id:
            raise ValueError("sampling recovery candidate parent is invalid")
        if not isinstance(candidate.get("attribution_role"), str) or not candidate["attribution_role"] or not isinstance(candidate.get("incremental_control"), str) or not candidate["incremental_control"]:
            raise ValueError("sampling recovery candidate attribution is incomplete")
        method_comparisons = _object(candidate.get("methods"), f"{case_id} candidate methods")
        if set(method_comparisons) != set(METHODS):
            raise ValueError("sampling recovery candidate method set differs")
        observed = by_case[case_id]
        improvements = []
        legacy_passes = []
        for method in METHODS:
            candidate_method = _object(method_comparisons[method], f"{case_id}/{method} candidate comparisons")
            if set(candidate_method) != {"versus_legacy", "versus_parent"}:
                raise ValueError(f"{case_id}/{method} candidate comparison set differs")
            candidate_metrics = _validate_metrics(
                observed[method]["metrics"],
                name=f"{case_id}/{method} candidate observation metrics",
            )
            legacy_metrics = _validate_metrics(
                reference[method]["metrics"],
                name=f"{case_id}/{method} legacy reference metrics",
            )
            legacy_summary = _validate_candidate_comparison(
                candidate_method["versus_legacy"],
                candidate_metrics=candidate_metrics,
                reference_metrics=legacy_metrics,
                real_artifact_metrics=artifact_metrics,
                name=f"{case_id}/{method} versus legacy",
            )
            parent_metrics = _validate_metrics(
                by_case[parent][method]["metrics"],
                name=f"{case_id}/{method} parent reference metrics",
            )
            _validate_candidate_comparison(
                candidate_method["versus_parent"],
                candidate_metrics=candidate_metrics,
                reference_metrics=parent_metrics,
                real_artifact_metrics=artifact_metrics,
                name=f"{case_id}/{method} versus parent",
            )
            legacy_passes.append(legacy_summary["passes"])
            parent_fid = legacy_metrics["fid"]
            candidate_fid = candidate_metrics["fid"]
            improvements.append((parent_fid - candidate_fid) / parent_fid)
        expected_shared_pass = all(legacy_passes)
        if candidate.get("passes_shared_recovery_screen") is not expected_shared_pass or expected_shared_pass:
            raise ValueError("sampling recovery candidate weakens the no-candidate result")
        expected_worst = min(improvements)
        actual_worst = _finite(candidate.get("worst_method_fid_relative_improvement"), "candidate worst-method FID improvement")
        if not math.isclose(actual_worst, expected_worst, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("sampling recovery candidate improvement is not reproducible")
    if seen_candidates != CANDIDATE_CASE_IDS:
        raise ValueError("sampling recovery candidate case set is incomplete")
    return {
        "schema": RESULT_SCHEMA,
        "status": "pass",
        "scientific_status": result["scientific_status"],
        "selection_status": result["selection_status"],
        "generation_advantage_proven": False,
        "observation_count": len(observations),
        "case_count": len(by_case),
        "candidate_count": len(candidates),
        "physical_identity_count": len(cache),
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


__all__ = [
    "AUTHORIZATION_BOUNDARY",
    "CASE_IDS",
    "CLAIM_BOUNDARY",
    "RESULT_SCHEMA",
    "SELECTION_CONTRACT",
    "validate_result",
]
