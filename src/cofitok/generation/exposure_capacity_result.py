"""Fail-closed audit for the bounded 100K-to-110K exposure continuation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity_authorization import (
    AUTHORIZATION_BOUNDARY,
    EVALUATION_CONTRACT,
    SOURCE_CHECKOUT,
    TARGET_STEP,
    identity,
    read_object,
    validate_authorization_contract,
)
from cofitok.generation.exposure_capacity_gate import HORIZON_EXTENSION_SCHEDULER_POLICY
from cofitok.inference_replay import reject_symlink_chain
from cofitok.image_integrity import sample_set_sha256
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.checkpointing import resolve_latest_checkpoint, verify_training_checkpoint


RESULT_SCHEMA = "cofitok_generation_exposure_capacity_continuation_result_v1"
RESULT_ROLE = "bounded_exposure_capacity_continuation_result"
VALIDATION_RECEIPT_SCHEMA = (
    "cofitok_generation_exposure_capacity_result_validation_receipt_v1"
)
VALIDATION_RECEIPT_ROLE = "content_addressed_exposure_capacity_result_validation"
VALIDATION_RECEIPT_BOUNDARY = {
    "receipt_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "capacity_screen_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
SOURCE_STEP = 100_000
EXPECTED_IMAGES_SEEN = TARGET_STEP * 64
EXPECTED_SAMPLING = copy.deepcopy(EVALUATION_CONTRACT)
EXPECTED_PREFIXES = {"cofitok": 8, "dense_identity": 1}
HORIZON_EXTENSION_SCHEMA_VERSION = 1
EXPECTED_ROLLOUT_PROTOCOL = {
    "num_images": 64,
    "batch_size": 4,
    "teacher_timesteps": [999, 900, 750, 500, 250, 100, 10],
    "sample_steps": 100,
    "guidance_scale": 1.5,
    "teacher_guidance_scale": 1.0,
    "guidance_rescale": 0.0,
    "cfg_batch_mode": "batched",
    "clip_x0": True,
    "precision": "bf16",
    "seed": 2029,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its finite domain")
    return result


def _canonical_object_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialized_identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    path = row.get("path")
    size = row.get("bytes")
    digest = row.get("sha256")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name} identity path is missing")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"{name} identity byte count is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} identity SHA256 is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _clean_checkout_identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} Git identity fields differ")
    revision = row.get("revision")
    tree = row.get("tree")
    branch = row.get("branch")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or revision != revision.lower()
        or any(character not in "0123456789abcdef" for character in revision)
        or not isinstance(tree, str)
        or len(tree) != 40
        or tree != tree.lower()
        or any(character not in "0123456789abcdef" for character in tree)
        or not isinstance(branch, str)
        or not branch
        or row.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{name} must identify one exact clean checkout")
    return {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }


def _validation_source_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    methods = ("cofitok", "dense_identity")
    root_sources = {
        name: _serialized_identity(result.get(name), f"result {name}")
        for name in (
            "authorization",
            "candidate_gate",
            "preparation",
            "standing_authorization",
        )
    }
    configs = _object(result.get("configs"), "result configs")
    if set(configs) != set(methods):
        raise ValueError("result config identity set differs")
    normalized_configs = {
        method: _serialized_identity(
            configs.get(method), f"result {method} config"
        )
        for method in methods
    }
    training = _object(result.get("training"), "result training evidence")
    sampling = _object(result.get("sampling"), "result sampling evidence")
    quality = _object(result.get("quality"), "result quality evidence")
    class_fidelity = _object(
        result.get("class_fidelity"), "result class-fidelity evidence"
    )
    mechanism = _object(result.get("mechanism"), "result mechanism evidence")
    rollout = _object(result.get("rollout"), "result rollout evidence")
    for label, rows in (
        ("training", training),
        ("sampling", sampling),
        ("quality", quality),
        ("class fidelity", class_fidelity),
        ("mechanism", mechanism),
        ("rollout", rollout),
    ):
        if set(rows) != set(methods):
            raise ValueError(f"result {label} method set differs")
    method_sources: dict[str, Any] = {}
    for method in methods:
        training_row = _object(training.get(method), f"result {method} training")
        sampling_row = _object(sampling.get(method), f"result {method} sampling")
        quality_row = _object(quality.get(method), f"result {method} quality")
        class_row = _object(
            class_fidelity.get(method), f"result {method} class fidelity"
        )
        mechanism_row = _object(
            mechanism.get(method), f"result {method} mechanism"
        )
        rollout_row = _object(rollout.get(method), f"result {method} rollout")
        method_sources[method] = {
            "training_report": _serialized_identity(
                training_row.get("report"), f"result {method} training report"
            ),
            "checkpoint": _serialized_identity(
                training_row.get("checkpoint"), f"result {method} checkpoint"
            ),
            "checkpoint_integrity_manifest": _serialized_identity(
                training_row.get("integrity_manifest"),
                f"result {method} checkpoint integrity manifest",
            ),
            "latest": _serialized_identity(
                training_row.get("latest"), f"result {method} latest pointer"
            ),
            "metrics": _serialized_identity(
                training_row.get("metrics"), f"result {method} training metrics"
            ),
            "sampling_report": _serialized_identity(
                sampling_row.get("report"), f"result {method} sampling report"
            ),
            "generation_metrics_report": _serialized_identity(
                quality_row.get("report"),
                f"result {method} generation metrics report",
            ),
            "class_fidelity_report": _serialized_identity(
                class_row.get("report"),
                f"result {method} class-fidelity report",
            ),
            "checkpoint_evaluation_report": _serialized_identity(
                mechanism_row.get("report"),
                f"result {method} checkpoint evaluation report",
            ),
            "rollout_stability_report": _serialized_identity(
                rollout_row.get("report"),
                f"result {method} rollout stability report",
            ),
        }
    return {
        "root": root_sources,
        "configs": normalized_configs,
        "methods": method_sources,
    }


def _verified_final_checkpoint(
    report: Mapping[str, Any],
    *,
    method: str,
    expected_run_dir: Path,
    report_path: Path,
    expected_source: Mapping[str, Any],
    expected_execution_checkout: Mapping[str, Any],
    expected_config_identity: Mapping[str, Any],
    expected_dataset_identity_sha256: str,
    expected_runtime_environment_sha256: str,
) -> dict[str, Any]:
    if report.get("training_complete") is not True or report.get("completed_steps") != TARGET_STEP:
        raise ValueError(f"{method} training report is not a completed 110K continuation")
    if report.get("target_steps") != TARGET_STEP:
        raise ValueError(f"{method} training target horizon differs")
    final_metrics = _object(report.get("final_metrics"), f"{method} final metrics")
    if final_metrics.get("step") != TARGET_STEP or final_metrics.get("samples_seen") != EXPECTED_IMAGES_SEEN:
        raise ValueError(f"{method} final metrics exposure differs")
    config = _object(report.get("config"), f"{method} training config")
    if config.get("runtime", {}).get("steps") != TARGET_STEP:
        raise ValueError(f"{method} training config horizon differs")
    config_identity = _object(expected_config_identity, f"{method} continuation config identity")
    if report.get("config_path") != config_identity.get("path"):
        raise ValueError(f"{method} training report config path differs")
    if config.get("data", {}).get("dataset") != "imagenet_256":
        raise ValueError(f"{method} training dataset differs")
    if config.get("data", {}).get("batch_size") != 64:
        raise ValueError(f"{method} training effective batch differs")
    git = _object(report.get("git"), f"{method} training Git")
    if (
        git.get("revision") != expected_execution_checkout.get("revision")
        or git.get("branch") != expected_execution_checkout.get("branch")
        or git.get("dirty") is not False
    ):
        raise ValueError(f"{method} training Git provenance differs from execution checkout")
    dataset = _object(report.get("dataset_provenance"), f"{method} dataset provenance")
    if dataset.get("identity_sha256") != expected_dataset_identity_sha256 or dataset.get("status") != "pass":
        raise ValueError(f"{method} dataset provenance differs from authorization")
    if report.get("runtime_environment_sha256") != expected_runtime_environment_sha256:
        raise ValueError(f"{method} runtime environment differs from authorization")

    source = _object(expected_source, f"{method} source checkpoint binding")
    source_checkpoint = _object(source.get("checkpoint"), f"{method} source checkpoint")
    source_checkpoint_path = str(source_checkpoint.get("path", ""))
    source_checkpoint_filename = Path(source_checkpoint_path).name
    if not source_checkpoint_path or not source_checkpoint_filename:
        raise ValueError(f"{method} authorized source checkpoint path is malformed")
    extension = _object(
        report.get("horizon_extension"),
        f"{method} horizon extension provenance",
    )
    scheduler = _object(
        extension.get("scheduler"),
        f"{method} horizon extension scheduler provenance",
    )
    extension_source = _object(
        extension.get("source_checkpoint"),
        f"{method} horizon extension source checkpoint",
    )
    expected_sidecar = _object(
        source.get("integrity_manifest"),
        f"{method} source checkpoint sidecar",
    )
    if (
        extension.get("schema_version") != HORIZON_EXTENSION_SCHEMA_VERSION
        or extension.get("kind") != "bounded_training_horizon_extension"
        or extension.get("source_horizon_steps") != SOURCE_STEP
        or extension.get("target_horizon_steps") != TARGET_STEP
        or extension.get("additional_horizon_steps") != TARGET_STEP - SOURCE_STEP
        or extension.get("source_checkpoint_step") != source.get("step")
        or extension.get("allowed_config_mismatch_paths")
        != ["config.name", "config.runtime.steps"]
        or extension.get("target_config_sha256") != _canonical_object_sha256(config)
        or extension_source.get("path") != source_checkpoint.get("path")
        or extension_source.get("filename") != source_checkpoint_filename
        or extension_source.get("step") != SOURCE_STEP
        or extension_source.get("bytes") != source_checkpoint.get("bytes")
        or extension_source.get("sha256") != source_checkpoint.get("sha256")
        or extension_source.get("integrity_manifest") != expected_sidecar.get("path")
        or extension_source.get("integrity_manifest_bytes")
        != expected_sidecar.get("bytes")
        or extension_source.get("integrity_manifest_sha256")
        != expected_sidecar.get("sha256")
        or scheduler.get("policy") != HORIZON_EXTENSION_SCHEDULER_POLICY
        or scheduler.get("source_horizon_steps") != SOURCE_STEP
        or scheduler.get("effective_horizon_steps") != SOURCE_STEP
        or scheduler.get("target_horizon_steps") != TARGET_STEP
        or scheduler.get("explicit_resume_target_steps_required") is not True
        or int(scheduler.get("restored_last_epoch", -1)) != int(
            extension.get("source_checkpoint_step", -1)
        )
    ):
        raise ValueError(f"{method} horizon extension provenance differs")
    transition = _object(
        report.get("resume_revision_transition"),
        f"{method} resume revision transition",
    )
    transition_source = _object(
        transition.get("source_checkpoint"),
        f"{method} resume transition source checkpoint",
    )
    if (
        report.get("resume") != source_checkpoint.get("path")
        or transition.get("schema_version") != 1
        or transition.get("reason") != "sampler_rng_state_device_compatibility"
        or transition.get("source_revision") != SOURCE_CHECKOUT["revision"]
        or transition.get("target_revision") != expected_execution_checkout.get("revision")
        or transition.get("branch") != expected_execution_checkout.get("branch")
        or transition_source.get("path") != source_checkpoint.get("path")
        or transition_source.get("filename") != source_checkpoint_filename
        or transition_source.get("bytes") != source_checkpoint.get("bytes")
        or transition_source.get("sha256") != source_checkpoint.get("sha256")
        or transition_source.get("step") != SOURCE_STEP
        or transition_source.get("integrity_manifest")
        != _object(source.get("integrity_manifest"), f"{method} source sidecar").get("path")
        or transition_source.get("git_branch") != SOURCE_CHECKOUT["branch"]
    ):
        raise ValueError(f"{method} resume provenance differs from the authorized source")
    if report.get("metrics_resume_reconciliation") is None:
        raise ValueError(f"{method} metrics resume reconciliation is missing")
    output_dir = reject_symlink_chain(report.get("output_dir", ""), name=f"{method} training output")
    if output_dir.resolve() != expected_run_dir.resolve():
        raise ValueError(f"{method} training output directory differs")
    latest = expected_run_dir / "latest.json"
    checkpoint = resolve_latest_checkpoint(expected_run_dir)
    if latest.resolve() != checkpoint.parent.joinpath("latest.json").resolve():
        raise ValueError(f"{method} latest pointer path differs")
    integrity = verify_training_checkpoint(checkpoint)
    if int(integrity.get("step", -1)) != TARGET_STEP:
        raise ValueError(f"{method} final checkpoint step differs")
    latest_payload = read_object(latest, name=f"{method} latest pointer")
    expected = {
        "checkpoint": checkpoint.name,
        "step": TARGET_STEP,
        "checkpoint_bytes": int(integrity["checkpoint_bytes"]),
        "checkpoint_sha256": str(integrity["checkpoint_sha256"]),
        "integrity_manifest": checkpoint.with_name(
            f"{checkpoint.name}.integrity.json"
        ).name,
    }
    for key, value in expected.items():
        if latest_payload.get(key) != value:
            raise ValueError(f"{method} latest pointer {key} differs from checkpoint")
    latest_report = _object(report.get("latest_checkpoint"), f"{method} report latest checkpoint")
    if (
        latest_report.get("checkpoint") != checkpoint.name
        or latest_report.get("checkpoint_sha256") != integrity["checkpoint_sha256"]
        or latest_report.get("checkpoint_bytes") != integrity["checkpoint_bytes"]
        or latest_report.get("step") != TARGET_STEP
    ):
        raise ValueError(f"{method} training report latest checkpoint differs")
    metrics_path = expected_run_dir / "train_metrics.jsonl"
    if not metrics_path.is_file() or metrics_path.is_symlink():
        raise ValueError(f"{method} training metrics history is missing")
    rows: list[dict[str, Any]] = []
    previous_step = 0
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = _object(json.loads(line), f"{method} training metric row")
        step = int(row.get("step", -1))
        samples_seen = int(row.get("samples_seen", -1))
        if step <= previous_step or samples_seen != step * 64 or step > TARGET_STEP:
            raise ValueError(f"{method} training metrics are not strictly increasing")
        previous_step = step
        rows.append(row)
    if not rows or rows[-1].get("step") != TARGET_STEP or rows[-1].get("samples_seen") != EXPECTED_IMAGES_SEEN:
        raise ValueError(f"{method} training metrics do not reach the 110K exposure")
    if rows[-1] != final_metrics:
        raise ValueError(f"{method} final metrics are not the canonical metrics tail")
    return {
        "report": identity(report_path),
        "run_dir": expected_run_dir.resolve().as_posix(),
        "checkpoint": identity(checkpoint),
        "integrity_manifest": identity(
            checkpoint.with_name(f"{checkpoint.name}.integrity.json")
        ),
        "latest": identity(latest),
        "git": git,
        "config": copy.deepcopy(_object(report.get("config"), f"{method} training config")),
        "metrics": identity(metrics_path),
        "parameter_count": report.get("parameter_count"),
    }


def _sampling_provenance(
    report_path: Path,
    *,
    expected_checkpoint: Mapping[str, Any],
    method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from scripts.evaluate_generation_metrics import find_images, validate_sampling_provenance
    except ModuleNotFoundError:  # pragma: no cover - direct script invocation fallback
        from evaluate_generation_metrics import find_images, validate_sampling_provenance

    report = read_object(report_path, name=f"{method} sampling report")
    if report.get("status") != "completed":
        raise ValueError(f"{method} sampling is not completed")
    sampling = _object(report.get("sampling"), f"{method} sampling protocol")
    expected = {
        "sample_steps": 100,
        "num_samples": 10_000,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "seed": 2027,
        "start_index": 0,
        "class_schedule": "balanced_modulo",
    }
    for key, value in expected.items():
        if sampling.get(key) != value:
            raise ValueError(f"{method} sampling {key} differs from the bounded protocol")
    if report.get("checkpoint_step") != TARGET_STEP or report.get("checkpoint_sha256") != expected_checkpoint["sha256"]:
        raise ValueError(f"{method} sampling checkpoint differs from final continuation")
    if report.get("weights") != "ema":
        raise ValueError(f"{method} sampling weights are not EMA")
    prefix = EXPECTED_PREFIXES[method]
    output_dirs = _object(report.get("output_dirs"), f"{method} sampling output dirs")
    generated_dir = reject_symlink_chain(
        output_dirs.get(str(prefix), ""), name=f"{method} generated sample directory"
    )
    images = find_images(generated_dir)
    if len(images) != 10_000:
        raise ValueError(f"{method} generated sample count differs")
    provenance = validate_sampling_provenance(report_path, generated_dir, images)
    if provenance["sampling"] != sampling:
        raise ValueError(f"{method} sampling protocol differs from validated provenance")
    if provenance["checkpoint_step"] != TARGET_STEP or provenance["weights"] != "ema":
        raise ValueError(f"{method} sampling provenance is not the final EMA checkpoint")
    if provenance["sample_set_sha256"] != sample_set_sha256(images):
        raise ValueError(f"{method} sample-set digest cannot be reproduced")
    normalized_sampling = copy.deepcopy(sampling)
    normalized_sampling["weights"] = report["weights"]
    return report, {
        "report": identity(report_path),
        "generated_dir": generated_dir.resolve().as_posix(),
        "sample_count": len(images),
        "sample_set_sha256": provenance["sample_set_sha256"],
        "sampling": normalized_sampling,
        "provenance": provenance,
    }


def _metrics_report(
    report_path: Path,
    *,
    method: str,
    sampling: Mapping[str, Any],
    expected_count: int = 10_000,
) -> dict[str, Any]:
    report = read_object(report_path, name=f"{method} metrics report")
    if report.get("status") != "completed" or report.get("protocol") != "torch_fidelity_directory_metrics":
        raise ValueError(f"{method} metrics report is not completed torch-fidelity evidence")
    counts = _object(report.get("counts"), f"{method} metrics counts")
    if counts.get("generated_image_count") != expected_count:
        raise ValueError(f"{method} metrics generated count differs")
    if report.get("sample_provenance") != sampling["provenance"]:
        raise ValueError(f"{method} metrics report is bound to another sampling report")
    metrics = _object(report.get("metrics"), f"{method} metrics")
    values = {
        name: _finite(metrics.get(name), f"{method} {name}", minimum=0.0)
        for name in (
            "frechet_inception_distance",
            "inception_score_mean",
            "inception_score_std",
            "precision",
            "recall",
        )
    }
    if values["inception_score_mean"] <= 0.0 or values["precision"] > 1.0 or values["recall"] > 1.0:
        raise ValueError(f"{method} distribution metrics are outside their domains")
    return {
        "report": identity(report_path),
        "metrics": values,
        "counts": counts,
        "sample_provenance": copy.deepcopy(report["sample_provenance"]),
    }


def _class_report(
    report_path: Path,
    *,
    method: str,
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    from scripts.evaluate_generation_class_fidelity import validate_class_fidelity_report

    report = read_object(report_path, name=f"{method} class-fidelity report")
    validate_class_fidelity_report(report)
    if report.get("sample_provenance") != sampling["provenance"]:
        raise ValueError(f"{method} class fidelity is bound to another sample set")
    metrics = _object(report.get("metrics"), f"{method} class-fidelity metrics")
    if metrics.get("sample_count") != 10_000 or metrics.get("requested_class_count") != 1000:
        raise ValueError(f"{method} class-fidelity sample/class count differs")
    return {
        "report": identity(report_path),
        "metrics": {
            "top1_accuracy": _finite(metrics.get("top1_accuracy"), f"{method} class top1"),
            "top5_accuracy": _finite(metrics.get("top5_accuracy"), f"{method} class top5"),
            "predicted_class_fraction": _finite(
                metrics.get("predicted_class_fraction"), f"{method} predicted class fraction"
            ),
            "normalized_predicted_class_entropy": _finite(
                metrics.get("normalized_predicted_class_entropy"),
                f"{method} class entropy",
            ),
        },
        "sample_provenance": copy.deepcopy(report["sample_provenance"]),
    }


def _checkpoint_eval(report_path: Path, *, method: str, expected_checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    report = read_object(report_path, name=f"{method} mechanism report")
    if report.get("status") != "completed":
        raise ValueError(f"{method} mechanism evaluation is not completed")
    if report.get("checkpoint_step") != TARGET_STEP or report.get("checkpoint_sha256") != expected_checkpoint["sha256"]:
        raise ValueError(f"{method} mechanism evaluation checkpoint differs")
    request = _object(report.get("request"), f"{method} mechanism request")
    metrics = _object(report.get("metrics"), f"{method} mechanism metrics")
    expected_random_orders = {"cofitok": 4, "dense_identity": 0}[method]
    if (
        request.get("num_images") != 1024
        or request.get("timestep") != 500
        or request.get("random_orders") != expected_random_orders
        or request.get("weights") != "ema"
    ):
        raise ValueError(f"{method} mechanism request differs")
    if metrics.get("evaluated_images") != 1024 or metrics.get("timestep") != 500:
        raise ValueError(f"{method} mechanism image/timestep count differs")
    return {
        "report": identity(report_path),
        "request": request,
        "metrics": {
            "evaluated_images": metrics["evaluated_images"],
            "timestep": metrics["timestep"],
            "order_count": metrics.get("order_count"),
        },
    }


def _rollout_report(report_path: Path, *, method: str, expected_checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    report = read_object(report_path, name=f"{method} rollout report")
    if report.get("status") != "completed" or report.get("checkpoint_step") != TARGET_STEP:
        raise ValueError(f"{method} rollout report is not a completed 110K evaluation")
    if report.get("checkpoint_sha256") != expected_checkpoint["sha256"]:
        raise ValueError(f"{method} rollout report checkpoint differs")
    if report.get("weights") != "ema":
        raise ValueError(f"{method} rollout weights are not EMA")
    protocol = _object(report.get("protocol"), f"{method} rollout protocol")
    for key, expected in EXPECTED_ROLLOUT_PROTOCOL.items():
        if protocol.get(key) != expected:
            raise ValueError(f"{method} rollout protocol differs at {key}")
    unexpected = sorted(set(protocol) - set(EXPECTED_ROLLOUT_PROTOCOL))
    if unexpected:
        raise ValueError(
            f"{method} rollout protocol has unexpected fields: {', '.join(unexpected)}"
        )
    return {"report": identity(report_path), "protocol": protocol}


def build_result(
    *,
    authorization: Mapping[str, Any],
    gate: Mapping[str, Any],
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    gate_identity: Mapping[str, Any],
    standing_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    source_checkout: Mapping[str, Any] | None = None,
    config_identities: Mapping[str, Any],
    authorization_identity: Mapping[str, Any],
    candidate_gate_identity: Mapping[str, Any],
    cofitok_run_dir: str | Path,
    dense_run_dir: str | Path,
    cofitok_training_report: str | Path,
    dense_training_report: str | Path,
    cofitok_sampling_report: str | Path,
    dense_sampling_report: str | Path,
    cofitok_metrics_report: str | Path,
    dense_metrics_report: str | Path,
    cofitok_class_report: str | Path,
    dense_class_report: str | Path,
    cofitok_checkpoint_report: str | Path,
    dense_checkpoint_report: str | Path,
    cofitok_rollout_report: str | Path,
    dense_rollout_report: str | Path,
) -> dict[str, Any]:
    validated_auth = validate_authorization_contract(
        authorization,
        gate=gate,
        preparation=preparation,
        preparation_identity=preparation_identity,
        gate_identity=gate_identity,
        standing_identity=standing_identity,
        execution_checkout=execution_checkout,
        source_checkout=source_checkout,
        config_identities=config_identities,
    )
    gate_live = _object(gate.get("live_prelaunch"), "candidate gate live snapshot")
    training_runtime_environment_sha256 = str(
        gate_live.get("runtime_environment_sha256", "")
    )
    if (
        len(training_runtime_environment_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in training_runtime_environment_sha256
        )
    ):
        raise ValueError("candidate gate training runtime SHA256 is malformed")
    training = {}
    sampling = {}
    quality = {}
    class_fidelity = {}
    mechanism = {}
    rollout = {}
    for method, run_dir, training_path, sampling_path, metrics_path, class_path, checkpoint_path, rollout_path in (
        (
            "cofitok",
            Path(cofitok_run_dir),
            Path(cofitok_training_report),
            Path(cofitok_sampling_report),
            Path(cofitok_metrics_report),
            Path(cofitok_class_report),
            Path(cofitok_checkpoint_report),
            Path(cofitok_rollout_report),
        ),
        (
            "dense_identity",
            Path(dense_run_dir),
            Path(dense_training_report),
            Path(dense_sampling_report),
            Path(dense_metrics_report),
            Path(dense_class_report),
            Path(dense_checkpoint_report),
            Path(dense_rollout_report),
        ),
    ):
        report = read_object(training_path, name=f"{method} training report")
        training[method] = _verified_final_checkpoint(
            report,
            method=method,
            expected_run_dir=run_dir,
            report_path=training_path,
            expected_source=_object(
                validated_auth["source_checkpoints"].get(method),
                f"{method} authorized source checkpoint",
            ),
            expected_execution_checkout=execution_checkout,
            expected_config_identity=_object(
                config_identities.get(method), f"{method} config identity"
            ),
            expected_dataset_identity_sha256=str(
                _object(validated_auth["live_prelaunch"], "authorization live snapshot")[
                    "dataset_identity_sha256"
                ]
            ),
            expected_runtime_environment_sha256=training_runtime_environment_sha256,
        )
        sampling_report, sampling[method] = _sampling_provenance(
            sampling_path,
            expected_checkpoint=training[method]["checkpoint"],
            method=method,
        )
        del sampling_report
        quality[method] = _metrics_report(
            metrics_path, method=method, sampling=sampling[method]
        )
        class_fidelity[method] = _class_report(
            class_path, method=method, sampling=sampling[method]
        )
        mechanism[method] = _checkpoint_eval(
            checkpoint_path,
            method=method,
            expected_checkpoint=training[method]["checkpoint"],
        )
        rollout[method] = _rollout_report(
            rollout_path,
            method=method,
            expected_checkpoint=training[method]["checkpoint"],
        )

    cofitok_sampling = sampling["cofitok"]["sampling"]
    dense_sampling = sampling["dense_identity"]["sampling"]
    for key in ("sample_steps", "num_samples", "weights", "guidance_scale", "guidance_rescale", "cfg_batch_mode", "eta", "seed", "start_index", "class_schedule"):
        if cofitok_sampling.get(key) != dense_sampling.get(key):
            raise ValueError(f"matched sampling protocol differs at {key}")
    cofitok_rollout_protocol = rollout["cofitok"]["protocol"]
    dense_rollout_protocol = rollout["dense_identity"]["protocol"]
    if cofitok_rollout_protocol != dense_rollout_protocol:
        raise ValueError("matched rollout protocol differs")
    revisions = {training[method]["git"].get("revision") for method in training}
    branches = {training[method]["git"].get("branch") for method in training}
    datasets = {
        training[method]["config"].get("data", {}).get("dataset")
        for method in training
    }
    runtime_shas = {
        read_object(
            cofitok_training_report if method == "cofitok" else dense_training_report,
            name=f"{method} training report",
        ).get("runtime_environment_sha256")
        for method in training
    }
    if len(revisions) != 1 or len(branches) != 1 or datasets != {"imagenet_256"} or len(runtime_shas) != 1:
        raise ValueError("matched continuation training provenance differs")
    cofitok_fid = quality["cofitok"]["metrics"]["frechet_inception_distance"]
    dense_fid = quality["dense_identity"]["metrics"]["frechet_inception_distance"]
    cofitok_recall = quality["cofitok"]["metrics"]["recall"]
    dense_recall = quality["dense_identity"]["metrics"]["recall"]
    return {
        "schema_version": RESULT_SCHEMA,
        "role": RESULT_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "hold",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "decision": "bounded_exposure_continuation_completed_without_claim_upgrade",
        "authorization": dict(authorization_identity),
        "candidate_gate": dict(candidate_gate_identity),
        "preparation": dict(preparation_identity),
        "standing_authorization": dict(standing_identity),
        "execution_checkout": dict(execution_checkout),
        "configs": copy.deepcopy(dict(config_identities)),
        "source_checkpoints": copy.deepcopy(validated_auth["source_checkpoints"]),
        "target": {
            "source_step": SOURCE_STEP,
            "target_step": TARGET_STEP,
            "additional_steps": TARGET_STEP - SOURCE_STEP,
            "effective_batch_size": 64,
            "methods": ["cofitok", "dense_identity"],
            "evaluation": copy.deepcopy(EXPECTED_SAMPLING),
        },
        "training": training,
        "sampling": sampling,
        "quality": {
            method: {
                "report": row["report"],
                "metrics": row["metrics"],
            }
            for method, row in quality.items()
        },
        "relative_quality": {
            "cofitok_fid_minus_dense": cofitok_fid - dense_fid,
            "cofitok_fid_over_dense": cofitok_fid / dense_fid,
            "cofitok_recall_minus_dense": cofitok_recall - dense_recall,
            "absolute_quality_claim_allowed": False,
        },
        "class_fidelity": class_fidelity,
        "mechanism": mechanism,
        "rollout": rollout,
        "claim_guards": {
            "terminal_hold_preserved": True,
            "generation_advantage_proven": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "release_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
        "validated_authorization": {
            "schema_version": validated_auth["schema_version"],
            "decision": validated_auth["decision"],
            "scope": validated_auth["scope"],
        },
    }


def build_validation_receipt(
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("role") != RESULT_ROLE
        or result.get("status") != "completed"
        or result.get("operational_status") != "pass"
        or result.get("scientific_status") != "hold"
        or result.get("terminal_status") != "hold"
        or result.get("generation_advantage_proven") is not False
    ):
        raise ValueError("continuation result is not the canonical completed hold")
    claim_guards = _object(result.get("claim_guards"), "result claim guards")
    if claim_guards != {
        "terminal_hold_preserved": True,
        "generation_advantage_proven": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_allowed": False,
        "release_allowed": False,
    }:
        raise ValueError("continuation result claim guards differ")
    serialized_result = _serialized_identity(result_identity, "continuation result")
    git = _clean_checkout_identity(validator_git, "result validator")
    execution = _clean_checkout_identity(
        result.get("execution_checkout"), "result execution checkout"
    )
    source_evidence = _validation_source_evidence(result)
    basis = {
        "result": serialized_result,
        "execution_checkout": execution,
        "validator_git": git,
        "source_evidence": source_evidence,
    }
    return {
        "schema_version": VALIDATION_RECEIPT_SCHEMA,
        "role": VALIDATION_RECEIPT_ROLE,
        "status": "pass",
        "result": serialized_result,
        "execution_checkout": execution,
        "validator_git": git,
        "source_evidence": source_evidence,
        "validation_basis_sha256": _canonical_object_sha256(basis),
        "scientific_status": "hold",
        "generation_advantage_proven": False,
        "authorization_boundary": copy.deepcopy(VALIDATION_RECEIPT_BOUNDARY),
    }


def validate_validation_receipt(
    receipt: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_object = _object(receipt, "result validation receipt")
    validator_git = _clean_checkout_identity(
        receipt_object.get("validator_git"), "result validator"
    )
    expected = build_validation_receipt(
        result=result,
        result_identity=result_identity,
        validator_git=validator_git,
    )
    if receipt_object != expected:
        raise ValueError("result validation receipt is not reproducible")
    return copy.deepcopy(expected)


__all__ = [
    "RESULT_ROLE",
    "RESULT_SCHEMA",
    "VALIDATION_RECEIPT_BOUNDARY",
    "VALIDATION_RECEIPT_ROLE",
    "VALIDATION_RECEIPT_SCHEMA",
    "build_result",
    "build_validation_receipt",
    "validate_validation_receipt",
]
