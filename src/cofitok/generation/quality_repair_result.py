from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from cofitok.environment import runtime_environment_sha256
from cofitok.generation.quality_repair import (
    EPSILON_STABILITY_SAMPLING_DESIGN_SCHEMA,
    build_epsilon_stability_sampling_design,
    materialize_epsilon_stability_case_protocol,
)
from cofitok.generation.quality_repair_artifact import (
    EPSILON_STABILITY_ARTIFACT_METRIC_KEYS,
    EPSILON_STABILITY_ARTIFACT_REPORT_ROLE,
    EPSILON_STABILITY_ARTIFACT_REPORT_SCHEMA_VERSION,
    EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY,
    normalize_epsilon_stability_artifact_metrics,
    normalize_epsilon_stability_boundary,
    normalize_epsilon_stability_identity,
    validate_epsilon_stability_artifact_report,
)
from cofitok.generation_class_fidelity import (
    CLASS_FIDELITY_REPORT_ROLE,
    CLASS_FIDELITY_REPORT_SCHEMA_VERSION,
    validate_class_fidelity_report,
)


EPSILON_STABILITY_EXECUTION_AUTHORIZATION_SCHEMA = (
    "cofitok_matched_epsilon_stability_execution_authorization_v1"
)
EPSILON_STABILITY_CASE_OBSERVATION_SCHEMA = (
    "cofitok_matched_epsilon_stability_case_observation_v1"
)
EPSILON_STABILITY_OBSERVATION_MANIFEST_SCHEMA = (
    "cofitok_matched_epsilon_stability_observation_manifest_v1"
)
EPSILON_STABILITY_REAL_ARTIFACT_REFERENCE_SCHEMA = (
    "cofitok_epsilon_stability_real_artifact_reference_v1"
)
EPSILON_STABILITY_RESULT_SCHEMA = (
    "cofitok_matched_epsilon_stability_sampling_result_v1"
)

METHODS = ("cofitok", "dense_identity")
REPORT_IDENTITY_KEYS = (
    "sampling_report",
    "metrics_report",
    "class_fidelity_report",
    "artifact_report",
)
METRIC_KEYS = (
    "fid",
    "inception_score",
    "class_top1",
    "class_top5",
    *EPSILON_STABILITY_ARTIFACT_METRIC_KEYS,
)
EPSILON_STABILITY_EXECUTION_SOURCE_BINDINGS = (
    "post_reconciliation_decision",
    "post_reconciliation_verification",
    "cross_protocol_reconciliation",
    "quality_bridge_result",
    "training_pair_report",
    "dataset",
    "real_set",
    "runtime_environment",
    "evaluator",
    "classifier",
    "classifier_report",
)
EPSILON_STABILITY_EXECUTION_ACTIONS = {
    "matched_1000_sample_sampling": True,
    "associated_non_formal_evaluation": True,
    "independent_10000_confirmation": False,
    "training": False,
    "full_300k": False,
    "promotion": False,
    "inference_export": False,
    "release": False,
    "process_signals": False,
}
RUNTIME_ROLES = ("sampling", "metrics", "class_fidelity", "artifact")


def _sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} SHA256 is malformed")
    return text


def _identity(value: Any, *, label: str) -> dict[str, Any]:
    return normalize_epsilon_stability_identity(value, label=label)


def _git_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "revision",
        "branch",
        "tracked_dirty",
    }:
        raise ValueError(f"{label} Git identity is malformed")
    revision = str(value["revision"])
    branch = str(value["branch"])
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not branch
        or value["tracked_dirty"] is not False
    ):
        raise ValueError(f"{label} Git identity is malformed")
    return {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }


def _execution_report_git(execution: Mapping[str, Any]) -> dict[str, Any]:
    git = execution["git"]
    return {
        "revision": git["revision"],
        "branch": git["branch"],
        "tracked_dirty": False,
    }


def _evaluator_report_git(execution: Mapping[str, Any]) -> dict[str, Any]:
    return dict(execution["evaluator_git"])


def _metric(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"metric {name} is invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metric {name} is invalid") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"metric {name} is invalid")
    if name in {
        "class_top1",
        "class_top5",
        "channel_saturation_fraction",
        "median_filter_residual_fraction",
    } and result > 1.0:
        raise ValueError(f"metric {name} is outside [0, 1]")
    if name == "total_variation" and result > 2.0:
        raise ValueError("metric total_variation exceeds 2")
    if name == "inception_score" and result <= 0.0:
        raise ValueError("metric inception_score must be positive")
    return result


def _metrics(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(METRIC_KEYS):
        raise ValueError(f"{label} metric set is incomplete")
    return {name: _metric(value[name], name=name) for name in METRIC_KEYS}


def _real_set(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "digest_schema",
        "sha256",
        "root",
        "image_count",
    }:
        raise ValueError(f"{label} real-set identity is malformed")
    root = str(value["root"])
    image_count = value["image_count"]
    if (
        not PurePosixPath(root).is_absolute()
        or isinstance(image_count, bool)
        or not isinstance(image_count, int)
        or image_count < 1_000
        or not str(value["digest_schema"])
    ):
        raise ValueError(f"{label} real-set identity is malformed")
    return {
        "digest_schema": str(value["digest_schema"]),
        "sha256": _sha256(value["sha256"], label=f"{label} real set"),
        "root": root,
        "image_count": image_count,
    }


def _runtime_sha(report: Mapping[str, Any], *, label: str) -> str:
    environment = report.get("runtime_environment")
    if not isinstance(environment, Mapping):
        raise ValueError(f"{label} runtime environment is missing")
    actual = runtime_environment_sha256(dict(environment))
    if report.get("runtime_environment_sha256") != actual:
        raise ValueError(f"{label} runtime environment SHA256 differs")
    return actual


def _report_source(value: Any, *, label: str) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"identity", "payload"}:
        raise ValueError(f"{label} source is malformed")
    identity = _identity(value["identity"], label=label)
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} payload is missing")
    return identity, payload


def _validate_execution_authorization(
    authorization: Mapping[str, Any],
    *,
    design_identity: dict[str, Any],
    design: Mapping[str, Any],
) -> dict[str, Any]:
    if set(authorization) != {
        "schema",
        "status",
        "scope",
        "preparation_identity",
        "design_identity",
        "terminal_route_receipt_identity",
        "separate_execution_authorization_identity",
        "git",
        "random_stream",
        "output_root",
        "output_root_non_overlapping",
        "evaluator_git",
        "runtime_environment_sha256s",
        "real_set",
        "methods",
        "source_bindings",
        "authorized_actions",
    }:
        raise ValueError("epsilon-stability execution authorization fields differ")
    if (
        authorization.get("schema")
        != EPSILON_STABILITY_EXECUTION_AUTHORIZATION_SCHEMA
    ):
        raise ValueError("epsilon-stability execution authorization schema mismatch")
    if authorization.get("status") != "approved":
        raise ValueError("epsilon-stability execution authorization is not approved")
    if authorization.get("design_identity") != design_identity:
        raise ValueError("execution authorization binds another design")
    if authorization.get("scope") != (
        "matched_1000_sample_epsilon_stability_sampling_diagnostic_only"
    ):
        raise ValueError("epsilon-stability execution scope is not exact")
    preparation_identity = _identity(
        authorization.get("preparation_identity"),
        label="epsilon-stability preparation",
    )
    route_identity = _identity(
        authorization.get("terminal_route_receipt_identity"),
        label="terminal route receipt",
    )
    user_authorization_identity = _identity(
        authorization.get("separate_execution_authorization_identity"),
        label="separate execution authorization",
    )
    git = authorization.get("git")
    if not isinstance(git, Mapping) or set(git) != {
        "revision",
        "tree",
        "branch",
        "tracked_dirty",
    }:
        raise ValueError("epsilon-stability execution Git identity is malformed")
    revision = str(git["revision"])
    tree = str(git["tree"])
    branch = str(git["branch"])
    if (
        len(revision) != 40
        or len(tree) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or any(character not in "0123456789abcdef" for character in tree)
        or not branch
        or git["tracked_dirty"] is not False
    ):
        raise ValueError("epsilon-stability execution Git identity is malformed")
    random_stream = authorization.get("random_stream")
    if not isinstance(random_stream, Mapping) or set(random_stream) != {
        "seed",
        "start_index",
        "namespace",
        "fresh",
    }:
        raise ValueError("epsilon-stability random stream is malformed")
    seed = random_stream["seed"]
    start_index = random_stream["start_index"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("epsilon-stability random stream seed is invalid")
    if (
        start_index
        != design["common_sampling_contract"]["random_stream"]["start_index"]
    ):
        raise ValueError("epsilon-stability random stream start index is incompatible")
    namespace = str(random_stream["namespace"])
    if not namespace or random_stream["fresh"] is not True:
        raise ValueError("epsilon-stability random stream is not fresh")
    output_root = str(authorization.get("output_root", ""))
    if (
        not PurePosixPath(output_root).is_absolute()
        or not output_root.startswith(
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        )
        or authorization.get("output_root_non_overlapping") is not True
    ):
        raise ValueError("epsilon-stability output root is not isolated")
    evaluator_git = _git_identity(
        authorization.get("evaluator_git"),
        label="epsilon-stability evaluator",
    )
    runtimes = authorization.get("runtime_environment_sha256s")
    if not isinstance(runtimes, Mapping) or set(runtimes) != set(RUNTIME_ROLES):
        raise ValueError("epsilon-stability runtime identities are incomplete")
    normalized_runtimes = {
        role: _sha256(runtimes[role], label=f"{role} runtime environment")
        for role in RUNTIME_ROLES
    }
    normalized_real_set = _real_set(
        authorization.get("real_set"),
        label="epsilon-stability authorization",
    )
    methods = authorization.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(METHODS):
        raise ValueError("epsilon-stability authorized method set is incomplete")
    normalized_methods: dict[str, Any] = {}
    for method in METHODS:
        row = methods[method]
        expected = design["matched_methods"][method]
        if not isinstance(row, Mapping) or set(row) != {
            "checkpoint_step",
            "prefix_budget",
            "checkpoint",
            "integrity_sidecar",
            "latest",
            "training_report",
        }:
            raise ValueError(f"{method} execution source is malformed")
        if (
            row["checkpoint_step"] != expected["checkpoint_step"]
            or row["prefix_budget"] != expected["prefix_budget"]
        ):
            raise ValueError(f"{method} execution source differs from the design")
        normalized_methods[method] = {
            "checkpoint_step": int(row["checkpoint_step"]),
            "prefix_budget": int(row["prefix_budget"]),
            "checkpoint": _identity(
                row["checkpoint"], label=f"{method} checkpoint"
            ),
            "integrity_sidecar": _identity(
                row["integrity_sidecar"],
                label=f"{method} checkpoint sidecar",
            ),
            "latest": _identity(row["latest"], label=f"{method} latest"),
            "training_report": _identity(
                row["training_report"], label=f"{method} training report"
            ),
        }
    source_bindings = authorization.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != set(
        EPSILON_STABILITY_EXECUTION_SOURCE_BINDINGS
    ):
        raise ValueError("epsilon-stability execution source bindings are incomplete")
    normalized_bindings = {
        name: _identity(source_bindings[name], label=name)
        for name in EPSILON_STABILITY_EXECUTION_SOURCE_BINDINGS
    }
    if (
        authorization.get("authorized_actions")
        != EPSILON_STABILITY_EXECUTION_ACTIONS
    ):
        raise ValueError("epsilon-stability authorized action set is not exact")
    return {
        "schema": authorization["schema"],
        "status": authorization["status"],
        "scope": authorization["scope"],
        "preparation_identity": preparation_identity,
        "design_identity": design_identity,
        "terminal_route_receipt_identity": route_identity,
        "separate_execution_authorization_identity": user_authorization_identity,
        "git": {
            "revision": revision,
            "tree": tree,
            "branch": branch,
            "tracked_dirty": False,
        },
        "random_stream": {
            "seed": seed,
            "start_index": start_index,
            "namespace": namespace,
            "fresh": True,
        },
        "output_root": output_root,
        "output_root_non_overlapping": True,
        "evaluator_git": evaluator_git,
        "runtime_environment_sha256s": normalized_runtimes,
        "real_set": normalized_real_set,
        "methods": normalized_methods,
        "source_bindings": normalized_bindings,
        "authorized_actions": dict(EPSILON_STABILITY_EXECUTION_ACTIONS),
    }


def validate_epsilon_stability_execution_authorization(
    authorization: Mapping[str, Any],
    *,
    design_identity: Mapping[str, Any],
    design: Mapping[str, Any],
) -> dict[str, Any]:
    return _validate_execution_authorization(
        authorization,
        design_identity=_identity(design_identity, label="sampling design"),
        design=design,
    )


def _validate_sampling_report(
    source: Mapping[str, Any],
    *,
    expected_sampling: Mapping[str, Any],
    execution: Mapping[str, Any],
    method: str,
    case_id: str,
) -> dict[str, Any]:
    identity, report = _report_source(source, label="sampling report")
    method_source = execution["methods"][method]
    prefix_budget = int(method_source["prefix_budget"])
    expected_git = _execution_report_git(execution)
    if (
        report.get("schema_version") != 6
        or report.get("status") != "completed"
        or report.get("git") != expected_git
        or report.get("weights") != "ema"
        or report.get("sampling") != dict(expected_sampling)
        or report.get("checkpoint") != method_source["checkpoint"]["path"]
        or report.get("checkpoint_sha256")
        != method_source["checkpoint"]["sha256"]
        or report.get("checkpoint_integrity_manifest")
        != method_source["integrity_sidecar"]["path"]
        or report.get("checkpoint_step") != method_source["checkpoint_step"]
    ):
        raise ValueError(f"{method}/{case_id} sampling report contract differs")
    runtime_sha = _runtime_sha(report, label=f"{method}/{case_id} sampling")
    if runtime_sha != execution["runtime_environment_sha256s"]["sampling"]:
        raise ValueError(f"{method}/{case_id} sampling runtime is unauthorized")
    output_dirs = report.get("output_dirs")
    sample_sets = report.get("sample_sets")
    budget_key = str(prefix_budget)
    if (
        not isinstance(output_dirs, Mapping)
        or set(output_dirs) != {budget_key}
        or not PurePosixPath(str(output_dirs[budget_key])).is_absolute()
        or not isinstance(sample_sets, Mapping)
        or set(sample_sets) != {budget_key}
        or not isinstance(sample_sets[budget_key], Mapping)
        or set(sample_sets[budget_key]) != {"count", "sha256"}
        or sample_sets[budget_key]["count"] != 1_000
    ):
        raise ValueError(f"{method}/{case_id} sampling output set differs")
    sample_sha256 = _sha256(
        sample_sets[budget_key]["sha256"],
        label=f"{method}/{case_id} sample set",
    )
    output_root = PurePosixPath(str(execution["output_root"]))
    if (
        PurePosixPath(identity["path"]).parent.parent.parent.parent
        != output_root
        and output_root not in PurePosixPath(identity["path"]).parents
    ):
        raise ValueError(f"{method}/{case_id} sampling report is outside output root")
    if output_root not in PurePosixPath(str(output_dirs[budget_key])).parents:
        raise ValueError(f"{method}/{case_id} samples are outside output root")
    if (
        not PurePosixPath(str(report.get("sampling_progress", ""))).is_absolute()
    ):
        raise ValueError(f"{method}/{case_id} sampling completion evidence differs")
    _sha256(
        report.get("sampling_manifest_sha256"),
        label=f"{method}/{case_id} sampling manifest",
    )
    try:
        elapsed_seconds = float(report["elapsed_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{method}/{case_id} sampling elapsed time is invalid"
        ) from error
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise ValueError(f"{method}/{case_id} sampling elapsed time is invalid")
    return {
        "identity": identity,
        "sample_set_sha256": sample_sha256,
        "generated_dir": str(output_dirs[budget_key]),
        "runtime_environment_sha256": runtime_sha,
        "sampling": dict(expected_sampling),
        "git": expected_git,
    }


def _validate_sample_provenance(
    provenance: Any,
    *,
    sampling: Mapping[str, Any],
    execution: Mapping[str, Any],
    method: str,
    case_id: str,
    label: str,
) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{method}/{case_id} {label} sample provenance is missing")
    method_source = execution["methods"][method]
    progress = provenance.get("sampling_progress")
    if (
        provenance.get("report") != sampling["identity"]["path"]
        or provenance.get("report_identity") != sampling["identity"]
        or provenance.get("checkpoint") != method_source["checkpoint"]["path"]
        or provenance.get("checkpoint_sha256")
        != method_source["checkpoint"]["sha256"]
        or provenance.get("checkpoint_integrity_manifest")
        != method_source["integrity_sidecar"]["path"]
        or provenance.get("checkpoint_step") != method_source["checkpoint_step"]
        or provenance.get("weights") != "ema"
        or provenance.get("git") != sampling["git"]
        or provenance.get("runtime_environment_sha256")
        != sampling["runtime_environment_sha256"]
        or provenance.get("selected_prefix_budget")
        != method_source["prefix_budget"]
        or provenance.get("image_shape") != [3, 256, 256]
        or provenance.get("sample_set_sha256")
        != sampling["sample_set_sha256"]
        or provenance.get("sampling") != sampling["sampling"]
        or not isinstance(progress, Mapping)
        or progress.get("status") != "completed"
        or progress.get("completed_samples") != 1_000
    ):
        raise ValueError(f"{method}/{case_id} {label} sample provenance differs")
    _identity(
        provenance.get("manifest_identity"),
        label=f"{method}/{case_id} sampling manifest",
    )
    _identity(
        progress.get("identity"),
        label=f"{method}/{case_id} sampling progress",
    )


def _validate_metrics_report(
    source: Mapping[str, Any],
    *,
    sampling: Mapping[str, Any],
    execution: Mapping[str, Any],
    method: str,
    case_id: str,
) -> dict[str, Any]:
    identity, report = _report_source(source, label="metrics report")
    if (
        report.get("schema_version") != 3
        or report.get("role") != "generation_directory_metrics_report"
        or report.get("status") != "completed"
        or report.get("protocol") != "torch_fidelity_directory_metrics"
        or report.get("git") != _evaluator_report_git(execution)
    ):
        raise ValueError(f"{method}/{case_id} metrics report contract differs")
    if (
        _runtime_sha(report, label=f"{method}/{case_id} metrics evaluator")
        != execution["runtime_environment_sha256s"]["metrics"]
    ):
        raise ValueError(f"{method}/{case_id} metrics runtime is unauthorized")
    counts = report.get("counts")
    paths = report.get("paths")
    if (
        not isinstance(counts, Mapping)
        or counts.get("generated_image_count") != 1_000
        or not isinstance(counts.get("real_image_count"), int)
        or counts["real_image_count"] < 1_000
        or not isinstance(paths, Mapping)
        or paths.get("sampling_report") != sampling["identity"]["path"]
        or paths.get("generated_dir") != sampling["generated_dir"]
    ):
        raise ValueError(f"{method}/{case_id} metrics source paths differ")
    _validate_sample_provenance(
        report.get("sample_provenance"),
        sampling=sampling,
        execution=execution,
        method=method,
        case_id=case_id,
        label="metrics",
    )
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{method}/{case_id} metrics payload is missing")
    fid = _metric(metrics.get("frechet_inception_distance"), name="fid")
    inception = _metric(metrics.get("inception_score_mean"), name="inception_score")
    real_set = _real_set(
        report.get("real_set"), label=f"{method}/{case_id} metrics"
    )
    if real_set != execution["real_set"]:
        raise ValueError(f"{method}/{case_id} metrics real set is unauthorized")
    return {
        "identity": identity,
        "fid": fid,
        "inception_score": inception,
        "real_set": real_set,
    }


def _validate_class_fidelity_source(
    source: Mapping[str, Any],
    *,
    sampling: Mapping[str, Any],
    execution: Mapping[str, Any],
    method: str,
    case_id: str,
) -> dict[str, Any]:
    identity, report = _report_source(source, label="class-fidelity report")
    if (
        report.get("schema_version") != CLASS_FIDELITY_REPORT_SCHEMA_VERSION
        or report.get("role") != CLASS_FIDELITY_REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("git") != _evaluator_report_git(execution)
    ):
        raise ValueError(f"{method}/{case_id} class-fidelity report contract differs")
    validate_class_fidelity_report(dict(report))
    if (
        _runtime_sha(report, label=f"{method}/{case_id} class fidelity")
        != execution["runtime_environment_sha256s"]["class_fidelity"]
    ):
        raise ValueError(f"{method}/{case_id} class-fidelity runtime is unauthorized")
    _validate_sample_provenance(
        report.get("sample_provenance"),
        sampling=sampling,
        execution=execution,
        method=method,
        case_id=case_id,
        label="class fidelity",
    )
    paths = report.get("paths")
    classifier = report.get("classifier")
    expected_classifier = execution["source_bindings"]["classifier"]
    if (
        not isinstance(paths, Mapping)
        or paths.get("sampling_report") != sampling["identity"]["path"]
        or paths.get("generated_dir") != sampling["generated_dir"]
        or not isinstance(classifier, Mapping)
        or classifier.get("weights_path") != expected_classifier["path"]
        or classifier.get("weights_bytes") != expected_classifier["bytes"]
        or classifier.get("weights_sha256") != expected_classifier["sha256"]
    ):
        raise ValueError(f"{method}/{case_id} class-fidelity source differs")
    metrics = report["metrics"]
    if metrics.get("sample_count") != 1_000:
        raise ValueError(f"{method}/{case_id} class-fidelity count differs")
    return {
        "identity": identity,
        "class_top1": _metric(metrics["top1_accuracy"], name="class_top1"),
        "class_top5": _metric(metrics["top5_accuracy"], name="class_top5"),
    }


def _validate_generated_artifact_source(
    source: Mapping[str, Any],
    *,
    sampling: Mapping[str, Any],
    execution: Mapping[str, Any],
    method: str,
    case_id: str,
) -> dict[str, Any]:
    identity, report = _report_source(source, label="artifact report")
    normalized = validate_epsilon_stability_artifact_report(
        report,
        expected_source_kind="generated",
        expected_git=_evaluator_report_git(execution),
        expected_runtime_environment_sha256=execution[
            "runtime_environment_sha256s"
        ]["artifact"],
    )
    expected_binding = {
        "sampling_report": sampling["identity"],
        "generated_dir": sampling["generated_dir"],
        "sample_set_sha256": sampling["sample_set_sha256"],
    }
    if (
        normalized["sample_count"] != 1_000
        or normalized["sample_set_sha256"] != sampling["sample_set_sha256"]
        or normalized["source_binding"] != expected_binding
    ):
        raise ValueError(f"{method}/{case_id} artifact source differs")
    return {"identity": identity, "metrics": normalized["metrics"]}


def build_epsilon_stability_case_observation(
    *,
    design: Mapping[str, Any],
    design_identity: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    execution_authorization_identity: Mapping[str, Any],
    case_id: str,
    method: str,
    report_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if design.get("schema") != EPSILON_STABILITY_SAMPLING_DESIGN_SCHEMA:
        raise ValueError("epsilon-stability sampling design schema mismatch")
    if dict(design) != build_epsilon_stability_sampling_design():
        raise ValueError("epsilon-stability sampling design is not canonical")
    normalized_design_identity = _identity(
        design_identity, label="sampling design"
    )
    normalized_execution_identity = _identity(
        execution_authorization_identity,
        label="execution authorization",
    )
    execution = _validate_execution_authorization(
        execution_authorization,
        design_identity=normalized_design_identity,
        design=design,
    )
    if not isinstance(report_sources, Mapping) or set(report_sources) != set(
        REPORT_IDENTITY_KEYS
    ):
        raise ValueError("epsilon-stability native source-report set is incomplete")
    expected_sampling = materialize_epsilon_stability_case_protocol(
        dict(design),
        case_id,
        method,
        seed=int(execution["random_stream"]["seed"]),
        start_index=int(execution["random_stream"]["start_index"]),
    )
    sampling = _validate_sampling_report(
        report_sources["sampling_report"],
        expected_sampling=expected_sampling,
        execution=execution,
        method=method,
        case_id=case_id,
    )
    distribution = _validate_metrics_report(
        report_sources["metrics_report"],
        sampling=sampling,
        execution=execution,
        method=method,
        case_id=case_id,
    )
    class_fidelity = _validate_class_fidelity_source(
        report_sources["class_fidelity_report"],
        sampling=sampling,
        execution=execution,
        method=method,
        case_id=case_id,
    )
    artifact = _validate_generated_artifact_source(
        report_sources["artifact_report"],
        sampling=sampling,
        execution=execution,
        method=method,
        case_id=case_id,
    )
    metrics = {
        "fid": distribution["fid"],
        "inception_score": distribution["inception_score"],
        "class_top1": class_fidelity["class_top1"],
        "class_top5": class_fidelity["class_top5"],
        **artifact["metrics"],
    }
    return {
        "schema": EPSILON_STABILITY_CASE_OBSERVATION_SCHEMA,
        "status": "pass",
        "design_identity": normalized_design_identity,
        "execution_authorization_identity": normalized_execution_identity,
        "case_id": case_id,
        "method": method,
        "sample_count": 1_000,
        "sample_set_sha256": sampling["sample_set_sha256"],
        "sampling": expected_sampling,
        "checkpoint": execution["methods"][method]["checkpoint"],
        "integrity_sidecar": execution["methods"][method][
            "integrity_sidecar"
        ],
        "real_set": distribution["real_set"],
        "source_reports": {
            "sampling_report": sampling["identity"],
            "metrics_report": distribution["identity"],
            "class_fidelity_report": class_fidelity["identity"],
            "artifact_report": artifact["identity"],
        },
        "metrics": _metrics(metrics, label=f"{method}/{case_id}"),
        "authorization_boundary": dict(
            EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
        ),
    }


def _validate_observation(
    source: Mapping[str, Any],
    *,
    design: Mapping[str, Any],
    design_identity: dict[str, Any],
    execution_authorization: Mapping[str, Any],
    execution_authorization_identity: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(source, Mapping) or set(source) != {
        "identity",
        "payload",
        "reports",
    }:
        raise ValueError("case observation source is malformed")
    identity = _identity(source["identity"], label="case observation")
    payload = source["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("case observation payload is missing")
    case_id = str(payload.get("case_id", ""))
    method = str(payload.get("method", ""))
    expected = build_epsilon_stability_case_observation(
        design=design,
        design_identity=design_identity,
        execution_authorization=execution_authorization,
        execution_authorization_identity=execution_authorization_identity,
        case_id=case_id,
        method=method,
        report_sources=source["reports"],
    )
    if dict(payload) != expected:
        raise ValueError(f"{method}/{case_id} observation does not replay")
    return {"identity": identity, **expected}


def build_epsilon_stability_observation_manifest(
    *,
    design_identity: Mapping[str, Any],
    execution_authorization_identity: Mapping[str, Any],
    observation_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_design_identity = _identity(
        design_identity, label="sampling design"
    )
    normalized_execution_identity = _identity(
        execution_authorization_identity,
        label="execution authorization",
    )
    rows = []
    for source in observation_sources:
        if not isinstance(source, Mapping) or set(source) != {
            "identity",
            "payload",
        }:
            raise ValueError("observation manifest source is malformed")
        identity = _identity(source["identity"], label="case observation")
        payload = source["payload"]
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema")
            != EPSILON_STABILITY_CASE_OBSERVATION_SCHEMA
            or payload.get("status") != "pass"
            or payload.get("design_identity") != normalized_design_identity
            or payload.get("execution_authorization_identity")
            != normalized_execution_identity
        ):
            raise ValueError("observation manifest source contract differs")
        source_reports = payload.get("source_reports")
        if not isinstance(source_reports, Mapping) or set(source_reports) != set(
            REPORT_IDENTITY_KEYS
        ):
            raise ValueError("observation manifest source reports are incomplete")
        rows.append(
            {
                "case_id": str(payload.get("case_id", "")),
                "method": str(payload.get("method", "")),
                "observation": identity,
                "source_reports": {
                    key: _identity(
                        source_reports[key],
                        label=f"observation manifest {key}",
                    )
                    for key in REPORT_IDENTITY_KEYS
                },
            }
        )
    rows.sort(key=lambda row: (row["case_id"], row["method"]))
    keys = [(row["case_id"], row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("observation manifest contains duplicate rows")
    return {
        "schema": EPSILON_STABILITY_OBSERVATION_MANIFEST_SCHEMA,
        "status": "pass",
        "design_identity": normalized_design_identity,
        "execution_authorization_identity": normalized_execution_identity,
        "observation_count": len(rows),
        "observations": rows,
        "authorization_boundary": dict(
            EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
        ),
    }


def _validate_observation_manifest(
    source: Mapping[str, Any],
    *,
    design_identity: Mapping[str, Any],
    execution_authorization_identity: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identity, payload = _report_source(source, label="observation manifest")
    expected = build_epsilon_stability_observation_manifest(
        design_identity=design_identity,
        execution_authorization_identity=execution_authorization_identity,
        observation_sources=[
            {
                "identity": observation["identity"],
                "payload": observation,
            }
            for observation in observations
        ],
    )
    if dict(payload) != expected:
        raise ValueError("epsilon-stability observation manifest does not replay")
    return {"identity": identity, "payload": expected}


def build_epsilon_stability_real_artifact_reference(
    *,
    design: Mapping[str, Any],
    design_identity: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    execution_authorization_identity: Mapping[str, Any],
    artifact_report_source: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_design_identity = _identity(
        design_identity, label="sampling design"
    )
    normalized_execution_identity = _identity(
        execution_authorization_identity,
        label="execution authorization",
    )
    execution = _validate_execution_authorization(
        execution_authorization,
        design_identity=normalized_design_identity,
        design=design,
    )
    artifact_identity, artifact_report = _report_source(
        artifact_report_source,
        label="real artifact report",
    )
    normalized = validate_epsilon_stability_artifact_report(
        artifact_report,
        expected_source_kind="real_reference",
        expected_git=_evaluator_report_git(execution),
        expected_runtime_environment_sha256=execution[
            "runtime_environment_sha256s"
        ]["artifact"],
    )
    source_binding = normalized["source_binding"]
    if not isinstance(source_binding, Mapping) or set(source_binding) != {
        "real_set_identity",
        "real_set",
        "subset_manifest_identity",
    }:
        raise ValueError("real artifact source binding is incomplete")
    real_set_identity = _identity(
        source_binding["real_set_identity"], label="authorized real set"
    )
    subset_manifest_identity = _identity(
        source_binding["subset_manifest_identity"],
        label="real artifact subset manifest",
    )
    if real_set_identity != execution["source_bindings"]["real_set"]:
        raise ValueError("real artifact reference binds another real set")
    if normalized["sample_count"] != 1_000:
        raise ValueError("real artifact reference must contain exactly 1000 samples")
    normalized_real_set = _real_set(
        source_binding["real_set"], label="real artifact reference"
    )
    if normalized_real_set != execution["real_set"]:
        raise ValueError("real artifact reference content differs from authorization")
    return {
        "schema": EPSILON_STABILITY_REAL_ARTIFACT_REFERENCE_SCHEMA,
        "status": "pass",
        "design_identity": normalized_design_identity,
        "execution_authorization_identity": normalized_execution_identity,
        "real_set_identity": real_set_identity,
        "real_set": normalized_real_set,
        "sample_count": 1_000,
        "sample_set_sha256": normalized["sample_set_sha256"],
        "source_reports": {
            "artifact_report": artifact_identity,
            "subset_manifest": subset_manifest_identity,
        },
        "metrics": normalized["metrics"],
        "authorization_boundary": dict(
            EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
        ),
    }


def _validate_real_reference(
    source: Mapping[str, Any],
    *,
    design: Mapping[str, Any],
    design_identity: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    execution_authorization_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(source, Mapping) or set(source) != {
        "identity",
        "payload",
        "artifact_report",
    }:
        raise ValueError("real artifact reference source is malformed")
    identity = _identity(source["identity"], label="real artifact reference")
    payload = source["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("real artifact reference payload is missing")
    expected = build_epsilon_stability_real_artifact_reference(
        design=design,
        design_identity=design_identity,
        execution_authorization=execution_authorization,
        execution_authorization_identity=execution_authorization_identity,
        artifact_report_source=source["artifact_report"],
    )
    if dict(payload) != expected:
        raise ValueError("real artifact reference does not replay")
    return {"identity": identity, **expected}


def _comparison(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    real_artifact_metrics: Mapping[str, float],
) -> dict[str, Any]:
    candidate_metrics = candidate["metrics"]
    reference_metrics = reference["metrics"]
    artifact_distance = {
        name: abs(candidate_metrics[name] - real_artifact_metrics[name])
        for name in EPSILON_STABILITY_ARTIFACT_METRIC_KEYS
    }
    reference_artifact_distance = {
        name: abs(reference_metrics[name] - real_artifact_metrics[name])
        for name in EPSILON_STABILITY_ARTIFACT_METRIC_KEYS
    }
    checks = {
        "fid_strictly_lower": candidate_metrics["fid"]
        < reference_metrics["fid"],
        "class_top1_non_regression": candidate_metrics["class_top1"]
        >= reference_metrics["class_top1"],
        "class_top5_non_regression": candidate_metrics["class_top5"]
        >= reference_metrics["class_top5"],
        **{
            f"{name}_real_distance_non_regression": artifact_distance[name]
            <= reference_artifact_distance[name]
            for name in EPSILON_STABILITY_ARTIFACT_METRIC_KEYS
        },
    }
    fid_relative_improvement = (
        (reference_metrics["fid"] - candidate_metrics["fid"])
        / reference_metrics["fid"]
        if reference_metrics["fid"] > 0.0
        else 0.0
    )
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "fid_relative_improvement": fid_relative_improvement,
        "metrics": candidate_metrics,
        "reference_metrics": reference_metrics,
        "artifact_distance_to_real": artifact_distance,
        "reference_artifact_distance_to_real": reference_artifact_distance,
    }


def build_epsilon_stability_sampling_result(
    *,
    design: Mapping[str, Any],
    design_identity: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    execution_authorization_identity: Mapping[str, Any],
    observation_sources: Sequence[Mapping[str, Any]],
    observation_manifest: Mapping[str, Any],
    real_artifact_reference: Mapping[str, Any],
) -> dict[str, Any]:
    if design.get("schema") != EPSILON_STABILITY_SAMPLING_DESIGN_SCHEMA:
        raise ValueError("epsilon-stability sampling design schema mismatch")
    normalized_design_identity = _identity(
        design_identity, label="sampling design"
    )
    normalized_execution_identity = _identity(
        execution_authorization_identity,
        label="execution authorization",
    )
    normalized_execution = _validate_execution_authorization(
        execution_authorization,
        design_identity=normalized_design_identity,
        design=design,
    )
    normalized_observations = [
        _validate_observation(
            source,
            design=design,
            design_identity=normalized_design_identity,
            execution_authorization=normalized_execution,
            execution_authorization_identity=normalized_execution_identity,
        )
        for source in observation_sources
    ]
    normalized_manifest = _validate_observation_manifest(
        observation_manifest,
        design_identity=normalized_design_identity,
        execution_authorization_identity=normalized_execution_identity,
        observations=normalized_observations,
    )
    real_reference = _validate_real_reference(
        real_artifact_reference,
        design=design,
        design_identity=normalized_design_identity,
        execution_authorization=normalized_execution,
        execution_authorization_identity=normalized_execution_identity,
    )
    expected_keys = {
        (str(case["case_id"]), method)
        for case in design["cases"]
        for method in METHODS
    }
    observations_by_key = {
        (row["case_id"], row["method"]): row
        for row in normalized_observations
    }
    if len(observations_by_key) != len(normalized_observations):
        raise ValueError("epsilon-stability observations contain duplicate rows")
    if set(observations_by_key) != expected_keys:
        raise ValueError("epsilon-stability observation matrix is incomplete")
    if any(
        row["real_set"] != real_reference["real_set"]
        for row in normalized_observations
    ):
        raise ValueError("epsilon-stability observations do not share the real set")

    baseline_case = "legacy_terminal_hard_clip"
    candidate_rows = []
    for case in design["cases"]:
        case_id = str(case["case_id"])
        if case_id == baseline_case:
            continue
        method_comparisons = {}
        for method in METHODS:
            candidate = observations_by_key[(case_id, method)]
            baseline = observations_by_key[(baseline_case, method)]
            parent = observations_by_key[(str(case["parent_case_id"]), method)]
            method_comparisons[method] = {
                "versus_legacy": _comparison(
                    candidate,
                    baseline,
                    real_artifact_metrics=real_reference["metrics"],
                ),
                "versus_parent": _comparison(
                    candidate,
                    parent,
                    real_artifact_metrics=real_reference["metrics"],
                ),
            }
        passes = all(
            method_comparisons[method]["versus_legacy"]["passes"]
            for method in METHODS
        )
        candidate_rows.append(
            {
                "case_id": case_id,
                "parent_case_id": case["parent_case_id"],
                "incremental_control": case["incremental_control"],
                "attribution_role": case["attribution_role"],
                "passes_shared_recovery_screen": passes,
                "methods": method_comparisons,
                "worst_method_fid_relative_improvement": min(
                    method_comparisons[method]["versus_legacy"]
                    ["fid_relative_improvement"]
                    for method in METHODS
                ),
            }
        )
    passing = [
        row for row in candidate_rows if row["passes_shared_recovery_screen"]
    ]
    passing.sort(
        key=lambda row: (
            -float(row["worst_method_fid_relative_improvement"]),
            str(row["case_id"]),
        )
    )
    selected_case_id = passing[0]["case_id"] if passing else None
    selection_status = (
        "candidate_identified_for_separately_authorized_10k_confirmation"
        if selected_case_id is not None
        else "no_shared_sampling_recovery_candidate"
    )
    return {
        "schema": EPSILON_STABILITY_RESULT_SCHEMA,
        "status": "pass",
        "scientific_status": "screening_only",
        "selection_status": selection_status,
        "selected_case_id": selected_case_id,
        "generation_advantage_proven": False,
        "source_bindings": {
            "design": normalized_design_identity,
            "execution_authorization": normalized_execution_identity,
            "preparation": normalized_execution["preparation_identity"],
            "observation_manifest": normalized_manifest["identity"],
            "real_artifact_reference": real_reference["identity"],
            "terminal_route_receipt": normalized_execution[
                "terminal_route_receipt_identity"
            ],
            "separate_execution_authorization": normalized_execution[
                "separate_execution_authorization_identity"
            ],
        },
        "execution": normalized_execution,
        "real_artifact_reference": {
            key: real_reference[key]
            for key in (
                "real_set_identity",
                "real_set",
                "sample_count",
                "sample_set_sha256",
                "source_reports",
                "metrics",
            )
        },
        "selection_contract": {
            "reference_case": baseline_case,
            "required_methods": list(METHODS),
            "required_per_method_checks": [
                "FID strictly lower than the matched legacy case",
                "class top-1 and top-5 do not regress",
                (
                    "all artifact-metric distances to the same real reference "
                    "do not regress"
                ),
            ],
            "inception_score_role": "descriptive_only_at_1000_samples",
            "candidate_ranking": (
                "descending worst-method relative FID improvement, then case_id"
            ),
            "single_factor_attribution_requires_parent_comparison": True,
        },
        "candidates": candidate_rows,
        "observations": normalized_observations,
        "claim_boundary": {
            "one_thousand_sample_screening_only": True,
            "selected_case_is_not_confirmed": selected_case_id is not None,
            "independent_matched_10000_confirmation_required": True,
            "min_snr_training_tested": False,
            "cofitok_generation_advantage_claim_allowed": False,
        },
        "authorization_boundary": dict(
            EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
        ),
    }
