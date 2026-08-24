from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from cofitok.environment import runtime_environment_sha256
from cofitok.generation.quality_repair import (
    EPSILON_STABILITY_SAMPLING_DESIGN_SCHEMA,
    build_epsilon_stability_sampling_design,
)
from cofitok.generation.quality_repair_artifact import (
    EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY,
    normalize_epsilon_stability_identity,
)
from cofitok.generation.quality_repair_result import (
    EPSILON_STABILITY_EXECUTION_ACTIONS,
    EPSILON_STABILITY_EXECUTION_AUTHORIZATION_SCHEMA,
    EPSILON_STABILITY_EXECUTION_SOURCE_BINDINGS,
    validate_epsilon_stability_execution_authorization,
)


EPSILON_STABILITY_PREPARATION_SCHEMA = (
    "cofitok_matched_epsilon_stability_preparation_v1"
)
EPSILON_STABILITY_USER_AUTHORIZATION_SCHEMA = (
    "cofitok_matched_epsilon_stability_user_authorization_receipt_v1"
)
EPSILON_STABILITY_RUNTIME_BINDING_SCHEMA = (
    "cofitok_matched_epsilon_stability_runtime_binding_v1"
)
EPSILON_STABILITY_SCOPE = (
    "matched_1000_sample_epsilon_stability_sampling_diagnostic_only"
)
EPSILON_STABILITY_USER_AUTHORIZATION_MESSAGE = (
    "Approve the non-authorizing matched 1000-sample sampling-recovery "
    "diagnostic only."
)
METHODS = ("cofitok", "dense_identity")
RUNTIME_ROLES = ("sampling", "metrics", "class_fidelity", "artifact")


def _identity(value: Any, *, label: str) -> dict[str, Any]:
    return normalize_epsilon_stability_identity(value, label=label)


def _source(
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"identity", "payload"}:
        raise ValueError(f"{label} source is malformed")
    identity = _identity(value["identity"], label=label)
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} payload is missing")
    return identity, payload


def _sha256(value: Any, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} SHA256 is malformed")
    return digest


def _repair_git(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "revision",
        "tree",
        "branch",
        "tracked_dirty",
    }:
        raise ValueError("epsilon-stability repair Git identity is malformed")
    revision = str(value["revision"])
    tree = str(value["tree"])
    branch = str(value["branch"])
    if (
        len(revision) != 40
        or len(tree) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or any(character not in "0123456789abcdef" for character in tree)
        or not branch
        or value["tracked_dirty"] is not False
    ):
        raise ValueError("epsilon-stability repair Git identity is malformed")
    return {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }


def _report_git(value: Any, *, label: str) -> dict[str, Any]:
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


def _real_set(value: Any) -> dict[str, Any]:
    candidate = value.get("real_set", value) if isinstance(value, Mapping) else value
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "digest_schema",
        "sha256",
        "root",
        "image_count",
    }:
        raise ValueError("epsilon-stability real-set contract is malformed")
    root = str(candidate["root"])
    count = candidate["image_count"]
    if (
        not PurePosixPath(root).is_absolute()
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1_000
        or not str(candidate["digest_schema"])
    ):
        raise ValueError("epsilon-stability real-set contract is malformed")
    return {
        "digest_schema": str(candidate["digest_schema"]),
        "sha256": _sha256(candidate["sha256"], label="real set"),
        "root": root,
        "image_count": count,
    }


def build_epsilon_stability_user_authorization_receipt(
    *,
    task_id: str,
    user_message: str,
) -> dict[str, Any]:
    if not task_id:
        raise ValueError("epsilon-stability authorization task id is missing")
    if user_message != EPSILON_STABILITY_USER_AUTHORIZATION_MESSAGE:
        raise ValueError("epsilon-stability user authorization wording differs")
    return {
        "schema": EPSILON_STABILITY_USER_AUTHORIZATION_SCHEMA,
        "status": "approved",
        "scope": EPSILON_STABILITY_SCOPE,
        "source": {
            "kind": "codex_user_message",
            "task_id": task_id,
            "message": user_message,
        },
        "authorized_actions": dict(EPSILON_STABILITY_EXECUTION_ACTIONS),
        "limitations": {
            "screening_only": True,
            "independent_10000_confirmation_requires_new_authorization": True,
            "generation_advantage_claim_allowed": False,
            "authorization_cannot_be_reused_for_another_output_root": True,
        },
    }


def validate_epsilon_stability_user_authorization_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if set(receipt) != {
        "schema",
        "status",
        "scope",
        "source",
        "authorized_actions",
        "limitations",
    }:
        raise ValueError("epsilon-stability user authorization fields differ")
    source = receipt.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "kind",
        "task_id",
        "message",
    }:
        raise ValueError("epsilon-stability user authorization source is malformed")
    expected = build_epsilon_stability_user_authorization_receipt(
        task_id=str(source["task_id"]),
        user_message=str(source["message"]),
    )
    if dict(receipt) != expected:
        raise ValueError("epsilon-stability user authorization contract differs")
    return expected


def _validate_terminal_route(
    *,
    decision_identity: dict[str, Any],
    decision: Mapping[str, Any],
    verification: Mapping[str, Any],
    reconciliation_identity: dict[str, Any],
    quality_bridge_result_identity: dict[str, Any],
) -> None:
    next_stage = decision.get("recommended_next_stage")
    boundary = decision.get("authorization_boundary")
    source_evidence = decision.get("source_evidence")
    if (
        decision.get("schema_version") != 1
        or decision.get("role")
        != "generation_100k_post_reconciliation_experiment_decision"
        or decision.get("status") != "completed"
        or decision.get("operational_status") != "pass"
        or decision.get("generation_advantage_proven") is not False
        or not isinstance(next_stage, Mapping)
        or next_stage.get("id")
        != "prepare_matched_100k_epsilon_stability_sampling_diagnostic"
        or next_stage.get("execution_ready") is not False
        or next_stage.get("gpu_execution_allowed") is not False
        or next_stage.get("sampling_launch_allowed") is not False
        or next_stage.get("training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
        or not isinstance(boundary, Mapping)
        or boundary.get("new_source_bound_execution_gate_required") is not True
        or boundary.get("sampling_launch_allowed") is not False
        or boundary.get("training_launch_allowed") is not False
        or boundary.get("process_signals_allowed") is not False
        or not isinstance(source_evidence, Mapping)
        or source_evidence.get("cross_protocol_reconciliation")
        != reconciliation_identity
        or source_evidence.get("quality_bridge_result")
        != quality_bridge_result_identity
    ):
        raise ValueError("epsilon-stability terminal route decision differs")
    if (
        verification.get("schema_version") != 1
        or verification.get("role")
        != "generation_100k_post_reconciliation_decision_verification"
        or verification.get("status") != "verified"
        or verification.get("decision") != decision_identity
        or verification.get("recommended_next_stage") != next_stage
        or verification.get("authorization_boundary") != boundary
    ):
        raise ValueError("epsilon-stability terminal route verification differs")


def _normalize_runtime_binding(
    source: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    identity, payload = _source(source, label="runtime binding")
    runtimes = payload.get("runtime_environment_sha256s")
    environments = payload.get("runtime_environments")
    if (
        set(payload) != {
            "schema",
            "status",
            "runtime_environments",
            "runtime_environment_sha256s",
        }
        or payload.get("schema") != EPSILON_STABILITY_RUNTIME_BINDING_SCHEMA
        or payload.get("status") != "pass"
        or not isinstance(runtimes, Mapping)
        or set(runtimes) != set(RUNTIME_ROLES)
        or not isinstance(environments, Mapping)
        or set(environments) != set(RUNTIME_ROLES)
    ):
        raise ValueError("epsilon-stability runtime binding differs")
    normalized = {
        role: _sha256(runtimes[role], label=f"{role} runtime environment")
        for role in RUNTIME_ROLES
    }
    for role in RUNTIME_ROLES:
        environment = environments[role]
        if (
            not isinstance(environment, Mapping)
            or runtime_environment_sha256(dict(environment)) != normalized[role]
        ):
            raise ValueError(
                f"epsilon-stability {role} runtime binding does not replay"
            )
    return identity, normalized


def _normalize_method(
    *,
    method: str,
    source: Any,
    design: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str, str]:
    if not isinstance(source, Mapping) or set(source) != {
        "checkpoint",
        "integrity_sidecar",
        "latest",
        "training_report",
    }:
        raise ValueError(f"{method} preparation source set is malformed")
    checkpoint = _identity(source["checkpoint"], label=f"{method} checkpoint")
    sidecar_identity, sidecar = _source(
        source["integrity_sidecar"], label=f"{method} checkpoint sidecar"
    )
    latest_identity, latest = _source(source["latest"], label=f"{method} latest")
    training_identity, training = _source(
        source["training_report"], label=f"{method} training report"
    )
    expected = design["matched_methods"][method]
    expected_checkpoint_name = PurePosixPath(checkpoint["path"]).name
    expected_sidecar_name = PurePosixPath(sidecar_identity["path"]).name
    training_git = training.get("git")
    dataset = training.get("dataset_provenance")
    runtime_sha256 = str(training.get("runtime_environment_sha256", ""))
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("step") != expected["checkpoint_step"]
        or sidecar.get("checkpoint") != expected_checkpoint_name
        or sidecar.get("checkpoint_bytes") != checkpoint["bytes"]
        or sidecar.get("checkpoint_sha256") != checkpoint["sha256"]
        or latest.get("schema_version") != 1
        or latest.get("step") != expected["checkpoint_step"]
        or latest.get("checkpoint") != expected_checkpoint_name
        or latest.get("integrity_manifest") != expected_sidecar_name
        or latest.get("checkpoint_bytes") != checkpoint["bytes"]
        or latest.get("checkpoint_sha256") != checkpoint["sha256"]
        or training.get("training_complete") is not True
        or training.get("completed_steps") != expected["checkpoint_step"]
        or training.get("target_steps") != expected["checkpoint_step"]
        or training.get("latest_checkpoint") != latest
        or not isinstance(training_git, Mapping)
        or training_git.get("dirty") is not False
        or not isinstance(dataset, Mapping)
        or dataset.get("status") != "pass"
        or dataset.get("formal") is not True
        or sidecar.get("dataset_identity_sha256")
        != dataset.get("identity_sha256")
        or latest.get("dataset_identity_sha256")
        != dataset.get("identity_sha256")
        or sidecar.get("runtime_environment_sha256") != runtime_sha256
        or latest.get("runtime_environment_sha256") != runtime_sha256
        or sidecar.get("git_revision") != training_git.get("revision")
        or latest.get("git_revision") != training_git.get("revision")
        or sidecar.get("git_branch") != training_git.get("branch")
        or latest.get("git_branch") != training_git.get("branch")
    ):
        raise ValueError(f"{method} 100K checkpoint lineage differs")
    decision_method = (
        decision.get("source_evidence", {})
        .get("terminal_methods", {})
        .get(method)
    )
    if (
        not isinstance(decision_method, Mapping)
        or decision_method.get("checkpoint_payload") != checkpoint
        or decision_method.get("checkpoint_sidecar") != sidecar_identity
        or decision_method.get("latest") != latest_identity
    ):
        raise ValueError(f"{method} terminal decision checkpoint binding differs")
    return (
        {
            "checkpoint_step": int(expected["checkpoint_step"]),
            "prefix_budget": int(expected["prefix_budget"]),
            "checkpoint": checkpoint,
            "integrity_sidecar": sidecar_identity,
            "latest": latest_identity,
            "training_report": training_identity,
        },
        str(training_git["revision"]),
        str(dataset["identity_sha256"]),
        runtime_sha256,
    )


def build_epsilon_stability_preparation(
    *,
    design_source: Mapping[str, Any],
    decision_source: Mapping[str, Any],
    decision_verification_source: Mapping[str, Any],
    reconciliation_source: Mapping[str, Any],
    quality_bridge_result_source: Mapping[str, Any],
    training_pair_report_source: Mapping[str, Any],
    method_sources: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    real_set_source: Mapping[str, Any],
    runtime_binding_source: Mapping[str, Any],
    evaluator_source: Mapping[str, Any],
    classifier_identity: Mapping[str, Any],
    classifier_report_source: Mapping[str, Any],
    repair_git: Mapping[str, Any],
    seed: int,
    random_stream_namespace: str,
    output_root: str,
) -> dict[str, Any]:
    design_identity, design = _source(design_source, label="sampling design")
    if (
        design.get("schema") != EPSILON_STABILITY_SAMPLING_DESIGN_SCHEMA
        or dict(design) != build_epsilon_stability_sampling_design()
    ):
        raise ValueError("epsilon-stability sampling design does not replay")
    decision_identity, decision = _source(
        decision_source, label="post-reconciliation decision"
    )
    verification_identity, verification = _source(
        decision_verification_source,
        label="post-reconciliation decision verification",
    )
    reconciliation_identity, reconciliation = _source(
        reconciliation_source, label="cross-protocol reconciliation"
    )
    quality_identity, quality = _source(
        quality_bridge_result_source, label="quality bridge result"
    )
    pair_identity, pair = _source(
        training_pair_report_source, label="training pair report"
    )
    if (
        reconciliation.get("role")
        != "generation_100k_cross_protocol_reconciliation"
        or reconciliation.get("status") != "completed"
        or reconciliation.get("operational_status") != "pass"
        or reconciliation.get("terminal_status") != "hold"
        or quality.get("role") != "stability_full_data_quality_bridge_result"
        or quality.get("status") != "completed"
        or quality.get("authorization_boundary", {}).get(
            "report_is_promotion_gate"
        )
        is not False
        or pair.get("status") != "pass"
        or pair.get("stage") != "complete"
        or pair.get("issues") != []
    ):
        raise ValueError("epsilon-stability terminal source status differs")
    _validate_terminal_route(
        decision_identity=decision_identity,
        decision=decision,
        verification=verification,
        reconciliation_identity=reconciliation_identity,
        quality_bridge_result_identity=quality_identity,
    )
    if not isinstance(method_sources, Mapping) or set(method_sources) != set(
        METHODS
    ):
        raise ValueError("epsilon-stability preparation method set is incomplete")
    normalized_methods = {}
    training_revisions = set()
    dataset_sha256s = set()
    training_runtime_sha256s = set()
    for method in METHODS:
        normalized, revision, dataset_sha256, runtime_sha256 = _normalize_method(
            method=method,
            source=method_sources[method],
            design=design,
            decision=decision,
        )
        normalized_methods[method] = normalized
        training_revisions.add(revision)
        dataset_sha256s.add(dataset_sha256)
        training_runtime_sha256s.add(runtime_sha256)
    if len(training_revisions) != 1 or len(dataset_sha256s) != 1:
        raise ValueError("epsilon-stability 100K training pair is not matched")
    dataset = _identity(dataset_identity, label="dataset manifest")
    first_training = method_sources["cofitok"]["training_report"]["payload"]
    provenance = first_training["dataset_provenance"]
    manifest = provenance["manifest"]
    expected_dataset_path = (
        PurePosixPath(str(provenance["dataset_root"]))
        / str(manifest["relative_path"])
    )
    if (
        PurePosixPath(dataset["path"]) != expected_dataset_path
        or dataset["bytes"] != manifest["bytes"]
        or dataset["sha256"] != manifest["sha256"]
        or provenance["identity_sha256"] not in dataset_sha256s
    ):
        raise ValueError("epsilon-stability dataset manifest differs")
    real_identity, real_payload = _source(real_set_source, label="real set")
    normalized_real_set = _real_set(real_payload)
    runtime_identity, runtimes = _normalize_runtime_binding(runtime_binding_source)
    evaluator, evaluator_payload = _source(
        evaluator_source, label="evaluator source manifest"
    )
    classifier = _identity(classifier_identity, label="classifier weights")
    classifier_report_identity, classifier_report = _source(
        classifier_report_source, label="classifier report"
    )
    classifier_payload = classifier_report.get("classifier")
    if (
        not isinstance(classifier_payload, Mapping)
        or classifier_payload.get("weights_path") != classifier["path"]
        or classifier_payload.get("weights_bytes") != classifier["bytes"]
        or classifier_payload.get("weights_sha256") != classifier["sha256"]
    ):
        raise ValueError("epsilon-stability classifier binding differs")
    normalized_git = _repair_git(repair_git)
    if (
        evaluator_payload.get("schema")
        != "cofitok_matched_epsilon_stability_evaluator_manifest_v1"
        or evaluator_payload.get("status") != "pass"
        or evaluator_payload.get("git") != normalized_git
        or not isinstance(evaluator_payload.get("sources"), Mapping)
        or not evaluator_payload["sources"]
    ):
        raise ValueError("epsilon-stability evaluator source manifest differs")
    evaluator_git = _report_git(
        {
            "revision": normalized_git["revision"],
            "branch": normalized_git["branch"],
            "tracked_dirty": False,
        },
        label="epsilon-stability evaluator",
    )
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed <= 0
        or not random_stream_namespace
    ):
        raise ValueError("epsilon-stability fresh random stream is invalid")
    output = PurePosixPath(output_root)
    if (
        not output.is_absolute()
        or not str(output).startswith(
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        )
        or any(
            output == PurePosixPath(identity["path"])
            or output in PurePosixPath(identity["path"]).parents
            for identity in (
                design_identity,
                decision_identity,
                verification_identity,
                reconciliation_identity,
                quality_identity,
                pair_identity,
                dataset,
                real_identity,
                runtime_identity,
                evaluator,
                classifier,
            )
        )
    ):
        raise ValueError("epsilon-stability output root is not isolated")
    source_bindings = {
        "post_reconciliation_decision": decision_identity,
        "post_reconciliation_verification": verification_identity,
        "cross_protocol_reconciliation": reconciliation_identity,
        "quality_bridge_result": quality_identity,
        "training_pair_report": pair_identity,
        "dataset": dataset,
        "real_set": real_identity,
        "runtime_environment": runtime_identity,
        "evaluator": evaluator,
        "classifier": classifier,
        "classifier_report": classifier_report_identity,
    }
    if set(source_bindings) != set(EPSILON_STABILITY_EXECUTION_SOURCE_BINDINGS):
        raise AssertionError("epsilon-stability source-binding implementation drift")
    return {
        "schema": EPSILON_STABILITY_PREPARATION_SCHEMA,
        "status": "prepared",
        "execution_ready": False,
        "scope": EPSILON_STABILITY_SCOPE,
        "design_identity": design_identity,
        "terminal_route_receipt_identity": decision_identity,
        "repair_git": normalized_git,
        "random_stream": {
            "seed": seed,
            "start_index": int(
                design["common_sampling_contract"]["random_stream"][
                    "start_index"
                ]
            ),
            "namespace": random_stream_namespace,
            "fresh": True,
        },
        "output_root": str(output),
        "output_root_non_overlapping": True,
        "evaluator_git": evaluator_git,
        "runtime_environment_sha256s": runtimes,
        "training_runtime_environment_sha256s": sorted(
            training_runtime_sha256s
        ),
        "real_set": normalized_real_set,
        "methods": normalized_methods,
        "source_bindings": source_bindings,
        "authorization_boundary": dict(
            EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
        ),
    }


def build_epsilon_stability_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    design: Mapping[str, Any],
    user_authorization: Mapping[str, Any],
    user_authorization_identity: Mapping[str, Any],
) -> dict[str, Any]:
    required_preparation_fields = {
        "schema",
        "status",
        "execution_ready",
        "scope",
        "design_identity",
        "terminal_route_receipt_identity",
        "repair_git",
        "random_stream",
        "output_root",
        "output_root_non_overlapping",
        "evaluator_git",
        "runtime_environment_sha256s",
        "training_runtime_environment_sha256s",
        "real_set",
        "methods",
        "source_bindings",
        "authorization_boundary",
    }
    if (
        set(preparation) != required_preparation_fields
        or preparation.get("schema") != EPSILON_STABILITY_PREPARATION_SCHEMA
        or preparation.get("status") != "prepared"
        or preparation.get("execution_ready") is not False
        or preparation.get("scope") != EPSILON_STABILITY_SCOPE
        or preparation.get("authorization_boundary")
        != EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
    ):
        raise ValueError("epsilon-stability preparation contract differs")
    normalized_user = validate_epsilon_stability_user_authorization_receipt(
        user_authorization
    )
    if normalized_user["scope"] != preparation["scope"]:
        raise ValueError("epsilon-stability approval binds another scope")
    candidate = {
        "schema": EPSILON_STABILITY_EXECUTION_AUTHORIZATION_SCHEMA,
        "status": "approved",
        "scope": EPSILON_STABILITY_SCOPE,
        "preparation_identity": _identity(
            preparation_identity, label="epsilon-stability preparation"
        ),
        "design_identity": preparation["design_identity"],
        "terminal_route_receipt_identity": preparation[
            "terminal_route_receipt_identity"
        ],
        "separate_execution_authorization_identity": _identity(
            user_authorization_identity,
            label="separate execution authorization",
        ),
        "git": preparation["repair_git"],
        "random_stream": preparation["random_stream"],
        "output_root": preparation["output_root"],
        "output_root_non_overlapping": preparation[
            "output_root_non_overlapping"
        ],
        "evaluator_git": preparation["evaluator_git"],
        "runtime_environment_sha256s": preparation[
            "runtime_environment_sha256s"
        ],
        "real_set": preparation["real_set"],
        "methods": preparation["methods"],
        "source_bindings": preparation["source_bindings"],
        "authorized_actions": dict(EPSILON_STABILITY_EXECUTION_ACTIONS),
    }
    return validate_epsilon_stability_execution_authorization(
        candidate,
        design_identity=preparation["design_identity"],
        design=design,
    )
