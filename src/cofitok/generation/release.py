from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.generation.artifact import (
    inference_export_manifest_path,
    verify_inference_artifact,
    verify_inference_export_manifest,
)
from cofitok.generation_authorization import validate_generation_gate_binding
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.authorization import validate_generation_training_authorization


GENERATION_RELEASE_RECEIPT_SCHEMA_VERSION = 1
GENERATION_RELEASE_RECEIPT_TYPE = "cofitok_generation_release_receipt"
_LARGE_SCALE_REQUIRED_CHECKS = (
    "ten_percent_matched_training",
    "controlled_revision_transition",
    "generation_storage_capacity",
    "full_training_operational_monitor",
    "scaling_promotion_gate",
    "full_matched_training",
    "full_training_runtime_environment",
    "full_training_checkpoint_code_provenance",
    "reproducible_full_checkpoint_files",
    "full_runtime_selection",
    "full_training_audits",
    "full_milestone_evaluations",
    "formal_50k_generation",
    "formal_sampling_runtime_selection",
    "deterministic_visual_quality_audit",
    "deployable_ema_inference_artifacts",
    "final_generation_gate",
    "final_comparison_report",
)
_STABILITY_REQUIRED_CHECKS = (
    "stability_5k_authorization",
    "stability_50k_monitor",
    "stability_50k_pair_summary",
    "stability_50k_checkpoint_integrity",
    "stability_scaling_gate",
    "stability_full_training_readiness",
    "stability_full_readiness_revision_bridge",
    "stability_full_launch_receipt",
    "stability_full_monitor",
    "stability_full_training_pair",
    "stability_full_runtime_selection",
    "stability_full_storage_capacity",
    "stability_full_milestones",
    "stability_full_formal_generation",
    "stability_full_runtime_and_visual",
    "stability_final_gate",
    "stability_strong_baseline_comparison",
    "stability_release_authorized_inference",
)
_CAPACITY_FULL_REQUIRED_CHECKS = (
    "capacity_full_training_supervisor_deployment",
    "capacity_full_training_completion",
    "capacity_full_training_integrity",
    "capacity_full_milestones",
    "capacity_full_posteval_supervisor_deployment",
    "capacity_full_posteval_result",
    "capacity_full_posteval_supervisor_status",
    "capacity_full_formal_generation",
    "capacity_full_runtime_and_visual",
    "capacity_full_final_gate",
    "capacity_full_strong_comparison",
    "capacity_full_release_authorized_inference",
)
_COMPLETION_PROFILES = {
    "large_scale_generation_v1": {
        "status": "complete",
        "check": "deployable_ema_inference_artifacts",
    },
    "stability_generation_system_v1": {
        "status": "pass",
        "check": "stability_release_authorized_inference",
    },
    "capacity_full_generation_system_v1": {
        "status": "pass",
        "check": "capacity_full_release_authorized_inference",
    },
}
_COMPLETION_REQUIRED_CHECKS = {
    "large_scale_generation_v1": _LARGE_SCALE_REQUIRED_CHECKS,
    "stability_generation_system_v1": _STABILITY_REQUIRED_CHECKS,
    "capacity_full_generation_system_v1": _CAPACITY_FULL_REQUIRED_CHECKS,
}
_COMPLETION_EXPECTATION_FIELDS = {
    "large_scale_generation_v1": (
        "deployment_source",
        "ten_percent_training",
        "full_training",
    ),
    "stability_generation_system_v1": (
        "decision_source_revision",
        "scaling_training_revision",
        "scaling_evaluation_revision",
        "full_training_revision",
        "full_evaluation_revision",
        "export_revision",
        "decision_sha256",
        "scaling_gate_sha256",
        "full_readiness_sha256",
        "full_launch_receipt_sha256",
        "final_gate_sha256",
        "scaling_training_branch",
        "scaling_evaluation_branch",
        "full_training_branch",
        "full_evaluation_branch",
        "export_branch",
    ),
    "capacity_full_generation_system_v1": (
        "training_revision",
        "training_tree",
        "training_branch",
        "evaluation_revision",
        "evaluation_tree",
        "evaluation_branch",
        "export_revision",
        "export_branch",
        "training_supervisor_deployment_sha256",
        "posteval_supervisor_deployment_sha256",
        "training_launch_receipt_sha256",
        "final_gate_sha256",
    ),
}
_COMPLETION_SHA1_EXPECTATIONS = {
    "large_scale_generation_v1": _COMPLETION_EXPECTATION_FIELDS[
        "large_scale_generation_v1"
    ],
    "stability_generation_system_v1": (
        "decision_source_revision",
        "scaling_training_revision",
        "scaling_evaluation_revision",
        "full_training_revision",
        "full_evaluation_revision",
        "export_revision",
    ),
    "capacity_full_generation_system_v1": (
        "training_revision",
        "training_tree",
        "evaluation_revision",
        "evaluation_tree",
        "export_revision",
    ),
}
_COMPLETION_SHA256_EXPECTATIONS = {
    "large_scale_generation_v1": (),
    "stability_generation_system_v1": (
        "decision_sha256",
        "scaling_gate_sha256",
        "full_readiness_sha256",
        "full_launch_receipt_sha256",
        "final_gate_sha256",
    ),
    "capacity_full_generation_system_v1": (
        "training_supervisor_deployment_sha256",
        "posteval_supervisor_deployment_sha256",
        "training_launch_receipt_sha256",
        "final_gate_sha256",
    ),
}
_COMPLETION_BRANCH_EXPECTATIONS = {
    "large_scale_generation_v1": (),
    "stability_generation_system_v1": (
        "scaling_training_branch",
        "scaling_evaluation_branch",
        "full_training_branch",
        "full_evaluation_branch",
        "export_branch",
    ),
    "capacity_full_generation_system_v1": (
        "training_branch",
        "evaluation_branch",
        "export_branch",
    ),
}
_COMPLETION_ARTIFACT_BINDINGS = {
    "large_scale_generation_v1": {
        "source_revision": "full_training",
        "source_branch": None,
        "source_branch_value": "scale/generative-system",
        "execution_revision": None,
        "execution_branch": None,
        "training_gate_sha256": None,
        "training_gate_check": None,
        "training_stage": "scaling",
        "training_decision": "promote_to_full_imagenet256",
        "final_gate_sha256": None,
        "final_gate_check": "final_generation_gate",
    },
    "stability_generation_system_v1": {
        "source_revision": "full_training_revision",
        "source_branch": "full_training_branch",
        "source_branch_value": None,
        "execution_revision": "export_revision",
        "execution_branch": "export_branch",
        "training_gate_sha256": "scaling_gate_sha256",
        "training_gate_check": "stability_scaling_gate",
        "training_stage": "scaling",
        "training_decision": "promote_to_full_imagenet256",
        "final_gate_sha256": "final_gate_sha256",
        "final_gate_check": "stability_final_gate",
    },
    "capacity_full_generation_system_v1": {
        "source_revision": "training_revision",
        "source_branch": "training_branch",
        "source_branch_value": None,
        "execution_revision": "export_revision",
        "execution_branch": "export_branch",
        "training_gate_sha256": "training_launch_receipt_sha256",
        "training_gate_check": None,
        "training_stage": "capacity_full_experimental",
        "training_decision": "authorize_fresh_matched_300k_training",
        "final_gate_sha256": "final_gate_sha256",
        "final_gate_check": "capacity_full_final_gate",
    },
}
_COMPLETION_FORMAL_GENERATION_BINDINGS = {
    "large_scale_generation_v1": ("formal_50k_generation", None),
    "stability_generation_system_v1": (
        "stability_full_formal_generation",
        "methods",
    ),
    "capacity_full_generation_system_v1": (
        "capacity_full_formal_generation",
        "methods",
    ),
}
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_METHODS = ("cofitok", "dense_identity")


def _read_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{name} is missing: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"release source is missing: {source}")
    return {
        "path": source.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def _validated_completion_checks(
    audit: Mapping[str, Any],
    *,
    profile: str,
    contract: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    checks = audit.get("checks")
    if not isinstance(checks, list):
        raise ValueError("generation completion audit check list is missing")
    expected = _COMPLETION_REQUIRED_CHECKS[profile]
    rows: dict[str, Mapping[str, Any]] = {}
    names: list[str] = []
    for index, row in enumerate(checks):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"generation completion audit check row {index} is malformed"
            )
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"generation completion audit check row {index} lacks a name"
            )
        if name in rows:
            raise ValueError(
                f"generation completion audit contains duplicate check: {name}"
            )
        names.append(name)
        rows[name] = row

    missing = [name for name in expected if name not in rows]
    if missing:
        raise ValueError(
            "generation completion audit is missing required checks: "
            + ", ".join(missing)
        )
    unknown = [name for name in names if name not in expected]
    if unknown:
        raise ValueError(
            "generation completion audit contains unknown checks: "
            + ", ".join(unknown)
        )
    if tuple(names) != expected:
        raise ValueError(
            f"generation completion audit check order differs from {profile} contract"
        )
    non_passing = [name for name in expected if rows[name].get("status") != "pass"]
    if non_passing:
        raise ValueError(
            "generation completion audit requires every check to pass: "
            + ", ".join(non_passing)
        )
    missing_evidence = [
        name
        for name in expected
        if not isinstance(rows[name].get("evidence"), Mapping)
    ]
    if missing_evidence:
        raise ValueError(
            "generation completion audit lacks structured check evidence: "
            + ", ".join(missing_evidence)
        )
    return rows


def _completion_profile(
    audit: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Mapping[str, Any]]]:
    raw_profile = audit.get("profile")
    profile = (
        "large_scale_generation_v1"
        if raw_profile is None
        else str(raw_profile)
    )
    contract = _COMPLETION_PROFILES.get(profile)
    if contract is None:
        raise ValueError(f"unsupported generation completion profile: {profile}")
    if audit.get("schema_version") != 1:
        raise ValueError("unsupported generation completion audit schema")
    if (
        audit.get("status") != contract["status"]
        or audit.get("complete") is not True
        or audit.get("failed_checks") != []
        or audit.get("missing_checks") != []
    ):
        raise ValueError("generation completion audit did not pass")
    checks = _validated_completion_checks(
        audit,
        profile=profile,
        contract=contract,
    )
    return profile, contract, checks


def _validated_completion_expectations(
    profile: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    raw = (
        audit.get("expected_revisions")
        if profile == "large_scale_generation_v1"
        else audit.get("expectations")
    )
    if not isinstance(raw, Mapping):
        raise ValueError("generation completion expectations are missing")
    expected_fields = _COMPLETION_EXPECTATION_FIELDS[profile]
    if set(raw) != set(expected_fields):
        raise ValueError(
            f"generation completion expectations differ from {profile} contract"
        )
    for name in _COMPLETION_SHA1_EXPECTATIONS[profile]:
        value = raw.get(name)
        if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
            raise ValueError(f"generation completion expectation {name} is invalid")
    for name in _COMPLETION_SHA256_EXPECTATIONS[profile]:
        value = raw.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"generation completion expectation {name} is invalid")
    for name in _COMPLETION_BRANCH_EXPECTATIONS[profile]:
        if not isinstance(raw.get(name), str) or not raw[name]:
            raise ValueError(f"generation completion expectation {name} is invalid")
    return {name: raw[name] for name in expected_fields}


def _validate_completion_contract_evidence(
    profile: str,
    expectations: Mapping[str, Any],
    checks: Mapping[str, Mapping[str, Any]],
) -> None:
    def evidence(name: str) -> Mapping[str, Any]:
        value = checks[name].get("evidence")
        if not isinstance(value, Mapping):
            raise ValueError(
                f"generation completion {name} evidence is malformed"
            )
        return value

    if profile == "large_scale_generation_v1":
        scaling = evidence("ten_percent_matched_training")
        transition = evidence("controlled_revision_transition")
        full_training = evidence("full_matched_training")
        if scaling.get("expected_revision") != expectations["ten_percent_training"]:
            raise ValueError(
                "generation completion ten-percent training revision differs"
            )
        if (
            transition.get("training_revision")
            != expectations["deployment_source"]
            or transition.get("target_revision") != expectations["full_training"]
        ):
            raise ValueError("generation completion deployment revision differs")
        if (
            full_training.get("expected_revision")
            != expectations["full_training"]
            or full_training.get("expected_branch") != "scale/generative-system"
        ):
            raise ValueError("generation completion full-training Git differs")
        return

    if profile == "stability_generation_system_v1":
        decision = evidence("stability_5k_authorization")
        readiness = evidence("stability_full_training_readiness")
        launch = evidence("stability_full_launch_receipt")
        if (
            decision.get("source_revision")
            != expectations["decision_source_revision"]
            or decision.get("decision_sha256") != expectations["decision_sha256"]
        ):
            raise ValueError("generation completion stability decision differs")
        if readiness.get("readiness_sha256") != expectations["full_readiness_sha256"]:
            raise ValueError("generation completion stability readiness differs")
        if (
            launch.get("launch_receipt_sha256")
            != expectations["full_launch_receipt_sha256"]
            or launch.get("readiness_sha256")
            != expectations["full_readiness_sha256"]
        ):
            raise ValueError("generation completion stability launch receipt differs")
        return

    training_git = {
        "revision": expectations["training_revision"],
        "tree": expectations["training_tree"],
        "branch": expectations["training_branch"],
        "tracked_dirty": False,
    }
    evaluation_git = {
        "revision": expectations["evaluation_revision"],
        "tree": expectations["evaluation_tree"],
        "branch": expectations["evaluation_branch"],
        "tracked_dirty": False,
    }
    training_deployment = evidence("capacity_full_training_supervisor_deployment")
    training_completion = evidence("capacity_full_training_completion")
    posteval_deployment = evidence("capacity_full_posteval_supervisor_deployment")
    training_identity = training_deployment.get("identity")
    posteval_identity = posteval_deployment.get("identity")
    training_receipt = training_completion.get("training_launch_receipt")
    training_authorization = training_completion.get("authorization")
    if (
        not isinstance(training_identity, Mapping)
        or training_identity.get("sha256")
        != expectations["training_supervisor_deployment_sha256"]
        or training_deployment.get("git") != training_git
    ):
        raise ValueError(
            "generation completion capacity training deployment differs"
        )
    if (
        not isinstance(posteval_identity, Mapping)
        or posteval_identity.get("sha256")
        != expectations["posteval_supervisor_deployment_sha256"]
        or posteval_deployment.get("git") != evaluation_git
    ):
        raise ValueError(
            "generation completion capacity post-eval deployment differs"
        )
    if (
        not isinstance(training_receipt, Mapping)
        or training_receipt.get("sha256")
        != expectations["training_launch_receipt_sha256"]
        or not isinstance(training_authorization, Mapping)
        or training_authorization.get("gate_sha256")
        != expectations["training_launch_receipt_sha256"]
    ):
        raise ValueError(
            "generation completion capacity training authorization differs"
        )
    expected_receipt_identity = {
        "path": training_authorization.get("gate_path"),
        "bytes": training_authorization.get("gate_bytes"),
        "sha256": training_authorization.get("gate_sha256"),
    }
    if dict(training_receipt) != expected_receipt_identity:
        raise ValueError(
            "generation completion capacity training receipt identity differs"
        )


def _validate_completion_artifact_provenance(
    profile: str,
    expectations: Mapping[str, Any],
    checks: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> None:
    rows = [artifacts[method] for method in _METHODS]
    if len({str(row["artifact_sha256"]) for row in rows}) != len(_METHODS):
        raise ValueError("generation completion artifacts must contain distinct bytes")
    if len({str(row["source_checkpoint_sha256"]) for row in rows}) != len(
        _METHODS
    ):
        raise ValueError(
            "generation completion artifacts must use distinct source checkpoints"
        )
    shared_fields = (
        "source_git",
        "source_runtime_environment_sha256",
        "training_authorization",
        "release_authorization",
        "execution_git",
        "export_runtime_environment_sha256",
        "execution_runtime_environment_sha256",
    )
    for name in shared_fields:
        if rows[0].get(name) != rows[1].get(name):
            raise ValueError(
                f"generation completion artifacts used different {name}"
            )

    formal_check, methods_field = _COMPLETION_FORMAL_GENERATION_BINDINGS[profile]
    formal_evidence = checks[formal_check].get("evidence")
    if not isinstance(formal_evidence, Mapping):
        raise ValueError("generation completion formal generation evidence is malformed")
    formal_methods = (
        formal_evidence
        if methods_field is None
        else formal_evidence.get(methods_field)
    )
    if not isinstance(formal_methods, Mapping) or set(formal_methods) != set(
        _METHODS
    ):
        raise ValueError("generation completion formal generation evidence is incomplete")
    for method, row in zip(_METHODS, rows):
        method_evidence = formal_methods.get(method)
        if (
            not isinstance(method_evidence, Mapping)
            or method_evidence.get("checkpoint_sha256")
            != row["source_checkpoint_sha256"]
        ):
            raise ValueError(
                f"generation completion {method} evaluated checkpoint differs"
            )

    binding = _COMPLETION_ARTIFACT_BINDINGS[profile]
    source_git = rows[0].get("source_git")
    if (
        not isinstance(source_git, Mapping)
        or set(source_git) != {"revision", "branch", "dirty"}
        or source_git.get("revision")
        != expectations[binding["source_revision"]]
        or source_git.get("dirty") is not False
    ):
        raise ValueError("generation completion artifact source Git differs")
    source_branch_field = binding["source_branch"]
    expected_source_branch = (
        binding["source_branch_value"]
        if source_branch_field is None
        else expectations[source_branch_field]
    )
    if source_git.get("branch") != expected_source_branch:
        raise ValueError("generation completion artifact source branch differs")
    if not isinstance(source_git.get("branch"), str) or not source_git["branch"]:
        raise ValueError("generation completion artifact source branch is invalid")
    if _SHA256.fullmatch(
        str(rows[0].get("source_runtime_environment_sha256", ""))
    ) is None:
        raise ValueError("generation completion artifact source environment is invalid")

    execution_revision_field = binding["execution_revision"]
    execution_git = rows[0].get("execution_git")
    if execution_revision_field is not None:
        if (
            not isinstance(execution_git, Mapping)
            or set(execution_git) != {"revision", "branch", "tracked_dirty"}
            or execution_git.get("revision")
            != expectations[execution_revision_field]
            or execution_git.get("branch")
            != expectations[binding["execution_branch"]]
            or execution_git.get("tracked_dirty") is not False
        ):
            raise ValueError("generation completion artifact execution Git differs")
        for name in (
            "export_runtime_environment_sha256",
            "execution_runtime_environment_sha256",
        ):
            if _SHA256.fullmatch(str(rows[0].get(name, ""))) is None:
                raise ValueError(
                    f"generation completion artifact {name} is invalid"
                )
    elif execution_git is not None:
        if (
            not isinstance(execution_git, Mapping)
            or set(execution_git) != {"revision", "branch", "tracked_dirty"}
            or _SHA1.fullmatch(str(execution_git.get("revision", ""))) is None
            or not isinstance(execution_git.get("branch"), str)
            or not execution_git["branch"]
            or execution_git.get("tracked_dirty") is not False
        ):
            raise ValueError("generation completion artifact execution Git is invalid")

    training_authorization = rows[0].get("training_authorization")
    release_authorization = rows[0].get("release_authorization")
    if not isinstance(training_authorization, Mapping) or not isinstance(
        release_authorization,
        Mapping,
    ):
        raise ValueError("generation completion artifact authorization is missing")
    try:
        validated_training_authorization = validate_generation_training_authorization(
            training_authorization
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "generation completion training authorization is invalid"
        ) from error
    try:
        validated_release_authorization = validate_generation_gate_binding(
            release_authorization,
            expected_stage="full",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "generation completion release authorization is invalid"
        ) from error
    if (
        validated_training_authorization.get("stage") != binding["training_stage"]
        or validated_training_authorization.get("decision")
        != binding["training_decision"]
    ):
        raise ValueError(
            "generation completion training authorization identity differs"
        )
    if profile == "capacity_full_generation_system_v1" and (
        training_authorization
        != checks["capacity_full_training_completion"]["evidence"].get(
            "authorization"
        )
    ):
        raise ValueError(
            "generation completion capacity training authorization evidence differs"
        )
    if (
        validated_release_authorization.get("stage") != "full"
        or validated_release_authorization.get("decision")
        != "large_scale_generation_ready"
    ):
        raise ValueError(
            "generation completion release authorization identity differs"
        )
    training_gate_field = binding["training_gate_sha256"]
    if training_gate_field is not None and (
        training_authorization.get("gate_sha256")
        != expectations[training_gate_field]
    ):
        raise ValueError("generation completion training gate differs")
    training_gate_check = binding["training_gate_check"]
    if training_gate_field is not None and training_gate_check is not None:
        training_gate_evidence = checks[training_gate_check].get("evidence")
        if (
            not isinstance(training_gate_evidence, Mapping)
            or training_gate_evidence.get("gate_sha256")
            != expectations[training_gate_field]
        ):
            raise ValueError(
                "generation completion training-gate evidence differs"
            )
        if profile == "stability_generation_system_v1" and (
            training_gate_evidence.get("provenance")
            != {
                "training_revision": expectations[
                    "scaling_training_revision"
                ],
                "training_branch": expectations["scaling_training_branch"],
                "evaluation_revision": expectations[
                    "scaling_evaluation_revision"
                ],
                "evaluation_branch": expectations[
                    "scaling_evaluation_branch"
                ],
            }
        ):
            raise ValueError(
                "generation completion training-gate provenance differs"
            )
    final_gate_field = binding["final_gate_sha256"]
    if final_gate_field is not None:
        expected_gate_sha = expectations[final_gate_field]
        if release_authorization.get("gate_sha256") != expected_gate_sha:
            raise ValueError("generation completion release gate differs")
        gate_evidence = checks[binding["final_gate_check"]].get("evidence")
        if (
            not isinstance(gate_evidence, Mapping)
            or gate_evidence.get("gate_sha256") != expected_gate_sha
        ):
            raise ValueError("generation completion final-gate evidence differs")
        expected_provenance = {
            "training_revision": expectations[binding["source_revision"]],
            "training_branch": expectations[binding["source_branch"]],
            "evaluation_revision": expectations[
                "full_evaluation_revision"
                if profile == "stability_generation_system_v1"
                else "evaluation_revision"
            ],
            "evaluation_branch": expectations[
                "full_evaluation_branch"
                if profile == "stability_generation_system_v1"
                else "evaluation_branch"
            ],
        }
        if gate_evidence.get("provenance") != expected_provenance:
            raise ValueError("generation completion final-gate provenance differs")


def _artifact_receipt_row(method: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = evidence.get("artifact_path")
    if not isinstance(raw_path, str):
        raise ValueError(f"{method} completion artifact path is not absolute")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(f"{method} completion artifact path is not absolute")
    sha256 = evidence.get("artifact_sha256")
    source_checkpoint_sha256 = evidence.get("source_checkpoint_sha256")
    artifact_bytes = evidence.get("artifact_bytes")
    source_checkpoint_bytes = evidence.get("source_checkpoint_bytes")
    export_manifest = evidence.get("export_manifest")
    smoke_sha256 = evidence.get("smoke_output_sha256")
    smoke_output_count = evidence.get("smoke_output_count")
    if (
        not isinstance(sha256, str)
        or _SHA256.fullmatch(sha256) is None
        or not isinstance(source_checkpoint_sha256, str)
        or _SHA256.fullmatch(source_checkpoint_sha256) is None
        or type(artifact_bytes) is not int
        or artifact_bytes < 1
        or type(source_checkpoint_bytes) is not int
        or source_checkpoint_bytes < 1
        or not isinstance(evidence.get("source_git"), Mapping)
        or not isinstance(evidence.get("training_authorization"), Mapping)
        or not isinstance(evidence.get("release_authorization"), Mapping)
        or not isinstance(export_manifest, Mapping)
        or set(export_manifest) != {"path", "bytes", "sha256"}
        or not isinstance(export_manifest.get("path"), str)
        or not Path(str(export_manifest.get("path", ""))).is_absolute()
        or type(export_manifest.get("bytes")) is not int
        or export_manifest["bytes"] < 1
        or not isinstance(export_manifest.get("sha256"), str)
        or _SHA256.fullmatch(export_manifest["sha256"]) is None
        or type(smoke_output_count) is not int
        or smoke_output_count < 1
        or not isinstance(smoke_sha256, list)
        or len(smoke_sha256) != smoke_output_count
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in smoke_sha256
        )
    ):
        raise ValueError(f"{method} completion artifact evidence is malformed")
    return {
        "path": path.resolve().as_posix(),
        "artifact_sha256": sha256,
        "artifact_bytes": artifact_bytes,
        "checkpoint_step": 300_000,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_checkpoint_bytes": source_checkpoint_bytes,
        "source_runtime_environment_sha256": evidence.get(
            "source_runtime_environment_sha256"
        ),
        "source_git": dict(evidence["source_git"]),
        "execution_git": evidence.get("execution_git"),
        "export_runtime_environment_sha256": evidence.get(
            "export_runtime_environment_sha256"
        ),
        "execution_runtime_environment_sha256": evidence.get(
            "execution_runtime_environment_sha256"
        ),
        "training_authorization": dict(evidence["training_authorization"]),
        "release_authorization": dict(evidence["release_authorization"]),
        "export_manifest": dict(export_manifest),
        "smoke_output_count": smoke_output_count,
        "smoke_output_sha256": list(smoke_sha256),
    }


def _receipt_payload(
    completion_audit_path: str | Path,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    profile, contract, checks = _completion_profile(audit)
    evidence = checks[contract["check"]].get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != set(_METHODS):
        raise ValueError("generation completion inference evidence is incomplete")
    expectations = _validated_completion_expectations(profile, audit)
    _validate_completion_contract_evidence(profile, expectations, checks)
    artifacts = {
        method: _artifact_receipt_row(method, evidence[method])
        for method in _METHODS
    }
    if len({row["path"] for row in artifacts.values()}) != len(_METHODS):
        raise ValueError("generation completion artifacts must use distinct paths")
    _validate_completion_artifact_provenance(
        profile,
        expectations,
        checks,
        artifacts,
    )
    return {
        "schema_version": GENERATION_RELEASE_RECEIPT_SCHEMA_VERSION,
        "receipt_type": GENERATION_RELEASE_RECEIPT_TYPE,
        "status": "completed",
        "completion_profile": profile,
        "completion_audit": _file_identity(completion_audit_path),
        "completion_expectations": dict(expectations),
        "artifacts": artifacts,
    }


def _validate_artifact_against_row(
    method: str,
    row: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> None:
    source_git = {
        "revision": integrity.get("source_git_revision"),
        "branch": integrity.get("source_git_branch"),
        "dirty": integrity.get("source_git_dirty"),
    }
    if (
        row.get("artifact_sha256") != integrity.get("artifact_sha256")
        or int(row.get("artifact_bytes", -1))
        != int(integrity.get("artifact_bytes", -2))
        or int(row.get("checkpoint_step", -1)) != int(integrity.get("step", -2))
        or row.get("source_checkpoint_sha256")
        != integrity.get("source_checkpoint_sha256")
        or row.get("source_runtime_environment_sha256")
        != integrity.get("source_runtime_environment_sha256")
        or row.get("source_git") != source_git
        or row.get("training_authorization")
        != integrity.get("source_training_authorization")
        or row.get("release_authorization")
        != integrity.get("release_authorization")
    ):
        raise ValueError(f"{method} inference artifact differs from release receipt")


def _validate_manifest_against_row(
    method: str,
    row: Mapping[str, Any],
    *,
    verify_sources: bool,
) -> None:
    artifact = Path(str(row.get("path", ""))).resolve()
    expected_path = inference_export_manifest_path(artifact).resolve()
    descriptor = row.get("export_manifest")
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{method} release receipt lacks export manifest")
    actual_descriptor = _file_identity(expected_path)
    if dict(descriptor) != actual_descriptor:
        raise ValueError(f"{method} inference export manifest differs from release receipt")
    if verify_sources:
        manifest = verify_inference_export_manifest(
            expected_path,
            expected_artifact=artifact,
        )
    else:
        manifest = _read_object(
            expected_path,
            name=f"{method} inference export manifest",
        )
    source = manifest.get("source")
    target = manifest.get("target")
    execution = manifest.get("execution")
    training_authorization = row.get("training_authorization")
    expected_manifest_authorization = (
        None
        if not isinstance(training_authorization, Mapping)
        else {
            "authorization_stage": training_authorization.get("stage"),
            "authorization_decision": training_authorization.get("decision"),
            "authorization_gate_bytes": training_authorization.get("gate_bytes"),
            "authorization_gate_sha256": training_authorization.get("gate_sha256"),
            "authorization_gate_identity_sha256": training_authorization.get(
                "gate_identity_sha256"
            ),
        }
    )
    execution_git = None if not isinstance(execution, Mapping) else execution.get("git")
    execution_environment_sha = (
        None
        if not isinstance(execution, Mapping)
        else execution.get("runtime_environment_sha256")
    )
    if (
        not isinstance(source, Mapping)
        or source.get("sha256") != row.get("source_checkpoint_sha256")
        or int(source.get("bytes", -1))
        != int(row.get("source_checkpoint_bytes", -2))
        or int(source.get("step", -1)) != int(row.get("checkpoint_step", -2))
        or source.get("runtime_environment_sha256")
        != row.get("source_runtime_environment_sha256")
        or source.get("git") != row.get("source_git")
        or source.get("training_authorization")
        != expected_manifest_authorization
        or not isinstance(target, Mapping)
        or target.get("artifact") != artifact.as_posix()
        or manifest.get("release_authorization")
        != row.get("release_authorization")
        or execution_git != row.get("execution_git")
        or execution_environment_sha
        != row.get("export_runtime_environment_sha256")
    ):
        raise ValueError(f"{method} inference export manifest provenance differs")


def write_generation_release_receipt(
    completion_audit: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    audit = _read_object(completion_audit, name="generation completion audit")
    payload = _receipt_payload(completion_audit, audit)
    for method, row in payload["artifacts"].items():
        integrity = verify_inference_artifact(row["path"])
        _validate_artifact_against_row(method, row, integrity)
        _validate_manifest_against_row(method, row, verify_sources=True)
    target = Path(output)
    if target.exists():
        existing = _read_object(target, name="generation release receipt")
        if existing != payload:
            raise ValueError("existing generation release receipt differs")
        return existing
    write_json_report(target, payload)
    written = _read_object(target, name="generation release receipt")
    if written != payload:
        raise ValueError("written generation release receipt differs")
    return written


def verify_generation_release_receipt(
    receipt_path: str | Path,
    artifact_path: str | Path,
    *,
    artifact_integrity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = _read_object(receipt_path, name="generation release receipt")
    if (
        receipt.get("schema_version") != GENERATION_RELEASE_RECEIPT_SCHEMA_VERSION
        or receipt.get("receipt_type") != GENERATION_RELEASE_RECEIPT_TYPE
        or receipt.get("status") != "completed"
    ):
        raise ValueError("generation release receipt header is invalid")
    audit_descriptor = receipt.get("completion_audit")
    if not isinstance(audit_descriptor, Mapping):
        raise ValueError("generation release receipt lacks completion audit identity")
    audit_path = Path(str(audit_descriptor.get("path", "")))
    actual_audit_identity = _file_identity(audit_path)
    if dict(audit_descriptor) != actual_audit_identity:
        raise ValueError("generation completion audit changed after release")
    audit = _read_object(audit_path, name="generation completion audit")
    expected = _receipt_payload(audit_path, audit)
    if receipt != expected:
        raise ValueError("generation release receipt differs from completion audit")

    artifact = Path(artifact_path).resolve()
    matches = [
        (method, row)
        for method, row in receipt["artifacts"].items()
        if Path(str(row.get("path", ""))).resolve() == artifact
    ]
    if len(matches) != 1:
        raise ValueError("inference artifact is not authorized by release receipt")
    method, row = matches[0]
    integrity = (
        artifact_integrity
        if artifact_integrity is not None
        else verify_inference_artifact(artifact)
    )
    _validate_artifact_against_row(method, row, integrity)
    _validate_manifest_against_row(method, row, verify_sources=False)
    receipt_identity = _file_identity(receipt_path)
    return {
        "receipt": receipt_identity,
        "completion_audit": actual_audit_identity,
        "completion_profile": receipt["completion_profile"],
        "completion_expectations": receipt["completion_expectations"],
        "method": method,
    }
