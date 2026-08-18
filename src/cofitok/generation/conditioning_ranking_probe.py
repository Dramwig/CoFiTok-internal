from __future__ import annotations

import copy
from typing import Any, Mapping

from cofitok.generation_pair import generation_pair_contract


PROBE_SCHEMA_VERSION = 1
PROBE_ROLE = "generation_conditioning_ranking_four_arm_probe_preparation"
PROBE_SCOPE = "imagenet256_10pct_four_arm_class_ranking_probe1k_only"
PROBE_AUTHORIZATION_TEXT = (
    "Approve the non-authorizing four-arm 1000-step "
    "class-conditioning-ranking probe only."
)
RANKING_FIELDS = (
    "class_conditioning_ranking_weight",
    "class_conditioning_ranking_start_step",
    "class_conditioning_ranking_warmup_steps",
    "class_conditioning_ranking_batch_fraction",
    "class_conditioning_ranking_margin",
    "class_conditioning_ranking_wrong_label_offset",
    "class_conditioning_ranking_min_timestep",
)
CONTROL_RANKING_CONFIG = {
    "class_conditioning_ranking_weight": 0.0,
    "class_conditioning_ranking_start_step": 0,
    "class_conditioning_ranking_warmup_steps": 0,
    "class_conditioning_ranking_batch_fraction": 0.0625,
    "class_conditioning_ranking_margin": 0.0,
    "class_conditioning_ranking_wrong_label_offset": 1,
    "class_conditioning_ranking_min_timestep": 0,
}
RANKED_RANKING_CONFIG = {
    "class_conditioning_ranking_weight": 0.05,
    "class_conditioning_ranking_start_step": 100,
    "class_conditioning_ranking_warmup_steps": 200,
    "class_conditioning_ranking_batch_fraction": 0.0625,
    "class_conditioning_ranking_margin": 0.01,
    "class_conditioning_ranking_wrong_label_offset": 500,
    "class_conditioning_ranking_min_timestep": 500,
}
EXECUTION_BOUNDARY = {
    "training_allowed": True,
    "training_runs": [
        "control_cofitok",
        "control_dense_identity",
        "ranked_cofitok",
        "ranked_dense_identity",
    ],
    "steps_per_run": 1_000,
    "dataset": "imagenet_256_10pct",
    "sampling_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "release_authorization_allowed": False,
}


def _ranking_config(config: Mapping[str, Any]) -> dict[str, Any]:
    loss = config.get("loss")
    if not isinstance(loss, Mapping):
        return {}
    return {field: loss.get(field) for field in RANKING_FIELDS}


def _without_probe_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    normalized.pop("name", None)
    loss = normalized.get("loss")
    if isinstance(loss, dict):
        for field in RANKING_FIELDS:
            loss.pop(field, None)
    return normalized


def conditioning_ranking_probe_contract(
    *,
    control_cofitok: dict[str, Any],
    control_dense: dict[str, Any],
    ranked_cofitok: dict[str, Any],
    ranked_dense: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    pairs = {
        "control": generation_pair_contract(control_cofitok, control_dense),
        "ranked": generation_pair_contract(ranked_cofitok, ranked_dense),
    }
    for name, contract in pairs.items():
        issues.extend(f"{name}_pair: {issue}" for issue in contract["issues"])

    for label, config, expected in (
        ("control_cofitok", control_cofitok, CONTROL_RANKING_CONFIG),
        ("control_dense", control_dense, CONTROL_RANKING_CONFIG),
        ("ranked_cofitok", ranked_cofitok, RANKED_RANKING_CONFIG),
        ("ranked_dense", ranked_dense, RANKED_RANKING_CONFIG),
    ):
        actual = _ranking_config(config)
        if actual != expected:
            issues.append(f"{label} ranking config differs from the probe contract")
        runtime = config.get("runtime", {})
        data = config.get("data", {})
        optimization = config.get("optimization", {})
        if runtime.get("steps") != 1_000:
            issues.append(f"{label} runtime.steps is not 1000")
        if data.get("dataset") != "imagenet_256_10pct":
            issues.append(f"{label} dataset is not imagenet_256_10pct")
        if int(data.get("batch_size", 0)) * int(
            optimization.get("gradient_accumulation_steps", 0)
        ) != 64:
            issues.append(f"{label} effective batch size is not 64")

    for method, control, ranked in (
        ("cofitok", control_cofitok, ranked_cofitok),
        ("dense_identity", control_dense, ranked_dense),
    ):
        if _without_probe_fields(control) != _without_probe_fields(ranked):
            issues.append(
                f"{method} control/ranked configs differ outside the ranking fields"
            )

    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "pass" if not issues else "fail",
        "role": PROBE_ROLE,
        "scope": PROBE_SCOPE,
        "valid": not issues,
        "issues": issues,
        "ranking_fields": list(RANKING_FIELDS),
        "control_ranking_config": copy.deepcopy(CONTROL_RANKING_CONFIG),
        "ranked_ranking_config": copy.deepcopy(RANKED_RANKING_CONFIG),
        "pair_contracts": pairs,
        "execution_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
        "authorization_required": True,
        "gpu_execution_authorized": False,
    }


def validate_conditioning_ranking_probe_approval(
    approval: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_preparation_sha256: str,
    expected_output_root: str,
) -> dict[str, Any]:
    expected = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "role": "generation_conditioning_ranking_probe_execution_approval",
        "status": "approved",
        "scope": PROBE_SCOPE,
        "user_authorization_text": PROBE_AUTHORIZATION_TEXT,
        "authorized_revision": expected_revision,
        "preparation_report_sha256": expected_preparation_sha256,
        "output_root": expected_output_root,
        "authorization_boundary": EXECUTION_BOUNDARY,
    }
    if dict(approval) != expected:
        raise ValueError("conditioning-ranking probe approval does not match exact scope")
    return copy.deepcopy(expected)
