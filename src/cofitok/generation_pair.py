from __future__ import annotations

from typing import Any

MATCHED_CONFIG_SECTIONS = ("data", "diffusion", "runtime", "optimization")
RESTRICTED_COFITOK_SYNTHESIS_MODES = {"restricted", "fixed_basis"}
FACTORIZATION_MODEL_FIELDS = {
    "token_count",
    "token_channels",
    "predictor_use_feedback",
    "synthesis_mode",
    "synthesis_kernel_size",
    "gamma_mode",
    "synthesis_token_strides",
    "synthesis_active_token_channels",
    "token_channel_schedule",
    "token_spatial_strides",
    "deep_synthesis_hidden_channels",
    "deep_synthesis_depth",
}
SHARED_TRAINING_LOSS_FIELDS = {
    "epsilon_weight",
    "min_snr_gamma",
    "rollout_consistency_weight",
    "rollout_consistency_start_step",
    "rollout_consistency_warmup_steps",
    "rollout_consistency_timestep_delta",
    "rollout_consistency_unroll_steps",
    "rollout_consistency_batch_fraction",
    "rollout_consistency_clip_x0",
    "rollout_consistency_mode",
    "ema_teacher_consistency_weight",
    "ema_teacher_consistency_start_step",
    "ema_teacher_consistency_warmup_steps",
    "ema_teacher_consistency_batch_fraction",
}


def generation_pair_contract(
    cofitok_config: dict[str, Any],
    dense_config: dict[str, Any],
) -> dict[str, Any]:
    mismatched_sections = [
        section
        for section in MATCHED_CONFIG_SECTIONS
        if cofitok_config.get(section) != dense_config.get(section)
    ]
    cofitok_model = cofitok_config.get("model", {})
    dense_model = dense_config.get("model", {})
    shared_model_fields = sorted(
        (set(cofitok_model) | set(dense_model)) - FACTORIZATION_MODEL_FIELDS
    )
    mismatched_model_fields = [
        field
        for field in shared_model_fields
        if cofitok_model.get(field) != dense_model.get(field)
    ]
    issues = []
    if mismatched_sections:
        issues.append("mismatched config sections: " + ", ".join(mismatched_sections))
    if mismatched_model_fields:
        issues.append(
            "mismatched shared model fields: " + ", ".join(mismatched_model_fields)
        )

    identities = {
        "cofitok_token_count": int(cofitok_model.get("token_count", 0)),
        "dense_token_count": int(dense_model.get("token_count", 0)),
        "cofitok_feedback": cofitok_model.get("predictor_use_feedback"),
        "dense_feedback": dense_model.get("predictor_use_feedback"),
        "cofitok_synthesis": cofitok_model.get("synthesis_mode"),
        "dense_synthesis": dense_model.get("synthesis_mode"),
    }
    if identities["cofitok_token_count"] <= 1 or identities["dense_token_count"] != 1:
        issues.append("factorized/dense token-count identities are invalid")
    if (
        identities["cofitok_feedback"] is not True
        or identities["dense_feedback"] is not False
    ):
        issues.append("factorized/dense predictor-feedback identities are invalid")
    if (
        identities["cofitok_synthesis"] not in RESTRICTED_COFITOK_SYNTHESIS_MODES
        or identities["dense_synthesis"] != "dense_identity"
    ):
        issues.append("factorized/dense synthesis identities are invalid")

    cofitok_loss = cofitok_config.get("loss", {})
    dense_loss = dense_config.get("loss", {})
    mismatched_shared_loss_fields = [
        field
        for field in sorted(SHARED_TRAINING_LOSS_FIELDS)
        if cofitok_loss.get(field) != dense_loss.get(field)
    ]
    if mismatched_shared_loss_fields:
        issues.append(
            "mismatched shared training loss fields: "
            + ", ".join(mismatched_shared_loss_fields)
        )
    cofitok_epsilon = float(cofitok_loss.get("epsilon_weight", 0.0))
    dense_epsilon = float(dense_loss.get("epsilon_weight", 0.0))
    if cofitok_epsilon <= 0.0 or cofitok_epsilon != dense_epsilon:
        issues.append("primary epsilon loss weights are not matched and positive")
    dense_nonzero_auxiliary = sorted(
        key
        for key, value in dense_loss.items()
        if key not in SHARED_TRAINING_LOSS_FIELDS
        and key.endswith("_weight")
        and isinstance(value, (int, float))
        and float(value) != 0.0
    )
    if dense_nonzero_auxiliary:
        issues.append(
            "dense baseline has nonzero factorization auxiliary losses: "
            + ", ".join(dense_nonzero_auxiliary)
        )

    return {
        "valid": not issues,
        "matched_config_sections": list(MATCHED_CONFIG_SECTIONS),
        "factorization_model_fields": sorted(FACTORIZATION_MODEL_FIELDS),
        "shared_model_fields": shared_model_fields,
        "mismatched_config_sections": mismatched_sections,
        "mismatched_shared_model_fields": mismatched_model_fields,
        "identities": identities,
        "primary_epsilon_weight": cofitok_epsilon,
        "shared_training_loss_fields": sorted(SHARED_TRAINING_LOSS_FIELDS),
        "mismatched_shared_training_loss_fields": mismatched_shared_loss_fields,
        "dense_nonzero_auxiliary_losses": dense_nonzero_auxiliary,
        "issues": issues,
    }
